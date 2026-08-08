# Adapted from https://github.com/primatrix/pallas-kernel (rev 41431b1, release/v0.4)
# Patch: feat/gla_varlen @ a7682e72 — caller no longer needs per-seq chunk padding.
# Vendored to remove external dependency after the upstream repository went private.
#
# This file merges the following modules into a single file:
#   - tops/utils.py (assert_shape, assert_shape_or_none)
#   - tops/ops/utils.py (exp, get_interpret)
#   - tops/ops/simple_gla/fused_recurrent.py (fused_recurrent_simple_gla)
#   - tops/ops/common/chunk_h.py (_build_chunk_map, _chunk_fwd_h_kernel_varlen, chunk_fwd_h_kernel_varlen)
#   - tops/ops/common/chunk_o.py (_chunk_fwd_o_kernel, _chunk_fwd_o_pl, chunk_fwd_o)
#   - tops/ops/simple_gla/chunk.py (chunk_simple_gla_fwd_varlen + align/unalign helpers)
#   - tops/ops/simple_gla/__init__.py (SimpleGLAKernelMode, simple_gla_fwd)

from __future__ import annotations

import enum
import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

# =============================================================================
# Utilities (from tops/utils.py and tops/ops/utils.py)
# =============================================================================


def assert_shape_or_none(
    x: jax.Array | list[jax.Array | None] | tuple[jax.Array | None, ...] | None,
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
    if x is None:
        return
    if isinstance(x, list | tuple):
        has_names = isinstance(name, list | tuple) and len(name) == len(x)
        for i, tensor in enumerate(x):
            if tensor is not None:
                curr_name = name[i] if has_names else f"{name}_{i}"
                assert (
                    tensor.shape == expected_shape
                ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def assert_shape(
    x: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
    if isinstance(x, list | tuple):
        has_names = isinstance(name, list | tuple) and len(name) == len(x)
        for i, tensor in enumerate(x):
            curr_name = name[i] if has_names else f"{name}_{i}"
            assert (
                tensor.shape == expected_shape
            ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def exp(x):
    return jnp.exp(x.astype(jnp.float32))


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


# =============================================================================
# Fused recurrent (from tops/ops/simple_gla/fused_recurrent.py)
# Pure JAX implementation using jax.lax.scan, decode-friendly.
# =============================================================================


def _scan_segment(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over one dense segment."""
    if reverse:
        q = jnp.flip(q, axis=1)
        k = jnp.flip(k, axis=1)
        v = jnp.flip(v, axis=1)
        if g is not None:
            g = jnp.flip(g, axis=1)

    B, _T, H, K = q.shape
    V = v.shape[-1]
    h0 = initial_state if initial_state is not None else jnp.zeros((B, H, K, V), dtype=q.dtype)

    q_t = jnp.swapaxes(q, 0, 1)
    k_t = jnp.swapaxes(k, 0, 1)
    v_t = jnp.swapaxes(v, 0, 1)
    g_t = jnp.swapaxes(g, 0, 1) if g is not None else jnp.zeros((q_t.shape[0], B, H), dtype=q.dtype)
    use_g = g is not None

    def step(h, xs):
        q_i, k_i, v_i, g_i = xs
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma[None, :]
        else:
            decay = jnp.broadcast_to(g_gamma[None, :], (B, H))

        h = h * jnp.exp(decay)[:, :, None, None]
        h = h + k_i[:, :, :, None] * v_i[:, :, None, :]
        o_i = jnp.sum(h * (q_i[:, :, :, None] * scale), axis=2)
        return h, o_i

    h_final, o_t = jax.lax.scan(step, h0, (q_t, k_t, v_t, g_t))
    o = jnp.swapaxes(o_t, 0, 1)

    if reverse:
        o = jnp.flip(o, axis=1)

    return o, h_final


def _scan_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
    cu_seqlens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over packed varlen data with one JAX scan."""
    _B, T, H, K = q.shape
    V = v.shape[-1]
    N = cu_seqlens.shape[0] - 1

    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    seq_starts = cu_seqlens[:-1]
    seq_ends = cu_seqlens[1:]
    token_starts = seq_starts[seq_ids]
    token_ends = seq_ends[seq_ids]

    if reverse:
        scan_order = token_ends - 1 - (token_idx - token_starts)
        reset_mask = token_idx == (token_ends - 1)
    else:
        scan_order = token_idx
        reset_mask = token_idx == token_starts

    scan_seq_ids = seq_ids[scan_order]
    q_s = q[0, scan_order]
    k_s = k[0, scan_order]
    v_s = v[0, scan_order]
    g_s = g[0, scan_order] if g is not None else jnp.zeros((T, H), dtype=q.dtype)

    h0_all = initial_state if initial_state is not None else jnp.zeros((N, H, K, V), dtype=q.dtype)
    use_g = g is not None

    def step(carry, xs):
        h_prev, final_states = carry
        seq_id, do_reset, q_i, k_i, v_i, g_i = xs

        h = jnp.where(do_reset, h0_all[seq_id], h_prev)
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma
        else:
            decay = g_gamma

        h = h * jnp.exp(decay)[:, None, None]
        h = h + k_i[:, :, None] * v_i[:, None, :]
        o_i = jnp.sum(h * (q_i[:, :, None] * scale), axis=1)

        final_states = final_states.at[seq_id].set(h)
        return (h, final_states), o_i

    init_carry = (
        jnp.zeros((H, K, V), dtype=q.dtype),
        h0_all,
    )
    (h_last, final_states), o_scan = jax.lax.scan(
        step,
        init_carry,
        (scan_seq_ids, reset_mask[scan_order], q_s, k_s, v_s, g_s),
    )
    del h_last

    inv_order = jnp.argsort(scan_order)
    o = o_scan[inv_order][None, ...]
    return o, final_states


def fused_recurrent_simple_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Simple GLA fused recurrent forward for decode-friendly execution.

    Args:
        q: [B, T, H, K] queries.
        k: [B, T, H, K] keys.
        v: [B, T, H, V] values.
        g: [B, T, H] optional per-token log gate.
        g_gamma: [H] optional per-head constant log decay.
        scale: Optional query scaling factor. Defaults to K ** -0.5.
        initial_state: [N, H, K, V] optional recurrent state, where N=B for dense
            mode and N=len(cu_seqlens)-1 for varlen mode.
        output_final_state: Whether to return the final recurrent state.
        reverse: Whether to process each sequence in reverse time order.
        cu_seqlens: [N+1] cumulative sequence lengths for packed varlen inputs.

    Returns:
        Tuple of output [B, T, H, V] in q.dtype and optional final state
        [N, H, K, V] in the input dtype.
    """
    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert v.ndim == 4, f"v must be 4D [B,T,H,V], got {v.ndim}D"

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.shape[:3] == q.shape[:3], f"v shape {v.shape} incompatible with q"
    assert g is not None or g_gamma is not None, "At least one of g or g_gamma must be provided"
    if g is not None:
        assert g.ndim == 3 and g.shape == (B, T, H), f"g shape {g.shape} != {(B, T, H)}"
    if g_gamma is not None:
        assert (
            g_gamma.ndim == 1 and g_gamma.shape[0] == H
        ), f"g_gamma shape {g_gamma.shape} != ({H},)"
    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
    if initial_state is not None:
        assert initial_state.shape == (
            N,
            H,
            K,
            V,
        ), f"initial_state shape {initial_state.shape} != expected {(N, H, K, V)}"

    if scale is None:
        scale = K**-0.5
    scale = float(scale)

    q_f = q
    k_f = k
    v_f = v
    g_f = g
    g_gamma_f = g_gamma
    h0_f = initial_state

    if cu_seqlens is None:
        o, ht = _scan_segment(
            q_f,
            k_f,
            v_f,
            g=g_f,
            g_gamma=g_gamma_f,
            scale=scale,
            initial_state=h0_f,
            reverse=reverse,
        )
        return o, (ht if output_final_state else None)

    cu_f = jnp.asarray(cu_seqlens, dtype=jnp.int32)
    o, ht = _scan_varlen(
        q_f,
        k_f,
        v_f,
        g=g_f,
        g_gamma=g_gamma_f,
        scale=scale,
        initial_state=h0_f,
        reverse=reverse,
        cu_seqlens=cu_f,
    )
    return o, (ht if output_final_state else None)


# =============================================================================
# Chunk forward h — varlen path (from tops/ops/common/chunk_h.py)
# Pallas TPU kernel for computing hidden states with variable-length sequences.
# =============================================================================


def _build_chunk_map(cu_seqlens, T_sum, BT):
    NT = T_sum // BT
    chunk_ids = lax.iota(jnp.int32, NT)
    chunk_pos = chunk_ids * BT
    N = cu_seqlens.shape[-1] - 1
    seq_idx = jnp.searchsorted(cu_seqlens[1:], chunk_pos, side="right")
    seq_idx = jnp.clip(seq_idx, 0, N - 1)
    return seq_idx


def _chunk_fwd_h_kernel_varlen(
    k_ref,  # [1, BT, BK]
    v_ref,  # [1, BT, BV]
    h0_ref,  # [N, 1, BK, BV]
    gk_ref,  # [1, BT, BK]
    g_gamma_ref,  # [H,]
    cu_seqlens_ref,  # [num_seq+1]
    chunk_to_seq,  # [T_sum/BT]
    seq_real_lens_ref,  # [N] real (non-padded) seq length, or None
    h_ref,  # [NS, 1, BK, BV]
    ht_ref,  # [N, 1, BK , BV]
    scratch_ref,  # [BK, BV]
    *,
    BT,
    BS,
):
    BT, BK = k_ref.shape[1], k_ref.shape[2]
    BV = v_ref.shape[2]

    NTS = BS // BT
    b_h_start = jnp.zeros((BK, BV), dtype=jnp.float32)

    i_h, _i_k, _i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3)

    if g_gamma_ref is not None:
        b_g = g_gamma_ref[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    t0 = i_t * BT

    seq_idx = chunk_to_seq[i_t]

    bos = cu_seqlens_ref[seq_idx]
    eos = cu_seqlens_ref[seq_idx + 1]

    @pl.when(bos != eos)
    def _():
        # reset h state
        @pl.when(t0 == bos)
        def reset_state():
            if h0_ref is not None:
                scratch_ref[...] = h0_ref[seq_idx, 0].astype(scratch_ref.dtype)
            else:
                scratch_ref[...] = b_h_start

        # store intermediate state
        @pl.when(i_t % NTS == 0)
        def store_fn():
            s_i = i_t // NTS
            h_ref[s_i, 0] = scratch_ref[...].astype(h_ref.dtype)
            return None

        k_tile = k_ref[(0, slice(None), slice(None))]  # [BT,BK]
        v_tile = v_ref[(0, slice(None), slice(None))]  # [BT,BV]

        if g_gamma_ref is not None:
            # Use real (non-padded) length when seq_real_lens is provided so b_g_last
            # only covers real tokens — sequences padded to chunk_size internally
            # would otherwise accumulate decay over zero-padding tail.
            if seq_real_lens_ref is not None:
                real_eos = bos + seq_real_lens_ref[seq_idx]
                effective_remaining = jnp.maximum(real_eos - t0, 0)
            else:
                effective_remaining = eos - t0
            # tpu not support scalar bf16 mul
            L_chunk = jnp.minimum(BT, effective_remaining)
            b_g_last = (g_gamma_ref[i_h].astype(jnp.float32) * L_chunk).astype(g_gamma_ref.dtype)
            scratch_ref[...] *= exp(b_g_last)
            # Mask exponent to avoid NaN (0 * inf) in padding positions
            v_decay_exp = b_g_last - b_g
            v_decay_exp = jnp.where(jnp.arange(BT) < L_chunk, v_decay_exp, -1e9)
            v_tile = (v_tile * exp(v_decay_exp)[:, None]).astype(v_tile.dtype)

        if gk_ref is not None:
            gk_tile = gk_ref[(0, slice(None), slice(None))]  # BT * BK
            g_last = gk_tile[-1, :]
            decay = exp(g_last)
            scratch_ref[...] = scratch_ref[...] * decay[:, None]  # [BK, BV] * [BK,1]
            k_tile = (k_tile * exp(g_last[None, :] - gk_tile)).astype(k_tile.dtype)

        # state update
        scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
            k_tile.astype(jnp.float32).T,
            v_tile.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        @pl.when(t0 + BT >= eos)
        def write_final():
            if ht_ref is not None:
                ht_ref[seq_idx, 0] = scratch_ref[...].astype(jnp.float32)


def _chunk_fwd_h_decay_matrix_states(
    k,  # (H, T_sum, K) — chunk-major, already transposed
    v,  # (H, T_sum, V)
    g_gamma,  # (H,)
    h0,  # (N, H, K, V) or None
    cu_seqlens_dev,  # (N+1,)
    chunk_to_seq,  # (NT,)
    seq_real_lens,  # (N,) or None
    *,
    BT,
    NTS,
    NS,
    N,
    output_final_state,
    states_in_fp32,
):
    """Closed-form solve of the chunk-state recurrence, as one decay matrix.

    The incumbent Pallas stage walks the chunk axis sequentially, carrying the
    ``(BK, BV)`` state in scratch: ``S_{j+1} = d_j * S_j + C_j`` with per-chunk
    decay ``d_j = exp(gamma * L_j)`` and chunk contribution
    ``C_j = K_j^T (V_j * exp(gamma * (L_j - i - 1)))``. Every step of that walk is
    one grid program, and the whole program is replicated per head, so the stage's
    cost is dominated by the number of steps rather than by the arithmetic.

    Unrolling the recurrence gives the entrance state at chunk ``r`` in closed
    form::

        S_r = exp(A_r - A_f) h0 + sum_{f <= j < r} exp(A_r - A_{j+1}) C_j

    where ``A_r = sum_{j<r} gamma * L_j`` and ``f`` is the row's first chunk. The
    chunk contributions ``C`` are one batched einsum over (head, chunk) and the
    carry becomes one strictly-lower-triangular ``(NT+1, NT)`` decay matrix per
    head, so the sequential axis disappears entirely.

    Every exponent is a difference ``A_r - A_{j+1}`` taken over an interval of
    ``gamma * L`` terms with ``gamma <= 0`` and ``L >= 0``, hence non-positive:
    the factored form ``exp(A_r) * sum exp(-A_{j+1}) C_j`` would overflow after a
    few tens of chunks, and keeping the difference inside the matrix avoids it.
    The result is the same sum re-associated — nothing is dropped or approximated.
    """
    H, T_sum, K = k.shape
    V = v.shape[2]
    NT = T_sum // BT
    f32 = jnp.float32

    gamma = g_gamma.astype(f32)  # (H,)
    chunk_idx = jnp.arange(NT, dtype=jnp.int32)
    t0 = chunk_idx * BT  # (NT,)
    seq = chunk_to_seq  # (NT,)
    bos = cu_seqlens_dev[seq]
    eos = cu_seqlens_dev[seq + 1]
    # Real (non-padded) chunk occupancy, matching the sequential stage's
    # `L_chunk = min(BT, effective_remaining)`. Chunks of an empty sequence are
    # skipped there (`@pl.when(bos != eos)`), i.e. they neither decay nor
    # accumulate — L = 0 reproduces exactly that.
    real_eos = bos + seq_real_lens[seq] if seq_real_lens is not None else eos
    L = jnp.clip(real_eos - t0, 0, BT).astype(f32)
    L = jnp.where(bos != eos, L, 0.0)  # (NT,)

    # A[h, r] = sum_{j < r} gamma_h * L_j, for the NT+1 chunk boundaries.
    log_d = gamma[:, None] * L[None, :]  # (H, NT)
    A = jnp.concatenate(
        [jnp.zeros((H, 1), f32), jnp.cumsum(log_d, axis=1)], axis=1
    )  # (H, NT+1)

    # Per-chunk contribution C[h, j] = K_j^T (V_j * exp(gamma*(L_j - i - 1))).
    k_c = k.reshape(H, NT, BT, K)
    v_c = v.reshape(H, NT, BT, V)
    row = jnp.arange(BT, dtype=f32)
    v_decay_exp = gamma[:, None, None] * (L[None, :, None] - (row + 1)[None, None, :])
    in_chunk = row[None, None, :] < L[None, :, None]
    v_decay = jnp.where(in_chunk, exp(jnp.minimum(v_decay_exp, 0.0)), 0.0)
    v_scaled = (v_c.astype(f32) * v_decay[..., None]).astype(v.dtype)
    contrib = jnp.einsum(
        "hjtk,hjtv->hjkv",
        k_c.astype(f32),
        v_scaled.astype(f32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=f32,
    )  # (H, NT, K, V)

    # One strictly-lower-triangular decay matrix per head over the NT+1 boundary
    # rows: M[h, r, j] = exp(A_r - A_{j+1}) when chunk j precedes row r inside the
    # same sequence, else 0.
    boundary = jnp.arange(NT + 1, dtype=jnp.int32)
    row_seq = jnp.concatenate([seq, seq[-1:]])  # (NT+1,)
    same_seq = row_seq[:, None] == seq[None, :]
    strictly_before = chunk_idx[None, :] < boundary[:, None]
    keep = same_seq & strictly_before
    decay_exp = jnp.minimum(A[:, :, None] - A[:, None, 1:], 0.0)
    decay = jnp.where(keep[None], exp(decay_exp), 0.0)  # (H, NT+1, NT)

    states = jnp.einsum(
        "hrj,hjkv->hrkv",
        decay,
        contrib,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=f32,
    )  # (H, NT+1, K, V)
    if h0 is not None:
        # Initial state carried from the row's own sequence start f: exp(A_r - A_f).
        first_chunk = cu_seqlens_dev[row_seq] // BT
        a_first = jnp.take_along_axis(A, first_chunk[None, :], axis=1)
        carry = exp(jnp.minimum(A - a_first, 0.0))  # (H, NT+1)
        h0_h = jnp.transpose(h0, (1, 0, 2, 3)).astype(f32)  # (H, N, K, V)
        states = states + carry[:, :, None, None] * h0_h[:, row_seq]

    h_dtype = f32 if states_in_fp32 else k.dtype
    h_rows = jnp.arange(NS, dtype=jnp.int32) * NTS
    h = jnp.transpose(states[:, h_rows], (1, 0, 2, 3)).astype(h_dtype)
    if not output_final_state:
        return h, None
    # The final state of a sequence is the boundary row just past its last chunk;
    # padding chunks beyond `eos` have L = 0, so any later boundary of the same
    # sequence carries the identical value — exactly what the sequential stage's
    # repeated `t0 + BT >= eos` writes leave behind.
    last_boundary = jnp.max(
        jnp.where(
            seq[None, :] == jnp.arange(N, dtype=jnp.int32)[:, None],
            chunk_idx[None, :] + 1,
            0,
        ),
        axis=1,
    )
    ht = jnp.transpose(states[:, last_boundary], (1, 0, 2, 3)).astype(f32)
    return h, ht


@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
        "enable_chunk_fwd_h_kernel_varlen_variant",
    ],
)
def chunk_fwd_h_kernel_varlen(
    k: jax.Array,  # [B,T,H,K]
    v: jax.Array,  # [B,T,H,V]
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 128,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    seq_real_lens: jax.Array | None = None,  # [N]
    enable_chunk_fwd_h_kernel_varlen_variant: bool = False,
):
    interpret = get_interpret()
    assert g is None, "g should be None."
    assert gv is None, "gv should be None."
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"

    T_sum = B * T
    chunk_to_seq = _build_chunk_map(cu_seqlens=cu_seqlens_dev, T_sum=T_sum, BT=BT)

    N, NS = (
        len(cu_seqlens_dev) - 1,
        T_sum // BS,
    )

    k = jnp.reshape(k, (T_sum, H, K))
    v = jnp.reshape(v, (T_sum, H, V))

    k = jnp.transpose(k, (1, 0, 2))  # (H,B*T,K)
    v = jnp.transpose(v, (1, 0, 2))  # (H,B*T,V)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K))
        gk = jnp.transpose(gk, (1, 0, 2))  # (H,B*T,K)

    # Selected schedule for the chunk-state recurrence. The incumbent (False) is
    # the sequential Pallas walk below; the variant resolves the same recurrence
    # in closed form as one per-head decay matrix. Both produce the same states.
    if enable_chunk_fwd_h_kernel_varlen_variant:
        assert gk is None, "the decay-matrix state solve requires gk=None."
        assert g_gamma is not None, "the decay-matrix state solve requires g_gamma."
    closed_form_states = (
        _chunk_fwd_h_decay_matrix_states(
            k,
            v,
            g_gamma,
            h0,
            cu_seqlens_dev,
            chunk_to_seq,
            seq_real_lens,
            BT=BT,
            NTS=BS // BT,
            NS=NS,
            N=N,
            output_final_state=output_final_state,
            states_in_fp32=states_in_fp32,
        )
        if enable_chunk_fwd_h_kernel_varlen_variant
        else None
    )
    if closed_form_states is not None:
        return closed_form_states

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV), T_sum // BT)

    def k_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def gk_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def v_index_map(head_index, _, v_index, t_index):
        return head_index, t_index, v_index

    def h0_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def ht_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def h_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((N, 1, BK, BV), ht_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, BT, BK), k_index_map),
        pl.BlockSpec((1, BT, BV), v_index_map),
    ]
    if h0 is not None:
        in_specs.append(pl.BlockSpec((N, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, BT, BK), gk_index_map))
    else:
        in_specs.append(None)

    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)

    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    if seq_real_lens is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)
    scratch = pltpu.VMEM((BK, BV), jnp.float32)
    scratch_shapes = [scratch]
    kernel = functools.partial(
        _chunk_fwd_h_kernel_varlen,
        BT=BT,
        BS=BS,
    )
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            vmem_limit_bytes=128 * 1024 * 1024,
        ),
    )(k, v, h0, gk, g_gamma, cu_seqlens_dev, chunk_to_seq, seq_real_lens)
    if output_final_state:
        return h, ht
    return h, None


# =============================================================================
# Chunk forward o (from tops/ops/common/chunk_o.py)
# Pallas TPU kernel for computing chunk output.
# =============================================================================


# Row count of the output stage's second-level sub-chunk. 128 is the TPU lane
# width and this kernel's pinned BK/BV, so a sub-chunk score tile and the running
# (K, BV) state are both exactly one native tile wide.
_OUTPUT_SUBCHUNK_ROWS = 128


def _output_subchunk_rows(chunk_rows: int) -> int | None:
    """Sub-chunk height for the two-level output schedule.

    ``None`` means the chunk is already at most one sub-chunk tall, so the
    two-level walk would degenerate into the incumbent single-tile schedule.
    """
    if chunk_rows <= _OUTPUT_SUBCHUNK_ROWS:
        return None
    return _OUTPUT_SUBCHUNK_ROWS


def _chunk_fwd_o_subchunk_body(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    g_ref,
    g_gamma_ref,
    scale_ref,
    o_ref,
    *,
    value_tile_base,
    BT: int,
    ZERO_STATE_OUTPUT: bool,
    OUTPUT_VALUE_TILES: int,
    SUBCHUNK_ROWS: int,
):
    """Two-level output schedule: split a chunk into ``SUBCHUNK_ROWS``-row sub-chunks.

    The incumbent schedule materializes one masked ``BT x BT`` score tile per output
    tile, so both its intra-chunk score arithmetic and — far more expensive here —
    the elementwise decay/mask traffic over that tile grow with the SQUARE of the
    chunk length. At ``BT=2048`` that tile is 16 MiB of fp32 and roughly half of it
    is masked away again.

    This variant applies the kernel's own chunked recurrence one level deeper: a
    sub-chunk pays a ``SUBCHUNK_ROWS x SUBCHUNK_ROWS`` score tile for its own rows
    and reads every older row of the same chunk through a ``(K, BV)`` state, exactly
    as a chunk reads the preceding chunks through ``h``. Score work and score-tile
    traffic drop from ``BT * BT`` to ``BT * SUBCHUNK_ROWS`` and the sub-chunk states
    add a fixed ``BT * K * V``.

    The sub-chunk axis stays a BATCH dimension of single contractions rather than a
    loop, and the state recurrence is resolved as one ``(num_sub, num_sub)`` decay
    matrix applied to the per-sub-chunk contributions, so the schedule keeps the
    incumbent's operation count while shrinking every operand. Nothing is dropped or
    approximated: it is the same linear-attention sum re-associated, and each
    sub-chunk state carries the same chunk-entrance convention ``h`` arrives in, so
    both gate forms stay exact.
    """
    sub = SUBCHUNK_ROWS
    num_sub = BT // sub
    row = jnp.arange(sub)
    sub_mask = row[:, None] >= row[None, :]
    older = jnp.arange(num_sub)[:, None] > jnp.arange(num_sub)[None, :]
    scale = scale_ref[0].astype(jnp.float32)
    K = q_ref.shape[-1]
    BV = v_ref.shape[-1]

    for local_value_tile in range(OUTPUT_VALUE_TILES):
        # Chunk-cumulative log decay per row. Both gate forms are additive in log
        # space, matching the incumbent's two independent exp() factors.
        b_log = None
        if g_ref is not None:
            b_log = g_ref[local_value_tile, 0, :, 0].astype(jnp.float32)  # (BT,)
        if g_gamma_ref is not None:
            value_tile_idx = value_tile_base + local_value_tile
            b_gamma = g_gamma_ref[value_tile_idx].astype(jnp.float32)
            b_g_gamma = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)
            b_log = b_g_gamma if b_log is None else b_log + b_g_gamma

        s_q = q_ref[local_value_tile, 0].reshape(num_sub, sub, K)
        s_k = k_ref[local_value_tile, 0].reshape(num_sub, sub, K)
        s_v = v_ref[local_value_tile, 0].astype(jnp.float32).reshape(num_sub, sub, BV)
        if b_log is not None:
            # ``s_log`` is the chunk-cumulative decay; ``entrance``/``exit`` are its
            # values just before and at the end of each sub-chunk. Every exponent
            # formed below is a difference of a later minus an earlier position, so
            # all of them stay non-positive for a decaying gate.
            s_log = b_log.reshape(num_sub, sub)
            exit_log = s_log[:, sub - 1]  # (num_sub,)
            entrance_log = jnp.concatenate(
                [jnp.zeros((1,), jnp.float32), exit_log[: num_sub - 1]]
            )

        # Intra-sub-chunk attention: one masked (sub, sub) score tile per sub-chunk.
        s_A = jnp.einsum(
            "nik,njk->nij", s_q, s_k, preferred_element_type=jnp.float32
        )
        if b_log is not None:
            log_diff = s_log[:, :, None] - s_log[:, None, :]
            s_A = s_A * exp(jnp.where(sub_mask, log_diff, 0.0))
        s_A = jnp.where(sub_mask, s_A, 0.0)
        # Keep the score tile in fp32 for precision; the values are upcast instead.
        b_o = jnp.einsum(
            "nij,njd->nid",
            s_A,
            s_v,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        # Each sub-chunk's own contribution to the states the later sub-chunks read,
        # expressed at that sub-chunk's exit so no positive exponent ever appears.
        k_decayed = s_k.astype(jnp.float32)
        if b_log is not None:
            k_decayed = k_decayed * exp(exit_log[:, None] - s_log)[:, :, None]
        b_u = jnp.einsum(
            "nik,nid->nkd",
            k_decayed,
            s_v,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        # Resolve the sub-chunk state recurrence in closed form: sub-chunk n reads
        # every strictly older sub-chunk's contribution, rebased onto its entrance.
        decay_matrix = older.astype(jnp.float32)
        if b_log is not None:
            decay_matrix = decay_matrix * exp(
                jnp.where(older, entrance_log[:, None] - exit_log[None, :], 0.0)
            )
        b_state = jnp.einsum(
            "nb,bkd->nkd",
            decay_matrix,
            b_u,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        if not ZERO_STATE_OUTPUT:
            b_h = h_ref[local_value_tile, 0].astype(jnp.float32)  # (K, BV)
            entrance_decay = (
                jnp.ones((num_sub,), jnp.float32)
                if b_log is None
                else exp(entrance_log)
            )
            b_state = b_state + entrance_decay[:, None, None] * b_h[None]

        b_inter = jnp.einsum(
            "nik,nkd->nid", s_q, b_state, preferred_element_type=jnp.float32
        )
        if b_log is not None:
            b_inter = b_inter * exp(s_log - entrance_log[:, None])[:, :, None]
        b_o = (b_o + b_inter) * scale
        o_ref[local_value_tile, 0] = b_o.reshape(BT, BV).astype(o_ref.dtype)


def _chunk_fwd_o_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    g_ref,
    g_gamma_ref,
    scale_ref,
    o_ref,
    *,
    BT: int,
    ZERO_STATE_OUTPUT: bool,
    OUTPUT_VALUE_TILES: int,
    SUBCHUNK_ROWS: int | None = None,
):
    """Pallas kernel for chunk_fwd_o.

    Grid: (ceil((H * num_v_tiles) / OUTPUT_VALUE_TILES), total_NT)
    Refs (after block spec indexing):
      q_ref/k_ref: (OUTPUT_VALUE_TILES, 1, BT, K)
      v_ref: (OUTPUT_VALUE_TILES, 1, BT, BV)
      h_ref: (OUTPUT_VALUE_TILES, 1, K, BV), or None when elided
      g_ref: (OUTPUT_VALUE_TILES, 1, BT, 128) or None
      g_gamma_ref: [H * num_v_tiles] via SMEM or ANY
      scale_ref: (1,) via SMEM or ANY
      o_ref: (OUTPUT_VALUE_TILES, 1, BT, BV)

    Each local tile retains its own q/k/v/gamma arithmetic. Grouping adjacent
    head-value tiles changes only program ownership, amortizing grid/program
    overhead without mixing heads or value slices.

    ``SUBCHUNK_ROWS`` is None on the incumbent single-tile schedule and names the
    sub-chunk height of the two-level schedule otherwise.
    """
    first_value_tile = pl.program_id(0) * OUTPUT_VALUE_TILES
    if SUBCHUNK_ROWS is not None:
        _chunk_fwd_o_subchunk_body(
            q_ref,
            k_ref,
            v_ref,
            h_ref,
            g_ref,
            g_gamma_ref,
            scale_ref,
            o_ref,
            value_tile_base=first_value_tile,
            BT=BT,
            ZERO_STATE_OUTPUT=ZERO_STATE_OUTPUT,
            OUTPUT_VALUE_TILES=OUTPUT_VALUE_TILES,
            SUBCHUNK_ROWS=SUBCHUNK_ROWS,
        )
        return
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    scale = scale_ref[0].astype(jnp.float32)

    for local_value_tile in range(OUTPUT_VALUE_TILES):
        b_q = q_ref[local_value_tile, 0]  # (BT, K)
        b_k = k_ref[local_value_tile, 0]  # (BT, K)
        b_v = v_ref[local_value_tile, 0]  # (BT, BV)

        if not ZERO_STATE_OUTPUT:
            b_h = h_ref[local_value_tile, 0]  # (K, BV)
            b_o = jnp.dot(
                b_q,
                b_h,
                preferred_element_type=jnp.float32,
            )
        b_A = jnp.dot(
            b_q,
            b_k.T,
            preferred_element_type=jnp.float32,
        )

        if g_ref is not None:
            b_g = g_ref[local_value_tile, 0, :, 0].astype(jnp.float32)  # (BT,)
            if not ZERO_STATE_OUTPUT:
                b_o = b_o * exp(b_g)[:, None]
            g_diff = b_g[:, None] - b_g[None, :]
            safe_g_diff = jnp.where(mask, g_diff, 0.0)
            b_A = b_A * exp(safe_g_diff)

        if g_gamma_ref is not None:
            value_tile_idx = first_value_tile + local_value_tile
            b_gamma = g_gamma_ref[value_tile_idx].astype(jnp.float32)
            b_g_gamma = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)
            if not ZERO_STATE_OUTPUT:
                b_o = b_o * exp(b_g_gamma)[:, None]
            g_gamma_diff = b_g_gamma[:, None] - b_g_gamma[None, :]
            safe_g_gamma_diff = jnp.where(mask, g_gamma_diff, 0.0)
            b_A = b_A * exp(safe_g_gamma_diff)

        b_A = jnp.where(mask, b_A, 0.0)

        # Keep b_A in fp32 for precision; upcast b_v instead. When the state
        # contribution is zero, do not materialize or decay/scale that zero term.
        b_intra = jnp.dot(
            b_A,
            b_v.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        if ZERO_STATE_OUTPUT:
            b_o = b_intra * scale
        else:
            b_o = b_o * scale + b_intra * scale
        o_ref[local_value_tile, 0] = b_o.astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "zero_state_output_elision",
        "output_value_tiles",
        "enable__chunk_fwd_o_pl_variant",
    ),
)
def _chunk_fwd_o_pl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array | None,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float,
    chunk_size: int = 64,
    zero_state_output_elision: bool = False,
    output_value_tiles: int = 1,
    enable__chunk_fwd_o_pl_variant: bool = False,
    output_gather: jax.Array | None = None,
) -> jax.Array:
    """Pallas launcher for chunk_fwd_o on the uniform-length path.

    ``enable__chunk_fwd_o_pl_variant`` selects the resident output schedule: the
    chunk-wide score tile is replaced by a two-level sub-chunked walk, and the
    launch delivers its rows straight into the caller-supplied ``output_gather``
    token order instead of an aligned staging buffer a later pass must restore.
    ``output_gather`` is ignored unless the variant is selected.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    def _reshape_bt(x, D):
        return x.reshape(B, NT, BT, H, D).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, D)

    _q = _reshape_bt(q, K)  # (H, total_NT, BT, K)
    _k = _reshape_bt(k, K)  # (H, total_NT, BT, K)
    _v = _reshape_bt(v, V)  # (H, total_NT, BT, V)
    _h = None
    if not zero_state_output_elision:
        assert h is not None, "h is required unless its zero output term is elided"
        _h = h.reshape(B, NT, H, K, V).transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _g = None
    if g is not None:
        _g = g.reshape(B, NT, BT, H).transpose(3, 0, 1, 2).reshape(H, total_NT, BT)
        _g = jnp.broadcast_to(_g[:, :, :, None], (H, total_NT, BT, 128))  # 4D for TPU alignment

    BV = 128 if V % 128 == 0 else V
    num_v_tiles = V // BV

    if num_v_tiles > 1:
        # Split V into tiles and merge with H: (H, ..., V) -> (H*num_v_tiles, ..., BV)
        _v = (
            _v.reshape(H, total_NT, BT, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, BT, BV)
        )
        if _h is not None:
            _h = (
                _h.reshape(H, total_NT, K, num_v_tiles, BV)
                .transpose(0, 3, 1, 2, 4)
                .reshape(H * num_v_tiles, total_NT, K, BV)
            )
        # g_gamma: repeat each head value for its V-tiles
        if g_gamma is not None:
            g_gamma = jnp.repeat(g_gamma, num_v_tiles)  # (H * num_v_tiles,)

    H_VT = H * num_v_tiles
    assert output_value_tiles > 0, "output_value_tiles must be positive"
    assert H_VT % output_value_tiles == 0, (
        f"H*num_v_tiles={H_VT} must be divisible by output_value_tiles="
        f"{output_value_tiles}"
    )

    # Present q/k/g in the same flattened head-value-tile order as v/h. This
    # lets one physical Pallas program own several independent full-BV tiles.
    # The common V=128 path has num_v_tiles=1 and needs no replication.
    if num_v_tiles > 1:
        _q = jnp.repeat(_q, num_v_tiles, axis=0)
        _k = jnp.repeat(_k, num_v_tiles, axis=0)
        if _g is not None:
            _g = jnp.repeat(_g, num_v_tiles, axis=0)

    grid = (H_VT // output_value_tiles, total_NT)

    # Every block owns output_value_tiles adjacent flattened head-value tiles.
    spec_qk = pl.BlockSpec(
        (output_value_tiles, 1, BT, K),
        index_map=lambda value_tile_block, nt_idx: (value_tile_block, nt_idx, 0, 0),
    )
    spec_v = pl.BlockSpec(
        (output_value_tiles, 1, BT, BV),
        index_map=lambda value_tile_block, nt_idx: (value_tile_block, nt_idx, 0, 0),
    )
    spec_h = (
        None
        if zero_state_output_elision
        else pl.BlockSpec(
            (output_value_tiles, 1, K, BV),
            index_map=lambda value_tile_block, nt_idx: (value_tile_block, nt_idx, 0, 0),
        )
    )
    interpret = get_interpret()
    spec_g = (
        None
        if _g is None
        else pl.BlockSpec(
            (output_value_tiles, 1, BT, 128),
            index_map=lambda value_tile_block, nt_idx: (value_tile_block, nt_idx, 0, 0),
        )
    )
    spec_gamma = (
        None
        if g_gamma is None
        else pl.BlockSpec(memory_space=pltpu.ANY if interpret else pltpu.SMEM)
    )
    spec_scale = pl.BlockSpec(memory_space=pltpu.ANY if interpret else pltpu.SMEM)

    subchunk_rows = (
        _output_subchunk_rows(BT) if enable__chunk_fwd_o_pl_variant else None
    )
    emit_order = output_gather if enable__chunk_fwd_o_pl_variant else None

    o = pl.pallas_call(
        functools.partial(
            _chunk_fwd_o_kernel,
            BT=BT,
            ZERO_STATE_OUTPUT=zero_state_output_elision,
            OUTPUT_VALUE_TILES=output_value_tiles,
            SUBCHUNK_ROWS=subchunk_rows,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[spec_qk, spec_qk, spec_v, spec_h, spec_g, spec_gamma, spec_scale],
            out_specs=pl.BlockSpec(
                (output_value_tiles, 1, BT, BV),
                index_map=lambda value_tile_block, nt_idx: (value_tile_block, nt_idx, 0, 0),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((H_VT, total_NT, BT, BV), v.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            # Full-sequence BT=2048 uses a 16 MiB fp32 score tile plus
            # elementwise temporaries. Match the state stage's VMEM ceiling
            # so Mosaic can schedule that exact tile without spilling it.
            vmem_limit_bytes=128 * 1024 * 1024,
        ),
        interpret=interpret,
    )(_q, _k, _v, _h, _g, g_gamma, jnp.asarray(scale, dtype=jnp.float32).reshape(1))

    if num_v_tiles > 1:
        o = (
            o.reshape(H, num_v_tiles, total_NT, BT, BV)
            .transpose(0, 2, 3, 1, 4)
            .reshape(H, total_NT, BT, V)
        )

    o = o.reshape(H, B, NT, BT, V).transpose(1, 2, 3, 0, 4)
    o = o.reshape(B, T, H, V)
    if emit_order is not None:
        # Land the rows in their final token order as part of this stage's own
        # layout step. The incumbent instead hands back the aligned staging
        # layout, which the caller has to restore with a scatter over the
        # aligned index space; expressing the identical permutation over the
        # OUTPUT index space is one gather and moves each row exactly once.
        o = o[:, emit_order]
    return o


def chunk_fwd_o(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array | None,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    zero_state_output_elision: bool = False,
    output_value_tiles: int = 1,
    enable__chunk_fwd_o_pl_variant: bool = False,
    output_gather: jax.Array | None = None,
) -> jax.Array:
    """Chunk forward output computation.

    ``zero_state_output_elision`` omits the state input and its identically-zero
    contribution for an output stage whose sole chunk entrance is zero.

    ``output_value_tiles`` groups that many independent full-width BV tiles in
    one Pallas program. It changes program ownership only; no head or value
    arithmetic is shared.

    ``enable__chunk_fwd_o_pl_variant`` selects the resident output schedule: a
    per-sub-chunk score tile plus a running intra-chunk state in place of the
    chunk-wide score tile, and delivery straight into the ``output_gather`` token
    order. ``output_gather`` is ignored unless the variant is selected.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if scale is None:
        scale = K**-0.5

    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert T % C == 0, f"Sequence length T={T} must be divisible by chunk_size={C}"
    assert (cu_seqlens_cpu is None) or (
        cu_seqlens_cpu % chunk_size == 0
    ).all(), "All sequence lengths must be divisible by chunk_size"
    if cu_seqlens_cpu is not None or cu_seqlens_dev is not None:
        assert B == 1, f"Packed varlen chunk_fwd_o expects B=1, got B={B}"
    assert zero_state_output_elision or h is not None, (
        "h is required unless its zero output term is elided"
    )
    assert scale is not None

    return _chunk_fwd_o_pl(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        chunk_size=chunk_size,
        zero_state_output_elision=zero_state_output_elision,
        output_value_tiles=output_value_tiles,
        enable__chunk_fwd_o_pl_variant=enable__chunk_fwd_o_pl_variant,
        output_gather=output_gather,
    )


# =============================================================================
# Chunk forward varlen + simple_gla_fwd entry point
# (from tops/ops/simple_gla/chunk.py and tops/ops/simple_gla/__init__.py)
# =============================================================================


def _build_align_gather_idx(cu_seqlens, aligned_cu, T_aligned):
    """For each position in the aligned layout, return (orig_pos, is_valid).
    Padding positions are mapped to 0 with is_valid=False; caller masks them."""
    N = cu_seqlens.shape[0] - 1
    real_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    pos = jnp.arange(T_aligned, dtype=jnp.int32)
    seq_idx = jnp.searchsorted(aligned_cu[1:], pos, side="right")
    seq_idx = jnp.clip(seq_idx, 0, N - 1)
    offset_in_seq = pos - aligned_cu[seq_idx]
    orig_pos = cu_seqlens[seq_idx] + offset_in_seq
    is_valid = offset_in_seq < real_lens[seq_idx]
    gather_idx = jnp.where(is_valid, orig_pos, jnp.int32(0))
    return gather_idx, is_valid


def _compute_t_aligned(T_orig, N, chunk_size, compact_alignment=False):
    """Static upper bound for the per-sequence aligned packed length.

    Each aligned sequence length is a multiple of ``BT`` and their sum is at
    most ``T_orig + N * (BT - 1)``.  Therefore the greatest ``BT`` multiple not
    exceeding that expression remains a safe upper bound.  The shipped path
    conservatively rounds it up once more; ``compact_alignment`` exposes the
    tight grid extent while retaining the same gather/mask layout semantics.
    """
    BT = chunk_size
    T_max = T_orig + N * (BT - 1)
    if compact_alignment:
        return (T_max // BT) * BT
    return ((T_max + BT - 1) // BT) * BT


def _align_varlen_inputs(q, k, v, cu_seqlens_dev, chunk_size, T_aligned):
    """Pad each sequence to a multiple of chunk_size and rebuild cu_seqlens."""
    BT = chunk_size
    N = cu_seqlens_dev.shape[0] - 1
    real_lens = cu_seqlens_dev[1:] - cu_seqlens_dev[:-1]
    aligned_lens = ((real_lens + BT - 1) // BT) * BT
    aligned_cu = jnp.zeros(N + 1, dtype=jnp.int32)
    aligned_cu = aligned_cu.at[1:].set(jnp.cumsum(aligned_lens))
    gather_idx, is_valid = _build_align_gather_idx(cu_seqlens_dev, aligned_cu, T_aligned)

    def _gather_and_mask(x):
        gathered = x[0, gather_idx]
        gathered = jnp.where(is_valid[:, None, None], gathered, 0)
        return gathered[None]

    q_a = _gather_and_mask(q)
    k_a = _gather_and_mask(k)
    v_a = _gather_and_mask(v)
    return q_a, k_a, v_a, aligned_cu, real_lens


def _build_unalign_gather_idx(cu_seqlens, aligned_cu, T_orig):
    """For each ORIGINAL token position, the aligned-layout row that carries it.

    This is the same permutation ``_unalign_output`` applies, indexed over the
    OUTPUT space instead of the aligned space. Every original position belongs to
    exactly one sequence and therefore to exactly one aligned row, so restoring
    the layout by reading each output row once is exact — no position is written
    twice and none is left unwritten.
    """
    N = cu_seqlens.shape[0] - 1
    pos = jnp.arange(T_orig, dtype=jnp.int32)
    seq_idx = jnp.searchsorted(cu_seqlens[1:], pos, side="right")
    seq_idx = jnp.clip(seq_idx, 0, N - 1)
    return aligned_cu[seq_idx] + (pos - cu_seqlens[seq_idx])


def _unalign_output(o_aligned, cu_seqlens_orig, aligned_cu, T_orig):
    """Scatter the aligned-layout output back to the original packed layout."""
    T_aligned = o_aligned.shape[1]
    gather_idx, is_valid = _build_align_gather_idx(cu_seqlens_orig, aligned_cu, T_aligned)
    o_out = jnp.zeros(
        (1, T_orig, o_aligned.shape[2], o_aligned.shape[3]),
        dtype=o_aligned.dtype,
    )
    o_out = o_out.at[0, gather_idx].add(jnp.where(is_valid[:, None, None], o_aligned[0], 0))
    return o_out


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_ht",
        "chunk_size",
        "compact_alignment",
        "single_chunk_state_elision",
        "zero_state_output_elision",
        "output_value_tiles",
        "enable__chunk_fwd_o_pl_variant",
        "enable_chunk_fwd_h_kernel_varlen_variant",
        "output_impl",
        "chunk_impl",
        "walk_impl",
    ],
)
def chunk_simple_gla_fwd_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    compact_alignment: bool = False,
    single_chunk_state_elision: bool = False,
    zero_state_output_elision: bool = False,
    output_value_tiles: int = 1,
    enable__chunk_fwd_o_pl_variant: bool = False,
    enable_chunk_fwd_h_kernel_varlen_variant: bool = False,
    output_impl: str = "incumbent",
    chunk_impl: str = "incumbent",
    walk_impl: str = "incumbent",
) -> tuple[jax.Array, jax.Array | None]:
    """Chunked varlen Simple GLA.

    ``compact_alignment`` selects the tight static upper bound for the number
    of aligned ``chunk_size`` tiles.  It changes only padding/grid extent; the
    gather, masking, recurrence order, and output unalignment remain exact.

    ``single_chunk_state_elision`` skips the state-update launch when the
    aligned input is exactly one logical chunk, no initial state is supplied,
    and no final state is requested.  In that case the output stage consumes
    only the all-zero state at the sole chunk boundary; the state produced
    after the chunk is terminal and otherwise unused.

    ``zero_state_output_elision`` removes the matching state load and ``q @ h``
    term from the output stage when single-chunk state elision proves that
    ``h`` is exactly zero.

    ``output_value_tiles`` controls how many independent full-BV output tiles
    each Pallas program computes, exposing the program tile grouping to callers.

    ``enable__chunk_fwd_o_pl_variant`` selects the output stage's resident
    schedule. The incumbent (False) keeps one chunk-wide masked score tile per
    output tile and returns the aligned staging layout, which is then restored by
    a scatter over the aligned index space. The variant walks the chunk in
    fixed-height sub-chunks and reads older rows through a running intra-chunk
    state — so the intra-chunk score tile stops growing with the square of
    ``chunk_size`` — and lands its rows in token order inside the same stage, so
    every output row is moved exactly once. Both paths emit the same values.

    ``enable_chunk_fwd_h_kernel_varlen_variant`` selects the chunk-state stage's
    schedule. The incumbent (False) walks the chunk axis sequentially inside one
    Pallas program per head, carrying the state in scratch. The variant unrolls
    that recurrence into its closed form and resolves the whole carry as a single
    per-head decay matrix, so the number of launched steps stops scaling with the
    chunk count. Both paths emit the same states.

    ``output_impl`` selects which handle executes the output stage.
    ``"incumbent"`` keeps ``chunk_fwd_o`` and the schedule toggles documented
    above; ``"batched_value_tiles"`` dispatches to the standalone
    ``batched_value_tile_fwd_o`` handle, which runs the same two-level walk with
    the value-tile axis as a batch dimension of every contraction instead of a
    Python loop bound (see simple_gla_batched_output.py). That handle owns the
    whole output schedule, so it consumes no output schedule toggle.

    ``chunk_impl`` selects which handle executes the WHOLE chunked recurrence.
    ``"incumbent"`` keeps the shipped two-stage pipeline (state stage, then
    output stage, communicating through the ``(NT, H, K, V)`` entrance-state
    buffer) together with every schedule toggle documented above;
    ``"decay_rescaled"`` dispatches to the standalone ``decay_rescaled_chunk_fwd``
    handle, which folds the scalar gate into a per-sub-chunk rescaling of ``k``
    and a deferred per-row scaling of the output, so the gate ratio tile
    disappears and the single ``k^T v`` contraction serves both the sub-chunk
    states and the chunk carry (see simple_gla_decay_rescaled.py). That handle
    owns both stages, so it consumes neither the state nor the output schedule
    toggle and emits the final recurrent state itself.

    ``walk_impl`` selects which handle executes the whole recurrence as a
    gate-complete walk. ``"incumbent"`` changes nothing. ``"fused_walk"``
    dispatches to the standalone ``fused_walk_chunk_fwd`` handle (see
    simple_gla_fused_walk.py), which carries BOTH halves of the factored gate
    inside the staged operands -- the row half and ``scale`` ride on ``q`` -- so
    no contraction result is rescaled afterwards, and which issues the sub-chunk
    prefix and the chunk carry as ONE contraction against the shared ``k^T v``
    by stacking the carry weights as the last row of the decay operator; the
    whole walk is contracted at the reduced pass count.
    ``"fused_walk_exact_state"`` is the same walk with that merged state/carry
    operator -- the one contraction whose result propagates across chunks and
    into decode -- kept at the full pass count. Like ``chunk_impl``, this handle
    owns both stages and consumes neither stage's schedule toggle.
    """
    # Imported here, not at module scope: the handles reuse this module's
    # helpers, so a top-level import would be circular.
    from sgl_jax.srt.kernels.simple_gla.simple_gla_batched_output import (
        batched_value_tile_fwd_o,
    )
    from sgl_jax.srt.kernels.simple_gla.simple_gla_decay_rescaled import (
        decay_rescaled_chunk_fwd,
    )
    from sgl_jax.srt.kernels.simple_gla.simple_gla_fused_walk import (
        fused_walk_chunk_fwd,
    )

    B, T_orig, H, K, V = *q.shape, v.shape[-1]
    N = cu_seqlens_dev.shape[0] - 1 if cu_seqlens_dev is not None else B

    assert_shape(q, (B, T_orig, H, K))
    assert_shape(k, (B, T_orig, H, K))
    assert_shape(v, (B, T_orig, H, V))
    assert_shape_or_none(g, (B, T_orig, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(h0, (N, H, K, V))
    assert cu_seqlens_cpu is None, "cu_seqlens_cpu must be None."
    assert cu_seqlens_dev is not None, "cu_seqlens_dev must not be None."
    assert (K % 128 == 0) and (V % 128 == 0)
    assert B == 1, "B must be 1."

    T_aligned = _compute_t_aligned(
        T_orig,
        N,
        chunk_size,
        compact_alignment=compact_alignment,
    )
    q_a, k_a, v_a, aligned_cu, real_seq_lens = _align_varlen_inputs(
        q,
        k,
        v,
        cu_seqlens_dev,
        chunk_size,
        T_aligned,
    )

    if zero_state_output_elision:
        assert single_chunk_state_elision, (
            "zero-state output elision requires single-chunk state elision"
        )

    # Gate-complete whole-recurrence dispatch. The fused walk owns the state
    # stage and the output stage together, and additionally carries BOTH halves
    # of the factored gate inside its staged operands, so no contraction result
    # is rescaled afterwards and the sub-chunk prefix and the chunk carry are one
    # contraction against the shared k^T v.
    walk_impl_handle = (
        fused_walk_chunk_fwd
        if walk_impl in ("fused_walk", "fused_walk_exact_state")
        else None
    )
    # The merged state/carry operator is the one contraction whose result leaves
    # the chunk, so its pass count is selectable independently of the four large
    # contractions the walk reduces.
    walk_exact_state_operator = walk_impl == "fused_walk_exact_state"
    if walk_impl_handle is not None:
        assert chunk_impl == "incumbent", (
            "gla.walk_impl and gla.chunk_impl both own the whole recurrence"
        )
        assert not single_chunk_state_elision, (
            "the fused walk owns the state stage, so it does not consume the "
            "state-elision toggles"
        )
        o, ht = walk_impl_handle(
            q_a,
            k_a,
            v_a,
            g_gamma=g_gamma,
            h0=h0,
            scale=scale,
            cu_seqlens_dev=aligned_cu,
            seq_real_lens=real_seq_lens,
            chunk_size=chunk_size,
            output_value_tiles=output_value_tiles,
            output_final_state=use_ht,
            output_gather=_build_unalign_gather_idx(
                cu_seqlens_dev, aligned_cu, T_orig
            ),
            exact_state_operator=walk_exact_state_operator,
        )
        if use_ht and ht is not None:
            # Same zero-length repair the two-stage path performs: a sequence
            # with no real token keeps its incoming state.
            zero_len_mask = (real_seq_lens == 0)[:, None, None, None]
            ht = jnp.where(zero_len_mask, h0 if h0 is not None else 0.0, ht)
        return o, ht

    # Whole-recurrence dispatch. The decay-rescaled handle owns the state stage
    # and the output stage together, so it replaces both launches and the state
    # buffer between them rather than sitting inside either one.
    chunk_impl_handle = (
        decay_rescaled_chunk_fwd if chunk_impl == "decay_rescaled" else None
    )
    if chunk_impl_handle is not None:
        assert not single_chunk_state_elision, (
            "the decay-rescaled forward owns the state stage, so it does not "
            "consume the state-elision toggles"
        )
        o, ht = chunk_impl_handle(
            q_a,
            k_a,
            v_a,
            g_gamma=g_gamma,
            h0=h0,
            scale=scale,
            cu_seqlens_dev=aligned_cu,
            seq_real_lens=real_seq_lens,
            chunk_size=chunk_size,
            output_value_tiles=output_value_tiles,
            output_final_state=use_ht,
            output_gather=_build_unalign_gather_idx(
                cu_seqlens_dev, aligned_cu, T_orig
            ),
        )
        if use_ht and ht is not None:
            # Same zero-length repair the two-stage path performs: a sequence
            # with no real token keeps its incoming state.
            zero_len_mask = (real_seq_lens == 0)[:, None, None, None]
            ht = jnp.where(zero_len_mask, h0 if h0 is not None else 0.0, ht)
        return o, ht

    if single_chunk_state_elision:
        assert N == 1, "single-chunk state elision requires exactly one sequence"
        assert T_aligned == chunk_size, (
            "single-chunk state elision requires one aligned chunk; "
            f"got T_aligned={T_aligned}, chunk_size={chunk_size}"
        )
        assert h0 is None, "single-chunk state elision requires h0=None"
        assert not use_ht, "single-chunk state elision cannot return a final state"
        # chunk_fwd_h stores the state at each chunk's *entrance*.  The sole
        # entrance state is exactly zero; its post-update state is terminal and
        # unobserved when use_ht=False, so no state Pallas launch is necessary.
        h = (
            None
            if zero_state_output_elision
            else jnp.zeros((1, H, K, V), dtype=k_a.dtype)
        )
        ht = None
    else:
        h, ht = chunk_fwd_h_kernel_varlen(
            k=k_a,
            v=v_a,
            g=g,
            g_gamma=g_gamma,
            gk=None,
            gv=None,
            h0=h0,
            output_final_state=use_ht,
            states_in_fp32=False,
            cu_seqlens_dev=aligned_cu,
            chunk_size=chunk_size,
            seq_real_lens=real_seq_lens,
            enable_chunk_fwd_h_kernel_varlen_variant=(
                enable_chunk_fwd_h_kernel_varlen_variant
            ),
        )
    # Pallas output buffers are NOT zero-initialized on TPU. Zero-length
    # sequences are skipped by @pl.when(bos != eos), leaving their ht
    # entries undefined. Replace with h0 (or zeros) so downstream scatter
    # doesn't write garbage into the recurrent state pool.
    if use_ht and ht is not None:
        zero_len_mask = (real_seq_lens == 0)[:, None, None, None]
        if h0 is not None:
            ht = jnp.where(zero_len_mask, h0, ht)
        else:
            ht = jnp.where(zero_len_mask, 0.0, ht)
    # The resident output schedule restores the token layout inside the output
    # stage, so the separate aligned-space scatter pass is not launched at all.
    chunk_fwd_o_handle = (
        batched_value_tile_fwd_o if output_impl == "batched_value_tiles" else None
    )
    # The batched handle always delivers in token order, exactly as the resident
    # schedule does, so both restore the layout inside the output stage and the
    # separate aligned-space scatter pass is not launched at all.
    unalign_gather = (
        _build_unalign_gather_idx(cu_seqlens_dev, aligned_cu, T_orig)
        if (enable__chunk_fwd_o_pl_variant or chunk_fwd_o_handle is not None)
        else None
    )
    if chunk_fwd_o_handle is not None:
        o = chunk_fwd_o_handle(
            q=q_a,
            k=k_a,
            v=v_a,
            g=g,
            g_gamma=g_gamma,
            h=h,
            scale=scale,
            chunk_size=chunk_size,
            output_value_tiles=output_value_tiles,
            output_gather=unalign_gather,
        )
    else:
        o = chunk_fwd_o(
            q=q_a,
            k=k_a,
            v=v_a,
            g=g,
            g_gamma=g_gamma,
            h=h,
            scale=scale,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cu_seqlens_dev=aligned_cu,
            chunk_size=chunk_size,
            zero_state_output_elision=zero_state_output_elision,
            output_value_tiles=output_value_tiles,
            enable__chunk_fwd_o_pl_variant=enable__chunk_fwd_o_pl_variant,
            output_gather=unalign_gather,
        )

    if unalign_gather is None:
        o = _unalign_output(o, cu_seqlens_dev, aligned_cu, T_orig)
    return o, ht


class SimpleGLAKernelMode(enum.Enum):
    """Simple GLA kernel implementation mode."""

    CHUNK = "chunk"
    FUSED_CHUNK = "fused_chunk"


def simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    compact_alignment: bool = False,
    single_chunk_state_elision: bool = False,
    zero_state_output_elision: bool = False,
    output_value_tiles: int = 1,
    enable__chunk_fwd_o_pl_variant: bool = False,
    enable_chunk_fwd_h_kernel_varlen_variant: bool = False,
    output_impl: str = "incumbent",
    chunk_impl: str = "incumbent",
    walk_impl: str = "incumbent",
    mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK,
):
    if cu_seqlens_dev is None:
        raise NotImplementedError(
            f"Non-varlen simple_gla_fwd (mode={mode}) is not vendored. "
            "Only the varlen path (cu_seqlens_dev != None) is supported."
        )
    return chunk_simple_gla_fwd_varlen(
        q,
        k,
        v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        h0=h0,
        use_ht=use_ht,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_dev=cu_seqlens_dev,
        chunk_size=chunk_size,
        compact_alignment=compact_alignment,
        single_chunk_state_elision=single_chunk_state_elision,
        zero_state_output_elision=zero_state_output_elision,
        output_value_tiles=output_value_tiles,
        enable__chunk_fwd_o_pl_variant=enable__chunk_fwd_o_pl_variant,
        enable_chunk_fwd_h_kernel_varlen_variant=(
            enable_chunk_fwd_h_kernel_varlen_variant
        ),
        output_impl=output_impl,
        chunk_impl=chunk_impl,
        walk_impl=walk_impl,
    )
