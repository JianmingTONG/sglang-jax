"""Fused resident-state KDA prefill pipeline (standalone JAX-callable handle).

The shipped `chunk_kda_fwd` path runs the chunked delta rule as four separate
Pallas launches — gate cumsum, intra-chunk triangular solve, inter-chunk state
propagation, output — and hands every intermediate between them through HBM:

  * stage 2 solves `(I - L) x = [v*beta | k*exp2(g)*beta | I]`, i.e. `V + K + BT`
    right-hand-side columns, so that stage 3 can later form `v_new = u - w @ h`
    and stage 4 can re-read `Aqk`.  `u`, `w` and the explicit `A_inv` are all
    materialized as `[NC, BT, *]` tensors;
  * stages 2 and 4 address their chunks through a RUNTIME start-offset vector
    (`chunk_starts`): each stage gathers its tiles with a vmapped
    `dynamic_slice` and scatter-adds its results back into `T + BT`-row staging
    buffers through an `[NC, BT]` int32 position vector.

This module keeps the same recurrence but executes it as ONE pass with the
chunk state resident:

  * `v_new = (I - L)^-1 (v*beta - k*exp2(g)*beta @ h)`.  The solve is a linear
    operator, so folding stage 3's `u - w @ h` into the right-hand side is
    exact; the pass then solves `V` columns instead of `V + K + BT`, and
    neither `u`, nor `w`, nor `A_inv` is ever formed.  The chunk's entrance
    state `h` is what makes this possible, which is why it only exists once the
    intra solve and the state recurrence live in the same program.
  * the state stays in a VMEM scratch across the chunks of a group and across
    the grid's sequential block axis, so the `[NC, BT, BT]` operator, the
    `[NC, BT, K]` `w`, and the per-chunk entrance states never reach HBM.
  * every tile is placed by a `BlockSpec` index map over the BT-blocked time
    axis (`_align_seqs` guarantees each sequence starts on a group boundary),
    so the deterministic gather/scatter index round trip disappears entirely.

The handle is deliberately NARROWER than `chunk_kda_fwd`: owning the whole
chunk pipeline, it fixes the micro-schedule the incumbent stages expose
separately (`compute_block_chunks`, `state_block_chunks`, `scalar_intra_solve`)
and consumes only the axes it still honours.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.kda.kda import (
    _RCP_LN2,
    _align_seqs,
    _solve_unit_lower_triangular,
    _unalign_output,
    assert_shape,
    assert_shape_or_none,
    exp2,
    get_interpret,
    kda_gate_chunk_cumsum,
    pallas_kda_gate_cumsum,
    prepare_chunk_indices,
)

_HIGHEST = jax.lax.Precision.HIGHEST

# How many logical BT chunks one physical program keeps resident. The group is
# the unit the varlen layout is aligned to, so it is capped at a divisor of the
# workload's chunk count: a group never forces padding the packed sequence.
_GROUP_CHUNKS = 4


def resident_group_chunks(num_chunks: int, group_chunks: int = _GROUP_CHUNKS) -> int:
    """Largest group <= ``group_chunks`` that exactly divides the chunk count."""
    group = max(1, int(group_chunks))
    if num_chunks <= 0:
        return 1
    group = min(group, num_chunks)
    while group > 1 and num_chunks % group:
        group -= 1
    return group


def _resident_pipeline_kernel(
    seqlens_ref,
    q_ref,
    k_ref,
    v_ref,
    gk_ref,
    beta_ref,
    h0_ref,
    o_ref,
    ht_ref,
    state_ref,
    *,
    BT,
    GROUP,
    K,
    V,
    scale,
    intra_block_size,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
):
    idx_n = pl.program_id(0)
    idx_nb = pl.program_id(2)

    bos = seqlens_ref[idx_n]
    eos = seqlens_ref[idx_n + 1]
    real_NT = (eos - bos) // BT

    @pl.when(idx_nb == 0)
    def _():
        if USE_INITIAL_STATE:
            state_ref[...] = h0_ref[0, 0].astype(jnp.float32)
        else:
            state_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)

    causal_bt = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    strict_bt = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32), k=-1)

    # Sequences are aligned to GROUP * BT, so a group is either entirely inside
    # this sequence or entirely past its end — one predicate covers the group.
    @pl.when(idx_nb * GROUP < real_NT)
    def _():
        for i_chunk in range(GROUP):
            rows = dslice(i_chunk * BT, BT)
            q = q_ref[0, 0, rows, :].astype(jnp.float32)
            k = k_ref[0, 0, rows, :].astype(jnp.float32)
            v = v_ref[0, 0, rows, :].astype(jnp.float32)
            g = gk_ref[0, 0, rows, :].astype(jnp.float32)
            beta = beta_ref[0, 0, rows, 0:1].astype(jnp.float32)

            # Same gated intra-chunk operators as the shipped stage 2: build the
            # decay from exp2(g[i] - g[j]) directly so a per-step gate larger
            # than ~127 cannot overflow the split normalization.
            g_diff = g[:, None, :] - g[None, :, :]
            g_diff = jnp.where(causal_bt[:, :, None] > 0, g_diff, -126.0)
            decay = exp2(jnp.maximum(g_diff, -126.0))

            Aqk = scale * jnp.sum(q[:, None, :] * decay * k[None, :, :], axis=-1)
            Aqk = Aqk * causal_bt
            L = (
                jnp.sum(k[:, None, :] * decay * k[None, :, :], axis=-1)
                * beta
                * strict_bt
            )

            # The chunk's entrance state is already here, so the delta-rule
            # correction enters the solve's RHS instead of being applied to a
            # separately solved `w` afterwards.
            state = state_ref[...]
            k_eg_beta = k * exp2(g) * beta
            rhs = v * beta - jnp.dot(
                k_eg_beta,
                state,
                precision=_HIGHEST,
                preferred_element_type=jnp.float32,
            )
            v_new = _solve_unit_lower_triangular(L, rhs, block_size=intra_block_size)

            # Inter-chunk term against the entrance state, then the intra term.
            # g[0] is the largest cumsum in the chunk; factoring it into the
            # state keeps both exponentials in (0, 1].
            g_head = g[0:1, :]
            qg = q * exp2(jnp.maximum(g - g_head, -126.0))
            state_head = state * exp2(jnp.maximum(g_head[0], -126.0))[:, None]
            o = jnp.dot(
                qg, state_head, precision=_HIGHEST, preferred_element_type=jnp.float32
            )
            o = o * scale + jnp.dot(
                Aqk, v_new, precision=_HIGHEST, preferred_element_type=jnp.float32
            )
            o_ref[0, 0, rows, :] = o.astype(o_ref.dtype)

            # Advance the resident state over this chunk.
            g_last = g[BT - 1]
            kg = k * exp2(g_last[None, :] - g)
            state_ref[...] = state * exp2(g_last)[:, None] + jnp.dot(
                kg.T, v_new, precision=_HIGHEST, preferred_element_type=jnp.float32
            )

    if STORE_FINAL_STATE:

        @pl.when((idx_nb + 1) * GROUP == real_NT)
        def _():
            ht_ref[0, 0] = state_ref[...].astype(ht_ref.dtype)


def resident_pipeline_stage(
    q,
    k,
    v,
    gk,
    beta,
    initial_state,
    scale,
    cu_seqlens,
    *,
    chunk_size,
    group,
    intra_block_size,
    output_final_state,
):
    """One fused Pallas launch: intra solve + state recurrence + output."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    GBT = BT * group
    N = cu_seqlens.shape[0] - 1
    assert T % GBT == 0, f"T={T} must be divisible by the group extent {GBT}"

    # One sentinel group so a clamped index map always lands in range.
    T_alloc = T + GBT

    def _pack(x):
        x = jnp.pad(x, ((0, 0), (0, GBT), (0, 0), (0, 0)))
        return jnp.transpose(x, (0, 2, 1, 3)).astype(jnp.float32)

    q_t, k_t, v_t, gk_t = _pack(q), _pack(k), _pack(v), _pack(gk)
    beta_t = _pack(beta.reshape(B, T, H, 1))
    h0 = None if initial_state is None else initial_state.astype(jnp.float32)

    def _t_index_map(n, h, nb, seqlens_ref):
        bos = pl.multiple_of(seqlens_ref[n], GBT)
        block_idx = jnp.minimum(bos // GBT + nb, T // GBT)
        return (0, h, block_idx, 0)

    def _state_index_map(n, h, nb, seqlens_ref):
        return (n, h, 0, 0)

    def _time_spec(width):
        return pl.BlockSpec([1, 1, GBT, width], index_map=_t_index_map)

    state_spec = pl.BlockSpec([1, 1, K, V], index_map=_state_index_map)
    o_shape = jax.ShapeDtypeStruct([B, H, T_alloc, V], jnp.float32)
    ht_shape = (
        jax.ShapeDtypeStruct([N, H, K, V], jnp.float32) if output_final_state else None
    )

    o_out, ht_out = pl.pallas_call(
        functools.partial(
            _resident_pipeline_kernel,
            BT=BT,
            GROUP=group,
            K=K,
            V=V,
            scale=scale,
            intra_block_size=intra_block_size,
            USE_INITIAL_STATE=h0 is not None,
            STORE_FINAL_STATE=output_final_state,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(N, H, T // GBT),
            in_specs=[
                _time_spec(K),
                _time_spec(K),
                _time_spec(V),
                _time_spec(K),
                _time_spec(1),
                None if h0 is None else state_spec,
            ],
            out_specs=[
                _time_spec(V),
                state_spec if output_final_state else None,
            ],
            scratch_shapes=[pltpu.VMEM((K, V), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        out_shape=[o_shape, ht_shape],
        interpret=get_interpret(),
    )(cu_seqlens.astype(jnp.int32), q_t, k_t, v_t, gk_t, beta_t, h0)

    return jnp.transpose(o_out[:, :, :T, :], (0, 2, 1, 3)), ht_out


def resident_pipeline_kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    initial_state: jax.Array,
    output_final_state: bool,
    cu_seqlens: jax.Array,
    *,
    chunk_size: int = 64,
    intra_block_size: int = 16,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
):
    """Fused resident-state KDA prefill for variable-length (packed) sequences.

    Same contract as ``chunk_kda_fwd``: ``cu_seqlens`` must not be None and B
    must be 1. Returns the same 12-tuple; the intermediates this pipeline never
    materializes (``Aqk``, ``Akk``, ``w``, ``u``, ``qg``, ``kg``, ``v_new``,
    ``h``) are returned as None, exactly as the shipped entry already releases
    its own.
    """
    B, T, H, K = q.shape
    BT = chunk_size
    N = cu_seqlens.shape[-1] - 1

    assert cu_seqlens is not None, "cu_seqlens must not be None for varlen path"
    assert B == 1, f"varlen requires B=1 (packed layout), got B={B}"
    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, v.shape[-1]), "v")
    assert_shape(beta, (B, T, H), "beta")
    assert_shape_or_none(initial_state, (N, H, K, v.shape[-1]), "initial_state")

    group = resident_group_chunks(T // BT if T % BT == 0 else 0)
    GBT = BT * group

    _orig_cu_seqlens = cu_seqlens
    T_input = T
    # A single packed sequence whose static extent already spans whole groups
    # needs no repacking; otherwise align every sequence to the group extent so
    # each group belongs to exactly one sequence.
    if not (N == 1 and T % GBT == 0):
        [q, k, v, g], [beta], cu_seqlens, _ = _align_seqs(
            [q, k, v, g], [beta], cu_seqlens, align=GBT
        )
    T = q.shape[1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, max_T=T)

    if use_gate_in_kernel:
        # _align_seqs pads g with 0 and softplus(0 + dt_bias) != 0, which would
        # leak gate activation into padding positions; neutralize them first.
        orig_lens = _orig_cu_seqlens[1:] - _orig_cu_seqlens[:-1]
        aligned_starts = cu_seqlens[:-1]
        pos = jnp.arange(T)
        in_range = (pos[None, :] >= aligned_starts[:, None]) & (
            pos[None, :] < (aligned_starts + orig_lens)[:, None]
        )
        g = jnp.where(in_range.any(axis=0)[None, :, None, None], g, -1e4)
        assert A_log is not None
        g_cumsum = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=BT,
            scale=_RCP_LN2,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        g_cumsum = pallas_kda_gate_cumsum(
            g=g,
            scale=_RCP_LN2,
            chunk_size=BT,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    o, final_state = resident_pipeline_stage(
        q,
        k,
        v,
        g_cumsum,
        beta,
        initial_state,
        scale,
        cu_seqlens,
        chunk_size=BT,
        group=group,
        intra_block_size=intra_block_size,
        output_final_state=output_final_state,
    )

    o = _unalign_output(o.astype(q.dtype), _orig_cu_seqlens, cu_seqlens, T_input)
    if use_gate_in_kernel:
        g_cumsum = None
    return (
        o,
        final_state,
        g_cumsum,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        initial_state,
    )
