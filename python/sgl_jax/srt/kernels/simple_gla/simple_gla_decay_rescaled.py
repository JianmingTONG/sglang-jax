"""Decay-rescaled single-pass GLA chunk forward (standalone JAX-callable handle).

The shipped chunked prefill spends its time carrying the gate around as an
explicit exponential tile and computing the same contraction twice.

*The gate as a tile.* ``simple_gla``'s output stage forms the intra-sub-chunk
scores as a raw ``q @ k^T`` and then multiplies it by a full
``(G, num_sub, sub, sub)`` matrix of gate ratios: it materializes
``log_diff = s_log[..., :, None] - s_log[..., None, :]``, exponentiates it, and
multiplies -- three passes over a tile that is as large as the score tile itself
(8 MiB at ``tiny-linear-serving/gla-long``) -- and then pays a fourth pass to
rescale the inter-chunk term by ``exp(s_log - entrance)``.

*The contraction, twice.* The gate ``g_gamma`` is a per-head SCALAR decay, so
``exp(s_i - s_j)`` factors as ``exp(s_i - a) * exp(a - s_j)`` around ANY pivot
``a``. Rescaling the rows once -- ``q_hat = q * exp(s - a)``,
``k_hat = k * exp(a - s)`` with ``a`` the sub-chunk entrance -- makes
``q_hat . k_hat`` already carry the decay, so the score tile needs only its
causal MASK, the inter-chunk term needs no rescaling pass at all (``q_hat``
already carries it), and ``k_hat^T v`` is simultaneously the sub-chunk's state
contribution. That last identity is what removes the second contraction: the
shipped pipeline computes ``k^T v`` once inside ``chunk_fwd_h_kernel_varlen``
(over its own ``(H, T, K)``/``(H, T, V)`` transposes, to build the chunk
entrance states) and again inside the output stage (per sub-chunk, to build the
sub-chunk entrance states). Here the sub-chunk contributions ``w = k_hat^T v``
are summed onto the chunk's real exit and become the chunk carry itself, so the
state stage, its transposes, its ``(NT, H, K, V)`` buffer and its round trip all
disappear: one launch consumes ``q/k/v`` and emits both the token output and the
final recurrent state.

Numerically this is the same sum re-associated. Every exponent formed on the
``q`` side and on the state side is a difference of a later minus an earlier
position (non-positive); the only growing factor is ``exp(a - s_j)`` on
``k_hat``, whose span is bounded by ONE sub-chunk (``|gamma| * sub``, i.e. about
``e^12.8`` for the repo's gate range) and which always meets its matching
``exp(s_i - a)`` inside the same dot product, so the retained products are
identical to the shipped ratio form and the discarded (strictly upper) ones are
masked away.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.simple_gla.simple_gla import (
    _OUTPUT_SUBCHUNK_ROWS,
    _build_chunk_map,
    assert_shape,
    assert_shape_or_none,
    exp,
    get_interpret,
)

_HIGHEST = jax.lax.Precision.HIGHEST


def decay_rescaled_subchunk_rows(chunk_rows: int) -> int:
    """Sub-chunk height of the decay-rescaled walk.

    The pivot of the rescaling is the sub-chunk entrance, so the sub-chunk height
    is exactly the span of the one growing exponent this schedule forms. A chunk
    no taller than one sub-chunk is its own sub-chunk.
    """
    return min(_OUTPUT_SUBCHUNK_ROWS, chunk_rows)


def _decay_rescaled_kernel(
    q_ref,
    k_ref,
    v_ref,
    h0_ref,
    g_gamma_ref,
    cu_ref,
    chunk_to_seq_ref,
    real_lens_ref,
    scale_ref,
    o_ref,
    ht_ref,
    state_ref,
    *,
    BT: int,
    OUTPUT_VALUE_TILES: int,
    SUBCHUNK_ROWS: int,
):
    """One chunk of ``G`` head-value tiles: output and state carry in one body.

    The whole body is STRAIGHT-LINE. The sequence-entrance reset, the padded tail
    and the final-state write are selects and masked weights rather than
    ``pl.when`` regions, because a predicated region costs more here than the
    stage it would guard.
    """
    G = OUTPUT_VALUE_TILES
    sub = SUBCHUNK_ROWS
    num_sub = BT // sub
    K = q_ref.shape[-1]
    value_width = v_ref.shape[-1]
    f32 = jnp.float32
    scale = scale_ref[0].astype(f32)

    i_tile, i_t = pl.program_id(0), pl.program_id(1)
    first_value_tile = i_tile * G
    # g_gamma lives in SMEM/ANY and is read one scalar per tile, exactly as the
    # shipped output stage reads it; the stack is what makes it a batch axis.
    gamma = jnp.stack(
        [g_gamma_ref[first_value_tile + i].astype(f32) for i in range(G)]
    )  # (G,)

    seq_idx = chunk_to_seq_ref[i_t]
    bos = cu_ref[seq_idx]
    eos = cu_ref[seq_idx + 1]
    t0 = i_t * BT
    # Real (non-padded) occupancy of this chunk, matching the shipped state
    # stage's `L_chunk = min(BT, effective_remaining)`; an empty sequence
    # contributes nothing and does not decay.
    real_eos = bos + real_lens_ref[seq_idx]
    L = jnp.clip(real_eos - t0, 0, BT).astype(f32)
    L = jnp.where(bos != eos, L, 0.0)

    # Chunk entrance state: the row's own sequence start reads h0, every later
    # chunk reads the carry. `jnp.where` is a select, so the untaken branch is
    # never arithmetic on the (still unwritten) scratch of the first step.
    zero_state = jnp.zeros((G, K, value_width), f32)
    h0_tile = zero_state if h0_ref is None else h0_ref[:, seq_idx].astype(f32)
    state_in = jnp.where(t0 == bos, h0_tile, state_ref[...])

    row = jnp.arange(sub, dtype=f32)
    sub_mask = jnp.arange(sub)[:, None] >= jnp.arange(sub)[None, :]
    older = jnp.arange(num_sub)[:, None] > jnp.arange(num_sub)[None, :]

    # Chunk-cumulative log decay per tile and row, and the log at each
    # sub-chunk's entrance -- the pivot of the rescaling.
    s_log = (
        gamma[:, None, None]
        * (jnp.arange(num_sub, dtype=f32)[None, :, None] * sub + row[None, None, :] + 1.0)
    )  # (G, num_sub, sub)
    entrance_log = gamma[:, None] * (
        jnp.arange(num_sub, dtype=f32)[None, :] * sub
    )  # (G, num_sub)

    s_q = q_ref[:, 0].reshape(G, num_sub, sub, K)
    s_k = k_ref[:, 0].reshape(G, num_sub, sub, K)
    s_v = v_ref[:, 0].astype(f32).reshape(G, num_sub, sub, value_width)

    # Only the COLUMN half of the factored gate is ever materialized. The row
    # half `exp(s_i - a)` is a per-row scalar that multiplies the intra term and
    # the inter term alike, so it rides out to the single output scaling below
    # instead of paying its own pass over a (G, num_sub, sub, K) tile.
    k_hat = s_k.astype(f32) * exp(entrance_log[:, :, None] - s_log)[..., None]

    # Intra-sub-chunk scores: the column gate is already inside the operand, so
    # this tile needs its causal mask and nothing else.
    s_A = jnp.einsum("gnik,gnjk->gnij", s_q.astype(f32), k_hat,
                     preferred_element_type=f32)
    s_A = jnp.where(sub_mask, s_A, 0.0)

    # Each sub-chunk's contribution, expressed at its own entrance pivot. This is
    # the ONLY k-v contraction in the pipeline: it feeds both the sub-chunk state
    # recurrence below and the chunk carry at the end.
    b_w = jnp.einsum(
        "gnik,gnid->gnkd",
        k_hat,
        s_v,
        precision=_HIGHEST,
        preferred_element_type=f32,
    )

    # Sub-chunk entrance states in closed form: sub-chunk n reads every strictly
    # older sub-chunk rebased from that sub-chunk's entrance onto its own, plus
    # the chunk entrance. The entrance state enters as one extra column of the
    # same decay matrix, so it costs a contraction column instead of a separate
    # broadcast-add pass over a (G, num_sub, K, value_width) tile.
    decay_matrix = jnp.concatenate(
        [
            jnp.where(
                older,
                exp(entrance_log[:, :, None] - entrance_log[:, None, :]),
                0.0,
            ),
            exp(entrance_log)[:, :, None],
        ],
        axis=2,
    )  # (G, num_sub, num_sub + 1)
    b_state = jnp.einsum(
        "gnb,gbkd->gnkd",
        decay_matrix,
        jnp.concatenate([b_w, state_in[:, None]], axis=1),
        precision=_HIGHEST,
        preferred_element_type=f32,
    )

    b_o = jnp.einsum(
        "gnij,gnjd->gnid",
        s_A,
        s_v,
        precision=_HIGHEST,
        preferred_element_type=f32,
    )
    b_o = b_o + jnp.einsum(
        "gnik,gnkd->gnid", s_q.astype(f32), b_state, preferred_element_type=f32
    )
    # The deferred row half of the gate, applied once to the summed row.
    row_gate = scale * exp(s_log - entrance_log[:, :, None])
    o_ref[:, 0] = (b_o * row_gate[..., None]).reshape(G, BT, value_width).astype(
        o_ref.dtype
    )

    # Chunk carry: the same sub-chunk contributions, rebased from their entrance
    # onto the chunk's REAL exit. A sub-chunk beyond the real tail contributes
    # exactly zero (its k rows are zero), so clamping its weight to one is safe
    # and keeps every formed exponent non-positive.
    exit_log = gamma * L  # (G,)
    carry = exp(jnp.minimum(exit_log[:, None] - entrance_log, 0.0))  # (G, num_sub)
    state_out = exp(exit_log)[:, None, None] * state_in + jnp.einsum(
        "gb,gbkd->gkd",
        carry,
        b_w,
        precision=_HIGHEST,
        preferred_element_type=f32,
    )
    state_ref[...] = state_out
    if ht_ref is not None:
        # The final state of a sequence is written at its last chunk; padded
        # chunks past the real tail carry L = 0 and leave the state untouched,
        # so a repeated write stores the identical value.
        ht_ref[:, seq_idx] = jnp.where(
            t0 + BT >= eos, state_out, ht_ref[:, seq_idx].astype(f32)
        ).astype(ht_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size", "output_value_tiles", "output_final_state"),
)
def decay_rescaled_chunk_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g_gamma: jax.Array,
    h0: jax.Array | None = None,
    scale: float | None = None,
    cu_seqlens_dev: jax.Array,
    seq_real_lens: jax.Array | None = None,
    chunk_size: int = 64,
    output_value_tiles: int = 1,
    output_final_state: bool = False,
    output_gather: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Whole chunked GLA forward as one launch, with the gate folded into q/k.

    Same contract and same emitted values as the shipped
    ``chunk_fwd_h_kernel_varlen`` + output-stage pair: token output in the
    ``output_gather`` order plus the final recurrent state. The handle owns the
    entire recurrence, so it consumes no state or output schedule toggle; it
    requires the scalar per-head gate (``g``/``gk`` gate tensors keep the shipped
    two-stage path).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT
    N = cu_seqlens_dev.shape[0] - 1

    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(h0, (N, H, K, V))
    assert g_gamma is not None, "the decay-rescaled forward requires g_gamma."
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    if scale is None:
        scale = K**-0.5
    sub = decay_rescaled_subchunk_rows(BT)
    assert BT % sub == 0, f"chunk_size={BT} must be a multiple of {sub}"

    def _reshape_bt(x, D):
        return (
            x.reshape(B, NT, BT, H, D)
            .transpose(3, 0, 1, 2, 4)
            .reshape(H, total_NT, BT, D)
        )

    _q = _reshape_bt(q, K)
    _k = _reshape_bt(k, K)
    _v = _reshape_bt(v, V)

    BV = 128 if V % 128 == 0 else V
    num_v_tiles = V // BV
    _h0 = None
    if h0 is not None:
        _h0 = h0.transpose(1, 0, 2, 3)  # (H, N, K, V)
    if num_v_tiles > 1:
        _v = (
            _v.reshape(H, total_NT, BT, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, BT, BV)
        )
        if _h0 is not None:
            _h0 = (
                _h0.reshape(H, N, K, num_v_tiles, BV)
                .transpose(0, 3, 1, 2, 4)
                .reshape(H * num_v_tiles, N, K, BV)
            )
        g_gamma = jnp.repeat(g_gamma, num_v_tiles)
        _q = jnp.repeat(_q, num_v_tiles, axis=0)
        _k = jnp.repeat(_k, num_v_tiles, axis=0)

    H_VT = H * num_v_tiles
    assert output_value_tiles > 0, "output_value_tiles must be positive"
    assert H_VT % output_value_tiles == 0, (
        f"H*num_v_tiles={H_VT} must be divisible by "
        f"output_value_tiles={output_value_tiles}"
    )

    G = output_value_tiles
    chunk_to_seq = _build_chunk_map(
        cu_seqlens=cu_seqlens_dev, T_sum=B * T, BT=BT
    )
    if seq_real_lens is None:
        seq_real_lens = cu_seqlens_dev[1:] - cu_seqlens_dev[:-1]

    grid = (H_VT // G, total_NT)
    interpret = get_interpret()

    def _tile_spec(width, rows):
        return pl.BlockSpec(
            (G, 1, rows, width),
            index_map=lambda value_tile_block, nt_idx: (value_tile_block, nt_idx, 0, 0),
        )

    def _seq_spec(width, rows):
        return pl.BlockSpec(
            (G, N, rows, width),
            index_map=lambda value_tile_block, nt_idx: (value_tile_block, 0, 0, 0),
        )

    smem = pl.BlockSpec(memory_space=pltpu.ANY if interpret else pltpu.SMEM)
    out_shape = [jax.ShapeDtypeStruct((H_VT, total_NT, BT, BV), v.dtype)]
    out_specs = [_tile_spec(BV, BT)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct((H_VT, N, K, BV), jnp.float32))
        out_specs.append(_seq_spec(BV, K))
    else:
        out_shape.append(None)
        out_specs.append(None)

    o, ht = pl.pallas_call(
        functools.partial(
            _decay_rescaled_kernel,
            BT=BT,
            OUTPUT_VALUE_TILES=G,
            SUBCHUNK_ROWS=sub,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                _tile_spec(K, BT),
                _tile_spec(K, BT),
                _tile_spec(BV, BT),
                None if _h0 is None else _seq_spec(BV, K),
                smem,
                smem,
                smem,
                smem,
                smem,
            ],
            out_specs=out_specs,
            scratch_shapes=[pltpu.VMEM((G, K, BV), jnp.float32)],
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            # The chunk axis carries the recurrent state in scratch, so it is
            # sequential; the value-tile axis stays independent.
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=128 * 1024 * 1024,
        ),
        interpret=interpret,
    )(
        _q,
        _k,
        _v,
        _h0,
        g_gamma,
        cu_seqlens_dev,
        chunk_to_seq,
        seq_real_lens,
        jnp.asarray(scale, dtype=jnp.float32).reshape(1),
    )

    if num_v_tiles > 1:
        o = (
            o.reshape(H, num_v_tiles, total_NT, BT, BV)
            .transpose(0, 2, 3, 1, 4)
            .reshape(H, total_NT, BT, V)
        )
    o = o.reshape(H, B, NT, BT, V).transpose(1, 2, 3, 0, 4).reshape(B, T, H, V)
    if output_gather is not None:
        # Single-move token-order delivery: the permutation is expressed over the
        # OUTPUT index space, so each row moves exactly once and no aligned
        # staging pass is launched.
        o = o[:, output_gather]
    if ht is not None:
        if num_v_tiles > 1:
            ht = (
                ht.reshape(H, num_v_tiles, N, K, BV)
                .transpose(0, 2, 3, 1, 4)
                .reshape(H, N, K, V)
            )
        ht = ht.transpose(1, 0, 2, 3)
    return o, ht
