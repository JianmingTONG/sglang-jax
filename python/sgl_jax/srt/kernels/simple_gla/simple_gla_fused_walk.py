"""Fused-operand GLA chunk walk (standalone JAX-callable handle).

The shipped chunked prefill still spends most of its time on contraction
*passes* it never had to issue, for two independent reasons.

*The gate rides outside the operands.* The decay-rescaled walk folds only the
COLUMN half of the gate into ``k``; the row half plus ``scale`` stays outside
every contraction and is applied afterwards, as a full elementwise pass over the
``(G, num_sub, sub, V)`` output tile (8 MiB at ``tiny-linear-serving/gla-long``).
Because the row factor is a per-ROW scalar it commutes with every contraction on
that row, so folding it -- together with ``scale`` -- into the staged ``q``
instead makes the whole walk operand-complete: after ``q_hat`` and ``k_hat`` are
formed, NO contraction result is ever rescaled again, and the trailing pass over
the output tile disappears into the staging fusion that was already reading
``q``.

*The same operand is contracted twice against two operators.* The sub-chunk
prefix ``b_state`` and the chunk carry are two separate contractions over the
SAME ``b_w = k_hat^T v`` and the SAME sub-chunk axis -- one weighted by the
strictly-older decay matrix, one by the exit-rebased carry weights. They are two
rows of one operator: stacking the carry as the last row of the decay matrix
issues them as a single ``(num_sub + 1, num_sub)`` contraction, so the walk
launches one contraction fewer and the carry's own reduction disappears.

*The pass count itself.* A contraction is not one matrix-unit operation. A TPU
MXU consumes bf16 operands, so an f32 x f32 dot is reproduced by SIX passes;
the error-compensated form is three, and which one a contraction gets is a
per-einsum ``jax.lax.Precision`` literal scattered through the shipped kernels
(the decay-rescaled walk pins three of its five contractions to ``HIGHEST`` and
leaves two at the unstated default). This handle makes the pass count ONE
explicit property of the walk. ``exact_state_operator`` keeps the single
contraction whose result propagates across chunks -- the merged state/carry
operator, whose output is the recurrent state consumed by decode -- at the full
pass count, so a deployment can buy back the state's last digits for about one
percent of the call while the four large contractions stay reduced.

Numerically nothing is re-associated relative to the decay-rescaled walk beyond
moving a per-row scalar to the other side of a linear map: every exponent formed
on the ``q`` side is a later-minus-earlier difference (non-positive), and the
only growing factor, ``exp(a - s_j)`` on ``k_hat``, still has its span bounded by
one sub-chunk and still meets its matching row factor inside the same dot
product.
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

# The reduced pass count: three error-compensated bf16 passes on a TPU MXU
# instead of the six an f32 x f32 dot is expanded into.
_REDUCED = jax.lax.Precision.HIGH
# The full pass count, kept for the operator that emits the recurrent state when
# ``exact_state_operator`` is selected.
_FULL = jax.lax.Precision.HIGHEST


def fused_walk_subchunk_rows(chunk_rows: int) -> int:
    """Sub-chunk height of the fused walk.

    The pivot of the column rescaling is the sub-chunk entrance, so the sub-chunk
    height is exactly the span of the one growing exponent this schedule forms. A
    chunk no taller than one sub-chunk is its own sub-chunk.
    """
    return min(_OUTPUT_SUBCHUNK_ROWS, chunk_rows)


def _fused_walk_kernel(
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
    STATE_PRECISION,
):
    """One chunk of ``G`` head-value tiles: output and state carry in one body.

    The body is STRAIGHT-LINE -- the sequence-entrance reset, the padded tail and
    the final-state write are selects and masked contraction weights rather than
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
    gamma = jnp.stack(
        [g_gamma_ref[first_value_tile + i].astype(f32) for i in range(G)]
    )  # (G,)

    seq_idx = chunk_to_seq_ref[i_t]
    bos = cu_ref[seq_idx]
    eos = cu_ref[seq_idx + 1]
    t0 = i_t * BT
    # Real (non-padded) occupancy of this chunk; an empty sequence contributes
    # nothing and does not decay.
    real_eos = bos + real_lens_ref[seq_idx]
    L = jnp.clip(real_eos - t0, 0, BT).astype(f32)
    L = jnp.where(bos != eos, L, 0.0)

    zero_state = jnp.zeros((G, K, value_width), f32)
    h0_tile = zero_state if h0_ref is None else h0_ref[:, seq_idx].astype(f32)
    state_in = jnp.where(t0 == bos, h0_tile, state_ref[...])

    row = jnp.arange(sub, dtype=f32)
    sub_mask = jnp.arange(sub)[:, None] >= jnp.arange(sub)[None, :]
    older = jnp.arange(num_sub)[:, None] > jnp.arange(num_sub)[None, :]

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

    # BOTH halves of the factored gate live inside the operands. The row half is a
    # per-row scalar, so it commutes with every contraction that consumes this row
    # -- carrying it (and `scale`) on `q` costs one multiply inside the pass that
    # already stages `q` and leaves no contraction result to rescale afterwards.
    q_hat = s_q.astype(f32) * (scale * exp(s_log - entrance_log[:, :, None]))[..., None]
    k_hat = s_k.astype(f32) * exp(entrance_log[:, :, None] - s_log)[..., None]

    # The only k-v contraction of the walk: each sub-chunk's contribution
    # expressed at its own entrance pivot.
    b_w = jnp.einsum(
        "gnik,gnid->gnkd",
        k_hat,
        s_v,
        precision=_REDUCED,
        preferred_element_type=f32,
    )

    # ONE operator for both readers of `b_w`. Rows 0..num_sub-1 are the
    # strictly-older decay matrix that rebases every earlier sub-chunk onto each
    # sub-chunk's entrance; the last row is the carry, rebasing the same
    # contributions onto the chunk's REAL exit. A sub-chunk beyond the real tail
    # contributes exactly zero (its k rows are zero), so clamping its weight to
    # one is safe and keeps every formed exponent non-positive.
    exit_log = gamma * L  # (G,)
    carry = exp(jnp.minimum(exit_log[:, None] - entrance_log, 0.0))  # (G, num_sub)
    state_operator = jnp.concatenate(
        [
            jnp.where(older, exp(entrance_log[:, :, None] - entrance_log[:, None, :]), 0.0),
            carry[:, None, :],
        ],
        axis=1,
    )  # (G, num_sub + 1, num_sub)
    walked = jnp.einsum(
        "gnb,gbkd->gnkd",
        state_operator,
        b_w,
        precision=STATE_PRECISION,
        preferred_element_type=f32,
    )  # (G, num_sub + 1, K, value_width)
    b_state = walked[:, :num_sub] + exp(entrance_log)[:, :, None, None] * state_in[:, None]

    # Intra-sub-chunk scores: both gate halves are already inside the operands, so
    # this tile needs its causal mask and nothing else.
    s_A = jnp.einsum(
        "gnik,gnjk->gnij", q_hat, k_hat, precision=_REDUCED, preferred_element_type=f32
    )
    s_A = jnp.where(sub_mask, s_A, 0.0)

    b_o = jnp.einsum(
        "gnij,gnjd->gnid", s_A, s_v, precision=_REDUCED, preferred_element_type=f32
    )
    b_o = b_o + jnp.einsum(
        "gnik,gnkd->gnid", q_hat, b_state, precision=_REDUCED, preferred_element_type=f32
    )
    o_ref[:, 0] = b_o.reshape(G, BT, value_width).astype(o_ref.dtype)

    state_out = exp(exit_log)[:, None, None] * state_in + walked[:, num_sub]
    state_ref[...] = state_out
    if ht_ref is not None:
        # The final state of a sequence is written at its last chunk; padded
        # chunks past the real tail carry L = 0 and leave the state untouched, so
        # a repeated write stores the identical value.
        ht_ref[:, seq_idx] = jnp.where(
            t0 + BT >= eos, state_out, ht_ref[:, seq_idx].astype(f32)
        ).astype(ht_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "output_value_tiles",
        "output_final_state",
        "exact_state_operator",
    ),
)
def fused_walk_chunk_fwd(
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
    exact_state_operator: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Whole chunked GLA forward as one launch with gate-complete operands.

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
    assert g_gamma is not None, "the fused walk requires g_gamma."
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    if scale is None:
        scale = K**-0.5
    sub = fused_walk_subchunk_rows(BT)
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
            _fused_walk_kernel,
            BT=BT,
            OUTPUT_VALUE_TILES=G,
            SUBCHUNK_ROWS=sub,
            STATE_PRECISION=_FULL if exact_state_operator else _REDUCED,
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
