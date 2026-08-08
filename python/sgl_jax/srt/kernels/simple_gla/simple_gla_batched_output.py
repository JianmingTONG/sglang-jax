"""Batched value-tile GLA output stage (standalone JAX-callable handle).

``output_value_tiles`` lets one Pallas program own several independent
head-value tiles, which amortizes the grid and the per-program prologue. The
shipped output kernel then walks those tiles with a PYTHON LOOP
(``for local_value_tile in range(OUTPUT_VALUE_TILES)`` in
``_chunk_fwd_o_kernel`` / ``_chunk_fwd_o_subchunk_body``), so a program that
owns ``G`` tiles emits ``G`` independent copies of the whole output schedule:
``G`` score contractions, ``G`` decay-matrix solves, ``G`` state contractions.
The measured effect is decisive on this stage — at the plans the model DP
selects (``tiny-linear-serving/gla-long`` at ``chunk_size=2048`` and
``gla-short`` at ``512``, both with ``output_value_tiles=8``) the whole varlen
forward traces 933 operations and replacing the output launch with zeros
removes 68% / 76% of its measured time, while the arithmetic itself runs at
roughly a tenth of the device's fp32 rate: the stage is paying for the NUMBER of
launched contractions, not for their size.

This module rebuilds that stage with the value-tile axis as a leading BATCH
DIMENSION of every contraction instead of a loop bound. One program's ``G``
tiles become a single ``einsum`` per stage —
``'gnik,gnjk->gnij'`` for the intra-sub-chunk scores, ``'gnb,gbkd->gnkd'`` for
the sub-chunk decay-matrix carry, ``'gnik,gnkd->gnid'`` for the inter term — so
the traced program of a ``G``-tile program stops scaling with ``G`` while every
operand keeps exactly the tile extents the loop used. Nothing is shared between
heads or value slices: the batch axis is the tile axis, so each tile's q/k/v/h
and its own gate still meet only themselves, and the emitted values are the same
re-associated sum.

The handle owns the whole output schedule (both levels of the two-level walk and
the token-order delivery), so it consumes neither ``enable__chunk_fwd_o_pl_variant``
nor a separate unalignment pass.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.simple_gla.simple_gla import (
    _OUTPUT_SUBCHUNK_ROWS,
    assert_shape,
    assert_shape_or_none,
    exp,
    get_interpret,
)

_HIGHEST = jax.lax.Precision.HIGHEST


def batched_output_subchunk_rows(chunk_rows: int) -> int:
    """Sub-chunk height of the batched output schedule.

    A chunk no taller than one sub-chunk is its own sub-chunk: the walk then has
    a single level and reproduces the shipped single-tile schedule exactly (an
    empty strictly-older set, so the state term is the chunk entrance itself).
    """
    return min(_OUTPUT_SUBCHUNK_ROWS, chunk_rows)


def _batched_value_tile_o_kernel(
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
    OUTPUT_VALUE_TILES: int,
    SUBCHUNK_ROWS: int,
):
    """All ``G`` head-value tiles of one chunk, as one batched schedule.

    Refs (after block-spec indexing) carry the tile axis first, exactly as the
    shipped kernel receives them:
      q_ref/k_ref (G, 1, BT, K); v_ref/o_ref (G, 1, BT, value_width);
      h_ref (G, 1, K, value_width); g_ref (G, 1, BT, 128) or None.
    Every stage below contracts over that axis as a batch dimension, so the
    program's operation count is the same for G = 1 and G = 8.
    """
    G = OUTPUT_VALUE_TILES
    sub = SUBCHUNK_ROWS
    num_sub = BT // sub
    K = q_ref.shape[-1]
    value_width = v_ref.shape[-1]
    scale = scale_ref[0].astype(jnp.float32)

    row = jnp.arange(sub)
    sub_mask = row[:, None] >= row[None, :]
    older = jnp.arange(num_sub)[:, None] > jnp.arange(num_sub)[None, :]

    # Chunk-cumulative log decay per tile and row. Both gate forms are additive
    # in log space, matching the shipped stage's two independent exp() factors.
    b_log = None
    if g_ref is not None:
        b_log = g_ref[:, 0, :, 0].astype(jnp.float32)  # (G, BT)
    if g_gamma_ref is not None:
        first_value_tile = pl.program_id(0) * G
        # g_gamma lives in SMEM/ANY and is read one scalar per tile, as the
        # shipped kernel reads it; the stack is what makes it a batch axis.
        b_gamma = jnp.stack(
            [g_gamma_ref[first_value_tile + i].astype(jnp.float32) for i in range(G)]
        )  # (G,)
        b_g_gamma = b_gamma[:, None] * (jnp.arange(BT) + 1).astype(jnp.float32)[None, :]
        b_log = b_g_gamma if b_log is None else b_log + b_g_gamma

    s_q = q_ref[:, 0].reshape(G, num_sub, sub, K)
    s_k = k_ref[:, 0].reshape(G, num_sub, sub, K)
    s_v = v_ref[:, 0].astype(jnp.float32).reshape(G, num_sub, sub, value_width)
    if b_log is not None:
        # ``s_log`` is the chunk-cumulative decay; ``entrance``/``exit`` are its
        # values just before and at the end of each sub-chunk. Every exponent
        # formed below is a difference of a later minus an earlier position, so
        # all of them stay non-positive for a decaying gate.
        s_log = b_log.reshape(G, num_sub, sub)
        exit_log = s_log[:, :, sub - 1]  # (G, num_sub)
        entrance_log = jnp.concatenate(
            [jnp.zeros((G, 1), jnp.float32), exit_log[:, : num_sub - 1]], axis=1
        )

    # Intra-sub-chunk attention: one masked (sub, sub) score tile per sub-chunk
    # of every tile, in a single contraction.
    s_A = jnp.einsum("gnik,gnjk->gnij", s_q, s_k, preferred_element_type=jnp.float32)
    if b_log is not None:
        log_diff = s_log[:, :, :, None] - s_log[:, :, None, :]
        s_A = s_A * exp(jnp.where(sub_mask, log_diff, 0.0))
    s_A = jnp.where(sub_mask, s_A, 0.0)
    # Keep the score tile in fp32 for precision; the values are upcast instead.
    b_o = jnp.einsum(
        "gnij,gnjd->gnid",
        s_A,
        s_v,
        precision=_HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Each sub-chunk's own contribution to the states the later sub-chunks read,
    # expressed at that sub-chunk's exit so no positive exponent ever appears.
    k_decayed = s_k.astype(jnp.float32)
    if b_log is not None:
        k_decayed = k_decayed * exp(exit_log[:, :, None] - s_log)[:, :, :, None]
    b_u = jnp.einsum(
        "gnik,gnid->gnkd",
        k_decayed,
        s_v,
        precision=_HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Resolve the sub-chunk state recurrence in closed form: sub-chunk n reads
    # every strictly older sub-chunk's contribution, rebased onto its entrance.
    # A single-level walk has no strictly-older sub-chunk, so that term is
    # identically zero and is not contracted at all.
    b_state = 0.0
    if num_sub > 1:
        decay_matrix = jnp.broadcast_to(
            older.astype(jnp.float32), (G, num_sub, num_sub)
        )
        if b_log is not None:
            decay_matrix = decay_matrix * exp(
                jnp.where(older, entrance_log[:, :, None] - exit_log[:, None, :], 0.0)
            )
        b_state = jnp.einsum(
            "gnb,gbkd->gnkd",
            decay_matrix,
            b_u,
            precision=_HIGHEST,
            preferred_element_type=jnp.float32,
        )
    b_h = h_ref[:, 0].astype(jnp.float32)  # (G, K, value_width)
    entrance_decay = (
        jnp.ones((G, num_sub), jnp.float32) if b_log is None else exp(entrance_log)
    )
    b_state = b_state + entrance_decay[:, :, None, None] * b_h[:, None]

    b_inter = jnp.einsum(
        "gnik,gnkd->gnid", s_q, b_state, preferred_element_type=jnp.float32
    )
    if b_log is not None:
        b_inter = b_inter * exp(s_log - entrance_log[:, :, None])[:, :, :, None]
    b_o = (b_o + b_inter) * scale
    o_ref[:, 0] = b_o.reshape(G, BT, value_width).astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size", "output_value_tiles"),
)
def batched_value_tile_fwd_o(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
    output_value_tiles: int = 1,
    output_gather: jax.Array | None = None,
) -> jax.Array:
    """GLA chunk output stage whose value-tile axis is a batch dimension.

    Same contract and same emitted values as ``chunk_fwd_o`` on the resident
    schedule: a two-level sub-chunk walk that lands its rows straight in the
    ``output_gather`` token order. The handle owns that schedule, so it takes no
    schedule toggle; it requires a chunk taller than one sub-chunk and a
    non-elided state term.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    assert h is not None, "batched value-tile output requires the state term"
    if scale is None:
        scale = K**-0.5
    sub = batched_output_subchunk_rows(BT)
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
    _h = h.reshape(B, NT, H, K, V).transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _g = None
    if g is not None:
        _g = g.reshape(B, NT, BT, H).transpose(3, 0, 1, 2).reshape(H, total_NT, BT)
        _g = jnp.broadcast_to(_g[:, :, :, None], (H, total_NT, BT, 128))

    BV = 128 if V % 128 == 0 else V
    num_v_tiles = V // BV
    if num_v_tiles > 1:
        _v = (
            _v.reshape(H, total_NT, BT, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, BT, BV)
        )
        _h = (
            _h.reshape(H, total_NT, K, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, K, BV)
        )
        if g_gamma is not None:
            g_gamma = jnp.repeat(g_gamma, num_v_tiles)
        _q = jnp.repeat(_q, num_v_tiles, axis=0)
        _k = jnp.repeat(_k, num_v_tiles, axis=0)
        if _g is not None:
            _g = jnp.repeat(_g, num_v_tiles, axis=0)

    H_VT = H * num_v_tiles
    assert output_value_tiles > 0, "output_value_tiles must be positive"
    assert H_VT % output_value_tiles == 0, (
        f"H*num_v_tiles={H_VT} must be divisible by "
        f"output_value_tiles={output_value_tiles}"
    )

    grid = (H_VT // output_value_tiles, total_NT)
    interpret = get_interpret()

    def _tile_spec(width, rows):
        return pl.BlockSpec(
            (output_value_tiles, 1, rows, width),
            index_map=lambda value_tile_block, nt_idx: (value_tile_block, nt_idx, 0, 0),
        )

    spec_gamma = (
        None
        if g_gamma is None
        else pl.BlockSpec(memory_space=pltpu.ANY if interpret else pltpu.SMEM)
    )

    o = pl.pallas_call(
        functools.partial(
            _batched_value_tile_o_kernel,
            BT=BT,
            OUTPUT_VALUE_TILES=output_value_tiles,
            SUBCHUNK_ROWS=sub,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                _tile_spec(K, BT),
                _tile_spec(K, BT),
                _tile_spec(BV, BT),
                _tile_spec(BV, K),
                None if _g is None else _tile_spec(128, BT),
                spec_gamma,
                pl.BlockSpec(memory_space=pltpu.ANY if interpret else pltpu.SMEM),
            ],
            out_specs=_tile_spec(BV, BT),
        ),
        out_shape=jax.ShapeDtypeStruct((H_VT, total_NT, BT, BV), v.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            # The batched schedule keeps the shipped resident stage's operand
            # extents, so it keeps its VMEM ceiling too.
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
    o = o.reshape(H, B, NT, BT, V).transpose(1, 2, 3, 0, 4).reshape(B, T, H, V)
    if output_gather is not None:
        # Same single-move token-order delivery the resident schedule performs:
        # the permutation is expressed over the OUTPUT index space, so each row
        # moves exactly once and no aligned staging pass is launched.
        o = o[:, output_gather]
    return o
