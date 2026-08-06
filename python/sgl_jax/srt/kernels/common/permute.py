"""Single-move row permutation — a reusable patch for a JAX API boundary gap.

JAX/jnp provides no primitive that moves each row of an array to a caller-chosen
destination row exactly once. The two expressible formulations both over-pay:

- ``x.at[idx].add`` / ``x.at[idx].set`` lowers to XLA scatter — a read-modify-write
  over the destination buffer with accumulate semantics that initializes and touches
  every destination row (DMA-bound int32 index-manipulation stages; the AKT campaign
  limiter — see akt/core/analysis/jax_api_limitations.md).
- ``jnp.take`` / gather materializes a second full buffer and needs the INVERSE
  permutation, which callers holding forward scatter indices must first compute.

Incumbent callers in this tree eat the scatter (simple_gla ``_unalign_output``,
kda's ``.at[flat_pos].add`` sites) or hand-fuse the permutation into one kernel
(simple_gla ``_chunk_fwd_o_pl`` ``emit_order``, not reusable). This module provides
the missing API once, for every kernel:

``single_move_permute(rows, dest_index, out_len)``
    out[dest_index[i]] = rows[i] for every in-range ``dest_index[i]``; rows whose
    destination is out of range (e.g. -1 for varlen padding) are dropped.

Two implementations, selected by ``backend``:

- ``"pallas"`` — a Pallas kernel (TPU and interpret-safe) that grids over source
  rows and DMA-copies each row straight into its destination block through a
  scalar-prefetched dynamic output index. Each destination row is written exactly
  once (no accumulate, no zero-init of the destination); dropped rows are routed
  to a trailing trash row that is sliced off.
- ``"jnp"`` — a pure-jnp fallback for hosts where Pallas cannot lower: ``take`` on
  the inverse permutation, computed via ``argsort`` since only forward indices are
  given.

PRECONDITION (both paths): the in-range entries of ``dest_index`` are unique and
cover ``0..out_len-1`` exactly once — a bijection onto the output, as in packed
varlen de-alignment (simple_gla's unalign pattern), where every original token
position is carried by exactly one aligned row. Under that precondition the result
is bit-exact with the scatter formulation
``zeros.at[dest].set(rows, mode="drop")`` (see test_permute_common.py).

This API is intentionally NOT wired into existing kernels here — it is introduced
as an existing, discoverable handle for the AKT loop to elevate per kernel.
"""

from __future__ import annotations

import os

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


def _get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


def _row_view(rows: jax.Array) -> jax.Array:
    """Collapse trailing dims so the kernel always moves 2D (row, width) blocks."""
    if rows.ndim == 1:
        return rows.reshape(rows.shape[0], 1)
    if rows.ndim == 2:
        return rows
    return rows.reshape(rows.shape[0], -1)


def _routed_dest(dest_index: jax.Array, out_len: int) -> jax.Array:
    """Map out-of-range destinations onto the trash row ``out_len``."""
    dest_index = dest_index.astype(jnp.int32)
    in_range = (dest_index >= 0) & (dest_index < out_len)
    return jnp.where(in_range, dest_index, jnp.int32(out_len))


def single_move_permute_reference(
    rows: jax.Array, dest_index: jax.Array, out_len: int
) -> jax.Array:
    """Pure-JAX reference semantics (the scatter formulation, for verification)."""
    dest = _routed_dest(dest_index, out_len)
    out = jnp.zeros((out_len,) + rows.shape[1:], dtype=rows.dtype)
    return out.at[dest].set(rows, mode="drop")


def _single_move_permute_jnp(
    rows: jax.Array, dest_index: jax.Array, out_len: int
) -> jax.Array:
    """Fallback: ``take`` on the inverse permutation (computed via ``argsort``)."""
    dest = _routed_dest(dest_index, out_len)
    # Stable sort of forward destinations: in-range destinations are exactly
    # 0..out_len-1 (bijection precondition), dropped rows sort last at out_len.
    inverse = jnp.argsort(dest)
    source_rows = inverse[:out_len]
    return jnp.take(rows, source_rows, axis=0)


def _permute_copy_kernel(dest_ref, x_ref, o_ref):
    # One source row per grid step, copied to its destination block exactly once.
    del dest_ref  # consumed by the output index map, not the body
    o_ref[...] = x_ref[...]


def _single_move_permute_pallas(
    rows: jax.Array, dest_index: jax.Array, out_len: int, interpret: bool
) -> jax.Array:
    rows2d = _row_view(rows)
    num_rows, width = rows2d.shape
    dest = _routed_dest(dest_index, out_len)
    out2d = pl.pallas_call(
        _permute_copy_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_rows,),
            in_specs=[pl.BlockSpec((1, width), lambda i, dest_ref: (i, 0))],
            # The destination block index is read from the scalar-prefetched
            # forward index: a direct single-move copy, not an accumulate.
            out_specs=pl.BlockSpec((1, width), lambda i, dest_ref: (dest_ref[i], 0)),
        ),
        # Row out_len is the trash row absorbing dropped sources; sliced off below.
        out_shape=jax.ShapeDtypeStruct((out_len + 1, width), rows2d.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )(dest, rows2d)
    return out2d[:out_len].reshape((out_len,) + rows.shape[1:])


def single_move_permute(
    rows: jax.Array,
    dest_index: jax.Array,
    out_len: int | None = None,
    *,
    backend: str = "auto",
    interpret: bool | None = None,
) -> jax.Array:
    """Move each source row to ``dest_index[i]`` exactly once (no accumulate).

    Args:
        rows: ``(num_rows, ...)`` source array; trailing dims are moved intact.
        dest_index: ``(num_rows,)`` integer forward indices. ``out[dest_index[i]]
            = rows[i]``; entries outside ``[0, out_len)`` (e.g. -1 padding) drop
            their row. In-range entries must be a bijection onto ``0..out_len-1``.
        out_len: number of output rows; defaults to ``num_rows``.
        backend: ``"pallas"`` (single-move DMA-copy kernel), ``"jnp"`` (take on
            the argsort-inverse permutation), or ``"auto"`` — pallas on TPU or
            under PALLAS_INTERPRET / explicit ``interpret=True``, jnp otherwise.
        interpret: force Pallas interpret mode; defaults to the PALLAS_INTERPRET
            env convention shared by the sibling kernels.

    Returns:
        ``(out_len, ...)`` array with every row landed by one move.
    """
    if rows.ndim < 1:
        raise ValueError("rows must have at least one (row) dimension")
    if dest_index.ndim != 1 or dest_index.shape[0] != rows.shape[0]:
        raise ValueError(
            "dest_index must be one forward index per source row: "
            f"rows={rows.shape[0]}, dest_index={dest_index.shape}"
        )
    if out_len is None:
        out_len = rows.shape[0]
    out_len = int(out_len)
    if out_len < 0:
        raise ValueError(f"out_len must be non-negative, got {out_len}")
    if interpret is None:
        interpret = _get_interpret()
    if backend == "auto":
        use_pallas = interpret or jax.default_backend() == "tpu"
        backend = "pallas" if use_pallas else "jnp"
    if backend == "pallas":
        return _single_move_permute_pallas(rows, dest_index, out_len, interpret)
    if backend == "jnp":
        return _single_move_permute_jnp(rows, dest_index, out_len)
    raise ValueError(f"unknown backend {backend!r}; expected auto|pallas|jnp")
