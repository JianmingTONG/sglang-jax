"""GLA (simple gated linear attention) — chunked prefill kernel (EDITABLE runner).

Tuned entry: `chunk_simple_gla_fwd_varlen(..., chunk_size,
compact_alignment, single_chunk_state_elision,
zero_state_output_elision, output_value_tiles,
enable__chunk_fwd_o_pl_variant,
enable_chunk_fwd_h_kernel_varlen_variant)` (Pallas).
The correctness contract (reference, tolerance, canonical inputs, native test) is
FROZEN in `akt/benchmark/refs/gla.py` — this runner only owns the DesignSpace and
the config->kernel run mapping.

Design space: base `chunk_size` plus capability-elevated `compact_alignment`,
`output_value_tiles`, `enable__chunk_fwd_o_pl_variant` and
`enable_chunk_fwd_h_kernel_varlen_variant`. `output_impl` dispatches the output
stage to the standalone `batched_value_tile_fwd_o` handle, which runs the same
two-level walk with the value-tile axis as a batch dimension of every
contraction instead of a Python loop bound; it owns that schedule, so the output
schedule toggle stays at its default for its configurations. Historical
state/output-elision axes remain visible as runner-only experiments but are
invalid in this serving objective because the nonzero initial state and final
recurrent state are both observed.
"""
from __future__ import annotations

import functools

import jax

from sgl_jax.srt.kernels.simple_gla.simple_gla import (
    _OUTPUT_SUBCHUNK_ROWS,
    chunk_simple_gla_fwd_varlen,
)

from akt.benchmark.refs.gla import (
    ATOL,
    NATIVE_TEST,
    RTOL,
    make_inputs,
    reference,
)
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob


_COMPACT_ALIGNMENT_CAPABILITY = "gla_compact_alignment"
_STATE_ELISION_CAPABILITY = "gla_single_chunk_state_elision"
_OUTPUT_ELISION_CAPABILITY = "gla_zero_state_output_elision"
_VALUE_TILE_GROUPING_CAPABILITY = "gla_value_tile_grouping"
_OUTPUT_SUBCHUNK_SCHEDULE_CAPABILITY = "gla_output_subchunk_schedule"
_STATE_DECAY_MATRIX_CAPABILITY = "gla_state_decay_matrix_closed_form"
_BATCHED_VALUE_TILE_OUTPUT_CAPABILITY = "gla_batched_value_tile_output_api"


@functools.lru_cache(maxsize=None)
def _jit_chunk(
    chunk_size: int,
    compact_alignment: bool,
    single_chunk_state_elision: bool,
    zero_state_output_elision: bool,
    output_value_tiles: int,
    enable__chunk_fwd_o_pl_variant: bool,
    enable_chunk_fwd_h_kernel_varlen_variant: bool,
    output_impl: str,
):
    def f(q, k, v, g_gamma, initial_state, cu):
        o, ht = chunk_simple_gla_fwd_varlen(
            q, k, v, g_gamma=g_gamma, scale=None,
            h0=initial_state, use_ht=True,
            cu_seqlens_dev=cu, chunk_size=chunk_size,
            compact_alignment=compact_alignment,
            single_chunk_state_elision=single_chunk_state_elision,
            zero_state_output_elision=zero_state_output_elision,
            output_value_tiles=output_value_tiles,
            enable__chunk_fwd_o_pl_variant=enable__chunk_fwd_o_pl_variant,
            enable_chunk_fwd_h_kernel_varlen_variant=(
                enable_chunk_fwd_h_kernel_varlen_variant),
            output_impl=output_impl)
        return o, ht
    return jax.jit(f)


def _run(inp, cfg):
    return _jit_chunk(
        int(cfg["chunk_size"]),
        bool(cfg.get("compact_alignment", False)),
        bool(cfg.get("single_chunk_state_elision", False)),
        bool(cfg.get("zero_state_output_elision", False)),
        int(cfg.get("output_value_tiles", 1)),
        bool(cfg.get("enable__chunk_fwd_o_pl_variant", False)),
        bool(cfg.get("enable_chunk_fwd_h_kernel_varlen_variant", False)),
        str(cfg.get("output_impl", "incumbent")),
    )(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g_gamma"],
        inp["initial_state"],
        inp["cu"],
    )


# Real supported set for `chunk_size` (the BT chunk tile). The kernel imposes NO
# tuned table — it is a free param whose only hard constraint is `T % chunk_size == 0`
# (asserted at simple_gla.py:445,771; BK/BV pinned to 128). The maintainers' convention
# (and every production caller — lightning_backend default 64) is powers of two, so the
# candidate set is all pow2 tiles from the existing lower bound up to a VMEM-sane cap.
_CHUNK_CANDIDATES = [16, 32, 64, 128, 256, 512, 1024, 2048]
# VMEM-sane cap: the intra-chunk score tile is BT×BT float32. A 16 MiB ceiling
# admits the full-sequence BT=2048 tile while remaining inside the Pallas launcher's
# explicit 128 MiB VMEM budget once its elementwise temporaries are included.
_SCORE_TILE_BYTES_CAP = 16 * 1024 * 1024  # -> chunk_size <= 2048


def _space(seqlen: int, heads: int):
    def _valid(c):
        cs = c["chunk_size"]
        state_elision = c.get("single_chunk_state_elision", False)
        output_elision = c.get("zero_state_output_elision", False)
        output_value_tiles = c.get("output_value_tiles", 1)
        output_impl = c.get("output_impl", "incumbent")
        if output_impl != "incumbent":
            # The batched value-tile handle OWNS the output schedule: it is the
            # two-level walk with its own token-order delivery, so it consumes
            # no output schedule toggle (that knob stays at its default) and it
            # is only defined for a chunk taller than one sub-chunk. The
            # deployment space therefore gains the chunk/alignment/value-tile
            # combinations the handle actually reads, not a duplicate of the
            # whole incumbent grid.
            if c.get("enable__chunk_fwd_o_pl_variant", False):
                return False
            if cs <= _OUTPUT_SUBCHUNK_ROWS or cs % _OUTPUT_SUBCHUNK_ROWS:
                return False
        return (
            cs > 0
            and (cs & (cs - 1)) == 0          # power of two (maintainer convention)
            and seqlen % cs == 0               # kernel asserts T % chunk_size == 0
            and cs * cs * 4 <= _SCORE_TILE_BYTES_CAP  # VMEM-sane BT×BT score tile
            and not state_elision  # serving observes initial + final recurrent state
            and not output_elision
            and output_value_tiles > 0
            and (output_value_tiles & (output_value_tiles - 1)) == 0
            and heads % output_value_tiles == 0
        )

    return DesignSpace(
        knobs=[
            Knob(
                "chunk_size",
                _CHUNK_CANDIDATES,
                default=64,
                programmer_control="gla.chunk_size",
            ),
            Knob(
                "compact_alignment",
                [False, True],
                default=False,
                elevated_by=_COMPACT_ALIGNMENT_CAPABILITY,
                programmer_control="gla.compact_alignment",
            ),
            Knob(
                "single_chunk_state_elision",
                [False, True],
                default=False,
                elevated_by=_STATE_ELISION_CAPABILITY,
            ),
            Knob(
                "zero_state_output_elision",
                [False, True],
                default=False,
                elevated_by=_OUTPUT_ELISION_CAPABILITY,
            ),
            Knob(
                "output_value_tiles",
                [1, 2, 4, 8],
                default=1,
                elevated_by=_VALUE_TILE_GROUPING_CAPABILITY,
                programmer_control="gla.output_value_tiles",
            ),
            Knob(
                "enable__chunk_fwd_o_pl_variant",
                [False, True],
                default=False,
                elevated_by=_OUTPUT_SUBCHUNK_SCHEDULE_CAPABILITY,
                programmer_control="gla.enable__chunk_fwd_o_pl_variant",
            ),
            Knob(
                "enable_chunk_fwd_h_kernel_varlen_variant",
                [False, True],
                default=False,
                elevated_by=_STATE_DECAY_MATRIX_CAPABILITY,
                programmer_control="gla.enable_chunk_fwd_h_kernel_varlen_variant",
            ),
            Knob(
                "output_impl",
                ["incumbent", "batched_value_tiles"],
                default="incumbent",
                elevated_by=_BATCHED_VALUE_TILE_OUTPUT_CAPABILITY,
                programmer_control="gla.output_impl",
            ),
        ],
        valid=_valid,
    )


CASES = [
    KernelCase(
        kernel_id="gla", shape_id=f"seq{sl}_h{h}",
        make_inputs=functools.partial(make_inputs, sl, h),
        run=_run, reference=reference, space=_space(sl, h),
        atol=ATOL, rtol=RTOL, native_test=NATIVE_TEST,
        note="chunked prefill GLA; ref=fused_recurrent (pure JAX), tol=repo 1e-3",
    )
    for (sl, h) in [(512, 8), (2048, 8)]
]
