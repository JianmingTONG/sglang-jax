"""GLA (simple gated linear attention) — chunked prefill kernel (EDITABLE runner).

Tuned entry: `chunk_simple_gla_fwd_varlen(..., chunk_size,
compact_alignment, single_chunk_state_elision,
zero_state_output_elision)` (Pallas).
The correctness contract (reference, tolerance, canonical inputs, native test) is
FROZEN in `akt/benchmark/refs/gla.py` — this runner only owns the DesignSpace and
the config->kernel run mapping.

Design space: base `chunk_size` plus capability-elevated `compact_alignment`,
`single_chunk_state_elision`, and `zero_state_output_elision`.  State elision
removes an unobserved terminal update when the aligned input is exactly one
logical chunk; output elision removes the matching zero state load and matmul.
"""
from __future__ import annotations

import functools

import jax

from sgl_jax.srt.kernels.simple_gla.simple_gla import chunk_simple_gla_fwd_varlen

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


@functools.lru_cache(maxsize=None)
def _jit_chunk(
    chunk_size: int,
    compact_alignment: bool,
    single_chunk_state_elision: bool,
    zero_state_output_elision: bool,
):
    def f(q, k, v, g_gamma, cu):
        o, _ht = chunk_simple_gla_fwd_varlen(
            q, k, v, g_gamma=g_gamma, scale=None,
            cu_seqlens_dev=cu, chunk_size=chunk_size,
            compact_alignment=compact_alignment,
            single_chunk_state_elision=single_chunk_state_elision,
            zero_state_output_elision=zero_state_output_elision)
        return o
    return jax.jit(f)


def _run(inp, cfg):
    return _jit_chunk(
        int(cfg["chunk_size"]),
        bool(cfg.get("compact_alignment", False)),
        bool(cfg.get("single_chunk_state_elision", False)),
        bool(cfg.get("zero_state_output_elision", False)),
    )(
        inp["q"], inp["k"], inp["v"], inp["g_gamma"], inp["cu"])


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


def _space(seqlen: int):
    def _valid(c):
        cs = c["chunk_size"]
        state_elision = c.get("single_chunk_state_elision", False)
        output_elision = c.get("zero_state_output_elision", False)
        return (
            cs > 0
            and (cs & (cs - 1)) == 0          # power of two (maintainer convention)
            and seqlen % cs == 0               # kernel asserts T % chunk_size == 0
            and cs * cs * 4 <= _SCORE_TILE_BYTES_CAP  # VMEM-sane BT×BT score tile
            and (
                not state_elision
                or (c.get("compact_alignment", False) and cs == seqlen)
            )
            and (not output_elision or state_elision)
        )

    return DesignSpace(
        knobs=[
            Knob("chunk_size", _CHUNK_CANDIDATES, default=64),
            Knob(
                "compact_alignment",
                [False, True],
                default=False,
                elevated_by=_COMPACT_ALIGNMENT_CAPABILITY,
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
        ],
        valid=_valid,
    )


CASES = [
    KernelCase(
        kernel_id="gla", shape_id=f"seq{sl}_h{h}",
        make_inputs=functools.partial(make_inputs, sl, h),
        run=_run, reference=reference, space=_space(sl),
        atol=ATOL, rtol=RTOL, native_test=NATIVE_TEST,
        note="chunked prefill GLA; ref=fused_recurrent (pure JAX), tol=repo 1e-3",
    )
    for (sl, h) in [(512, 8), (2048, 8)]
]
