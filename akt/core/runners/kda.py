"""KDA (Kimi Delta Attention) — chunked varlen prefill kernel.

Tuned entry: `chunk_kda_fwd(..., chunk_size, compute_block_chunks)` (Pallas,
threads an `interpret` flag via `get_interpret()` → runs here under
PALLAS_INTERPRET=1 on CPU).
Reference:   `naive_recurrent_kda(...)` (pure JAX per-step delta recurrence).
Design space: base `chunk_size`, capability-elevated `intra_block_size` for the
stage-2 triangular solve, exact `scalar_intra_solve` scheduling, and kept
`compute_block_chunks` for grouping the independent stage-2/stage-4 programs.
`state_block_chunks` separately groups stage-3 propagation, while
`state_dim_alignment` exposes its K/V state tile instead of always rounding
both dimensions to 128. `single_chunk_state_elision` removes an unobserved
terminal state update when the packed input is one full-sequence chunk.

The authoritative correctness contract (reference impl, tolerances, canonical
inputs, native-test wiring) lives in the FROZEN akt.benchmark.refs.kda module and
is imported below — this EDITABLE runner owns only the DesignSpace + the
config->kernel run-mapping + the CASES assembly.
"""
from __future__ import annotations

import functools

import jax

from sgl_jax.srt.kernels.kda import chunk_kda

from akt.benchmark.refs.kda import (
    ATOL,
    NATIVE_TEST,
    RTOL,
    check_out,
    make_inputs,
    reference,
)
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob


_INTRA_CAPABILITY = "kda_intra_solve_blocks"
_SCALAR_INTRA_CAPABILITY = "kda_scalar_intra_solve"
_COMPUTE_CAPABILITY = "kda_compute_block_chunks"
_STATE_CAPABILITY = "kda_state_block_chunks"
_STATE_DIM_CAPABILITY = "kda_state_dim_alignment"
_STATE_ELISION_CAPABILITY = "kda_single_chunk_state_elision"


@functools.lru_cache(maxsize=None)
def _jit_chunk(
    chunk_size: int,
    intra_block_size: int,
    scalar_intra_solve: bool,
    compute_block_chunks: int,
    state_block_chunks: int,
    state_dim_alignment: int,
    single_chunk_state_elision: bool,
    scale: float,
):
    # chunk_kda_fwd is itself jitted with static chunk_size/output_final_state;
    # initial_state=None, output_final_state=False → fresh state, output-only.
    def f(q, k, v, g, beta, cu):
        out = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale,
            None,
            False,
            cu,
            chunk_size=chunk_size,
            intra_block_size=intra_block_size,
            scalar_intra_solve=scalar_intra_solve,
            compute_block_chunks=compute_block_chunks,
            state_block_chunks=state_block_chunks,
            state_dim_alignment=state_dim_alignment,
            single_chunk_state_elision=single_chunk_state_elision,
        )
        return out[0]  # o
    return jax.jit(f)


def _run(inp, cfg):
    return _jit_chunk(
        int(cfg["chunk_size"]),
        int(cfg.get("intra_block_size", 16)),
        bool(cfg.get("scalar_intra_solve", False)),
        int(cfg.get("compute_block_chunks", 1)),
        int(cfg.get("state_block_chunks", 1)),
        int(cfg.get("state_dim_alignment", 128)),
        bool(cfg.get("single_chunk_state_elision", False)),
        float(inp["scale"]),
    )(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["cu"])


def _space(seqlen: int, head_dim: int) -> DesignSpace:
    # `chunk_size` is the BT time tile the four-stage pipeline blocks over. The
    # kernel's OWN constraints (no shipped tuned table for KDA) define the real
    # supported set:
    #   * power of 2 — kda.py:195 asserts `chunk_size == 2**(bit_length-1)`;
    #   * divides the sequence length — kda.py:1026 asserts `T % chunk_size == 0`
    #     (after varlen alignment pads T up to a multiple of BT). Restricting to
    #     divisors of the original T keeps the honest set (a non-divisor only
    #     "works" by over-padding T, e.g. 256 on a 128-seq — a degenerate config);
    #   * >= 16 — chunk_size 8 (and below) trips an AssertionError in the chunked
    #     pipeline, so 16 is the smallest supported tile (empirically verified).
    # The divisibility bound also caps chunk_size at T, so the space cannot explode.
    def _valid(c, T=seqlen):
        chunk_size = c["chunk_size"]
        intra_block_size = c.get("intra_block_size", 16)
        scalar_intra_solve = c.get("scalar_intra_solve", False)
        compute_block_chunks = c.get("compute_block_chunks", 1)
        state_block_chunks = c.get("state_block_chunks", 1)
        state_dim_alignment = c.get("state_dim_alignment", 128)
        state_elision = c.get("single_chunk_state_elision", False)
        return (
            chunk_size >= 16
            and (chunk_size & (chunk_size - 1)) == 0
            and T % chunk_size == 0
            and intra_block_size >= 8
            and (intra_block_size & (intra_block_size - 1)) == 0
            and chunk_size % intra_block_size == 0
            # Scalar mode has no blocked solve size. Anchor the existing knob at
            # its smallest value so search enumerates one, not three, identical
            # scalar executions for each physical chunk-grouping combination.
            and (not scalar_intra_solve or intra_block_size == 8)
            and compute_block_chunks >= 1
            and (compute_block_chunks & (compute_block_chunks - 1)) == 0
            # Stage 2 and Stage 4 group this many logical BT chunks per
            # physical Pallas program; keep their grouped extent exact.
            and (T // chunk_size) % compute_block_chunks == 0
            and state_block_chunks >= 1
            and (state_block_chunks & (state_block_chunks - 1)) == 0
            and (T // chunk_size) % state_block_chunks == 0
            and state_dim_alignment >= head_dim
            and (state_dim_alignment & (state_dim_alignment - 1)) == 0
            # Elision is exact only for this runner's output-only, zero-state,
            # single-sequence case with one full-sequence logical chunk. Anchor
            # the now-unused state knobs so search does not time duplicates.
            and (
                not state_elision
                or (
                    chunk_size == T
                    and state_block_chunks == 1
                    and state_dim_alignment == 128
                )
            )
        )

    return DesignSpace(
        knobs=[
            Knob("chunk_size", [16, 32, 64, 128, 256], default=64),
            Knob(
                "intra_block_size",
                [8, 16, 32],
                default=16,
                elevated_by=_INTRA_CAPABILITY,
            ),
            Knob(
                "scalar_intra_solve",
                [False, True],
                default=False,
                elevated_by=_SCALAR_INTRA_CAPABILITY,
            ),
            Knob(
                "compute_block_chunks",
                [1, 2],
                default=1,
                elevated_by=_COMPUTE_CAPABILITY,
            ),
            Knob(
                "state_block_chunks",
                [1, 2, 4],
                default=1,
                elevated_by=_STATE_CAPABILITY,
            ),
            Knob(
                "state_dim_alignment",
                [64, 128],
                default=128,
                elevated_by=_STATE_DIM_CAPABILITY,
            ),
            Knob(
                "single_chunk_state_elision",
                [False, True],
                default=False,
                elevated_by=_STATE_ELISION_CAPABILITY,
            ),
        ],
        valid=_valid,
    )


CASES = [
    KernelCase(
        kernel_id="kda", shape_id=f"seq{sl}_h{h}_d{d}",
        make_inputs=functools.partial(make_inputs, sl, h, d),
        run=_run, reference=reference, space=_space(sl, d),
        atol=ATOL, rtol=RTOL,
        check_out=check_out,  # _run/reference already return o
        native_test=NATIVE_TEST,
        regime_pref=("cpu-interpret",),
        note="chunked varlen KDA; ref=naive_recurrent (pure JAX); runs in CPU-interpret",
    )
    for (sl, h, d) in [(128, 2, 64), (256, 4, 128)]
]
