"""KDA (Kimi Delta Attention) — chunked varlen prefill kernel.

Tuned entry: `chunk_kda_fwd(..., chunk_size)` (Pallas, threads an `interpret`
flag via `get_interpret()` → runs here under PALLAS_INTERPRET=1 on CPU).
Reference:   `naive_recurrent_kda(...)` (pure JAX per-step delta recurrence).
Design space (base): `chunk_size` — the BT time tile the four-stage pipeline
(gate cumsum / intra solve / inter-chunk state / output) blocks over. K/V head
dims are padded to 128 inside the kernel; elevating independent BK/BV tiling
would be a natural CAPABILITY.

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


@functools.lru_cache(maxsize=None)
def _jit_chunk(chunk_size: int, scale: float):
    # chunk_kda_fwd is itself jitted with static chunk_size/output_final_state;
    # initial_state=None, output_final_state=False → fresh state, output-only.
    def f(q, k, v, g, beta, cu):
        out = chunk_kda(q, k, v, g, beta, scale, None, False, cu, chunk_size=chunk_size)
        return out[0]  # o
    return jax.jit(f)


def _run(inp, cfg):
    return _jit_chunk(int(cfg["chunk_size"]), float(inp["scale"]))(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["cu"])


def _space():
    return DesignSpace(
        knobs=[Knob("chunk_size", [16, 32, 64, 128], default=64)],
        valid=lambda c: c["chunk_size"] >= 16 and (c["chunk_size"] & (c["chunk_size"] - 1)) == 0,
    )


CASES = [
    KernelCase(
        kernel_id="kda", shape_id=f"seq{sl}_h{h}_d{d}",
        make_inputs=functools.partial(make_inputs, sl, h, d),
        run=_run, reference=reference, space=_space(),
        atol=ATOL, rtol=RTOL,
        check_out=check_out,  # _run/reference already return o
        native_test=NATIVE_TEST,
        regime_pref=("cpu-interpret",),
        note="chunked varlen KDA; ref=naive_recurrent (pure JAX); runs in CPU-interpret",
    )
    for (sl, h, d) in [(128, 2, 64), (256, 4, 128)]
]
