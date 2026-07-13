"""GLA (simple gated linear attention) — chunked prefill kernel (EDITABLE runner).

Tuned entry: `chunk_simple_gla_fwd_varlen(..., chunk_size)` (Pallas).
The correctness contract (reference, tolerance, canonical inputs, native test) is
FROZEN in `akt/benchmark/refs/gla.py` — this runner only owns the DesignSpace and
the config->kernel run mapping.

Design space (base): `chunk_size` — the BT chunk tile. BK/BV are pinned to the
head dim (128) inside the kernel; elevating independent BK/BV tiling is a natural
CAPABILITY (see akt/core/evolve/capabilities/README.md).
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


@functools.lru_cache(maxsize=None)
def _jit_chunk(chunk_size: int):
    def f(q, k, v, g_gamma, cu):
        o, _ht = chunk_simple_gla_fwd_varlen(
            q, k, v, g_gamma=g_gamma, scale=None,
            cu_seqlens_dev=cu, chunk_size=chunk_size)
        return o
    return jax.jit(f)


def _run(inp, cfg):
    return _jit_chunk(int(cfg["chunk_size"]))(
        inp["q"], inp["k"], inp["v"], inp["g_gamma"], inp["cu"])


def _space():
    return DesignSpace(
        knobs=[Knob("chunk_size", [16, 32, 64, 128, 256], default=64)],
        valid=lambda c: c["chunk_size"] <= 512,
    )


CASES = [
    KernelCase(
        kernel_id="gla", shape_id=f"seq{sl}_h{h}",
        make_inputs=functools.partial(make_inputs, sl, h),
        run=_run, reference=reference, space=_space(),
        atol=ATOL, rtol=RTOL, native_test=NATIVE_TEST,
        note="chunked prefill GLA; ref=fused_recurrent (pure JAX), tol=repo 1e-3",
    )
    for (sl, h) in [(512, 8), (2048, 8)]
]
