"""GLA (simple gated linear attention) — chunked prefill kernel.

Tuned entry: `chunk_simple_gla_fwd_varlen(..., chunk_size)` (Pallas).
Reference:   `fused_recurrent_simple_gla(...)` (pure JAX lax.scan).
Design space (base): `chunk_size` — the BT chunk tile. BK/BV are pinned to the
head dim (128) inside the kernel; elevating independent BK/BV tiling is a natural
CAPABILITY (see akt/core/evolve/capabilities/README.md).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.simple_gla.simple_gla import (
    chunk_simple_gla_fwd_varlen,
    fused_recurrent_simple_gla,
)

from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob

_HEAD_DIM = 128  # K = V = 128 (simple-GLA constraint, 128-aligned)


def _inputs(seqlen: int, heads: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.1, dtype=jnp.float32)
    q = mk(1, seqlen, heads, _HEAD_DIM)
    k = mk(1, seqlen, heads, _HEAD_DIM)
    v = mk(1, seqlen, heads, _HEAD_DIM)
    g_gamma = jnp.asarray(rng.uniform(-0.1, -0.01, size=(heads,)), dtype=jnp.float32)
    cu = jnp.asarray([0, seqlen], dtype=jnp.int32)          # one packed sequence
    return {"q": q, "k": k, "v": v, "g_gamma": g_gamma, "cu": cu, "seqlen": seqlen}


@functools.lru_cache(maxsize=None)
def _jit_chunk(chunk_size: int):
    def f(q, k, v, g_gamma, cu):
        o, _ht = chunk_simple_gla_fwd_varlen(
            q, k, v, g_gamma=g_gamma, scale=None,
            cu_seqlens_dev=cu, chunk_size=chunk_size)
        return o
    return jax.jit(f)


@functools.lru_cache(maxsize=1)
def _jit_ref():
    def f(q, k, v, g_gamma, cu):
        o, _ht = fused_recurrent_simple_gla(
            q, k, v, g_gamma=g_gamma, scale=None, cu_seqlens=cu)
        return o
    return jax.jit(f)


def _run(inp, cfg):
    return _jit_chunk(int(cfg["chunk_size"]))(
        inp["q"], inp["k"], inp["v"], inp["g_gamma"], inp["cu"])


def _ref(inp):
    return _jit_ref()(inp["q"], inp["k"], inp["v"], inp["g_gamma"], inp["cu"])


def _space():
    return DesignSpace(
        knobs=[Knob("chunk_size", [16, 32, 64, 128, 256], default=64)],
        valid=lambda c: c["chunk_size"] <= 512,
    )


CASES = [
    KernelCase(
        kernel_id="gla", shape_id=f"seq{sl}_h{h}",
        make_inputs=functools.partial(_inputs, sl, h),
        run=_run, reference=_ref, space=_space(),
        atol=2e-2, rtol=2e-2,
        note="chunked prefill GLA; ref=fused_recurrent (pure JAX)",
    )
    for (sl, h) in [(512, 8), (2048, 8)]
]
