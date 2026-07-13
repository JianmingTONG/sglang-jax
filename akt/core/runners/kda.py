"""KDA (Kimi Delta Attention) — chunked varlen prefill kernel.

Tuned entry: `chunk_kda_fwd(..., chunk_size)` (Pallas, threads an `interpret`
flag via `get_interpret()` → runs here under PALLAS_INTERPRET=1 on CPU).
Reference:   `naive_recurrent_kda(...)` (pure JAX per-step delta recurrence).
Design space (base): `chunk_size` — the BT time tile the four-stage pipeline
(gate cumsum / intra solve / inter-chunk state / output) blocks over. K/V head
dims are padded to 128 inside the kernel; elevating independent BK/BV tiling
would be a natural CAPABILITY.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.kda import chunk_kda, naive_recurrent_kda

from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob


def _inputs(seqlen: int, heads: int, head_dim: int, seed: int = 0):
    # B=1 packed varlen layout (one sequence: cu_seqlens=[0, T]); fp32 to keep
    # chunk-vs-recurrence noise well under atol. g is a small negative log-gate
    # (decay), matching the model's g = -exp(A)*softplus(.) sign convention.
    rng = np.random.default_rng(seed)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.1, dtype=jnp.float32)
    q = mk(1, seqlen, heads, head_dim)
    k = mk(1, seqlen, heads, head_dim)
    v = mk(1, seqlen, heads, head_dim)
    g = -jax.nn.softplus(
        jnp.asarray(rng.standard_normal((1, seqlen, heads, head_dim)) * 0.5, jnp.float32)
    ) * 0.1
    beta = jnp.asarray(rng.uniform(0.0, 1.0, size=(1, seqlen, heads)), dtype=jnp.float32)
    cu = jnp.asarray([0, seqlen], dtype=jnp.int32)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "cu": cu, "scale": head_dim ** -0.5}


@functools.lru_cache(maxsize=None)
def _jit_chunk(chunk_size: int, scale: float):
    # chunk_kda_fwd is itself jitted with static chunk_size/output_final_state;
    # initial_state=None, output_final_state=False → fresh state, output-only.
    def f(q, k, v, g, beta, cu):
        out = chunk_kda(q, k, v, g, beta, scale, None, False, cu, chunk_size=chunk_size)
        return out[0]  # o
    return jax.jit(f)


@functools.lru_cache(maxsize=None)
def _jit_ref(scale: float):
    def f(q, k, v, g, beta):
        o, _ht = naive_recurrent_kda(q, k, v, g, beta, scale=scale)
        return o
    return jax.jit(f)


def _run(inp, cfg):
    return _jit_chunk(int(cfg["chunk_size"]), float(inp["scale"]))(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["cu"])


def _ref(inp):
    return _jit_ref(float(inp["scale"]))(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])


def _space():
    return DesignSpace(
        knobs=[Knob("chunk_size", [16, 32, 64, 128], default=64)],
        valid=lambda c: c["chunk_size"] >= 16 and (c["chunk_size"] & (c["chunk_size"] - 1)) == 0,
    )


CASES = [
    KernelCase(
        kernel_id="kda", shape_id=f"seq{sl}_h{h}_d{d}",
        make_inputs=functools.partial(_inputs, sl, h, d),
        run=_run, reference=_ref, space=_space(),
        atol=1e-2, rtol=2e-2,
        check_out=lambda o: o,  # _run/_ref already return o
        regime_pref=("cpu-interpret",),
        note="chunked varlen KDA; ref=naive_recurrent (pure JAX); runs in CPU-interpret",
    )
    for (sl, h, d) in [(128, 2, 64), (256, 4, 128)]
]
