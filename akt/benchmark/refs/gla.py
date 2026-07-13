"""FROZEN correctness contract for the `gla` (simple gated linear attention) kernel.

Authoritative sglang-jax sources (imported / mirrored with citation):
  * REFERENCE — `fused_recurrent_simple_gla` (pure-JAX `lax.scan` recurrent path),
    from `python/sgl_jax/srt/kernels/simple_gla/simple_gla.py`. This is exactly the
    reference the maintainers' own kernel test validates the Pallas twin against
    (see `simple_gla_fused_test.py:23` — "Reference path (the kernel we are
    replacing)").
  * TOLERANCE — `_ATOL = _RTOL = 1e-3`, the EXACT tolerance used by the repo's
    kernel test `python/sgl_jax/test/kernels/simple_gla_fused_test.py:30-31`.
    That test covers the DECODE fused kernel; our runner tunes the CHUNK/prefill
    kernel (`chunk_simple_gla_fwd_varlen`). Both are validated against the same
    `fused_recurrent_simple_gla` reference, so we adopt the same 1e-3 tolerance.
    Empirically the chunk kernel matches the recurrent reference to ~1e-8 (float32
    machine precision) in Pallas interpret across the whole design space, so 1e-3
    passes with ~5 orders of magnitude of margin here — NO loosening was needed.
  * INPUT GENERATOR — canonical qkv/gamma construction mirrors the repo test's
    `_make_qkv` (standard-normal * 0.1) and `_make_g_gamma`
    (uniform(-0.1, -0.01) per head), `simple_gla_fused_test.py:37-61`. The chunk
    kernel is a PREFILL/varlen kernel, so inputs are the packed 4D `[B,T,H,D]`
    layout with a single `cu_seqlens` segment (the decode test builds 3D decode
    tensors — same statistics, prefill shape).

  * NATIVE_TEST — `simple_gla_fused_test.py` is the maintainers' simple_gla
    correctness check. It exercises the DECODE kernel (not the chunk kernel), but
    it is the repo's simple_gla test and it PASSES in Pallas interpret here
    (validated: 12 passed).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.simple_gla.simple_gla import fused_recurrent_simple_gla

# --- tolerance: EXACT repo test value (simple_gla_fused_test.py:30-31) ----------
ATOL = 1e-3
RTOL = 1e-3

# --- native repo test (validated: 12 passed in interpret here) ------------------
NATIVE_TEST = "python/sgl_jax/test/kernels/simple_gla_fused_test.py"

_HEAD_DIM = 128  # K = V = 128 (simple-GLA constraint, 128-aligned)


def make_inputs(seqlen: int, heads: int, seed: int = 0) -> dict:
    """Canonical packed-varlen prefill inputs for chunk simple-GLA.

    Input statistics mirror the repo decode test's `_make_qkv` (N(0,1)*0.1) and
    `_make_g_gamma` (uniform(-0.1,-0.01) per head), `simple_gla_fused_test.py:37-61`,
    reshaped to the 4D `[B, T, H, D]` packed layout the chunk kernel consumes with a
    single `cu_seqlens` segment.
    """
    rng = np.random.default_rng(seed)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.1, dtype=jnp.float32)
    q = mk(1, seqlen, heads, _HEAD_DIM)
    k = mk(1, seqlen, heads, _HEAD_DIM)
    v = mk(1, seqlen, heads, _HEAD_DIM)
    g_gamma = jnp.asarray(rng.uniform(-0.1, -0.01, size=(heads,)), dtype=jnp.float32)
    cu = jnp.asarray([0, seqlen], dtype=jnp.int32)  # one packed sequence
    return {"q": q, "k": k, "v": v, "g_gamma": g_gamma, "cu": cu, "seqlen": seqlen}


@functools.lru_cache(maxsize=1)
def _jit_ref():
    def f(q, k, v, g_gamma, cu):
        o, _ht = fused_recurrent_simple_gla(
            q, k, v, g_gamma=g_gamma, scale=None, cu_seqlens=cu)
        return o
    return jax.jit(f)


def reference(inp: dict):
    """Ground-truth output `o` [B,T,H,V] from the pure-JAX recurrent path."""
    return _jit_ref()(inp["q"], inp["k"], inp["v"], inp["g_gamma"], inp["cu"])
