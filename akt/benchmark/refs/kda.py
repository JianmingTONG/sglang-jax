"""FROZEN authoritative correctness reference for the KDA (Kimi Delta Attention)
chunked varlen prefill kernel.

All correctness material is sourced from sglang-jax's OWN code:

  * Reference implementation — ``sgl_jax.srt.kernels.kda.naive_recurrent_kda``
    (python/sgl_jax/srt/kernels/kda/naive.py): the pure-JAX per-step delta
    recurrence the chunked kernel must match. This is the SAME reference the repo
    test drives — python/sgl_jax/test/test_kda_attention.py:151 (ref_kda_attention)
    calls ``naive_recurrent_kda`` per sequence.

  * Tolerance — python/sgl_jax/test/test_kda_attention.py:391-392
    (``TestKDAAttention.setUp``): ``self.rtol = 2e-2`` / ``self.atol = 1e-2``,
    enforced by the ``np.testing.assert_allclose`` calls at lines 461-467.

  * Input convention — python/sgl_jax/test/test_kda_attention.py:24-29
    (``_scaled_randn``: ``standard_normal * 0.1`` shrinks the recurrent state so
    the shared atol holds) and the log-space gate sign convention
    ``g = -exp(A) * softplus(.)`` at lines 141-143 (``ref_kda_attention``).

``make_inputs`` builds ``q/k/v/g/beta/cu/scale`` DIRECTLY at the
``naive_recurrent_kda`` / ``chunk_kda`` call boundary — B=1 packed varlen, a
single sequence ``cu=[0, T]`` — i.e. the level the tuned Pallas kernel actually
operates at. The repo test's ``create_test_data`` produces the equivalent tensors
one abstraction up (after short-conv + l2-normalize + gate activation, through the
full ``RadixLinearAttention`` backend + ``RecurrentStatePool``), which is not
importable in this pure-Pallas interpret environment (see ``NATIVE_TEST``).

This module is FROZEN: it is the correctness contract. The EDITABLE runner
(akt/core/runners/kda.py) imports ``reference`` / ``ATOL`` / ``RTOL`` /
``make_inputs`` / ``NATIVE_TEST`` / ``check_out`` from here and owns only the
DesignSpace + the config->kernel run-mapping.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.kda import naive_recurrent_kda

# python/sgl_jax/test/test_kda_attention.py:391-392 (TestKDAAttention.setUp):
#   self.rtol = 2e-2 ; self.atol = 1e-2  (asserted at lines 461-467).
ATOL = 1e-2
RTOL = 2e-2

# NATIVE_TEST = None: the repo test python/sgl_jax/test/test_kda_attention.py cannot
# be collected or executed off-TPU in this interpret environment. Importing it pulls
# the full backend stack (kda_backend -> gdn_backend -> hybrid_linear_attn_backend ->
# model_executor.forward_batch_info -> configs.model_config -> `transformers`), and
# `transformers` is not installed here (ModuleNotFoundError at collection time). The
# test also drives the end-to-end RadixLinearAttention backend + RecurrentStatePool
# (bf16, sharded mesh) rather than chunk_kda directly, so it is not a self-contained
# interpret check. Wire it once `transformers` (and, for faithful latency, a TPU) are
# available; the reference above is the same naive_recurrent_kda the test asserts on.
NATIVE_TEST = None


def make_inputs(seqlen: int, heads: int, head_dim: int, seed: int = 0) -> dict:
    """Canonical KDA kernel inputs at the naive_recurrent_kda / chunk_kda boundary.

    B=1 packed varlen layout (one sequence: ``cu_seqlens=[0, T]``); fp32 to keep the
    chunk-vs-recurrence noise well under atol. ``g`` is a small negative log-gate
    (decay) matching the model's ``g = -softplus(.)`` sign convention
    (test_kda_attention.py:141-143); the ``* 0.1`` scaling mirrors ``_scaled_randn``
    (test_kda_attention.py:24-29).
    """
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
def _jit_ref(scale: float):
    def f(q, k, v, g, beta):
        o, _ht = naive_recurrent_kda(q, k, v, g, beta, scale=scale)
        return o
    return jax.jit(f)


def reference(inp: dict):
    """Ground-truth output ``o`` from sglang-jax's own pure-JAX delta recurrence."""
    return _jit_ref(float(inp["scale"]))(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])


def check_out(o):
    """reference/run both already return just the output tensor ``o``."""
    return o
