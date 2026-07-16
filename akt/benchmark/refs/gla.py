"""FROZEN correctness contract for the `gla` (simple gated linear attention) kernel.

Authoritative sglang-jax sources (imported / mirrored with citation):
  * REFERENCE — a frozen transcription of the `fused_recurrent_simple_gla`
    pure-JAX `lax.scan` recurrence from
    `python/sgl_jax/srt/kernels/simple_gla/simple_gla.py`. Keeping the recurrence
    here prevents an editable production-kernel patch from changing both the
    implementation and its supposed oracle in the same round.
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
    layout with a single `cu_seqlens` segment and a nonzero recurrent input state
    (the decode test builds 3D decode tensors — same statistics, prefill shape).
    Correctness covers both token output and the final state consumed by decode.

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
    single `cu_seqlens` segment. A nonzero state makes state propagation observable;
    the reference returns both token output and final recurrent state.
    """
    rng = np.random.default_rng(seed)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.1, dtype=jnp.float32)
    q = mk(1, seqlen, heads, _HEAD_DIM)
    k = mk(1, seqlen, heads, _HEAD_DIM)
    v = mk(1, seqlen, heads, _HEAD_DIM)
    g_gamma = jnp.asarray(rng.uniform(-0.1, -0.01, size=(heads,)), dtype=jnp.float32)
    initial_state = jnp.asarray(
        rng.standard_normal((1, heads, _HEAD_DIM, _HEAD_DIM)) * 0.01,
        dtype=jnp.float32,
    )
    cu = jnp.asarray([0, seqlen], dtype=jnp.int32)  # one packed sequence
    return {
        "q": q,
        "k": k,
        "v": v,
        "g_gamma": g_gamma,
        "initial_state": initial_state,
        "cu": cu,
        "seqlen": seqlen,
    }


@functools.lru_cache(maxsize=1)
def _jit_ref():
    def f(q, k, v, g_gamma, initial_state, cu):
        B, T, H, K = q.shape
        V = v.shape[-1]
        assert B == 1
        nseq = cu.shape[0] - 1
        token_idx = jnp.arange(T, dtype=cu.dtype)
        seq_ids = jnp.searchsorted(cu[1:], token_idx, side="right")
        reset = token_idx == cu[:-1][seq_ids]
        h0 = initial_state.astype(q.dtype)

        def step(carry, xs):
            h_prev = carry
            q_i, k_i, v_i, seq_id, do_reset = xs
            h = jnp.where(do_reset, h0[seq_id], h_prev)
            h = h * jnp.exp(g_gamma)[:, None, None]
            h = h + k_i[:, :, None] * v_i[:, None, :]
            out = jnp.sum(h * (q_i[:, :, None] * (K**-0.5)), axis=1)
            return h, (out, h)

        initial = jnp.zeros((H, K, V), dtype=q.dtype)
        _h, (out, states) = jax.lax.scan(
            step,
            initial,
            (q[0], k[0], v[0], seq_ids, reset),
        )
        final_state = states[cu[1:] - 1]
        return out[None], final_state

    return jax.jit(f)


def reference(inp: dict):
    """Ground-truth output and observed final state for serving prefill."""
    return _jit_ref()(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g_gamma"],
        inp["initial_state"],
        inp["cu"],
    )
