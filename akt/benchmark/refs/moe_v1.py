"""FROZEN authoritative correctness contract for the fused EP-MoE v1 kernel.

Everything here is sourced from sglang-jax's OWN kernel + test code:

- ``reference`` calls the repo's pure-JAX oracle
  ``sgl_jax.srt.kernels.fused_moe.v1.kernel.ref_moe`` (defined at kernel.py:344).
  ``ref_moe`` re-derives its own top-k routing from ``gating_output``, so the
  reference needs only (tokens, w1, w2, w3, gating, top_k) plus the two flags
  (``renormalize_topk_logits``, ``act_fn``) — it does NOT consume the precomputed
  ``topk_weights`` / ``topk_ids`` (those feed only the tuned kernel run path).

- ``ATOL`` / ``RTOL`` are the EXACT tolerance the repo test asserts with on the
  basic path: ``_test_moe`` defaults ``atol=2e-1, rtol=2e-1``
  (python/sgl_jax/test/kernels/fused_moe_v1_test.py:173-174), and ``test_basic``
  (fused_moe_v1_test.py:326-351) calls ``_test_moe`` without overriding them.

- ``make_inputs`` reproduces the repo test's input generator ``gen_moe_inputs``
  (fused_moe_v1_test.py:37-113) on its BASIC path (``has_bias=False``,
  ``has_shared_expert=False`` -> b*/w*_shared are None). The generator is mirrored
  verbatim rather than imported because importing the test module pulls
  ``sgl_jax.srt.layers.moe`` (via ``TopK`` / ``create_moe_weights_mapping``), which
  requires ``transformers`` — a package absent from this venv. The routing
  (``topk_weights`` / ``topk_ids``) is produced by ``_route_topk``, a pure-JAX
  replica of ``TopK._topk`` (gate.py:131-132) followed by renormalization
  (gate.py:122-123, 127) — identical math to the non-grouped, no-bias TopK the
  test uses. The ("data","tensor") mesh mirrors the test's
  ``create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])``
  (fused_moe_v1_test.py:143) as ``(1, ndev)`` on this box.

- ``NATIVE_TEST = None`` — the fused-MoE v1 Pallas/Mosaic kernel is TPU-only and
  threads NO ``interpret`` flag, so ``MoEKernelTest`` cannot run off-TPU (the
  ``fused_ep_moe`` dispatch errors under Pallas interpret on CPU/GPU). Its module
  also fails to import here (missing ``transformers``). See the reason comment on
  ``NATIVE_TEST`` below.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.fused_moe.v1.kernel import ref_moe

# --- tolerance: fused_moe_v1_test.py:173-174 (_test_moe defaults), used unchanged
#     by test_basic (fused_moe_v1_test.py:326-351).
ATOL = 2e-1
RTOL = 2e-1

# --- native repo test: None. The fused-MoE v1 Pallas kernel is TPU-only (no
#     interpret path -> fused_ep_moe errors under Pallas interpret on this
#     CPU/GPU box), and its test module (fused_moe_v1_test.py) is not even
#     importable here (it pulls sgl_jax.srt.layers.moe -> requires `transformers`,
#     absent in this venv). So there is no repo test that PASSES off-TPU to wire.
NATIVE_TEST = None


# --------------------------------------------------------------- inputs
# Mirrored verbatim from python/sgl_jax/test/kernels/fused_moe_v1_test.py:37-113
# (BASIC path: has_bias=False, has_shared_expert=False -> the b*/w*_shared tensors
# are None and are dropped, since this reference exercises the basic MoE only).
def gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 12)
    k0, k1, k2, k3, _, _, _, k7, k8 = keys[:9]

    a = jax.random.normal(k0, (num_tokens, hidden_size), dtype=jnp.float32).astype(dtype) / 10
    w1 = (
        jax.random.normal(k1, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w2 = (
        jax.random.normal(k2, (num_experts, intermediate_size, hidden_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w3 = (
        jax.random.normal(k3, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)

    # Deterministic, strictly-ordered top-k per token (top-1 > top-2 > ...).
    gating_output = jax.random.normal(k7, (num_tokens, num_experts), dtype=jnp.float32)
    token_keys = jax.random.split(k8, num_tokens)
    top_k_indices = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
        token_keys
    ).astype(jnp.int32)
    boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
    one_hot = jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32) * boosts[..., None],
        axis=1,
    )
    gating_output = (gating_output + one_hot).astype(dtype)
    return a, w1, w2, w3, gating_output


@functools.lru_cache(maxsize=1)
def _mesh():
    # Mesh axes ("data", "tensor"); ep = data * tensor, matching fused_ep_moe
    # defaults and the test's create_device_mesh(ici_parallelism=[1, -1], ...)
    # (fused_moe_v1_test.py:143). ici=[1,-1] -> data=1, tensor=ndev on this box.
    devs = jax.devices()
    return jax.sharding.Mesh(np.asarray(devs).reshape(1, len(devs)), ("data", "tensor"))


def _route_topk(gating, top_k, renormalize=True):
    """Pure-JAX replica of TopK._topk + renormalize (non-grouped, no bias).

    Mirrors sgl_jax.srt.layers.gate.TopK: router_logits.astype(f32) (gate.py:104),
    ._topk = jax.lax.top_k(router_logits, topk) (gate.py:131-132), renormalize
    topk_weights /= sum(topk_weights, -1, keepdims) (gate.py:122-123), and
    topk_weights.astype(f32) (gate.py:127).
    """
    logits = gating.astype(jnp.float32)
    w, ids = jax.lax.top_k(logits, top_k)
    if renormalize:
        w = w / jnp.sum(w, axis=-1, keepdims=True)
    return w.astype(jnp.float32), ids.astype(jnp.int32)


def make_inputs(dtype, top_k, num_experts, hidden, inter, num_tokens, seed=1234) -> dict:
    """Canonical inputs for one MoE problem. Returns both the reference inputs
    (tokens, w*, gating, top_k) and the tuned-kernel run inputs (mesh, topk_*)."""
    a, w1, w2, w3, gating = gen_moe_inputs(
        dtype, top_k, num_experts, hidden, inter, num_tokens, seed=seed
    )
    topk_weights, topk_ids = _route_topk(gating, top_k, renormalize=True)
    return {
        "mesh": _mesh(),
        "tokens": a,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "gating": gating,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "top_k": top_k,
    }


# --------------------------------------------------------------- reference
def reference(inp: dict):
    """sglang-jax's OWN pure-JAX oracle (kernel.py:344). Re-derives top-k routing
    from `gating` internally; matches the tuned kernel's renormalize + silu."""
    return ref_moe(
        inp["tokens"],
        inp["w1"],
        inp["w2"],
        inp["w3"],
        inp["gating"],
        inp["top_k"],
        renormalize_topk_logits=True,
        act_fn="silu",
    )
