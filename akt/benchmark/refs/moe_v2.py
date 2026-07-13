"""FROZEN authoritative correctness reference for the fused EP-MoE v2 kernel.

Every piece here is sourced from sglang-jax's OWN kernel/test code:

- Reference impl (the correctness contract):
    ``python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py::ref_moe``
    — the v2 kernel's own pure-JAX reference. It takes PRE-ROUTED top-k
    (``topk_weights``/``topk_ids``), not raw gating. IMPORTED, not reimplemented.

- Tolerance:
    ``python/sgl_jax/test/kernels/fused_moe_v2_test.py`` — ``_test_moe`` defaults
    ``atol=2e-1, rtol=2e-1`` (lines 151-152), which is the tolerance the basic
    bf16 path (``test_basic``, lines 255-257) asserts ``fused_ep_moe_v2`` against
    ``ref_moe`` with. (The fp8 tests tighten to 5e-2; we mirror the bf16 basic
    path, so 2e-1.)

- Canonical inputs:
    ``fused_moe_v2_test.py::gen_moe_inputs`` (bf16 basic path:
    ``has_shared_expert=False`` -> ``w*_shared = None``) is inlined VERBATIM
    below. It is inlined rather than imported because the test module pulls
    ``absl.testing`` / ``flax.nnx`` (via ``TopK``) at import time, and those deps
    are absent in this venv. The routing (``sgl_jax.srt.layers.moe.TopK._topk`` +
    renormalize, non-grouped / no-bias) is likewise replicated in pure JAX —
    identical math to ``TopK._topk``.

NATIVE_TEST = None. The ``fused_ep_moe_v2`` Pallas/Mosaic kernel is TPU-only and
threads NO ``interpret`` path, so ``MoEV2KernelTest`` cannot pass off-TPU (the
Mosaic call errors under PALLAS_INTERPRET on this GPU/CPU box). The case is
therefore tpu-deferred; only the reference (``ref_moe``, pure JAX) is validated
here. Wire a real repo nodeid only once a TPU backend is attached.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.fused_moe.v2.kernel import ref_moe

# Tolerance — fused_moe_v2_test.py:151-152 (_test_moe defaults; basic bf16 path).
ATOL = 2e-1
RTOL = 2e-1

# fused_ep_moe_v2 is TPU-only (no interpret path); MoEV2KernelTest cannot pass
# off-TPU here. See module docstring. tpu-deferred: only ref_moe is validated.
NATIVE_TEST = None


# --------------------------------------------------------------- inputs
# Inlined verbatim from fused_moe_v2_test.py::gen_moe_inputs (bf16 basic path:
# has_shared_expert=False -> w*_shared = None).
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
    k0, k1, k2, k3, k7, k8 = keys[0], keys[1], keys[2], keys[3], keys[7], keys[8]

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

    # Strictly-ordered, deterministic top-k per token (top-1 > top-2 > ...).
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
    # Mesh axes ("data", "tensor"); ep = data * tensor, matching fused_ep_moe_v2
    # defaults. ici_parallelism=[1, -1] -> data=1, tensor=ndev on this box.
    devs = jax.devices()
    return jax.sharding.Mesh(np.asarray(devs).reshape(1, len(devs)), ("data", "tensor"))


def _route_topk(gating, top_k, renormalize=True):
    """Pure-JAX replica of TopK._topk + renormalize (non-grouped, no bias)."""
    logits = gating.astype(jnp.float32)
    w, ids = jax.lax.top_k(logits, top_k)
    if renormalize:
        w = w / jnp.sum(w, axis=-1, keepdims=True)
    return w.astype(jnp.float32), ids.astype(jnp.int32)


def make_inputs(dtype, top_k, num_experts, hidden, inter, num_tokens, seed=1234):
    """Canonical inputs for the v2 case: the (bf16) gen_moe_inputs tensors plus
    the pre-routed top-k that both the kernel and ref_moe consume, and the mesh
    the Pallas kernel is sharded over. reference() ignores ``mesh``/``top_k`` keys
    it does not need."""
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
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "top_k": top_k,
    }


# --------------------------------------------------------------- reference
def reference(inp):
    """Authoritative ground truth: sglang-jax's own ``ref_moe`` (pure JAX), on the
    same pre-routed top-k the kernel consumes."""
    return ref_moe(
        inp["tokens"],
        inp["w1"],
        inp["w2"],
        inp["w3"],
        inp["topk_weights"],
        inp["topk_ids"],
        inp["top_k"],
        act_fn="silu",
    )
