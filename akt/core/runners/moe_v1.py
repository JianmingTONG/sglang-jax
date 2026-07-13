"""Fused EP-MoE v1 (TPU-Pallas) — autotuning runner.

Tuned entry: ``fused_ep_moe(mesh, tokens, w1, w2, w3, topk_weights, topk_ids,
top_k, ..., block_config=FusedMoEBlockConfig(...))`` — a Mosaic/Pallas kernel.
Reference:   ``ref_moe(tokens, w1, w2, w3, gating_output, top_k, ...)`` — pure JAX.

Design space (base): the block-config tile knobs ``bt`` (outer token tile),
``bf`` (intermediate tile), ``bd1``/``bd2`` (hidden tiles for w1 / w2). Each
config dict is mapped into a ``FusedMoEBlockConfig`` in :func:`_block_config`;
the un-tuned "compute" sub-tiles (``btc``/``bfc``/``bd1c``/``bd2c``) are pinned
to their block counterpart (compute-tile == block-tile).

Regime: the fused-MoE Pallas kernel is TPU-only and has NO ``interpret`` path,
so the kernel RUN is **tpu-deferred** on this GPU/CPU box (the Pallas call will
error here — expected). The runner is still fully wired: inputs, cfg->block
mapping, design space, and a runnable REFERENCE (validated on this box).

Notes on box-local reuse: the upstream test's ``gen_moe_inputs`` and the ``TopK``
routing module both pull deps (``absl`` / ``flax.nnx``) that are absent in this
venv, so the small-problem generator is inlined verbatim from
``fused_moe_v1_test.py`` and the (non-grouped, no-bias) TopK routing is
replicated in pure JAX — identical math to ``TopK._topk`` + renormalize.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe,
    ref_moe,
)

from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob


# --------------------------------------------------------------- inputs
# Inlined verbatim from python/sgl_jax/test/kernels/fused_moe_v1_test.py
# (basic path: has_bias=False, has_shared_expert=False -> b*/w*_shared = None).
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


def _inputs(dtype, top_k, num_experts, hidden, inter, num_tokens, seed=1234):
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


# --------------------------------------------------------------- cfg -> block
def _block_config(cfg: dict) -> FusedMoEBlockConfig:
    """Map the tuned tile knobs into a FusedMoEBlockConfig. The compute sub-tiles
    (*c) are pinned to their block counterpart; bse (SE block) tracks bf."""
    bt = int(cfg["bt"])
    bf = int(cfg["bf"])
    bd1 = int(cfg["bd1"])
    bd2 = int(cfg["bd2"])
    return FusedMoEBlockConfig(
        bt=bt,
        btc=bt,
        bf=bf,
        bfc=bf,
        bd1=bd1,
        bd1c=bd1,
        bd2=bd2,
        bd2c=bd2,
        bse=bf,
    )


def _run(inp, cfg):
    return fused_ep_moe(
        mesh=inp["mesh"],
        tokens=inp["tokens"],
        w1=inp["w1"],
        w2=inp["w2"],
        w3=inp["w3"],
        topk_weights=inp["topk_weights"],
        topk_ids=inp["topk_ids"],
        top_k=inp["top_k"],
        act_fn="silu",
        renormalize_topk_logits=True,
        block_config=_block_config(cfg),
        tp_axis_name="tensor",
    )


def _ref(inp):
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


def _space():
    return DesignSpace(
        knobs=[
            Knob("bt", [8, 16, 32, 64], default=32),
            Knob("bf", [256, 512, 1024], default=512),
            Knob("bd1", [1024, 2048], default=1024),
            Knob("bd2", [1024, 2048], default=1024),
        ],
    )


# Small MoE: 32 tokens, 8 experts, top-2 (bf16). hidden=2048 / intermediate=1024
# are the smallest dims for which EVERY enumerated config is validly applicable:
# hidden must be a multiple of both bd1/bd2 candidates {1024, 2048} and
# intermediate a multiple of the bf candidates {256, 512, 1024}. (So the run
# reaches the real Pallas/TPU dispatch rather than a config-alignment reject.)
_DTYPE = jnp.bfloat16
_TOPK = 2
_NE = 8
_H = 2048
_I = 1024
_NT = 32

CASES = [
    KernelCase(
        kernel_id="moe_v1",
        shape_id=f"t{_NT}_e{_NE}_k{_TOPK}_h{_H}_i{_I}",
        make_inputs=functools.partial(_inputs, _DTYPE, _TOPK, _NE, _H, _I, _NT),
        run=_run,
        reference=_ref,
        space=_space(),
        atol=2e-1,
        rtol=2e-1,
        regime_pref=("tpu-deferred",),
        note="fused EP-MoE v1 (Pallas); ref=ref_moe (pure JAX); tpu-deferred "
        "(no interpret path, TPU-only)",
    )
]
