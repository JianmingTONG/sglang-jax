"""Fused EP-MoE v1 (TPU-Pallas) — autotuning runner.

Tuned entry: ``fused_ep_moe(mesh, tokens, w1, w2, w3, topk_weights, topk_ids,
top_k, ..., block_config=FusedMoEBlockConfig(...))`` — a Mosaic/Pallas kernel.

The correctness contract (reference / tolerance / canonical inputs / native test)
is FROZEN in ``akt.benchmark.refs.moe_v1`` and imported below — this EDITABLE
runner owns ONLY the design space and the cfg->kernel run mapping, so a capability
can add tuning Knobs without weakening the correctness contract.

Design space (base): the block-config tile knobs ``bt`` (outer token tile),
``bf`` (intermediate tile), ``bd1``/``bd2`` (hidden tiles for w1 / w2). Each
config dict is mapped into a ``FusedMoEBlockConfig`` in :func:`_block_config`;
the un-tuned "compute" sub-tiles (``btc``/``bfc``/``bd1c``/``bd2c``) are pinned
to their block counterpart (compute-tile == block-tile).

Regime: the fused-MoE Pallas kernel is TPU-only and has NO ``interpret`` path,
so the kernel RUN is **tpu-deferred** on this GPU/CPU box (the Pallas call will
error here — expected). The runner is still fully wired: inputs, cfg->block
mapping, design space, and a runnable REFERENCE (validated on this box).
"""
from __future__ import annotations

import functools

from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe,
)

from akt.benchmark.refs.moe_v1 import (
    ATOL,
    RTOL,
    NATIVE_TEST,
    make_inputs,
    reference,
)
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob


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
import jax.numpy as jnp  # noqa: E402  (only for the shape constant below)

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
        make_inputs=functools.partial(make_inputs, _DTYPE, _TOPK, _NE, _H, _I, _NT),
        run=_run,
        reference=reference,
        space=_space(),
        atol=ATOL,
        rtol=RTOL,
        native_test=NATIVE_TEST,
        regime_pref=("tpu-deferred",),
        note="fused EP-MoE v1 (Pallas); ref=ref_moe (pure JAX); tpu-deferred "
        "(no interpret path, TPU-only)",
    )
]
