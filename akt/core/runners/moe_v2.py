"""Fused EP-MoE v2 (TPU-Pallas) — autotuning runner.

Tuned entry: ``fused_ep_moe_v2(mesh, tokens, w1, w2, w3, topk_weights, topk_ids,
top_k, ..., block_config=FusedMoEBlockConfig(...))`` — a Mosaic/Pallas kernel.

The correctness contract (reference impl, tolerance, canonical inputs, and the
NATIVE_TEST repo nodeid) lives in the FROZEN module
``akt.benchmark.refs.moe_v2`` and is imported below — this EDITABLE runner owns
ONLY the tuning DesignSpace, the cfg->FusedMoEBlockConfig mapping, and the CASES
assembly. See that module for reference/tolerance/input provenance.

Design space (base): the v2 block-config knobs ``bt`` (outer token tile), ``bf``
(intermediate tile), ``btc`` (compute token sub-tile), ``bse`` (SE intermediate
tile). The config dict maps directly into ``FusedMoEBlockConfig(bt, bf, btc,
bse)`` in :func:`_block_config`; ``bts`` defaults to ``bt`` inside the kernel.

Regime: the fused-MoE v2 Pallas kernel is TPU-only and has NO ``interpret`` path,
so the kernel RUN is **tpu-deferred** on this GPU/CPU box (the Pallas call will
error here — expected). The runner is still fully wired: inputs, cfg->block
mapping, design space, and a runnable REFERENCE (validated on this box).
"""
from __future__ import annotations

import functools

import jax.numpy as jnp

from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe_v2,
)

from akt.benchmark.refs.moe_v2 import (
    ATOL,
    NATIVE_TEST,
    RTOL,
    make_inputs,
    reference,
)
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob


# --------------------------------------------------------------- cfg -> block
def _block_config(cfg: dict) -> FusedMoEBlockConfig:
    """Map the tuned tile knobs into a v2 FusedMoEBlockConfig (bts defaults to bt
    inside the kernel's effective_for)."""
    return FusedMoEBlockConfig(
        bt=int(cfg["bt"]),
        bf=int(cfg["bf"]),
        btc=int(cfg["btc"]),
        bse=int(cfg["bse"]),
    )


def _run(inp, cfg):
    return fused_ep_moe_v2(
        inp["mesh"],
        inp["tokens"],
        inp["w1"],
        inp["w2"],
        inp["w3"],
        inp["topk_weights"],
        inp["topk_ids"],
        inp["top_k"],
        act_fn="silu",
        block_config=_block_config(cfg),
        dp_axis_name="data",
        tp_axis_name="tensor",
    )


def _space():
    return DesignSpace(
        knobs=[
            Knob("bt", [8, 16, 32, 64], default=32),
            Knob("bf", [256, 512], default=512),
            Knob("btc", [8, 32], default=32),
            Knob("bse", [128, 256], default=256),
        ],
        # compute token sub-tile must not exceed the outer token tile (mirrors the
        # kernel's effective_for: btc = min(btc, bts=bt)).
        valid=lambda c: c["btc"] <= c["bt"],
    )


# Small MoE: 32 tokens, 8 experts, top-2, hidden=intermediate=512 (bf16).
_DTYPE = jnp.bfloat16
_TOPK = 2
_NE = 8
_H = 512
_I = 512
_NT = 32

CASES = [
    KernelCase(
        kernel_id="moe_v2",
        shape_id=f"t{_NT}_e{_NE}_k{_TOPK}_h{_H}_i{_I}",
        make_inputs=functools.partial(make_inputs, _DTYPE, _TOPK, _NE, _H, _I, _NT),
        run=_run,
        reference=reference,
        space=_space(),
        atol=ATOL,
        rtol=RTOL,
        native_test=NATIVE_TEST,
        regime_pref=("tpu-deferred",),
        note="fused EP-MoE v2 (Pallas); ref=ref_moe (pure JAX); tpu-deferred "
        "(no interpret path, TPU-only)",
    )
]
