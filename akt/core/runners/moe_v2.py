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


def _vmem_ok(bt: int, bf: int, h: int, budget: int = 60 * 1024 * 1024) -> bool:
    """Conservative VMEM proxy mirroring the DOMINANT terms of bench_v2's
    ``_estimate_vmem_bytes_v2`` for the bf16 / no-shared-expert regime this case
    runs in (weight double-buffers dominate; per-``bts`` accumulators are small,
    ``bts == bt`` here). Prunes any tile whose live VMEM would blow past the
    kernel's ~64 MB budget, so widening the value lists cannot enumerate an
    OOM config. Deliberately over-counts (2-byte bf16 weights, x2 buffering)."""
    bts = bt
    # w1,w3 double buffers (2, h, bf) + w2 double buffer (2, bf, h), bf16 (2 bytes).
    w = 2 * (2 * h * bf * 2) + 2 * (bf * h * 2)
    # gate/up accumulators (bts, bf) f32 x2 + x/y_acc/y_stage (bts, h) + 2 output banks.
    acc = 2 * (bts * bf * 4) + bts * h * (2 + 4 + 2) + 2 * (bt * h * 2)
    return (w + acc) < budget


def _space(nt: int, h: int, i: int) -> DesignSpace:
    """The v2 block-config design space: the maintainer-SUPPORTED tile candidates,
    not a hand-picked subset.

    Value lists are the union of the shipped tuned table
    (``fused_moe/v2/tuned_block_configs.py::TUNED_BLOCK_CONFIGS``) and the
    ``bench_v2.py::generate_tune_candidates`` ladders:
      * ``bt``  — power-of-2 outer-token ladder {8..256} (tuned table + generator).
      * ``bf``  — intermediate-tile ladder {128,256,512,1024,2048} (generator
                  ``bf_list``; tuned table adds nothing beyond it).
      * ``btc`` — 8-aligned compute sub-tile ladder {8..128} (generator emits the
                  8-aligned divisors of ``bts``; power-of-2 ``bts`` -> power-of-2 btc).
      * ``bse`` — SE intermediate-tile ladder {128,256,512,1024,2048} (tuned table
                  {128,256,512,1024} + generator SE candidate set).

    The ``valid`` guard prunes to the configs the kernel actually supports for THIS
    shape (``nt`` tokens, hidden=``h``, intermediate=``i``), mirroring the kernel's
    own asserts (kernel.py:235-246) + ``effective_for`` + the bench VMEM filter.
    ``bts`` is left at ``bt`` (the runner passes ``bts=None`` -> ``effective_for``
    sets ``bts=bt``), so every bts-based check below uses ``bt``."""
    return DesignSpace(
        knobs=[
            Knob("bt", [8, 16, 32, 64, 128, 256], default=32),
            Knob("bf", [128, 256, 512, 1024, 2048], default=512),
            Knob("btc", [8, 16, 32, 64, 128], default=32),
            Knob("bse", [128, 256, 512, 1024, 2048], default=256),
        ],
        valid=lambda c: (
            # bf: divides the intermediate dim and is 128-lane aligned (kernel asserts).
            i % c["bf"] == 0 and c["bf"] % 128 == 0
            # btc: 8-aligned (VREG sublane) divisor of bts(=bt), and btc <= bt.
            and c["btc"] % 8 == 0 and c["btc"] <= c["bt"] and c["bt"] % c["btc"] == 0
            # bt: divides the (single-host) token count; bts=bt so the same bound.
            and c["bt"] <= nt and nt % c["bt"] == 0
            # bse: SE intermediate tile <= bf and divides the intermediate dim.
            and c["bse"] <= c["bf"] and i % c["bse"] == 0
            # VMEM sanity so the widened lists never enumerate an OOM tile.
            and _vmem_ok(c["bt"], c["bf"], h)
        ),
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
        space=_space(_NT, _H, _I),
        atol=ATOL,
        rtol=RTOL,
        native_test=NATIVE_TEST,
        regime_pref=("tpu-deferred",),
        note="fused EP-MoE v2 (Pallas); ref=ref_moe (pure JAX); tpu-deferred "
        "(no interpret path, TPU-only)",
    )
]
