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
import math

import jax

from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe,
    get_dtype_packing,
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


# Small MoE: 32 tokens, 8 experts, top-2 (bf16). hidden=2048 / intermediate=1024.
import jax.numpy as jnp  # noqa: E402  (only for the shape constant below)

_DTYPE = jnp.bfloat16
_TOPK = 2
_NE = 8
_H = 2048
_I = 1024
_NT = 32

# ep_size = the "tensor" axis of the (1, ndev) mesh built by the frozen ref
# (akt.benchmark.refs.moe_v1._mesh reshapes jax.devices() to (1, ndev)), so it is
# just the device count on this box. It sets local_num_tokens = num_tokens/ep_size,
# which bounds the outer token tile `bt`.
_EP = jax.device_count()
_LOCAL_NT = _NT // _EP if _EP > 0 else _NT
_T_PACKING = get_dtype_packing(_DTYPE)      # bf16 -> 2 (the 32-bit repack width)
_TILE_ALIGN = _T_PACKING * 128              # bf16 -> 256 (bd1/bd2 alignment)

# Kernel VMEM budget (bench_fused_moe.DEFAULT_TPU_VMEM_BUDGET_MB) + the same 0.90
# headroom the tuner's select_block_configs applies, used as a size sanity bound.
_VMEM_BUDGET_BYTES = 96 * 1024 * 1024
_VMEM_HEADROOM = 0.90


def _vmem_bytes(bt: int, bf: int, bd1: int, bd2: int) -> int:
    """Conservative VMEM estimate for THIS case (bf16 weights, no shared expert,
    no quant) — the dominant scratch buffers of the fused_moe v1 Pallas kernel,
    mirroring benchmark/moe/bench_fused_moe._estimate_vmem_bytes. bts defaults to
    bt (the runner leaves FusedMoEBlockConfig.bts=None)."""
    bts = bt
    tb = wb = jnp.dtype(_DTYPE).itemsize            # token/weight bytes (bf16 -> 2)
    a2a_max_tokens = ((bt * _EP + bts - 1) // bts) * bts
    acc_bt = math.gcd(bt, 16)
    a2a_g_acc = 2 * _TOPK * acc_bt * _H * tb
    b_output = 2 * bt * _H * tb
    w1 = 2 * bd1 * bf * wb
    w3 = 2 * bd1 * bf * wb
    w2 = 2 * bf * bd2 * wb
    b_acc = 2 * a2a_max_tokens * bf * 4
    t_stage = 2 * bts * (bd1 // _T_PACKING) * 4
    a2a_s_acc = 3 * bts * (bd2 // _T_PACKING) * 4
    return a2a_g_acc + b_output + w1 + w3 + w2 + b_acc + t_stage + a2a_s_acc


def _valid(cfg: dict) -> bool:
    """Only enumerate configs the kernel actually supports for this shape — i.e.
    the ones that pass validate_fused_moe_block_config (kernel.py:244) with the
    runner's cfg->block mapping (btc=bt, bfc=bf, bd1c=bd1, bd2c=bd2, bse=bf,
    bts=bt). All the compute (*c) tiles equal their block tile, so only the
    block-tile divisibility rules below are load-bearing; the VMEM bound keeps the
    space from exploding with OOM configs. (These are the same divisibility rules
    the tuner's select_block_configs applies before benchmarking.)"""
    bt = cfg["bt"]
    bf = cfg["bf"]
    bd1 = cfg["bd1"]
    bd2 = cfg["bd2"]
    # bt: outer token tile. t_packing-aligned; 2/4 (small-batch decode) or mult of
    # 8; and must evenly divide the per-core token count (=> bt <= local_num_tokens).
    if bt % _T_PACKING != 0:
        return False
    if not (bt in (2, 4) or bt % 8 == 0):
        return False
    if _LOCAL_NT % bt != 0:
        return False
    # bf / bse: 128-aligned and must tile the intermediate dim.
    if bf % 128 != 0 or _I % bf != 0:
        return False
    # bd1 / bd2: hidden tiles. tile_align(=t_packing*128)-aligned and must tile the
    # hidden dim. The tuner's candidate generator (and every tuned-table row) ties
    # bd1 == bd2 == bd, so restrict to that supported diagonal.
    if bd1 % _TILE_ALIGN != 0 or _H % bd1 != 0:
        return False
    if bd2 % _TILE_ALIGN != 0 or _H % bd2 != 0:
        return False
    if bd1 != bd2:
        return False
    # Size sanity: stay within the kernel's VMEM budget (never binds at this small
    # shape, but guards against enumerating OOM configs for larger shapes).
    if _vmem_bytes(bt, bf, bd1, bd2) > int(_VMEM_BUDGET_BYTES * _VMEM_HEADROOM):
        return False
    return True


def _space():
    # Candidate value sets = the real supported ranges the sglang-jax maintainers
    # tune over: benchmark/moe/bench_fused_moe.run_all CLI defaults
    #   bt_candidates=[2,4,8,16,32,64,128,256,512]
    #   bf_candidates=[128,256,512,1024,2048]  (bse tracks bf here)
    #   bd_candidates=[256,512,1024,2048,4096,8192]  (tuner ties bd1=bd2=bd)
    # matching the value spans present in TUNED_BLOCK_CONFIGS. The `valid` guard
    # prunes each to the subset applicable to this shape (H=2048, I=1024).
    return DesignSpace(
        knobs=[
            Knob("bt", [2, 4, 8, 16, 32, 64, 128, 256, 512], default=32),
            Knob("bf", [128, 256, 512, 1024, 2048], default=512),
            Knob("bd1", [256, 512, 1024, 2048, 4096, 8192], default=1024),
            Knob("bd2", [256, 512, 1024, 2048, 4096, 8192], default=1024),
        ],
        valid=_valid,
    )


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
