"""Fused MLP (SwiGLU) — TPU-Pallas kernel `apply_fused_mlp_sharded`.

Fuses gate/up projections + SiLU gating + down projection into one pipelined TPU
kernel (`srt/kernels/fused_mlp.py`). It is Mosaic-only (no interpret path), so it
is **tpu-deferred** on this box: fully wired (inputs, config->tiling map, design
space) with its definitive latency awaiting a TPU. The pure-JAX reference runs
here and is validated; the Pallas run raises on CPU/GPU.

The authoritative correctness contract (reference / tolerances / canonical inputs /
NATIVE_TEST) lives in the FROZEN `akt.benchmark.refs.fused_mlp`; this EDITABLE
runner owns only the DesignSpace + the config->kernel `run` mapping + CASES.

Weight layout. The kernel's fused weight `w_gu` is column-INTERLEAVED per
intermediate tile: block `i` is `[gate_i (b_inter) | up_i (b_inter)]`
(see `mlp_kernel_main`'s BlockSpec, which loads `w_gu[:, i*2*b_inter:(i+1)*2*b_inter]`
and splits it at `b_inter`). So the interleaving DEPENDS on `b_inter`. The ref keeps
`w_gu` in the canonical `[all-gate | all-up]` layout; `run` re-interleaves it for the
chosen `b_inter` before calling the kernel.

Design space (base). There is NO shipped tuned table and NO `tune_*.py` /
`get_block_spec_config*.py` candidate generator for `fused_mlp` (unlike gmm / mla /
rpa / quantized_matmul). The authoritative evidence of the maintainer-supported
`(b_seq, b_inter)` set is therefore (a) the kernel signature defaults
(`apply_fused_mlp_sharded(..., b_seq=64, b_inter=128)`) and (b) the one production
caller, `glm5_moe.py::GLM5MoE_MLP`: it selects `b_seq = 64 if seq_len <= 8 else 256`
(glm5_moe.py:569) and `b_inter ∈ {32, 64, 128}` by `local_inter_size`
(glm5_moe.py:492-497). Both knobs are always powers of two there. We widen each base
knob to the honest power-of-2 range that covers that witnessed set plus the finer /
wider tiles the kernel equally supports:
  `b_seq   ∈ {32, 64, 128, 256}`  (sequence tile → grid `seq // b_seq`)
  `b_inter ∈ {32, 64, 128, 256, 512, 1024}`  (intermediate tile → pipeline depth
                                               `inter // b_inter`)
`valid` keeps only truly-runnable configs: `b_seq` divides `seq` (the sharded kernel
grids on `seq // b_seq`), `b_inter` divides `inter` (`num_inter = inter // b_inter`,
and the per-tile gate|up interleave needs `inter % b_inter == 0`), and a VMEM
working-set bound (`_vmem_bytes <= _VMEM_BUDGET`) so a widened tile can't enumerate a
config that would OOM the TPU's VMEM. Defaults stay the kernel's own (64 / 128).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from sgl_jax.srt.kernels.fused_mlp import apply_fused_mlp_sharded

from akt.benchmark.refs.fused_mlp import (
    ATOL,
    NATIVE_TEST,
    RTOL,
    make_inputs,
    reference,
)
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob

# Single-device mesh; the kernel's shard_map shards along axis "tensor" and
# psum-reduces over it (identity for a 1-device mesh).
_MESH = Mesh(jax.devices()[:1], ("tensor",))


def _interleave(w_gu, b_inter: int):
    """[all-gate|all-up] -> per-tile [gate_i|up_i] blocks of width b_inter,
    the layout `apply_fused_mlp_sharded` expects for this b_inter."""
    hidden = w_gu.shape[0]
    inter = w_gu.shape[1] // 2
    nt = inter // b_inter
    wg = w_gu[:, :inter].reshape(hidden, nt, b_inter)
    wu = w_gu[:, inter:].reshape(hidden, nt, b_inter)
    return jnp.concatenate([wg, wu], axis=2).reshape(hidden, 2 * inter)


def _run(inp, cfg):
    b_seq, b_inter = int(cfg["b_seq"]), int(cfg["b_inter"])
    w_gu_il = _interleave(inp["w_gu"], b_inter)
    return apply_fused_mlp_sharded(inp["x"], w_gu_il, inp["wd"], _MESH, b_seq, b_inter)


# TPU VMEM working-set sanity bound (bytes). The kernel keeps resident, per
# `mlp_kernel_main`'s BlockSpecs + scratch: the seq tile as x_tile/y_tile plus the
# f32 y_scratch (3 * b_seq * hidden), the TRIPLE-buffered gate|up weight
# (3 * hidden * 2*b_inter) and TRIPLE-buffered down weight (3 * b_inter * hidden),
# and the f32 matmul-1 output (b_seq * 2*b_inter). All f32 in VMEM (weights upcast,
# accum is preferred_element_type=f32). 32 MiB comfortably admits every valid config
# for the frozen shapes (max ~12 MiB) while pruning tilings that would blow VMEM on
# a wider MLP — it is a size guard, not a perf model (these tpu-deferred configs are
# not timed here).
_VMEM_BUDGET = 32 * 1024 * 1024


def _vmem_bytes(b_seq: int, b_inter: int, hidden: int) -> int:
    f32 = 4
    seq_tiles = 3 * b_seq * hidden * f32            # x_tile + y_tile + y_scratch(f32)
    w_gu = 3 * hidden * (2 * b_inter) * f32         # triple-buffered gate|up
    wd = 3 * b_inter * hidden * f32                 # triple-buffered down
    hu = b_seq * (2 * b_inter) * f32                # matmul-1 output (f32)
    return seq_tiles + w_gu + wd + hu


def _space(seq: int, inter: int, hidden: int) -> DesignSpace:
    return DesignSpace(
        knobs=[
            Knob("b_seq", [32, 64, 128, 256], default=64),
            Knob("b_inter", [32, 64, 128, 256, 512, 1024], default=128),
        ],
        valid=lambda c, seq=seq, inter=inter, hidden=hidden: (
            seq % c["b_seq"] == 0
            and inter % c["b_inter"] == 0
            and _vmem_bytes(c["b_seq"], c["b_inter"], hidden) <= _VMEM_BUDGET),
    )


# (seq, hidden, inter). Small SwiGLU MLPs; both dims chosen so every knob divides.
_SHAPES = [(128, 256, 512), (256, 512, 512)]

CASES = [
    KernelCase(
        kernel_id="fused_mlp", shape_id=f"s{seq}_h{hidden}_i{inter}",
        make_inputs=functools.partial(make_inputs, seq, hidden, inter),
        run=_run, reference=reference, space=_space(seq, inter, hidden),
        atol=ATOL, rtol=RTOL, native_test=NATIVE_TEST,
        regime_pref=("tpu",),
        note="SwiGLU fused MLP; Mosaic-only -> tpu-deferred; ref=pure JAX SwiGLU",
    )
    for (seq, hidden, inter) in _SHAPES
]
