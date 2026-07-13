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

Design space (base): `b_seq ∈ {32,64,128}` (sequence tile, kernel grid), and
`b_inter ∈ {128,256,512}` (intermediate tile, pipeline depth) — the kernel's two
shipped block sizes. `valid` keeps configs where the tiles divide `seq`/`inter`.
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


def _space(seq: int, inter: int) -> DesignSpace:
    return DesignSpace(
        knobs=[
            Knob("b_seq", [32, 64, 128], default=64),
            Knob("b_inter", [128, 256, 512], default=128),
        ],
        valid=lambda c, seq=seq, inter=inter: (
            seq % c["b_seq"] == 0 and inter % c["b_inter"] == 0),
    )


# (seq, hidden, inter). Small SwiGLU MLPs; both dims chosen so every knob divides.
_SHAPES = [(128, 256, 512), (256, 512, 512)]

CASES = [
    KernelCase(
        kernel_id="fused_mlp", shape_id=f"s{seq}_h{hidden}_i{inter}",
        make_inputs=functools.partial(make_inputs, seq, hidden, inter),
        run=_run, reference=reference, space=_space(seq, inter),
        atol=ATOL, rtol=RTOL, native_test=NATIVE_TEST,
        regime_pref=("tpu",),
        note="SwiGLU fused MLP; Mosaic-only -> tpu-deferred; ref=pure JAX SwiGLU",
    )
    for (seq, hidden, inter) in _SHAPES
]
