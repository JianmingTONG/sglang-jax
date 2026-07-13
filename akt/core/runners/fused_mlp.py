"""Fused MLP (SwiGLU) — TPU-Pallas kernel `apply_fused_mlp_sharded`.

Fuses gate/up projections + SiLU gating + down projection into one pipelined TPU
kernel (`srt/kernels/fused_mlp.py`). It is Mosaic-only (no interpret path), so it
is **tpu-deferred** on this box: fully wired (inputs, config->tiling map, pure-JAX
reference, design space) with its definitive latency awaiting a TPU. The pure-JAX
reference below runs here and is validated; the Pallas run raises on CPU/GPU.

Weight layout. The kernel's fused weight `w_gu` is column-INTERLEAVED per
intermediate tile: block `i` is `[gate_i (b_inter) | up_i (b_inter)]`
(see `mlp_kernel_main`'s BlockSpec, which loads `w_gu[:, i*2*b_inter:(i+1)*2*b_inter]`
and splits it at `b_inter`). So the interleaving DEPENDS on `b_inter`. To keep the
reference a config-independent ground truth, `make_inputs` stores `w_gu` in the
canonical `[all-gate | all-up]` layout; the reference splits it at the midpoint,
and `run` re-interleaves it for the chosen `b_inter` before calling the kernel.
(Verified in numpy: the tiled kernel math over the interleaved weight reproduces
the reference bit-exactly for every `b_inter`.)

Design space (base): `b_seq ∈ {32,64,128}` (sequence tile, kernel grid), and
`b_inter ∈ {128,256,512}` (intermediate tile, pipeline depth) — the kernel's two
shipped block sizes. `valid` keeps configs where the tiles divide `seq`/`inter`.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.kernels.fused_mlp import apply_fused_mlp_sharded

from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob

# Single-device mesh; the kernel's shard_map shards along axis "tensor" and
# psum-reduces over it (identity for a 1-device mesh).
_MESH = Mesh(jax.devices()[:1], ("tensor",))


def _inputs(seq: int, hidden: int, inter: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.05, dtype=jnp.float32)
    x = mk(seq, hidden)
    w_gate = mk(hidden, inter)
    w_up = mk(hidden, inter)
    w_down = mk(inter, hidden)
    # Canonical [all-gate | all-up] layout (config-independent ground truth).
    w_gu = jnp.concatenate([w_gate, w_up], axis=1)   # [hidden, 2*inter]
    return {"x": x, "w_gu": w_gu, "wd": w_down}


def _reference(inp):
    """out = (silu(x @ W_gate) * (x @ W_up)) @ W_down.  W_gate|W_up = split(w_gu)."""
    x, w_gu, wd = inp["x"], inp["w_gu"], inp["wd"]
    inter = w_gu.shape[1] // 2
    w_gate = w_gu[:, :inter]
    w_up = w_gu[:, inter:]
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ wd


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
        make_inputs=functools.partial(_inputs, seq, hidden, inter),
        run=_run, reference=_reference, space=_space(seq, inter),
        atol=2e-2, rtol=2e-2,
        regime_pref=("tpu",),
        note="SwiGLU fused MLP; Mosaic-only -> tpu-deferred; ref=pure JAX SwiGLU",
    )
    for (seq, hidden, inter) in _SHAPES
]
