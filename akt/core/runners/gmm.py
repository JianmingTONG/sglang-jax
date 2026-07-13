"""GMM (grouped matmul) — megablox TPU-Pallas kernel.

Two wired paths over ONE shared tiling design space (`tile_m, tile_k, tile_n`,
gmm_v2's `TileSizes`):

* **interpret-runnable** (`kernel_id="gmm"`) — routes through the backend
  dispatch `megablox_gmm_backend.gmm(...)`, which on a non-TPU box auto-selects
  `interpret=True` and lowers to **gmm v1** (`megablox_gmm_kernel/gmm.py`). That
  runs on CPU here, so these cases probe as `regime=cpu` and are correctness-gated
  against a pure-JAX einsum reference. The tiling tuple `(tm,tk,tn)` is threaded to
  v1 verbatim. (The heavy backend module pulls in the sglang server stack — zmq
  etc.; when that import is unavailable we call v1 directly with `interpret=True`,
  which IS the backend's interpret dispatch, so behaviour is identical.)
* **tpu-deferred** (`kernel_id="gmm_v2"`) — calls `gmm_v2(...)` with the same knobs
  as a `TileSizes(tile_m,tile_k,tile_n)`. gmm_v2 is Mosaic-only (`get_tpu_info()`
  raises on CPU), so it is wired + reference-validated but its definitive latency
  awaits a TPU (probe = `failed` on this box).

Design space (base): `tile_m ∈ {64,128,256}`, `tile_k ∈ {256,512,1024}`,
`tile_n ∈ {256,512,1024}` — gmm_v2's shipped tiling axes. `valid` keeps configs
whose tiles divide the problem dims (clean tiling for both v1 and v2).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import TileSizes, gmm_v2

from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob

# --- interpret path: prefer the real backend dispatch; fall back to v1 -------
try:  # backend drags in the server stack (zmq/psutil/...) — may be absent here
    from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm as _backend_gmm

    def _gmm_interpret(lhs, rhs, group_sizes, tiling):
        # interpret=True forces the backend's non-TPU dispatch (-> gmm v1).
        return _backend_gmm(
            lhs, rhs, group_sizes,
            preferred_element_type=jnp.float32, tiling=tiling, interpret=True)

    _INTERP_VIA = "megablox_gmm_backend.gmm(interpret=True)->v1"
except Exception:  # noqa: BLE001
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm as _gmm_v1

    def _gmm_interpret(lhs, rhs, group_sizes, tiling):
        # This IS the backend's interpret branch: v1 with interpret=True.
        return _gmm_v1(
            lhs, rhs, group_sizes,
            preferred_element_type=jnp.float32, tiling=tiling, interpret=True)

    _INTERP_VIA = "gmm_v1(interpret=True) [backend import unavailable]"


def _group_sizes(m: int, g: int) -> jnp.ndarray:
    """Deterministic, non-uniform group split summing to m (straddles tiles)."""
    rng = np.random.default_rng(1234 + m + g)
    w = rng.uniform(0.5, 1.5, size=g)
    sizes = np.floor(w / w.sum() * m).astype(np.int64)
    sizes[-1] += m - int(sizes.sum())          # fix rounding so sum == m
    sizes = np.maximum(sizes, 0)
    sizes[-1] += m - int(sizes.sum())
    return jnp.asarray(sizes, dtype=jnp.int32)


def _inputs(m: int, k: int, n: int, g: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((m, k)) * 0.1, dtype=jnp.float32)
    rhs = jnp.asarray(rng.standard_normal((g, k, n)) * 0.1, dtype=jnp.float32)
    return {"lhs": lhs, "rhs": rhs, "group_sizes": _group_sizes(m, g)}


def _reference(inp):
    """Pure-JAX grouped matmul (einsum), mirroring gmm_test.reference_gmm's
    unquantized / no-bias path. Eager (data-dependent group slicing)."""
    lhs, rhs, gs = inp["lhs"], inp["rhs"], inp["group_sizes"]
    outs, start = [], 0
    for grp in range(int(gs.shape[0])):
        end = start + int(gs[grp])
        outs.append(jnp.einsum(
            "bd,dh->bh",
            lhs[start:end].astype(jnp.float32),
            rhs[grp].astype(jnp.float32)))
        start = end
    return jnp.concatenate(outs, axis=0).astype(lhs.dtype)


def _tiling(cfg) -> tuple[int, int, int]:
    return (int(cfg["tile_m"]), int(cfg["tile_k"]), int(cfg["tile_n"]))


def _run_interpret(inp, cfg):
    return _gmm_interpret(inp["lhs"], inp["rhs"], inp["group_sizes"], _tiling(cfg))


def _run_v2(inp, cfg):
    tm, tk, tn = _tiling(cfg)
    return gmm_v2(
        inp["lhs"], inp["rhs"], inp["group_sizes"],
        tile_info=TileSizes(tile_m=tm, tile_k=tk, tile_n=tn),
        preferred_element_type=jnp.float32,
        maybe_quantize_lhs=False)


def _space(m: int, k: int, n: int) -> DesignSpace:
    return DesignSpace(
        knobs=[
            Knob("tile_m", [64, 128, 256], default=128),
            Knob("tile_k", [256, 512, 1024], default=512),
            Knob("tile_n", [256, 512, 1024], default=512),
        ],
        # Clean tiling: tiles divide the problem dims (v1 requires tm | m; keeps
        # v2 tiles aligned too). Captures dims via default args.
        valid=lambda c, m=m, k=k, n=n: (
            m % c["tile_m"] == 0 and k % c["tile_k"] == 0 and n % c["tile_n"] == 0),
    )


# Small grouped matmuls. Interpret shapes run here (CPU); the v2 shape is deferred.
_INTERP_SHAPES = [(256, 512, 512, 4), (512, 512, 512, 8)]
_V2_SHAPE = (512, 1024, 1024, 8)

CASES = [
    KernelCase(
        kernel_id="gmm", shape_id=f"m{m}_k{k}_n{n}_g{g}",
        make_inputs=functools.partial(_inputs, m, k, n, g),
        run=_run_interpret, reference=_reference, space=_space(m, k, n),
        atol=2e-2, rtol=2e-2,
        regime_pref=("gpu", "cpu-interpret"),
        note=f"grouped matmul; interpret via {_INTERP_VIA}; ref=einsum (pure JAX)",
    )
    for (m, k, n, g) in _INTERP_SHAPES
] + [
    KernelCase(
        kernel_id="gmm_v2", shape_id="m{}_k{}_n{}_g{}".format(*_V2_SHAPE),
        make_inputs=functools.partial(_inputs, *_V2_SHAPE),
        run=_run_v2, reference=_reference, space=_space(*_V2_SHAPE[:3]),
        atol=2e-2, rtol=2e-2,
        regime_pref=("tpu",),
        note="gmm_v2 with TileSizes(tile_m,tile_k,tile_n); Mosaic-only -> tpu-deferred",
    )
]
