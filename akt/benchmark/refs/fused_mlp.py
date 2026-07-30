"""FROZEN authoritative correctness reference for the `fused_mlp` (SwiGLU) kernel.

Sources pulled from sglang-jax:
  - Kernel under test: python/sgl_jax/srt/kernels/fused_mlp.py
      `apply_fused_mlp_sharded` / `mlp_kernel_main` / `inner_mlp_kernel` — the
      optimized Gated-MLP (SwiGLU) Pallas kernel that fuses the gate/up
      projections, the SiLU gating, and the down projection into one pipeline.

NO NATIVE TEST. sglang-jax has NO native correctness test for `fused_mlp`:
nothing under python/sgl_jax/test/ references `fused_mlp` / `apply_fused_mlp_sharded`
(only the *fused_moe* v1/v2 kernels ship tests). There is therefore no repo
reference to import, so `reference` below is NOT a re-implementation of some other
oracle — it IS the mathematical spec the kernel realizes: SwiGLU

    out = ( silu(x @ W_gate) * (x @ W_up) ) @ W_down

which is exactly what `inner_mlp_kernel` computes (fused_mlp.py:36-46: one matmul
`x @ w_gu` -> split the columns at `b_inter` into h|u -> `silu(h) * u` -> second
matmul `a @ wd`, accumulated across intermediate tiles in float32). Because the
kernel is Mosaic-only (no `interpret` path) it cannot run off-TPU, and there is no
maintainer-authored pytest to wire, so NATIVE_TEST is None.

Weight layout. The kernel's fused weight `w_gu` is column-INTERLEAVED per
intermediate tile: block `i` is `[gate_i (b_inter) | up_i (b_inter)]` (see
`mlp_kernel_main`'s w_gu BlockSpec at fused_mlp.py:94-98 loading
`w_gu[:, i*2*b_inter:(i+1)*2*b_inter]` and the split at `b_inter` in
`inner_mlp_kernel`). So the interleaving DEPENDS on `b_inter`. To keep this
reference a config-independent ground truth, `make_inputs` stores `w_gu` in the
canonical `[all-gate | all-up]` layout; the reference splits it at the midpoint and
the EDITABLE runner re-interleaves it per chosen `b_inter` before calling the
kernel. (Verified in numpy that the tiled kernel math over the interleaved weight
reproduces this reference bit-exactly for every `b_inter`.)

Tolerance: ATOL = RTOL = 2e-2. There is no sglang-jax `fused_mlp` test to cite a
tolerance from. The nearest maintainer-authored SwiGLU oracle is the fused_moe v2
kernel test, whose float32 (non-fp8) SwiGLU path asserts atol=rtol=5e-2
(python/sgl_jax/test/kernels/fused_moe_v2_test.py:296-297; its fp8 paths loosen
further to 2e-1/3e-1 at v1_test.py:172, v2_test.py:151). This kernel's chain is
pure float32 (the kernel accumulates matmuls at `preferred_element_type=jnp.float32`,
fused_mlp.py:36,46) with no quantization, so the tighter 2e-2/2e-2 bound is
comfortably inside — and stricter than — that comparable fp32 SwiGLU reference.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# EXACT tolerance for the correctness check. See the module docstring: no native
# fused_mlp test exists; 2e-2/2e-2 is a documented fp32-SwiGLU tolerance, stricter
# than the comparable fused_moe fp32 SwiGLU path (fused_moe_v2_test.py:296-297,
# atol=rtol=5e-2).
ATOL = 2e-2
RTOL = 2e-2

# No repo pytest verifies this kernel off-TPU (Mosaic-only, no interpret path; and
# no maintainer-authored fused_mlp test exists at all). See module docstring.
NATIVE_TEST = None

# NOT declared bit-exact across configs, deliberately: the module docstring's
# "reproduces this reference bit-exactly for every b_inter" was verified in NUMPY,
# but on the MXU a K-wide matmul split into b_inter-sized blocks accumulates partial
# sums in a different order — float addition is non-associative, so cross-config
# bit-identity is NOT structurally guaranteed on real hardware. The family allclose
# tolerance above remains the correctness contract; flip this only with TPU evidence.
BITEXACT_INVARIANT = False


def make_inputs(seq: int, hidden: int, inter: int, seed: int = 0) -> dict:
    """Canonical SwiGLU-MLP inputs. `w_gu` is stored in the config-independent
    [all-gate | all-up] layout; the runner re-interleaves per `b_inter`."""
    rng = np.random.default_rng(seed)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.05, dtype=jnp.float32)
    x = mk(seq, hidden)
    w_gate = mk(hidden, inter)
    w_up = mk(hidden, inter)
    w_down = mk(inter, hidden)
    w_gu = jnp.concatenate([w_gate, w_up], axis=1)  # [hidden, 2*inter]
    return {"x": x, "w_gu": w_gu, "wd": w_down}


def reference(inp):
    """SwiGLU: out = (silu(x @ W_gate) * (x @ W_up)) @ W_down.

    W_gate | W_up = split(w_gu) at the midpoint. This IS the mathematical spec of
    `inner_mlp_kernel` (fused_mlp.py:36-46) — sglang-jax has no native fused_mlp
    correctness test, so this SwiGLU spec is the authoritative reference.
    """
    x, w_gu, wd = inp["x"], inp["w_gu"], inp["wd"]
    inter = w_gu.shape[1] // 2
    w_gate = w_gu[:, :inter]
    w_up = w_gu[:, inter:]
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ wd
