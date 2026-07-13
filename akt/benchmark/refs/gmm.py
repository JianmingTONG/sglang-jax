"""FROZEN correctness contract for the `gmm` (grouped matmul / megablox) kernel.

Authoritative sglang-jax sources (imported / cited):
  * REFERENCE — `reference_gmm`, imported verbatim from the maintainers' own kernel
    test `python/sgl_jax/test/kernels/gmm_test.py:96` (the pure-JAX grouped-matmul
    the repo's `test_gmm` validates the Pallas gmm v1/v2 kernels against). We drive
    only its unquantized / no-bias / group_offset=0 path (rhs_scale=rhs_bias=None,
    group_offset default [0]) — the exact configuration our interpret runner
    executes. This REPLACES the previously reimplemented einsum in the runner; the
    two are bit-identical on this path (validated: 0.00e+00 maxabs), but importing
    the repo's function makes the maintainers' reference the single source of truth.
  * TOLERANCE — the unquantized `test_gmm` asserts `self.assertArraysAllClose(
    actual, expected)` with NO explicit tolerance (gmm_test.py:206), i.e. jtu's
    per-dtype default. That test runs bfloat16 operands, whose jtu default bar is
    atol=rtol=1e-2 (`jax._src.public_test_util._default_tolerance['bfloat16']==0.01`).
    Our interpret runner uses float32 operands (preferred_element_type=float32), so
    its numeric error vs `reference_gmm` is ~6e-7 maxabs. We adopt atol=rtol=2e-2 —
    within the maintainers' bf16 bar and clearing our float32-interpret error by
    ~5 orders of magnitude. (The quantized tests use looser explicit tols
    3e-1/1.1 — not relevant to this unquantized path.)
  * INPUT GENERATOR — `make_inputs` mirrors `test_gmm`'s tensor structure: lhs
    [m,k], rhs [g,k,n], and a non-uniform `group_sizes` summing to m (the repo's
    `get_group_sizes`, gmm_test.py:64-68, draws a normalized random split). We keep
    the runner's original float32 operands + a deterministic numpy split (seeded,
    reproducible, straddling tile boundaries) so the interpret path stays at ~6e-7
    and probe_regime is stable; the reference consumed is the repo's, unchanged.
  * NATIVE_TEST — see the NATIVE_TEST comment below. The numeric `test_gmm` cannot
    run off-TPU (the repo's `gmm_wrapper` hardcodes the kernel call WITHOUT
    `interpret=True`, so CPU lowering raises "Only interpret mode is supported on
    CPU backend."). The only GmmTest test that passes in Pallas interpret here is
    the block-size regression `test_default_block_m_divides_m`, which we wire.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Authoritative reference: the maintainers' own pure-JAX grouped matmul used by
# python/sgl_jax/test/kernels/gmm_test.py::GmmTest::test_gmm (gmm_test.py:96).
from sgl_jax.test.kernels.gmm_test import reference_gmm

# --- tolerance: repo unquantized `test_gmm` uses assertArraysAllClose defaults ---
# (gmm_test.py:206). bf16 jtu default = 1e-2; our float32-interpret error ~6e-7.
# 2e-2/2e-2 sits within the maintainers' bar and clears our error by ~5 orders.
ATOL = 2e-2
RTOL = 2e-2

# --- native repo test (validated: 18 passed in interpret here) -------------------
# The numeric `test_gmm` cannot run off-TPU: `gmm_wrapper` (gmm_test.py:32) calls
# gmm() without interpret=True, so on CPU Pallas raises "Only interpret mode is
# supported on CPU backend." The block-size regression `test_default_block_m_divides_m`
# (gmm_test.py:131, guards the gmm default-tile selection) is pure-Python and DOES
# pass in interpret — it is the only GmmTest test that passes off-TPU, so it is the
# repo's own gmm check wired here. (Primary numeric correctness is the per-config
# allclose vs `reference_gmm` above.)
NATIVE_TEST = "python/sgl_jax/test/kernels/gmm_test.py -k default_block_m_divides_m"


def _group_sizes(m: int, g: int) -> jnp.ndarray:
    """Deterministic, non-uniform group split summing to m (straddles tiles).

    Same intent as the repo test's `get_group_sizes` (gmm_test.py:64-68 — a
    normalized random split summing to batch_size), made deterministic via numpy
    so the interpret gate is reproducible."""
    rng = np.random.default_rng(1234 + m + g)
    w = rng.uniform(0.5, 1.5, size=g)
    sizes = np.floor(w / w.sum() * m).astype(np.int64)
    sizes[-1] += m - int(sizes.sum())          # fix rounding so sum == m
    sizes = np.maximum(sizes, 0)
    sizes[-1] += m - int(sizes.sum())
    return jnp.asarray(sizes, dtype=jnp.int32)


def make_inputs(m: int, k: int, n: int, g: int, seed: int = 0) -> dict:
    """Canonical grouped-matmul inputs: lhs [m,k], rhs [g,k,n], group_sizes [g].

    Tensor structure mirrors `test_gmm` (gmm_test.py:171-184); float32 operands
    (matching the runner's preferred_element_type=float32) keep the interpret error
    at ~6e-7. Consumed by both the interpret v1 path and the tpu-deferred v2 path."""
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((m, k)) * 0.1, dtype=jnp.float32)
    rhs = jnp.asarray(rng.standard_normal((g, k, n)) * 0.1, dtype=jnp.float32)
    return {"lhs": lhs, "rhs": rhs, "group_sizes": _group_sizes(m, g)}


def reference(inp: dict):
    """Ground-truth grouped matmul via the repo's authoritative `reference_gmm`
    (unquantized / no-bias / group_offset=0 path)."""
    return reference_gmm(inp["lhs"], inp["rhs"], inp["group_sizes"])
