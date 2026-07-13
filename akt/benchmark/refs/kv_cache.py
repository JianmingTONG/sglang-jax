"""FROZEN authoritative correctness reference for the `kv_cache` update kernel.

Every piece here is sourced from sglang-jax's OWN kernel/test code:

- Kernel under test:
    ``python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py`` — the
    scalar-prefetch Pallas DMA kernel ``kv_cache_update_kernel`` (driven by the
    ``kv_cache_update`` shard_map wrapper the runner tunes, and by the
    ``kv_cache_update_impl`` static-grid wrapper the repo test drives ON TPU). Both
    wrappers copy each slice ``(kv_cache_start, new_kv_start, slice_len)`` from
    ``new_kv`` into ``kv_cache`` — i.e. token j is scattered to ``cache[loc[j]]``.

- Reference construction (the correctness contract) — FAITHFULLY MIRRORED from
    ``python/sgl_jax/test/mem_cache/test_kv_cache.py::TestKVCache.expected_update_kv_cache``
    (lines 109-122): build the expected cache by a per-token scatter
    ``expected = expected.at[loc[i]].set(new_kv[i])``, skipping padding rows
    (``loc[i] == -1``). This REPLACES the old ``dynamic_update_slice`` contiguous
    shortcut with the repo's own loc-based scatter — the authoritative notion of
    "update kv cache". (It is mirrored, not imported: the test method is bound to a
    ``unittest.TestCase`` and operates on separate 3D k/v against a zeros cache in
    the 5D-fused path; here there is a single fused 3D ``new_kv`` scattered into the
    existing ``cache``, since ``kv_cache_update`` aliases and preserves untouched
    rows — ``input_output_aliases`` in update_kv_cache.py:189/279.)

- Canonical inputs — the loc map is the test's own contiguous form
    ``loc = jnp.arange(total_tokens) + base`` (test_kv_cache.py:96, base=10 there;
    ``_OFFSET`` here). Because loc is contiguous, the per-token reference scatter and
    the kernel's page-sized *slice* copy produce the IDENTICAL result for every
    ``page_size`` / ``num_slices_per_block`` — so this single reference is an exact,
    cfg-independent ground truth across the whole design space. k/v are drawn with
    ``jax.random.uniform(..., bf16)`` as in the test (test_kv_cache.py:73-82); the
    cache is drawn random (not zeros) so the reference also pins the untouched-region
    preservation the aliasing kernel guarantees.

- Tolerance:
    ``jnp.allclose`` defaults ``atol=1e-8, rtol=1e-5`` — exactly what
    ``test_kv_cache.py`` asserts the updated cache against ``expected`` with
    (``_run_and_verify``, test_kv_cache.py:140-147, plain ``jnp.allclose``; the
    per-segment checks additionally pin ``rtol=1e-5``, e.g. line 236). The DMA copy
    is bit-exact, so this tolerance is comfortably met.

NATIVE_TEST = None — the kernel is genuinely TPU-only. ``kv_cache_update`` /
``kv_cache_update_impl`` hardcode ``pl.pallas_call`` with no ``interpret`` flag and
never thread ``get_interpret()``, so off-TPU they error "Only interpret mode is
supported on CPU backend" (confirmed by running the runner's ``_run`` on CPU). The
repo test ``test_kv_cache.py`` DOES pass under ``PALLAS_INTERPRET=1 JAX_PLATFORMS=cpu``
(7 passed) — but ONLY because its sibling conftest
(``python/sgl_jax/test/mem_cache/conftest.py``:50-51) MONKEYPATCHES
``update_fused_kv_cache_vectorized`` with a pure-JAX CPU scatter shim
(``_cpu_fused_kv_scatter``) whenever ``jax.default_backend() != "tpu"``. So off-TPU
the test verifies a CPU STAND-IN, not the Pallas kernel this case tunes. Wiring it
would report green while exercising a shim — violating base.py's native_test
contract ("exercises the actual Pallas kernel"). Hence None (base.py's "TPU-only"
clause); the kernel's real off-TPU verification is that same shim's identical loc
scatter, which is exactly the ``reference`` below.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Tolerance — jnp.allclose defaults, matching test_kv_cache.py:140-147 (the repo
# asserts the updated cache against `expected` with plain jnp.allclose; per-segment
# checks pin rtol=1e-5, e.g. test_kv_cache.py:236).
ATOL = 1e-8
RTOL = 1e-5

# TPU-only kernel: no interpret path. test_kv_cache.py only "passes" off-TPU because
# its conftest (mem_cache/conftest.py:50-51) monkeypatches the entry with a pure-JAX
# CPU shim — it does NOT exercise the Pallas kernel this case tunes. See docstring.
NATIVE_TEST = None

# Contiguous write base into the cache (leaves a sentinel prefix untouched). Mirrors
# the test's `loc = arange(total_tokens) + base` contiguous mapping (test uses 10).
_OFFSET = 16


def make_inputs(head_num: int, cache_len: int, new_len: int, head_dim: int, seed: int = 42):
    """Canonical inputs for the 3D `kv_cache_update` scatter.

    Mirrors test_kv_cache.py's data generation: k/v via `jax.random.uniform(..., bf16)`
    (test lines 73-82) and a contiguous loc map `arange(new_len) + _OFFSET`
    (test line 96). `head_dim` must be 128-aligned; `head_num` even (kernel asserts).
    Returns new_kv/cache (3D bf16), the scalar `new_len`, and `loc` (the per-token
    cache-target map the reference scatters along, `-1` = padding).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    new_kv = jax.random.uniform(keys[1], (new_len, head_num, head_dim), dtype=jnp.bfloat16)
    cache = jax.random.uniform(keys[2], (cache_len, head_num, head_dim), dtype=jnp.bfloat16)
    loc = jnp.arange(new_len, dtype=jnp.int32) + _OFFSET
    return {"new_kv": new_kv, "cache": cache, "new_len": new_len, "loc": loc}


def reference(inp):
    """Authoritative ground truth: the repo's own per-token loc scatter.

    Faithful mirror of test_kv_cache.py::expected_update_kv_cache (lines 109-122):
    for each token i, if `loc[i] != -1` scatter `new_kv[i]` into `out[loc[i]]`.
    Starts from the existing `cache` (the aliasing kernel preserves untouched rows)
    rather than the test's zeros buffer; identical scatter otherwise.
    """
    out = inp["cache"]
    new_kv = inp["new_kv"]
    loc = inp["loc"]
    for i in range(loc.shape[0]):
        dst = int(loc[i])
        if dst != -1:  # skip padding rows, as expected_update_kv_cache does
            out = out.at[dst].set(new_kv[i])
    return out
