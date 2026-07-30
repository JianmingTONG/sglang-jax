"""KV-cache update — paged scatter of new K/V into the cache (Pallas DMA kernel).

Tuned entry: `kv_cache_update(new_kv, slices, kv_cache, num_kv_update_slices,
*, page_size, num_slices_per_block, kv_partition_axis)` — a scalar-prefetch
Pallas kernel that DMAs each slice (a contiguous run of ≤page_size tokens) from
new_kv into kv_cache. It has NO interpret path (pallas_call is hardcoded without
`interpret`), so it is TPU-ONLY: off-TPU the shard_map wrapper errors here (GPU:
"dynamic grid bounds not supported in the Triton backend"; CPU: "Only interpret
mode is supported on CPU backend") → the case is `tpu-deferred` (wired +
reference-validated; its definitive latency awaits a TPU). The repo's own
test_kv_cache.py only runs off-TPU via a conftest CPU shim (not the kernel), so
NATIVE_TEST is None — see akt.benchmark.refs.kv_cache.

The authoritative correctness contract (reference / tolerances / canonical inputs /
NATIVE_TEST) lives in the FROZEN `akt.benchmark.refs.kv_cache`; this EDITABLE runner
owns only the DesignSpace + the config->kernel `run` mapping + CASES.

Design space (base): `num_slices_per_block` (the DMA block tiling — the true
autotuning axis, swept in bench_update_kv_cache.py) and `page_size` (slice
granularity). Both are kernel static args here. Because the input `loc` map is
contiguous, the kernel result is INVARIANT to both knobs — the single reference
stays exact across the whole design space.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    VMEM_SIZE,
    get_slot_mapping,
    kv_cache_update,
)

from akt.benchmark.refs.kv_cache import (
    ATOL,
    BITEXACT_INVARIANT,
    NATIVE_TEST,
    RTOL,
    make_inputs,
    reference,
)
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob

# Single-device (data, tensor) mesh so the entry's shard_map traces here; on a
# real TPU host this would be the model's parallelism mesh.
_MESH = jax.make_mesh((1, 1), ("data", "tensor"))


def _build_slices(loc, new_len: int, page_size: int, num_slices_per_block: int):
    # Page p covers new_kv[p*ps : p*ps+len] -> cache[loc[p*ps] : ...]. loc is the
    # (contiguous) per-token cache-target map from make_inputs, so cache_start is
    # read straight off it — keeping the kernel scatter aligned with reference().
    n_pages = (new_len + page_size - 1) // page_size
    cache_start = jnp.asarray([int(loc[p * page_size]) for p in range(n_pages)], jnp.int32)
    new_start = jnp.asarray([p * page_size for p in range(n_pages)], jnp.int32)
    lens = jnp.asarray([min(page_size, new_len - p * page_size) for p in range(n_pages)], jnp.int32)
    # get_slot_mapping pads to a multiple of num_slices_per_block (zero-len no-ops).
    slices = get_slot_mapping(num_slices_per_block, cache_start, new_start, lens)
    return slices, jnp.asarray([n_pages], jnp.int32)


def _run(inp, cfg):
    ps, nspb = int(cfg["page_size"]), int(cfg["num_slices_per_block"])
    slices, nkus = _build_slices(inp["loc"], int(inp["new_len"]), ps, nspb)
    with jax.set_mesh(_MESH):
        return kv_cache_update(
            inp["new_kv"], slices, inp["cache"], nkus,
            page_size=ps, num_slices_per_block=nspb, kv_partition_axis="tensor")


def _space(head_num: int, head_dim: int):
    """Base design space, widened to the SHIPPED autotuning set.

    `num_slices_per_block` is the true tuned axis: the maintainers' bench
    (benchmark/kernels/update_kv_cache/bench_update_kv_cache.py::full_benchmark)
    sweeps the full power-of-2 ladder [2 .. 4096] and their tuned table
    (update_kv_cache/tuned_block_sizes.py::best_num_slices_per_block_config)
    contains best values up to 4096 — so those are the real supported candidates
    (previously the runner only enumerated up to 64). `page_size` stays the
    kernel's supported slice granularities {1, 64, 128}.

    VMEM guard mirrors the kernel's own cap (get_num_slices_per_block): the
    per-block scratch VMEM((nspb, page_size, head_num, head_dim), bf16) must fit
    VMEM_SIZE. The bench enforces this by `min(get_num_slices_per_block(...), nspb)`;
    here we prune any config whose scratch would exceed VMEM (which would OOM /
    fail the pallas_call vmem_limit_bytes), so only truly-supported configs are
    enumerated. bf16 => 2 bytes/element.
    """
    bytes_per_elt = 2  # bf16 (make_inputs draws new_kv/cache as bfloat16)

    def _fits_vmem(cfg: dict) -> bool:
        nspb, ps = int(cfg["num_slices_per_block"]), int(cfg["page_size"])
        # kernel: max_num_slices_per_block = VMEM // (bytes * page_size * heads * head_dim)
        max_nspb = VMEM_SIZE // (bytes_per_elt * ps * head_num * head_dim)
        return 1 <= nspb <= max_nspb

    return DesignSpace(
        knobs=[
            Knob("num_slices_per_block",
                 [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], default=8),
            Knob("page_size", [1, 64, 128], default=64),
        ],
        valid=_fits_vmem,
    )


CASES = [
    KernelCase(
        kernel_id="kv_cache", shape_id=f"h{hn}_cache{cl}_new{nl}",
        make_inputs=functools.partial(make_inputs, hn, cl, nl, 128),
        run=_run, reference=reference, space=_space(hn, 128),
        atol=ATOL, rtol=RTOL, native_test=NATIVE_TEST,
        bitexact_invariant=BITEXACT_INVARIANT,
        regime_pref=("tpu-deferred",),
        note="tpu-deferred (kernel TPU-only, no interpret path); "
             "ref=repo loc-scatter (expected_update_kv_cache); NATIVE_TEST=None (conftest shim)",
    )
    for (hn, cl, nl) in [(8, 4096, 256), (16, 8192, 512)]
]
