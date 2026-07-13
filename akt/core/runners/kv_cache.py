"""KV-cache update — paged scatter of new K/V into the cache (Pallas DMA kernel).

Tuned entry: `kv_cache_update(new_kv, slices, kv_cache, num_kv_update_slices,
*, page_size, num_slices_per_block, kv_partition_axis)` — a scalar-prefetch
Pallas kernel that DMAs each slice (a contiguous run of ≤page_size tokens) from
new_kv into kv_cache. It has NO interpret path, so it is TPU-ONLY: the kernel
call errors here ("Only interpret mode is supported on CPU backend") → the case
is `tpu-deferred` (wired + reference-validated; latency awaits a TPU).

Reference: a pure-JAX contiguous scatter (`dynamic_update_slice`). We lay the
new tokens out contiguously in the cache (token j → cache[OFFSET+j]); the paged
slice list is then just a tiling of that contiguous copy, so the result is
INVARIANT to both `page_size` and `num_slices_per_block` — which lets the single
cfg-independent reference stay exact across the whole design space.

Design space (base): `num_slices_per_block` (the DMA block tiling — the true
autotuning axis, swept in bench_update_kv_cache.py) and `page_size` (slice
granularity). Both are kernel static args here.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    get_slot_mapping,
    kv_cache_update,
)

from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob

# Single-device (data, tensor) mesh so the entry's shard_map traces here; on a
# real TPU host this would be the model's parallelism mesh.
_MESH = jax.make_mesh((1, 1), ("data", "tensor"))
_OFFSET = 16  # contiguous write base into the cache (leaves a sentinel prefix)


def _inputs(head_num: int, cache_len: int, new_len: int, head_dim: int, seed: int = 42):
    # new_kv/cache mirror bench_update_kv_cache's create_bench_data (3D,
    # bf16, random). head_dim must be 128-aligned; head_num even.
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    new_kv = jax.random.normal(keys[1], (new_len, head_num, head_dim), dtype=jnp.bfloat16)
    cache = jax.random.normal(keys[2], (cache_len, head_num, head_dim), dtype=jnp.bfloat16)
    return {"new_kv": new_kv, "cache": cache, "new_len": new_len}


def _build_slices(new_len: int, page_size: int, num_slices_per_block: int):
    # Contiguous tiling: page p covers new_kv[p*ps : p*ps+len] -> cache[OFFSET+p*ps : ...].
    n_pages = (new_len + page_size - 1) // page_size
    cache_start = jnp.asarray([_OFFSET + p * page_size for p in range(n_pages)], jnp.int32)
    new_start = jnp.asarray([p * page_size for p in range(n_pages)], jnp.int32)
    lens = jnp.asarray([min(page_size, new_len - p * page_size) for p in range(n_pages)], jnp.int32)
    # get_slot_mapping pads to a multiple of num_slices_per_block (zero-len no-ops).
    slices = get_slot_mapping(num_slices_per_block, cache_start, new_start, lens)
    return slices, jnp.asarray([n_pages], jnp.int32)


def _run(inp, cfg):
    ps, nspb = int(cfg["page_size"]), int(cfg["num_slices_per_block"])
    slices, nkus = _build_slices(int(inp["new_len"]), ps, nspb)
    with jax.set_mesh(_MESH):
        return kv_cache_update(
            inp["new_kv"], slices, inp["cache"], nkus,
            page_size=ps, num_slices_per_block=nspb, kv_partition_axis="tensor")


def _ref(inp):
    # cfg-independent ground truth: write new_kv contiguously at OFFSET.
    return jax.lax.dynamic_update_slice(inp["cache"], inp["new_kv"], (_OFFSET, 0, 0))


def _space():
    return DesignSpace(
        knobs=[
            Knob("num_slices_per_block", [2, 4, 8, 16, 32, 64], default=8),
            Knob("page_size", [1, 64, 128], default=64),
        ],
    )


CASES = [
    KernelCase(
        kernel_id="kv_cache", shape_id=f"h{hn}_cache{cl}_new{nl}",
        make_inputs=functools.partial(_inputs, hn, cl, nl, 128),
        run=_run, reference=_ref, space=_space(),
        atol=0.0, rtol=0.0,  # exact bit copy on TPU
        regime_pref=("tpu-deferred",),
        note="tpu-deferred (no interpret path); ref=contiguous JAX scatter",
    )
    for (hn, cl, nl) in [(8, 4096, 256), (16, 8192, 512)]
]
