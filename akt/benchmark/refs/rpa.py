"""FROZEN authoritative correctness reference for RPA v3 (Ragged Paged Attention).

Sources (all sglang-jax's OWN code — imported, not reimplemented):
  * Reference impl:  sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3
        ``ref_ragged_paged_attention`` — the pure jnp.einsum ground truth the repo's
        FlashAttention tests validate the Mosaic kernel against.
  * Tolerance:       python/sgl_jax/test/flashattention_common.py:488-489
        ``rtol = 2e-2`` / ``atol = 1e-2`` (AttentionTestBase.run_test's np.allclose).
  * Canonical inputs: this module's ``make_inputs`` builds a small DECODE workload
        (q_len==1/seq) in BOTH the reference view (separate k/v pages + 2D page table)
        and the kernel view (fused interleaved KV cache + flat page table), matching the
        layout ``merge_kv`` / ``create_kv_cache_data`` produce and the way
        flashattention_common.py::AttentionTestBase feeds ``ref_ragged_paged_attention``
        (q reshaped to [tokens, num_q_heads, head_dim]; k/v as [pages, page_size, kv, hd];
        per-seq page table; ``num_seqs`` as i32[1]).

NATIVE_TEST: None. RPA v3 is a TPU-only Mosaic/Pallas kernel with NO interpret path;
the repo tests that exercise it (python/sgl_jax/test/test_flashattention_{mha,gqa,dp,misc}.py
via AttentionTestBase in flashattention_common.py) drive the FlashAttention backend, which
lowers ``ragged_paged_attention`` to Mosaic and cannot run off-TPU. So there is no repo
pytest that passes in interpret here — the reference below is validated (builds + shapes),
and the kernel's definitive check is TPU-deferred.

This dir is FROZEN: the EDITABLE runner (akt/core/runners/rpa.py) imports
``reference``/``ATOL``/``RTOL``/``make_inputs``/``NATIVE_TEST``/``check_out`` from here and
owns only the DesignSpace + config->d_block_sizes mapping + CASES assembly.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ref_ragged_paged_attention,
)

# EXACT sglang-jax test tolerance — flashattention_common.py:488-489
# (AttentionTestBase.run_test: rtol = 2e-2 / atol = 1e-2 in np.allclose).
RTOL = 2e-2
ATOL = 1e-2

# get_dtype_packing(float32) = 32 // 32 = 1 (fused-KV cache packing factor for f32).
_KV_PACKING_F32 = 1

# TPU-only kernel — no interpret path, so no repo pytest runs here (see module docstring).
NATIVE_TEST = None  # reason: tpu-only kernel, native test TPU-deferred


def make_inputs(
    num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    kv_lens_list: tuple[int, ...],
    seed: int = 0,
) -> dict:
    """Build a SMALL decode workload (q_len == 1 per seq). Produces both the
    reference view (separate k/v pages + 2D page table) and the kernel view
    (fused interleaved KV cache + flat page table)."""
    rng = np.random.default_rng(seed)
    scale = 0.1
    kv_lens_np = np.asarray(kv_lens_list, dtype=np.int32)
    assert kv_lens_np.shape == (num_seqs,)

    pages_per_seq = int(np.max((kv_lens_np + page_size - 1) // page_size))
    total_num_pages = num_seqs * pages_per_seq

    # 2D page table: seq i owns a contiguous unique page block, padded with 0.
    page_indices_2d = np.zeros((num_seqs, pages_per_seq), dtype=np.int32)
    for i in range(num_seqs):
        npg = int((kv_lens_np[i] + page_size - 1) // page_size)
        page_indices_2d[i, :npg] = np.arange(
            i * pages_per_seq, i * pages_per_seq + npg, dtype=np.int32
        )

    # 1 query token per sequence (decode).
    q = (rng.standard_normal((num_seqs, num_q_heads, head_dim)) * scale).astype(np.float32)
    k_pages = (
        rng.standard_normal((total_num_pages, page_size, num_kv_heads, head_dim)) * scale
    ).astype(np.float32)
    v_pages = (
        rng.standard_normal((total_num_pages, page_size, num_kv_heads, head_dim)) * scale
    ).astype(np.float32)

    # New-token k/v the kernel writes into the cache == the last cache slot of
    # each seq, so the write is a no-op and the fused cache stays self-consistent
    # with what the reference attends over.
    keys = np.zeros((num_seqs, num_kv_heads, head_dim), dtype=np.float32)
    values = np.zeros((num_seqs, num_kv_heads, head_dim), dtype=np.float32)
    for i in range(num_seqs):
        last = int(kv_lens_np[i] - 1)
        pg = int(page_indices_2d[i, last // page_size])
        sl = last % page_size
        keys[i] = k_pages[pg, sl]
        values[i] = v_pages[pg, sl]

    cu_q_lens = np.concatenate([[0], np.cumsum(np.ones(num_seqs, dtype=np.int32))]).astype(np.int32)
    cu_kv_lens = np.concatenate([[0], np.cumsum(kv_lens_np)]).astype(np.int32)
    distribution = np.array([num_seqs, num_seqs, num_seqs], dtype=np.int32)  # all-decode

    # Fused KV cache: interleave [K0,V0,K1,V1,...] then split by packing (=1 for
    # float32) — exactly the layout merge_kv / create_kv_cache_data produce.
    kv_pages = np.concatenate([k_pages, v_pages], axis=-1).reshape(
        total_num_pages, page_size, num_kv_heads * 2, head_dim
    )
    kv_cache_fused = kv_pages.reshape(
        total_num_pages, page_size, num_kv_heads * 2 // _KV_PACKING_F32, _KV_PACKING_F32, head_dim
    )

    return {
        # reference view
        "queries": jnp.asarray(q),
        "k_pages": jnp.asarray(k_pages),
        "v_pages": jnp.asarray(v_pages),
        "kv_lens": jnp.asarray(kv_lens_np),
        "page_indices_2d": jnp.asarray(page_indices_2d),
        "cu_q_lens": jnp.asarray(cu_q_lens),
        "num_seqs": jnp.asarray([num_seqs], dtype=jnp.int32),
        # kernel view
        "keys": jnp.asarray(keys),
        "values": jnp.asarray(values),
        "kv_cache_fused": jnp.asarray(kv_cache_fused),
        "page_indices_flat": jnp.asarray(page_indices_2d.reshape(-1)),
        "cu_kv_lens": jnp.asarray(cu_kv_lens),
        "distribution": jnp.asarray(distribution),
        # statics
        "sm_scale": float(head_dim ** -0.5),
        "page_size": int(page_size),
    }


def reference(inp: dict):
    """Authoritative ground truth: sglang-jax's own ``ref_ragged_paged_attention``.

    Eager (NOT jitted): the reference has data-dependent Python control flow over
    num_seqs / kv_lens, matching how flashattention_common.py calls it."""
    return ref_ragged_paged_attention(
        inp["queries"],
        inp["k_pages"],
        inp["v_pages"],
        inp["kv_lens"],
        inp["page_indices_2d"],
        inp["cu_q_lens"],
        inp["num_seqs"],
        causal=True,
        sm_scale=inp["sm_scale"],
    )


def check_out(o):
    """The kernel returns (out, updated_cache); the reference returns out. Compare out."""
    return o[0] if isinstance(o, tuple) else o
