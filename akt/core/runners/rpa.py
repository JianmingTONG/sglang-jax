"""RPA v3 (Ragged Paged Attention) — TPU-Pallas decode kernel (EDITABLE runner).

Tuned entry: `ragged_paged_attention(...)` (Pallas/Mosaic). We focus on the
DECODE path, whose tiling is the static 4-tuple `d_block_sizes =
(bq_sz, bkv_sz, bq_csz, bkv_csz)`.

The authoritative correctness contract (reference, tolerance, canonical inputs,
NATIVE_TEST) lives in the FROZEN akt/benchmark/refs/rpa.py; this runner imports it
and owns ONLY the DesignSpace + config->d_block_sizes mapping + the CASES assembly.

Design space (base): the decode KV block `bkv_sz`. Per the shipped sweep
(`benchmark/kernels/flash_attention/get_block_spec_config_v3.py`) the decode
`bq_sz` is forced to 1 and the sub-tiles are PINNED (`bq_csz==bq_sz`,
`bkv_csz==bkv_sz`); the run mapping ties them here but the config dict keeps all
four keys so a later CAPABILITY can UNPIN `bq_csz`/`bkv_csz` as new knobs.

REGIME: RPA v3 is a TPU-only Mosaic kernel with NO interpret path, so on this
(GPU/CPU) box it is **tpu-deferred** — we fully WIRE it (build inputs, map
config -> d_block_sizes, provide the pure-JAX reference) and validate the
reference + shapes; the Pallas `run` call itself errors here (expected).
"""
from __future__ import annotations

import functools

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import align_to, next_power_of_2

from akt.benchmark.refs.rpa import (
    ATOL,
    NATIVE_TEST,
    RTOL,
    check_out,
    make_inputs,
    reference,
)
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob

# Shipped decode bkv candidate set (aligned to page/kv_packing), from
# get_block_spec_config_v3.py::_bkv_candidates (minus the 4096 upper probe).
_BKV_CANDIDATES = [256, 512, 1024, 2048]
_KV_PACKING_F32 = 1  # get_dtype_packing(float32) = 32 // 32


def _heuristic_decode_bkv(max_kv: int, head_dim: int, num_kv_heads: int, page_size: int) -> int:
    """Mirror get_default_block_sizes' DECODE branch (bkv = min(peak, max_kv),
    page-aligned), then clamp to the nearest shipped candidate <= it. We can't
    call get_default_block_sizes directly here: it needs a TPU (get_tpu_version
    raises off-TPU)."""
    num_kv_heads_x2 = next_power_of_2(align_to(num_kv_heads * 2, _KV_PACKING_F32))
    hd = align_to(head_dim, 128)
    min_bkv_to_peak = 16 * 1024 * 1024 * _KV_PACKING_F32 // 4 // hd // num_kv_heads_x2
    heur = align_to(min(min_bkv_to_peak, max_kv), max(page_size, _KV_PACKING_F32))
    le = [c for c in _BKV_CANDIDATES if c <= heur]
    return max(le) if le else min(_BKV_CANDIDATES)


def _block_sizes(cfg: dict) -> tuple[int, int, int, int]:
    """Map a config to the decode 4-tuple. bq_sz pinned to 1; sub-tiles tied to
    their parents unless a capability elevated them into the config."""
    bq_sz = int(cfg.get("bq_sz", 1))
    bkv_sz = int(cfg["bkv_sz"])
    bq_csz = int(cfg.get("bq_csz", bq_sz))
    bkv_csz = int(cfg.get("bkv_csz", bkv_sz))
    return (bq_sz, bkv_sz, bq_csz, bkv_csz)


def _run(inp: dict, cfg: dict):
    d_block_sizes = _block_sizes(cfg)
    out, _updated_cache = ragged_paged_attention(
        inp["queries"],
        inp["keys"],
        inp["values"],
        inp["kv_cache_fused"],
        inp["kv_lens"],
        inp["page_indices_flat"],
        inp["cu_q_lens"],
        inp["cu_kv_lens"],
        inp["distribution"],
        None,  # custom_mask
        causal=1,
        sm_scale=inp["sm_scale"],
        d_block_sizes=d_block_sizes,
    )
    return out


def _space(page_size: int, default_bkv: int) -> DesignSpace:
    return DesignSpace(
        knobs=[Knob("bkv_sz", list(_BKV_CANDIDATES), default=default_bkv)],
        # kernel requires bkv_sz (and bkv_csz==bkv_sz) divisible by page_size.
        valid=lambda c: c["bkv_sz"] % page_size == 0,
    )


# Small decode shapes: (num_seqs, num_q_heads, num_kv_heads, head_dim, page_size, kv_lens)
_SHAPES = [
    (2, 4, 2, 128, 256, (512, 384)),   # GQA 4:2, ctx up to 512
    (3, 8, 8, 128, 256, (1024, 768, 512)),  # MHA 8:8, ctx up to 1024
]


def _make_case(spec) -> KernelCase:
    num_seqs, num_q_heads, num_kv_heads, head_dim, page_size, kv_lens = spec
    max_kv = max(kv_lens)
    default_bkv = _heuristic_decode_bkv(max_kv, head_dim, num_kv_heads, page_size)
    shape_id = f"d_s{num_seqs}_q{num_q_heads}kv{num_kv_heads}_hd{head_dim}_p{page_size}_ctx{max_kv}"
    return KernelCase(
        kernel_id="rpa_v3",
        shape_id=shape_id,
        make_inputs=functools.partial(
            make_inputs, num_seqs, num_q_heads, num_kv_heads, head_dim, page_size, kv_lens
        ),
        run=_run,
        reference=reference,
        space=_space(page_size, default_bkv),
        atol=ATOL,
        rtol=RTOL,
        check_out=check_out,
        native_test=NATIVE_TEST,
        regime_pref=("tpu-deferred",),
        note="RPA v3 decode; tpu-deferred (Pallas TPU-only, no interpret path); "
        "ref=ref_ragged_paged_attention (pure jnp.einsum). Knob=bkv_sz "
        "(bq_sz pinned 1, sub-tiles tied — later capability unpins bq_csz/bkv_csz).",
    )


CASES = [_make_case(s) for s in _SHAPES]
