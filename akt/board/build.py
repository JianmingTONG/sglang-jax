"""akt-board data builder for the sglang-jax kernel autotuning loop.

Joins the loop's bookkeeping (evolve_history.jsonl + evolve_state.json + capability
manifests + the freshest gate eval) into a single board.json the static index.html
renders. Rebuilt by the loop after every submit (write_board), and runnable by hand:
    cd akt/board && python build.py && python -m http.server 8777
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

BOARD = Path(__file__).resolve().parent
ROOT = BOARD.parent                      # akt/
REPO = ROOT.parent                       # sglang-jax/
for p in (str(REPO), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

HIST = ROOT / "optimization_history/evolve_history.jsonl"
STATE = ROOT / "optimization_history/evolve_state.json"
EVAL = ROOT / "optimization_history/.evolve_eval.json"
STATUS = BOARD / "evolve_status.json"
CAPS = ROOT / "core/evolve/capabilities"


def _load_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def _load_json(p, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return default if default is not None else {}


def _manifests():
    m = {}
    if CAPS.is_dir():
        for p in CAPS.glob("*.json"):
            d = _load_json(p)
            if d.get("name"):
                m[d["name"]] = d
    return m


def _kernels_from_eval():
    """Per-kernel latency card from the freshest gate eval."""
    s = _load_json(EVAL, {})
    out = []
    for r in s.get("results", []):
        d, b = r.get("default_s"), r.get("forward_s")
        nt = r.get("native_test") or {}
        out.append({
            "case": r.get("case"), "kernel": r.get("kernel"), "shape": r.get("shape"),
            "regime": r.get("regime", "tpu-deferred"),
            "default_s": d, "best_s": b,
            "speedup": (d / b) if (d and b) else None,
            "space_size": r.get("space_size"),
            "best_config": r.get("best_config"),
            "correct": r.get("correct"),
            "native_test": nt.get("status"),          # sglang-jax's own pytest verdict
            "native_test_id": r.get("native_test_id"),
            "note": (r.get("search_note") or r.get("reason") or r.get("note") or "")[:160],
        })
    return out, s


# ---- Single 4-level flexibility graph (curated, jax-free) --------------------
# LOWERING = top->bottom: a workload category lowers into operators -> JAX/Pallas API
# capabilities -> hardware flexibilities (the stack CAN execute this). EXPOSURE =
# bottom->top: the low-level flexibility is named / selectable back up the stack. An
# edge present in LOWERING but ABSENT in EXPOSURE (top->bottom but not bottom->up) is a
# FLEXIBILITY GAP: the lowering executes it, but no API/config names it. Gap edges are
# the frontier the loop elevates (kept in sync with adapter.FRONTIER).
_FLEX_LEVELS = [
    {"id": "L1", "title": "Workload categories", "nodes": [
        {"id": "cat.attn", "label": "Attention (ragged/paged)"},
        {"id": "cat.moe", "label": "MoE / grouped GEMM"},
        {"id": "cat.lin", "label": "Linear attention"},
        {"id": "cat.mlp", "label": "Fused MLP (SwiGLU)"},
        {"id": "cat.kv", "label": "KV-cache update"}]},
    {"id": "L2", "title": "Operators  (compute | memory-reorg)", "nodes": [
        {"id": "op.mm", "label": "Matrix multiply", "kind": "compute"},
        {"id": "op.vec", "label": "Vectorized arithmetic", "kind": "compute"},
        {"id": "op.scan", "label": "Reduction / scan", "kind": "compute"},
        {"id": "op.layout", "label": "Layout reorganization", "kind": "memory"},
        {"id": "op.gather", "label": "Gather / scatter", "kind": "memory"},
        {"id": "op.mask", "label": "Mask / pad", "kind": "memory"}]},
    {"id": "L3", "title": "JAX / Pallas API capabilities", "nodes": [
        {"id": "api.dot", "label": "lax.dot_general"},
        {"id": "api.scan", "label": "lax.scan / assoc"},
        {"id": "api.block", "label": "pl.BlockSpec tiling"},
        {"id": "api.pipe", "label": "pltpu.emit_pipeline"},
        {"id": "api.grid", "label": "PrefetchScalarGridSpec"},
        {"id": "api.mem", "label": "HBM / VMEM spaces"},
        {"id": "api.shard", "label": "shard_map / mesh"}]},
    {"id": "L4", "title": "Hardware-level flexibility", "nodes": [
        {"id": "hw.mxu", "label": "MXU 128x128 tiling"},
        {"id": "hw.vpu", "label": "VPU lanes (8x128)"},
        {"id": "hw.vmem", "label": "VMEM alloc / buffer depth"},
        {"id": "hw.dma", "label": "HBM<->VMEM DMA / prefetch"},
        {"id": "hw.core", "label": "Multi-core / grid parallel"},
        {"id": "hw.sublane", "label": "Sublane / lane sub-tiling"}]},
]
_FLEX_LOWERING = [
    # L1 -> L2 (a category decomposes into operators)
    ("cat.attn", "op.mm"), ("cat.attn", "op.vec"), ("cat.attn", "op.gather"), ("cat.attn", "op.mask"),
    ("cat.moe", "op.mm"), ("cat.moe", "op.gather"), ("cat.moe", "op.layout"),
    ("cat.lin", "op.mm"), ("cat.lin", "op.scan"), ("cat.lin", "op.vec"),
    ("cat.mlp", "op.mm"), ("cat.mlp", "op.vec"), ("cat.mlp", "op.layout"),
    ("cat.kv", "op.gather"), ("cat.kv", "op.layout"),
    # L2 -> L3 (an operator maps to JAX/Pallas API capabilities)
    ("op.mm", "api.dot"), ("op.mm", "api.block"), ("op.mm", "api.shard"),
    ("op.vec", "api.block"),
    ("op.scan", "api.scan"),
    ("op.layout", "api.block"), ("op.layout", "api.pipe"),
    ("op.gather", "api.grid"), ("op.gather", "api.mem"), ("op.gather", "api.shard"),
    ("op.mask", "api.block"),
    # L3 -> L4 (an API capability exercises hardware flexibility)
    ("api.dot", "hw.mxu"),
    ("api.block", "hw.mxu"), ("api.block", "hw.sublane"),
    ("api.scan", "hw.vpu"),
    ("api.pipe", "hw.vmem"), ("api.pipe", "hw.dma"),
    ("api.grid", "hw.dma"), ("api.grid", "hw.core"),
    ("api.mem", "hw.vmem"),
    ("api.shard", "hw.core"),
]
# lowering-reachable but NOT exposed -> the flexibility gaps (cite the live frontier).
_FLEX_GAPS = {
    ("api.pipe", "hw.vmem"): "fused_mlp: emit_pipeline buffer_count hardcoded=3 — VMEM double-buffer depth executes but no knob names it",
    ("api.grid", "hw.dma"): "moe_v2: cross_expert_prefetch / interleave toggles — DMA prefetch schedule executes but is unsearched",
    ("api.block", "hw.sublane"): "rpa bq_csz/bkv_csz pinned + gla/kda BK/BV pinned — sublane sub-tiling executes but is not exposed",
    ("api.grid", "hw.core"): "kv_cache: num_slices_per_block tuned table is dead code — grid tile executes but is not selected",
    ("api.block", "hw.mxu"): "gmm_v2: no per-shape tuned tile table (v1 has one) — MXU tile selection executes but is unnamed",
}


def _flexgraph(eval_summary):
    gapset = set(_FLEX_GAPS)
    exposure = [[b, a] for (a, b) in _FLEX_LOWERING if (a, b) not in gapset]
    nodes = [{**n, "level": lv["id"]} for lv in _FLEX_LEVELS for n in lv["nodes"]]
    return {
        "levels": [{"id": lv["id"], "title": lv["title"],
                    "node_ids": [n["id"] for n in lv["nodes"]]} for lv in _FLEX_LEVELS],
        "nodes": nodes,
        "lowering_edges": [list(e) for e in _FLEX_LOWERING],
        "exposure_edges": exposure,
        "gap_edges": [{"from": a, "to": b, "why": why} for (a, b), why in _FLEX_GAPS.items()],
        "n_exposed": len(_FLEX_LOWERING) - len(_FLEX_GAPS), "n_gap": len(_FLEX_GAPS),
        "note": ("Single graph. An edge present in BOTH lowering (top→bottom, the stack "
                 "can execute it) and exposure (bottom→top, the API names it) is drawn "
                 "GREEN with arrows omitted (bidirectional). An edge present in lowering "
                 "but NOT exposure is drawn RED with a directed down-arrow — a FLEXIBILITY "
                 "GAP: executable in the lowering, unnamed at the top; the frontier the "
                 "loop elevates."),
    }


def build():
    st = _load_json(STATE, {})
    manifests = _manifests()
    hist = _load_jsonl(HIST)
    kernels, eval_summary = _kernels_from_eval()

    trail = []
    for r in hist:
        mm = manifests.get(r.get("capability"), {})
        trail.append({
            "round": r.get("round"), "capability": r.get("capability"),
            "gap": r.get("gap") or mm.get("gap"),
            "decision": r.get("decision"),
            "delta_pct": r.get("delta_pct"), "target_pct": r.get("target_pct"),
            "estimated_relief_pct": (r.get("estimated_relief_pct")
                                     if r.get("estimated_relief_pct") is not None
                                     else mm.get("estimated_relief_pct")),
            "geomean_s": r.get("geomean_s"), "incumbent_geomean": r.get("incumbent_geomean"),
            "reason": r.get("reason"), "hypothesis": mm.get("hypothesis"),
            "search_dimension": mm.get("search_dimension"),
            "files_touched": r.get("files_touched") or mm.get("files_touched"),
        })
    kept = [t for t in trail if t["decision"] in ("keep", "baseline")]
    rejected = [t for t in trail if t["decision"] not in ("keep", "baseline")]

    try:
        from akt.benchmark.adapter import FRONTIER
    except Exception:  # noqa: BLE001
        FRONTIER = []

    board = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": "geomean kernel latency (interpret proxy on this box; TPU when attached)",
        "incumbent": {
            "shipped_default_geomean_s": st.get("shipped_default_geomean"),
            "base_search_geomean_s": st.get("base_search_geomean"),
            "current_geomean_s": st.get("incumbent_geomean"),
            "choices": st.get("incumbent_choices"),
            "round": st.get("round"),
            "target_pct": round((st.get("target_improvement") or 0.02) * 100, 1),
            "commit": (st.get("incumbent_commit") or "")[:8],
        },
        "eval": {"all_correct": eval_summary.get("all_correct"),
                 "allclose_correct": eval_summary.get("allclose_correct"),
                 "native_ok": eval_summary.get("native_ok"),
                 "native_summary": eval_summary.get("native_summary"),
                 "n_runnable": eval_summary.get("n_runnable"),
                 "n_deferred": eval_summary.get("n_deferred"),
                 "search_geomean_s": eval_summary.get("geomean_s"),
                 "default_geomean_s": eval_summary.get("geomean_default_s")},
        "kernels": kernels,
        "frontier": FRONTIER,
        "trail": trail, "kept": kept, "rejected": rejected,
        "n_rounds": len(trail),
    }
    (BOARD / "board.json").write_text(json.dumps(board, indent=1, allow_nan=False))
    fg = _flexgraph(eval_summary)
    (BOARD / "flexgraph.json").write_text(json.dumps(fg, indent=1, allow_nan=False))
    n_run = sum(1 for k in kernels if k.get("best_s"))
    print(f"akt-board: {len(kernels)} kernels ({n_run} runnable, {len(kernels)-n_run} deferred), "
          f"{len(trail)} round(s) ({len(kept)} kept, {len(rejected)} rejected), "
          f"{len(FRONTIER)} frontier gaps; flexgraph {fg['n_exposed']} exposed + "
          f"{fg['n_gap']} gap nodes. -> board.json + flexgraph.json")


if __name__ == "__main__":
    build()
