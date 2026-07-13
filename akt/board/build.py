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


def _flexgraph(eval_summary):
    """Lowering graph vs exposure graph over a SHARED node set (the AI-domain analog
    of the FHE two-graph flexgap instrument). Middle nodes = flexibilities. A knob the
    Mosaic/Pallas lowering CAN execute (top->bottom edge present in the LOWERING graph)
    but the config API does NOT name (bottom->top edge absent in the EXPOSURE graph) is
    a FLEXIBILITY GAP — the exact target a capability elevates."""
    try:
        from akt.benchmark.adapter import FRONTIER
    except Exception:  # noqa: BLE001
        FRONTIER = []
    exposed = {}                       # kernel -> {knob_name: elevated_by}
    for r in eval_summary.get("results", []):
        for kb in r.get("knobs", []):
            exposed.setdefault(r.get("kernel"), {})[kb["name"]] = kb.get("elevated_by")
    exposed_nodes = [{"id": f"exp:{ker}", "kernel": ker, "kind": "exposed", "label": ker,
                      "knobs": sorted(kn),
                      "elevated": sorted({v for v in kn.values() if v})}
                     for ker, kn in sorted(exposed.items()) if ker]
    gap_nodes = [{"id": f"gap:{i}", "kind": "gap", "status": f.get("status"),
                  "kernel": f.get("interface", "").split(":")[0].strip(),
                  "label": f.get("interface", ""), "what": f.get("what", "")}
                 for i, f in enumerate(FRONTIER)]
    API, LOW = "api", "lowering"
    mids = exposed_nodes + gap_nodes
    knob_ids = [n["id"] for n in mids]
    exp_ids = [n["id"] for n in exposed_nodes]
    nodes = ([{"id": API, "kind": "pole", "layer": "top",
               "label": "Kernel config API — what the schema names"}]
             + [{**n, "layer": "mid"} for n in mids]
             + [{"id": LOW, "kind": "pole", "layer": "bottom",
                 "label": "Pallas / Mosaic lowering — what the backend executes"}])
    return {
        "nodes": nodes,
        # lowering graph: top->knob->bottom for EVERY knob (all are lowering-executable)
        "lowering_edges": [[API, k] for k in knob_ids] + [[k, LOW] for k in knob_ids],
        # exposure graph: bottom->knob->top for EXPOSED knobs only
        "exposure_edges": [[LOW, k] for k in exp_ids] + [[k, API] for k in exp_ids],
        "gaps": [n["id"] for n in gap_nodes],
        "n_exposed": len(exposed_nodes), "n_gap": len(gap_nodes),
        "note": ("Same nodes, two edge sets. A middle node with a LOWERING (top->bottom) "
                 "edge but NO EXPOSURE (bottom->top) edge is a FLEXIBILITY GAP: the "
                 "Mosaic lowering can execute it, but no config knob names it — the "
                 "exact target a capability elevates."),
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
