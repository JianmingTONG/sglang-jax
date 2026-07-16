"""akt-board data builder for the sglang-jax kernel autotuning loop.

Joins the loop's bookkeeping (evolve_history.jsonl + evolve_state.json + capability
manifests + the freshest gate eval) into a single board.json the static index.html
renders. Rebuilt by the loop after every submit (write_board), and runnable by hand:
    cd akt/board && python build.py && python -m http.server 8777
"""
from __future__ import annotations

import ast
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
CURRENT_OBJECTIVE_SCOPE = "stateful-serving-deployable-v1"


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
            "research_space_size": r.get("research_space_size"),
            "best_config": r.get("best_config"),
            "correct": r.get("correct"),
            "native_test": nt.get("status"),          # sglang-jax's own pytest verdict
            "native_test_id": r.get("native_test_id"),
            "note": (r.get("search_note") or r.get("reason") or r.get("note") or "")[:160],
        })
    return out, s


def _control_inventory():
    """Read live runner metadata without importing JAX in the board builder."""
    exposed = set()
    runner_only = set()
    for path in (ROOT / "core/runners").glob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "Knob" or not node.args:
                continue
            try:
                knob = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            control_node = keywords.get("programmer_control")
            if control_node is not None:
                try:
                    control = ast.literal_eval(control_node)
                except (ValueError, TypeError):
                    control = None
                if isinstance(control, str) and control:
                    exposed.add(control)
                    continue
            elevated = keywords.get("elevated_by")
            if elevated is not None and not (
                isinstance(elevated, ast.Constant) and elevated.value is None
            ):
                runner_only.add(f"{path.stem}.{knob}")
    return {
        "programmer_controls": sorted(exposed),
        "runner_only_knobs": sorted(runner_only),
    }


# ---- Flexibility graph: the TPU compiler LOWERING GRAPH ----------------------
# Nodes = the 3-layer taxonomy (JAX/Pallas -> Mosaic TPU IR -> TPU hardware) + a hidden
# band, across 5 category columns; edges verified against the jax 0.8.1 Pallas/Mosaic
# source and annotated with which sglang-jax kernels exercise them. Data + assembly live
# in akt/core/analysis/flexgraph_spec.py (jax-free). A lowering edge into a HIDDEN node
# has no exposure back to the top -> the flexibility gap.
def _flexgraph(eval_summary):
    # Prefer the AUTO-EXTRACTED graph: flexgraph_extract.py navigates the live stack
    # (jax/Pallas/Mosaic lowering source + the serving kernels) and writes
    # flexgraph_generated.json. Fall back to the hand-authored spec if not generated.
    gen = ROOT / "core/analysis/flexgraph_generated.json"
    try:
        if gen.exists():
            g = json.loads(gen.read_text())
            if g.get("nodes"):
                return g
    except Exception:  # noqa: BLE001
        pass
    try:
        from akt.core.analysis.flexgraph_spec import build_graph
        return build_graph()
    except Exception as e:  # noqa: BLE001
        return {"kind": "unavailable", "error": repr(e)[:200], "nodes": [],
                "lowering_edges": [], "exposure_edges": [], "gap_edges": [],
                "n_gap": 0, "n_hidden": 0, "note": ""}


def _clean(o):
    """Coerce NaN/Inf floats to None throughout a nested structure. board.json/
    flexgraph.json are dumped with allow_nan=False (browsers reject NaN/Inf JSON), so a
    single non-finite metric (e.g. an all-deferred geomean, or a failed-eval latency)
    would otherwise raise ValueError and leave the board un-rebuilt."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _valid_latency(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _apply_incumbent_envelope(trail, state):
    """Annotate rounds with the correctness-qualified best-so-far latency.

    The envelope starts at the campaign's original searched baseline. Candidate
    measurements remain visible, but only a correctness-qualified KEEP/baseline
    may lower the incumbent, and noisy measurements can never make it rise.
    """
    ordered = sorted(trail, key=lambda row: row.get("round") or 0)
    objective_scope = state.get("objective_scope")
    baseline_candidates = [
        state.get("objective_baseline_geomean"),
        state.get("base_search_geomean"),
        *(row.get("incumbent_geomean") for row in ordered),
        state.get("incumbent_geomean"),
    ]
    baseline = next((value for value in baseline_candidates if _valid_latency(value)), None)
    best = baseline

    for row in ordered:
        performance_status = row.get("performance_status") or ""
        objective_compatible = (
            objective_scope is None or row.get("objective_scope") == objective_scope
        )
        historical_timing_is_valid = (
            performance_status != "post-run-invalid"
            and not performance_status.startswith("superseded")
        )
        eligible = (
            row.get("decision") in ("keep", "baseline")
            and row.get("correct") is True
            and historical_timing_is_valid
            and objective_compatible
        )
        candidate = row.get("geomean_s")
        previous = best
        if eligible and _valid_latency(candidate):
            best = candidate if best is None else min(best, candidate)
        row["incumbent_eligible"] = eligible
        row["objective_compatible"] = objective_compatible
        row["advances_incumbent"] = best is not None and (previous is None or best < previous)
        row["incumbent_after_s"] = best

    return baseline, best


def build():
    st = _load_json(STATE, {})
    manifests = _manifests()
    hist = _load_jsonl(HIST)
    kernels, eval_summary = _kernels_from_eval()

    trail = []
    for r in hist:
        mm = manifests.get(r.get("capability"), {})
        accuracy = r.get("accuracy") or {}
        best_configs = [
            {"case": case, "config": evidence.get("best_config")}
            for case, evidence in accuracy.items()
            if evidence.get("best_config") is not None
        ]
        trail.append({
            "round": r.get("round"), "capability": r.get("capability"),
            "gap": r.get("gap") or mm.get("gap"),
            "decision": r.get("decision"),
            "delta_pct": r.get("delta_pct"), "target_pct": r.get("target_pct"),
            "estimated_relief_pct": (r.get("estimated_relief_pct")
                                     if r.get("estimated_relief_pct") is not None
                                     else mm.get("estimated_relief_pct")),
            "geomean_s": r.get("geomean_s"), "incumbent_geomean": r.get("incumbent_geomean"),
            "reason": r.get("reason"),
            "hypothesis": r.get("hypothesis") or mm.get("hypothesis"),
            "search_dimension": r.get("search_dimension") or mm.get("search_dimension"),
            "files_touched": r.get("files_touched") or mm.get("files_touched"),
            "correct": r.get("correct"),
            "performance_status": r.get("performance_status"),
            "correctness_regime": r.get("correctness_regime"),
            "n_verified": r.get("n_verified"),
            "n_deferred": r.get("n_deferred"),
            "missing_cases": r.get("missing_cases") or [],
            "truncated_cases": r.get("truncated_cases") or [],
            "best_configs": best_configs,
            "programmer_exposure": r.get("programmer_exposure"),
            "objective_scope": r.get("objective_scope"),
            "timestamp": r.get("timestamp"),
        })
    original_geomean, replayed_geomean = _apply_incumbent_envelope(trail, st)
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
            "original_geomean_s": original_geomean,
            "shipped_default_geomean_s": st.get("shipped_default_geomean"),
            "base_search_geomean_s": st.get("base_search_geomean"),
            "current_geomean_s": st.get("incumbent_geomean"),
            "replayed_geomean_s": replayed_geomean,
            "choices": st.get("incumbent_choices"),
            "round": st.get("round"),
            "target_pct": round((st.get("target_improvement") or 0.02) * 100, 1),
            "objective_scope": st.get("objective_scope"),
            "objective_baseline_geomean_s": st.get("objective_baseline_geomean"),
            "objective_baseline_round": st.get("objective_baseline_round"),
            "commit": (st.get("incumbent_commit") or "")[:8],
        },
        "eval": {"all_correct": eval_summary.get("all_correct"),
                 "objective_scope": eval_summary.get("objective_scope"),
                 "allclose_correct": eval_summary.get("allclose_correct"),
                 "native_ok": eval_summary.get("native_ok"),
                 "native_summary": eval_summary.get("native_summary"),
                 "n_runnable": eval_summary.get("n_runnable"),
                 "n_deferred": eval_summary.get("n_deferred"),
                 "n_choices": eval_summary.get("n_choices"),
                 "search_geomean_s": eval_summary.get("geomean_s"),
                 "default_geomean_s": eval_summary.get("geomean_default_s")},
        "objective_scope": st.get("objective_scope"),
        "required_objective_scope": CURRENT_OBJECTIVE_SCOPE,
        "objective_stale": st.get("objective_scope") != CURRENT_OBJECTIVE_SCOPE,
        "kernels": kernels,
        "controls": _control_inventory(),
        "frontier": FRONTIER,
        "trail": trail, "kept": kept, "rejected": rejected,
        "n_rounds": len(trail),
    }
    (BOARD / "board.json").write_text(json.dumps(_clean(board), indent=1, allow_nan=False))
    fg = _flexgraph(eval_summary)
    (BOARD / "flexgraph.json").write_text(json.dumps(_clean(fg), indent=1, allow_nan=False))
    n_run = sum(1 for k in kernels if k.get("best_s"))
    print(f"akt-board: {len(kernels)} kernels ({n_run} runnable, {len(kernels)-n_run} deferred), "
          f"{len(trail)} round(s) ({len(kept)} kept, {len(rejected)} rejected), "
          f"{len(FRONTIER)} frontier gaps; flexgraph {len(fg.get('nodes',[]))} nodes / "
          f"{len(fg.get('lowering_edges',[]))} lowering / {fg.get('n_gap',0)} gap edges "
          f"({fg.get('n_hidden',0)} hidden). -> board.json + flexgraph.json")


if __name__ == "__main__":
    build()
