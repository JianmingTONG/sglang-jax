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
CURRENT_OBJECTIVE_SCOPE = "model-serving-empirical-dp-v2"


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
    # The current target-hardware gate records exhaustive per-case measurement
    # tables. Keep the legacy ``results`` adapter below so old campaign artifacts
    # remain readable.
    if s.get("case_search"):
        backend = (s.get("target_hardware") or {}).get("backend", "target")
        for case_id, result in s["case_search"].items():
            measurements = result.get("measurements") or []
            best = min(
                measurements,
                key=lambda row: row.get("latency_s", math.inf),
                default=None,
            )
            defaults = {
                knob.get("name"): knob.get("default")
                for knob in result.get("knobs") or []
            }
            default = next(
                (row for row in measurements if row.get("config") == defaults),
                None,
            )
            out.append({
                "case": case_id,
                "kernel": case_id.split(":", 1)[0],
                "regime": backend,
                "default_s": (default or {}).get("latency_s"),
                "best_s": (best or {}).get("latency_s"),
                "speedup": (
                    default["latency_s"] / best["latency_s"]
                    if default and best and best.get("latency_s")
                    else None
                ),
                "space_size": result.get("valid_configs"),
                "research_space_size": result.get("space_size"),
                "best_config": (best or {}).get("config"),
                "correct": result.get("all_configs_correct"),
                "native_test": None,
                "note": (
                    f"exhaustive {result.get('correct_configs', 0)}/"
                    f"{result.get('valid_configs', 0)} configs"
                ),
            })
        return out, s
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


# ---- Flexibility graph: compiler context + canonical AKT actions -------------
def _flexgraph(eval_summary, state=None):
    try:
        graph = _load_json(ROOT / "core/analysis/flexgraph_generated.json")
        required = ("nodes", "lowering_edges", "hidden_lowering_edges", "action_edges")
        if not all(isinstance(graph.get(key), list) for key in required):
            raise ValueError("generated flexibility graph is missing v2 edge tables")
        if not graph["nodes"]:
            raise ValueError("generated flexibility graph has no nodes")
        from akt.core.evolve.action_catalog import (
            action_catalog_context,
            load_action_graph,
        )

        validated, _catalog = load_action_graph(REPO)
        if graph != validated:
            raise ValueError("displayed graph differs from the canonical action graph")
        fingerprint = action_catalog_context(REPO)["fingerprint"]
        expected = (state or {}).get("action_graph_fingerprint") or (
            eval_summary or {}
        ).get("action_graph_fingerprint")
        status = (
            "unbound"
            if not expected
            else "current"
            if expected == fingerprint
            else "stale"
        )
        return {
            **graph,
            "action_graph_fingerprint": fingerprint,
            "campaign_action_graph_fingerprint": expected,
            "action_context_status": status,
            "actions_executable_for_campaign": status == "current",
        }
    except Exception as e:  # noqa: BLE001
        return {"kind": "unavailable", "error": repr(e)[:200], "nodes": [],
                "lowering_edges": [], "exposure_edges": [],
                "hidden_lowering_edges": [], "action_edges": [],
                "n_open_action_edges": 0, "n_hidden": 0,
                "note": f"Generated action graph unavailable: {e}"}


def _compact_empirical_status(record):
    certificate = (record or {}).get("empirical_dp_certificate") or {}
    if not certificate:
        return {"status": "not_run", "ok": None, "models": 0, "passed_models": 0}
    models = certificate.get("models") or []
    return {
        "status": certificate.get("status") or (
            "validated" if certificate.get("ok") is True else "inconclusive"
        ),
        "ok": certificate.get("ok"),
        "models": len(models),
        "passed_models": sum(model.get("ok") is True for model in models),
    }


def _compact_live_status(record):
    record = record or {}
    gate = record.get("live_model_gate") or {}
    document = record.get("live_model_evaluation") or {}
    candidates = document.get("candidates") or []
    eligible = gate.get("eligible_candidates")
    if eligible is None:
        eligible = sum(
            (candidate.get("eligibility") or {}).get("eligible") is True
            for candidate in candidates
        )
    skipped = gate.get("topology_skipped_candidates")
    if skipped is None:
        skipped = sum(
            candidate.get("status") == "skipped"
            and candidate.get("applicability") == "direct-capability"
            and (candidate.get("eligibility") or {}).get("eligible") is not True
            for candidate in candidates
        )
    not_applicable = gate.get("not_applicable_candidates")
    if not_applicable is None:
        not_applicable = sum(
            candidate.get("applicability") == "not_applicable"
            for candidate in candidates
        )
    if not gate and not document:
        status = "not_run"
        ok = None
    elif not_applicable and gate.get("ok") is not False:
        status = "not_applicable"
        ok = gate.get("ok", True)
    elif skipped and not eligible and gate.get("ok") is not False:
        status = "topology_skipped"
        ok = gate.get("ok", True)
    else:
        status = document.get("status") or (
            "passed" if gate.get("ok") is True else "failed"
        )
        ok = gate.get("ok", document.get("all_passed"))
    return {
        "status": status,
        "ok": ok,
        "eligible": eligible,
        "passed": gate.get("passed_candidates", 0),
        "not_applicable": not_applicable,
        "topology_skipped": skipped,
    }


def _compact_models(record):
    models = []
    for model in (record or {}).get("models") or []:
        certificate = model.get("certificate") or {}
        models.append({
            "model": model.get("model"),
            "candidate_s": model.get("candidate_s"),
            "incumbent_s": model.get("incumbent_s"),
            "ratio": model.get("ratio"),
            "correct": model.get("selected_correct"),
            "dp_ok": (
                certificate.get("certified_additive_optimum") is True
                and certificate.get("bounded_dp_matches_bruteforce") is True
            ),
            "empirical": _compact_empirical_status(model),
        })
    return models


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


def _trail_record(record, manifest):
    """Normalize current and historical round schemas into one compact board row."""

    accuracy = record.get("accuracy") or {}
    best_configs = []
    for label, evidence in accuracy.items():
        if not isinstance(evidence, dict):
            continue
        if evidence.get("best_config") is not None:
            best_configs.append({"case": label, "config": evidence["best_config"]})
        selected_plan = evidence.get("selected_plan")
        if isinstance(selected_plan, dict):
            best_configs.extend(
                {"case": callsite, "config": config}
                for callsite, config in selected_plan.items()
                if isinstance(config, dict)
            )

    estimate = record.get("estimate")
    if not isinstance(estimate, dict):
        estimate = manifest.get("estimate") or {}
    dimensions = record.get("search_dimensions")
    if not isinstance(dimensions, list):
        dimensions = manifest.get("search_dimensions") or []
    dimension_summary = record.get("search_dimension") or manifest.get(
        "search_dimension"
    )
    if not dimension_summary and dimensions:
        dimension_summary = "; ".join(
            f"{dimension.get('control')}={dimension.get('candidate_values')}"
            for dimension in dimensions
            if isinstance(dimension, dict)
        )

    action_snapshot = record.get("action_snapshot") or {}
    contract = record.get("capability_contract") or {}
    gap_id = record.get("gap_id") or manifest.get("gap_id")
    delta = record.get("delta_pct")
    if delta is None:
        delta = record.get("improvement_pct")
    estimated_relief = record.get("estimated_relief_pct")
    if estimated_relief is None:
        estimated_relief = estimate.get("expected_relief_pct")

    return {
        "round": record.get("round"),
        "capability": record.get("capability"),
        "gap": record.get("gap") or manifest.get("gap") or gap_id,
        "gap_id": gap_id,
        "source_evidence": contract.get("source_evidence")
        or action_snapshot.get("evidence"),
        "decision": record.get("decision"),
        "delta_pct": delta,
        "target_pct": record.get("target_pct"),
        "estimated_relief_pct": estimated_relief,
        "geomean_s": record.get("geomean_s"),
        "incumbent_geomean": record.get("incumbent_geomean"),
        "reason": record.get("reason"),
        "hypothesis": record.get("hypothesis") or manifest.get("hypothesis"),
        "search_dimension": dimension_summary,
        "search_dimensions": dimensions,
        "files_touched": record.get("files_touched") or manifest.get("files_touched"),
        "correct": record.get("correct"),
        "performance_status": record.get("performance_status"),
        "correctness_regime": record.get("correctness_regime"),
        "n_verified": record.get("n_verified"),
        "n_deferred": record.get("n_deferred"),
        "missing_cases": record.get("missing_cases") or [],
        "truncated_cases": record.get("truncated_cases") or [],
        "best_configs": best_configs,
        "programmer_exposure": record.get("programmer_exposure"),
        "empirical_dp": _compact_empirical_status(record),
        "live_kimi": _compact_live_status(record),
        "objective_scope": record.get("objective_scope"),
        "timestamp": record.get("timestamp"),
    }


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
        trail.append(_trail_record(r, mm))
    original_geomean, replayed_geomean = _apply_incumbent_envelope(trail, st)
    kept = [t for t in trail if t["decision"] in ("keep", "baseline")]
    rejected = [t for t in trail if t["decision"] not in ("keep", "baseline")]

    try:
        from akt.benchmark.adapter import FRONTIER
    except Exception:  # noqa: BLE001
        FRONTIER = []

    board = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": "paired latency across three frozen serving traces on target TPU",
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
                 "models": _compact_models(eval_summary),
                 "empirical_dp": _compact_empirical_status(eval_summary),
                 "live_kimi": _compact_live_status(eval_summary),
                 "n_runnable": (len(eval_summary.get("case_search") or {})
                                if eval_summary.get("case_search") is not None
                                else eval_summary.get("n_runnable")),
                 "n_deferred": eval_summary.get("n_deferred"),
                 "n_choices": eval_summary.get("n_choices"),
                 "search_geomean_s": (eval_summary.get("candidate_geomean_s")
                                      or eval_summary.get("geomean_s")),
                 "default_geomean_s": (eval_summary.get("incumbent_geomean_s")
                                       or eval_summary.get("geomean_default_s")),
                 "paired_ratio_geomean": eval_summary.get("paired_ratio_geomean")},
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
    fg = _flexgraph(eval_summary, st)
    (BOARD / "flexgraph.json").write_text(json.dumps(_clean(fg), indent=1, allow_nan=False))
    n_run = sum(1 for k in kernels if k.get("best_s"))
    print(f"akt-board: {len(kernels)} kernels ({n_run} runnable, {len(kernels)-n_run} deferred), "
          f"{len(trail)} round(s) ({len(kept)} kept, {len(rejected)} rejected), "
          f"{len(FRONTIER)} frontier gaps; flexgraph {len(fg.get('nodes',[]))} nodes / "
          f"{len(fg.get('lowering_edges',[]))} lowering / "
          f"{len(fg.get('hidden_lowering_edges',[]))} hidden / "
          f"{len(fg.get('action_edges',[]))} actions. -> board.json + flexgraph.json")


if __name__ == "__main__":
    build()
