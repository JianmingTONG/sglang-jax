"""Automated compliance checker for the AKT rule registry (akt/RULES.md).

Each check_rN_* function verifies one rule and returns (ok, evidence). Run all:

    PYTHONPATH=python:. .venv/bin/python akt/core/verify/rules_check.py

Exit status is non-zero on any violation. `pytest akt/benchmark/test_rules_check.py`
runs the same checks inside the battery, so a modification that breaks a rule
fails CI-style before it reaches a campaign.

These checks are deliberately host-portable (interpret/pure-Python): they verify
the CONTRACTS — the on-target measurements themselves are guarded by the gate.
"""

from __future__ import annotations

import os

os.environ.setdefault("PALLAS_INTERPRET", "1")

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _loop():
    spec = importlib.util.spec_from_file_location(
        "akt_loop_for_rules", ROOT / "akt/core/evolve/loop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ R1
def check_r1_unified_workloads():
    """Every active workload is expressible+reproducible in the formal layer."""
    from akt.benchmark.model_specs import library
    from akt.benchmark.model_workloads import FROZEN_MODEL_WORKLOADS, MODEL_WORKLOADS
    from akt.benchmark.workload_spec import lower, verify_legacy_roundtrip

    verify_legacy_roundtrip()                       # raises on drift
    if MODEL_WORKLOADS[: len(FROZEN_MODEL_WORKLOADS)] != FROZEN_MODEL_WORKLOADS:
        return False, "frozen prefix of MODEL_WORKLOADS was reordered/edited"
    lib = library()
    for workload in MODEL_WORKLOADS[len(FROZEN_MODEL_WORKLOADS):]:
        spec = lib.get(workload.model_id)
        if spec is None:
            return False, f"extended workload {workload.model_id!r} has no formal spec"
        lowered_ids = {c.case_id for c in lower(spec).workload.calls}
        active_ids = {c.case_id for c in workload.calls}
        if not active_ids <= lowered_ids:
            return False, (f"extended workload {workload.model_id!r} carries calls "
                           f"its spec cannot derive: {sorted(active_ids - lowered_ids)}")
    return True, (f"legacy roundtrip exact; {len(MODEL_WORKLOADS)-len(FROZEN_MODEL_WORKLOADS)} "
                  "extended workload(s), all re-derivable from the spec library")


# ------------------------------------------------------------------ R2
def check_r2_campaign_diagram_bottleneck():
    """The oracle-facing bottleneck comes from the measure→trace→verdict pipeline."""
    from akt.benchmark.model_workloads import callsite_inventory
    from akt.core.analysis import campaign

    try:
        diagnosis = campaign.diagnose()
    except campaign.CampaignUnavailable as error:
        return False, f"campaign diagnosis unavailable: {error}"
    levels = diagnosis.get("levels") or []
    if not levels:
        return False, "diagnosis has no levels"
    l0 = levels[0].get("segments") or []
    if not l0:
        return False, "L0 has no segments"
    missing_source = [s["name"] for s in l0 if not s.get("latency_source")]
    if missing_source:
        return False, f"L0 segments without a named latency source: {missing_source}"
    limiter_text = (diagnosis.get("limiter") or {}).get("text") or ""
    inventory = set(callsite_inventory())
    named = [site for site in inventory if site in limiter_text]
    if not named:
        return False, f"LIMITER does not name a measured callsite: {limiter_text[:120]!r}"
    if len(levels) > 1:
        l1 = levels[1].get("segments") or []
        undeclared = [s["name"] for s in l1 if "estimated" not in
                      (s.get("latency_share_basis") or "")]
        if undeclared:
            return False, f"L1 stages without a declared estimation basis: {undeclared[:4]}"
    return True, (f"diagnosis live: {len(l0)} measured L0 segments, limiter names "
                  f"{named[0]!r}, L1 shares declare their estimation basis")


# ------------------------------------------------------------------ R3
def check_r3_global_scale():
    """Parent utilization == duration-weighted mean of its children (global scale)."""
    from akt.core.analysis import campaign

    try:
        diagnosis = campaign.diagnose()
    except campaign.CampaignUnavailable as error:
        return False, f"campaign diagnosis unavailable: {error}"
    levels = diagnosis.get("levels") or []
    if len(levels) < 2:
        return True, "no L1 in diagnosis (single-level host) — vacuously consistent"
    l0 = levels[0]["segments"]
    l1 = levels[1]["segments"]
    dominant = max(l0, key=lambda s: s["share"])
    cmax = max(s["flop_mass"] / s["latency_s"] for s in l0 if s.get("latency_s"))
    bmax = max(s["byte_mass"] / s["latency_s"] for s in l0 if s.get("latency_s"))
    parent_c = dominant["flop_mass"] / dominant["latency_s"] / cmax
    parent_b = dominant["byte_mass"] / dominant["latency_s"] / bmax
    child_c = child_b = 0.0
    for stage in l1:
        if not stage.get("latency_s"):
            continue
        child_c += stage["flop_mass"] / stage["latency_s"] / cmax * stage["share"]
        child_b += stage["byte_mass"] / stage["latency_s"] / bmax * stage["share"]
    def close(a, b):
        return abs(a - b) <= max(0.01, 0.01 * max(a, b))
    if not (close(parent_c, child_c) and close(parent_b, child_b)):
        return False, (f"parent ({parent_c:.4f},{parent_b:.4f}) != weighted children "
                       f"({child_c:.4f},{child_b:.4f}) on the global scale")
    return True, (f"{dominant['name']}: parent cmp/bw ({parent_c:.4f},{parent_b:.4f}) "
                  f"== weighted mean of {len(l1)} stages within 1%")


# ------------------------------------------------------------------ R4
def check_r4_serialization():
    """ELEVATE before INVENT, probed on synthetic state."""
    import types

    loop = _loop()
    actions = [{"access": "existing", "gap_id": "fam:axis:pipeline-depth",
                "model_callsites": ["tiny-linear-serving/gla-short"]}]
    from akt.core.evolve import action_catalog
    original = action_catalog.action_catalog_context
    original_hist = loop.HIST
    try:
        action_catalog.action_catalog_context = lambda _r: {"actions": actions,
                                                            "fingerprint": "fp"}
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            loop.HIST = Path(temp) / "h.jsonl"
            state = {"expected_callsites": ["tiny-linear-serving/gla-short"]}
            novel = {"proposed_action": {"gap_id": "x"}}
            try:
                loop.enforce_phase_serialization(novel, state)
                return False, "novel manifest admitted while a red-link was un-attempted"
            except ValueError:
                pass
            loop.HIST.write_text(json.dumps({
                "gap_id": "fam:axis:pipeline-depth", "decision": "reject",
                "objective_scope": loop.OBJECTIVE_SCOPE}) + "\n")
            loop.enforce_phase_serialization(novel, state)   # must now pass
    finally:
        action_catalog.action_catalog_context = original
        loop.HIST = original_hist
    return True, "novel blocked while red-link un-attempted; unblocked after recorded attempt"


# ------------------------------------------------------------------ R5
def check_r5_flexgraph_interface():
    """Invention is confined to declared canonical action records."""
    from akt.core.evolve.action_catalog import (
        MINABLE_INTERFACE_CATEGORIES, load_action_graph,
    )
    from akt.core.evolve.capability_contract import validate_proposed_action

    if MINABLE_INTERFACE_CATEGORIES != frozenset({"pipeline-depth", "schedule-toggle"}):
        return False, f"minable interface set changed: {sorted(MINABLE_INTERFACE_CATEGORIES)}"
    load_action_graph(ROOT)                          # catalog must validate
    _p, errors = validate_proposed_action({"proposed_action": {"gap_id": "junk"}}, ROOT)
    if not errors:
        return False, "malformed proposed_action was accepted"
    return True, "catalog validates; malformed novel declarations rejected"


# ------------------------------------------------------------------ R6
def check_r6_default_is_incumbent():
    """Empty policy == dataclass defaults for every registered control."""
    from sgl_jax.srt.configs.kernel_control import PROGRAMMER_CONTROL_REGISTRY
    import dataclasses

    from sgl_jax.srt.configs import kernel_control as kc

    control_types = getattr(kc, "_CONTROL_TYPES", None)
    if not isinstance(control_types, dict):
        return False, "kernel_control lost its family -> typed-controls mapping"
    for family, controls in PROGRAMMER_CONTROL_REGISTRY.items():
        control_type = control_types.get(family)
        if control_type is None or not dataclasses.is_dataclass(control_type):
            return False, f"no typed control dataclass for family {family!r}"
        defaults = {f.name: f.default for f in dataclasses.fields(control_type)}
        for control in controls:
            if control not in defaults:
                return False, f"registered control {family}.{control} has no typed default"
        for banned in ("single_chunk_state_elision", "zero_state_output_elision"):
            if banned in controls:
                return False, f"elision flag {family}.{banned} leaked into the registry"
    return True, (f"{sum(len(c) for c in PROGRAMMER_CONTROL_REGISTRY.values())} registered "
                  "controls all carry typed defaults; elision flags excluded")


# ------------------------------------------------------------------ R7
def check_r7_frozen_fail_closed():
    """FROZEN covers the measurement definition; lowering accounts for every op."""
    loop = _loop()
    required_frozen = [
        "akt/benchmark/", "akt/core/analysis/flexgraph_extract.py",
        "akt/core/evolve/loop.py", "akt/core/evolve/capability_contract.py",
        "akt/core/evolve/exposure.py", "akt/core/evolve/action_catalog.py",
        "akt/core/search/model_dp.py", "python/sgl_jax/test/",
    ]
    missing = [p for p in required_frozen if p not in loop.FROZEN]
    if missing:
        return False, f"measurement-defining paths absent from FROZEN: {missing}"
    from akt.benchmark.model_specs import dense_transformer_spec
    from akt.benchmark.workload_spec import lower
    spec = dense_transformer_spec("probe", hidden=512, layers=2, heads=8,
                                  kv_heads=2, head_dim=64, inter=512, seqs=(256,))
    lowered = lower(spec)
    expanded = sum(len(b.ops) for b in spec.blocks)
    if len(lowered.coverage) != expanded:
        return False, (f"lowering dropped ops silently: coverage {len(lowered.coverage)} "
                       f"!= expanded {expanded}")
    return True, (f"FROZEN covers {len(required_frozen)} required paths; lowering "
                  f"accounts for all {expanded} ops (measurable + coverage gaps)")


# ------------------------------------------------------------------ R8
def check_r8_keep_bar():
    """The KEEP bar and guard conjunction are in force."""
    loop_source = (ROOT / "akt/core/evolve/loop.py").read_text()
    if 'max(float(st.get("target_improvement", 0.02)), 0.02)' not in loop_source:
        return False, "the 2% KEEP floor expression is gone from loop.py"
    history_path = ROOT / "akt/optimization_history/evolve_history.jsonl"
    below_bar_keeps = []
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("decision") != "keep":
                continue
            bar = record.get("target_pct")
            paired = record.get("improvement_pct")
            if isinstance(bar, (int, float)) and isinstance(paired, (int, float)):
                if abs(paired) <= bar:
                    below_bar_keeps.append((record.get("round"), paired))
    if below_bar_keeps:
        return False, f"history contains KEEPs at/below the bar: {below_bar_keeps}"
    return True, "2% floor present; no historical KEEP at or below its bar"


CHECKS = [
    ("R1 unified workload definition", check_r1_unified_workloads),
    ("R2 bottleneck via campaign diagram", check_r2_campaign_diagram_bottleneck),
    ("R3 one global utilization scale", check_r3_global_scale),
    ("R4 ELEVATE before INVENT", check_r4_serialization),
    ("R5 invention via flexgraph interface", check_r5_flexgraph_interface),
    ("R6 default path == incumbent", check_r6_default_is_incumbent),
    ("R7 frozen measurement, fail-closed", check_r7_frozen_fail_closed),
    ("R8 measured, guarded KEEP bar", check_r8_keep_bar),
]


def run_all():
    results = []
    for name, check in CHECKS:
        try:
            ok, evidence = check()
        except Exception as error:  # noqa: BLE001 — a crashed check is a failure
            ok, evidence = False, f"check crashed: {type(error).__name__}: {error}"
        results.append((name, ok, evidence))
    return results


def main():
    results = run_all()
    width = max(len(name) for name, _ok, _e in results)
    failures = 0
    for name, ok, evidence in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{mark}] {name:<{width}}  {evidence}")
    print(json.dumps({"AKT_RULES_CHECK": {
        "n_rules": len(results), "n_fail": failures,
        "failed": [name for name, ok, _e in results if not ok]}}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
