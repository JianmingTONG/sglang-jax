"""Benchmark ADAPTER — the execution/monitoring contract the AKT core consumes.

The AKT loop (akt/core/evolve/loop.py) never imports benchmark code; it shells out
to THIS file and reads one JSON line per subcommand, so the whole benchmark domain
(kernels, metrics, bottleneck, red-link action space) is swappable without touching core/.

CONTRACT
  adapter.py bottleneck  -> "AKT_BOTTLENECK {json}": {title, lines[], note}
      what dominates the suite latency (ranked kernels), pre-rendered for the QUERY.
  adapter.py gaps        -> "AKT_GAPS {json}":       {title, lines[], note}
      the validated red-link action catalog: source-derived low-level flexibilities
      the programmer API does NOT currently expose.
  adapter.py space [--kernel K] -> plain text: per-case design-space size + knobs
      (evidence the enlarged space is enumerated, used by loop.search_audit).

This file is part of the FROZEN harness: the loop may not edit anything under
akt/benchmark/.
"""
from __future__ import annotations

import os

os.environ.setdefault("PALLAS_INTERPRET", "1")

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]       # sglang-jax/
AKT = REPO / "akt"
for p in (str(REPO), str(REPO / "python"), str(AKT)):
    if p not in sys.path:
        sys.path.insert(0, p)

EVAL_OUT = AKT / "optimization_history/.evolve_eval.json"


# ------------------------------------------------------------------ bottleneck
def cmd_bottleneck(_args):
    """Rank models and selected callsites from the freshest target-HW evaluation."""
    if not EVAL_OUT.exists():
        print("AKT_BOTTLENECK " + json.dumps(
            {"missing": "no eval yet — run akt/benchmark/gates/model_eval.py --out "
                        f"{EVAL_OUT} (or the loop's `init`)"}))
        return
    s = json.loads(EVAL_OUT.read_text())
    if s.get("models"):
        from akt.benchmark.model_workloads import callsite_inventory

        inventory = callsite_inventory()
        lines = []
        rejected_candidate = s.get("gate_decision") == "reject"
        latency_key = "incumbent_s" if rejected_candidate else "candidate_s"
        plan_key = "incumbent_plan" if rejected_candidate else "selected_plan"
        role = "RETAINED INCUMBENT" if rejected_candidate else "CURRENT INCUMBENT"
        models = sorted(
            s["models"],
            key=lambda model: -(model.get(latency_key) or 0.0),
        )
        total = sum(model.get(latency_key) or 0.0 for model in models) or 1.0
        for model in models:
            latency = model.get(latency_key)
            ratio = model.get("ratio")
            lines.append(
                f"{role} MODEL {model.get('model')}: {latency * 1e3:.3f}ms "
                f"({100 * latency / total:.1f}% of three-model total), "
                f"paired candidate/incumbent={ratio:.4f}"
            )
            selected = model.get(plan_key) or {}
            call_rows = []
            for site, config in selected.items():
                if site not in model.get("callsites", []):
                    continue
                call = inventory.get(site)
                search = (s.get("case_search") or {}).get(call.case_id if call else "", {})
                measured = next(
                    (
                        row.get("latency_s")
                        for row in search.get("measurements", [])
                        if row.get("config") == config
                    ),
                    None,
                )
                if measured:
                    call_rows.append((measured, site, config))
            for measured, site, config in sorted(call_rows, reverse=True)[:4]:
                lines.append(
                    f"  CALL {site}: modeled {measured * 1e3:.3f}ms selected={config}"
                )
        runtime = s.get("runtime_evidence") or {}
        if runtime.get("required"):
            lines.append(
                ("REJECTED CANDIDATE " if rejected_candidate else "")
                + "RUNTIME EVIDENCE: "
                + ("PASS" if runtime.get("ok") else "FAIL")
                + ("; " + "; ".join(runtime.get("errors") or []) if runtime.get("errors") else "")
            )
            for control in runtime.get("controls") or []:
                for event in (control.get("matched") or [])[:4]:
                    lines.append(
                        f"  TRACE {control.get('control')}: {event.get('model')}/"
                        f"{str(event.get('callsite', '')).rsplit('/', 1)[-1]} -> "
                        f"{event.get('backend')}({event.get('value')!r}) "
                        f"probe={'verified' if event.get('verified_backend_probe') else 'untrusted'}"
                    )
        print(
            "AKT_BOTTLENECK "
            + json.dumps(
                {
                    "title": "paired target-hardware three-model latency and selected callsites",
                    "lines": lines,
                    "models": models,
                    "runtime_evidence": runtime,
                    "note": (
                        "Choose a source-derived open gap that maps to a dominant model callsite. "
                        + ("The latest candidate was rejected; rankings above use its paired retained-incumbent branch. "
                           if rejected_candidate else "")
                        + "The next round must remeasure every local configuration, certify the "
                        "model DP, and beat the paired incumbent on target hardware."
                    ),
                },
                allow_nan=False,
            )
        )
        return
    print("AKT_BOTTLENECK " + json.dumps({"missing": "evaluation has no model results"}))


def _auto_gap_lines():
    """All graph-linked red actions executable by the frozen three-model gate."""
    from akt.core.evolve.action_catalog import action_catalog_context

    context = action_catalog_context(REPO)
    actions = context["actions"]
    if not actions:
        # A VALID catalog with zero open actions is CONVERGENCE, not an error: every
        # source-proven red link has been closed (or none exists). Distinct from an
        # invalid/stale catalog, which raises above and reports "missing".
        note = (
            "CONVERGED: the validated action catalog is EMPTY — every source-proven "
            "red-link action has been closed or none remains discoverable. There is "
            "no executable gap left to elevate under this objective; widen the "
            "objective (new workloads/backends) or stop the campaign."
        )
        return [], note, context
    lines = []
    for gap in actions:
        sites = gap["model_callsites"]
        eligibility = "models=" + ",".join(sites)
        edge = gap["action_edge"]
        lines.append(
            f"gap_id={gap['gap_id']} ({gap['category']}) "
            f"{gap['source_evidence']['detail'][:150]} "
            f"<{gap['source_evidence']['path']}:{gap['source_evidence']['line']}> "
            f"[red-link {edge['source']} -> {edge['target']}; {eligibility}]"
        )
    note = (
        f"Complete fingerprinted action catalog ({len(actions)} actions). Each red link "
        "is an existing low-level choice that may be exposed; other graph findings and "
        "hidden compiler boundaries are context, not executable actions."
    )
    return lines, note, context


def cmd_gaps(_args):
    try:
        lines, note, context = _auto_gap_lines()
    except Exception as error:  # noqa: BLE001
        print("AKT_GAPS " + json.dumps({
            "missing": f"generated red-link action catalog is invalid: {error}"
        }))
        return
    title = "complete executable red-link AKT action catalog"
    print("AKT_GAPS " + json.dumps({
        "title": title,
        "lines": lines,
        "n_actions": len(lines),
        "converged": not lines,
        "action_graph_fingerprint": context["fingerprint"],
        "action_context_version": context["version"],
        "note": note,
    }, allow_nan=False))


# ------------------------------------------------------------------ gate contract
def cmd_gate(_args):
    """Self-describing gate metric contract — AKT_GATE {json}.

    The loop stores this at init/rebaseline and reads the objective's summary keys
    and direction from it instead of hardwired literals, so swapping the benchmark
    objective (e.g. a throughput metric where higher is better) is an adapter-side
    change; the frozen loop needs no edit. The values below mirror the current
    model_eval objective exactly."""
    print("AKT_GATE " + json.dumps({
        "objective_name": "paired_model_geomean_s",
        "objective_unit": "s",
        "objective_scope": "model-serving-empirical-dp-v2",
        "lower_is_better": True,
        "summary_keys": {
            "candidate": "candidate_geomean_s",
            "incumbent": "incumbent_geomean_s",
            "paired_ratio": "paired_ratio_geomean",
        },
        "eval_script": "akt/benchmark/gates/model_eval.py",
    }, allow_nan=False))


# ------------------------------------------------------------------ space report
def cmd_space(args):
    from akt.benchmark.suites import load_cases
    cases = load_cases("full")
    if args.kernel:
        cases = [c for c in cases if args.kernel in c.case_id or args.kernel == c.kernel_id]
    if not cases:
        print(f"[adapter] no cases match --kernel {args.kernel!r}")
        return
    print(f"design-space report ({len(cases)} case(s)):")
    for c in cases:
        deploy = c.space.deployment_space()
        knobs = [(k.name,
                  len(k.values) if k.programmer_control else 1,
                  "+" + k.elevated_by if k.elevated_by else "base",
                  k.programmer_control or "runner-only")
                 for k in c.space.knobs]
        print(f"  {c.case_id}: deployable_size={deploy.size()} "
              f"research_size={c.space.size()} knobs={knobs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bottleneck", help="emit AKT_BOTTLENECK json for the QUERY")
    sub.add_parser("gaps", help="emit AKT_GAPS json (canonical red-link action space)")
    sub.add_parser("gate", help="emit AKT_GATE json (self-describing metric contract)")
    ps = sub.add_parser("space", help="design-space size/knobs report (search evidence)")
    ps.add_argument("--kernel", default=None)
    args = ap.parse_args()
    {"bottleneck": cmd_bottleneck, "gaps": cmd_gaps, "gate": cmd_gate,
     "space": cmd_space}[args.cmd](args)


if __name__ == "__main__":
    main()
