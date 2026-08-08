"""Benchmark ADAPTER — the execution/monitoring contract the AKT core consumes.

The AKT loop (akt/core/evolve/loop.py) never imports benchmark code; it shells out
to THIS file and reads one JSON line per subcommand, so the whole benchmark domain
(kernels, metrics, bottleneck, red-link action space) is swappable without touching core/.

CONTRACT
  adapter.py bottleneck  -> "AKT_BOTTLENECK {json}": {title, lines[], note}
      what dominates the suite latency (ranked kernels), pre-rendered for the QUERY.
  adapter.py gaps        -> "AKT_GAPS {json}":       {title, lines[], relief{}, note}
      the validated red-link action catalog: source-derived low-level flexibilities
      the programmer API does NOT currently expose. Every catalog entry (red-link
      action, frontier slot, perpetual standalone slot) carries a `relief` evidence
      string: measured default-vs-best-alternative latency from the latest eval for
      red-link actions, campaign family-share/limiter diagnosis for slots.
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
CAMPAIGN_OUT = AKT / "board/campaign.json"


# ------------------------------------------------------------------ bottleneck
def _campaign_lines():
    """Recursive campaign-diagram diagnosis (latency x compute/bandwidth
    utilization per level) so the oracle sees WHY, not just WHICH. Never raises:
    returns (lines, ok)."""
    try:
        from akt.core.analysis.campaign import campaign_query_block

        return campaign_query_block().splitlines(), True
    except Exception as error:  # noqa: BLE001
        return [f"(campaign diagnosis unavailable: {error})"], False


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
        campaign_extra, campaign_ok = _campaign_lines()
        lines.extend(campaign_extra)
        print(
            "AKT_BOTTLENECK "
            + json.dumps(
                {
                    "title": "paired target-hardware three-model latency and selected callsites",
                    "lines": lines,
                    "campaign": campaign_ok,
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
    campaign_extra, campaign_ok = _campaign_lines()
    print("AKT_BOTTLENECK " + json.dumps({
        "missing": "evaluation has no model results",
        "lines": campaign_extra,
        "campaign": campaign_ok,
    }))


def _catalog_standalone_slots():
    """Perpetual `<family>:new_api:standalone` slots, straight from the generated graph."""
    from akt.core.evolve.action_catalog import GRAPH

    try:
        graph = json.loads((REPO / GRAPH).read_text())
    except Exception:  # noqa: BLE001
        return []
    return [
        record
        for record in graph.get("standalone_frontier_actions") or []
        if isinstance(record, dict) and isinstance(record.get("gap_id"), str)
    ]


def _measured_axis_relief(action, summary):
    """Concrete measured speedup evidence for one red-link action.

    Reads the latest gate eval's exhaustive per-case measurement tables: if the
    action's axis is a knob in a measured case of the action's kernel(s), report
    the best latency at the incumbent value vs the best alternative value —
    "measured[case]: default=Xms best_alt(axis=v)=Yms (±Z%)". A testbench-
    deferred case or an unmeasured axis is reported honestly, never invented.
    """
    axis = action.get("source_axis")
    incumbent = action.get("incumbent_value")
    kernels = set(action.get("kernel_ids") or [])
    case_search = (summary or {}).get("case_search") or {}
    parts = []
    for case_id in sorted(case_search):
        if case_id.split(":", 1)[0] not in kernels:
            continue
        best_by_value = {}
        for row in (case_search[case_id] or {}).get("measurements") or []:
            config = row.get("config") or {}
            latency = row.get("latency_s")
            if axis not in config or not isinstance(latency, (int, float)):
                continue
            key = json.dumps(config[axis], sort_keys=True)
            if key not in best_by_value or latency < best_by_value[key][0]:
                best_by_value[key] = (latency, config[axis])
        default_key = json.dumps(incumbent, sort_keys=True)
        default = best_by_value.get(default_key)
        alternatives = [v for k, v in best_by_value.items() if k != default_key]
        if default is None or not alternatives:
            continue
        alt_latency, alt_value = min(alternatives, key=lambda pair: pair[0])
        change = 100.0 * (alt_latency - default[0]) / default[0]
        parts.append(
            f"measured[{case_id}]: default={default[0] * 1e3:.2f}ms "
            f"best_alt({axis}={alt_value})={alt_latency * 1e3:.2f}ms "
            f"({change:+.1f}%)"
        )
    return "; ".join(parts) if parts else "unmeasured on testbench (tpu-deferred)"


def _campaign_diagnosis_relief(slot, campaign):
    """Campaign-diagram evidence for a frontier/standalone slot.

    Attaches the slot family's measured L0 latency share and verdict from the
    campaign diagnosis, plus an explicit LIMITER match and frontier-hint match
    when the diagnosis points at this family/slot."""
    if not isinstance(campaign, dict) or not campaign.get("levels"):
        return "diagnosis: campaign diagnosis unavailable"
    kernels = set(slot.get("kernel_ids") or [])
    share, verdicts, matched = 0.0, [], 0
    for level in campaign.get("levels") or []:
        if level.get("level") != 0:
            continue
        for segment in level.get("segments") or []:
            case = segment.get("case") or ""
            if case.split(":", 1)[0] not in kernels:
                continue
            matched += 1
            share += segment.get("share") or 0.0
            verdict = segment.get("verdict")
            if verdict and verdict not in verdicts:
                verdicts.append(verdict)
    extras = []
    limiter = campaign.get("limiter") or {}
    if limiter.get("kernel_id") in kernels:
        extras.append(
            "LIMITER match: "
            f"{limiter.get('stage')} {limiter.get('engine')} {limiter.get('verdict')}"
        )
    if slot.get("gap_id") in (campaign.get("hints") or []):
        extras.append("campaign frontier hint")
    if not matched:
        return "diagnosis: family unmeasured on testbench (tpu-deferred); no L0 share"
    text = (
        f"diagnosis: family share={100.0 * share:.1f}% "
        f"verdict={'/'.join(verdicts) or 'undetermined'} "
        f"(L0 {matched} callsite(s))"
    )
    return "; ".join([text, *extras])


def relief_evidence(context=None, summary=None, campaign=None):
    """gap_id -> concrete, honest relief evidence for EVERY catalog entry.

    Red-link actions carry measured default-vs-best-alternative latency from
    the latest eval summary (or an explicit unmeasured-on-testbench note);
    frontier and perpetual standalone slots carry campaign-diagnosis evidence
    (family L0 share, verdict, limiter/hint match). This makes the elevate-vs-
    invent choice evidence-based per entry — numbers are never invented."""
    if context is None:
        from akt.core.evolve.action_catalog import action_catalog_context

        context = action_catalog_context(REPO)
    if summary is None:
        try:
            summary = json.loads(EVAL_OUT.read_text()) if EVAL_OUT.exists() else {}
        except Exception:  # noqa: BLE001
            summary = {}
    if campaign is None:
        try:
            campaign = (
                json.loads(CAMPAIGN_OUT.read_text()) if CAMPAIGN_OUT.exists() else {}
            )
        except Exception:  # noqa: BLE001
            campaign = {}
    relief = {}
    for action in context.get("actions") or []:
        relief[action["gap_id"]] = _measured_axis_relief(action, summary)
    for slot in context.get("frontier_actions") or []:
        relief[slot["gap_id"]] = _campaign_diagnosis_relief(slot, campaign)
    for slot in _catalog_standalone_slots():
        relief[slot["gap_id"]] = _campaign_diagnosis_relief(slot, campaign)
    return relief


def _auto_gap_lines():
    """All graph-linked red actions executable by the frozen three-model gate."""
    from akt.core.evolve.action_catalog import action_catalog_context

    context = action_catalog_context(REPO)
    actions = context["actions"]
    frontier = context.get("frontier_actions") or []
    standalone = _catalog_standalone_slots()
    relief = relief_evidence(context=context)
    lines = []
    for gap in actions:
        sites = gap["model_callsites"]
        eligibility = "models=" + ",".join(sites)
        edge = gap["action_edge"]
        lines.append(
            f"gap_id={gap['gap_id']} ({gap['category']}) "
            f"{gap['source_evidence']['detail'][:150]} "
            f"<{gap['source_evidence']['path']}:{gap['source_evidence']['line']}> "
            f"[red-link {edge['source']} -> {edge['target']}; {eligibility}] "
            f"relief: {relief.get(gap['gap_id'], 'unavailable')}"
        )
    if not actions:
        # A VALID catalog with zero open actions is RED-LINK EXHAUSTION, not an
        # error: every source-proven red link has been closed (or none exists).
        # Distinct from an invalid/stale catalog, which raises above and reports
        # "missing". Rounds may continue with NOVEL-algorithm proposals declared
        # in the flexgraph interface (manifest `proposed_action`) — the FRONTIER
        # slots listed below (if any) are the extractor-ranked candidates.
        note = (
            "RED-LINK SPACE EXHAUSTED: the validated action catalog is EMPTY — "
            "every source-proven red-link action has been closed or none remains "
            "discoverable. No existing gap is left to elevate; further rounds must "
            "declare a NOVEL algorithm via `proposed_action`, or widen the "
            "objective (new workloads/backends)."
        )
    else:
        note = (
            f"Complete fingerprinted action catalog ({len(actions)} actions). Each red link "
            "is an existing low-level choice that may be exposed; other graph findings and "
            "hidden compiler boundaries are context, not executable actions."
        )
    for gap in frontier:
        sites = gap["model_callsites"]
        lines.append(
            f"FRONTIER(novel-slot) gap_id={gap['gap_id']} ({gap['category']}) "
            f"family={gap['family']} fn={gap['source_function']} "
            f"{gap['source_evidence']['detail'][:150]} "
            f"<{gap['source_evidence']['path']}:{gap['source_evidence']['line']}> "
            f"[no red link yet; models={','.join(sites)}] "
            f"relief: {relief.get(gap['gap_id'], 'unavailable')}"
        )
    if frontier:
        note += (
            f" {len(frontier)} FRONTIER slot(s) listed above are extractor-declared "
            "novel interfaces with no implemented source sink and no red action edge; "
            "elevating one means implementing the declared variant so the regenerated "
            "graph mines it as an executable action."
        )
    for slot in standalone:
        lines.append(
            f"STANDALONE(perpetual-api-slot) gap_id={slot['gap_id']} "
            f"({slot.get('category')}) family={slot.get('family')} "
            f"{(slot.get('detail') or slot.get('what') or '')[:150]} "
            f"<{slot.get('evidence')}> "
            f"[never closes; models={','.join(slot.get('model_callsites') or [])}] "
            f"relief: {relief.get(slot['gap_id'], 'unavailable')}"
        )
    if standalone:
        note += (
            f" {len(standalone)} PERPETUAL standalone-API slot(s) listed above never "
            "close; introducing one API does not exhaust the family."
        )
    note += (
        " Every line ends with its concrete `relief:` evidence — measured "
        "default-vs-best-alternative latency for red-link actions, campaign "
        "family-share/limiter diagnosis for frontier/standalone slots — so the "
        "elevate-vs-invent choice is evidence-based, never a subjective preference."
    )
    return lines, note, context, relief, len(standalone)


def cmd_gaps(_args):
    try:
        lines, note, context, relief, n_standalone = _auto_gap_lines()
    except Exception as error:  # noqa: BLE001
        print("AKT_GAPS " + json.dumps({
            "missing": f"generated red-link action catalog is invalid: {error}"
        }))
        return
    title = "complete executable red-link AKT action catalog"
    print("AKT_GAPS " + json.dumps({
        "title": title,
        "lines": lines,
        "n_actions": len(context["actions"]),
        "n_frontier": len(context.get("frontier_actions") or []),
        "n_standalone": n_standalone,
        "converged": not context["actions"],
        "relief": relief,
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
