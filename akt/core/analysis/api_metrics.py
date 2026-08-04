"""ISA-grounded API-novelty metrics for the AKT gate (two calibrated guardrails).

The campaign's audit showed every kept capability was a schedule/packaging change:
efficient use of the EXISTING abstraction, not a new one. These guardrails filter
that class out of the KEEP path so only abstraction-level changes survive.

1. DIRECTIONAL, ISA-GROUNDED REDUNDANCY.
   For a candidate program N (the affected kernel case run with the new control at
   its selected non-default value) and an existing program E (an incumbent
   configuration of the same case), both are lowered under the same operation
   pattern, shape, dtype and compiler configuration to their operation DEPENDENCY
   GRAPHS: the jaxpr equation DAG, recursing into Pallas kernel bodies (this is
   the operation stream the Mosaic/TPU compiler receives; dependency graphs, not
   consecutive instruction sequences, so compiler reordering cannot hide a match).
   Semantic compatibility is the frozen reference check the gate already enforces.

       R(N, E) = matched_weight(N reproducible by E) / total_weight(N)

   computed as a maximum-weight dependency-preserving subgraph approximation via
   Weisfeiler-Lehman signatures (label = primitive + output shapes/dtypes,
   iterated over parent/child neighborhoods; equal signatures imply locally
   isomorphic dependency contexts). Weights are instruction counts by default
   (``weight_mode="count"``); ``"size"`` weighs each operation by its output
   element count as a first cycle/resource proxy.

       D(N) = max_E R(N, E)          (duplicate risk; E* = argmax)

   Directionality: R(N, E*) and R(E*, N) high together => duplicate;
   R(N, E*) low while R(E*, N) high => N GENERALIZES E* (admissible).

   Threshold calibration: leave-one-out nearest-neighbor D among each family's
   existing one-knob-off programs — the canonical parameter-tweak population —
   summarized as catalog PERCENTILES (never the mean alone).

2. GENERICITY = UPPER-LAYER COVERAGE.

       G_delta(N) = |unique upper-layer patterns with a valid u -> N -> l path
                     and local benefit >= delta| / |relevant upper-layer patterns|

   Upper-layer patterns are the frozen model callsites (each counted once);
   "valid path with benefit" means the callsite's best measured configuration
   selects the new control at a non-default value and improves the callsite's
   best incumbent-configuration latency by at least delta. Lower-layer fan-out
   (how many kernel functions the control forwards into) is reported separately
   as ``implementation_breadth`` — it is NOT genericity.

Run ``profile`` on the target host to build the calibration baseline
(``api_baseline.json``); the loop pins it at init/rebaseline and the gate keeps
only candidates that clear the calibrated percentiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PALLAS_INTERPRET", "1")  # before any kernel import

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "akt/core/analysis/api_baseline.json"

DEFAULT_WL_ITERS = 2
DEFAULT_WEIGHT_MODE = "count"
DEFAULT_DELTA = 0.02
# Catalog-percentile policy (stored in the baseline so the gate and the profile
# agree): a candidate is non-redundant when its duplicate risk sits BELOW the
# parameter-tweak population's low tail; generalization needs the reverse
# direction at or above the population median.
DEFAULT_REDUNDANCY_PERCENTILE = 10
DEFAULT_GENERALIZATION_PERCENTILE = 50
DEFAULT_GENERICITY_PERCENTILE = 50
# The redundancy cut never rises above this ceiling: a candidate that an
# existing API can reproduce at >=90% weight is a duplicate no matter how
# self-similar the calibration population is (a degenerate tweak population
# with p10 = 1.0 must not make the guard vacuous).
DEFAULT_REDUNDANCY_CEILING = 0.9
# The genericity cut never drops below covering HALF the relevant upper-layer
# patterns, even when a proxy-host calibration degenerates (e.g. interpret
# timings flattening every existing control's G to zero). A new abstraction
# must be broadly useful, not a single-callsite special case.
DEFAULT_GENERICITY_FLOOR = 0.5


# ------------------------------------------------------------ operation graphs


def _sub_jaxprs(value):
    """Yield every Jaxpr nested inside one eqn-params value (any container)."""
    import jax.extend.core as jex_core

    if isinstance(value, jex_core.ClosedJaxpr):
        yield value.jaxpr
        return
    if isinstance(value, jex_core.Jaxpr):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _sub_jaxprs(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _sub_jaxprs(item)


def _eqn_label(eqn, label_mode: str = "rank") -> str:
    """Matching label for one operation.

    ``rank`` (default) abstracts shape EXTENTS away (primitive + dtype + rank):
    a tile/chunk-size re-parameterization of the same algorithm then matches its
    peers almost fully (high R => correctly flagged as a parameter tweak), while
    a different operation composition/wiring still mismatches. ``shape`` keeps
    full extents for finer structural studies; extents always contribute via the
    ``size`` weight mode regardless of label mode.
    """
    if label_mode == "shape":
        outs = ",".join(
            f"{getattr(v.aval, 'dtype', '?')}{list(getattr(v.aval, 'shape', []))}"
            for v in eqn.outvars
        )
    else:
        outs = ",".join(
            f"{getattr(v.aval, 'dtype', '?')}r{len(getattr(v.aval, 'shape', []) or ())}"
            for v in eqn.outvars
        )
    return f"{eqn.primitive.name}|{outs}"


def _eqn_weight(eqn, mode: str) -> float:
    if mode == "count":
        return 1.0
    size = 1
    for v in eqn.outvars:
        shape = getattr(v.aval, "shape", None) or ()
        n = 1
        for dim in shape:
            n *= int(dim)
        size = max(size, n)
    return float(size)


def operation_graph(
    trace_fn,
    *,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
    label_mode: str = "rank",
) -> dict:
    """Dependency graph of every operation the program emits, at every nesting
    level (pjit / scan / cond / pallas_call kernel bodies included). Edges are
    data dependencies within each jaxpr level; sub-jaxpr boundaries are level
    breaks (documented approximation — the intra-kernel structure is what
    distinguishes algorithms)."""
    import jax

    closed = jax.make_jaxpr(trace_fn)()
    labels: list[str] = []
    weights: list[float] = []
    edges: list[tuple[int, int]] = []

    def visit(jaxpr):
        producer: dict = {}
        for eqn in jaxpr.eqns:
            index = len(labels)
            labels.append(_eqn_label(eqn, label_mode))
            weights.append(_eqn_weight(eqn, weight_mode))
            for var in eqn.invars:
                key = id(var)
                if key in producer:
                    edges.append((producer[key], index))
            for var in eqn.outvars:
                producer[id(var)] = index
            for sub in _sub_jaxprs(eqn.params):
                visit(sub)

    visit(closed.jaxpr)
    return {"labels": labels, "weights": weights, "edges": edges}


def _digest(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def wl_signatures(graph: dict, iters: int = DEFAULT_WL_ITERS) -> list[str]:
    """Ancestry Weisfeiler-Lehman signature per node: label refined by the
    sorted PARENT signature multiset only. An operation is "reproducible by E"
    when E computes the same value the same way — that is a property of its
    ancestor structure, not of what happens to consume it later, so children are
    deliberately excluded (a superset program must still match every op of the
    program it extends). Equal signatures => locally isomorphic dependency
    ancestries, the dependency-preserving matching unit."""
    n = len(graph["labels"])
    parents: list[list[int]] = [[] for _ in range(n)]
    for src, dst in graph["edges"]:
        parents[dst].append(src)
    sigs = [_digest(label) for label in graph["labels"]]
    for _ in range(iters):
        sigs = [
            _digest(
                sigs[i]
                + "|P:" + ",".join(sorted(sigs[p] for p in parents[i]))
            )
            for i in range(n)
        ]
    return sigs


def redundancy(
    graph_n: dict, graph_e: dict, *, wl_iters: int = DEFAULT_WL_ITERS
) -> float:
    """R(N, E): weight fraction of N's operations reproducible by E, matched by
    WL signature multisets (a deterministic under-approximation of the
    maximum-weight dependency-preserving common subgraph)."""
    total = sum(graph_n["weights"])
    if total <= 0:
        return 0.0
    sig_n = wl_signatures(graph_n, wl_iters)
    sig_e = wl_signatures(graph_e, wl_iters)
    counts_e: dict[str, int] = {}
    for sig in sig_e:
        counts_e[sig] = counts_e.get(sig, 0) + 1
    matched = 0.0
    budget = dict(counts_e)
    # ops sharing a signature share a label, hence a weight — greedy is exact here
    for sig, weight in zip(sig_n, graph_n["weights"]):
        if budget.get(sig, 0) > 0:
            budget[sig] -= 1
            matched += weight
    return matched / total


def duplicate_risk(
    graph_n: dict,
    catalog: dict[str, dict],
    *,
    wl_iters: int = DEFAULT_WL_ITERS,
) -> dict:
    """D(N) = max_E R(N, E) over the catalog, with the directional pair for E*."""
    best_id, best_r = None, -1.0
    for program_id, graph_e in catalog.items():
        score = redundancy(graph_n, graph_e, wl_iters=wl_iters)
        if score > best_r:
            best_id, best_r = program_id, score
    if best_id is None:
        return {"D": 0.0, "closest": None, "R_ne": 0.0, "R_en": 0.0}
    reverse = redundancy(catalog[best_id], graph_n, wl_iters=wl_iters)
    return {"D": best_r, "closest": best_id, "R_ne": best_r, "R_en": reverse}


# ------------------------------------------------------------ program catalogs


def _config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def program_graph(case, config: dict, *, weight_mode: str = DEFAULT_WEIGHT_MODE) -> dict:
    inputs = case.make_inputs()
    return operation_graph(lambda: case.run(inputs, config), weight_mode=weight_mode)


def one_knob_programs(case, *, exclude_knobs: set[str] | None = None) -> dict[str, dict]:
    """The canonical existing-API population of one case: the default program and
    every single-knob deviation (the parameter-tweak neighborhood). Returns
    {program_id: config}."""
    exclude = exclude_knobs or set()
    space = case.space
    default = space.default_config()
    programs: dict[str, dict] = {}
    if space.valid(default):
        programs[f"default:{_config_key(default)}"] = default
    for knob in space.knobs:
        if knob.name in exclude:
            continue
        for value in knob.values:
            if value == knob.default:
                continue
            config = dict(default)
            config[knob.name] = value
            if not space.valid(config):
                continue
            programs[f"{knob.name}={value!r}"] = config
    return programs


def _percentiles(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    ordered = sorted(scores)
    out = {}
    for pct in (5, 10, 25, 50, 75, 90, 95):
        rank = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
        out[f"p{pct}"] = ordered[rank]
    out["n"] = len(ordered)
    out["min"], out["max"] = ordered[0], ordered[-1]
    return out


def loo_duplicate_scores(
    case,
    *,
    wl_iters: int = DEFAULT_WL_ITERS,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
) -> list[dict]:
    """Leave-one-out nearest-neighbor D among one case's existing programs."""
    programs = one_knob_programs(case)
    graphs: dict[str, dict] = {}
    for program_id, config in programs.items():
        try:
            graphs[program_id] = program_graph(case, config, weight_mode=weight_mode)
        except Exception as error:  # noqa: BLE001 - untraceable => excluded, recorded
            graphs[program_id] = {"error": f"{type(error).__name__}: {str(error)[:120]}"}
    scores = []
    valid = {pid: g for pid, g in graphs.items() if "error" not in g}
    for program_id, graph in valid.items():
        rest = {pid: g for pid, g in valid.items() if pid != program_id}
        if not rest:
            continue
        result = duplicate_risk(graph, rest, wl_iters=wl_iters)
        scores.append({"program": program_id, **result})
    errors = {pid: g["error"] for pid, g in graphs.items() if "error" in g}
    if errors:
        scores.append({"trace_errors": errors})
    return scores


# ------------------------------------------------------------------ genericity


def genericity_from_case_search(
    case_search: dict,
    *,
    control_knob: str,
    relevant_case_to_callsites: dict[str, list[str]],
    delta: float = DEFAULT_DELTA,
) -> dict:
    """G_delta over the frozen model callsites (each upper pattern counted once).

    benefit(callsite) = (best latency among configs with the control at DEFAULT
    minus best among NON-default) / best-default, from the gate's own measured
    ``case_search`` population.
    """
    covered, relevant, per_site = [], [], {}
    for case_id, callsites in relevant_case_to_callsites.items():
        result = case_search.get(case_id) or {}
        knob = next(
            (k for k in result.get("knobs") or [] if k.get("name") == control_knob),
            None,
        )
        measurements = result.get("measurements") or []
        for site in callsites:
            relevant.append(site)
            if knob is None or not measurements:
                per_site[site] = {"benefit": None, "reason": "control absent"}
                continue
            default = knob.get("default")
            base = [
                m["latency_s"]
                for m in measurements
                if m.get("config", {}).get(control_knob, default) == default
                and isinstance(m.get("latency_s"), (int, float))
            ]
            alt = [
                m["latency_s"]
                for m in measurements
                if m.get("config", {}).get(control_knob, default) != default
                and isinstance(m.get("latency_s"), (int, float))
            ]
            if not base or not alt:
                per_site[site] = {"benefit": None, "reason": "one-sided population"}
                continue
            benefit = (min(base) - min(alt)) / min(base)
            per_site[site] = {"benefit": benefit}
            if benefit >= delta:
                covered.append(site)
    return {
        "G": (len(covered) / len(relevant)) if relevant else 0.0,
        "delta": delta,
        "covered_patterns": sorted(covered),
        "relevant_patterns": sorted(relevant),
        "per_pattern": per_site,
    }


def implementation_breadth(forwarding_paths: dict) -> int:
    """Lower-layer fan-out: distinct kernel functions the control reaches.
    Reported separately from genericity, per the metric definition."""
    functions = set()
    for trail in (forwarding_paths or {}).values():
        for step in trail or []:
            functions.add((step.get("path"), step.get("function")))
    return len(functions)


# ------------------------------------------------------------------- baseline


def _load_cases():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from akt.benchmark.suites import load_cases

    return load_cases("full")


def _callsites_by_case() -> dict[str, list[str]]:
    from akt.benchmark.model_workloads import callsite_inventory

    mapping: dict[str, list[str]] = {}
    for site, call in callsite_inventory().items():
        mapping.setdefault(call.case_id, []).append(site)
    return mapping


def build_baseline(
    *,
    wl_iters: int = DEFAULT_WL_ITERS,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
    delta: float = DEFAULT_DELTA,
    genericity_iters: int = 5,
) -> dict:
    """Profile the EXISTING APIs to calibrate both guardrail thresholds.

    Redundancy: per-family leave-one-out D percentiles over one-knob-off
    programs. Genericity: G_delta of every already-registered programmer control,
    measured with the runner timer over the same one-knob population (recomputed
    on the target host at rebaseline; the recorded regime marks proxy timings).
    """
    import jax

    from akt.benchmark.runners.base import time_config

    cases = _load_cases()
    case_to_sites = _callsites_by_case()
    families: dict[str, list[float]] = {}
    per_case: dict[str, dict] = {}
    for case in cases:
        scores = loo_duplicate_scores(case, wl_iters=wl_iters, weight_mode=weight_mode)
        d_values = [s["D"] for s in scores if "D" in s]
        families.setdefault(case.kernel_id, []).extend(d_values)
        per_case[case.case_id] = {
            "scores": scores,
            "percentiles": _percentiles(d_values),
        }

    all_scores = [d for values in families.values() for d in values]
    genericity_scores: dict[str, dict] = {}
    for case in cases:
        controls = [k for k in case.space.knobs if k.programmer_control]
        if not controls:
            continue
        try:
            timings = {}
            for program_id, config in one_knob_programs(case).items():
                timings[program_id] = {
                    "config": config,
                    "latency_s": time_config(case, config, iters=genericity_iters)[
                        "median_s"
                    ],
                }
        except Exception as error:  # noqa: BLE001 - untimeable on this host
            for knob in controls:
                genericity_scores.setdefault(
                    knob.programmer_control,
                    {"error": f"{type(error).__name__}: {str(error)[:120]}"},
                )
            continue
        case_search = {
            case.case_id: {
                "knobs": [
                    {
                        "name": k.name,
                        "values": k.values,
                        "default": k.default,
                        "programmer_control": k.programmer_control,
                    }
                    for k in case.space.knobs
                ],
                "measurements": list(timings.values()),
            }
        }
        for knob in controls:
            entry = genericity_scores.setdefault(
                knob.programmer_control, {"G_by_case": {}}
            )
            if "G_by_case" not in entry:
                continue
            result = genericity_from_case_search(
                case_search,
                control_knob=knob.name,
                relevant_case_to_callsites={
                    case.case_id: case_to_sites.get(case.case_id, [])
                },
                delta=delta,
            )
            entry["G_by_case"][case.case_id] = result["G"]

    g_values = []
    for control, entry in genericity_scores.items():
        by_case = entry.get("G_by_case") or {}
        if by_case:
            entry["G"] = sum(by_case.values()) / len(by_case)
            g_values.append(entry["G"])

    baseline = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": jax.default_backend(),
        "pallas_interpret": os.environ.get("PALLAS_INTERPRET") == "1",
        "wl_iters": wl_iters,
        "weight_mode": weight_mode,
        "delta": delta,
        "redundancy": {
            "per_family": {
                family: _percentiles(values) for family, values in families.items()
            },
            "global": _percentiles(all_scores),
            "per_case": per_case,
        },
        "genericity": {
            "per_control": genericity_scores,
            "global": _percentiles(g_values),
        },
        "policy": {
            "redundancy_percentile": DEFAULT_REDUNDANCY_PERCENTILE,
            "generalization_percentile": DEFAULT_GENERALIZATION_PERCENTILE,
            "genericity_percentile": DEFAULT_GENERICITY_PERCENTILE,
        },
    }
    raw = json.dumps(baseline, sort_keys=True, separators=(",", ":"))
    baseline["fingerprint"] = hashlib.sha256(raw.encode()).hexdigest()
    return baseline


# ------------------------------------------------------------ gate-side check


def _threshold(percentiles: dict, pct: int, fallback: float) -> float:
    value = (percentiles or {}).get(f"p{pct}")
    return value if isinstance(value, (int, float)) else fallback


def gate_metrics(
    *,
    baseline: dict,
    capability_controls: list[dict],
    case_search: dict,
    relevant_case_to_callsites: dict[str, list[str]],
    forwarding_paths: dict | None = None,
    delta: float | None = None,
) -> dict:
    """Compute both guardrails for one pending capability and apply the pinned,
    catalog-percentile thresholds.

    ``capability_controls``: [{"case_id", "knob", "selected_value"}] — the new
    control per affected case with the plan-selected non-default value. The
    candidate program N is that case at default+{knob: selected}; the existing
    catalog E is the case's one-knob-off population with the new knob held at
    default (the incumbent APIs). Uniform for ELEVATE and NOVEL rounds: a pure
    parameter tweak scores D ~= tweak-population percentiles and fails.
    """
    wl_iters = baseline.get("wl_iters", DEFAULT_WL_ITERS)
    weight_mode = baseline.get("weight_mode", DEFAULT_WEIGHT_MODE)
    policy = baseline.get("policy") or {}
    delta = baseline.get("delta", DEFAULT_DELTA) if delta is None else delta
    cases = {case.case_id: case for case in _load_cases()}

    per_case = []
    errors = []
    worst = {"D": -1.0}
    for record in capability_controls:
        case = cases.get(record["case_id"])
        knob_name = record["knob"]
        if case is None:
            errors.append(f"unknown case {record['case_id']!r}")
            continue
        default_config = case.space.default_config()
        candidate_config = dict(default_config)
        candidate_config[knob_name] = record["selected_value"]
        try:
            graph_n = program_graph(case, candidate_config, weight_mode=weight_mode)
            catalog = {}
            for program_id, config in one_knob_programs(
                case, exclude_knobs={knob_name}
            ).items():
                catalog[program_id] = program_graph(
                    case, config, weight_mode=weight_mode
                )
        except Exception as error:  # noqa: BLE001 - fail closed
            errors.append(
                f"cannot trace {record['case_id']}: "
                f"{type(error).__name__}: {str(error)[:160]}"
            )
            continue
        result = duplicate_risk(graph_n, catalog, wl_iters=wl_iters)
        family = case.kernel_id
        family_pct = (baseline.get("redundancy") or {}).get("per_family", {}).get(
            family
        ) or (baseline.get("redundancy") or {}).get("global") or {}
        cut = min(
            _threshold(
                family_pct,
                policy.get("redundancy_percentile", DEFAULT_REDUNDANCY_PERCENTILE),
                0.5,
            ),
            DEFAULT_REDUNDANCY_CEILING,
        )
        general_floor = _threshold(
            family_pct,
            policy.get(
                "generalization_percentile", DEFAULT_GENERALIZATION_PERCENTILE
            ),
            0.9,
        )
        generalizes = result["R_ne"] < cut and result["R_en"] >= general_floor
        record_out = {
            **record,
            **result,
            "family": family,
            "redundancy_cut": cut,
            "generalization_floor": general_floor,
            "non_redundant": result["D"] < cut,
            "generalizes_closest": generalizes,
            "redundancy_ok": result["D"] < cut or generalizes,
        }
        per_case.append(record_out)
        if result["D"] > worst["D"]:
            worst = {**record_out}

    knob_names = {record["knob"] for record in capability_controls}
    genericity = {}
    for knob_name in sorted(knob_names):
        genericity[knob_name] = genericity_from_case_search(
            case_search,
            control_knob=knob_name,
            relevant_case_to_callsites=relevant_case_to_callsites,
            delta=delta,
        )
    g_cut = max(
        _threshold(
            (baseline.get("genericity") or {}).get("global") or {},
            policy.get("genericity_percentile", DEFAULT_GENERICITY_PERCENTILE),
            0.5,
        ),
        DEFAULT_GENERICITY_FLOOR,
    )
    g_min = min((entry["G"] for entry in genericity.values()), default=0.0)

    redundancy_ok = bool(per_case) and all(r["redundancy_ok"] for r in per_case)
    genericity_ok = bool(genericity) and g_min >= g_cut
    ok = redundancy_ok and genericity_ok and not errors
    return {
        "ok": ok,
        "redundancy_ok": redundancy_ok,
        "genericity_ok": genericity_ok,
        "per_case": per_case,
        "worst_duplicate_risk": worst if worst["D"] >= 0 else None,
        "genericity": genericity,
        "genericity_min": g_min,
        "genericity_cut": g_cut,
        "implementation_breadth": implementation_breadth(forwarding_paths or {}),
        "baseline_fingerprint": baseline.get("fingerprint"),
        "errors": errors,
    }


# ------------------------------------------------------------------------ CLI


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    profile = sub.add_parser(
        "profile", help="calibrate both guardrails from the existing APIs"
    )
    profile.add_argument("--out", default=str(BASELINE))
    profile.add_argument("--wl-iters", type=int, default=DEFAULT_WL_ITERS)
    profile.add_argument(
        "--weight-mode", choices=("count", "size"), default=DEFAULT_WEIGHT_MODE
    )
    profile.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    score = sub.add_parser("score", help="score one candidate config vs its case")
    score.add_argument("--case", required=True)
    score.add_argument("--knob", required=True)
    score.add_argument("--value", required=True, help="JSON value for the knob")
    gate = sub.add_parser(
        "gate", help="compute both guardrails for one pending capability"
    )
    gate.add_argument("--spec", required=True, help="JSON spec path (see gate_metrics)")
    gate.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.cmd == "gate":
        spec = json.loads(Path(args.spec).read_text())
        baseline = json.loads(Path(spec["baseline_path"]).read_text())
        summary = json.loads(Path(spec["summary_path"]).read_text())
        result = gate_metrics(
            baseline=baseline,
            capability_controls=spec["capability_controls"],
            case_search=summary.get("case_search") or {},
            relevant_case_to_callsites=spec["relevant_case_to_callsites"],
            forwarding_paths=spec.get("forwarding_paths") or {},
            delta=spec.get("delta"),
        )
        Path(args.out).write_text(json.dumps(result, indent=1, allow_nan=False))
        print("AKT_API_NOVELTY " + json.dumps(
            {
                "ok": result["ok"],
                "redundancy_ok": result["redundancy_ok"],
                "genericity_ok": result["genericity_ok"],
                "genericity_min": result["genericity_min"],
                "worst_D": (result.get("worst_duplicate_risk") or {}).get("D"),
            }
        ))
        return

    if args.cmd == "profile":
        baseline = build_baseline(
            wl_iters=args.wl_iters, weight_mode=args.weight_mode, delta=args.delta
        )
        Path(args.out).write_text(json.dumps(baseline, indent=1))
        red = baseline["redundancy"]["global"]
        gen = baseline["genericity"]["global"]
        print(
            f"[api-metrics] baseline -> {args.out}\n"
            f"[api-metrics] redundancy LOO D percentiles (global): "
            + ", ".join(f"{k}={v:.3f}" for k, v in red.items() if k.startswith("p"))
            + f" (n={red.get('n', 0)})\n"
            f"[api-metrics] existing-control genericity percentiles: "
            + (
                ", ".join(
                    f"{k}={v:.3f}" for k, v in gen.items() if k.startswith("p")
                )
                if gen
                else "(no timed controls on this host)"
            )
            + f"\n[api-metrics] AKT_API_BASELINE {json.dumps({'fingerprint': baseline['fingerprint'], 'backend': baseline['backend']})}"
        )
        return

    cases = {case.case_id: case for case in _load_cases()}
    case = cases[args.case]
    knob_value = json.loads(args.value)
    config = case.space.default_config()
    config[args.knob] = knob_value
    graph_n = program_graph(case, config)
    catalog = {
        pid: program_graph(case, cfg)
        for pid, cfg in one_knob_programs(case, exclude_knobs={args.knob}).items()
    }
    print(json.dumps(duplicate_risk(graph_n, catalog), indent=1))


if __name__ == "__main__":
    main()
