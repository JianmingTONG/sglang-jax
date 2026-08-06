"""API COVERAGE GRAPH — how every suite API relates to the existing APIs.

Node set: each kernel case's DEFAULT-config program, plus one node per
non-default value of every dispatch/variant knob (a knob elevated by a
capability, or a programmer_control whose values are non-boolean strings),
traced at default+that value and labeled ``case_id@knob=value``.

Edges reuse the delta machinery in akt/core/analysis/api_metrics.py (imported,
never modified): for an ordered node pair (A, B),

    f_AB = 1 - subtract_catalog(graph_A, {B: graph_B})["delta_pct"]

is the fraction of A's substantive op mass the single-program catalog {B}
reproduces (instance cover + ancestry match). The unordered pair is classified:

    equivalent   f_AB > 0.98 and f_BA > 0.98
    generalizes  one direction > 0.9, the other <= 0.9 (edge normalized so
                 ``a`` is the SUPERSET: a reproduces b, i.e. f_ba > 0.9)
    overlaps     max(f_AB, f_BA) in (0.3, 0.9] — or both > 0.9 without
                 clearing the equivalent bar (mutual-coverage guard)
    disjoint     otherwise

Runtime bounds: cross-case pairs are restricted to case DEFAULT nodes; variant
nodes compare only against case defaults (their own case's + the others');
total nodes are capped at 40 (defaults kept first, drops logged) and a wall
budget truncates the edge sweep rather than overrunning.

CLI (run from the repo root with PYTHONPATH=python:.):
    python akt/core/analysis/coverage.py [--out akt/board/coverage.json]
writes coverage.json and prints one "AKT_COVERAGE {json}" summary line.
Degrades to an ``unavailable`` record if api_metrics or tracing fails.
"""
from __future__ import annotations

import os

os.environ.setdefault("PALLAS_INTERPRET", "1")  # before any kernel import

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # akt/
REPO = ROOT.parent                              # sglang-jax/
for _p in (str(REPO), str(REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_OUT = ROOT / "board/coverage.json"
NODE_CAP = 40
EQUIVALENT_MIN = 0.98
GENERALIZES_MIN = 0.90
OVERLAP_MIN = 0.30
DEFAULT_BUDGET_S = 900.0


def _is_variant_knob(knob) -> bool:
    """Dispatch/variant axis: capability-elevated, or a programmer control
    whose candidate values are (non-boolean) strings."""
    if getattr(knob, "elevated_by", None):
        return True
    values = list(getattr(knob, "values", None) or [])
    return bool(
        getattr(knob, "programmer_control", None)
        and values
        and all(isinstance(v, str) for v in values)
    )


def _node_stats(api_metrics, graph: dict) -> dict:
    classes = api_metrics.classify_labels(graph["labels"])
    weights = graph["weights"]
    return {
        "ops": sum(1 for c in classes if c != "container"),
        "mass_compute": sum(
            w for w, c in zip(weights, classes) if c == "compute"
        ),
        "mass_memory": sum(
            w for w, c in zip(weights, classes) if c == "memory"
        ),
    }


def _collect_nodes(api_metrics, cases) -> tuple[list[dict], list[dict]]:
    """Trace the node set. Returns (nodes, skipped); every node carries its
    operation graph under the internal ``graph`` key (stripped before output)."""
    nodes: list[dict] = []
    skipped: list[dict] = []

    def _trace(node_id: str, case, config: dict, extra: dict) -> None:
        try:
            graph = api_metrics.program_graph(case, config)
        except Exception as error:  # noqa: BLE001 - untraceable on this host
            skipped.append({
                "id": node_id,
                "reason": f"{type(error).__name__}: {str(error)[:120]}",
            })
            return
        nodes.append({
            "id": node_id,
            "case": case.case_id,
            "kernel": case.kernel_id,
            **extra,
            **_node_stats(api_metrics, graph),
            "graph": graph,
        })

    for case in cases:
        space = case.space
        default = space.default_config()
        before = len(nodes)
        _trace(case.case_id, case, default, {})
        if len(nodes) == before:
            continue  # default untraceable -> variants of it are meaningless
        for knob in space.knobs:
            if not _is_variant_knob(knob):
                continue
            for value in knob.values:
                if value == knob.default:
                    continue
                node_id = f"{case.case_id}@{knob.name}={value}"
                config = dict(default)
                config[knob.name] = value
                try:
                    valid = space.valid(config)
                except Exception:  # noqa: BLE001
                    valid = False
                if not valid:
                    skipped.append({"id": node_id, "reason": "invalid config"})
                    continue
                _trace(node_id, case, config,
                       {"knob": knob.name, "value": value})
    return nodes, skipped


def _coverage_fraction(api_metrics, a: dict, b: dict) -> float:
    record = api_metrics.subtract_catalog(a["graph"], {b["id"]: b["graph"]})
    return max(0.0, min(1.0, 1.0 - float(record["delta_pct"])))


def _classify(f_ab: float, f_ba: float) -> tuple[str, str | None]:
    """(relation, superset) where superset names which node reproduces the
    other for 'generalizes' ('a': a covers b; 'b': b covers a)."""
    if f_ab > EQUIVALENT_MIN and f_ba > EQUIVALENT_MIN:
        return "equivalent", None
    if f_ab > GENERALIZES_MIN and f_ba <= GENERALIZES_MIN:
        return "generalizes", "b"       # b reproduces a
    if f_ba > GENERALIZES_MIN and f_ab <= GENERALIZES_MIN:
        return "generalizes", "a"       # a reproduces b
    if f_ab > GENERALIZES_MIN and f_ba > GENERALIZES_MIN:
        # Mutual coverage above the generalizes bar but below the equivalent
        # bar: closest honest label is overlap (the spec's classification has
        # no bucket for this band; "disjoint" would be misleading).
        return "overlaps", None
    if max(f_ab, f_ba) > OVERLAP_MIN:
        return "overlaps", None
    return "disjoint", None


def _pair_plan(nodes: list[dict]) -> list[tuple[dict, dict]]:
    defaults = [n for n in nodes if "knob" not in n]
    variants = [n for n in nodes if "knob" in n]
    pairs = []
    for i, a in enumerate(defaults):
        for b in defaults[i + 1:]:
            pairs.append((a, b))
    for v in variants:
        for d in defaults:
            pairs.append((v, d))
    return pairs


def build(suite: str = "full", budget_s: float = DEFAULT_BUDGET_S) -> dict:
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    started = time.monotonic()
    try:
        from akt.core.analysis import api_metrics
        from akt.benchmark.suites import load_cases

        cases = load_cases(suite)
    except Exception as error:  # noqa: BLE001 - degrade gracefully
        return {
            "generated": generated,
            "unavailable": f"{type(error).__name__}: {str(error)[:200]}",
            "suite": suite,
            "nodes": [],
            "edges": [],
            "note": "api_metrics import or suite load failed on this host",
        }

    nodes, skipped = _collect_nodes(api_metrics, cases)
    if not nodes:
        return {
            "generated": generated,
            "unavailable": "no case traced on this host",
            "suite": suite,
            "nodes": [],
            "edges": [],
            "skipped": skipped,
            "note": "every default-config trace failed",
        }

    # Cap total nodes; defaults survive first so cross-case edges stay intact.
    ordered = ([n for n in nodes if "knob" not in n]
               + [n for n in nodes if "knob" in n])
    kept, dropped = ordered[:NODE_CAP], [n["id"] for n in ordered[NODE_CAP:]]

    edges: list[dict] = []
    edge_errors: list[str] = []
    truncated_pairs = 0
    pairs = _pair_plan(kept)
    for a, b in pairs:
        if time.monotonic() - started > budget_s:
            truncated_pairs += 1
            continue
        try:
            f_ab = _coverage_fraction(api_metrics, a, b)
            f_ba = _coverage_fraction(api_metrics, b, a)
        except Exception as error:  # noqa: BLE001 - skip the pair, keep going
            edge_errors.append(
                f"{a['id']}~{b['id']}: {type(error).__name__}: {str(error)[:80]}"
            )
            continue
        relation, superset = _classify(f_ab, f_ba)
        if relation == "generalizes" and superset == "b":
            a, b, f_ab, f_ba = b, a, f_ba, f_ab   # normalize: a is the superset
        edges.append({
            "a": a["id"],
            "b": b["id"],
            "relation": relation,
            "f_ab": round(f_ab, 4),
            "f_ba": round(f_ba, 4),
        })

    relations: dict[str, int] = {}
    for edge in edges:
        relations[edge["relation"]] = relations.get(edge["relation"], 0) + 1

    note = (
        "f_ab = fraction of a's substantive op mass reproducible from b alone "
        "(api_metrics.subtract_catalog two-pass delta); 'generalizes' means a "
        "reproduces b (a ⊇ b). Cross-case pairs restricted to case-default "
        f"nodes; variant nodes compare against case defaults only. suite={suite}."
    )
    if dropped:
        note += f" {len(dropped)} node(s) dropped by the {NODE_CAP}-node cap."
    if truncated_pairs:
        note += (f" {truncated_pairs}/{len(pairs)} pair(s) skipped by the "
                 f"{budget_s:.0f}s budget.")

    return {
        "generated": generated,
        "suite": suite,
        "nodes": [
            {k: v for k, v in node.items() if k != "graph"} for node in kept
        ],
        "edges": edges,
        "relations": relations,
        "dropped_nodes": dropped,
        "skipped": skipped,
        "edge_errors": edge_errors,
        "runtime_s": round(time.monotonic() - started, 2),
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="where to write coverage.json (default: akt/board/)")
    parser.add_argument("--suite", default="full", choices=("full", "fast"),
                        help="which case suite supplies the node set")
    parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S,
                        help="wall budget for the pairwise edge sweep")
    args = parser.parse_args()

    document = build(suite=args.suite, budget_s=args.budget_s)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, allow_nan=False, default=str))
    print("AKT_COVERAGE " + json.dumps({
        "n_nodes": len(document["nodes"]),
        "n_edges": len(document["edges"]),
        "relations": document.get("relations") or {},
        "unavailable": document.get("unavailable"),
        "runtime_s": document.get("runtime_s"),
        "out": str(out),
    }, default=str))


if __name__ == "__main__":
    main()
