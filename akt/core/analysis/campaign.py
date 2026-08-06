"""Recursive campaign-diagram bottleneck diagnosis for the AKT loop.

Latency alone says WHICH kernel is slow; pairing it with compute+bandwidth
utilization proxies per level says WHY (campaign_diagram_tools style):

  cmp high / bw low   -> compute-bound   -> overlap/offload work into the load phase
  bw high  / cmp low  -> memory-bound    -> cut bytes moved / keep data resident
  both low            -> dependency-bound-> pipeline / overlap the stages

Levels, recursing into the dominant segment until the limiting resource is named:
  L0  serving-trace callsites (frozen model workloads, fast-suite cases)
  L1  kernel stages of the dominant callsite (top-level pallas_call/pjit segments)
  L2  engines of the dominant stage (MXU / VPU / DMA)
ending with a frontier-slot hint (flexgraph frontier_actions matching the
dominant kernel family).

All utilizations are PROXIES: masses come from the jaxpr operation graph
(weight_mode="size" = output element counts; bytes assume fp32), and there are
no absolute hardware peaks on this host — utils are normalized so the busiest
segment per level is 1.0.

CLI (run from the repo root with PYTHONPATH=python:.):
    python akt/core/analysis/campaign.py [--out akt/board/campaign.json]
prints the oracle-facing text block plus one "AKT_CAMPAIGN {json}" line.
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

EVAL = ROOT / "optimization_history/.evolve_eval.json"
FLEXGRAPH = ROOT / "core/analysis/flexgraph_generated.json"
DEFAULT_OUT = ROOT / "board/campaign.json"

BYTES_PER_ELEM = 4.0                            # fp32 proxy
CONTAINER_PRIMS = {"pallas_call", "pjit", "jit"}
_TEXT_TOP_SEGMENTS = 6                          # per-level lines in the query block


class CampaignUnavailable(RuntimeError):
    """The diagnosis inputs (graph tools / cases / latencies) are unusable."""


# ------------------------------------------------------------------ masses
# Fallback operation classifier, used only when api_metrics (concurrently being
# rewritten) does not yet export classify_labels. Same contract: one of
# {"compute", "memory", "container"} per label.
_MEMORY_PRIMS = {
    "slice", "dynamic_slice", "dynamic_update_slice", "gather", "scatter",
    "scatter-add", "scatter_add", "concatenate", "broadcast_in_dim", "reshape",
    "transpose", "squeeze", "expand_dims", "pad", "rev", "copy", "roll",
    "convert_element_type", "bitcast_convert_type", "iota", "get", "swap",
    "load", "store", "masked_load", "masked_store", "dma_start", "dma_wait",
    "device_put", "split", "gather_nd",
}


def _prim(label: str) -> str:
    return label.split("|", 1)[0]


def _fallback_classify(labels):
    out = []
    for label in labels:
        prim = _prim(label)
        if prim in CONTAINER_PRIMS or prim in (
            "scan", "while", "cond", "custom_vjp_call", "custom_jvp_call",
            "custom_vjp_call_jaxpr", "remat", "checkpoint", "closed_call",
            "core_call",
        ):
            out.append("container")
        elif prim in _MEMORY_PRIMS:
            out.append("memory")
        else:
            out.append("compute")
    return out


def _graph_tools():
    """(operation_graph, classify_fn, classifier_note). Raises CampaignUnavailable."""
    try:
        from akt.core.analysis import api_metrics
    except Exception as error:  # noqa: BLE001
        raise CampaignUnavailable(f"api_metrics import failed: {error}") from error
    operation_graph = getattr(api_metrics, "operation_graph", None)
    if operation_graph is None:
        raise CampaignUnavailable("api_metrics has no operation_graph")
    classify = getattr(api_metrics, "classify_labels", None)
    if classify is not None:
        return operation_graph, classify, "api_metrics.classify_labels"
    return operation_graph, _fallback_classify, "campaign fallback classifier"


def _masses(labels, weights, classes, indices=None):
    """(flop_mass, byte_mass) proxies over a node subset (default: all nodes)."""
    idx = range(len(labels)) if indices is None else indices
    flop = 0.0
    byte = 0.0
    for i in idx:
        if classes[i] == "compute":
            flop += weights[i]
        elif classes[i] == "memory":
            byte += weights[i] * BYTES_PER_ELEM
    return flop, byte


# ------------------------------------------------------------------ latencies
def _eval_latency_table() -> dict:
    """case_id -> {latency_s, source} from the freshest gate eval, if any."""
    try:
        summary = json.loads(EVAL.read_text())
    except Exception:  # noqa: BLE001
        return {}
    table = {}
    for case_id, result in (summary.get("case_search") or {}).items():
        rows = [
            row for row in result.get("measurements") or []
            if isinstance(row.get("latency_s"), (int, float)) and row["latency_s"] > 0
        ]
        if rows:
            best = min(rows, key=lambda row: row["latency_s"])
            table[case_id] = {"latency_s": float(best["latency_s"]),
                              "source": "eval case_search (best measured config)"}
    if table:
        return table
    for result in summary.get("results") or []:               # legacy schema
        latency = result.get("forward_s") or result.get("default_s")
        if isinstance(latency, (int, float)) and latency > 0:
            table[result.get("case")] = {"latency_s": float(latency),
                                         "source": "eval legacy results (best searched)"}
    return table


def _case_latency(case, table) -> tuple[float, str] | None:
    row = table.get(case.case_id)
    if row:
        return row["latency_s"], row["source"]
    try:                                                       # fallback: time here
        from akt.benchmark.runners.base import time_config
        timing = time_config(case, case.space.default_config(), iters=5)
        return float(timing["median_s"]), "live time_config iters=5 (default config)"
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ verdicts
_RULES = {
    "compute-bound": "cmp high / bw low -> overlap or offload work into the load phase",
    "memory-bound": "bw high / cmp low -> cut bytes moved / keep data resident",
    "dependency-bound": "both low -> pipeline / overlap the stages",
}


def _verdict(cmp_util: float, bw_util: float) -> str:
    if bw_util > 1.5 * cmp_util and bw_util > 0.4:
        return "memory-bound"
    if cmp_util > 1.5 * bw_util and cmp_util > 0.4:
        return "compute-bound"
    return "dependency-bound"


def _finish_segments(segments):
    """Fill share, normalized cmp/bw utils, and the verdict from raw rates."""
    total_lat = sum(s["_lat"] for s in segments) or 1.0
    max_cmp = max((s["_cmp_rate"] for s in segments), default=0.0) or 1.0
    max_bw = max((s["_bw_rate"] for s in segments), default=0.0) or 1.0
    for s in segments:
        s["share"] = s["_lat"] / total_lat
        s["cmp_util"] = s["_cmp_rate"] / max_cmp
        s["bw_util"] = s["_bw_rate"] / max_bw
        s["verdict"] = _verdict(s["cmp_util"], s["bw_util"])
        s["rule"] = _RULES[s["verdict"]]
        for key in ("_lat", "_cmp_rate", "_bw_rate"):
            s.pop(key, None)
    return segments


# ------------------------------------------------------------------ levels
def _trace_case(case, operation_graph):
    inputs = case.make_inputs()
    config = case.space.default_config()
    return operation_graph(lambda: case.run(inputs, config), weight_mode="size")


def _level0(cases, callsites_by_case, operation_graph, classify, table):
    """One segment per fast-suite serving-trace callsite."""
    segments = []
    graphs = {}
    skipped = []
    for case in cases:
        sites = callsites_by_case.get(case.case_id) or [case.case_id]
        latency = _case_latency(case, table)
        if latency is None:
            skipped.append(case.case_id)
            continue
        latency_s, source = latency
        graph = _trace_case(case, operation_graph)
        classes = list(classify(list(graph["labels"])))
        flop, byte = _masses(graph["labels"], graph["weights"], classes)
        graphs[case.case_id] = (graph, classes)
        for site in sites:
            segments.append({
                "name": site,
                "case": case.case_id,
                "latency_s": latency_s,
                "latency_source": source,
                "flop_mass": flop,
                "byte_mass": byte,
                "_lat": latency_s,
                "_cmp_rate": flop / latency_s,
                "_bw_rate": byte / latency_s,
            })
    return _finish_segments(segments), graphs, skipped


def _segment_graph(labels):
    """Split the flat node order into top-level container stages.

    operation_graph appends a container eqn (pallas_call / pjit-jit) and then
    recurses into its body, so a container's body nodes immediately FOLLOW it in
    node order: nodes after container k (up to the next container) are grouped
    into k's segment; leading non-container nodes group into the first following
    segment. Nested containers open new segments — a documented flattening
    approximation (the flat label list carries no nesting depth).
    """
    segments = []   # list of {"name", "indices"}
    pending = []
    for i, label in enumerate(labels):
        if _prim(label) in CONTAINER_PRIMS:
            segments.append({
                "name": f"#{len(segments) + 1} {label}",
                "prim": _prim(label),
                "indices": pending + [i],
            })
            pending = []
        elif segments:
            segments[-1]["indices"].append(i)
        else:
            pending.append(i)
    if pending:                                     # graph with no container at all
        segments.append({"name": "#1 (flat op stream)", "prim": "none",
                         "indices": pending})
    return segments


def _level1(dominant, graph, classes):
    """Stages of the dominant callsite; latency shares estimated from mass."""
    labels, weights = graph["labels"], graph["weights"]
    raw = _segment_graph(labels)
    segments = []
    for seg in raw:
        flop, byte = _masses(labels, weights, classes, seg["indices"])
        mass = flop + byte
        segments.append({
            "name": seg["name"],
            "prim": seg["prim"],
            "n_ops": len(seg["indices"]),
            "flop_mass": flop,
            "byte_mass": byte,
            "latency_share_basis": "estimated proportional to flop+byte mass",
            "_indices": seg["indices"],
            "_lat": mass,
            "_cmp_rate": (flop / mass) if mass else 0.0,
            "_bw_rate": (byte / mass) if mass else 0.0,
        })
    _finish_segments(segments)
    for seg in segments:                    # estimated absolute latency per stage
        seg["latency_s"] = dominant["latency_s"] * seg["share"]
    return segments


def _level2(l1_segment, graph, classes):
    """Engine shares inside the dominant stage: MXU / VPU / DMA."""
    labels, weights = graph["labels"], graph["weights"]
    mxu = vpu = dma = 0.0
    for i in l1_segment["_indices"]:
        if classes[i] == "compute":
            if "dot" in _prim(labels[i]):
                mxu += weights[i]
            else:
                vpu += weights[i]
        elif classes[i] == "memory":
            dma += weights[i] * BYTES_PER_ELEM
    total = (mxu + vpu + dma) or 1.0
    flop_total = (mxu + vpu) or 1.0
    engines = [
        ("MXU", mxu, mxu / flop_total, 0.0),
        ("VPU", vpu, vpu / flop_total, 0.0),
        ("DMA", dma, 0.0, 1.0 if dma else 0.0),
    ]
    segments = []
    for name, mass, cmp_util, bw_util in engines:
        verdict = _verdict(cmp_util, bw_util)
        segments.append({
            "name": name,
            "mass": mass,
            "share": mass / total,             # busy% proxy of the stage total
            "cmp_util": cmp_util,
            "bw_util": bw_util,
            "verdict": verdict,
            "rule": _RULES[verdict],
        })
    return segments


# ------------------------------------------------------------------ hints
def _slot_hints(kernel_id: str) -> list[str]:
    try:
        flexgraph = json.loads(FLEXGRAPH.read_text())
    except Exception:  # noqa: BLE001
        return []
    hints = []
    for slot in flexgraph.get("frontier_actions") or []:
        kernel_ids = slot.get("kernel_ids") or []
        if kernel_id in kernel_ids or kernel_id == slot.get("family"):
            gap_id = slot.get("gap_id")
            if gap_id:
                hints.append(gap_id)
    return hints


# ------------------------------------------------------------------ diagnosis
def _dominant(segments):
    return max(segments, key=lambda s: s["share"]) if segments else None


def diagnose() -> dict:
    """Full recursive diagnosis; returns an 'unavailable' record on any failure."""
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        operation_graph, classify, classifier_note = _graph_tools()
        from akt.benchmark.model_workloads import callsite_inventory
        from akt.benchmark.suites import load_cases

        cases = load_cases("fast")
        callsites_by_case: dict[str, list[str]] = {}
        for site, call in callsite_inventory().items():
            callsites_by_case.setdefault(call.case_id, []).append(site)

        table = _eval_latency_table()
        l0, graphs, skipped = _level0(
            cases, callsites_by_case, operation_graph, classify, table)
        if not l0:
            raise CampaignUnavailable(
                "no fast-suite callsite could be measured "
                f"(skipped: {', '.join(skipped) or 'none'})")

        dom0 = _dominant(l0)
        graph, classes = graphs[dom0["case"]]
        l1 = _level1(dom0, graph, classes)
        dom1 = _dominant(l1)
        l2 = _level2(dom1, graph, classes)
        dom2 = _dominant(l2)
        for seg in l1:                       # strip working indices from the output
            seg.pop("_indices", None)

        kernel_id = dom0["case"].split(":", 1)[0]
        hints = _slot_hints(kernel_id)
        limiter = {
            "callsite": dom0["name"],
            "case": dom0["case"],
            "kernel_id": kernel_id,
            "stage": dom1["name"],
            "engine": dom2["name"],
            "verdict": dom2["verdict"],
            "rule": dom2["rule"],
            "text": (
                f"{dom0['name']} -> {dom1['name']} -> {dom2['name']} "
                f"{dom2['verdict']} ({dom2['rule']})"
            ),
        }
        levels = [
            {"level": 0,
             "title": "L0 serving-trace callsites (fast suite)",
             "segments": l0,
             "dominant": dom0["name"],
             "note": ("latencies are real per-case measurements; utils normalized "
                      "so the busiest callsite = 1.0; tpu-deferred callsites "
                      "excluded on this host"
                      + (f"; skipped (no latency): {', '.join(skipped)}"
                         if skipped else ""))},
            {"level": 1,
             "title": f"L1 kernel stages of {dom0['name']}",
             "segments": l1,
             "dominant": dom1["name"],
             "note": ("stages = top-level pallas_call/pjit containers in node "
                      "order; latency shares ESTIMATED proportional to "
                      "flop+byte mass")},
            {"level": 2,
             "title": f"L2 engines of {dom1['name']}",
             "segments": l2,
             "dominant": dom2["name"],
             "note": "MXU = dot-primitive compute mass, VPU = other compute, "
                     "DMA = memory mass x4B; busy% = share of the stage total"},
        ]
        return {
            "generated": generated,
            "proxy": True,
            "classifier": classifier_note,
            "levels": levels,
            "limiter": limiter,
            "hints": hints,
        }
    except Exception as error:  # noqa: BLE001 - never crash the caller
        return {
            "generated": generated,
            "proxy": True,
            "unavailable": f"{type(error).__name__}: {str(error)[:220]}",
            "levels": [],
            "hints": [],
        }


# ------------------------------------------------------------------ rendering
def _pct(value) -> str:
    return f"{100.0 * value:.0f}%"


def render_query_block(diagnosis: dict) -> str:
    lines = ["== CAMPAIGN DIAGNOSIS (proxy compute/bandwidth utilization) =="]
    if diagnosis.get("unavailable"):
        lines.append(f"(campaign diagnosis unavailable: {diagnosis['unavailable']})")
        return "\n".join(lines)
    for level in diagnosis.get("levels", []):
        lines.append(f"{level['title']}:")
        ranked = sorted(level["segments"], key=lambda s: -s["share"])
        shown = ranked[:_TEXT_TOP_SEGMENTS]
        for seg in shown:
            mark = " *" if seg["name"] == level.get("dominant") else ""
            lines.append(
                f"  {seg['name']} share={_pct(seg['share'])} "
                f"cmp={_pct(seg['cmp_util'])} bw={_pct(seg['bw_util'])} "
                f"-> {seg['verdict']}{mark}"
            )
        rest = ranked[_TEXT_TOP_SEGMENTS:]
        if rest:
            lines.append(
                f"  (+{len(rest)} smaller segments, "
                f"share={_pct(sum(s['share'] for s in rest))} total)"
            )
    limiter = diagnosis.get("limiter") or {}
    hints = diagnosis.get("hints") or []
    lines.append(
        "LIMITER: " + (limiter.get("text") or "undetermined")
        + " -> candidate slots: " + (", ".join(hints) if hints else "(none matched)")
    )
    return "\n".join(lines)


def campaign_query_block() -> str:
    """Compact text block for the oracle QUERY (never raises)."""
    try:
        return render_query_block(diagnose())
    except Exception as error:  # noqa: BLE001
        return ("== CAMPAIGN DIAGNOSIS (proxy compute/bandwidth utilization) ==\n"
                f"(campaign diagnosis unavailable: {error})")


def _clean(obj):
    """JSON-safe: coerce non-finite floats to None (board.json convention)."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="where to write campaign.json (default: akt/board/)")
    args = parser.parse_args()
    diagnosis = _clean(diagnose())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnosis, indent=1, allow_nan=False))
    print(render_query_block(diagnosis))
    print("AKT_CAMPAIGN " + json.dumps(diagnosis, allow_nan=False))


if __name__ == "__main__":
    main()
