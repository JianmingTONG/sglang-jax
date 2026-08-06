"""Delta-residual API-novelty guardrail (v2) for the AKT gate.

Start from the candidate program N, subtract everything the existing-API catalog
can reproduce, and judge the REMAINDER (Δ) on its own substance. Subtraction is
two-pass:

  PASS 1 — instance cover: catalog programs are hashed with typed input tokens
  (params → ⊥dtype·rank); every N op matching some catalog program's ENTRY
  micro-op opens a whole-instance embedding attempt (inputs may bind to ANY N
  values — a full embedding is itself the proof those ops are reproducible by
  calling that API). Matched instances are struck; the cover iterates to
  fixpoint, so chained / interleaved / intermediate-tapping compositions of
  existing APIs are recognized as duplicates.

  PASS 2 — per-op ancestry match: fixpoint Weisfeiler-Lehman signatures (one
  Merkle hash = an op's entire computation history) looked up in the union
  signature multiset of the catalog; found → struck, missing → Δ. Mismatch is
  the SOUND direction: everything left in Δ has a provably new derivation, and
  novelty is ancestry-closed (descendants of a new op are new).

The verdict reads Δ alone: per-class substantive mass (compute / MEMORY — the
roofline criterion: memory ops relocate bits across addresses, cost ∝ bytes;
compute ops produce new element values, cost ∝ FLOPs; classification is TOTAL,
containers recursed with zero self-weight, unknown primitives default to
compute + an `unclassified` audit list) against a MEASURED noise floor (p95 of
known one-knob tweaks' spurious residuals), a coherence tie-break (largest
connected Δ component), and a zero-embed self-check. δ = w(Δ)/w(N) is reported
as a composition reference only — the similarity ratio R ≡ 1−δ carries no
verdict and the v1 R/D scores are gone.

The genericity guardrail G_delta (upper-layer coverage) is unchanged from v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PALLAS_INTERPRET", "1")  # before any kernel import

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "akt/core/analysis/api_baseline.json"

DEFAULT_WEIGHT_MODE = "count"
DEFAULT_DELTA = 0.02
DEFAULT_COHERENCE_MIN = 3
DEFAULT_EPSILON_OPS = 4
DEFAULT_MIN_INSTANCE_OPS = 2
DEFAULT_NOISE_PERCENTILE = 95
DEFAULT_GENERICITY_PERCENTILE = 50
# The genericity cut never drops below covering HALF the relevant upper-layer
# patterns even when a proxy-host calibration degenerates.
DEFAULT_GENERICITY_FLOOR = 0.5
FIXPOINT_CAP = 96
COVER_MAP_CAP = 128

# ------------------------------------------------------------- classification
# Roofline criterion: MEMORY ⇔ relocate/replicate/reinterpret/select bits
# across addresses (cost ∝ bytes, no new element values); COMPUTE ⇔ new element
# values or lane-local transforms (cost ∝ FLOPs); CONTAINER ⇔ executes nothing
# itself (body recursed). Total: unknown primitives default to COMPUTE
# (fail-substantive — never dropped from Δ) and are logged for audit.

CONTAINER_PRIMS = frozenset(
    {
        "pjit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
        "custom_vjp_call_jaxpr", "remat", "checkpoint", "scan", "while", "cond",
        "pallas_call", "shard_map", "custom_partitioning", "xla_call",
    }
)
_MEMORY_TOKENS = (
    "dma", "copy", "swap", "gather", "scatter", "slice", "transpose",
    "broadcast", "reshape", "squeeze", "expand_dims", "bitcast", "concatenate",
    "pad", "rev", "roll", "masked_load", "masked_store", "get", "store",
    "load", "device_put",
)
_KNOWN_COMPUTE = frozenset(
    {
        "dot_general", "add", "add_any", "sub", "mul", "div", "rem", "neg",
        "exp", "exp2", "log", "log1p", "expm1", "tanh", "logistic", "erf",
        "erf_inv", "sin", "cos", "atan2", "sqrt", "rsqrt", "cbrt", "square",
        "abs", "sign", "floor", "ceil", "round", "clamp", "max", "min", "pow",
        "integer_pow", "cumsum", "cumlogsumexp", "cummax", "cummin", "cumprod",
        "reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and",
        "reduce_or", "argmax", "argmin", "select_n", "iota",
        "convert_element_type", "eq", "ne", "lt", "le", "gt", "ge", "and",
        "or", "xor", "not", "shift_left", "shift_right_logical",
        "shift_right_arithmetic", "is_finite", "nextafter", "population_count",
        "clz", "stop_gradient", "sort", "top_k", "erfc", "atanh", "asinh",
        "acosh", "sinh", "cosh", "tan", "asin", "acos", "atan", "real", "imag",
        "conj", "split", "log_sigmoid", "random_bits", "random_seed",
        "random_wrap", "random_unwrap", "threefry2x32", "reduce_precision",
    }
)


def classify_primitive(name: str) -> str:
    if name in CONTAINER_PRIMS:
        return "container"
    if any(token in name for token in _MEMORY_TOKENS):
        return "memory"
    return "compute"


def classify_labels(labels: list[str]) -> list[str]:
    """Total per-node classification from base labels (stable public seam)."""
    return [classify_primitive(label.split("|", 1)[0]) for label in labels]


def unclassified_primitives(labels: list[str]) -> list[str]:
    """Primitives that hit the compute DEFAULT rather than a known rule."""
    seen = []
    for label in labels:
        name = label.split("|", 1)[0]
        if (
            name not in CONTAINER_PRIMS
            and name not in _KNOWN_COMPUTE
            and not any(token in name for token in _MEMORY_TOKENS)
            and name not in seen
        ):
            seen.append(name)
    return seen


# ------------------------------------------------------------ operation graphs


def _sub_jaxprs(value):
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


def _aval_token(var, prefix: str) -> str:
    aval = getattr(var, "aval", None)
    dtype = getattr(aval, "dtype", "?")
    rank = len(getattr(aval, "shape", ()) or ())
    return f"{prefix}{dtype}r{rank}"


def _eqn_label(eqn) -> str:
    outs = ",".join(
        f"{getattr(v.aval, 'dtype', '?')}r{len(getattr(v.aval, 'shape', ()) or ())}"
        for v in eqn.outvars
    )
    return f"{eqn.primitive.name}|{outs}"


def _eqn_weight(eqn, mode: str) -> float:
    if eqn.primitive.name in CONTAINER_PRIMS:
        return 0.0  # containers execute nothing themselves; bodies are counted
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
    label_mode: str = "rank",  # retained for API stability; rank is the mode
) -> dict:
    """Dependency graph of every operation, at every nesting level.

    Returns {labels, weights, edges, in_tokens, contain}: edges are data
    dependencies within each jaxpr level; graph/sub-jaxpr inputs appear as
    TYPED tokens (⊥dtype·rank) folded into the consuming op's signature —
    that is the input-anonymization the instance cover relies on. `contain`
    edges (container → body op) exist for coherence only, never for ancestry.
    """
    import jax
    from jax._src import core as jcore

    closed = jax.make_jaxpr(trace_fn)()
    labels: list[str] = []
    weights: list[float] = []
    edges: list[tuple[int, int]] = []
    in_tokens: list[list[str]] = []
    contain: list[tuple[int, int]] = []

    def visit(jaxpr, parent: int | None):
        producer: dict = {}
        for eqn in jaxpr.eqns:
            index = len(labels)
            labels.append(_eqn_label(eqn))
            weights.append(_eqn_weight(eqn, weight_mode))
            tokens: list[str] = []
            for var in eqn.invars:
                if isinstance(var, jcore.Literal):
                    tokens.append(_aval_token(var, "L"))
                elif id(var) in producer:
                    edges.append((producer[id(var)], index))
                else:
                    tokens.append(_aval_token(var, "⊥"))
            in_tokens.append(tokens)
            if parent is not None:
                contain.append((parent, index))
            for var in eqn.outvars:
                producer[id(var)] = index
            for sub in _sub_jaxprs(eqn.params):
                visit(sub, index)

    visit(closed.jaxpr, None)
    return {
        "labels": labels,
        "weights": weights,
        "edges": edges,
        "in_tokens": in_tokens,
        "contain": contain,
    }


def _digest(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def signatures(graph: dict, *, iters: int | None = None) -> list[str]:
    """Fixpoint ancestry Weisfeiler-Lehman signatures.

    Base = own label + sorted typed input tokens; each round folds in the
    sorted parent-signature multiset; iterated until stable (Merkle hash of
    the full ancestry DAG). Different hash ⇒ provably different derivation.
    """
    n = len(graph["labels"])
    parents: list[list[int]] = [[] for _ in range(n)]
    for src, dst in graph["edges"]:
        parents[dst].append(src)
    sigs = [
        _digest(label + "|I:" + ",".join(sorted(tokens)))
        for label, tokens in zip(graph["labels"], graph["in_tokens"])
    ]
    cap = FIXPOINT_CAP if iters is None else iters
    for _ in range(cap):
        nxt = [
            _digest(sigs[i] + "|P:" + ",".join(sorted(sigs[p] for p in parents[i])))
            for i in range(n)
        ]
        if nxt == sigs:
            break
        sigs = nxt
    return sigs


# ------------------------------------------------------------- instance cover


def _parent_lists(graph: dict) -> list[list[int]]:
    parents: list[list[int]] = [[] for _ in graph["labels"]]
    for src, dst in graph["edges"]:
        parents[dst].append(src)
    return parents


def instance_cover(
    n_graph: dict,
    catalog: dict[str, dict],
    *,
    reverse: bool = False,
    min_instance_ops: int = DEFAULT_MIN_INSTANCE_OPS,
) -> tuple[set[int], list[dict]]:
    """PASS 1: strike whole embedded catalog programs out of N.

    Deterministic greedy cover: programs largest-first (then id), anchors in
    increasing N-node order (decreasing when ``reverse`` — the borderline
    retry), catalog program nodes matched in topological (index) order with
    min-index candidate choice. An instance's inputs bind to ANY N values;
    soundness comes from requiring the ENTIRE program to embed with exact
    per-op arity and mapped-parent consistency. Greedy failure only shrinks
    the cover (more Δ) — the strict, fail-closed direction.
    """
    n_labels = n_graph["labels"]
    n_parents = _parent_lists(n_graph)
    n_tokens = n_graph["in_tokens"]
    n_arity = [len(p) + len(t) for p, t in zip(n_parents, n_tokens)]
    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(n_labels):
        by_label.setdefault(label, []).append(i)

    classes_cache: dict[str, list[str]] = {}

    def substantive_ops(graph):
        key = id(graph)
        return sum(
            1 for c in classify_labels(graph["labels"]) if c != "container"
        )

    order = sorted(
        catalog.items(),
        key=lambda kv: (-len(kv[1]["labels"]), kv[0]),
    )
    struck: set[int] = set()
    instances: list[dict] = []

    for pid, e_graph in order:
        if substantive_ops(e_graph) < min_instance_ops:
            continue
        e_labels = e_graph["labels"]
        e_parents = _parent_lists(e_graph)
        e_tokens = e_graph["in_tokens"]
        e_arity = [len(p) + len(t) for p, t in zip(e_parents, e_tokens)]
        anchors = list(by_label.get(e_labels[0], []))
        if reverse:
            anchors = list(reversed(anchors))
        for anchor in anchors:
            if anchor in struck:
                continue
            mapping: dict[int, int] = {}
            used: set[int] = set()
            ok = True
            for k in range(len(e_labels)):
                needed = Counter(mapping[p] for p in e_parents[k])
                candidates = [anchor] if k == 0 else by_label.get(e_labels[k], [])
                if reverse and k > 0:
                    candidates = list(reversed(candidates))
                chosen = None
                for cand in candidates:
                    if cand in struck or cand in used:
                        continue
                    if n_arity[cand] != e_arity[k]:
                        continue
                    have = Counter(n_parents[cand])
                    if any(have[p] < c for p, c in needed.items()):
                        continue
                    # remaining (unmapped) operands of cand are free bindings —
                    # exactly the instance's ⊥ inputs; arity equality bounds them
                    chosen = cand
                    break
                if chosen is None:
                    ok = False
                    break
                mapping[k] = chosen
                used.add(chosen)
            if ok and len(mapping) == len(e_labels):
                struck |= set(mapping.values())
                instances.append(
                    {"program": pid, "anchor": anchor, "ops": len(mapping)}
                )
    return struck, instances


# ---------------------------------------------------------------- subtraction


def subtract_catalog(
    n_graph: dict,
    catalog: dict[str, dict],
    *,
    reverse: bool = False,
) -> dict:
    """Two-pass subtraction: instance cover, then per-op ancestry match.

    Returns the Δ record: per-class substantive masses, coherence, δ, the
    zero-embed self-check, and the minimized Δ subgraph.
    """
    struck, instances = instance_cover(n_graph, catalog, reverse=reverse)
    n_sigs = signatures(n_graph)
    budget: Counter = Counter()
    for e_graph in catalog.values():
        budget.update(signatures(e_graph))
    classes = classify_labels(n_graph["labels"])
    weights = n_graph["weights"]

    matched: set[int] = set()
    consumed: Counter = Counter()
    order = range(len(n_sigs)) if not reverse else reversed(range(len(n_sigs)))
    for i in order:
        if i in struck:
            continue
        sig = n_sigs[i]
        if budget[sig] > 0:
            budget[sig] -= 1
            consumed[sig] += 1
            matched.add(i)

    delta = [
        i
        for i in range(len(n_sigs))
        if i not in struck and i not in matched and classes[i] != "container"
    ]
    mass_cmp = sum(weights[i] for i in delta if classes[i] == "compute")
    mass_mem = sum(weights[i] for i in delta if classes[i] == "memory")
    total = sum(w for w, c in zip(weights, classes) if c != "container")
    # SELF-CHECK: a Δ op whose signature still has catalog budget left should
    # have been matched — nonzero means the subtraction itself is buggy.
    zero_embed = sum(1 for i in delta if budget[n_sigs[i]] > 0)

    delta_set = set(delta)
    adjacency: dict[int, set[int]] = {i: set() for i in delta}
    for src, dst in list(n_graph["edges"]) + list(n_graph["contain"]):
        if src in delta_set and dst in delta_set:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    components = 0
    largest = 0
    seen: set[int] = set()
    for start in delta:
        if start in seen:
            continue
        components += 1
        stack, size = [start], 0
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            size += 1
            stack.extend(adjacency[node] - seen)
        largest = max(largest, size)
    depth: dict[int, int] = {}
    for i in delta:  # index order is topological within each level
        preds = [
            src
            for src, dst in n_graph["edges"]
            if dst == i and src in delta_set
        ]
        depth[i] = 1 + max((depth.get(p, 0) for p in preds), default=0)
    max_depth = max(depth.values(), default=0)

    return {
        "struck_instances": instances,
        "n_struck": len(struck),
        "n_matched": len(matched),
        "delta_ops": len(delta),
        "mass": {"compute": mass_cmp, "memory": mass_mem},
        "total_mass": total,
        "delta_pct": ((mass_cmp + mass_mem) / total) if total else 0.0,
        "coherence": {
            "components": components,
            "largest": largest,
            "max_depth": max_depth,
        },
        "zero_embed": zero_embed,
        "unclassified": unclassified_primitives(
            [n_graph["labels"][i] for i in delta]
        ),
        "graph": {
            "labels": [n_graph["labels"][i] for i in delta],
            "classes": [classes[i] for i in delta],
            "edges": [
                [delta.index(src), delta.index(dst)]
                for src, dst in n_graph["edges"]
                if src in delta_set and dst in delta_set
            ],
        },
    }


# ------------------------------------------------------------ program helpers


def _config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def program_graph(case, config: dict, *, weight_mode: str = DEFAULT_WEIGHT_MODE) -> dict:
    inputs = case.make_inputs()
    return operation_graph(lambda: case.run(inputs, config), weight_mode=weight_mode)


def one_knob_programs(case, *, exclude_knobs: set[str] | None = None) -> dict[str, dict]:
    """Default + every single-knob deviation of one case ({program_id: config})."""
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


def _trace_many(case, programs: dict[str, dict], *, weight_mode: str) -> dict[str, dict]:
    graphs: dict[str, dict] = {}
    for pid, config in programs.items():
        try:
            graphs[pid] = program_graph(case, config, weight_mode=weight_mode)
        except Exception as error:  # noqa: BLE001 - untraceable → excluded, recorded
            graphs[pid] = {"error": f"{type(error).__name__}: {str(error)[:120]}"}
    return {pid: g for pid, g in graphs.items() if "error" not in g}


def catalog_for_case(
    cases: dict[str, object],
    case_id: str,
    *,
    exclude_knob: str | None = None,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
    cache: dict | None = None,
) -> dict[str, dict]:
    """⋃E for one candidate: the same case's one-knob population (minus the new
    knob) PLUS every other suite API's default program (cross-API cover)."""
    cache = cache if cache is not None else {}
    catalog: dict[str, dict] = {}
    case = cases[case_id]
    exclude = {exclude_knob} if exclude_knob else set()
    key = ("same", case_id, exclude_knob, weight_mode)
    if key not in cache:
        cache[key] = _trace_many(
            case, one_knob_programs(case, exclude_knobs=exclude), weight_mode=weight_mode
        )
    for pid, graph in cache[key].items():
        catalog[f"{case_id}:{pid}"] = graph
    for other_id, other in cases.items():
        if other_id == case_id:
            continue
        okey = ("default", other_id, weight_mode)
        if okey not in cache:
            cache[okey] = _trace_many(
                other,
                {"default": other.space.default_config()},
                weight_mode=weight_mode,
            )
        for pid, graph in cache[okey].items():
            catalog[f"{other_id}:{pid}"] = graph
    return catalog


def _percentiles(scores: list[float]) -> dict:
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


# ------------------------------------------------------------------ genericity
# (unchanged v1 guardrail — upper-layer coverage)


def genericity_from_case_search(
    case_search: dict,
    *,
    control_knob: str,
    relevant_case_to_callsites: dict[str, list[str]],
    delta: float = DEFAULT_DELTA,
) -> dict:
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
    weight_mode: str = DEFAULT_WEIGHT_MODE,
    delta: float = DEFAULT_DELTA,
    genericity_iters: int = 5,
) -> dict:
    """Calibrate both guardrails from the EXISTING APIs.

    Noise floor: run every one-knob tweak of every traceable case through the
    full two-pass subtraction against its own catalog — the residual substantive
    op counts are pure matcher noise, and their p95 is the floor a candidate's
    Δ must clear. Inter-API scale: pairwise Δ between different kernel families'
    default programs — what "as different as separate APIs" measures. Genericity
    percentiles as in v1.
    """
    import jax

    from akt.benchmark.runners.base import time_config

    cases = {case.case_id: case for case in _load_cases()}
    case_to_sites = _callsites_by_case()
    cache: dict = {}

    noise_masses: list[float] = []
    per_case_noise: dict[str, list[float]] = {}
    defaults: dict[str, dict] = {}
    kernel_of: dict[str, str] = {}
    for case_id, case in cases.items():
        kernel_of[case_id] = case.kernel_id
        traced = _trace_many(
            case, {"default": case.space.default_config()}, weight_mode=weight_mode
        )
        if "default" in traced:
            defaults[case_id] = traced["default"]

    for case_id, case in cases.items():
        key = ("same", case_id, None, weight_mode)
        cache[key] = _trace_many(case, one_knob_programs(case), weight_mode=weight_mode)
        programs = cache[key]
        if not programs:
            continue
        for pid, graph in programs.items():
            rest = {
                f"{case_id}:{other}": g
                for other, g in programs.items()
                if other != pid
            }
            for other_id, other_graph in defaults.items():
                if other_id != case_id:
                    rest[f"{other_id}:default"] = other_graph
            if not rest:
                continue
            record = subtract_catalog(graph, rest)
            mass = record["mass"]["compute"] + record["mass"]["memory"]
            noise_masses.append(mass)
            per_case_noise.setdefault(case_id, []).append(mass)

    inter_masses: list[float] = []
    inter_pairs: list[list] = []
    ids = sorted(defaults)
    for a in ids:
        for b in ids:
            if a >= b or kernel_of[a] == kernel_of[b]:
                continue
            record = subtract_catalog(defaults[a], {b: defaults[b]})
            mass = record["mass"]["compute"] + record["mass"]["memory"]
            inter_masses.append(mass)
            inter_pairs.append([a, b, mass])

    genericity_scores: dict[str, dict] = {}
    for case in cases.values():
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
        except Exception as error:  # noqa: BLE001
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
    for entry in genericity_scores.values():
        by_case = entry.get("G_by_case") or {}
        if by_case:
            entry["G"] = sum(by_case.values()) / len(by_case)
            g_values.append(entry["G"])

    baseline = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": jax.default_backend(),
        "pallas_interpret": os.environ.get("PALLAS_INTERPRET") == "1",
        "weight_mode": weight_mode,
        "delta": {
            "noise_floor": _percentiles(noise_masses),
            "noise_by_case": {
                case_id: _percentiles(values)
                for case_id, values in per_case_noise.items()
            },
            "inter_api": {**_percentiles(inter_masses), "pairs": inter_pairs},
            "epsilon_ops": DEFAULT_EPSILON_OPS,
            "coherence_min": DEFAULT_COHERENCE_MIN,
            "noise_percentile": DEFAULT_NOISE_PERCENTILE,
        },
        "genericity": {
            "per_control": genericity_scores,
            "global": _percentiles(g_values),
        },
        "policy": {
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


def delta_verdict(
    case,
    knob_name: str,
    selected_value,
    catalog: dict[str, dict],
    *,
    noise_floor: float,
    coherence_min: int,
    epsilon: float,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
) -> dict:
    """Judge one candidate program's Δ substance (with borderline retry)."""
    config = case.space.default_config()
    config[knob_name] = selected_value
    n_graph = program_graph(case, config, weight_mode=weight_mode)
    record = subtract_catalog(n_graph, catalog)
    mass = record["mass"]["compute"] + record["mass"]["memory"]
    exact_verified = None
    if abs(mass - noise_floor) <= epsilon:
        # borderline: deterministic retry with reversed cover/anchor order —
        # take the SMALLER Δ (most coverage found); certifies the region.
        retry = subtract_catalog(n_graph, catalog, reverse=True)
        retry_mass = retry["mass"]["compute"] + retry["mass"]["memory"]
        if retry_mass < mass:
            record, mass = retry, retry_mass
        exact_verified = True
    passes = (
        mass > noise_floor
        and record["coherence"]["largest"] >= coherence_min
        and record["zero_embed"] == 0
    )
    return {
        **record,
        "mass_total": mass,
        "noise_floor": noise_floor,
        "epsilon": epsilon,
        "coherence_min": coherence_min,
        "exact_verified": exact_verified,
        "pass": passes,
    }


def cover_map_for_case(
    case,
    knob_name: str,
    catalog: dict[str, dict],
    *,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
    cap: int = COVER_MAP_CAP,
) -> dict:
    """Per-config bidirectional exact cover: some config of N ≡ an existing API
    (signature multisets equal both ways — zero surplus either direction)."""
    catalog_sigs = {
        pid: Counter(signatures(graph)) for pid, graph in catalog.items()
    }
    result: dict[str, str] = {}
    space = case.space.deployment_space()
    for index, config in enumerate(space.enumerate(cap=cap)):
        try:
            graph = program_graph(case, config, weight_mode=weight_mode)
        except Exception:  # noqa: BLE001
            continue
        sigs = Counter(signatures(graph))
        for pid, e_sigs in catalog_sigs.items():
            if sigs == e_sigs:
                result[_config_key(config)] = pid
                break
        if index >= cap:
            break
    return result


def gate_metrics(
    *,
    baseline: dict,
    capability_controls: list[dict],
    case_search: dict,
    relevant_case_to_callsites: dict[str, list[str]],
    forwarding_paths: dict | None = None,
    delta: float | None = None,
) -> dict:
    """Both guardrails for one pending capability, on the pinned thresholds."""
    weight_mode = baseline.get("weight_mode", DEFAULT_WEIGHT_MODE)
    policy = baseline.get("policy") or {}
    dcfg = baseline.get("delta") or {}
    delta = (
        (baseline.get("genericity") or {}).get("delta", DEFAULT_DELTA)
        if delta is None
        else delta
    )
    global_floor = _threshold(
        dcfg.get("noise_floor") or {},
        dcfg.get("noise_percentile", DEFAULT_NOISE_PERCENTILE),
        2.0,
    )
    noise_by_case = dcfg.get("noise_by_case") or {}

    def floor_for(case_id: str) -> float:
        # Per-case calibration: a family whose mere re-parameterization already
        # restructures the unrolled program (e.g. kda's chunk loops) gets its own
        # measured floor; families whose tweaks leave zero residual get the
        # absolute minimum instead of a vacuous 0.
        case_floor = _threshold(
            noise_by_case.get(case_id) or {},
            dcfg.get("noise_percentile", DEFAULT_NOISE_PERCENTILE),
            global_floor,
        )
        return max(case_floor, float(dcfg.get("min_mass", DEFAULT_EPSILON_OPS)))

    epsilon = dcfg.get("epsilon_ops", DEFAULT_EPSILON_OPS)
    coherence_min = dcfg.get("coherence_min", DEFAULT_COHERENCE_MIN)
    cases = {case.case_id: case for case in _load_cases()}
    cache: dict = {}

    per_case = []
    errors = []
    worst = None
    delta_summary = None
    cover_map: dict = {}
    for record in capability_controls:
        case = cases.get(record["case_id"])
        knob_name = record["knob"]
        if case is None:
            errors.append(f"unknown case {record['case_id']!r}")
            continue
        try:
            catalog = catalog_for_case(
                cases,
                record["case_id"],
                exclude_knob=knob_name,
                weight_mode=weight_mode,
                cache=cache,
            )
            verdict = delta_verdict(
                case,
                knob_name,
                record["selected_value"],
                catalog,
                noise_floor=floor_for(record["case_id"]),
                coherence_min=coherence_min,
                epsilon=epsilon,
                weight_mode=weight_mode,
            )
        except Exception as error:  # noqa: BLE001 - fail closed
            errors.append(
                f"cannot evaluate {record['case_id']}: "
                f"{type(error).__name__}: {str(error)[:160]}"
            )
            continue
        entry = {
            **record,
            "mass_compute": verdict["mass"]["compute"],
            "mass_memory": verdict["mass"]["memory"],
            "mass_total": verdict["mass_total"],
            "largest_component": verdict["coherence"]["largest"],
            "zero_embed": verdict["zero_embed"],
            "exact_verified": verdict["exact_verified"],
            "pass": verdict["pass"],
        }
        per_case.append(entry)
        if worst is None or verdict["mass_total"] < worst["mass_total"]:
            worst = entry
            delta_summary = verdict
        if not cover_map:
            try:
                cover_map = cover_map_for_case(
                    case, knob_name, catalog, weight_mode=weight_mode
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"cover map failed: {str(error)[:120]}")

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

    redundancy_ok = bool(per_case) and all(r["pass"] for r in per_case)
    genericity_ok = bool(genericity) and g_min >= g_cut
    ok = redundancy_ok and genericity_ok and not errors
    return {
        "ok": ok,
        "redundancy_ok": redundancy_ok,
        "genericity_ok": genericity_ok,
        "delta": (
            {
                "mass": delta_summary["mass"],
                "total_ops": delta_summary["delta_ops"],
                "delta_pct": delta_summary["delta_pct"],
                "coherence": delta_summary["coherence"],
                "zero_embed": delta_summary["zero_embed"],
                "noise_floor": delta_summary["noise_floor"],
                "epsilon": epsilon,
                "exact_verified": delta_summary["exact_verified"],
                "unclassified": delta_summary["unclassified"],
                "graph": delta_summary["graph"],
                "per_case": per_case,
            }
            if delta_summary is not None
            else {"per_case": per_case, "noise_floor": global_floor}
        ),
        "cover_map": cover_map,
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
    gate.add_argument("--spec", required=True)
    gate.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.cmd == "profile":
        baseline = build_baseline(weight_mode=args.weight_mode, delta=args.delta)
        Path(args.out).write_text(json.dumps(baseline, indent=1))
        nf = baseline["delta"]["noise_floor"]
        ia = baseline["delta"]["inter_api"]
        print(
            f"[api-metrics] baseline -> {args.out}\n"
            f"[api-metrics] tweak-residual noise floor (substantive ops): "
            + (
                ", ".join(f"{k}={v:.1f}" for k, v in nf.items() if k.startswith("p"))
                if nf
                else "(no traceable tweaks)"
            )
            + f" (n={nf.get('n', 0)})\n"
            f"[api-metrics] inter-API Δ scale: "
            + (
                ", ".join(
                    f"{k}={v:.1f}"
                    for k, v in ia.items()
                    if k.startswith("p") and isinstance(v, (int, float))
                )
                if ia.get("n")
                else "(none)"
            )
            + "\n[api-metrics] AKT_API_BASELINE "
            + json.dumps(
                {
                    "fingerprint": baseline["fingerprint"],
                    "backend": baseline["backend"],
                }
            )
        )
        return

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
        d = result.get("delta") or {}
        print(
            "AKT_API_NOVELTY "
            + json.dumps(
                {
                    "ok": result["ok"],
                    "redundancy_ok": result["redundancy_ok"],
                    "genericity_ok": result["genericity_ok"],
                    "mass": d.get("mass"),
                    "noise_floor": d.get("noise_floor"),
                    "genericity_min": result["genericity_min"],
                }
            )
        )
        return

    cases = {case.case_id: case for case in _load_cases()}
    case = cases[args.case]
    knob_value = json.loads(args.value)
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    dcfg = baseline.get("delta") or {}
    catalog = catalog_for_case(cases, args.case, exclude_knob=args.knob)
    verdict = delta_verdict(
        case,
        args.knob,
        knob_value,
        catalog,
        noise_floor=_threshold(
            dcfg.get("noise_floor") or {}, DEFAULT_NOISE_PERCENTILE, 2.0
        ),
        coherence_min=dcfg.get("coherence_min", DEFAULT_COHERENCE_MIN),
        epsilon=dcfg.get("epsilon_ops", DEFAULT_EPSILON_OPS),
    )
    verdict.pop("graph", None)
    print(json.dumps(verdict, indent=1, default=str))


if __name__ == "__main__":
    main()
