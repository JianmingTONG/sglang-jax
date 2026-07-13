#!/usr/bin/env python3
"""AUTOMATED flexibility-gap graph extractor.

Navigates the *current* serving stack on its own to discover the nodes and the
connections of the TPU compiler lowering graph — rather than hand-authoring them
(cf. the static flexgraph_spec.py). It reads THREE real sources:

  1. The serving-stack kernels  (python/sgl_jax/srt/kernels/**):
     AST + token scan -> which JAX/Pallas primitives each kernel actually uses
     (the ACTIVE set of the Pallas layer).
  2. The installed JAX Pallas->Mosaic lowering source (jax/_src/pallas/mosaic/
     lowering.py, primitives.py): parse the `register_lowering_rule(prim)` registry
     and scan each rule body for the Mosaic dialect ops it emits
     (tpu.* / arith.* / vector.* / memref.* / !tpu.* / #tpu.memory_space<...>).
     -> the JAX-primitive nodes, the Mosaic-op nodes, and the LOWERING edges between
     them, verified against the code. Rules that `raise NotImplementedError` are
     flagged as PARTIAL (an unexposed corner = a gap).
  3. The Mosaic `tpu` dialect module (jax.experimental.mosaic.dialects.tpu):
     enumerate the full op surface for completeness.

The two pieces that are NOT in the Python source — the Mosaic-op -> hardware-unit
mapping and the "capabilities hidden below Mosaic" (physical register/bank/route/
schedule decisions that have no IR handle at all) — are applied as small, explicit
RULE tables (architectural knowledge), clearly separated below. Everything else is
discovered from the stack.

EXPOSURE is bottom->top for every node that has an IR/API handle; a lowering edge
into a HIDDEN node (no handle) has no exposure -> that edge is a FLEXIBILITY GAP.

Run (under the project venv, which has jax):
    PYTHONPATH=python:. .venv/bin/python akt/core/analysis/flexgraph_extract.py \
        --out akt/core/analysis/flexgraph_generated.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------- locate
def locate_sources():
    """Find the installed Pallas/Mosaic lowering source + the serving-stack kernels
    by importing jax and walking file paths — no hardcoded venv path."""
    import jax._src.pallas.mosaic.lowering as _low
    mosaic_dir = Path(_low.__file__).parent
    repo = Path(__file__).resolve().parents[3]            # akt/core/analysis -> sglang-jax/
    kernels_dir = repo / "python/sgl_jax/srt/kernels"
    return {"lowering": mosaic_dir / "lowering.py",
            "primitives": mosaic_dir / "primitives.py",
            "mosaic_dir": mosaic_dir, "kernels": kernels_dir, "repo": repo}


# ----------------------------------------------------- (1) lowering rule parse
_DIALECTS = ("tpu", "arith", "vector", "math", "scf", "memref", "cf")
_OP_RE = re.compile(r"\b(" + "|".join(_DIALECTS) + r")\.([a-z_][A-Za-z0-9_]*)\s*\(")
_TYPE_RE = re.compile(r"!tpu\.([a-z_]+)")
_MEMSPACE_RE = re.compile(r"#tpu\.memory_space<([a-z_]+)>")


def _callee(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _prim_short(prim_src: str) -> str:
    """'lax.dot_general_p' -> 'dot_general'; 'tpu_primitives.dma_start_p' -> 'dma_start'."""
    base = prim_src.split(".")[-1]
    return base[:-2] if base.endswith("_p") else base


def _emitted_ops(text: str) -> set[str]:
    ops = set()
    for m in _OP_RE.finditer(text):
        ops.add(f"{m.group(1)}.{m.group(2)}")
    for m in _TYPE_RE.finditer(text):
        ops.add(f"tpu.!{m.group(1)}")
    for m in _MEMSPACE_RE.finditer(text):
        ops.add(f"tpu.memory_space<{m.group(1)}>")
    return ops


def extract_lowering(lowering_path: Path):
    """Parse lowering.py: return {prim_short: {"srcs":set, "ops":set, "partial":bool}}
    discovered from the register_lowering_rule registry (decorator + direct forms)."""
    src = lowering_path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    fn_body = {}          # fn name -> source text
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fn_body[node.name] = "\n".join(lines[node.lineno - 1: node.end_lineno])
    rules: dict[str, dict] = {}

    def add(prim_src, body):
        short = _prim_short(prim_src)
        r = rules.setdefault(short, {"srcs": set(), "ops": set(), "partial": False})
        r["srcs"].add(prim_src)
        r["ops"] |= _emitted_ops(body)
        if "NotImplementedError" in body or "not supported" in body or "unsupported" in body.lower():
            r["partial"] = True

    # decorator form:  @register_lowering_rule(PRIM, ...)\n def _x_lowering_rule(...)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and _callee(dec.func) == "register_lowering_rule" and dec.args:
                    add(ast.unparse(dec.args[0]), fn_body.get(node.name, ""))
    # direct form:  register_lowering_rule(PRIM)(_x_lowering_rule)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
                and _callee(node.func.func) == "register_lowering_rule" and node.func.args):
            prim_src = ast.unparse(node.func.args[0])
            fn = ast.unparse(node.args[0]) if node.args else ""
            add(prim_src, fn_body.get(fn.split(".")[-1], ""))
    return rules


# ------------------------------------------------- (3) mosaic tpu dialect ops
def discover_tpu_ops() -> set[str]:
    try:
        from jax.experimental.mosaic.dialects import tpu
        return {f"tpu.{n}" for n in dir(tpu)
                if not n.startswith("_") and (n[0].islower() or n.endswith("Op"))}
    except Exception:  # noqa: BLE001
        return set()


# -------------------------------------------------- (2) kernel usage scanner
# API token -> the JAX primitive short-name(s) it binds (so kernel usage of the
# high-level Pallas/JAX API marks the underlying primitive ACTIVE). Discovered
# correspondences; kept small + explicit.
_API_ALIAS = {
    "make_async_copy": ["dma_start", "dma_wait"], "async_copy": ["dma_start", "dma_wait"],
    "emit_pipeline": ["dma_start", "dma_wait"], "copy": ["dma_start", "dma_wait"],
    "dot": ["dot_general"], "matmul": ["dot_general"],
    "roll": ["roll", "dynamic_rotate"], "transpose": ["transpose"], "swapaxes": ["transpose"],
    "reshape": ["reshape"], "broadcast_to": ["broadcast_in_dim"],  # broadcast_to is Triton-only (raises on TPU)
    "broadcast": ["broadcast_in_dim"], "repeat": ["repeat"], "concatenate": ["concatenate"],
    "take_along_axis": ["gather"], "take": ["gather"], "gather": ["gather"],
    "iota": ["iota"], "semaphore_wait": ["semaphore_wait"], "semaphore_signal": ["semaphore_signal"],
    "exp": ["exp"], "tanh": ["tanh"], "where": ["select_n"], "sum": ["reduce_sum"],
    "max": ["reduce_max"], "min": ["reduce_min"], "maximum": ["max"], "reciprocal": ["reciprocal"],
}


def scan_kernel_usage(kernels_dir: Path, prim_shorts: set[str]):
    """For each kernel .py, find which primitive-shorts it exercises (directly by name
    or via an API alias). Returns {prim_short: sorted[kernel_name]}."""
    active = defaultdict(set)
    kernel_tag = {  # file-substring -> short kernel label
        "ragged_paged_attention_v3": "rpa", "fused_moe/v1": "moe_v1", "fused_moe/v2": "moe_v2",
        "gmm_v2": "gmm", "gmm.py": "gmm", "kda/kda": "kda", "simple_gla": "gla",
        "fused_mlp": "fused_mlp", "update_kv_cache": "kv_cache",
    }
    for f in kernels_dir.rglob("*.py"):
        rel = str(f.relative_to(kernels_dir.parent))
        tag = next((v for k, v in kernel_tag.items() if k in rel), None)
        if tag is None:
            continue
        text = f.read_text()
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        for prim in prim_shorts:
            if prim in toks:
                active[prim].add(tag)
        for api, prims in _API_ALIAS.items():
            if api in toks:
                for p in prims:
                    if p in prim_shorts:
                        active[p].add(tag)
    return {k: sorted(v) for k, v in active.items()}


# ---------------------------------------------- RULE TABLES (architectural glue)
# (a) category of a node, by keyword on its name (first match wins).
_CAT_RULES = [   # first match wins; order chosen so ops land in the right column
    ("datamove", r"dma|copy|prefetch|\bload\b|\bstore\b|strided_(load|store)|shuffled|vector\.(load|store)"),
    ("sync", r"sem_|semaphore|barrier|device_id|\bcore_id\b|subcore|signal|program_id|num_programs|axis_index|iteration_bound|delay"),
    ("memory", r"memref|memory_space|alloca|!tpu|tiled|assume_layout|erase_layout|scratch|run_scoped|multiple_of|\bget\b|\bswap\b|assume_multiple"),
    ("layout", r"relayout|rotate|transpose|reshape|shape_cast|broadcast|gather|concat|split|slice|sublane|lane|vreg|repeat|\bscan\b|sort|squeeze|shuffle|\bpad\b|iota"),
    ("compute", r"matmul|dot|arith|math|elementwise|transcendental|logic|reductions|bit ops|dtype|mul|add|sub|div|exp|tanh|reduce|reciprocal|select|cmp|convert|bitcast|prng|pack"),
]


def categorize(name: str) -> str:
    n = name.lower()
    for cat, pat in _CAT_RULES:
        if re.search(pat, n):
            return cat
    return "compute"


# (b) Mosaic op -> TPU hardware unit(s), by keyword (the mapping is not in Python).
_HW_RULES = [
    (r"matmul", ["MXU"]),
    (r"enqueue_dma|wait_dma|dma_start|dma_wait|enqueue_indirect|wait_indirect", ["DMA engines"]),
    (r"vector\.load|strided_load|shuffled_load|^tpu\.load|iload", ["VMEM→VREG load"]),
    (r"vector_store|strided_store|shuffled_store|^tpu\.store|istore", ["VREG→VMEM store"]),
    (r"relayout|rotate|transpose|gather|concat|sublane|lane_|shuffle|shape_cast|broadcast|repeat", ["XLU"]),
    (r"reduce|scan|all_reduce|reduce_index|reduction", ["VPU", "XLU"]),   # cross-lane part on XLU
    (r"arith\.|math\.|reciprocal|iota|select|cmpf|cmpi|pack|unpack|bitcast|prng|vector\.(?!load)", ["VPU"]),
    (r"sem_|barrier", ["semaphore fabric"]),
    (r"device_id|core_id|subcore", ["scalar unit", "inter-core / ICI comm"]),
    (r"iteration_bound|delay", ["scalar unit"]),   # loop bound / fixed stall — scalar, not comm
    (r"memory_space<hbm|!tpu.*hbm", ["HBM"]),
    (r"memory_space<vmem|!tpu.*vmem", ["VMEM"]),
    (r"memory_space<smem|memref\.load|memref\.store", ["SMEM / scalar unit"]),
    (r"memory_space<cmem", ["CMEM"]),
    (r"memory_space<semaphore|!tpu.semaphore", ["semaphore memory"]),
    (r"memref|alloca|tiled|assume_layout|erase_layout", ["VMEM"]),
]


def coarsen(op: str) -> str:
    """Collapse fine generic MLIR ops into families so the graph stays readable; keep
    the meaningful tpu.* Mosaic ops verbatim (matmul, enqueue_dma, relayout, sem_*, …)."""
    d, _, name = op.partition(".")
    if d == "arith":
        if name == "constant":
            return "arith.constant"
        if re.match(r"(add|sub|mul|div|maximum|minimum|max|min|neg|rem|abs)", name):
            return "arith.* arithmetic"
        if re.match(r"(cmp|select|and|or|xor|shl|shr|not)", name):
            return "arith.* logic/compare"
        if re.match(r"(ext|trunc|fpto|sito|uito|bitcast|index_cast)", name):
            return "arith.* convert"
        return "arith.* other"
    if d == "math":
        return "math.* transcendental"
    if d == "scf":
        return "scf.* control flow"
    if d == "cf":
        return "cf.* branch"
    if d == "vector":
        if re.match(r"(load|store)", name):
            return op
        if "reduction" in name:
            return "vector.multi_reduction"
        return "vector.* layout"
    if d == "memref":
        if re.match(r"(load|store)", name):
            return op
        return "memref.* alloc/view"
    return op                       # tpu.* and !tpu.* kept verbatim


_PRIM_FAMILY = {
    **{p: "elementwise arithmetic" for p in
       ("add", "sub", "mul", "div", "max", "min", "neg", "abs", "pow", "rem",
        "integer_pow", "sign", "floor", "ceil", "round", "nextafter", "square")},
    **{p: "transcendental math" for p in
       ("exp", "exp2", "log", "log1p", "tanh", "logistic", "rsqrt", "sqrt", "sin",
        "cos", "tan", "erf", "cbrt", "expm1")},
    **{p: "logic / compare / select" for p in
       ("eq", "ne", "lt", "gt", "le", "ge", "and", "or", "not", "xor", "select_n", "is_finite")},
    **{p: "dtype convert" for p in
       ("convert_element_type", "reduce_precision", "bitcast_convert_type", "clamp")},
    **{p: "reductions" for p in
       ("reduce_sum", "reduce_max", "reduce_min", "reduce_and", "reduce_or",
        "reduce_prod", "argmax", "argmin", "cumsum", "cummax", "cumlogsumexp")},
    **{p: "bit ops" for p in
       ("shift_left", "shift_right_logical", "shift_right_arithmetic", "population_count", "clz")},
}


def coarsen_prim(p: str) -> str:
    """Collapse the many elementwise/logic/reduction JAX primitives into families so
    the Pallas layer stays readable; structural ops (dot_general, dma_start, gather,
    roll, transpose, reshape, iota, semaphore_*, load, swap, …) are kept verbatim."""
    return _PRIM_FAMILY.get(p, p)


def hw_units(op: str):
    out = []
    for pat, units in _HW_RULES:
        if re.search(pat, op):
            for u in units:
                if u not in out:
                    out.append(u)
    return out


# (c) HIDDEN below Mosaic — capabilities with NO Pallas API and NO Mosaic op (by
# construction: these are compiler/hardware decisions, so they cannot be discovered
# from the IR; they are the architectural floor). Each maps under a category column.
_HIDDEN = [
    ("physical VREG alloc", "layout"), ("register spill", "layout"),
    ("shuffle insertion schedule", "layout"),
    ("vector-layout assignment / relayout insertion", "layout"),
    ("VMEM bank mapping", "memory"), ("DMA channel selection", "datamove"),
    ("HW queue assignment", "datamove"), ("MXU push/pop staging", "compute"),
    ("instruction scheduling", "compute"), ("instruction encoding", "compute"),
    ("cycle-level unit overlap", "compute"), ("automatic pipeline overlap", "compute"),
    ("ICI route selection", "sync"), ("contention arbitration", "memory"),
]
# which hardware unit each hidden capability sits under (source of the gap edge).
_HIDDEN_FROM = {
    "physical VREG alloc": "hw:VREG", "register spill": "hw:VREG",
    "shuffle insertion schedule": "hw:XLU",
    "vector-layout assignment / relayout insertion": "hw:XLU",
    "VMEM bank mapping": "hw:VMEM", "DMA channel selection": "hw:DMA engines",
    "HW queue assignment": "hw:DMA engines", "MXU push/pop staging": "hw:MXU",
    # global VLIW-bundle decisions govern ALL units, not one — anchor to a scheduler pseudo-unit
    "instruction scheduling": "hw:VLIW issue scheduler",
    "instruction encoding": "hw:VLIW issue scheduler",
    "cycle-level unit overlap": "hw:VLIW issue scheduler",
    "automatic pipeline overlap": "hw:VLIW issue scheduler",
    "ICI route selection": "hw:inter-core / ICI comm",
    "contention arbitration": "hw:VMEM",   # bank/crossbar/DMA bandwidth arbitration, not the sem fabric
}


# ------------------------------------------------------------------- assemble
def _nid(layer, name):
    return f"{layer}:{name}"


# Emissions the AST scan misses because the rule DELEGATES to a helper function
# (verified from the lowering source / the stack investigation). Kept minimal + cited.
_ENRICH = {
    "reduce_sum": ["vector.multi_reduction"], "reduce_max": ["vector.multi_reduction"],
    "reduce_min": ["vector.multi_reduction"],
    "argmax": ["tpu.reduce_index"], "argmin": ["tpu.reduce_index"],
    # run_scoped delegates its scratch alloc to _alloc_value (memref.alloca + tpu.sem_alloc),
    # which the rule-body AST scan doesn't see (lowering.py _alloc_value).
    "run_scoped": ["memref.alloca", "tpu.sem_alloc"],
}
# reduce_prod / cumsum / cummax / cumlogsumexp have NO lowering rule in this Mosaic build
# (verified: 0 occurrences) so they never enter `rules` — dropped from _ENRICH as dead.

# Pure tracing/plumbing primitives with no TPU-flexibility meaning — dropped.
_SKIP_PRIMS = {
    "jit", "pjit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
    "custom_vjp_call_jaxpr", "custom_transpose", "remat", "remat2", "checkpoint",
    "run_state", "debug_callback", "debug_print", "assert", "custom_root", "custom_linear_solve",
    "check", "reduce_precision",
    "broadcast_to",   # Triton-only primitive; its Mosaic rule raises RuntimeError (dead on TPU)
}


def build_graph(focus_active: bool = True):
    src = locate_sources()
    rules = {p: v for p, v in extract_lowering(src["lowering"]).items() if p not in _SKIP_PRIMS}
    for p, extra in _ENRICH.items():       # delegated emissions the AST scan misses
        if p in rules:
            rules[p]["ops"] |= set(extra)
    prim_shorts = set(rules)
    active = scan_kernel_usage(src["kernels"], prim_shorts)
    tpu_ops_all = discover_tpu_ops()

    nodes, node_ids = {}, set()

    def node(layer, name, cat, nameability, activek=None):
        i = _nid(layer, name)
        if i not in node_ids:
            node_ids.add(i)
            nodes[i] = {"id": i, "label": name, "layer": layer, "category": cat,
                        "nameability": nameability, "active": sorted(activek or [])}
        elif activek:
            nodes[i]["active"] = sorted(set(nodes[i]["active"]) | set(activek))
        return i

    low_edges, gap_edges = [], []
    hw_seen = set()

    # Pallas primitive nodes + primitive->mosaic-op edges + mosaic-op->hw edges.
    for prim, info in sorted(rules.items()):
        pl = coarsen_prim(prim)
        ops = sorted(info["ops"])
        # BINARY nameability: a capability is either NAMEABLE/reschedulable at the top
        # or HIDDEN (reachable in the lowering but not nameable). A "grouped" op like
        # lax.dot is itself nameable; the internals it fuses that you cannot reschedule
        # are separate HIDDEN nodes it lowers into (the red gap edges) — so there is no
        # third tier. `fanout` is kept only as a tooltip hint, not a class.
        fanout = (pl != prim or len({o.split(".")[0] for o in ops}) > 1 or len(ops) > 2)
        p_id = node("pallas", pl, categorize(pl), "nameable", active.get(prim))
        if fanout:
            nodes[p_id]["fanout"] = True
        for op in ops:
            cop = coarsen(op)
            m_id = node("mosaic", cop, categorize(cop), "nameable", active.get(prim))
            low_edges.append([p_id, m_id])
            for u in hw_units(op):     # classify HW from the ORIGINAL (specific) op
                h_id = node("hw", u, categorize(u), "nameable", active.get(prim))
                hw_seen.add(u)
                low_edges.append([m_id, h_id])
        if info["partial"]:
            # a primitive Mosaic only partially lowers = an unexposed corner (gap flag)
            nodes[p_id]["partial"] = True

    # Hidden-below-Mosaic capabilities + gap edges from the hardware units.
    for name, cat in _HIDDEN:
        x_id = node("hidden", name, cat, "hidden")
        frm = _HIDDEN_FROM.get(name, "")
        if frm.startswith("hw:"):
            unit = frm[3:]
            h_id = node("hw", unit, categorize(unit), "nameable")   # anchor even if unreached
            low_edges.append([h_id, x_id])
            gap_edges.append([h_id, x_id])

    # connect the pseudo-anchor units (they sit BELOW the execution units, which are all
    # governed by them) so their gaps are reached from the active compute path.
    _ANCHOR_FEED = {"VLIW issue scheduler": ("MXU", "VPU", "XLU", "DMA engines"),
                    "VREG": ("VPU", "MXU", "XLU")}
    for anchor, feeders in _ANCHOR_FEED.items():
        a_id = _nid("hw", anchor)
        if a_id in node_ids:
            for u in feeders:
                s = _nid("hw", u)
                if s in node_ids:
                    low_edges.append([s, a_id])

    # de-dup edges
    low_edges = [list(e) for e in dict.fromkeys(map(tuple, low_edges))]
    gap_edges = [list(e) for e in dict.fromkeys(map(tuple, gap_edges))]
    # FOCUS: restrict to the subgraph the serving stack actually exercises — active
    # primitives + everything they lower into + the hidden gaps under those HW units.
    if focus_active:
        keep = {i for i, n in nodes.items() if n["active"]}
        keep |= {i for i, n in nodes.items() if n["nameability"] == "hidden"}  # keep the floor
        for a, b in gap_edges:                     # + the HW anchor of each gap
            keep.add(a); keep.add(b)
        nodes = {i: n for i, n in nodes.items() if i in keep}
        low_edges = [e for e in low_edges if e[0] in nodes and e[1] in nodes]
        gap_edges = [e for e in gap_edges if e[0] in nodes and e[1] in nodes]
    hidden_ids = {n["id"] for n in nodes.values() if n["nameability"] == "hidden"}
    exposure = [[b, a] for (a, b) in low_edges if b not in hidden_ids]
    shown_prims = sum(1 for n in nodes.values() if n["layer"] == "pallas")

    cats = [("memory", "Memory & placement"), ("datamove", "Data movement / DMA"),
            ("compute", "Compute"), ("layout", "Layout & vectorization"),
            ("sync", "Synchronization & topology")]
    layers = [("pallas", "JAX / Pallas primitives"), ("mosaic", "Mosaic TPU IR ops"),
              ("hw", "TPU hardware"), ("hidden", "Hidden below Mosaic")]
    n_partial = sum(1 for n in nodes.values() if n.get("partial"))
    return {
        "kind": "lowering-3layer", "generated_by": "flexgraph_extract.py (automated)",
        "categories": [{"id": c, "title": t} for c, t in cats],
        "layers": [{"id": l, "title": t} for l, t in layers],
        "nodes": list(nodes.values()),
        "lowering_edges": low_edges, "exposure_edges": exposure, "gap_edges": gap_edges,
        "n_gap": len(gap_edges), "n_hidden": len(hidden_ids), "n_partial": n_partial,
        "stats": {"jax_primitives_total": len(rules), "primitives_shown": shown_prims,
                  "mosaic_ops": sum(1 for n in nodes.values() if n["layer"] == "mosaic"),
                  "hw_units": sum(1 for n in nodes.values() if n["layer"] == "hw"),
                  "tpu_dialect_ops": len(tpu_ops_all),
                  "active_primitives": sum(1 for p in rules if active.get(p)),
                  "kernels_scanned": len({k for ks in active.values() for k in ks}),
                  "focus_active": focus_active},
        "note": ("AUTO-EXTRACTED from the stack: JAX/Pallas primitives + primitive→Mosaic-op "
                 "edges parsed from jax/_src/pallas/mosaic/lowering.py; ACTIVE marks primitives "
                 "the sglang-jax kernels use; Mosaic→hardware + the hidden-below-Mosaic floor "
                 "are rule tables (not in code). Nameability is BINARY: a capability is either "
                 "NAMEABLE / reschedulable at the top (green, bidirectional) or HIDDEN — reachable "
                 "in the lowering but not reschedulable from the top (red directed edge into it "
                 "= the flexibility gap). A high-level op like lax.dot is nameable; the fused "
                 "internals it can't reschedule are the hidden nodes it lowers into."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="akt/core/analysis/flexgraph_generated.json")
    ap.add_argument("--full", action="store_true",
                    help="show the WHOLE lowering surface, not just the serving-stack subgraph")
    args = ap.parse_args()
    g = build_graph(focus_active=not args.full)
    Path(args.out).write_text(json.dumps(g, indent=1))
    s = g["stats"]
    print(f"[flexgraph-extract] parsed {s['jax_primitives_total']} JAX primitives with Mosaic "
          f"lowering rules ({s['tpu_dialect_ops']} tpu.* ops in the dialect); "
          f"{s['active_primitives']} exercised across {s['kernels_scanned']} serving kernels.")
    print(f"[flexgraph-extract] graph ({'FULL' if args.full else 'serving-stack focus'}): "
          f"{len(g['nodes'])} nodes ({s['primitives_shown']} primitives, {s['mosaic_ops']} Mosaic "
          f"ops, {s['hw_units']} HW units), {len(g['lowering_edges'])} lowering edges, "
          f"{g['n_gap']} gap edges into {g['n_hidden']} hidden capabilities.")
    print(f"[flexgraph-extract] -> {args.out}")


if __name__ == "__main__":
    main()
