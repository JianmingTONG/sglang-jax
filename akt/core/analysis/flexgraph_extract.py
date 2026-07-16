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

# The EXACT way a kernel author INVOKES each box — the real, module-qualified call
# form(s), NOT the internal `*_p` primitive object. jnp = jax.numpy, lax = jax.lax,
# pl = jax.experimental.pallas, pltpu = jax.experimental.pallas.tpu. Grounded against
# the installed jax 0.8.1 API (pl.load/store/swap are DEPRECATED -> `ref[idx]`; there is
# no pltpu.copy -> async_copy; DMA is make_async_copy(...).start()/.wait()). Keyed by the
# coarsened box label for FAMILY boxes (representative forms) and by prim-short for 1:1.
_USAGE = {
    # coarsened FAMILY boxes -> representative real call forms
    "elementwise arithmetic":  ["a + b", "a * b", "a - b", "a / b", "jnp.maximum(a, b)", "abs(a)", "a ** n"],
    "transcendental math":     ["jnp.exp(x)", "jnp.tanh(x)", "jnp.log(x)", "lax.rsqrt(x)", "jnp.sqrt(x)", "jnp.sin(x)"],
    "logic / compare / select":["jnp.where(cond, a, b)", "a == b", "a < b", "a & b", "a | b", "~a"],
    "dtype convert":           ["x.astype(dtype)", "lax.convert_element_type(x, dtype)", "jnp.clip(x, lo, hi)"],
    "reductions":              ["jnp.sum(x, axis)", "jnp.max(x, axis)", "jnp.min(x, axis)", "jnp.argmax(x, axis)"],
    "bit ops":                 ["x << n", "x >> n", "lax.population_count(x)"],
    # 1:1 boxes -> the exact invocation
    "dot_general":  ["jnp.matmul(a, b)", "jnp.dot(a, b)", "pl.dot(a, b)", "lax.dot_general(a, b, dims)"],
    "dma_start":    ["pltpu.make_async_copy(src, dst, sem).start()", "pltpu.async_copy(src, dst, sem)"],
    "dma_wait":     ["pltpu.make_async_copy(src, dst, sem).wait()"],
    "load":         ["ref[idx]                     # pl.load is deprecated"],
    "get":          ["ref[idx]                     # scalar/array ref read"],
    "program_id":   ["pl.program_id(axis)"],
    "num_programs": ["pl.num_programs(axis)"],
    "run_scoped":   ["pl.run_scoped(fn, *scratch_types)"],
    "multiple_of":  ["pl.multiple_of(x, n)"],
    "reciprocal":   ["pl.reciprocal(x)", "1.0 / x"],
    "roll":         ["pltpu.roll(x, shift, axis)"],
    "repeat":       ["pltpu.repeat(x, repeats, axis)"],
    "bitcast":      ["pltpu.bitcast(x, new_dtype)"],
    "semaphore_signal": ["pltpu.semaphore_signal(sem)"],
    "semaphore_wait":   ["pltpu.semaphore_wait(sem)"],
    "iota":         ["lax.broadcasted_iota(dtype, shape, dim)", "lax.iota(dtype, n)"],
    "gather":       ["x[idx]", "jnp.take(x, idx, axis)", "jnp.take_along_axis(x, idx, axis)"],
    "concatenate":  ["jnp.concatenate([a, b], axis)"],
    "broadcast_in_dim": ["jnp.broadcast_to(x, shape)"],
    "reshape":      ["x.reshape(shape)", "jnp.reshape(x, shape)"],
    "transpose":    ["x.T", "jnp.transpose(x, axes)", "jnp.swapaxes(x, i, j)"],
    "squeeze":      ["jnp.squeeze(x, axis)"],
    "split":        ["jnp.split(x, n, axis)"],
    "slice":        ["x[lo:hi]", "lax.slice(x, starts, limits)"],
    "pad":          ["jnp.pad(x, width)", "lax.pad(x, val, config)"],
    "axis_index":   ["lax.axis_index(axis_name)"],
    "cond":         ["lax.cond(pred, true_fn, false_fn, *ops)", "pl.when(pred)"],
    "while":        ["lax.while_loop(cond_fn, body_fn, init)"],
    "scan":         ["lax.scan(body_fn, init, xs)"],
    # full-view-only (never in the serving subgraph, but keep --full tidy)
    "swap":         ["ref[idx] = val               # pl.store / pl.swap are deprecated"],
    "get_barrier_semaphore": ["pltpu.get_barrier_semaphore()"],
    "delay":        ["pl.delay(cycles)"],
    "stop_gradient":["lax.stop_gradient(x)"],
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


# (c) The "gap" nodes (no user handle) are re-homed to the LEVEL where they are realized
# — HW is the floor, so there is no below-hardware band:
#   - Mosaic BACKEND (LLO) decisions: a genuine lowering pass between Mosaic IR and the
#     hardware (register/VREG allocation, vector-layout assignment, instruction
#     scheduling). No user handle -> hidden. Each Mosaic op is routed through the LLO
#     stage for its category:  mosaic op -> llo stage -> hw unit.
#   - Hardware-internal micro-behaviours: decided inside a unit (no handle) -> hidden
#     nodes AT the hw level, attached to their parent unit.
_LLO_FOR_CAT = {   # op category -> (llo stage label, category column)
    "layout":   ("vector-layout assignment / relayout insertion", "layout"),
    "compute":  ("reg-alloc + instruction scheduling (VLIW)", "compute"),
    "datamove": ("reg-alloc + instruction scheduling (VLIW)", "compute"),
    "sync":     ("reg-alloc + instruction scheduling (VLIW)", "compute"),
    "memory":   ("VMEM / register allocation", "memory"),
}
_HW_MICRO = {   # hw unit -> its internal, non-nameable micro-behaviours (hidden, at hw)
    "MXU": ["MXU push/pop staging"],
    "DMA engines": ["DMA channel selection", "HW queue assignment"],
    "VMEM": ["VMEM bank mapping", "contention arbitration"],
    "inter-core / ICI comm": ["ICI route selection"],
}
_HW_DESC = {   # short hardware descriptor per unit (shown in the click detail panel)
    "MXU": "systolic matrix-multiply array",
    "VPU": "vector ALU (8 sublanes x 128 lanes)",
    "scalar unit": "scalar core / SREG",
    "XLU": "cross-lane unit (shuffles / rotates / transposes / gathers)",
    "VREG": "vector register file", "Lanes": "128 lanes", "Sublanes": "8 sublanes",
    "HBM": "main memory", "VMEM": "vector scratchpad", "SMEM": "scalar memory",
    "CMEM": "constant / compute memory", "DMA engines": "async copy engines",
    "semaphore fabric": "sync-flag / barrier network",
    "inter-core / ICI comm": "inter-core / inter-chip (ICI) interconnect",
    "VMEM->VREG load": "vector load path", "VREG->VMEM store": "vector store path",
    "HBM->VMEM": "DMA transfer", "VMEM->HBM": "DMA transfer", "remote DMA": "cross-chip DMA",
    "semaphore memory": "semaphore memory",
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


# ============================ SERVING-STACK GAP MINING ============================
# Auto-derive the flexibility-gap frontier the AKT loop evolves against — WITHOUT a
# hand-authored gap list (cf. the frozen adapter.FRONTIER). Investigate the FULL
# serving stack (every Pallas kernel under python/sgl_jax/srt/kernels/**), find each
# kernel's structural tiling / pipeline / schedule axes, and classify every axis as
# NAMED (a live tuned table or config selector chooses it) or a GAP (fixed in the
# shipped path at a non-searched value: hardcoded literal / pinned to another axis /
# dead tuned table / missing table / backend-gated / unsearched schedule toggle).
# A gap = "the Pallas/Mosaic lowering CAN execute this axis but no config names it" —
# exactly a candidate capability. Grounded entirely in the source (AST).

# structural-axis name patterns (what a tile / pipeline / schedule knob is called)
_TILE_RE = re.compile(
    r"(^tile_|_tile$|^b[df]\d?$|^bt$|^bf$|^bse$|^btc$|^bd\d$|^b_(seq|inter)$|"
    r"^bkv(_sz|_csz)?$|^bq(_sz|_csz)?$|block_size|_block_size$|^chunk_size$|"
    r"^page_size$|num_slices_per_block|^B[KVQTMND]$|^num_\w+_per_block$)")
_PIPE_RE = re.compile(r"(buffer_count|num_stages|n_buffers|num_pipeline|double_buffer|pipeline_depth)")
_SCHED_RE = re.compile(r"(prefetch|overlap|interleave|^enable_|_mode$|reorder|_bank$|fuse)")
_TABLE_RE = re.compile(r"(TUNED|tuned|best_\w*config|_TABLE|BLOCK_SIZES|block_config)")
_SELECTOR_RE = re.compile(r"(get_\w*(block|tile|slice|config|size)|_select|choose_)")
_BACKEND_RE = re.compile(r"(tpu_version|device_name|device_kind|platform|generation|chip|v6e?|v7)")
_PALLAS_MARK = ("pallas_call", "pltpu", "emit_pipeline", "pl.Buffered", "make_kernel")

# gap category -> (human label, the reachable lowering primitive it corresponds to,
# salience for ranking). The primitive links the gap to a node in the lowering graph
# (the "actual connection": this config axis, if named, would steer THAT capability).
_GAP_CATEGORIES = {
    "pipeline-depth":       ("pipeline / double-buffering depth", "dma_start", 6),
    "compute-tile-pinned":  ("compute sub-tile pinned to load tile", "dot_general", 5),
    "missing-tuned-table":  ("tiles reachable but kernel ships no tuned table", "dot_general", 5),
    "dead-tuned-table":     ("tuned table exists but selector never consults it", "dot_general", 5),
    "schedule-toggle":      ("schedule/fusion toggle never searched", "dma_start", 4),
    "backend-gated":        ("tuned table gated to one TPU generation", "dot_general", 2),
    "shape-pinned-tile":    ("tile derived from input shape (not a knob)", "dot_general", 3),
}

# family label -> the akt runner kernel_id(s) that tune it (the labels whose kernel_id
# doesn't string-match the source directory). Kept explicit so `mla/v2` etc. don't get
# mis-attributed by loose substring matching.
_SUITE_ALIAS = {
    "ragged_paged_attention": {"rpa_v3"}, "simple_gla": {"gla"},
    "update_kv_cache": {"kv_cache"}, "fused_moe/v1": {"moe_v1"},
    "fused_moe/v2": {"moe_v2"}, "kda": {"kda"}, "fused_mlp": {"fused_mlp"},
    "megablox_gmm_kernel": {"gmm", "gmm_v2"}, "gmm": {"gmm", "gmm_v2"},
}


def _short_family(rel: str) -> str:
    """kernels/<...>/x.py -> a stable family label (dir-based, v1/v2 kept)."""
    parts = Path(rel).parts
    parts = parts[1:] if parts and parts[0] == "kernels" else parts
    segs = [p for p in parts[:-1] if p not in ("__pycache__",)]
    if not segs:                       # a top-level kernel file (e.g. fused_mlp.py)
        return Path(rel).stem
    if segs[-1] in ("v1", "v2") and len(segs) >= 2:
        return f"{segs[-2]}/{segs[-1]}"
    return segs[-1] if len(segs) == 1 else "/".join(segs[-2:]) if segs[-1] in ("v1", "v2") else segs[-1]


# benchmark / test / reference scaffolding — NOT the shipped serving kernel; mining
# these pollutes gap evidence with driver code, so they're skipped in discovery.
_SKIP_FILE = re.compile(r"(^bench|_bench|^test_|_test\.py$|^native\.py$|^ref\.py$|"
                        r"^perf\.py$|^bench_)")


def discover_kernel_families(kernels_dir: Path):
    """Walk the WHOLE kernels tree; group .py files into families and mark which
    families actually build a Pallas kernel. No hardcoded kernel list."""
    fam = defaultdict(lambda: {"files": [], "pallas": False, "dir": None})
    for f in sorted(kernels_dir.rglob("*.py")):
        rel = str(f.relative_to(kernels_dir.parent))
        if "/__pycache__/" in rel or f.name in ("__init__.py", "_pathways_compat.py") \
                or _SKIP_FILE.search(f.name):
            continue
        label = _short_family(rel)
        try:
            text = f.read_text()
        except Exception:  # noqa: BLE001
            continue
        rec = fam[label]
        rec["files"].append(f)
        rec["dir"] = f.parent
        if any(m in text for m in _PALLAS_MARK):
            rec["pallas"] = True
    return dict(fam)


def runner_named_axes(repo: Path):
    """Parse runner knobs and their stable programmer-control identifiers.

    A searched runner axis and an end-to-end exposed capability are intentionally
    distinct; this prevents benchmark-only tuning from closing a flexibility gap.
    """
    out = defaultdict(dict)
    rdir = repo / "akt/core/runners"
    for f in sorted(rdir.glob("*.py")) if rdir.is_dir() else []:
        try:
            tree = ast.parse(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        kids, knobs = set(), {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee(node.func) == "Knob" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    control = None
                    for keyword in node.keywords:
                        if (
                            keyword.arg == "programmer_control"
                            and isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, str)
                        ):
                            control = keyword.value.value
                    knobs[a0.value] = control
            if isinstance(node, ast.keyword) and node.arg == "kernel_id" \
                    and isinstance(node.value, ast.Constant):
                kids.add(node.value.value)
        for kid in (kids or {f.stem}):
            out[kid].update(knobs)
    return dict(out)


def _const_repr(node):
    """A short literal repr if `node` is a constant/None, else None."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name) and node.id in ("None", "True", "False"):
        return node.id
    return None


def _axis_dependencies(tree):
    """Derive coarse name-flow edges for low-level axis aliases.

    Kernel code commonly renames public arguments (``BT = chunk_size``) or forwards
    them to helper parameters (``block_size=intra_block_size``). Following those AST
    edges lets the miner relate implementation axes to runner/API names without a
    hand-authored family alias table.
    """
    dependencies = defaultdict(set)
    local_functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def static_refs(node, scope):
        """Names in a scalar/static expression, or None for runtime data flow."""
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            return {(scope, node.id)}
        if isinstance(node, ast.Constant):
            return set()
        if isinstance(
            node,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.UnaryOp,
                ast.IfExp,
                ast.Compare,
            ),
        ):
            result = set()
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, ast.expr):
                    continue
                child_refs = static_refs(child, scope)
                if child_refs is None:
                    return None
                result.update(child_refs)
            return result
        return None

    def target_names(node):
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            return set().union(*(target_names(item) for item in node.elts))
        return set()

    class DependencyVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = "<module>"

        def _visit_function(self, node):
            previous = self.scope
            self.scope = node.name
            for statement in node.body:
                self.visit(statement)
            self.scope = previous

        def visit_FunctionDef(self, node):  # noqa: N802
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            self._visit_function(node)

        def visit_Assign(self, node):  # noqa: N802
            sources = static_refs(node.value, self.scope)
            if sources is None:
                self.generic_visit(node)
                return
            for target in node.targets:
                for name in target_names(target):
                    dependencies[(self.scope, name)].update(
                        source for source in sources if source[1] != name
                    )
            self.generic_visit(node)

        def visit_AnnAssign(self, node):  # noqa: N802
            sources = static_refs(node.value, self.scope)
            if sources is None:
                self.generic_visit(node)
                return
            for name in target_names(node.target):
                dependencies[(self.scope, name)].update(
                    source for source in sources if source[1] != name
                )
            self.generic_visit(node)

        def visit_Call(self, node):  # noqa: N802
            callee = local_functions.get(_callee(node.func))
            if callee is not None:
                parameters = callee.args.posonlyargs + callee.args.args
                for parameter, argument in zip(parameters, node.args):
                    if not (_TILE_RE.search(parameter.arg) or _PIPE_RE.search(parameter.arg)):
                        continue
                    sources = static_refs(argument, self.scope)
                    if sources is not None:
                        dependencies[(callee.name, parameter.arg)].update(sources)
                for keyword in node.keywords:
                    if keyword.arg is None or not (
                        _TILE_RE.search(keyword.arg) or _PIPE_RE.search(keyword.arg)
                    ):
                        continue
                    sources = static_refs(keyword.value, self.scope)
                    if sources is not None:
                        dependencies[(callee.name, keyword.arg)].update(sources)
            self.generic_visit(node)

    DependencyVisitor().visit(tree)
    return dependencies


def _resolve_axis(axis, named_axes, dependencies, scope=None):
    """Return runner axes that transitively determine one implementation axis."""
    found = set()
    pending = (
        [(scope, axis)]
        if scope
        else [key for key in dependencies if key[1] == axis] or [("<module>", axis)]
    )
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        current_scope, name = current
        if name in named_axes:
            found.add(name)
        pending.extend(dependencies.get(current, ()))
        module_key = ("<module>", name)
        if current_scope != "<module>" and module_key in dependencies:
            pending.append(module_key)
    return found


def _mine_file(path: Path, text: str):
    """AST-mine ONE kernel file for structural axes + fixed-value gap signatures."""
    try:
        tree = ast.parse(text)
    except Exception:  # noqa: BLE001
        return {
            "axes": [],
            "findings": [],
            "tables": set(),
            "table_refs": set(),
            "dependencies": {},
        }
    lines = text.splitlines()
    axes, findings = [], []
    tables, table_refs = set(), set()

    # module-level tuned tables (dict literals whose name looks like a tuned table)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for t in node.targets:
                if isinstance(t, ast.Name) and _TABLE_RE.search(t.id):
                    tables.add(t.id)
    # every reference to a tuned-table name anywhere (for defined-but-unused)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and _TABLE_RE.search(node.id):
            table_refs.add(node.id)

    def _param_axis(fn, arg, default_node):
        nm = arg.arg
        cat = ("pipeline-depth" if _PIPE_RE.search(nm)
               else "schedule-toggle" if _SCHED_RE.search(nm)
               else "tile" if _TILE_RE.search(nm) else None)
        if not cat:
            return
        # debug/verbose/logging flags are not perf schedule knobs — drop the noise
        if cat == "schedule-toggle" and re.search(r"debug|verbose|logg?|dump|trace|profile|assert|mask_mode", nm):
            return
        axes.append({"axis": nm, "kind": cat, "fn": fn.name,
                     "line": arg.lineno, "default": _const_repr(default_node) if default_node else None})
        # schedule toggle with a literal default = a never-searched schedule knob
        if cat == "schedule-toggle" and default_node is not None and _const_repr(default_node):
            findings.append({"category": "schedule-toggle", "axis": nm, "fn": fn.name,
                             "line": arg.lineno, "detail": f"default={_const_repr(default_node)}"})

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        a = fn.args
        pos = a.posonlyargs + a.args
        defs = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        for arg, d in zip(pos, defs):
            _param_axis(fn, arg, d)
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            _param_axis(fn, arg, d)

        # (a) DEAD tuned table: an unconditional top-level return BEFORE the body
        # ever references a tuned table -> the table code is unreachable.
        if _SELECTOR_RE.search(fn.name):
            returned_at = None
            for stmt in fn.body:
                if isinstance(stmt, ast.Return) and returned_at is None:
                    returned_at = stmt.lineno
            if returned_at is not None:
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Name) and _TABLE_RE.search(sub.id) \
                            and getattr(sub, "lineno", 0) > returned_at:
                        findings.append({"category": "dead-tuned-table", "axis": sub.id,
                                         "fn": fn.name, "line": returned_at,
                                         "detail": f"selector returns at L{returned_at}; "
                                                   f"table {sub.id} referenced only after (unreachable)"})
                        break

        # (b) BACKEND-GATED table lookup: a tuned-table reference inside an `if`
        # whose test mentions the TPU version / device name / platform.
        for node in ast.walk(fn):
            if isinstance(node, ast.If):
                test_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                test_names |= {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
                if any(_BACKEND_RE.search(t) for t in test_names):
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Name) and _TABLE_RE.search(sub.id):
                            findings.append({"category": "backend-gated", "axis": sub.id,
                                             "fn": fn.name, "line": node.lineno,
                                             "detail": "table lookup guarded by TPU-generation / device test"})
                            break

        # (c) PINNED tile: `x = x if x is not None else y`  OR  `Bx = ref.shape[i]`.
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                tnames = [t.id for t in targets if isinstance(t, ast.Name)]
                for tn in tnames:
                    if not _TILE_RE.search(tn):
                        continue
                    v = node.value
                    if isinstance(v, ast.IfExp):     # A if A is not None else B
                        els = v.orelse
                        if isinstance(els, ast.Name):
                            findings.append({"category": "compute-tile-pinned", "axis": tn,
                                             "fn": fn.name, "line": node.lineno,
                                             "detail": f"defaults to load tile `{els.id}` when unset"})
                    elif isinstance(v, ast.Subscript) and isinstance(v.value, ast.Attribute) \
                            and v.value.attr == "shape":
                        base = getattr(v.value.value, "id", "input")
                        findings.append({"category": "shape-pinned-tile", "axis": tn,
                                         "fn": fn.name, "line": node.lineno,
                                         "detail": f"derived from {base}.shape (tied to input dims, not a knob)"})

    # (d) HARDCODED pipeline / tile literal at a call site (Buffered(buffer_count=3),
    # emit_pipeline(..., buffer_count=N), pallas_call(..., <pipe>=N)).
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = _callee(node.func)
            for kw in node.keywords:
                if kw.arg and (_PIPE_RE.search(kw.arg)) and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, (int, float)):
                    findings.append({"category": "pipeline-depth", "axis": kw.arg, "fn": fname,
                                     "line": node.lineno,
                                     "detail": f"{fname}({kw.arg}={kw.value.value}) — hardcoded literal"})
    return {
        "axes": axes,
        "findings": findings,
        "tables": tables,
        "table_refs": table_refs,
        "dependencies": _axis_dependencies(tree),
    }


def mine_serving_stack(sources):
    """Investigate the full serving stack and emit the auto gap frontier + categories.
    Returns {serving_stack, gaps, gap_categories}. Each gap is a reachable structural
    axis that the shipped config path fixes without naming — a candidate capability."""
    kernels_dir, repo = sources["kernels"], sources["repo"]
    families = discover_kernel_families(kernels_dir)
    runner_axes = runner_named_axes(repo)           # kernel_id -> knob -> API control
    named = {kernel_id: set(axes) for kernel_id, axes in runner_axes.items()}
    live = set(named)
    # family label -> the akt kernel_id(s) that tune it. Alias-first (exact), then a
    # STRICT stem/exact fallback — no loose substring matching (which cross-linked
    # every "*_v2" kernel_id to any "*/v2" family).
    def _suite_ids(label):
        if label in _SUITE_ALIAS:
            return set(_SUITE_ALIAS[label]) & live
        stem = label.split("/")[-1]
        return {kid for kid in live if kid == label or kid == stem}

    gaps = []
    stack = []
    for label, rec in sorted(families.items()):
        if not rec["pallas"]:
            continue
        ids = _suite_ids(label)
        fam_named = set().union(*(named[k] for k in ids if k in named)) if ids else set()
        fam_controls = {
            knob: control
            for kernel_id in ids
            for knob, control in runner_axes.get(kernel_id, {}).items()
            if control
        }
        merged = {
            "axes": [],
            "findings": [],
            "tables": set(),
            "table_refs": set(),
            "dependencies": defaultdict(set),
        }
        per_file = []                          # (relpath, mine-result) for per-file checks
        for f in rec["files"]:
            try:
                m = _mine_file(f, f.read_text())
            except Exception:  # noqa: BLE001
                continue
            rel = str(f.relative_to(repo))
            per_file.append((rel, m))
            merged["axes"] += [{**a, "file": rel} for a in m["axes"]]
            merged["findings"] += [{**g, "file": rel} for g in m["findings"]]
            merged["tables"] |= m["tables"]; merged["table_refs"] |= m["table_refs"]
            for axis, dependencies in m["dependencies"].items():
                merged["dependencies"][axis].update(dependencies)
        tile_axes = sorted({a["axis"] for a in merged["axes"] if a["kind"] == "tile"})
        has_table = bool(merged["tables"] or merged["table_refs"])
        stack.append({"family": label, "in_akt_suite": sorted(ids),
                      "tile_axes": tile_axes, "has_tuned_table": has_table,
                      "n_files": len(rec["files"])})

        seen = set()
        def _emit(cat, axis, detail, file, line, fn=None):
            key = (label, cat, axis)
            if key in seen:
                return
            seen.add(key)
            # elevated iff THIS family's own runner already searches the axis — use
            # fam_named only (a global check cross-links same-named knobs, e.g. moe's
            # `bt` would falsely mark grouped_topk's `bt` as elevated).
            parts = [p for p in re.split(r"[,\s]+", axis) if p]
            resolved = [
                _resolve_axis(part, fam_named, merged["dependencies"], fn)
                for part in parts
            ]
            elevated = bool(parts) and all(matches for matches in resolved)
            programmer_exposed = elevated and all(
                all(fam_controls.get(knob) for knob in matches)
                for matches in resolved
            )
            matched_knobs = sorted(set().union(*resolved)) if resolved else []
            controls = sorted(
                {fam_controls[knob] for knob in matched_knobs if fam_controls.get(knob)}
            )
            gaps.append({
                "id": f"{label}:{axis}:{cat}", "family": label, "kernel_ids": sorted(ids),
                "axis": axis, "category": cat, "status": cat,
                "what": _GAP_CATEGORIES.get(cat, (cat, "", 0))[0],
                "detail": detail, "evidence": f"{file}:{line}",
                "reachable_prim": _GAP_CATEGORIES.get(cat, (None, None, 0))[1],
                "elevated_in_akt": elevated,
                "programmer_exposed": programmer_exposed,
                "runner_axes": matched_knobs,
                "programmer_controls": controls,
                "exposure_level": (
                    "programmer" if programmer_exposed else "runner" if elevated else "none"
                ),
                "in_akt_suite": bool(ids)})

        for g in merged["findings"]:
            _emit(
                g["category"],
                g["axis"],
                g["detail"],
                g["file"],
                g["line"],
                g.get("fn"),
            )
        # missing-tuned-table: per FILE — an entry that constructs its own tiles
        # (TileSizes / calculate_tiling / >=2 tile axes) but references NO tuned table,
        # even when a sibling file in the same family ships one (the gmm_v2-vs-v1 case).
        for rel, m in per_file:
            ftiles = sorted({a["axis"] for a in m["axes"] if a["kind"] == "tile"})
            txt = ""
            try:
                txt = (repo / rel).read_text()
            except Exception:  # noqa: BLE001
                pass
            builds_tiles = ("TileSizes" in txt or "calculate_tiling" in txt or len(ftiles) >= 2)
            if builds_tiles and not m["table_refs"] and any(m2 for _, m2 in [(rel, m)]):
                sib_has = has_table and not m["table_refs"]
                _emit("missing-tuned-table", ",".join(ftiles[:3]) or "tiles",
                      (f"{Path(rel).name} constructs tiles but references no tuned table"
                       + (" (a sibling file in this family ships one)" if sib_has else "")),
                      rel, 0)

    # rank: gaps in the tuned suite first, then by category salience, then un-elevated
    def _rank(g):
        sal = _GAP_CATEGORIES.get(g["category"], (None, None, 0))[2]
        return (
            0 if g["in_akt_suite"] else 1,
            0 if not g["programmer_exposed"] else 1,
            -sal,
        )
    gaps.sort(key=_rank)
    cats = {}
    for c, (lab, prim, sal) in _GAP_CATEGORIES.items():
        n = sum(1 for g in gaps if g["category"] == c)
        if n:
            cats[c] = {"label": lab, "reachable_prim": prim, "salience": sal, "count": n}
    return {"serving_stack": stack, "gaps": gaps, "gap_categories": cats}


# validation: the auto-miner should REDISCOVER the frozen hand-authored FRONTIER
# (proof it is really investigating the stack, not re-encoding a list). Maps each
# hand gap to an (family-substr, category) the miner must have produced.
_FRONTIER_EXPECT = [
    ("update_kv_cache", "dead-tuned-table"),      # num_slices_per_block dead selector
    ("megablox_gmm_kernel", "missing-tuned-table"),  # gmm_v2 no table
    ("fused_mlp", "pipeline-depth"),              # buffer_count=3 hardcoded
    ("ragged_paged_attention", "compute-tile-pinned"),  # bkv_csz pinned
    ("fused_moe/v2", "schedule-toggle"),          # moe_v2 toggles
    ("simple_gla", "shape-pinned-tile"),          # BK/BV from ref.shape
    ("ragged_paged_attention", "backend-gated"),  # TUNED_BLOCK_SIZES_V3 gen-gated
]


def frontier_coverage(mined):
    got = {(g["family"], g["category"]) for g in mined["gaps"]}
    rows = []
    for famsub, cat in _FRONTIER_EXPECT:
        hit = any(famsub in f and c == cat for (f, c) in got)
        rows.append((famsub, cat, hit))
    return rows


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
    # exact API/ISA "handles" per node, surfaced in the board's click detail panel.
    api_for = {}                          # jax primitive short -> Pallas/JAX API tokens
    for _tok, _prims in _API_ALIAS.items():
        for _p in _prims:
            api_for.setdefault(_p, set()).add(_tok)
    prim_handle, op_handle = {}, {}       # node id -> exact primitive srcs / raw Mosaic ops

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
        _ph = prim_handle.setdefault(p_id, {"prim": set(), "usage": []})
        _ph["prim"] |= set(info["srcs"])
        for _u in _USAGE.get(pl, []):        # real invocation form(s) for this box
            if _u not in _ph["usage"]:
                _ph["usage"].append(_u)
        if fanout:
            nodes[p_id]["fanout"] = True
        for op in ops:
            cop = coarsen(op)
            ocat = categorize(cop)
            m_id = node("mosaic", cop, ocat, "nameable", active.get(prim))
            op_handle.setdefault(m_id, set()).add(op)
            low_edges.append([p_id, m_id])
            # route the op through the Mosaic BACKEND (LLO) stage for its category, then
            # onto its hardware unit(s):  mosaic op -> llo stage (hidden) -> hw unit.
            lname, lcat = _LLO_FOR_CAT.get(ocat, _LLO_FOR_CAT["compute"])
            l_id = node("llo", lname, lcat, "hidden", active.get(prim))
            low_edges.append([m_id, l_id]); gap_edges.append([m_id, l_id])
            for u in hw_units(op):     # classify HW from the ORIGINAL (specific) op
                h_id = node("hw", u, categorize(u), "nameable", active.get(prim))
                hw_seen.add(u)
                low_edges.append([l_id, h_id])
        if info["partial"]:
            nodes[p_id]["partial"] = True   # a corner Mosaic can't lower = an unexposed gap

    # Hardware-internal micro-behaviours: hidden nodes AT the hw level, attached to their
    # parent unit (the unit is nameable; the internal decision it makes is not).
    for unit, micros in _HW_MICRO.items():
        u_id = _nid("hw", unit)
        if u_id in node_ids:
            for mname in micros:
                # inherit the parent unit's ACTIVE set — a micro-behaviour is exercised
                # whenever its unit is (else it would look falsely "unused by the suite").
                x_id = node("hw", mname, nodes[u_id]["category"], "hidden", nodes[u_id]["active"])
                low_edges.append([u_id, x_id]); gap_edges.append([u_id, x_id])

    # de-dup edges
    low_edges = [list(e) for e in dict.fromkeys(map(tuple, low_edges))]
    gap_edges = [list(e) for e in dict.fromkeys(map(tuple, gap_edges))]
    # FOCUS: restrict to the subgraph the serving stack actually exercises — active
    # primitives + everything they lower into + the hidden gaps under those HW units.
    if focus_active:
        keep = {i for i, n in nodes.items() if n["active"]}
        # keep only the hidden gaps that hang off an ACTIVE node, so the shown floor is
        # exactly the part the serving stack reaches — not ops/gaps the suite never triggers
        # (those belong to the --full view). This keeps the focused view fully-exercised.
        for a, b in gap_edges:
            if a in keep:
                keep.add(b)
        nodes = {i: n for i, n in nodes.items() if i in keep}
        low_edges = [e for e in low_edges if e[0] in nodes and e[1] in nodes]
        gap_edges = [e for e in gap_edges if e[0] in nodes and e[1] in nodes]

    # attach the exact API/ISA HANDLE to each node (1:1 for single-API boxes, the member
    # list for coarsened families; hidden nodes get no handle — that absence IS the gap).
    for i, n in nodes.items():
        if n["nameability"] == "hidden":
            continue
        if n["layer"] == "pallas" and i in prim_handle:
            h = prim_handle[i]
            # `usage` = the real invocation form (headline); `prim` = the internal
            # lowering-primitive object (secondary — it is NOT how you call it).
            n["handle"] = {"usage": h["usage"], "prim": sorted(h["prim"])}
        elif n["layer"] == "mosaic" and i in op_handle:
            n["handle"] = {"op": sorted(op_handle[i])}
        elif n["layer"] == "hw":
            n["handle"] = {"hw": _HW_DESC.get(n["label"], n["label"])}

    hidden_ids = {n["id"] for n in nodes.values() if n["nameability"] == "hidden"}
    exposure = [[b, a] for (a, b) in low_edges if b not in hidden_ids]
    shown_prims = sum(1 for n in nodes.values() if n["layer"] == "pallas")

    cats = [("memory", "Memory & placement"), ("datamove", "Data movement / DMA"),
            ("compute", "Compute"), ("layout", "Layout & vectorization"),
            ("sync", "Synchronization & topology")]
    layers = [("pallas", "JAX / Pallas primitives"), ("mosaic", "Mosaic TPU IR ops"),
              ("llo", "Mosaic backend (LLO): reg-alloc / scheduling"), ("hw", "TPU hardware")]
    n_partial = sum(1 for n in nodes.values() if n.get("partial"))

    # (3) AUTO-DERIVE the gap frontier + categories from the FULL serving stack, and
    # CONNECT each gap to the reachable lowering node it would steer (config axis ->
    # the Pallas primitive that already lowers, but which no config names). This is
    # the loop's evolve target list, generated — not hand-authored.
    mined = mine_serving_stack(src)
    for g in mined["gaps"]:
        rp = g.get("reachable_prim")
        g["graph_node"] = f"pallas:{rp}" if rp and f"pallas:{rp}" in nodes else None
    coverage = frontier_coverage(mined)
    n_open = sum(
        1
        for g in mined["gaps"]
        if g["in_akt_suite"] and not g["programmer_exposed"]
    )
    return {
        "kind": "lowering-4layer", "generated_by": "flexgraph_extract.py (automated)",
        "categories": [{"id": c, "title": t} for c, t in cats],
        "layers": [{"id": l, "title": t} for l, t in layers],
        "nodes": list(nodes.values()),
        "lowering_edges": low_edges, "exposure_edges": exposure, "gap_edges": gap_edges,
        "n_gap": len(gap_edges), "n_hidden": len(hidden_ids), "n_partial": n_partial,
        # auto-mined serving-stack investigation + evolve frontier
        "serving_stack": mined["serving_stack"], "gaps": mined["gaps"],
        "gap_categories": mined["gap_categories"],
        "frontier_coverage": [{"family": f, "category": c, "hit": h} for (f, c, h) in coverage],
        "stats": {"jax_primitives_total": len(rules), "primitives_shown": shown_prims,
                  "mosaic_ops": sum(1 for n in nodes.values() if n["layer"] == "mosaic"),
                  "hw_units": sum(1 for n in nodes.values() if n["layer"] == "hw"),
                  "tpu_dialect_ops": len(tpu_ops_all),
                  "active_primitives": sum(1 for p in rules if active.get(p)),
                  "kernels_scanned": len({k for ks in active.values() for k in ks}),
                  "pallas_families": len(mined["serving_stack"]),
                  "auto_gaps": len(mined["gaps"]), "open_suite_gaps": n_open,
                  "frontier_rediscovered": f"{sum(h for _,_,h in coverage)}/{len(coverage)}",
                  "focus_active": focus_active},
        "note": ("AUTO-EXTRACTED: JAX/Pallas primitives + primitive→Mosaic-op edges parsed from "
                 "jax/_src/pallas/mosaic/lowering.py; ACTIVE marks the primitives the sglang-jax "
                 "kernels use. FOUR lowering levels, TPU hardware at the FLOOR: Pallas → Mosaic IR "
                 "→ Mosaic backend (LLO: reg-alloc / scheduling) → hardware. Nameability is BINARY: "
                 "NAMEABLE (green, reschedulable at the top) vs HIDDEN (red) — reachable in the "
                 "lowering but with no handle. The LLO level is hidden (the backend's decisions have "
                 "no user handle) and every op must traverse it; hardware micro-behaviours (MXU "
                 "staging, VMEM bank, DMA channel, ICI route) are hidden nodes AT the hw level. A "
                 "red directed edge into any hidden node = a flexibility gap."),
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
    print(f"[flexgraph-extract] serving stack: {s['pallas_families']} Pallas kernel families "
          f"investigated -> {s['auto_gaps']} flexibility gaps auto-derived "
          f"({s['open_suite_gaps']} OPEN in the akt suite) across {len(g['gap_categories'])} "
          f"categories; hand-FRONTIER rediscovered {s['frontier_rediscovered']}.")
    miss = [c for c in g["frontier_coverage"] if not c["hit"]]
    if miss:
        print("[flexgraph-extract] WARNING frontier gaps NOT rediscovered: "
              + ", ".join(f"{c['family']}/{c['category']}" for c in miss))
    print(f"[flexgraph-extract] -> {args.out}")


if __name__ == "__main__":
    main()
