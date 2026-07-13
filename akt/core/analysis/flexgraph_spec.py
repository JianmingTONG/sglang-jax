"""Flexibility graph SPEC — the TPU compiler lowering graph (jax-free data).

Nodes are the taxonomy across three layers (JAX/Pallas -> Mosaic TPU IR -> TPU
hardware) organized into five category columns (memory, data-movement, compute,
layout, sync), plus a bottom HIDDEN band for capabilities the compiler/hardware
control. Edges are LOWERING (top->bottom), verified against the jax 0.8.1
Pallas/Mosaic source and annotated with which sglang-jax kernels actually exercise
each (ACTIVE, from akt/core/analysis stack investigation 2026-07-13).

EXPOSURE (bottom->top) is implicit: every nameable (direct/grouped) node is exposed;
a LOWERING edge whose target is a HIDDEN node has no exposure counterpart -> that edge
is a FLEXIBILITY GAP (executable in the lowering, unnamed at the top). Nameability:
  direct  = explicit API / IR op / attribute / hardware-unit name  (green)
  grouped = one high-level name that fans out into several lower actions  (amber)
  hidden  = compiler/hardware-controlled, no Pallas or Mosaic handle  (red)
"""

CATS = [("memory", "Memory & placement"), ("datamove", "Data movement / DMA"),
        ("compute", "Compute"), ("layout", "Layout & vectorization"),
        ("sync", "Synchronization & topology")]
LAYERS = [("pallas", "JAX / Pallas"), ("mosaic", "Mosaic TPU IR"),
          ("hw", "TPU hardware"), ("hidden", "Hidden below Mosaic")]

# (id, label, layer, category, nameability)
NODES = [
    # ---------------- L1  JAX / Pallas ----------------
    ("p.hbm", "HBM", "pallas", "memory", "direct"),
    ("p.vmem", "VMEM", "pallas", "memory", "direct"),
    ("p.smem", "SMEM", "pallas", "memory", "direct"),
    ("p.cmem", "CMEM", "pallas", "memory", "direct"),
    ("p.blockspec", "BlockSpec", "pallas", "memory", "direct"),
    ("p.tiling", "Tiling", "pallas", "memory", "direct"),
    ("p.scratch", "Scratch allocation", "pallas", "memory", "direct"),
    ("p.sync_copy", "sync_copy", "pallas", "datamove", "direct"),
    ("p.async_copy", "async_copy", "pallas", "datamove", "direct"),
    ("p.make_async_copy", "make_async_copy", "pallas", "datamove", "direct"),
    ("p.emit_pipeline", "emit_pipeline", "pallas", "datamove", "grouped"),
    ("p.jax_arith", "JAX arithmetic", "pallas", "compute", "direct"),
    ("p.vpu_elem", "VPU elementwise", "pallas", "compute", "direct"),
    ("p.lax_dot", "lax.dot", "pallas", "compute", "grouped"),
    ("p.matmul", "Matrix multiplication", "pallas", "compute", "grouped"),
    ("p.transpose", "Transpose", "pallas", "layout", "grouped"),
    ("p.reshape", "Reshape", "pallas", "layout", "grouped"),
    ("p.broadcast", "Broadcast", "pallas", "layout", "grouped"),
    ("p.gather", "Gather", "pallas", "layout", "grouped"),
    ("p.roll", "Roll", "pallas", "layout", "grouped"),
    ("p.layout_ref", "Layout-sensitive refs", "pallas", "layout", "grouped"),
    ("p.semtype", "SemaphoreType", "pallas", "sync", "direct"),
    ("p.barriers", "Barriers", "pallas", "sync", "direct"),
    ("p.tcmesh", "TensorCoreMesh", "pallas", "sync", "direct"),
    ("p.dimsem", "dimension_semantics", "pallas", "sync", "direct"),
    ("p.coremap", "core_map", "pallas", "sync", "direct"),
    # ---------------- L2  Mosaic TPU IR ----------------
    ("m.memspace", "memory-space attrs", "mosaic", "memory", "direct"),
    ("m.tiled", "tiled layouts", "mosaic", "memory", "direct"),
    ("m.assume_layout", "assume_layout", "mosaic", "memory", "direct"),
    ("m.erase_layout", "erase_layout", "mosaic", "memory", "direct"),
    ("m.memrefs", "memrefs", "mosaic", "memory", "direct"),
    ("m.enqueue_dma", "enqueue_dma", "mosaic", "datamove", "direct"),
    ("m.wait_dma", "wait_dma", "mosaic", "datamove", "direct"),
    ("m.indirect_dma", "enqueue_indirect_dma", "mosaic", "datamove", "direct"),
    ("m.vload", "vector load", "mosaic", "datamove", "direct"),
    ("m.vstore", "vector store", "mosaic", "datamove", "direct"),
    ("m.iload", "indexed load", "mosaic", "datamove", "direct"),
    ("m.istore", "indexed store", "mosaic", "datamove", "direct"),
    ("m.arith", "arithmetic ops", "mosaic", "compute", "direct"),
    ("m.vecops", "vector ops", "mosaic", "compute", "direct"),
    ("m.matmul", "tpu.matmul", "mosaic", "compute", "direct"),
    ("m.push_rhs", "matmul_push_rhs", "mosaic", "compute", "hidden"),
    ("m.acc_lhs", "matmul_acc_lhs", "mosaic", "compute", "hidden"),
    ("m.pop", "matmul_pop", "mosaic", "compute", "hidden"),
    ("m.relayout", "relayout", "mosaic", "layout", "direct"),
    ("m.dyn_rotate", "dynamic_rotate", "mosaic", "layout", "direct"),
    ("m.gather", "gather", "mosaic", "layout", "direct"),
    ("m.scan", "scan", "mosaic", "layout", "direct"),
    ("m.sort", "sort", "mosaic", "layout", "hidden"),
    ("m.vreg_layout", "VREG layout", "mosaic", "layout", "direct"),
    ("m.lane_layout", "lane layout", "mosaic", "layout", "direct"),
    ("m.sublane_layout", "sublane layout", "mosaic", "layout", "direct"),
    ("m.sem_alloc", "sem_alloc", "mosaic", "sync", "direct"),
    ("m.sem_wait", "sem_wait", "mosaic", "sync", "direct"),
    ("m.sem_signal", "sem_signal", "mosaic", "sync", "direct"),
    ("m.barrier", "barrier", "mosaic", "sync", "direct"),
    ("m.device_id", "device_id", "mosaic", "sync", "direct"),
    ("m.core_id", "core_id", "mosaic", "sync", "direct"),
    ("m.subcore_id", "subcore_id", "mosaic", "sync", "direct"),
    # ---------------- L3  TPU hardware ----------------
    ("h.hbm", "HBM", "hw", "memory", "direct"),
    ("h.vmem", "VMEM", "hw", "memory", "direct"),
    ("h.smem", "SMEM", "hw", "memory", "direct"),
    ("h.cmem", "CMEM", "hw", "memory", "direct"),
    ("h.sem_mem", "semaphore memory", "hw", "memory", "direct"),
    ("h.dma_engines", "DMA engines", "hw", "datamove", "direct"),
    ("h.hbm2vmem", "HBM→VMEM", "hw", "datamove", "direct"),
    ("h.vmem2hbm", "VMEM→HBM", "hw", "datamove", "direct"),
    ("h.vmem2vreg", "VMEM→VREG load", "hw", "datamove", "direct"),
    ("h.vreg2vmem", "VREG→VMEM store", "hw", "datamove", "direct"),
    ("h.remote_dma", "remote DMA", "hw", "datamove", "direct"),
    ("h.vpu", "VPU", "hw", "compute", "direct"),
    ("h.mxu", "MXU", "hw", "compute", "direct"),
    ("h.scalar", "scalar unit", "hw", "compute", "direct"),
    ("h.vreg", "VREG", "hw", "layout", "direct"),
    ("h.lanes", "Lanes", "hw", "layout", "direct"),
    ("h.sublanes", "Sublanes", "hw", "layout", "direct"),
    ("h.xlu", "XLU", "hw", "layout", "direct"),
    ("h.sem_fabric", "semaphore fabric", "hw", "sync", "direct"),
    ("h.intercore_sync", "inter-core sync", "hw", "sync", "direct"),
    ("h.intercore_comm", "inter-core comm", "hw", "sync", "direct"),
    ("h.interchip_comm", "inter-chip comm", "hw", "sync", "direct"),
    # ---------------- HIDDEN below Mosaic ----------------
    ("x.vmembank", "VMEM bank mapping", "hidden", "memory", "hidden"),
    ("x.dmachan", "DMA channel selection", "hidden", "datamove", "hidden"),
    ("x.queue", "HW queue assignment", "hidden", "datamove", "hidden"),
    ("x.mxu_stage", "MXU push/pop staging", "hidden", "compute", "hidden"),
    ("x.insched", "instruction scheduling", "hidden", "compute", "hidden"),
    ("x.overlap", "cycle-level unit overlap", "hidden", "compute", "hidden"),
    ("x.encoding", "instruction encoding", "hidden", "compute", "hidden"),
    ("x.pipetiming", "pipeline timing", "hidden", "compute", "hidden"),
    ("x.regalloc", "physical VREG alloc", "hidden", "layout", "hidden"),
    ("x.spill", "register spill", "hidden", "layout", "hidden"),
    ("x.interconnect", "ICI route selection", "hidden", "sync", "hidden"),
    ("x.contention", "contention arbitration", "hidden", "sync", "hidden"),
]

# LOWERING edges (top->bottom), verified against jax 0.8.1 Pallas/Mosaic source.
LOWERING = [
    # ---- memory path ----
    ("p.hbm", "m.memspace"), ("p.vmem", "m.memspace"), ("p.smem", "m.memspace"),
    ("p.cmem", "m.memspace"), ("p.blockspec", "m.tiled"), ("p.tiling", "m.tiled"),
    ("p.scratch", "m.memrefs"), ("p.blockspec", "m.memrefs"),
    ("m.tiled", "m.vreg_layout"), ("m.assume_layout", "m.vreg_layout"),
    ("m.erase_layout", "m.tiled"),
    ("m.memspace", "h.hbm"), ("m.memspace", "h.vmem"), ("m.memspace", "h.smem"),
    ("m.memspace", "h.cmem"), ("m.memrefs", "h.vmem"), ("m.memrefs", "h.hbm"),
    ("m.tiled", "x.vmembank"), ("h.vmem", "x.vmembank"),
    # ---- data-movement path ----
    ("p.sync_copy", "m.enqueue_dma"), ("p.sync_copy", "m.wait_dma"),
    ("p.async_copy", "m.enqueue_dma"), ("p.async_copy", "m.wait_dma"),
    ("p.make_async_copy", "m.enqueue_dma"), ("p.make_async_copy", "m.wait_dma"),
    ("p.emit_pipeline", "m.enqueue_dma"), ("p.emit_pipeline", "m.wait_dma"),
    ("p.emit_pipeline", "m.vload"), ("p.emit_pipeline", "m.vstore"),
    ("p.gather", "m.indirect_dma"),
    ("p.layout_ref", "m.vload"), ("p.layout_ref", "m.vstore"),
    ("p.layout_ref", "m.iload"), ("p.layout_ref", "m.istore"),
    ("m.enqueue_dma", "h.dma_engines"), ("m.wait_dma", "h.dma_engines"),
    ("m.indirect_dma", "h.dma_engines"),
    ("h.dma_engines", "h.hbm2vmem"), ("h.dma_engines", "h.vmem2hbm"),
    ("h.dma_engines", "h.remote_dma"),
    ("m.vload", "h.vmem2vreg"), ("m.iload", "h.vmem2vreg"),
    ("m.vstore", "h.vreg2vmem"), ("m.istore", "h.vreg2vmem"),
    ("m.enqueue_dma", "x.dmachan"), ("h.dma_engines", "x.queue"),
    ("h.remote_dma", "x.interconnect"),
    # ---- compute path ----
    ("p.jax_arith", "m.arith"), ("p.vpu_elem", "m.vecops"),
    ("p.lax_dot", "m.matmul"), ("p.matmul", "m.matmul"),
    ("m.matmul", "m.push_rhs"), ("m.matmul", "m.acc_lhs"), ("m.matmul", "m.pop"),
    ("m.arith", "h.vpu"), ("m.arith", "h.scalar"), ("m.vecops", "h.vpu"),
    ("m.matmul", "h.mxu"),
    ("m.push_rhs", "x.mxu_stage"), ("m.acc_lhs", "x.mxu_stage"), ("m.pop", "x.mxu_stage"),
    ("h.mxu", "x.insched"), ("h.mxu", "x.overlap"), ("h.vpu", "x.overlap"),
    ("h.mxu", "x.pipetiming"), ("h.vpu", "x.encoding"),
    # ---- layout path ----
    ("p.transpose", "m.relayout"), ("p.reshape", "m.relayout"),
    ("p.broadcast", "m.vecops"), ("p.broadcast", "m.relayout"),
    ("p.gather", "m.gather"), ("p.roll", "m.dyn_rotate"),
    ("p.layout_ref", "m.vreg_layout"),
    ("m.relayout", "h.xlu"), ("m.dyn_rotate", "h.xlu"), ("m.gather", "h.xlu"),
    ("m.scan", "h.vpu"), ("m.sort", "h.xlu"),
    ("m.vreg_layout", "h.vreg"), ("m.lane_layout", "h.lanes"),
    ("m.sublane_layout", "h.sublanes"),
    ("h.vreg", "x.regalloc"), ("h.vreg", "x.spill"), ("h.xlu", "x.overlap"),
    # ---- sync path ----
    ("p.semtype", "m.sem_alloc"), ("p.barriers", "m.sem_wait"),
    ("p.barriers", "m.barrier"), ("p.dimsem", "m.core_id"),
    ("p.coremap", "m.core_id"), ("p.coremap", "m.subcore_id"),
    ("p.tcmesh", "m.device_id"), ("p.tcmesh", "m.core_id"),
    ("m.sem_alloc", "h.sem_mem"), ("m.sem_alloc", "h.sem_fabric"),
    ("m.sem_wait", "h.sem_fabric"), ("m.sem_signal", "h.sem_fabric"),
    ("m.barrier", "h.intercore_sync"),
    ("m.device_id", "h.interchip_comm"), ("m.core_id", "h.intercore_comm"),
    ("m.subcore_id", "h.intercore_comm"),
    ("h.sem_fabric", "h.intercore_sync"),
    ("h.interchip_comm", "x.interconnect"), ("h.sem_fabric", "x.contention"),
    ("h.intercore_comm", "x.contention"),
]

# ACTIVE: which sglang-jax kernels exercise each L1 node (from the stack investigation).
# Names normalized (simple_gla->gla, fused_moe_v2->moe_v2). Not-used L1 nodes omitted
# (CMEM, sync_copy, TensorCoreMesh are unused across all 8 kernels).
ACTIVE_L1 = {
    "p.hbm": ["fused_mlp", "moe_v2", "gmm", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.vmem": ["fused_mlp", "moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.smem": ["moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.blockspec": ["fused_mlp", "moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.tiling": ["fused_mlp", "moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "gla"],
    "p.scratch": ["fused_mlp", "moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.async_copy": ["moe_v2", "gmm", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.make_async_copy": ["moe_v2", "gmm", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.emit_pipeline": ["fused_mlp", "gmm"],
    "p.jax_arith": ["fused_mlp", "moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.vpu_elem": ["fused_mlp", "moe_v2", "gmm", "kda", "moe_v1", "rpa", "gla"],
    "p.lax_dot": ["moe_v2", "gmm", "kda", "moe_v1", "rpa", "gla"],
    "p.matmul": ["fused_mlp", "moe_v2", "gmm", "kda", "moe_v1", "rpa", "gla"],
    "p.transpose": ["kda", "kv_cache", "rpa", "gla"],
    "p.reshape": ["moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.broadcast": ["moe_v2", "gmm", "kda", "moe_v1", "rpa", "gla"],
    "p.gather": ["gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.roll": ["gmm"],
    "p.layout_ref": ["fused_mlp", "moe_v2", "gmm", "kda", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.semtype": ["moe_v2", "gmm", "kv_cache", "moe_v1", "rpa", "gla"],
    "p.barriers": ["moe_v2", "moe_v1"],
    "p.dimsem": ["fused_mlp", "gmm", "kda", "rpa", "gla"],
    "p.coremap": ["fused_mlp", "moe_v2", "kv_cache", "moe_v1"],
}


def build_graph():
    """Assemble the flexgraph JSON (nodes + lowering + exposure + gaps + active).
    ACTIVE is propagated DOWN the lowering edges: any node reachable from an active
    L1 node is active (exercised by the suite)."""
    nodes = [{"id": i, "label": lb, "layer": ly, "category": c, "nameability": nm}
             for (i, lb, ly, c, nm) in NODES]
    idset = {n["id"] for n in nodes}
    low = [[a, b] for (a, b) in LOWERING if a in idset and b in idset]
    hidden_ids = {n["id"] for n in nodes if n["nameability"] == "hidden"}
    # exposure = reverse of every lowering edge whose target is NOT hidden
    exposure = [[b, a] for (a, b) in low if b not in hidden_ids]
    gap_edges = [[a, b] for (a, b) in low if b in hidden_ids]
    # propagate active downward
    active = {i: set(ks) for i, ks in ACTIVE_L1.items()}
    adj = {}
    for a, b in low:
        adj.setdefault(a, []).append(b)
    frontier = list(active)
    while frontier:
        a = frontier.pop()
        for b in adj.get(a, []):
            before = active.get(b, set())
            new = before | active.get(a, set())
            if new != before:
                active[b] = new
                frontier.append(b)
    for n in nodes:
        n["active"] = sorted(active.get(n["id"], []))
    return {
        "kind": "lowering-3layer",
        "categories": [{"id": c, "title": t} for c, t in CATS],
        "layers": [{"id": l, "title": t} for l, t in LAYERS],
        "nodes": nodes,
        "lowering_edges": low,
        "exposure_edges": exposure,
        "gap_edges": gap_edges,
        "n_gap": len(gap_edges), "n_hidden": len(hidden_ids),
        "note": ("Lowering ▼ (top→bottom): JAX/Pallas → Mosaic TPU IR → TPU hardware; "
                 "every edge verified against jax 0.8.1 Pallas/Mosaic source. A node the "
                 "compiler/hardware controls with NO Pallas or Mosaic handle is HIDDEN; a "
                 "lowering edge INTO a hidden node has no exposure ▲ back to the top — the "
                 "FLEXIBILITY GAP (red). Green = nameable & bidirectionally lowered/exposed; "
                 "amber = one grouped name that fans out into several lower actions. Node "
                 "fill brightness = exercised by the sglang-jax kernel suite."),
    }
