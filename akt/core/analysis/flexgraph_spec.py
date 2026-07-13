"""Flexibility graph SPEC — the TPU compiler lowering graph (jax-free data).

Nodes are the taxonomy across three layers (JAX/Pallas -> Mosaic TPU IR -> TPU
hardware) organized into five category columns (memory, data-movement, compute,
layout, sync), plus a bottom HIDDEN band for capabilities the compiler/hardware
control. Edges are LOWERING (top->bottom), verified against the jax 0.8.1
Pallas/Mosaic source and annotated with which sglang-jax kernels actually exercise
each (ACTIVE, from akt/core/analysis stack investigation 2026-07-13).

EXPOSURE (bottom->top) is implicit: every NAMEABLE node is exposed; a LOWERING edge
whose target is a HIDDEN node has no exposure counterpart -> that edge is a FLEXIBILITY
GAP (reachable in the lowering, not reschedulable from the top). Nameability is BINARY:
  nameable = reschedulable at the top: an explicit API / IR op / attribute / HW name
             (green). A high-level op that fuses several actions is still nameable —
             the internals it can't reschedule are the HIDDEN nodes it lowers into.
  hidden   = compiler/hardware-controlled, no Pallas or Mosaic handle -> the gap (red).
"""

CATS = [("memory", "Memory & placement"), ("datamove", "Data movement / DMA"),
        ("compute", "Compute"), ("layout", "Layout & vectorization"),
        ("sync", "Synchronization & topology")]
LAYERS = [("pallas", "JAX / Pallas"), ("mosaic", "Mosaic TPU IR"),
          ("hw", "TPU hardware"), ("hidden", "Hidden below Mosaic")]

# (id, label, layer, category, nameability)
NODES = [
    # ---------------- L1  JAX / Pallas ----------------
    ("p.hbm", "HBM", "pallas", "memory", "nameable"),
    ("p.vmem", "VMEM", "pallas", "memory", "nameable"),
    ("p.smem", "SMEM", "pallas", "memory", "nameable"),
    ("p.cmem", "CMEM", "pallas", "memory", "nameable"),
    ("p.blockspec", "BlockSpec", "pallas", "memory", "nameable"),
    ("p.tiling", "Tiling", "pallas", "memory", "nameable"),
    ("p.scratch", "Scratch allocation", "pallas", "memory", "nameable"),
    ("p.sync_copy", "sync_copy", "pallas", "datamove", "nameable"),
    ("p.async_copy", "async_copy", "pallas", "datamove", "nameable"),
    ("p.make_async_copy", "make_async_copy", "pallas", "datamove", "nameable"),
    ("p.emit_pipeline", "emit_pipeline", "pallas", "datamove", "nameable"),
    ("p.jax_arith", "JAX arithmetic", "pallas", "compute", "nameable"),
    ("p.vpu_elem", "VPU elementwise", "pallas", "compute", "nameable"),
    ("p.lax_dot", "lax.dot", "pallas", "compute", "nameable"),
    ("p.matmul", "Matrix multiplication", "pallas", "compute", "nameable"),
    ("p.transpose", "Transpose", "pallas", "layout", "nameable"),
    ("p.reshape", "Reshape", "pallas", "layout", "nameable"),
    ("p.broadcast", "Broadcast", "pallas", "layout", "nameable"),
    ("p.gather", "Gather", "pallas", "layout", "nameable"),
    ("p.roll", "Roll", "pallas", "layout", "nameable"),
    ("p.layout_ref", "Layout-sensitive refs", "pallas", "layout", "nameable"),
    ("p.semtype", "SemaphoreType", "pallas", "sync", "nameable"),
    ("p.barriers", "Barriers", "pallas", "sync", "nameable"),
    ("p.tcmesh", "TensorCoreMesh", "pallas", "sync", "nameable"),
    ("p.dimsem", "dimension_semantics", "pallas", "sync", "nameable"),
    ("p.coremap", "core_map", "pallas", "sync", "nameable"),
    # ---------------- L2  Mosaic TPU IR ----------------
    ("m.memspace", "memory-space attrs", "mosaic", "memory", "nameable"),
    ("m.tiled", "tiled layouts", "mosaic", "memory", "nameable"),
    ("m.assume_layout", "assume_layout", "mosaic", "memory", "nameable"),
    ("m.erase_layout", "erase_layout", "mosaic", "memory", "nameable"),
    ("m.memrefs", "memrefs", "mosaic", "memory", "nameable"),
    ("m.enqueue_dma", "enqueue_dma", "mosaic", "datamove", "nameable"),
    ("m.wait_dma", "wait_dma", "mosaic", "datamove", "nameable"),
    ("m.indirect_dma", "enqueue_indirect_dma", "mosaic", "datamove", "nameable"),
    ("m.vload", "vector load", "mosaic", "datamove", "nameable"),
    ("m.vstore", "vector store", "mosaic", "datamove", "nameable"),
    ("m.iload", "indexed load", "mosaic", "datamove", "nameable"),
    ("m.istore", "indexed store", "mosaic", "datamove", "nameable"),
    ("m.arith", "arithmetic ops", "mosaic", "compute", "nameable"),
    ("m.vecops", "vector ops", "mosaic", "compute", "nameable"),
    ("m.matmul", "tpu.matmul", "mosaic", "compute", "nameable"),
    ("m.push_rhs", "matmul_push_rhs", "mosaic", "compute", "hidden"),
    ("m.acc_lhs", "matmul_acc_lhs", "mosaic", "compute", "hidden"),
    ("m.pop", "matmul_pop", "mosaic", "compute", "hidden"),
    ("m.relayout", "relayout", "mosaic", "layout", "nameable"),
    ("m.dyn_rotate", "dynamic_rotate", "mosaic", "layout", "nameable"),
    ("m.gather", "gather", "mosaic", "layout", "nameable"),
    ("m.scan", "scan", "mosaic", "layout", "nameable"),
    ("m.sort", "sort", "mosaic", "layout", "hidden"),
    ("m.vreg_layout", "VREG layout", "mosaic", "layout", "nameable"),
    ("m.lane_layout", "lane layout", "mosaic", "layout", "nameable"),
    ("m.sublane_layout", "sublane layout", "mosaic", "layout", "nameable"),
    ("m.sem_alloc", "sem_alloc", "mosaic", "sync", "nameable"),
    ("m.sem_wait", "sem_wait", "mosaic", "sync", "nameable"),
    ("m.sem_signal", "sem_signal", "mosaic", "sync", "nameable"),
    ("m.barrier", "barrier", "mosaic", "sync", "nameable"),
    ("m.device_id", "device_id", "mosaic", "sync", "nameable"),
    ("m.core_id", "core_id", "mosaic", "sync", "nameable"),
    ("m.subcore_id", "subcore_id", "mosaic", "sync", "nameable"),
    # ---------------- L3  TPU hardware ----------------
    ("h.hbm", "HBM", "hw", "memory", "nameable"),
    ("h.vmem", "VMEM", "hw", "memory", "nameable"),
    ("h.smem", "SMEM", "hw", "memory", "nameable"),
    ("h.cmem", "CMEM", "hw", "memory", "nameable"),
    ("h.sem_mem", "semaphore memory", "hw", "memory", "nameable"),
    ("h.dma_engines", "DMA engines", "hw", "datamove", "nameable"),
    ("h.hbm2vmem", "HBM→VMEM", "hw", "datamove", "nameable"),
    ("h.vmem2hbm", "VMEM→HBM", "hw", "datamove", "nameable"),
    ("h.vmem2vreg", "VMEM→VREG load", "hw", "datamove", "nameable"),
    ("h.vreg2vmem", "VREG→VMEM store", "hw", "datamove", "nameable"),
    ("h.remote_dma", "remote DMA", "hw", "datamove", "nameable"),
    ("h.vpu", "VPU", "hw", "compute", "nameable"),
    ("h.mxu", "MXU", "hw", "compute", "nameable"),
    ("h.scalar", "scalar unit", "hw", "compute", "nameable"),
    ("h.vreg", "VREG", "hw", "layout", "nameable"),
    ("h.lanes", "Lanes", "hw", "layout", "nameable"),
    ("h.sublanes", "Sublanes", "hw", "layout", "nameable"),
    ("h.xlu", "XLU", "hw", "layout", "nameable"),
    ("h.sem_fabric", "semaphore fabric", "hw", "sync", "nameable"),
    ("h.intercore_sync", "inter-core sync", "hw", "sync", "nameable"),
    ("h.intercore_comm", "inter-core comm", "hw", "sync", "nameable"),
    ("h.interchip_comm", "inter-chip comm", "hw", "sync", "nameable"),
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
                 "every edge verified against jax 0.8.1 Pallas/Mosaic source. Nameability is "
                 "BINARY: a capability is NAMEABLE/reschedulable at the top (green, "
                 "bidirectional) or HIDDEN — reachable in the lowering but not reschedulable "
                 "(red directed edge into it = the flexibility gap). Node fill brightness = "
                 "exercised by the sglang-jax kernel suite."),
    }
