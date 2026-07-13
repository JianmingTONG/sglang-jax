"""Benchmark ADAPTER — the execution/monitoring contract the AKT core consumes.

The AKT loop (akt/core/evolve/loop.py) never imports benchmark code; it shells out
to THIS file and reads one JSON line per subcommand, so the whole benchmark domain
(kernels, metrics, bottleneck, gap frontier) is swappable without touching core/.

CONTRACT
  adapter.py bottleneck  -> "AKT_BOTTLENECK {json}": {title, lines[], note}
      what dominates the suite latency (ranked kernels), pre-rendered for the QUERY.
  adapter.py gaps        -> "AKT_GAPS {json}":       {title, lines[], note}
      the flexibility-gap frontier: lowering-reachable tiling/pipeline/fusion knobs
      the kernel config schema does NOT currently name — candidate capabilities.
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

HERE = Path(__file__).resolve().parent          # akt/benchmark
ROOT = Path(__file__).resolve().parents[2]       # akt/
REPO = ROOT.parent                               # sglang-jax/
for p in (str(REPO), str(REPO / "python"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

EVAL_OUT = ROOT / "akt/optimization_history/.evolve_eval.json"


# ------------------------------------------------------------------ bottleneck
def cmd_bottleneck(_args):
    """Rank kernels by measured latency from the freshest gate eval. On this box
    those are Pallas-interpret proxies (functional, NOT TPU-faithful); on TPU they
    are the real device latencies. The dominant kernel is THE target to relieve."""
    if not EVAL_OUT.exists():
        print("AKT_BOTTLENECK " + json.dumps(
            {"missing": "no eval yet — run akt/benchmark/gates/eval.py --out "
                        f"{EVAL_OUT} (or the loop's `init`)"}))
        return
    s = json.loads(EVAL_OUT.read_text())
    rows = []
    for r in s.get("results", []):
        lat = r.get("forward_s") or r.get("default_s")
        rows.append((r.get("case"), lat, r.get("regime", "tpu-deferred"),
                     r.get("best_config"), r.get("search_note", "")))
    ran = [x for x in rows if x[1]]
    tot = sum(x[1] for x in ran) or 1.0
    ran.sort(key=lambda x: -x[1])
    lines = [f"{c}: {lat*1e3:.2f}ms ({round(100*lat/tot)}% suite) [{reg}] best={cfg}"
             for (c, lat, reg, cfg, _n) in ran]
    deferred = [x[0] for x in rows if not x[1]]
    if deferred:
        lines.append("tpu-deferred (wired, latency awaits a TPU): " + ", ".join(deferred))
    print("AKT_BOTTLENECK " + json.dumps(
        {"title": "kernel-suite latency share (interpret proxy on this box)",
         "lines": lines,
         "note": ("%suite = share of the runnable-suite latency (THE target). "
                  "Interpret latencies are functional proxies, not TPU-faithful — "
                  "the config RANKING may differ on TPU; treat the dominant kernel + "
                  "its structural bottleneck as the target, not the exact ms.")},
        allow_nan=False))


# ------------------------------------------------------------------ gap frontier
# The flexibility-gap frontier — the AI-domain analog of the FHE flexgap.json.
# Each entry is a lowering-reachable capability (the Mosaic/Pallas backend CAN
# execute it) that the kernel's config schema does NOT currently name/select. These
# are the candidate capabilities the oracle elevates into a runner DesignSpace Knob.
FRONTIER = [
    {"interface": "kv_cache: get_best_num_slices_per_block", "status": "unexposed",
     "what": "the shape-keyed tuned num_slices_per_block table is DEAD CODE — the "
             "selector early-returns 4 if page==1 else page_size. Elevate the block/"
             "grid tile (num_slices_per_block) into the search so it is chosen per shape."},
    {"interface": "gmm_v2: calculate_tiling (TileSizes)", "status": "unexposed",
     "what": "GMM v2 tiles via a VMEM auto-tiler with NO tuned table (v1 has one). "
             "Elevate an autotuned (tile_m,tile_k,tile_n) table for v2, per shape."},
    {"interface": "fused_mlp: apply_fused_mlp_sharded(buffer_count,b_seq,b_inter)",
     "status": "unexposed",
     "what": "emit_pipeline double-buffering depth is hardcoded buffer_count=3, and "
             "b_seq/b_inter have no tuned table/selector (caller-supplied). Elevate "
             "pipeline depth + a per-shape (b_seq,b_inter) table."},
    {"interface": "rpa_v3: decode block sweep (bq_csz,bkv_csz)", "status": "pinned",
     "what": "the shipped tuner pins the compute sub-tiles bq_csz=bq_sz, bkv_csz=bkv_sz; "
             "the kernel accepts them independently. Elevate independent (bq_csz,bkv_csz) "
             "so the search can decouple the load tile from the MXU compute tile."},
    {"interface": "moe_v2: fused_ep_moe_v2 toggles", "status": "unexposed",
     "what": "the None block_config falls to a crude hardcoded config, and the boolean "
             "schedule toggles (cross_expert_prefetch_mode, interleave_bt, "
             "enable_bt_scatter_overlap) are never searched. Elevate them as knobs."},
    {"interface": "gla/kda: chunk kernels BK/BV", "status": "pinned",
     "what": "BK=BV are pinned to the head dim (128) inside the chunk kernels; the "
             "state-update matmul could tile K/V. Elevate BK/BV tiling as knobs."},
    {"interface": "rpa_v3: tuned_block_sizes_v3 lookup", "status": "floor-only",
     "what": "the shipped RPA tuned table is consulted only on TPU v7; v6e falls to the "
             "heuristic. Backend/config gating (not a kernel knob) — flag to the user."},
]


def cmd_gaps(_args):
    lines = [f"[{f['interface'][:52]}] ({f['status']}) {f['what'][:150]}"
             for f in FRONTIER]
    print("AKT_GAPS " + json.dumps(
        {"title": "flexibility gaps: lowering-reachable knobs the config schema can't name",
         "lines": lines,
         "note": ("each gap = a capability the Pallas/Mosaic lowering executes but no "
                  "config knob names/selects -> candidate to elevate into a runner "
                  "DesignSpace Knob; 'pinned' = the axis exists but is tied to another; "
                  "'floor-only' = backend/config gating, frozen for the loop (flag to user).")},
        allow_nan=False))


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
        knobs = [(k.name, len(k.values), "+" + k.elevated_by if k.elevated_by else "base")
                 for k in c.space.knobs]
        print(f"  {c.case_id}: size={c.space.size()} knobs={knobs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bottleneck", help="emit AKT_BOTTLENECK json for the QUERY")
    sub.add_parser("gaps", help="emit AKT_GAPS json (flexibility-gap frontier)")
    ps = sub.add_parser("space", help="design-space size/knobs report (search evidence)")
    ps.add_argument("--kernel", default=None)
    args = ap.parse_args()
    {"bottleneck": cmd_bottleneck, "gaps": cmd_gaps, "space": cmd_space}[args.cmd](args)


if __name__ == "__main__":
    main()
