"""AKT performance-evaluation gate for the sglang-jax kernel suite (FROZEN).

Per kernel case: run the autotuning SEARCH over the (possibly capability-enlarged)
design space — an exhaustive enumerate-correct-time-argmin, optimal within the
space — then gate on numerical correctness vs the kernel's pure-JAX reference and
report the searched-best latency. The incumbent baseline is the DEFAULT/shipped
config's measured latency (the "current implementation profiled on this machine").

Objective = geomean of the searched-best latencies over the runnable cases.
`geomean_default_s` = geomean of the default-config latencies (the incumbent set at
init). Cases that don't execute here (tpu-deferred) are wired + reference-checked
but excluded from the local geomean (they rejoin on TPU).

This file is part of the frozen harness: the AKT loop must not edit it.

Usage:
  python akt/benchmark/gates/eval.py --suite fast --runs 20 --out results.json
"""
from __future__ import annotations

# Force Pallas interpret so interpret-capable kernels run on this non-TPU box.
# Must precede any kernel import (get_interpret() reads this env).
import os
os.environ.setdefault("PALLAS_INTERPRET", "1")

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]           # akt/
REPO = ROOT.parent                                    # sglang-jax/
for p in (str(REPO), str(REPO / "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax  # noqa: E402
from akt.benchmark.suites import load_cases  # noqa: E402
from akt.benchmark.runners.base import (  # noqa: E402
    check_correct, search_best, time_config,
)


def eval_case(case, runs: int) -> dict:
    out = {"case": case.case_id, "kernel": case.kernel_id, "shape": case.shape_id,
           "space_size": case.space.size(), "note": case.note}
    default_cfg = case.space.default_config()
    # 1) does this case execute here? (probe the default config)
    ok, why = check_correct(case, default_cfg)
    if not ok:
        out.update(regime="tpu-deferred", correct=None, forward_s=None,
                   default_s=None, best_config=None, reason=why,
                   search_note=f"tpu-deferred ({why[:60]})")
        return out
    regime = jax.default_backend() + ("-interpret" if os.environ.get("PALLAS_INTERPRET") == "1" else "")
    out["regime"] = regime
    # 2) search the (possibly enlarged) design space
    try:
        r = search_best(case, iters=runs)
    except Exception as e:  # noqa: BLE001
        out.update(correct=False, forward_s=None, default_s=None,
                   error=repr(e)[:200], search_note="search failed")
        return out
    best_cfg = r["best_config"]
    correct = best_cfg is not None
    out.update(correct=correct, forward_s=r["best_median_s"],
               default_s=r["default_median_s"], best_config=best_cfg,
               default_config=default_cfg, n_correct=r["n_correct"],
               n_valid=r["n_valid"])
    # search evidence (the "enlarged space was searched, not sampled" proof)
    speedup = (r["default_median_s"] / r["best_median_s"]
               if r["best_median_s"] and r["default_median_s"] else None)
    out["search_note"] = (
        f"searched {r['n_correct']}/{r['space_size']} configs; "
        f"best={best_cfg} vs default={default_cfg}"
        + (f"; {speedup:.2f}x" if speedup else ""))
    return out


def _geomean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="fast", choices=["fast", "full"])
    ap.add_argument("--runs", type=int, default=20, help="timing iters per config")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = load_cases(args.suite)
    print(f"[akt-eval] backend={jax.default_backend()} interpret={os.environ.get('PALLAS_INTERPRET')} "
          f"suite={args.suite} cases={len(cases)}", flush=True)
    results = []
    for c in cases:
        try:
            r = eval_case(c, args.runs)
        except Exception as e:  # noqa: BLE001
            r = {"case": c.case_id, "correct": False, "error": repr(e)[:200]}
            traceback.print_exc()
        results.append(r)
        reg = r.get("regime", "tpu-deferred")
        fs = r.get("forward_s")
        ds = r.get("default_s")
        tag = ("OK  " if r.get("correct") else
               "DEF " if r.get("correct") is None else "FAIL")
        print(f"[akt-eval] {tag} {r['case']:<22} [{reg:>14}] "
              f"best={fs*1e3:.2f}ms " if fs else f"[akt-eval] {tag} {r['case']:<22} [{reg:>14}] best=—  ",
              end="")
        print((f"default={ds*1e3:.2f}ms  {r.get('search_note','')[:70]}"
               if ds else f" {r.get('search_note') or r.get('reason') or r.get('error','')}")[:110],
              flush=True)

    runnable = [r for r in results if r.get("correct")]
    geo = _geomean([r["forward_s"] for r in runnable])
    geo_def = _geomean([r["default_s"] for r in runnable])
    all_ok = all((r.get("correct") is not False) for r in results) and bool(runnable)
    n_choices = sum(r.get("space_size", 0) for r in results)
    summary = {"suite": args.suite, "all_correct": all_ok,
               "geomean_s": geo, "geomean_default_s": geo_def,
               "n_runnable": len(runnable), "n_deferred": sum(1 for r in results if r.get("correct") is None),
               "n_choices": n_choices, "results": results,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[akt-eval] SUMMARY all_correct={all_ok} search_geomean={geo*1e3:.2f}ms "
          f"default_geomean={geo_def*1e3:.2f}ms runnable={len(runnable)} "
          f"deferred={summary['n_deferred']} choices={n_choices}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
