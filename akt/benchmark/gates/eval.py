"""AKT performance-evaluation gate for the sglang-jax kernel suite (FROZEN).

Per kernel case: run the autotuning SEARCH over the (possibly capability-enlarged)
design space — an exhaustive enumerate-correct-time-argmin, optimal within the
space — then gate on numerical correctness vs the kernel's pure-JAX reference and
report the searched-best latency.

Objective = geomean of the searched-best deployable latencies over the runnable
cases. ``geomean_default_s`` is retained only as shipped-default context. Cases
whose frozen contract is TPU-only are excluded off TPU; a failure in a case that
advertises local execution fails closed instead of being mislabeled as deferred.

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

import re  # noqa: E402
import subprocess  # noqa: E402

import jax  # noqa: E402
from akt.benchmark.suites import load_cases  # noqa: E402
from akt.benchmark.runners.base import (  # noqa: E402
    LEGACY_OBJECTIVE_SCOPE,
    check_correct,
    search_best,
    time_config,
)


def run_native_test(nodeid: str, timeout: int = 300) -> dict:
    """Run sglang-jax's OWN pytest correctness test (the maintainer-authored check)
    in Pallas-interpret and report its verdict. `nodeid` is a pytest file/nodeid or
    `-k` expression. Returns {status: passed|failed|error|no-tests, passed, failed,
    detail}. This is an INDEPENDENT verification layered on top of the per-config
    allclose-vs-reference gate — it exercises the actual Pallas kernel under the
    repo's own assertions + tolerances."""
    env = dict(os.environ, PALLAS_INTERPRET="1", JAX_PLATFORMS="cpu",
               PYTHONPATH=str(REPO / "python"))
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header",
           "-p", "no:cacheprovider", *nodeid.split()]
    try:
        r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "detail": f"timeout {timeout}s"}
    tail = (r.stdout + r.stderr).strip().splitlines()
    summary = next((l for l in reversed(tail)
                    if re.search(r"passed|failed|error|no tests ran", l)), "")
    npass = int((re.search(r"(\d+) passed", summary) or [0, 0])[1])
    nfail = int((re.search(r"(\d+) (?:failed|error)", summary) or [0, 0])[1])
    status = ("passed" if npass and not nfail else
              "failed" if nfail else
              "no-tests" if "no tests ran" in summary else "error")
    return {"status": status, "passed": npass, "failed": nfail,
            "detail": summary.strip("= ")[:120]}


def eval_case(case, runs: int, native: bool = True) -> dict:
    deployment_space = case.space.deployment_space()
    out = {"case": case.case_id, "kernel": case.kernel_id, "shape": case.shape_id,
           "space_size": deployment_space.size(),
           "research_space_size": case.space.size(), "note": case.note,
           # the currently-EXPOSED tuning knobs (for the lowering/exposure graph):
           # elevated_by names the capability that added a knob (None = base/shipped).
           "knobs": [{"name": k.name, "n": len(k.values),
                      "deployment_n": (len(k.values)
                                       if k.programmer_control
                                       else 1),
                      "elevated_by": k.elevated_by,
                      "programmer_control": k.programmer_control}
                     for k in case.space.knobs]}
    # sglang-jax's own pytest correctness check (independent of our per-config gate).
    if native and getattr(case, "native_test", None):
        out["native_test"] = run_native_test(case.native_test)
        out["native_test_id"] = case.native_test
    default_cfg = case.space.default_config()
    # 1) does this case execute here? (probe the default config)
    ok, why = check_correct(case, default_cfg)
    if not ok:
        tpu_only = set(case.regime_pref).issubset({"tpu", "tpu-deferred"})
        deferred = jax.default_backend() != "tpu" and tpu_only
        regime = "tpu-deferred" if deferred else (
            jax.default_backend()
            + ("-interpret" if os.environ.get("PALLAS_INTERPRET") == "1" else "")
        )
        out.update(
            regime=regime,
            correct=None if deferred else False,
            forward_s=None,
            default_s=None,
            best_config=None,
            reason=why,
            search_note=(
                f"tpu-deferred ({why[:60]})"
                if deferred
                else f"default config failed ({why[:60]})"
            ),
        )
        return out
    regime = jax.default_backend() + ("-interpret" if os.environ.get("PALLAS_INTERPRET") == "1" else "")
    out["regime"] = regime
    # 2) search the (possibly enlarged) design space
    try:
        r = search_best(case, space=deployment_space, iters=runs)
    except Exception as e:  # noqa: BLE001
        out.update(correct=False, forward_s=None, default_s=None,
                   error=repr(e)[:200], search_note="search failed")
        return out
    best_cfg = r["best_config"]
    # A DesignSpace advertises every config accepted by valid() as supported.
    # Fail closed if any such config mismatches or errors; silently filtering it
    # would overstate the functional design space even if another config wins.
    correct = best_cfg is not None and r["n_correct"] == r["n_valid"]
    out.update(correct=correct, forward_s=r["best_median_s"],
               default_s=r["default_median_s"], best_config=best_cfg,
               default_config=default_cfg, n_correct=r["n_correct"],
               n_valid=r["n_valid"], truncated=r.get("truncated", False),
               incorrect_configs=r.get("incorrect_configs", []))
    if not correct and r.get("incorrect_configs"):
        first = r["incorrect_configs"][0]
        out["reason"] = (
            f"{r['n_valid'] - r['n_correct']} advertised config(s) incorrect; "
            f"first={first['config']}: {first['reason']}"
        )
    # search evidence (the "enlarged space was searched, not sampled" proof)
    speedup = (r["default_median_s"] / r["best_median_s"]
               if r["best_median_s"] and r["default_median_s"] else None)
    out["search_note"] = (
        f"searched {r['n_correct']}/{r['n_valid']} deployable valid configs"
        + (f"; {r['n_valid'] - r['n_correct']} incorrect" if not correct else "")
        + (" [TRUNCATED — cap hit, NOT exhaustive]" if r.get("truncated") else "")
        + f"; best={best_cfg} vs default={default_cfg}"
        + (f"; {speedup:.2f}x" if speedup else ""))
    return out


def _geomean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def native_verdict_ok(verdicts):
    """A configured native check supplies evidence only when it passes."""
    return all(verdict.get("status") == "passed" for verdict in verdicts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="fast", choices=["fast", "full"])
    ap.add_argument("--runs", type=int, default=20, help="timing iters per config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-native", action="store_true",
                    help="skip sglang-jax's own pytest correctness checks (faster)")
    args = ap.parse_args()

    cases = load_cases(args.suite)
    print(f"[akt-eval] backend={jax.default_backend()} interpret={os.environ.get('PALLAS_INTERPRET')} "
          f"suite={args.suite} cases={len(cases)} native_tests={not args.no_native}", flush=True)
    results = []
    for c in cases:
        try:
            r = eval_case(c, args.runs, native=not args.no_native)
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
        nt = r.get("native_test")
        nt_s = (f"  native[{r.get('kernel')}]:{nt['status']}" if nt else "")
        print(((f"default={ds*1e3:.2f}ms  {r.get('search_note','')[:60]}"
               if ds else f" {r.get('search_note') or r.get('reason') or r.get('error','')}")[:100]) + nt_s,
              flush=True)

    runnable = [r for r in results if r.get("correct")]
    geo = _geomean([r["forward_s"] for r in runnable])
    geo_def = _geomean([r["default_s"] for r in runnable])
    all_ok = all((r.get("correct") is not False) for r in results) and bool(runnable)
    n_choices = sum(r.get("space_size", 0) for r in results)
    # sglang-jax native-test verdicts (independent maintainer-authored checks)
    nats = [r["native_test"] for r in results if r.get("native_test")]
    native_summary = {"ran": len(nats),
                      "passed": sum(1 for n in nats if n["status"] == "passed"),
                      "failed": sum(1 for n in nats if n["status"] == "failed"),
                      "other": sum(1 for n in nats if n["status"] not in ("passed", "failed"))}
    # Every configured native test must positively pass. A timeout, collection
    # error, or "no tests" result is not evidence of correctness and fails closed.
    native_ok = native_verdict_ok(nats)
    truncated = [r["case"] for r in results if r.get("truncated")]
    summary = {"suite": args.suite, "objective_scope": LEGACY_OBJECTIVE_SCOPE,
               "all_correct": all_ok and native_ok,
               "allclose_correct": all_ok, "native_ok": native_ok,
               "geomean_s": geo, "geomean_default_s": geo_def,
               "n_runnable": len(runnable), "n_deferred": sum(1 for r in results if r.get("correct") is None),
               "truncated_cases": truncated,   # searches that hit the cap (not exhaustive)
               "native_summary": native_summary,
               "n_choices": n_choices, "results": results,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if truncated:
        print(f"[akt-eval] WARNING: search TRUNCATED (cap hit, not exhaustive) for: "
              f"{', '.join(truncated)}", flush=True)
    print(f"[akt-eval] SUMMARY all_correct={all_ok and native_ok} (allclose={all_ok} "
          f"native={native_summary['passed']}/{native_summary['ran']} pass) "
          f"search_geomean={geo*1e3:.2f}ms default_geomean={geo_def*1e3:.2f}ms "
          f"runnable={len(runnable)} deferred={summary['n_deferred']} choices={n_choices}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
