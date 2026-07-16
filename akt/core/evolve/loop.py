"""akt CAPABILITY-ELEVATION loop — the deterministic scaffold around this Claude
Code session, whose job is to GROW the programmer's design space, not tune it.

WHY THIS EXISTS (vs the retired parameter-sweep loop)
  The parameter-sweep loop (tag: akt-paramsweep-v1) toggles flags/params that
  already exist and converged vn_go to its *executable* optimum. But the campaign
  diagram keeps showing the same bottleneck (FC giant-step rotations) because the
  thing that would relieve it is a low-level flexibility the LOWERING can execute
  yet the top-level programming model cannot NAME — a flexibility gap. Tuning
  can't reach it; only ELEVATING it into the design space can.

  This loop drives exactly that. Each round the agent:

    (1) ESTIMATE  — read the QUERY (bottleneck + ranked flexibility-gap frontier)
                    and pick ONE gap whose elevation should relieve the bottleneck,
                    with a quantified estimate of the relief.
    (2) IMPLEMENT — build the WHOLE STACK that exposes this gap as a new design
                    choice (backend / FFI / evaluator / optimizer / nn as needed),
                    and REGISTER its new dimension in the vn_go layout enumerator
                    (taxonomy/flags) so the search can pick it.
    (3) SEARCH    — the loop brute-forces the ENLARGED space to the new global
                    optimum (DP-accelerated; the exact==B&B EXACTNESS REPORT proves
                    the new dimension was searched, not sampled).
    (4) GATE      — real-HW evaluation of the 3 small MNIST models; KEEP the whole
                    add-on IFF the new optimum beats the incumbent by > target
                    (default 2%) AND stays correct; else RESTORE (revert every file
                    the capability touched, back to the incumbent commit).

  The script owns (3) and (4) and the keep/restore bookkeeping. The agent owns the
  intelligence of (1) and (2) and the search-dimension wiring inside (2). A
  capability is described by a small MANIFEST the agent writes
  (akt/core/evolve/capabilities/<name>.json) so the loop can search it, gate it,
  and cleanly revert it.

FROZEN (the capability may touch anything EXCEPT these — they define the
measurement, so letting a capability edit them would let it game its own gate):
  - akt/benchmark/**      (adapter, suites, gates/eval.py + its ORION reference
                           MAEs/latencies — the frozen HW gate)
  - akt/core/evolve/loop.py  (this file)
Everything else is fair game — orion/backend/** (Go/FFI), orion/core/packing.py,
orion/core/vn_go/**, orion/nn/** — because "implement the whole stack" is the point.
(Contrast the param-sweep loop, which froze backend/packing.)

USAGE — two modes, same gate
  python akt/core/evolve/loop.py init   [--hours H] [--target 0.02] [--runs 3]

  # (A) AUTONOMOUS — the LOOP drives an LLM oracle to implement each round, then gates it.
  #     Pluggable oracle (Claude Code today; codex / a model-API harness are drop-ins).
  #     Needs a CLEAN incumbent tree (it reverts void/rejected rounds).
  python akt/core/evolve/loop.py run --rounds N [--oracle claude|codex] [--oracle-cmd '...'] [--audit]

  # (B) MANUAL — a human/agent implements the capability + writes the manifest, then submits.
  python akt/core/evolve/loop.py status                     # the Capability QUERY
  python akt/core/evolve/loop.py submit --capability <name> [--runs 3] [--audit]
  python akt/core/evolve/loop.py restore --capability <name>   # manual revert
"""
from __future__ import annotations

import argparse, json, os, re, signal, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "akt/optimization_history/evolve_state.json"
HIST = ROOT / "akt/optimization_history/evolve_history.jsonl"
CAPS = Path(__file__).resolve().parent / "capabilities"
EVOLVE_JSON = ROOT / "akt/board/evolve.json"
STATUS = ROOT / "akt/board/evolve_status.json"   # live heartbeat -> board WIP time bar
# All harness subprocesses run under the project venv with the kernel import path
# (python/ for sgl_jax, repo root for akt) and Pallas interpret so the TPU kernels
# that support it run on this non-TPU box.
PY = os.environ.get("AKT_PY", ".venv/bin/python")
PYENV = f'PALLAS_INTERPRET=1 PYTHONPATH="python:." {PY}'
ADAPTER = f"{PYENV} akt/benchmark/adapter.py"
EVAL = f"{PYENV} akt/benchmark/gates/eval.py"

# The capability may edit anything except the measurement harness + this loop.
FROZEN = ("akt/benchmark/", "akt/core/evolve/loop.py")

# ---- Oracle seam: the loop DRIVES an LLM oracle to implement each round -------
# `run` hands the QUERY to an oracle, which edits the repo + writes a manifest; the
# loop then gates it. Pluggable by name (--oracle) or command (--oracle-cmd). The
# command runs in ROOT with {prompt} = a file holding the round prompt (fed on stdin).
# Claude Code today; codex is a drop-in; a model-API harness (tools to read/write/bash)
# would be a new entry that satisfies the same contract (edit tree + write manifest).
ORACLES = {
    # stream-json + verbose so the loop can render live WIP (tool calls, messages) as
    # the oracle works, instead of one silent block at the end.
    "claude": ("cat {prompt} | claude -p --dangerously-skip-permissions "
               "--output-format stream-json --verbose"),
    "codex":  "cat {prompt} | codex exec --dangerously-bypass-approvals-and-sandbox -",
}
ORACLE_PROMPT = """You are the ORACLE for the AKT capability-elevation loop, working in \
the sglang-jax TPU-Pallas kernel repo (cwd = repo root). Do EXACTLY ONE capability this \
round, then STOP — do NOT run the evolve loop or `submit`; the harness gates you.

{query}

DOMAIN: each kernel is wrapped as a KernelCase (akt/core/runners/<kernel>.py) with a
DesignSpace of tunable tiling/config Knobs. The loop's SEARCH autotunes that space
(enumerate -> check-correct-vs-reference -> time -> argmin). A CAPABILITY elevates a NEW
low-level flexibility that the Mosaic/Pallas lowering can execute but the kernel's config
schema does not currently NAME, into a new Knob (or a new value range) the search can pick.

YOUR TASK:
 1. From the FRONTIER above pick ONE flexibility gap whose elevation can beat the incumbent \
by >{target_pct:.0f}% on the kernel suite. State a one-line relief estimate first.
 2. IMPLEMENT it end-to-end: expose the knob in the kernel (python/sgl_jax/srt/kernels/**) \
so the tiling/pipeline/fusion/dtype choice is actually plumbed through, then REGISTER its \
new Knob (or widened value set) in the kernel's runner DesignSpace so the search enumerates \
it. Keep the run() mapping in lock-step with what the kernel executes (cost == execution).
 3. Verify correctness: the searched-best config must still match the pure-JAX reference \
within the case's atol/rtol (PALLAS_INTERPRET=1 PYTHONPATH=python:. .venv/bin/python \
akt/benchmark/gates/eval.py --suite fast must stay all_correct=True). Prefer exact/structural \
retiling over lossy approximations — a config that is fast but wrong is rejected.
 4. Write akt/core/evolve/capabilities/<name>.json (schema: that dir's README) with gap, \
hypothesis, estimated_relief_pct, search_dimension, audit_case, and files_touched listing \
EVERY file you changed or created. Set "status":"pending".

HARD CONSTRAINTS:
 - NEVER edit anything under akt/benchmark/ or akt/core/evolve/loop.py (FROZEN — the \
measurement/harness; touching them VOIDS the round and your work is reverted).
 - Do NOT git commit; leave changes in the working tree + the manifest.
 - Your FINAL line must be exactly:  CAPABILITY: <name>
"""


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def load():
    return json.loads(STATE.read_text())


def save(st):
    STATE.write_text(json.dumps(st, indent=1))


def git_head():
    rc, out = sh("git rev-parse HEAD", timeout=30)
    return out.strip() if rc == 0 else None


def write_status(state, round=None, capability=None, phase=None, last=None, reset=False):
    """Live run-status heartbeat for the board's WIP time bar. `state` is
    'running' | 'idle'. Preserves init_ts (loop age) and the started_ts of the
    CURRENT running round across mid-round phase updates. `reset=True` (on init)
    starts a fresh campaign clock."""
    now = time.time()
    cur = {}
    if not reset:
        try:
            cur = json.loads(STATUS.read_text())
        except Exception:
            cur = {}
    init_ts = cur.get("init_ts", now)
    # keep the round's start across phase updates; new start when idle->running
    started = (cur.get("started_ts") if (state == "running" and cur.get("state") == "running")
               else now if state == "running" else None)
    out = {"state": state, "updated_ts": now, "init_ts": init_ts,
           "round": round if round is not None else cur.get("round"),
           "capability": capability, "phase": phase, "started_ts": started,
           "last": last if last is not None else cur.get("last")}
    try:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(out, indent=1, allow_nan=False))
    except Exception:
        pass


# ---------------------------------------------------------------- QUERY blocks

def _adapter_json(sub, tag):
    """Shell one adapter subcommand and return its last AKT_<tag> {json} dict."""
    try:
        _rc, out = sh(f"{ADAPTER} {sub}", timeout=180)
        ms = re.findall(rf"AKT_{tag} (\{{.*\}})", out)
        return json.loads(ms[-1]) if ms else None
    except Exception:
        return None


def bottleneck_block():
    d = _adapter_json("bottleneck", "BOTTLENECK")
    if not d:
        return "== BOTTLENECK unavailable"
    if d.get("missing"):
        return f"== BOTTLENECK: {d['missing']}"
    body = "\n".join("   " + l for l in d.get("lines", []))
    note = f"\n   {d['note']}" if d.get("note") else ""
    return f"== BOTTLENECK ({d.get('title', 'monitor')})\n{body}{note}"


def frontier_block():
    """The ranked flexibility-gap frontier (AKT_GAPS): the candidate add-ons. The
    order is the adapter/flexgap ranking (schedule-keyword salience); the agent
    picks the one whose elevation best matches the current bottleneck."""
    d = _adapter_json("gaps", "GAPS")
    if not d:
        return "== FRONTIER unavailable (no gaps subcommand / flexgap.json missing)"
    if d.get("missing"):
        return f"== FRONTIER: {d['missing']}"
    lines = d.get("lines", [])
    body = "\n".join(f"   [{i}] {l}" for i, l in enumerate(lines))
    note = f"\n   {d['note']}" if d.get("note") else ""
    return (f"== FRONTIER ({d.get('title', 'flexibility gaps to elevate')})\n"
            f"{body}{note}")


def _cap_summaries():
    kept, rejected = [], []
    for p in sorted(CAPS.glob("*.json")) if CAPS.is_dir() else []:
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        (kept if m.get("status") == "kept" else
         rejected if m.get("status") == "rejected" else []).append(m)
    return kept, rejected


def query(st):
    unit = st.get("objective_unit", "s")
    cond = ("CONTINUE" if time.time() < st["deadline_ts"] else "FINISH(DEADLINE)")
    kept, rejected = _cap_summaries()
    kepts = "; ".join(f"{m['name']}(+{m.get('delta_pct', '?')}%)" for m in kept) or "(none yet)"
    rejs = "; ".join(f"{m['name']}[{m.get('reject_reason', 'rejected')[:32]}]"
                     for m in rejected) or "(none)"
    tgt = st["target_improvement"] * 100
    return (
        f"== CAPABILITY QUERY round={st['round'] + 1} -> {cond}\n"
        f"   incumbent {st['objective_name']}={st['incumbent_geomean']:.4f}{unit} "
        f"(choices={st['incumbent_choices']}); KEEP bar = beat it by > {tgt:.0f}% on the kernel suite\n"
        f"   KEPT capabilities: {kepts}\n"
        f"   REJECTED (do not re-attempt unmodified): {rejs}\n"
        f"{bottleneck_block()}\n"
        f"{frontier_block()}\n"
        f"== DECIDE ONE gap to elevate. Then, off-loop:\n"
        f"   (1) ESTIMATE the bottleneck relief; (2) IMPLEMENT it end-to-end (plumb the\n"
        f"   knob through python/sgl_jax/srt/kernels/**) AND register its Knob in the\n"
        f"   kernel's runner DesignSpace (akt/core/runners/) so the search picks it;\n"
        f"   write the manifest akt/core/evolve/capabilities/<name>.json (schema in README);\n"
        f"   then: python akt/core/evolve/loop.py submit --capability <name>")


# ---------------------------------------------------------------- gate helpers

def hw_eval(runs, suite="full"):
    """Real-HW evaluation of the suite through the FROZEN gate (eval.py). Returns
    (all_correct, geomean_s, per_case, results, n_choices) — results carry per-case
    search_note (the enlarged-search evidence produced during compile)."""
    out_path = ROOT / "akt/optimization_history/.evolve_eval.json"
    # Delete any prior-round result FIRST: if this eval crashes before writing, the
    # missing file is detected below instead of the stale previous result being read
    # back as if it were fresh (a crashed eval must fail the round, not pass on old data).
    out_path.unlink(missing_ok=True)
    rc, log = sh(f"{EVAL} --suite {suite} --runs {runs} --out {out_path}", timeout=3600)
    if not out_path.exists():
        print(f"[evolve] eval produced no output (rc={rc}):\n{log[-800:]}")
        return False, float("nan"), {}, [], None, float("nan")
    s = json.loads(out_path.read_text())
    per = {r["case"]: r.get("forward_s") for r in s.get("results", [])}
    return (bool(s.get("all_correct")), float(s.get("geomean_s", float("nan"))),
            per, s.get("results", []), s.get("n_choices"),
            float(s.get("geomean_default_s", float("nan"))))


def search_audit(case, exact=False):
    """Evidence that the ENLARGED design space is actually enumerated (searched,
    not sampled): report the per-case design-space size + knob axes for the audit
    kernel. The exhaustive enumerate-correct-time-argmin in runners/base.search_best
    is optimal within that space by construction. Returns the adapter's space report."""
    rc, out = sh(f"{ADAPTER} space --kernel {case}", timeout=300)
    tail = "\n".join(out.strip().splitlines()[-18:])
    return rc, tail


def load_manifest(name):
    p = CAPS / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no manifest at {p.relative_to(ROOT)} — write it first "
            f"(schema: akt/core/evolve/capabilities/README.md)")
    m = json.loads(p.read_text())
    bad = [f for f in m.get("files_touched", [])
           if any(f.startswith(x) or x in f for x in FROZEN)]
    if bad:
        raise PermissionError(
            f"manifest lists FROZEN files (would game the gate): {bad}")
    return m, p


def restore_capability(m, incumbent_commit):
    """Revert ONLY the files the capability touched back to the incumbent commit.
    A file tracked at the incumbent is checked out; a NEW file the capability
    created (untracked at incumbent) is removed. The gitignored Lattigo .so is
    never rm'd here — if the capability rebuilt the backend, revert the .go source
    then REBUILD the .so back to the incumbent (build_lattigo)."""
    files = list(m.get("files_touched", []))
    for f in files:
        rc, _ = sh(f"git cat-file -e {incumbent_commit}:{f}", timeout=30)
        if rc == 0:
            sh(f"git checkout {incumbent_commit} -- {f}", timeout=60)
        else:                                     # capability-created new file -> remove
            (ROOT / f).unlink(missing_ok=True)
    # Kernels are pure Python/Pallas — no compiled artifact to rebuild on revert.
    print(f"[evolve] restored {len(files)} file(s) to incumbent {incumbent_commit[:8]}")


def write_board():
    # build.py joins evolve_history + manifests + evolve_state + the freshest eval into
    # board.json + flexgraph.json. It is jax-free (reads JSON + adapter.FRONTIER), so
    # plain python3 suffices — no venv needed for the board refresh.
    sh("python3 akt/board/build.py", timeout=300)


# ---------------------------------------------------------------- commands

def cmd_init(args):
    correct, geo, _per, _res, n_choices, geo_def = hw_eval(args.runs)
    if not correct or geo != geo:
        print(f"[evolve] init ABORTED: incumbent gate failed (correct={correct}, geo={geo}).")
        return
    # Snapshot the FULL set of suite cases the incumbent eval produced. Every later round
    # must reproduce this set; a capability that breaks a runner import (silently dropping
    # a kernel from the eval, and thus from the geomean) then fails the missing-case guard.
    expected_cases = sorted(r.get("case") for r in _res if r.get("case"))
    CAPS.mkdir(exist_ok=True)
    # Incumbent = the BEST performance achievable with the EXISTING knobs (the
    # base-space autotuning optimum), NOT the shipped default (a suboptimal config).
    # A capability must therefore beat the best you can already do by tuning the
    # current design space — so a KEEP reflects a genuinely NEW flexibility, not mere
    # autotuning of knobs the kernel already exposes. The shipped default is kept only
    # as context (shipped_default_geomean).
    incumbent = geo if geo == geo else geo_def
    st = {"start_ts": time.time(), "deadline_ts": time.time() + args.hours * 3600,
          "round": 0, "target_improvement": args.target,
          "incumbent_geomean": incumbent, "incumbent_choices": n_choices,
          "shipped_default_geomean": geo_def, "base_search_geomean": geo,
          "incumbent_commit": git_head(), "expected_cases": expected_cases,
          "objective_name": "geomean_s", "objective_unit": "s"}
    save(st)
    write_status("idle", round=0, reset=True)   # start a fresh campaign clock
    print(f"[evolve] initialized: incumbent (best of EXISTING knobs) geomean={incumbent*1e3:.2f}ms "
          f"| shipped default={geo_def*1e3:.2f}ms (context only) | choices={n_choices} "
          f"commit={str(st['incumbent_commit'])[:8]}, KEEP bar > {args.target*100:.0f}%.")
    print(query(st))


def cmd_status(_):
    print(query(load()))


def gate_capability(st, m, mpath, args):
    """Steps (3)+(4): DP-search the enlarged space (optional audit), real-HW gate the
    3 MNIST models, KEEP iff >target else RESTORE. Advances the round; used by both the
    manual `submit` and the autonomous `run`. Returns the decision string."""
    st["round"] += 1
    rnd = st["round"]
    incumbent = st["incumbent_geomean"]
    write_status("running", round=rnd, capability=m["name"], phase="starting")
    print(f"== ROUND {rnd} SUBMIT capability='{m['name']}' gap='{m.get('gap','?')[:60]}'")

    # (3) enlarged-space search evidence for the target model (optional heavy audit)
    audit_tail = ""
    if args.audit and m.get("audit_case"):
        write_status("running", round=rnd, capability=m["name"], phase="enlarged-search audit")
        _rc, audit_tail = search_audit(m["audit_case"], exact=args.audit_exact)
        print("== ENLARGED-SEARCH EXACTNESS (target model):")
        print("\n".join("   " + l for l in audit_tail.splitlines()))

    # (4) HW gate on the kernel suite (interpret proxy here; TPU when attached)
    write_status("running", round=rnd, capability=m["name"], phase="HW eval (kernel suite)")
    correct, geo, per, results, n_choices, _geo_def = hw_eval(args.runs)
    notes = {r["case"]: (r.get("search_note") or "")[:100] for r in results}
    tgt = st["target_improvement"]
    delta_pct = round((geo / incumbent - 1) * 100, 2) if geo == geo else None

    # --- FUNCTIONAL-CORRECTNESS GUARD ------------------------------------------
    # eval.py's per-case `correct` is allclose(searched-best output, pure-JAX
    # reference) within the case's atol/rtol. This guard makes the decision decisive
    # and auditable: NO capability is kept unless every case that RAN stays correct,
    # and a failure names the offending kernel — so a fast-but-wrong retiling can't
    # slip through on a speed win.
    acc = {r["case"]: {"regime": r.get("regime"),
                       "correct": r.get("correct"),
                       "best_config": r.get("best_config"),
                       "why": r.get("reason") or r.get("error") or ""}
           for r in results}
    # A case that DOESN'T run here (correct is None -> tpu-deferred) is wired but not
    # gated locally; only a case that RAN and MISMATCHED its reference (correct is
    # False) fails the guard. On TPU every case runs and is gated. n_verified/n_deferred
    # are recorded so a KEEP whose touched kernel was never correctness-checked here is
    # VISIBLE rather than silent.
    failing = [f"{c}[{(v['why'] or 'mismatch')[:40]}]"
               for c, v in acc.items() if v.get("correct") is False]
    n_verified = sum(1 for v in acc.values() if v.get("correct") is True)
    n_deferred = sum(1 for v in acc.values() if v.get("correct") is None)
    # MISSING-CASE GUARD: every suite case the incumbent produced must reappear. A
    # capability that breaks a runner import silently drops that kernel from the eval
    # (and from the geomean) — that must fail, not pass on a smaller suite.
    present = {r.get("case") for r in results if r.get("case")}
    missing = sorted(c for c in (st.get("expected_cases") or []) if c not in present)
    # TRUNCATION GUARD: a search that hit the enumeration cap did NOT prove its optimum,
    # so a KEEP built on it is unsound.
    truncated = [r.get("case") for r in results if r.get("truncated")]
    guard_ok = (bool(correct) and not failing and not missing and not truncated
                and geo == geo)

    if not guard_ok:
        decision = "reject"
        if failing:
            reason = "CORRECTNESS GUARD FAILED: " + "; ".join(failing)
        elif missing:
            reason = ("MISSING-CASE GUARD FAILED: eval dropped " + ", ".join(missing)
                      + " (a runner likely failed to import) — geomean is over a partial suite")
        elif truncated:
            reason = ("SEARCH-TRUNCATION GUARD FAILED: search hit the cap for "
                      + ", ".join(truncated) + " — optimum not proven within the space")
        else:
            reason = "correctness gate FAILED (eval error / NaN)"
    elif geo < incumbent * (1 - tgt):
        decision, reason = "keep", f"new optimum {geo:.4f} beats incumbent {incumbent:.4f} by {-delta_pct:.2f}% (> {tgt*100:.0f}%)"
    else:
        decision, reason = "reject", f"insufficient gain ({geo:.4f} vs {incumbent:.4f}, {delta_pct:+.2f}% <= {tgt*100:.0f}% bar)"

    rec = {"round": rnd, "capability": m["name"], "gap": m.get("gap"),
           "hypothesis": m.get("hypothesis"),
           "estimated_relief_pct": m.get("estimated_relief_pct"),
           "search_dimension": m.get("search_dimension"),
           "files_touched": m.get("files_touched"),
           "correct": guard_ok, "accuracy": acc, "geomean_s": (None if geo != geo else geo),
           "incumbent_geomean": incumbent, "delta_pct": delta_pct,
           "n_verified": n_verified, "n_deferred": n_deferred,
           "missing_cases": missing, "truncated_cases": truncated,
           "target_pct": tgt * 100, "decision": decision, "reason": reason,
           "per_case": per, "search_notes": notes, "audit": bool(args.audit),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    HIST.open("a").write(json.dumps(rec, allow_nan=False) + "\n")

    if decision == "keep":
        m["status"] = "kept"; m["delta_pct"] = -delta_pct if delta_pct else None
        m["measured_geomean_s"] = geo
        mpath.write_text(json.dumps(m, indent=1))
        st["incumbent_geomean"] = geo
        st["incumbent_choices"] = n_choices
        write_board()
        sh("git add -A", timeout=60)
        msg = f"akt evolve R{rnd} KEEP capability={m['name']}: {reason[:60]}"
        sh(f"git commit -q -m {json.dumps(msg)} "
           f"-m 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>'", timeout=120)
        st["incumbent_commit"] = git_head()
        save(st)
    else:
        m["status"] = "rejected"; m["reject_reason"] = reason
        # revert code before recording the rejection in the manifest/board
        restore_capability(m, st["incumbent_commit"])
        mpath.write_text(json.dumps(m, indent=1))   # keep the rejected record
        write_board()
        save(st)

    write_status("idle", round=rnd, capability=m["name"], last={
        "round": rnd, "capability": m["name"], "decision": decision, "reason": reason,
        "geomean_s": (None if geo != geo else geo), "delta_pct": delta_pct,
        "finished_ts": time.time()})
    unit = st["objective_unit"]
    print(f"== ROUND {rnd} {decision.upper()}: {reason} | geomean_s={geo:.4f}{unit} "
          f"incumbent={st['incumbent_geomean']:.4f}{unit}")
    print(query(st))
    return decision


def cmd_submit(args):
    """Manual round: the capability is already implemented in the tree + its manifest."""
    st = load()
    m, mpath = load_manifest(args.capability)
    gate_capability(st, m, mpath, args)


def cmd_restore(args):
    st = load()
    m, _p = load_manifest(args.capability)
    restore_capability(m, st["incumbent_commit"])
    print(f"[evolve] manually restored capability '{m['name']}'.")


# ---------------------------------------------------------------- oracle driver

def _oracle_cmd(args):
    if getattr(args, "oracle_cmd", None):
        return args.oracle_cmd
    name = getattr(args, "oracle", "claude")
    if name not in ORACLES:
        raise SystemExit(f"unknown oracle '{name}'; known: {list(ORACLES)} (or --oracle-cmd)")
    return ORACLES[name]


def _manifest_names():
    return {p.stem for p in CAPS.glob("*.json")} if CAPS.is_dir() else set()


def _worktree_changed():
    """Repo-relative paths with uncommitted changes (tracked mods + untracked)."""
    rc, out = sh("git status --porcelain --untracked-files=all", timeout=60)
    paths = []
    for l in out.splitlines():
        p = l[3:].strip()
        if p:
            paths.append(p.split(" -> ")[-1].strip('"'))
    return paths


def touched_frozen():
    return [p for p in _worktree_changed()
            if any(p.startswith(x) or x in p for x in FROZEN)]


def _frozen_fingerprint():
    """md5 of every FROZEN file — snapshot before/after the oracle so the guard
    catches ONLY the oracle's edits to the measurement harness, independent of any
    other uncommitted changes in the tree."""
    import hashlib
    fp = {}
    files = []
    for x in FROZEN:
        p = ROOT / x
        if p.is_dir():
            files += [q for q in p.rglob("*") if q.is_file()]
        elif p.is_file():
            files.append(p)
    for q in files:
        try:
            fp[str(q)] = hashlib.md5(q.read_bytes()).hexdigest()
        except Exception:
            pass
    return fp


def _frozen_diff(pre):
    post = _frozen_fingerprint()
    return sorted(str(Path(k).relative_to(ROOT))
                  for k in set(pre) | set(post) if pre.get(k) != post.get(k))


def revert_worktree(incumbent):
    """Discard a void round's uncommitted edits back to the incumbent commit, EXCEPT
    the loop's own bookkeeping (optimization_history / board), so history survives."""
    for f in _worktree_changed():
        if f.startswith("akt/optimization_history") or f.startswith("akt/board"):
            continue
        rc, _ = sh(f"git cat-file -e {incumbent}:{f}", timeout=30)
        if rc == 0:
            sh(f"git checkout {incumbent} -- {f}", timeout=60)
        else:
            (ROOT / f).unlink(missing_ok=True)


def _render_oracle_line(line):
    """One-line WIP summary of a streamed oracle event (Claude stream-json), or the
    raw line for a plain-text oracle (codex). None = nothing worth showing."""
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
    except Exception:
        return line[:120]                              # non-JSON oracle -> raw
    if not isinstance(ev, dict):
        return line[:120]                              # valid JSON scalar/array -> raw
    t = ev.get("type")
    if t == "assistant":
        out = []
        for c in ev.get("message", {}).get("content", []):
            k = c.get("type")
            if k == "tool_use":
                inp = c.get("input", {}) or {}
                tgt = (inp.get("file_path") or inp.get("path")
                       or str(inp.get("command", ""))[:56] or "")
                out.append(f"tool:{c.get('name')} {tgt}".strip())
            elif k == "text" and c.get("text", "").strip():
                out.append("say: " + c["text"].strip().replace("\n", " ")[:90])
        return "  ".join(out) or None
    if t == "result":
        dur = (ev.get("duration_ms") or 0) // 1000
        return f"DONE — {ev.get('subtype', '')} ({ev.get('num_turns', '?')} turns, {dur}s)"
    if t == "rate_limit_event":
        info = ev.get("rate_limit_info", {}) or {}
        return f"RATE-LIMIT {info.get('status', '?')} ({info.get('rateLimitType', '?')}, resets {info.get('resetsAt', '?')})"
    return None


def _scan_rate_limit(line):
    """Rate-limit reset timestamp from one stream-json line, or None. Covers the
    explicit rate_limit_event and a 429 result."""
    try:
        ev = json.loads(line)
    except Exception:
        return None
    if not isinstance(ev, dict):
        return None
    if ev.get("type") == "rate_limit_event":
        info = ev.get("rate_limit_info", {}) or {}
        if info.get("status") == "rejected":
            return float(info.get("resetsAt") or (time.time() + 1800))
    if ev.get("type") == "result" and ev.get("api_error_status") == 429:
        return float(time.time() + 1800)     # 429 without an explicit reset time
    return None


def invoke_oracle(st, args):
    """Drive the LLM oracle for one round: hand it the QUERY, STREAM its work live
    (terminal + a tail-able log + a board heartbeat), and return the capability name
    it produced (a new pending manifest, or a `CAPABILITY: <name>` line)."""
    CAPS.mkdir(exist_ok=True)
    before = _manifest_names()
    pf = ROOT / "akt/optimization_history/.oracle_prompt.txt"
    pf.write_text(ORACLE_PROMPT.format(query=query(st),
                                       target_pct=st["target_improvement"] * 100))
    cmd = _oracle_cmd(args).replace("{prompt}", str(pf))
    logf = ROOT / "akt/optimization_history/.oracle.log"
    nxt = st["round"] + 1
    print(f"[evolve] oracle started (timeout {args.oracle_timeout}s). Live WIP below; "
          f"also `tail -f {logf}` and the board WIP bar (round {nxt}).")
    proc = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)

    def kill_oracle_group():
        write_status("running", round=nxt, phase="oracle timeout; terminating process group")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        time.sleep(5)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    watchdog = threading.Timer(args.oracle_timeout, kill_oracle_group)
    watchdog.start()
    started, out_lines = time.time(), []
    heartbeat_stop = threading.Event()
    last_msg = {"text": "starting", "ts": started}

    def heartbeat():
        while not heartbeat_stop.wait(10):
            quiet = int(time.time() - last_msg["ts"])
            suffix = f" (no output {quiet}s)" if quiet >= 60 else ""
            write_status("running", round=nxt,
                         phase=f"oracle: {last_msg['text'][:56]}{suffix}")

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    rate_reset = None
    try:
        with logf.open("w") as lg:
            for line in proc.stdout:
                lg.write(line); lg.flush(); out_lines.append(line)
                rl = _scan_rate_limit(line)
                if rl:
                    if '"rate_limit_event"' in line:   # authoritative resetsAt wins
                        rate_reset = rl
                    elif rate_reset is None:           # 429 fallback only if nothing better
                        rate_reset = rl
                msg = _render_oracle_line(line)
                now = time.time()
                if msg:
                    last_msg.update(text=msg, ts=now)
                    print(f"  [oracle +{int(now - started)}s] {msg}", flush=True)
                if msg:                                # immediate board activity update
                    write_status("running", round=nxt,
                                 phase=f"oracle: {(msg or 'working')[:56]}")
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)
        watchdog.cancel()
        proc.wait()
    out = "".join(out_lines)
    new = [n for n in (_manifest_names() - before) if _pending(n)]
    if new:
        print(f"[evolve] oracle finished ({int(time.time()-started)}s) -> capability '{new[0]}'.")
        return new[0], None
    for nm in reversed(re.findall(r"CAPABILITY:\s*([A-Za-z0-9_\-]+)", out)):
        if (CAPS / f"{nm}.json").exists():
            return nm, None
    print(f"[evolve] oracle produced no pending manifest (exit={proc.returncode}); see {logf}."
          + (f" RATE-LIMITED until {time.strftime('%H:%M:%S', time.localtime(rate_reset))}."
             if rate_reset else ""))
    return None, rate_reset


def _pending(name):
    try:
        return json.loads((CAPS / f"{name}.json").read_text()).get("status") == "pending"
    except Exception:
        return False


def cmd_run(args):
    """AUTONOMOUS loop: for each round the LOOP drives the oracle to implement one
    capability, then gates it (keep/restore). This is the loop DRIVING the LLM.

    Precondition: a CLEAN incumbent tree — `run` reverts void/rejected rounds to the
    incumbent commit, so any pre-existing uncommitted work (outside the loop's own
    bookkeeping) would be lost. Commit or stash first."""
    dirty = [p for p in _worktree_changed()
             if not (p.startswith("akt/optimization_history") or p.startswith("akt/board"))]
    if dirty:
        print("[evolve] `run` needs a CLEAN incumbent tree (it reverts void/rejected rounds).")
        print(f"         commit or stash these first ({len(dirty)}): {dirty[:8]}"
              + (" ..." if len(dirty) > 8 else ""))
        return
    gated = 0                # rounds that actually reached the gate (the --rounds budget)
    consec_void = 0          # non-rate-limit voids in a row (backstop against error loops)
    MAX_CONSEC_VOID = 5
    while gated < args.rounds:
        st = load()
        if time.time() >= st["deadline_ts"]:
            print(f"[evolve] deadline reached after {gated} gated round(s) — stopping."); break
        nxt = st["round"] + 1
        print(f"\n{'#'*70}\n# AUTONOMOUS ROUND {nxt} — oracle={args.oracle_cmd or args.oracle}\n{'#'*70}")
        write_status("running", round=nxt, phase=f"oracle ({args.oracle}) implementing")
        pre_fp = _frozen_fingerprint()
        name, rate_reset = invoke_oracle(st, args)
        if rate_reset and not name:
            # ORACLE RATE-LIMITED (e.g. the 5h window): don't burn rounds hammering the
            # API — sleep until the reset (+2 min slack), capped at the campaign deadline.
            wake = rate_reset + 120
            if wake >= st["deadline_ts"]:
                print(f"[evolve] rate-limit reset ({time.strftime('%H:%M', time.localtime(rate_reset))}) "
                      f"is at/after the deadline — stopping.")
                break
            wait = max(0, wake - time.time())
            if not wait:                              # reset already passed -> retry now
                continue
            print(f"[evolve] oracle RATE-LIMITED; sleeping {int(wait/60)} min until "
                  f"{time.strftime('%H:%M:%S', time.localtime(wake))}.")
            write_status("idle", round=st["round"], last={
                "round": nxt, "capability": None, "decision": "rate-limited",
                "reason": f"oracle 429; sleeping until {time.strftime('%H:%M', time.localtime(wake))}",
                "finished_ts": time.time()})
            time.sleep(wait)
            continue                                  # retry the round after the reset
        frozen_changed = _frozen_diff(pre_fp)        # ONLY the oracle's frozen edits
        void = (f"oracle modified FROZEN harness {frozen_changed[:2]}" if frozen_changed
                else "oracle produced no manifest" if not name else None)
        if void:
            consec_void += 1
            print(f"[evolve] VOID round: {void} — reverting the working tree "
                  f"({consec_void}/{MAX_CONSEC_VOID} consecutive).")
            revert_worktree(st["incumbent_commit"])
            write_status("idle", round=st["round"], last={"round": nxt, "capability": name,
                         "decision": "void", "reason": void, "finished_ts": time.time()})
            if consec_void >= MAX_CONSEC_VOID:
                print(f"[evolve] {MAX_CONSEC_VOID} consecutive VOID rounds — oracle is failing "
                      f"systematically; stopping (see .oracle.log).")
                break
            continue
        try:
            m, mpath = load_manifest(name)
        except Exception as e:
            consec_void += 1
            print(f"[evolve] VOID round: bad manifest '{name}': {e} — reverting "
                  f"({consec_void}/{MAX_CONSEC_VOID} consecutive).")
            revert_worktree(st["incumbent_commit"])
            if consec_void >= MAX_CONSEC_VOID:
                print(f"[evolve] {MAX_CONSEC_VOID} consecutive VOID rounds — stopping.")
                break
            continue
        consec_void = 0
        gated += 1
        gate_capability(st, m, mpath, args)          # keep/restore + board + history + status
    print("[evolve] autonomous run finished.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("init")
    p0.add_argument("--hours", type=float, default=6.0)
    p0.add_argument("--target", type=float, default=0.02, help="KEEP improvement bar (fraction)")
    p0.add_argument("--runs", type=int, default=3)
    sub.add_parser("status")
    p2 = sub.add_parser("submit")
    p2.add_argument("--capability", required=True)
    p2.add_argument("--runs", type=int, default=3)
    p2.add_argument("--audit", action="store_true", help="run dp_audit EXACTNESS REPORT on manifest.audit_case")
    p2.add_argument("--audit-exact", action="store_true", help="force exact==B&B in the audit")
    p3 = sub.add_parser("restore"); p3.add_argument("--capability", required=True)
    # autonomous: the loop DRIVES an LLM oracle to implement each round, then gates it
    pr = sub.add_parser("run", help="autonomously drive an LLM oracle for N rounds")
    pr.add_argument("--rounds", type=int, default=1)
    pr.add_argument("--oracle", default="claude", help=f"oracle name {list(ORACLES)}")
    pr.add_argument("--oracle-cmd", default=None,
                    help="explicit oracle command ({prompt} = prompt file); overrides --oracle")
    pr.add_argument("--oracle-timeout", type=int, default=5400,
                    help="seconds allowed for the oracle to implement one capability")
    pr.add_argument("--runs", type=int, default=3, help="HW-eval runs at the gate")
    pr.add_argument("--audit", action="store_true")
    pr.add_argument("--audit-exact", action="store_true")
    args = ap.parse_args()
    {"init": cmd_init, "status": cmd_status, "submit": cmd_submit,
     "restore": cmd_restore, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
