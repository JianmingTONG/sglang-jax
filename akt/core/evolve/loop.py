"""AKT capability-elevation loop for the sglang-jax serving stack.

The loop grows programmer control over performance-critical Pallas/Mosaic
capabilities. A capability is not merely another benchmark parameter: it must form
an end-to-end path from a low-level kernel argument, through a production
layer/backend, into the stable ``KernelControlPolicy`` API, and into an AKT runner
dimension used to measure its value.

Each round:
  1. selects one stable gap_id from the generated flexibility-graph action catalog;
  2. estimates model-level relief and exposes the capability through the full stack;
  3. measures every local configuration and runs exact DP on three frozen models;
  4. empirically challenges the additive DP with complete-plan interaction panels;
  5. pairs candidate and incumbent synthetic serving-trace bundles on target hardware;
  6. conditionally runs the real Kimi-Linear checkpoint on a compatible TPU; and
  7. keeps the change only when every applicable guard passes and aggregate gain
     exceeds 2%.

The frozen harness defines the workload, references, measurement, and exposure
contract. Production kernels, programmer controls, serving consumers, and editable
runners are the implementation surface.

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

import argparse, json, math, os, re, shlex, signal, subprocess, sys, tempfile, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "akt/optimization_history/evolve_state.json"
HIST = ROOT / "akt/optimization_history/evolve_history.jsonl"
CAPS = Path(__file__).resolve().parent / "capabilities"
EVOLVE_JSON = ROOT / "akt/board/evolve.json"
STATUS = ROOT / "akt/board/evolve_status.json"   # live heartbeat -> board WIP time bar
BOOKKEEPING_PATHS = frozenset(
    {
        "akt/optimization_history/evolve_state.json",
        "akt/optimization_history/evolve_history.jsonl",
        "akt/board/evolve_status.json",
    }
)
# All harness subprocesses run under the project venv with the kernel import path
# (python/ for sgl_jax, repo root for akt) and Pallas interpret so the TPU kernels
# that support it run on this non-TPU box.
PY = tuple(shlex.split(os.environ.get("AKT_PY", ".venv/bin/python")))
if not PY:
    raise RuntimeError("AKT_PY must name a Python executable")
ADAPTER = "akt/benchmark/adapter.py"
MODEL_EVAL = "akt/benchmark/gates/model_eval.py"
LIVE_MODEL_EVAL = "akt/benchmark/gates/live_model_eval.py"
OBJECTIVE_SCOPE = "model-serving-empirical-dp-v2"
_CAPABILITY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_REPO_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

# The capability may edit production kernels and runners, but never the gate,
# workload/reference contracts, or maintainer-authored native tests.
FROZEN = (
    "akt/benchmark/",
    "akt/board/build.py",
    "akt/board/index.html",
    "akt/board/chart.umd.min.js",
    "akt/board/test_build.py",
    "akt/core/analysis/flexgraph_extract.py",
    "akt/core/analysis/flexgraph_generated.json",
    "akt/core/evolve/loop.py",
    "akt/core/evolve/exposure.py",
    "akt/core/evolve/action_catalog.py",
    "akt/core/evolve/capability_contract.py",
    "akt/core/search/model_dp.py",
    "python/sgl_jax/test/",
)

# Some TPU-only references currently share a source file with the editable kernel.
# Freeze only their reference AST (plus imports) so the oracle can tune the kernel
# without being able to move the correctness goalpost in the same file.
FROZEN_REFERENCE_SYMBOLS = {
    "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py": (
        "activation_fn",
        "ref_moe",
    ),
    "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py": (
        "swigluoai",
        "activation_fn",
        "ref_moe",
    ),
    "python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py": (
        "DEFAULT_MASK_VALUE",
        "ref_ragged_paged_attention",
    ),
}

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

DOMAIN: the validated flexibility-graph snapshot below is a FROZEN guardrail for this
round, not brainstorming material. Each action is one source-proven, already-existing
low-level selection point that is not exposed to programmers. Its stable gap_id, graph
fingerprint, family, source axis/function/sink, finite candidate domain, evidence, and exact model callsites are
foreign keys in your manifest. You may add plumbing needed to expose that existing
selection point, but may not add a new low-level algorithm, kernel path, schedule mode,
or semantic behavior unrelated to that source axis. Every kernel is a KernelCase
whose deployable configurations become stages in an exact three-model DP. A CAPABILITY must
make one previously programmer-inaccessible low-level behavior controllable through the
whole stack: kernel/backend -> production consumer -> KernelControlPolicy -> runner Knob ->
model planner. A new knob alone, a default change, or widening an existing control is tuning.

{action_contract}

YOUR TASK:
 1. Pick ONE executable red-link gap_id. Use the latest paired \
model/callsite timings to estimate baseline share and expected aggregate relief >{target_pct:.0f}%; \
record the calculation and do not repeat an unchanged failed estimate.
 2. IMPLEMENT it end-to-end. Promote an existing runner/backend argument, or lift the exact
graph-fingerprinted hardcoded selection into an entry argument while preserving its incumbent
default exactly (copy the action's incumbent_value, including JSON type). The runner must use
the action's complete candidate_values domain; do not copy that graph-owned field into the
manifest or invent another value. Do not invent a
replacement implementation. Register a validated control under \
python/sgl_jax/srt/configs/kernel_control.py, and forward that control from a production \
layer/model/backend (python/sgl_jax/srt/layers/**, models/**, or model_executor/**) to the kernel. Then add \
a runner Knob with programmer_control="<family>.<key>". Name the exact \
production kernel function and control in the manifest; the control key is the derived backend argument. \
The frozen evaluator instruments that callable \
and records the actual selected argument, so do not add a self-reported runtime event.
 3. Verify exact semantics against the frozen pure-JAX reference. Run \
`PYTHONPATH=python:. .venv/bin/python akt/benchmark/gates/dp_verify.py`; target-hardware \
model search and paired measurement are run only by the frozen gate.
 4. Write akt/core/evolve/capabilities/<name>.json using the exact schema in that directory. \
It must foreign-key gap_id and action_graph_fingerprint, provide the quantitative \
estimate, describe every search dimension using the minimal schema, and list EVERY \
changed/created file. Set status="pending".

HARD CONSTRAINTS:
 - NEVER edit anything under akt/benchmark/, python/sgl_jax/test/, or the frozen \
flexgraph extractor, action-catalog validator, model DP, loop, capability contract, or \
exposure gate. Pure-JAX \
reference symbols in shared kernel files are fingerprinted too. Changing any VOIDS the round.
 - Every graph-authorized value must be correct. All three models must run with zero deferred cases on \
the target backend; DP must equal bounded brute force; a non-default new value must win and emit a \
matching runtime event; both paired and stored-baseline absolute aggregate model latency must improve by >{target_pct:.0f}%.
 - Live-checkpoint attribution is strict. Kimi-Linear runs only when the selected action changes
the existing GMM-v2 route (`megablox_gmm_kernel:buffer_count:pipeline-depth`); unrelated actions
record an explicit no-attempt `not_applicable` result.
 - Do NOT git commit; leave changes in the working tree + the manifest.
 - Your FINAL line must be exactly:  CAPABILITY: <name>
"""


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def run_argv(argv, *, timeout=3600, env=None):
    """Run a non-shell command and return the loop's standard ``(rc, output)``."""

    result = subprocess.run(
        [str(value) for value in argv],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def harness_env(*, interpret):
    env = os.environ.copy()
    env["PALLAS_INTERPRET"] = "1" if interpret else "0"
    env["PYTHONPATH"] = "python:."
    return env


def git(*args, timeout=60):
    return run_argv(("git", *args), timeout=timeout)


def load():
    return json.loads(STATE.read_text())


def save(st):
    STATE.write_text(json.dumps(st, indent=1))


def git_head():
    rc, out = git("rev-parse", "HEAD", timeout=30)
    return out.strip() if rc == 0 else None


def validate_incumbent_head(state):
    """Require the measured incumbent revision to be the current Git parent."""

    expected = state.get("incumbent_commit")
    actual = git_head()
    if not expected or not actual or actual != expected:
        raise RuntimeError(
            "Git HEAD does not match the measured incumbent; rebaseline before "
            f"submitting another round (state={expected!r}, HEAD={actual!r})"
        )


def _validate_git_commit(commit):
    """Fail closed unless ``commit`` resolves to a commit object."""

    if not isinstance(commit, str) or not commit:
        raise RuntimeError(f"invalid incumbent commit {commit!r}")
    rc, output = git("cat-file", "-e", f"{commit}^{{commit}}", timeout=30)
    if rc:
        raise RuntimeError(
            f"cannot resolve incumbent commit {commit!r}: {output.strip()[-300:]}"
        )


def _tracked_at_commit(commit, relative):
    """Return whether one exact repository path is present at a valid commit."""

    rc, output = git(
        "ls-tree", "-z", "--name-only", commit, "--", relative, timeout=30
    )
    if rc:
        raise RuntimeError(
            f"cannot inspect {relative!r} at {commit!r}: {output.strip()[-300:]}"
        )
    return relative in output.split("\0")


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
        _rc, out = run_argv(
            (*PY, ADAPTER, sub), timeout=180, env=harness_env(interpret=True)
        )
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


def action_space_block():
    """Complete graph-linked action catalog executable by the frozen model gate."""
    d = _adapter_json("gaps", "GAPS")
    if not d:
        return "== RED-LINK ACTION SPACE unavailable (generated graph missing)"
    if d.get("missing"):
        return f"== RED-LINK ACTION SPACE: {d['missing']}"
    lines = d.get("lines", [])
    body = "\n".join(f"   [{i}] {l}" for i, l in enumerate(lines))
    note = f"\n   {d['note']}" if d.get("note") else ""
    fingerprint = d.get("action_graph_fingerprint")
    identity = (
        f"; graph_fingerprint={fingerprint}; context_v={d.get('action_context_version')}"
        if fingerprint
        else ""
    )
    return (f"== RED-LINK ACTION SPACE ({d.get('title', 'flexibilities to expose')})\n"
            f"{body}{note}{identity}")


def action_contract_block():
    """Canonical, machine-checkable graph context handed to the oracle.

    Human-rendered adapter lines are useful for ranking, but they are not an
    authorization boundary.  The fingerprinted records below are the exact foreign
    keys later checked by ``capability_contract`` before any TPU work starts.
    """
    from akt.core.evolve.action_catalog import action_catalog_context

    context = action_catalog_context(ROOT)
    actions = context.get("actions") or []
    if not actions:
        raise RuntimeError("validated flexibility graph has no executable actions")
    return (
        "== FROZEN FLEXIBILITY-GRAPH ACTION CONTRACT\n"
        + json.dumps(context, indent=2, sort_keys=True, allow_nan=False)
    )


def current_action_graph_fingerprint():
    from akt.core.evolve.action_catalog import action_catalog_context

    return action_catalog_context(ROOT)["fingerprint"]


def validate_campaign_action_graph(state):
    expected = state.get("action_graph_fingerprint")
    actual = current_action_graph_fingerprint()
    if not expected:
        raise RuntimeError("campaign has no pinned action-graph fingerprint; rebaseline")
    if actual != expected:
        raise RuntimeError(
            "action graph changed outside an accepted round: "
            f"state={expected}, current={actual}"
        )
    return actual


def failed_estimates_block():
    """Rejected estimates are explicit negative evidence for the next oracle."""
    if not HIST.exists():
        return "== FAILED ESTIMATES: (none)"
    rows = []
    for line in HIST.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("decision") != "reject"
            or record.get("objective_scope") != OBJECTIVE_SCOPE
        ):
            continue
        estimate = record.get("estimate") or {}
        rows.append(
            f"gap_id={record.get('gap_id', '?')} callsite={estimate.get('callsite', '?')} "
            f"estimated={estimate.get('expected_relief_pct', '?')}% "
            f"measured={record.get('improvement_pct', '?')}%: "
            f"{record.get('reason', '')[:120]}"
        )
    return "== FAILED ESTIMATES\n" + ("\n".join("   " + row for row in rows[-12:]) or "   (none)")


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
    scope_ok = st.get("objective_scope") == OBJECTIVE_SCOPE
    cond = (
        "REBASELINE_REQUIRED"
        if not scope_ok
        else "CONTINUE" if time.time() < st["deadline_ts"] else "FINISH(DEADLINE)"
    )
    kept, rejected = _cap_summaries()
    # Scope-less historical state and scope-less manifests must not match each other:
    # evidence is current only when it explicitly names the required objective.
    evidence_scope = OBJECTIVE_SCOPE
    active_kept = [m for m in kept if m.get("objective_scope") == evidence_scope]
    legacy_kept = [m for m in kept if m.get("objective_scope") != evidence_scope]
    active_rejected = [m for m in rejected if m.get("objective_scope") == evidence_scope]
    kepts = "; ".join(
        f"{m['name']}(+{m.get('delta_pct', '?')}%)" for m in active_kept
    ) or "(none yet)"
    legacies = "; ".join(m["name"] for m in legacy_kept) or "(none)"
    rejs = "; ".join(
        f"{m['name']}[{m.get('reject_reason', 'rejected')[:32]}]"
        for m in active_rejected
    ) or "(none)"
    tgt = st["target_improvement"] * 100
    scope_line = f"   objective_scope={st.get('objective_scope', 'legacy-output-only-v0')}"
    scope_line += (
        f" (required={OBJECTIVE_SCOPE}; run `loop.py rebaseline`)\n"
        if not scope_ok
        else "\n"
    )
    return (
        f"== CAPABILITY QUERY round={st['round'] + 1} -> {cond}\n"
        f"{scope_line}"
        f"   action_graph_fingerprint={st.get('action_graph_fingerprint', 'unversioned')}\n"
        f"   incumbent {st['objective_name']}={st['incumbent_geomean']:.6f}{unit} "
        f"(measured local configs={st['incumbent_choices']}); KEEP bar = paired and "
        f"stored-baseline absolute aggregate synthetic-trace improvement > {tgt:.0f}% "
        f"on target hardware\n"
        f"   KEPT under active objective: {kepts}\n"
        f"   LEGACY kept (code is in baseline; old deltas are not evidence): {legacies}\n"
        f"   REJECTED under active objective (do not re-attempt unmodified): {rejs}\n"
        f"{bottleneck_block()}\n"
        f"{action_space_block()}\n"
        f"{failed_estimates_block()}\n"
        f"== DECIDE ONE executable red-link gap_id to elevate. Then, off-loop:\n"
        f"   (1) estimate model-level bottleneck relief; (2) implement it end-to-end across\n"
        f"   backend -> production consumer -> KernelControlPolicy -> runner -> model DP;\n"
        f"   (3) report concrete selected backend execution;\n"
        f"   write the manifest akt/core/evolve/capabilities/<name>.json (schema in README);\n"
        f"   then: python akt/core/evolve/loop.py submit --capability <name>")


# ---------------------------------------------------------------- gate helpers


def _stream_gate(
    command,
    log_path,
    *,
    progress_prefixes,
    capability,
    timeout=12 * 3600,
    env=None,
):
    """Run one long gate with a bounded watchdog, persistent log, and progress."""

    proc = subprocess.Popen(
        [str(value) for value in command],
        shell=False,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def terminate():
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

    watchdog = threading.Timer(timeout, terminate)
    watchdog.start()
    tail = []
    try:
        with log_path.open("w") as log_file:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                rendered = line.strip()
                if not rendered:
                    continue
                tail = [*tail[-59:], rendered]
                if rendered.startswith(progress_prefixes):
                    print(rendered[:500], flush=True)
                    write_status(
                        "running",
                        capability=capability,
                        phase=rendered[:100],
                    )
        proc.wait()
    finally:
        watchdog.cancel()
    if proc.returncode and tail:
        print("\n".join(tail[-20:]))
    return proc.returncode


def _live_model_hw_eval(
    synthetic_path,
    *,
    runs,
    capability=None,
):
    """Run conditional real-checkpoint candidates after the JAX gate has exited."""

    out_path = ROOT / "akt/optimization_history/.live_model_eval.json"
    log_path = ROOT / "akt/optimization_history/.live_model_eval.log"
    out_path.unlink(missing_ok=True)
    command = [
        *PY,
        LIVE_MODEL_EVAL,
        "--synthetic-summary",
        str(synthetic_path),
        "--out",
        str(out_path),
        "--runs",
        str(max(1, int(runs))),
    ]
    if capability:
        command.extend(("--capability", capability))
    print("[evolve] conditional live-model gate (Kimi-Linear on compatible TP4 TPU)")
    rc = _stream_gate(
        command,
        log_path,
        progress_prefixes=(
            "[accuracy-runner]",
            "[perf-runner]",
            "AKT_LIVE_MODEL_EVAL",
        ),
        capability=capability,
        env=harness_env(interpret=False),
    )
    if not out_path.exists():
        return {
            "schema_version": "akt.live-model-eval.v1",
            "suite": "akt-live-model-candidates",
            "status": "failed",
            "all_passed": False,
            "candidates": [],
            "error": f"live-model evaluator produced no output (rc={rc})",
        }
    return json.loads(out_path.read_text())


def live_model_gate_evidence(live_document):
    """Accept direct Kimi evidence, a topology skip, or clean non-applicability."""

    errors = []
    candidates = (
        live_document.get("candidates")
        if isinstance(live_document, dict)
        else None
    )
    if not isinstance(candidates, list) or not candidates:
        errors.append("live-model document contains no candidate records")
        candidates = []
    eligible = []
    topology_skipped = []
    not_applicable = []
    for record in candidates:
        if not isinstance(record, dict):
            errors.append("live-model candidate record is not an object")
            continue
        applicability = record.get("applicability")
        if applicability == "not_applicable":
            not_applicable.append(record)
            if record.get("attempted") is True or record.get("status") != "skipped":
                errors.append("non-applicable Kimi route was not a clean no-attempt skip")
            continue
        if applicability != "direct-capability":
            errors.append(f"unknown live-model applicability {applicability!r}")
            continue
        eligibility = record.get("eligibility") or {}
        is_eligible = eligibility.get("eligible") is True
        if is_eligible:
            eligible.append(record)
            if record.get("attempted") is not True or record.get("status") != "passed":
                candidate_id = (record.get("candidate") or {}).get("candidate_id", "?")
                errors.append(
                    f"eligible candidate {candidate_id} did not pass: "
                    f"status={record.get('status')!r}"
                )
            coverage = record.get("policy_coverage") or {}
            if record.get("status") == "passed" and not (
                coverage.get("capability_applies") is True
                and coverage.get("evaluation_intent") == "direct-capability"
            ):
                errors.append("passing Kimi record lacks direct capability attestation")
        else:
            topology_skipped.append(record)
            if record.get("attempted") is True or record.get("status") != "skipped":
                errors.append("ineligible candidate was not a clean no-attempt skip")
    return {
        "ok": not errors,
        "eligible_candidates": len(eligible),
        "passed_candidates": sum(r.get("status") == "passed" for r in eligible),
        "topology_skipped_candidates": len(topology_skipped),
        "not_applicable_candidates": len(not_applicable),
        "errors": errors,
    }


def model_hw_eval(
    runs,
    incumbent_plans=None,
    capability=None,
):
    """Run synthetic TPU search, then the separate conditional live-model gate."""
    out_path = ROOT / "akt/optimization_history/.evolve_eval.json"
    incumbent_path = ROOT / "akt/optimization_history/.incumbent_plans.json"
    out_path.unlink(missing_ok=True)
    incumbent_path.unlink(missing_ok=True)
    command = [*PY, MODEL_EVAL, "--runs", str(int(runs)), "--out", str(out_path)]
    if incumbent_plans is not None:
        incumbent_path.write_text(json.dumps(incumbent_plans, indent=1))
        command.extend(("--incumbent-plans", str(incumbent_path)))
    if capability:
        command.extend(("--capability", capability))
    log_path = ROOT / "akt/optimization_history/.model_eval.log"
    rc = _stream_gate(
        command,
        log_path,
        progress_prefixes=("[akt-model-eval]",),
        capability=capability,
        env=harness_env(interpret=False),
    )
    if not out_path.exists():
        return {
            "all_correct": False,
            "target_hardware_ok": False,
            "objective_scope": OBJECTIVE_SCOPE,
            "error": f"model evaluator produced no output (rc={rc})",
            "models": [],
        }
    summary = json.loads(out_path.read_text())
    # The synthetic evaluator process has fully exited at this point, so its JAX
    # client and device allocations cannot contend with the live server process.
    live = _live_model_hw_eval(
        out_path,
        runs=runs,
        capability=capability,
    )
    summary["live_model_evaluation"] = live
    summary["live_model_gate"] = live_model_gate_evidence(live)
    out_path.write_text(json.dumps(summary, indent=1, allow_nan=False))
    return summary


def _case_space_inventory(summary):
    return {
        case_id: [measurement["config"] for measurement in result.get("measurements") or []]
        for case_id, result in (summary.get("case_search") or {}).items()
    }


def search_audit(case):
    """Evidence that the ENLARGED design space is actually enumerated (searched,
    not sampled): report the per-case design-space size + knob axes for the audit
    kernel. The exhaustive enumerate-correct-time-argmin in runners/base.search_best
    is optimal within that space by construction. Returns the adapter's space report."""
    rc, out = run_argv(
        (*PY, ADAPTER, "space", "--kernel", case),
        timeout=300,
        env=harness_env(interpret=True),
    )
    tail = "\n".join(out.strip().splitlines()[-18:])
    return rc, tail


def load_manifest(name, require_pending=False):
    if not isinstance(name, str) or not _CAPABILITY_RE.fullmatch(name):
        raise ValueError(f"invalid capability name {name!r}")
    p = CAPS / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no manifest at {p.relative_to(ROOT)} — write it first "
            f"(schema: akt/core/evolve/capabilities/README.md)")
    m = json.loads(p.read_text())
    if m.get("name") != name or (require_pending and m.get("status") != "pending"):
        raise ValueError(
            f"manifest identity/status mismatch: name={m.get('name')!r}, "
            f"status={m.get('status')!r}"
        )
    if require_pending:
        required = {
            "name",
            "gap_id",
            "action_graph_fingerprint",
            "hypothesis",
            "estimate",
            "search_dimensions",
            "files_touched",
            "status",
        }
        missing = sorted(required - set(m))
        if missing:
            raise ValueError(f"pending manifest is missing required fields: {missing}")
        extra = sorted(set(m) - required - {"audit_case"})
        if extra:
            raise ValueError(f"pending manifest has unsupported fields: {extra}")
    if not isinstance(m.get("files_touched"), list) or not m["files_touched"]:
        raise ValueError("manifest files_touched must be a non-empty list")
    if any(not isinstance(rel, str) for rel in m["files_touched"]):
        raise ValueError("manifest files_touched entries must be strings")
    if len(m["files_touched"]) != len(set(m["files_touched"])):
        raise ValueError("manifest files_touched contains duplicates")
    for rel in m["files_touched"]:
        if not _REPO_PATH_RE.fullmatch(rel):
            raise ValueError(f"manifest path contains unsupported characters: {rel!r}")
        path = Path(rel)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != rel
            or rel.startswith("./")
        ):
            raise ValueError(f"manifest path must be normalized and repo-relative: {rel!r}")
    bad = [f for f in m.get("files_touched", [])
           if any(f.startswith(x) or x in f for x in FROZEN)]
    if bad:
        raise PermissionError(
            f"manifest lists FROZEN files (would game the gate): {bad}")
    if require_pending:
        from akt.core.evolve.capability_contract import validate_action_reference

        action = validate_action_reference(m, ROOT)
        if not action.get("ok"):
            raise ValueError(
                "manifest does not select an executable red-link action: "
                + "; ".join(action.get("errors") or [])
            )
    return m, p


def validate_manifest_scope(m, changed_before=()):
    """Require the manifest to name every non-bookkeeping oracle edit, exactly."""
    before = set(changed_before)
    actual = {
        path
        for path in _worktree_changed()
        if path not in before
        and not _is_bookkeeping(path)
    }
    declared = set(m["files_touched"])
    undeclared = sorted(actual - declared)
    untouched = sorted(declared - actual)
    if undeclared or untouched:
        raise ValueError(
            f"files_touched mismatch: undeclared={undeclared}, unchanged={untouched}"
        )


def validate_estimate_threshold(m, st):
    expected = (m.get("estimate") or {}).get("expected_relief_pct")
    required = max(float(st.get("target_improvement", 0.02)), 0.02) * 100
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        raise ValueError("estimate.expected_relief_pct must be numeric")
    if expected <= required:
        raise ValueError(
            f"estimated aggregate relief must be strictly > {required:.1f}%; got {expected}"
        )


def restore_capability(m, incumbent_commit):
    """Revert ONLY the files the capability touched back to the incumbent commit.
    A file tracked at the incumbent is checked out; a NEW file the capability
    created (untracked at incumbent) is removed."""
    _validate_git_commit(incumbent_commit)
    files = list(m.get("files_touched", []))
    for f in files:
        if _tracked_at_commit(incumbent_commit, f):
            checkout_rc, output = git(
                "checkout", incumbent_commit, "--", f, timeout=60
            )
            if checkout_rc:
                raise RuntimeError(f"cannot restore {f!r}: {output.strip()[-300:]}")
        else:                                     # capability-created new file -> remove
            path = ROOT / f
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                raise RuntimeError(f"cannot remove new file {f!r}: {error}") from error
    # Kernels are pure Python/Pallas — no compiled artifact to rebuild on revert.
    print(f"[evolve] restored {len(files)} file(s) to incumbent {incumbent_commit[:8]}")


def write_board():
    # build.py joins evolve_history + manifests + evolve_state + the freshest eval into
    # board.json + flexgraph.json. It is jax-free and reads the generated action graph, so
    # plain python3 suffices — no venv needed for the board refresh.
    sh("python3 akt/board/build.py", timeout=300)


def candidate_action_graph_closure(manifest):
    """Regenerate and validate the graph without replacing the incumbent snapshot.

    A KEEP must close exactly the selected red action in the source-mined catalog.
    The candidate may make the original source pattern disappear by lifting a literal
    into an existing entry argument, so closure means the ID is no longer selectable;
    if the finding remains, it must explicitly report programmer exposure. Unrelated
    incumbent actions may not disappear as collateral damage.
    """
    from akt.core.evolve.action_catalog import (
        action_semantic_record,
        action_catalog_context,
        load_action_catalog,
        load_action_graph,
    )

    gap_id = manifest.get("gap_id")
    before = load_action_catalog(ROOT)
    candidate_path = ROOT / "akt/optimization_history/.candidate_flexgraph.json"
    candidate_path.unlink(missing_ok=True)
    rc, output = run_argv(
        (*PY, "akt/core/analysis/flexgraph_extract.py", "--out", candidate_path),
        timeout=300,
        env=harness_env(interpret=True),
    )
    errors = []
    if rc or not candidate_path.exists():
        errors.append(
            "candidate graph regeneration failed: "
            + (output[-500:] if output else f"exit={rc}")
        )
        return {"ok": False, "gap_id": gap_id, "errors": errors}

    raw = candidate_path.read_text()
    try:
        graph = json.loads(raw)
        with tempfile.TemporaryDirectory(prefix="akt-candidate-graph-") as temp:
            temp_root = Path(temp)
            graph_path = temp_root / "akt/core/analysis/flexgraph_generated.json"
            graph_path.parent.mkdir(parents=True)
            graph_path.write_text(raw)
            _validated_graph, after = load_action_graph(temp_root)
            after_context = action_catalog_context(temp_root)
    except Exception as error:  # noqa: BLE001
        errors.append(f"candidate graph is invalid: {error}")
        return {"ok": False, "gap_id": gap_id, "errors": errors}
    finally:
        candidate_path.unlink(missing_ok=True)

    if gap_id in after:
        errors.append(f"selected action {gap_id!r} remains open after implementation")
    unrelated_missing = sorted((set(before) - {gap_id}) - set(after))
    if unrelated_missing:
        errors.append(
            "candidate closed unrelated graph actions: " + ", ".join(unrelated_missing)
        )
    unrelated_added = sorted(set(after) - set(before))
    if unrelated_added:
        errors.append(
            "candidate introduced unrelated graph actions: " + ", ".join(unrelated_added)
        )
    unrelated_changed = sorted(
        action_id
        for action_id in (set(before) & set(after)) - {gap_id}
        if action_semantic_record(before[action_id])
        != action_semantic_record(after[action_id])
    )
    if unrelated_changed:
        errors.append(
            "candidate changed unrelated graph action semantics: "
            + ", ".join(unrelated_changed)
        )
    finding = next(
        (item for item in graph.get("gaps") or [] if item.get("gap_id") == gap_id),
        None,
    )
    if finding is not None and not (
        finding.get("programmer_exposed") is True or finding.get("open") is False
    ):
        errors.append(
            f"selected finding {gap_id!r} is not marked programmer-exposed/closed"
        )
    return {
        "ok": not errors,
        "gap_id": gap_id,
        "before_action_count": len(before),
        "after_action_count": len(after),
        "unrelated_missing": unrelated_missing,
        "unrelated_added": unrelated_added,
        "unrelated_changed": unrelated_changed,
        "selected_finding": finding,
        "candidate_action_graph_fingerprint": after_context.get("fingerprint"),
        # Retain the validated document in memory for atomic installation only after
        # all other gates choose KEEP; history receives the compact fields above.
        "_graph": graph,
        "errors": errors,
    }


def install_candidate_action_graph(closure):
    graph = closure.get("_graph") if isinstance(closure, dict) else None
    if not isinstance(graph, dict):
        return False
    path = ROOT / "akt/core/analysis/flexgraph_generated.json"
    staged = path.with_suffix(".json.akt-candidate")
    staged.write_text(json.dumps(graph, indent=1, allow_nan=False))
    staged.replace(path)
    print("[evolve] installed validated post-elevation flexibility graph")
    return True


def _commit_kept_candidate(
    manifest,
    manifest_path,
    graph_path,
    incumbent_graph,
    *,
    round_number,
    reason,
    previous_head,
):
    """Commit one accepted candidate, restoring the graph on any Git failure."""

    actual_head = git_head()
    if actual_head != previous_head:
        graph_path.write_text(incumbent_graph)
        return {
            "ok": False,
            "head": previous_head,
            "error": (
                "KEEP COMMIT ABORTED: Git HEAD changed during evaluation "
                f"(expected={previous_head!r}, actual={actual_head!r})"
            ),
        }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    commit_paths = sorted(
        set(manifest["files_touched"])
        | {"akt/core/analysis/flexgraph_generated.json"}
    )
    add_rc, add_output = git("add", "--", *commit_paths, timeout=60)
    message = (
        f"akt evolve R{round_number} KEEP capability={manifest['name']}: "
        f"{reason[:60]}"
    )
    commit_rc, commit_output = (
        git(
            "commit",
            "-q",
            "-m",
            message,
            "--only",
            "--",
            *commit_paths,
            timeout=120,
        )
        if add_rc == 0
        else (add_rc, add_output)
    )
    committed_head = git_head() if commit_rc == 0 else None
    parent = None
    parent_error = ""
    if commit_rc == 0 and committed_head:
        parent_rc, parent_output = git(
            "rev-parse", f"{committed_head}^", timeout=30
        )
        parent = parent_output.strip() if parent_rc == 0 else None
        if parent_rc:
            parent_error = parent_output.strip()[-300:]
    valid_commit = (
        commit_rc == 0
        and committed_head
        and committed_head != previous_head
        and parent == previous_head
    )
    if not valid_commit:
        rollback_errors = []
        if commit_rc == 0 and not committed_head:
            rollback_errors.append("committed HEAD could not be resolved")
        if commit_rc == 0 and committed_head and committed_head != previous_head:
            reset_target = parent or previous_head
            reset_head_rc, reset_head_output = git(
                "reset", "--soft", reset_target, timeout=60
            )
            if reset_head_rc:
                rollback_errors.append(
                    "HEAD rollback failed: " + reset_head_output.strip()[-300:]
                )
        reset_rc, reset_output = git(
            "reset", "-q", "HEAD", "--", *commit_paths, timeout=60
        )
        if reset_rc:
            rollback_errors.append(
                "index rollback failed: " + reset_output.strip()[-300:]
            )
        try:
            graph_path.write_text(incumbent_graph)
        except OSError as error:
            rollback_errors.append(f"graph restoration failed: {error}")
        if rollback_errors:
            raise RuntimeError(
                "KEEP transaction failed and cleanup was incomplete: "
                + "; ".join(rollback_errors)
            )
        detail = (commit_output or add_output or parent_error).strip()[-300:]
        if commit_rc == 0 and parent != previous_head:
            detail = (
                "commit parent mismatch: "
                f"expected={previous_head!r}, actual={parent!r}"
            )
        return {
            "ok": False,
            "head": previous_head,
            "error": "KEEP COMMIT FAILED" + (f": {detail}" if detail else ""),
        }
    return {"ok": True, "head": committed_head, "error": None}


# ---------------------------------------------------------------- commands

def cmd_init(args):
    dirty = _implementation_changes()
    if dirty:
        print("[evolve] init needs a clean implementation tree.")
        print(f"         commit or stash these first ({len(dirty)}): {dirty[:8]}"
              + (" ..." if len(dirty) > 8 else ""))
        return
    summary = model_hw_eval(args.runs)
    models = summary.get("models") or []
    geo = summary.get("candidate_geomean_s")
    if (
        not summary.get("all_correct")
        or not summary.get("target_hardware_ok")
        or not (summary.get("live_model_gate") or {}).get("ok")
        or summary.get("n_deferred") != 0
        or summary.get("objective_scope") != OBJECTIVE_SCOPE
        or len(models) != 3
        or not isinstance(geo, (int, float))
    ):
        print(
            "[evolve] init ABORTED: strict three-model target-hardware gate failed: "
            + str(summary.get("error") or {
                "all_correct": summary.get("all_correct"),
                "target_hardware_ok": summary.get("target_hardware_ok"),
                "n_deferred": summary.get("n_deferred"),
                "models": len(models),
            })
        )
        return
    CAPS.mkdir(exist_ok=True)
    incumbent_plans = {model["model"]: model["selected_plan"] for model in models}
    expected_callsites = sorted(
        site for model in models for site in model.get("callsites") or []
    )
    st = {"start_ts": time.time(), "deadline_ts": time.time() + args.hours * 3600,
          "round": 0, "target_improvement": max(args.target, 0.02),
          "incumbent_geomean": geo, "incumbent_choices": summary.get("n_choices"),
          "base_search_geomean": geo, "incumbent_plans": incumbent_plans,
          "incumbent_case_spaces": _case_space_inventory(summary),
          "incumbent_commit": git_head(),
          "expected_models": sorted(model["model"] for model in models),
          "expected_callsites": expected_callsites,
          "workload_fingerprint": summary.get("workload_fingerprint"),
          "model_contract_fingerprint": summary.get("model_contract_fingerprint"),
          "action_graph_fingerprint": current_action_graph_fingerprint(),
          "target_hardware_fingerprint": (summary.get("target_hardware") or {}).get("fingerprint"),
          "objective_name": "paired_model_geomean_s", "objective_unit": "s",
          "objective_scope": summary.get("objective_scope"),
          "objective_baseline_geomean": geo,
          "objective_baseline_round": 0}
    save(st)
    write_status("idle", round=0, reset=True)   # start a fresh campaign clock
    print(f"[evolve] initialized: paired three-model incumbent geomean={geo*1e3:.2f}ms "
          f"| measured local configs={summary.get('n_choices')} "
          f"commit={str(st['incumbent_commit'])[:8]}, "
          f"KEEP bar > {st['target_improvement']*100:.0f}%.")
    print(query(st))


def cmd_status(_):
    print(query(load()))


def cmd_rebaseline(args):
    """Migrate history to the strict three-model target-hardware objective."""
    dirty = _implementation_changes()
    if dirty:
        print(
            "[evolve] rebaseline needs a CLEAN code tree so rollback points at the "
            f"measured revision; commit or stash these first ({len(dirty)}): {dirty[:8]}"
        )
        return
    st = load()
    summary = model_hw_eval(args.runs)
    models = summary.get("models") or []
    geo = summary.get("candidate_geomean_s")
    if (
        not summary.get("all_correct")
        or not summary.get("target_hardware_ok")
        or not (summary.get("live_model_gate") or {}).get("ok")
        or summary.get("n_deferred") != 0
        or summary.get("objective_scope") != OBJECTIVE_SCOPE
        or len(models) != 3
        or not isinstance(geo, (int, float))
    ):
        print(
            "[evolve] rebaseline ABORTED: strict three-model target-hardware gate failed: "
            + str(summary.get("error") or summary.get("target_hardware"))
        )
        return
    st.update(
        incumbent_geomean=geo,
        incumbent_choices=summary.get("n_choices"),
        incumbent_plans={model["model"]: model["selected_plan"] for model in models},
        incumbent_case_spaces=_case_space_inventory(summary),
        expected_models=sorted(model["model"] for model in models),
        expected_callsites=sorted(
            site for model in models for site in model.get("callsites") or []
        ),
        workload_fingerprint=summary.get("workload_fingerprint"),
        model_contract_fingerprint=summary.get("model_contract_fingerprint"),
        action_graph_fingerprint=current_action_graph_fingerprint(),
        target_hardware_fingerprint=(summary.get("target_hardware") or {}).get("fingerprint"),
        objective_name="paired_model_geomean_s",
        objective_scope=summary.get("objective_scope"),
        objective_baseline_geomean=geo,
        objective_baseline_round=st.get("round", 0),
        deadline_ts=time.time() + args.hours * 3600,
        incumbent_commit=git_head(),
        target_improvement=max(st.get("target_improvement", 0.02), 0.02),
    )
    save(st)
    write_status("idle", round=st.get("round", 0))
    write_board()
    print(
        f"[evolve] rebaselined round {st.get('round', 0)} to {OBJECTIVE_SCOPE}: "
        f"three-model geomean={geo * 1e3:.2f}ms, "
        f"measured local configs={summary.get('n_choices')}. Earlier measurements are legacy."
    )


def gate_capability(st, m, mpath, args):
    """Run every proof required for one model-level capability round."""
    st["round"] += 1
    rnd = st["round"]
    incumbent = st["incumbent_geomean"]
    write_status("running", round=rnd, capability=m["name"], phase="starting")
    print(
        f"== ROUND {rnd} SUBMIT capability='{m['name']}' "
        f"gap_id='{m.get('gap_id', '?')}'"
    )

    write_status(
        "running",
        round=rnd,
        capability=m["name"],
        phase="post-implementation graph closure",
    )
    try:
        graph_closure = candidate_action_graph_closure(m)
    except Exception as error:  # noqa: BLE001
        graph_closure = {
            "ok": False,
            "gap_id": m.get("gap_id"),
            "errors": [f"candidate graph validation failed: {error}"],
        }
    graph_closure_record = {
        key: value for key, value in graph_closure.items() if key != "_graph"
    }

    # (3) enlarged-space enumeration evidence for the affected kernel family
    audit_tail = ""
    if args.audit and m.get("audit_case"):
        write_status("running", round=rnd, capability=m["name"], phase="enlarged-search audit")
        _rc, audit_tail = search_audit(m["audit_case"])
        print("== ENLARGED-SPACE ENUMERATION (affected kernel family):")
        print("\n".join("   " + l for l in audit_tail.splitlines()))

    # Exhaustive local measurement -> exact model DP -> bounded brute certificate ->
    # paired synthetic serving-trace timing. model_eval fails immediately off target hardware.
    write_status(
        "running",
        round=rnd,
        capability=m["name"],
        phase="target-HW three-model search and paired evaluation",
    )
    if graph_closure.get("ok"):
        summary = model_hw_eval(
            args.runs,
            incumbent_plans=st.get("incumbent_plans"),
            capability=m["name"],
        )
    else:
        print(
            "[evolve] target-HW evaluation skipped: graph closure failed: "
            + "; ".join(graph_closure.get("errors") or [])
        )
        summary = {
            "all_correct": False,
            "target_hardware_ok": False,
            "n_deferred": 0,
            "objective_scope": OBJECTIVE_SCOPE,
            "models": [],
            "case_search": {},
            "runtime_evidence": {
                "required": True,
                "ok": False,
                "errors": ["not evaluated after graph-closure failure"],
            },
            "error": "post-implementation flexibility graph did not close",
        }
    models = summary.get("models") or []
    geo = summary.get("candidate_geomean_s")
    paired_ratio = summary.get("paired_ratio_geomean")
    tgt = st["target_improvement"]
    delta_pct = (
        round((paired_ratio - 1) * 100, 2)
        if isinstance(paired_ratio, (int, float)) and math.isfinite(paired_ratio)
        else None
    )
    improvement_pct = -delta_pct if delta_pct is not None else None
    absolute_improvement_pct = (
        round((1 - geo / incumbent) * 100, 2)
        if isinstance(geo, (int, float))
        and math.isfinite(geo)
        and isinstance(incumbent, (int, float))
        and incumbent > 0
        else None
    )

    # --- PROGRAMMER-EXPOSURE GUARD ---------------------------------------------
    # A benchmark runner dimension is not itself a capability elevation. Require
    # the same control to exist in the stable API registry and to be forwarded by
    # a production layer/backend into the low-level kernel argument.
    write_status(
        "running",
        round=rnd,
        capability=m["name"],
        phase="programmer exposure gate",
    )
    from akt.core.evolve.exposure import validate_programmer_exposure

    exposure = validate_programmer_exposure(m, summary, ROOT)
    exposure_ok = bool(exposure.get("ok"))

    write_status(
        "running", round=rnd, capability=m["name"], phase="gap and prior-access contract"
    )
    from akt.core.evolve.capability_contract import (
        validate_capability_contract,
        validate_space_extension,
    )

    capability_contract = validate_capability_contract(
        m,
        summary,
        ROOT,
        st["incumbent_commit"],
    )
    space_extension = validate_space_extension(
        m,
        summary,
        st.get("incumbent_case_spaces") or {},
        ROOT,
    )
    runtime_evidence = summary.get("runtime_evidence") or {}
    case_search = summary.get("case_search") or {}
    failing_configs = {
        case_id: result.get("failures") or []
        for case_id, result in case_search.items()
        if not result.get("all_configs_correct")
    }
    model_accuracy = {
        model.get("model"): {
            "correct": model.get("selected_correct"),
            "errors": model.get("correctness_errors") or [],
            "candidate_s": model.get("candidate_s"),
            "incumbent_s": model.get("incumbent_s"),
            "ratio": model.get("ratio"),
            "selected_plan": model.get("selected_plan"),
        }
        for model in models
    }
    present_models = {model.get("model") for model in models if model.get("model")}
    expected_models = set(st.get("expected_models") or [])
    present_callsites = {
        site for model in models for site in model.get("callsites") or []
    }
    expected_callsites = set(st.get("expected_callsites") or [])
    missing_models = sorted(expected_models - present_models)
    added_models = sorted(present_models - expected_models)
    missing_callsites = sorted(expected_callsites - present_callsites)
    added_callsites = sorted(present_callsites - expected_callsites)
    certificates = {
        model.get("model"): model.get("certificate") or {}
        for model in models
    }
    bad_certificates = sorted(
        model
        for model, certificate in certificates.items()
        if not certificate.get("certified_additive_optimum")
        or not certificate.get("bounded_dp_matches_bruteforce")
    )
    empirical_dp = summary.get("empirical_dp_certificate") or {}
    empirical_dp_ok = (
        empirical_dp.get("scope") == "synthetic_trace_additivity"
        and empirical_dp.get("ok") is True
        and len(empirical_dp.get("models") or []) == 3
    )
    live_model_evaluation = summary.get("live_model_evaluation") or {}
    live_model_gate = summary.get("live_model_gate") or {}
    live_model_ok = live_model_gate.get("ok") is True
    paired_ok = (
        len(models) == 3
        and all(
            isinstance(model.get("candidate_s"), (int, float))
            and isinstance(model.get("incumbent_s"), (int, float))
            and model["candidate_s"] > 0
            and model["incumbent_s"] > 0
            for model in models
        )
    )
    target_hardware = summary.get("target_hardware") or {}
    hardware_matches = (
        target_hardware.get("fingerprint") == st.get("target_hardware_fingerprint")
    )
    workload_matches = (
        summary.get("workload_fingerprint") == st.get("workload_fingerprint")
        and summary.get("model_contract_fingerprint")
        == st.get("model_contract_fingerprint")
    )
    guard_ok = (
        bool(graph_closure.get("ok"))
        and bool(summary.get("all_correct"))
        and bool(summary.get("target_hardware_ok"))
        and summary.get("n_deferred") == 0
        and not failing_configs
        and all(model.get("selected_correct") for model in models)
        and not missing_models
        and not added_models
        and not missing_callsites
        and not added_callsites
        and not bad_certificates
        and empirical_dp_ok
        and live_model_ok
        and paired_ok
        and workload_matches
        and hardware_matches
        and exposure_ok
        and capability_contract.get("ok")
        and space_extension.get("ok")
        and runtime_evidence.get("ok")
        and summary.get("objective_scope") == st.get("objective_scope") == OBJECTIVE_SCOPE
        and isinstance(geo, (int, float))
        and math.isfinite(geo)
    )

    if not guard_ok:
        decision = "reject"
        if not graph_closure.get("ok"):
            reason = (
                "FLEXIBILITY-GRAPH CLOSURE GUARD FAILED: "
                + "; ".join(graph_closure.get("errors") or [])
            )
        elif not summary.get("target_hardware_ok"):
            reason = "TARGET-HARDWARE GUARD FAILED: " + str(
                summary.get("error") or target_hardware
            )
        elif summary.get("n_deferred") != 0:
            reason = f"NO-DEFERRED GUARD FAILED: {summary.get('n_deferred')} call(s) deferred"
        elif failing_configs:
            reason = "ALL-CONFIG CORRECTNESS GUARD FAILED: " + ", ".join(failing_configs)
        elif not all(model.get("selected_correct") for model in models):
            reason = "MODEL CORRECTNESS GUARD FAILED: " + repr(model_accuracy)
        elif missing_models or added_models or missing_callsites or added_callsites:
            reason = (
                "FROZEN-WORKLOAD GUARD FAILED: "
                f"models missing={missing_models} added={added_models}; "
                f"calls missing={missing_callsites} added={added_callsites}"
            )
        elif not workload_matches:
            reason = "FROZEN-WORKLOAD FINGERPRINT CHANGED"
        elif not hardware_matches:
            reason = "TARGET-HARDWARE FINGERPRINT CHANGED FROM INCUMBENT"
        elif bad_certificates:
            reason = "DP/BRUTE OPTIMALITY GUARD FAILED: " + ", ".join(bad_certificates)
        elif not empirical_dp_ok:
            reason = (
                "EMPIRICAL ADDITIVE-DP GUARD FAILED: "
                + str(empirical_dp.get("status") or empirical_dp)
            )
        elif not live_model_ok:
            reason = (
                "LIVE KIMI-LINEAR GUARD FAILED: "
                + "; ".join(live_model_gate.get("errors") or [
                    str(live_model_evaluation.get("status") or "missing result")
                ])
            )
        elif not paired_ok:
            reason = "PAIRED SYNTHETIC SERVING-TRACE MEASUREMENT FAILED"
        elif not exposure_ok:
            reason = (
                "PROGRAMMER-EXPOSURE GUARD FAILED: "
                + "; ".join(exposure.get("errors") or ["missing exposure evidence"])
            )
        elif not capability_contract.get("ok"):
            reason = (
                "CAPABILITY-CONTRACT GUARD FAILED: "
                + "; ".join(capability_contract.get("errors") or [])
            )
        elif not space_extension.get("ok"):
            reason = (
                "DESIGN-SPACE EXTENSION GUARD FAILED: "
                + "; ".join(space_extension.get("errors") or [])
            )
        elif not runtime_evidence.get("ok"):
            reason = (
                "SELECTED RUNTIME-EVIDENCE GUARD FAILED: "
                + "; ".join(runtime_evidence.get("errors") or [])
            )
        elif summary.get("objective_scope") != st.get("objective_scope"):
            reason = (
                "OBJECTIVE-SCOPE GUARD FAILED: state="
                f"{st.get('objective_scope')!r}, eval={summary.get('objective_scope')!r}"
            )
        else:
            reason = "model gate FAILED (evaluation error or non-finite objective)"
    elif (
        improvement_pct is not None
        and absolute_improvement_pct is not None
        and improvement_pct > tgt * 100
        and absolute_improvement_pct > tgt * 100
    ):
        decision, reason = (
            "keep",
            f"paired three-model candidate beats incumbent by {improvement_pct:.2f}% "
            f"(paired={improvement_pct:.2f}%, absolute={absolute_improvement_pct:.2f}%; "
            f"both strictly > {tgt * 100:.0f}%)",
        )
    else:
        decision, reason = (
            "reject",
            f"insufficient aggregate gain (paired={improvement_pct}%, "
            f"absolute={absolute_improvement_pct}%; both must exceed {tgt * 100:.0f}%)",
        )

    graph_path = ROOT / "akt/core/analysis/flexgraph_generated.json"
    incumbent_graph = graph_path.read_text()
    if decision == "keep" and not install_candidate_action_graph(graph_closure):
        decision = "reject"
        guard_ok = False
        reason = "FLEXIBILITY-GRAPH INSTALL GUARD FAILED"

    if decision == "keep":
        m.update(
            status="kept",
            delta_pct=improvement_pct,
            measured_geomean_s=geo,
            objective_scope=summary.get("objective_scope"),
        )
        previous_head = st["incumbent_commit"]
        commit_result = _commit_kept_candidate(
            m,
            mpath,
            graph_path,
            incumbent_graph,
            round_number=rnd,
            reason=reason,
            previous_head=previous_head,
        )
        if not commit_result["ok"]:
            decision = "reject"
            guard_ok = False
            reason = commit_result["error"]
        else:
            st.update(
                incumbent_geomean=geo,
                incumbent_choices=summary.get("n_choices"),
                incumbent_plans={
                    model["model"]: model["selected_plan"] for model in models
                },
                incumbent_case_spaces=_case_space_inventory(summary),
                action_graph_fingerprint=graph_closure.get(
                    "candidate_action_graph_fingerprint"
                ),
                incumbent_commit=commit_result["head"],
            )

    if decision == "reject":
        m.pop("delta_pct", None)
        m.pop("measured_geomean_s", None)
        m.update(
            status="rejected",
            reject_reason=reason,
            objective_scope=summary.get("objective_scope"),
        )
        restore_capability(m, st["incumbent_commit"])
        mpath.write_text(json.dumps(m, indent=1))

    # Persist bookkeeping only after KEEP has both installed the graph and committed
    # successfully. A commit failure follows the same rollback path as any rejection.
    summary.update(
        gate_decision=decision,
        gate_reason=reason,
        capability=m["name"],
        improvement_pct=improvement_pct,
        absolute_improvement_pct=absolute_improvement_pct,
        retained_plan_role="selected_plan" if decision == "keep" else "incumbent_plan",
    )
    (ROOT / "akt/optimization_history/.evolve_eval.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False)
    )
    rec = {
        "round": rnd,
        "capability": m["name"],
        "gap_id": m.get("gap_id"),
        "action_graph_fingerprint": m.get("action_graph_fingerprint"),
        "action_snapshot": capability_contract.get("gap"),
        "hypothesis": m.get("hypothesis"),
        "estimate": m.get("estimate"),
        "search_dimensions": capability_contract.get("derived_dimensions")
        or m.get("search_dimensions"),
        "files_touched": m.get("files_touched"),
        "correct": guard_ok,
        "accuracy": model_accuracy,
        "geomean_s": geo,
        "incumbent_geomean": incumbent,
        "improvement_pct": improvement_pct,
        "absolute_improvement_pct": absolute_improvement_pct,
        "n_verified": len(present_callsites),
        "n_deferred": summary.get("n_deferred"),
        "model_results": model_accuracy,
        "dp_certificates": certificates,
        "empirical_dp_certificate": empirical_dp,
        "live_model_gate": live_model_gate,
        "runtime_evidence": runtime_evidence,
        "graph_closure": graph_closure_record,
        "programmer_exposure": exposure,
        "capability_contract": capability_contract,
        "space_extension": space_extension,
        "target_hardware": target_hardware,
        "workload_fingerprint": summary.get("workload_fingerprint"),
        "objective_scope": summary.get("objective_scope"),
        "target_pct": tgt * 100,
        "decision": decision,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    HIST.open("a").write(json.dumps(rec, allow_nan=False) + "\n")
    save(st)
    write_board()

    write_status("idle", round=rnd, capability=m["name"], last={
        "round": rnd, "capability": m["name"], "decision": decision, "reason": reason,
        "geomean_s": geo if isinstance(geo, (int, float)) and math.isfinite(geo) else None,
        "delta_pct": delta_pct,
        "finished_ts": time.time()})
    unit = st["objective_unit"]
    shown_geo = geo if isinstance(geo, (int, float)) else float("nan")
    print(f"== ROUND {rnd} {decision.upper()}: {reason} | geomean_s={shown_geo:.6f}{unit} "
          f"incumbent={st['incumbent_geomean']:.4f}{unit}")
    print(query(st))
    return decision


def cmd_submit(args):
    """Manual round: the capability is already implemented in the tree + its manifest."""
    st = load()
    if st.get("objective_scope") != OBJECTIVE_SCOPE:
        raise RuntimeError("campaign objective is stale; run loop.py rebaseline first")
    if (
        not st.get("incumbent_plans")
        or not st.get("incumbent_case_spaces")
        or len(st.get("expected_models") or []) != 3
    ):
        raise RuntimeError("campaign lacks a certified three-model incumbent; run rebaseline")
    validate_incumbent_head(st)
    validate_campaign_action_graph(st)
    m, mpath = load_manifest(args.capability, require_pending=True)
    validate_estimate_threshold(m, st)
    validate_manifest_scope(m)
    gate_capability(st, m, mpath, args)


def cmd_restore(args):
    st = load()
    validate_incumbent_head(st)
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
    rc, out = git("status", "--porcelain", "--untracked-files=all", timeout=60)
    if rc:
        raise RuntimeError(f"cannot inspect Git worktree: {out.strip()[-300:]}")
    paths = []
    for l in out.splitlines():
        p = l[3:].strip()
        if p:
            paths.append(p.split(" -> ")[-1].strip('"'))
    return paths


def _is_bookkeeping(path):
    if path in BOOKKEEPING_PATHS:
        return True
    prefix = "akt/core/evolve/capabilities/"
    if not path.startswith(prefix) or not path.endswith(".json"):
        return False
    try:
        manifest = json.loads((ROOT / path).read_text())
    except Exception:
        return False
    return (
        manifest.get("status") == "rejected"
        and manifest.get("name") == Path(path).stem
    )


def _implementation_changes():
    return [path for path in _worktree_changed() if not _is_bookkeeping(path)]


def touched_frozen():
    return [p for p in _worktree_changed()
            if any(p.startswith(x) or x in p for x in FROZEN)]


def _frozen_fingerprint():
    """Hash frozen files and shared-file reference AST before/after the oracle."""
    import ast
    import hashlib

    fp = {}
    files = []

    def fingerprintable(path):
        return (
            "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )

    for x in FROZEN:
        p = ROOT / x
        if p.is_dir():
            files += [q for q in p.rglob("*") if q.is_file() and fingerprintable(q)]
        elif p.is_file():
            files.append(p)
    # The loop owns these mutable records, but the oracle must not rewrite them
    # while it is running between this function's pre/post snapshots.
    files += [p for p in (STATE, HIST) if p.is_file()]
    for q in files:
        try:
            fp[str(q)] = hashlib.sha256(q.read_bytes()).hexdigest()
        except Exception:
            pass
    for rel, names in FROZEN_REFERENCE_SYMBOLS.items():
        q = ROOT / rel
        key = f"{q}::frozen-reference-ast"
        try:
            tree = ast.parse(q.read_text())

            def import_names(node):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    return set()
                return {
                    alias.asname or alias.name.split(".")[0]
                    for alias in node.names
                }

            imports = [
                node
                for node in tree.body
                if import_names(node).intersection({"jax", "jnp", "lax"})
            ]

            def declared_names(node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    return {node.name}
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    return {t.id for t in targets if isinstance(t, ast.Name)}
                return set()

            selected = [
                node for node in tree.body if declared_names(node).intersection(names)
            ]
            found = set().union(*(declared_names(node) for node in selected))
            missing = sorted(set(names) - found)
            if missing:
                raise ValueError(f"missing frozen symbols: {missing}")
            payload = ast.dump(
                ast.Module(body=imports + selected, type_ignores=[]),
                include_attributes=False,
            ).encode()
            fp[key] = hashlib.sha256(payload).hexdigest()
        except Exception as exc:  # a missing/unparseable reference must also trip the diff
            fp[key] = f"ERROR:{type(exc).__name__}:{exc}"
    return fp


def _frozen_diff(pre):
    post = _frozen_fingerprint()

    def display(key):
        path, sep, suffix = key.partition("::")
        rel = str(Path(path).relative_to(ROOT))
        return f"{rel}::{suffix}" if sep else rel

    return sorted(display(k) for k in set(pre) | set(post) if pre.get(k) != post.get(k))


def revert_worktree(incumbent):
    """Discard a void round's uncommitted edits back to the incumbent commit, EXCEPT
    the loop's own bookkeeping (optimization_history / board), so history survives."""
    _validate_git_commit(incumbent)
    for f in _worktree_changed():
        if _is_bookkeeping(f):
            continue
        if _tracked_at_commit(incumbent, f):
            checkout_rc, output = git("checkout", incumbent, "--", f, timeout=60)
            if checkout_rc:
                raise RuntimeError(f"cannot revert {f!r}: {output.strip()[-300:]}")
        else:
            try:
                (ROOT / f).unlink(missing_ok=True)
            except OSError as error:
                raise RuntimeError(f"cannot remove untracked file {f!r}: {error}") from error


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
    # Fail closed before launching an unconstrained code-writing process.  The
    # manifest gate independently recomputes this context, so a stale prompt or a
    # graph changed during the round cannot authorize a proposal.
    contract = action_contract_block()
    pf.write_text(
        ORACLE_PROMPT.format(
            query=query(st),
            action_contract=contract,
            target_pct=st["target_improvement"] * 100,
        )
    )
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
    current = load()
    if current.get("objective_scope") != OBJECTIVE_SCOPE:
        print("[evolve] campaign objective is stale; run `loop.py rebaseline` first.")
        return
    if (
        not current.get("incumbent_plans")
        or not current.get("incumbent_case_spaces")
        or len(current.get("expected_models") or []) != 3
    ):
        print("[evolve] campaign has no certified three-model incumbent; run `loop.py rebaseline`.")
        return
    try:
        validate_campaign_action_graph(current)
        action_contract_block()
    except Exception as error:  # noqa: BLE001
        print(
            "[evolve] oracle NOT started: flexibility-graph guardrail is invalid: "
            f"{error}"
        )
        return
    try:
        validate_incumbent_head(current)
    except RuntimeError as error:
        print(f"[evolve] oracle NOT started: {error}")
        return
    dirty = _implementation_changes()
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
        changed_before = set(_worktree_changed())
        pre_fp = _frozen_fingerprint()
        name, rate_reset = invoke_oracle(st, args)
        frozen_changed = _frozen_diff(pre_fp)        # ONLY the oracle's frozen edits
        if rate_reset and not name:
            # A rate-limited oracle may already have made partial edits. Never
            # carry unmanifested work into the retry or bypass frozen integrity.
            revert_worktree(st["incumbent_commit"])
            rate_reason = (
                f"oracle modified FROZEN harness {frozen_changed[:2]} before rate limit"
                if frozen_changed
                else "oracle rate-limited; partial edits reverted"
            )
            # ORACLE RATE-LIMITED (e.g. the 5h window): don't burn rounds hammering the
            # API — sleep until the reset (+2 min slack), capped at the campaign deadline.
            wake = rate_reset + 120
            if wake >= st["deadline_ts"]:
                print(f"[evolve] rate-limit reset ({time.strftime('%H:%M', time.localtime(rate_reset))}) "
                      f"is at/after the deadline — stopping.")
                write_status("idle", round=st["round"], last={
                    "round": nxt, "capability": None, "decision": "rate-limited",
                    "reason": rate_reason, "finished_ts": time.time()})
                break
            wait = max(0, wake - time.time())
            if not wait:                              # reset already passed -> retry now
                write_status("idle", round=st["round"], last={
                    "round": nxt, "capability": None, "decision": "rate-limited",
                    "reason": rate_reason, "finished_ts": time.time()})
                continue
            print(f"[evolve] oracle RATE-LIMITED; sleeping {int(wait/60)} min until "
                  f"{time.strftime('%H:%M:%S', time.localtime(wake))}.")
            write_status("idle", round=st["round"], last={
                "round": nxt, "capability": None, "decision": "rate-limited",
                "reason": rate_reason + "; sleeping until "
                + time.strftime('%H:%M', time.localtime(wake)),
                "finished_ts": time.time()})
            time.sleep(wait)
            continue                                  # retry the round after the reset
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
            m, mpath = load_manifest(name, require_pending=True)
            validate_estimate_threshold(m, st)
            validate_manifest_scope(m, changed_before)
        except Exception as e:
            consec_void += 1
            reason = f"bad manifest '{name}': {e}"
            print(f"[evolve] VOID round: {reason} — reverting "
                  f"({consec_void}/{MAX_CONSEC_VOID} consecutive).")
            revert_worktree(st["incumbent_commit"])
            write_status("idle", round=st["round"], last={
                "round": nxt, "capability": name, "decision": "void",
                "reason": reason, "finished_ts": time.time()})
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
    pb = sub.add_parser("rebaseline", help="preserve history and adopt the current objective")
    pb.add_argument("--runs", type=int, default=3)
    pb.add_argument("--hours", type=float, default=6.0)
    sub.add_parser("status")
    p2 = sub.add_parser("submit")
    p2.add_argument("--capability", required=True)
    p2.add_argument("--runs", type=int, default=3)
    p2.add_argument("--audit", action="store_true", help="print enlarged-space enumeration evidence")
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
    args = ap.parse_args()
    {"init": cmd_init, "rebaseline": cmd_rebaseline, "status": cmd_status, "submit": cmd_submit,
     "restore": cmd_restore, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
