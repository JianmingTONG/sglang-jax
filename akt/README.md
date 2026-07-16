# AKT — the capability-elevation loop (sglang-jax kernels)

AKT is an autonomous loop that improves the sglang-jax TPU-Pallas kernels by
**elevating one low-level flexibility per round** through the production serving path
and stable programmer API, then keeping the change only if exposure, correctness, and
performance gates all pass. A Claude/LLM session is the *oracle* that implements each
round; the loop owns the trajectory, measurement, and keep/restore rule.

```
init      measure the incumbent (best config in the deployable design space)
rebaseline
          preserve campaign history but replace a legacy objective with the
          current stateful-serving, deployable-space baseline
run       for each round: hand the QUERY to the oracle -> it elevates ONE gap ->
          re-search the enlarged space -> gate (>2% on the suite) -> KEEP or REVERT
status    print the current Capability QUERY (bottleneck + ranked gap frontier)
submit    gate a manually-implemented capability
```

```bash
python akt/core/evolve/loop.py init  --hours 6 --target 0.02
python akt/core/evolve/loop.py rebaseline --hours 6 --runs 3   # legacy campaigns
python akt/core/evolve/loop.py run   --rounds 3 --oracle claude   # autonomous
python akt/core/evolve/loop.py status                             # see the QUERY
python akt/core/evolve/loop.py submit --capability <name>         # manual
```

---

## Running the loop — setup, execution, and memory

### Before you launch — environment + setup scripts

Environment (this AI-kernel port, `/home/ubuntu/work/sglang-jax`): the project
**`.venv`** (jax 0.8.1) — *not* the conda `jaxite` env, which belongs to the FHE/maple
tree — with **`PALLAS_INTERPRET=1`** and **`PYTHONPATH="python:."`**. There is **no
Go/Lattigo build** (kernels are pure Pallas). The loop injects those two env vars into
its own subprocesses, so you only need to export them for **hand-run** commands (the
adapter / eval / extractor); a bare `loop.py …` invocation sets them itself.

Run these in order before `run`:

| # | Command | Required? | Why |
|---|---------|-----------|-----|
| 1 | `cd /home/ubuntu/work/sglang-jax` | **yes** | all loop paths are repo-relative and the git keep/restore logic assumes the repo root is CWD |
| 2 | `.venv/bin/python -c "import jax; print(jax.__version__)"` | optional | sanity-check the interpreter the loop shells out to (expect `0.8.1`) |
| 3 | `PALLAS_INTERPRET=1 PYTHONPATH="python:." .venv/bin/python akt/core/analysis/flexgraph_extract.py` | recommended | refresh the initial auto-derived gap frontier (`flexgraph_generated.json`). The loop refreshes it after every KEEP; running it here makes the first QUERY reflect source edits made outside AKT |
| 4 | `git status --porcelain` | **yes** | `init`, `rebaseline`, and `run` require a **clean implementation tree** because the incumbent commit is the rollback point. Commit or stash code changes first |
| 5a | `.venv/bin/python akt/core/evolve/loop.py init --hours 6 --target 0.02 --runs 3` | **fresh campaign only** | exhaustively measures the best **deployable** configuration under the stateful-serving objective and writes `evolve_state.json` plus `.evolve_eval.json` |
| 5b | `.venv/bin/python akt/core/evolve/loop.py rebaseline --hours 6 --runs 3` | **existing legacy campaign only** | preserves rounds/manifests but replaces the old output-only incumbent with a correctness-checked stateful-serving baseline. `status` and the board explicitly say `REBASELINE_REQUIRED` / `OBJECTIVE STALE` when this migration is required |
| 6 | `.venv/bin/python akt/core/evolve/loop.py run --rounds N --oracle claude` | **yes** | the target command (autonomous mode needs the `claude` CLI on `PATH`) |

Do not use `init` merely to renew an existing campaign: it starts a new round-zero
ratchet. Use `rebaseline` to retain historical evidence while changing objective scope.
Both commands refresh the deadline. Historical output-only rounds remain reviewable on
the board but are excluded from the stateful incumbent envelope. A truncated search
cannot initialize or rebaseline a campaign.

### What runs during a round

Everything below is **automatic** — `run` spawns it; you don't invoke any of it by hand:

- **`adapter.py bottleneck` + `adapter.py gaps`** — shelled to build the QUERY (and again to reprint it after the gate); `gaps` reads `flexgraph_generated.json`.
- **the oracle** — `cat .oracle_prompt.txt | claude -p … --output-format stream-json` (the loop *drives* the LLM; streamed to `.oracle.log` + the board heartbeat).
- **FROZEN-fingerprint guard** — hashes `akt/benchmark/**`, board source, the loop, the native-test tree, and reference ASTs that share editable kernel files; any oracle change to those contracts voids the round.
- **Manifest-scope guard** — `files_touched` must exactly equal the oracle's non-bookkeeping worktree edits; campaign state/history are also protected during the oracle window.
- **Frozen case-contract validator** — every runner must retain the exact workload IDs, canonical input partials, reference functions, tolerances, output projection, execution-regime declaration, and native-test wiring declared by `benchmark/suites.py`.
- **`akt/benchmark/gates/eval.py --suite full`** — THE GATE: exhaustively autotunes each kernel's **deployable** space, requires *every* advertised configuration to match the frozen reference, and times it. Recurrent KDA/GLA cases start from a nonzero state and compare both token output and final state, matching production prefill semantics. Configured sglang-jax native pytest checks must positively pass; errors/timeouts fail closed. The runnable case set is fixed at `init`/`rebaseline`, so a candidate cannot improve the geomean by turning a slow case into `tpu-deferred`.
- **Programmer-exposure gate** — requires a new runner dimension to name a registered `KernelControlPolicy` control and verifies that a production layer/backend forwards it to the declared kernel argument. A larger candidate list, new default, or runner-only switch does not qualify.
- **`flexgraph_extract.py` after KEEP** — re-mines the live serving stack so the next round sees the updated programmer-exposure frontier.
- **`akt/board/build.py`** — rebuilds the dashboard on every keep/reject.
- **git** — `git commit` on KEEP, `git checkout <incumbent> -- <file>` on REJECT.

> `build.py` only reads the generated graph. `loop.py` regenerates it after a successful
> KEEP; run the extractor manually before `init` when the stack changed outside the loop.

Run these **alongside**, to watch (optional, read-only): `loop.py status` (reprint the
QUERY), `tail -f akt/optimization_history/.oracle.log` (live oracle work),
`cd akt/board && python -m http.server 8777` (dashboard),
`tail -f akt/optimization_history/evolve_history.jsonl` (per-round decisions).

### Does the loop consider the changes it made in past runs? — **Yes, it is a ratchet**

The loop is **cumulative, not fresh-each-run**. Each round is measured against the *best
result kept so far*, and past decisions persist:

- **KEEP advances the incumbent.** On a kept round the loop lowers `incumbent_geomean`,
  `git commit`s the capability's edits, and sets `incumbent_commit = git HEAD`. Since the
  next round reloads that state and never reverts kept edits, **every later round builds on
  top of all kept capabilities** and must beat the *improved* incumbent.
- **REJECT reverts cleanly.** A rejected round `git checkout`s every touched file back to
  the incumbent commit (or deletes newly-added files); the incumbent is left untouched.
- **The oracle is told the history.** The QUERY hands it the **KEPT** capabilities (with
  their `+Δ%`) and the **REJECTED** ones marked *"do not re-attempt unmodified"* — persisted
  as `kept`/`rejected` status in `capabilities/*.json`.
- **The gap frontier shrinks only after production exposure.** The extractor separately
  records `elevated_in_akt` (the benchmark runner can search it) and
  `programmer_exposed` (the production API can select it). A runner-only experiment stays
  open; a matching registered control removes the gap after the post-KEEP refresh.
- **Objective scopes never mix.** The incumbent and chart envelope can advance only
  from measurements tagged `stateful-serving-deployable-v1`. Legacy output-only rounds
  are retained as audit history but cannot compete with or seed the current objective.

State lives in `akt/optimization_history/evolve_state.json` (the ratchet),
`evolve_history.jsonl` (append-only per-round ledger), and `core/evolve/capabilities/*.json`
(the kept/rejected memory).

---

## Capability elevation versus parameter tuning

The old loop drifted toward model-specific tuning because its contract stopped at the
benchmark runner: adding a `Knob`, searching two fixed GLA shapes or two fixed KDA shapes,
and improving their geomean was sufficient for KEEP. It did not require a server option,
a production backend consumer, or evidence that another workload could select the new
path. Large full-sequence chunks and output-only state elision were therefore rewarded by
the canonical shapes even when stateful serving could not use them.

That setup optimized a small empirical table, not the software abstraction. A proposal
could win by choosing a better value for the benchmark's exact sequence/head dimensions
without making any capability selectable by a model server. Worse, the old recurrent
cases used no incoming state and did not observe the outgoing state, so eliminating state
work looked valid despite changing the contract needed for subsequent decode.

The current contract distinguishes three levels:

1. **Kernel argument** — low-level machinery exists.
2. **Runner-only experiment** — AKT can measure it, but application code cannot select it.
3. **Programmer control** — a validated, shape-aware policy reaches the argument through
   the production serving backend. Only this level is capability elevation.

The API is [`KernelControlPolicy`](../python/sgl_jax/srt/configs/kernel_control.py).
It matches static per-device shape/state context, not model or checkpoint names. Pass a
JSON object or JSON file at server startup:

```bash
PYTHONPATH=python .venv/bin/python -m sgl_jax.launch_server \
  --model-path <model> \
  --kernel-control-config /path/to/kernel-controls.json
```

```json
{
  "kda": {
    "default": {"intra_block_size": 16, "compute_block_chunks": 2},
    "rules": [
      {
        "when": {"head_dim": 64, "max_sequence_length": 512},
        "set": {"state_dim_alignment": 64, "state_block_chunks": 2}
      }
    ]
  },
  "gla": {
    "rules": [
      {
        "when": {"min_sequence_length": 1024},
        "set": {"chunk_size": 256, "output_value_tiles": 4}
      }
    ]
  }
}
```

Unknown controls and invalid combinations fail at server startup; shape-dependent
constraints fail before the kernel launch. An empty policy preserves kernel defaults.
The state-elision experiments remain runner-only because SGLang serving observes the
final recurrent state for subsequent decode, while those optimizations are exact only
when that state is unobserved.

The policy is deliberately **shape-aware, not model-name-aware**. Its context consists
of kernel family, tensor dimensions, sequence bounds, and whether recurrent state is
present/observed. This lets one exposed capability apply across checkpoints and workload
mixtures while still permitting shape-specific selection. Searching values already
present in an exposed control remains autotuning; adding a previously unavailable,
end-to-end selectable axis is the capability elevation AKT accepts.

AKT still has to time concrete workload shapes: performance is hardware- and
shape-dependent. The distinction is that a shape may select a value for a generic
control, but it cannot be the only place that control exists. The current production
policy exposes **nine controls across KDA and GLA**. Pre-AKT axes in other runners remain
part of the inherited incumbent search contract; they are not retroactively claimed as
stable server APIs. A future AKT round touching those families must add the same typed
policy and production-consumer path before it can KEEP.

The present two-shape KDA and two-shape GLA cases establish correctness and performance
only for those sampled regimes. Broader generalization still requires representative
model-derived shapes and an attached-TPU run. What is now generic is the **elevation
contract and programmer abstraction**, not a claim that one tuned value is optimal for
every model.

---

## Two things people always ask

### (1) Where is the flexibility gap documented, and how does the loop use it?

**The gaps are stored, as auto-generated data, in
[`akt/core/analysis/flexgraph_generated.json`](core/analysis/flexgraph_generated.json)**
(top-level `gaps` / `gap_categories` / `serving_stack` keys). Each gap is one
self-describing record — the axis, its category, the source `evidence` (`file:line`),
whether it is in the AKT suite, whether a runner searches it, and whether the production
programmer API exposes it:

```json
{ "id": "fused_mlp:buffer_count:pipeline-depth",
  "axis": "buffer_count", "category": "pipeline-depth",
  "detail": "Buffered(buffer_count=3) — hardcoded literal",
  "evidence": "python/sgl_jax/srt/kernels/fused_mlp.py:97",
  "reachable_prim": "dma_start", "graph_node": "pallas:dma_start",
  "in_akt_suite": true, "elevated_in_akt": true,
  "programmer_exposed": false, "exposure_level": "runner" }
```

This file is **not hand-authored** — it is regenerated from the source tree by
[`akt/core/analysis/flexgraph_extract.py`](core/analysis/flexgraph_extract.py), which
AST-scans every Pallas kernel to find tiling/pipeline/schedule axes the lowering *can*
execute but no config *names*. (A `gap` = reachable-but-unnamed.)

**The contract for how the loop consumes gaps is documented in
[`akt/benchmark/adapter.py`](benchmark/adapter.py)** (top-of-file docstring + the
`gaps` subcommand). The data flows to the loop like this:

```
flexgraph_generated.json                         (storage: the gap records)
   │  adapter.py  gaps  →  AKT_GAPS {title, lines[], note}   (filter to actionable, rank, flatten)
   ▼
loop.py::frontier_block()  →  the "== FRONTIER" block of the Capability QUERY
   ▼
run mode:  QUERY written to akt/optimization_history/.oracle_prompt.txt → handed to the oracle
status mode: QUERY printed to the terminal
```

The gap record is a **pointer** — it tells the oracle *what* axis to elevate and *where*
in the source (`evidence`). The loop never auto-applies it; it hands over the ranked
frontier and then gates the result. A new runner `Knob` flips `elevated_in_akt`; only a
corresponding production `KernelControlPolicy` path flips `programmer_exposed` and removes
the gap from the OPEN frontier. This prevents benchmark-only tuning from masquerading as
stack capability elevation.

> The frozen fallback: if the extractor has not run, `adapter.py` falls back to a
> hand-curated `FRONTIER` list. The auto-miner is validated to rediscover that list
> **7/7**, so it is a fresh superset, not a replacement of unknown fidelity.

### (2) Does the LLM oracle choose which gap to elevate?

**Yes.** The loop *generates and ranks* the frontier but does **not** decide. In
autonomous `run` mode the ranked list is embedded in the QUERY prompt and the **LLM
oracle picks ONE gap** to implement — the one it judges best relieves the current
bottleneck. The loop then checks end-to-end exposure, re-searches the enlarged design
space, validates correctness, and applies the performance threshold. In manual mode a
human picks instead.

The extractor's ranking (`_rank`: in-suite first, un-elevated first, then category
salience) is only a *suggested ordering* — the oracle is free to choose any gap.

---

## Map of the subsystem

| Path | Role |
|------|------|
| `akt/core/evolve/loop.py` | THE loop — init/run/submit/gate/keep-restore; builds the QUERY, drives the oracle |
| `akt/core/evolve/exposure.py` | Frozen gate proving runner knob → stable API → production consumer → kernel keyword |
| `akt/core/analysis/flexgraph_extract.py` | Investigates the serving stack; **auto-derives the gap frontier + the lowering graph** |
| `akt/core/analysis/flexgraph_generated.json` | **Where the gaps are stored** (generated) |
| `akt/benchmark/adapter.py` | **The gaps/bottleneck contract** the loop shells out to (FROZEN) |
| `akt/benchmark/` | The frozen measurement harness (suites, gates/eval.py, runners/base.py) |
| `akt/core/runners/*.py` | Editable per-kernel design spaces — where an elevated gap becomes a `Knob` |
| `python/sgl_jax/srt/configs/kernel_control.py` | Stable, validated programmer-facing control policy and registry |
| `akt/core/evolve/capabilities/*.json` | One manifest per elevated capability ([schema](core/evolve/capabilities/README.md)) |
| `akt/board/` | The dashboard (trajectory, flexibility graph, serving-stack gap frontier) |

**FROZEN** (defines the measurement and frontier — the loop/oracle may not edit):
everything under `akt/benchmark/**`, `python/sgl_jax/test/**`, the source-derived
`flexgraph_extract.py` miner, the loop, and its exposure gate. Pure-JAX
reference symbols that still share an editable production-kernel file are protected by
AST fingerprints. Production kernels and runners remain editable, but the frozen suite
validates that runners cannot replace their workload/reference/tolerance/test contract.
