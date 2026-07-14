# AKT — the capability-elevation loop (sglang-jax kernels)

AKT is an autonomous loop that improves the sglang-jax TPU-Pallas kernels by
**elevating one low-level flexibility per round** into the tunable design space, then
keeping the change only if it measurably beats the incumbent. A Claude/LLM session is
the *oracle* that implements each round; the loop owns the trajectory, the measurement,
and the keep/restore rule.

```
init      measure the incumbent (best config in the existing design space)
run       for each round: hand the QUERY to the oracle -> it elevates ONE gap ->
          re-search the enlarged space -> gate (>2% on the suite) -> KEEP or REVERT
status    print the current Capability QUERY (bottleneck + ranked gap frontier)
submit    gate a manually-implemented capability
```

```bash
python akt/core/evolve/loop.py init  --hours 6 --target 0.02
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
| 3 | `PALLAS_INTERPRET=1 PYTHONPATH="python:." .venv/bin/python akt/core/analysis/flexgraph_extract.py` | optional | refresh the auto-derived gap frontier (`flexgraph_generated.json`). **Off-loop** — the loop never runs this. If absent, the adapter falls back to the hand `FRONTIER`, so the QUERY is non-empty either way |
| 4 | `.venv/bin/python akt/core/evolve/loop.py init --hours 6 --target 0.02 --runs 3` | **yes** | measures the incumbent (best config in the *existing* design space) and, crucially, **writes the two files `run` depends on**: `evolve_state.json` (loaded every round) and `.evolve_eval.json` (the **only** source `adapter bottleneck` reads — no eval → empty BOTTLENECK). Also resets the campaign deadline, so re-run `init` if a prior deadline has expired |
| 5 | `git status --porcelain` | **yes** | `run` refuses a **dirty tree** (it git-reverts void/rejected rounds and would clobber uncommitted work) — commit or stash first |
| 6 | `.venv/bin/python akt/core/evolve/loop.py run --rounds N --oracle claude` | **yes** | the target command (autonomous mode needs the `claude` CLI on `PATH`) |

### What runs during a round

Everything below is **automatic** — `run` spawns it; you don't invoke any of it by hand:

- **`adapter.py bottleneck` + `adapter.py gaps`** — shelled to build the QUERY (and again to reprint it after the gate); `gaps` reads `flexgraph_generated.json`.
- **the oracle** — `cat .oracle_prompt.txt | claude -p … --output-format stream-json` (the loop *drives* the LLM; streamed to `.oracle.log` + the board heartbeat).
- **FROZEN-fingerprint guard** — md5 of `akt/benchmark/**` + the loop file, snapshotted before/after the oracle, so tampering with the measurement voids the round.
- **`akt/benchmark/gates/eval.py --suite full`** — THE GATE: autotune each kernel, check correctness, time it; it also subprocess-runs sglang-jax's **own pytest** per case.
- **`akt/board/build.py`** — rebuilds the dashboard on every keep/reject.
- **git** — `git commit` on KEEP, `git checkout <incumbent> -- <file>` on REJECT.

> Note: **`flexgraph_extract.py` is NOT part of the loop** — neither `loop.py` nor
> `build.py` ever calls it. Regenerate the gap frontier yourself (step 3 above) when the
> kernels change; the loop only *reads* the JSON it left behind.

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
- **The gap frontier shrinks as gaps get elevated.** `flexgraph_extract.py` re-derives
  `elevated_in_akt` from the *live* runner `Knob`s, so once a gap has been elevated it drops
  off the actionable OPEN frontier the oracle sees on the next regeneration.

State lives in `akt/optimization_history/evolve_state.json` (the ratchet),
`evolve_history.jsonl` (append-only per-round ledger), and `core/evolve/capabilities/*.json`
(the kept/rejected memory).

---

## Two things people always ask

### (1) Where is the flexibility gap documented, and how does the loop use it?

**The gaps are stored, as auto-generated data, in
[`akt/core/analysis/flexgraph_generated.json`](core/analysis/flexgraph_generated.json)**
(top-level `gaps` / `gap_categories` / `serving_stack` keys). Each gap is one
self-describing record — the axis, its category, the source `evidence` (`file:line`),
whether it is in the akt suite, and whether a runner has already elevated it:

```json
{ "id": "fused_mlp:buffer_count:pipeline-depth",
  "axis": "buffer_count", "category": "pipeline-depth",
  "detail": "Buffered(buffer_count=3) — hardcoded literal",
  "evidence": "python/sgl_jax/srt/kernels/fused_mlp.py:97",
  "reachable_prim": "dma_start", "graph_node": "pallas:dma_start",
  "in_akt_suite": true, "elevated_in_akt": false }
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
frontier and then only measures the result. Once a gap is elevated (a `Knob` is added to
the kernel's runner in `akt/core/runners/`), the next extractor run flips it
`elevated_in_akt: true` and it drops off the OPEN frontier — that is the feedback edge
that closes the loop.

> The frozen fallback: if the extractor has not run, `adapter.py` falls back to a
> hand-curated `FRONTIER` list. The auto-miner is validated to rediscover that list
> **7/7**, so it is a fresh superset, not a replacement of unknown fidelity.

### (2) Does the LLM oracle choose which gap to elevate?

**Yes.** The loop *generates and ranks* the frontier but does **not** decide. In
autonomous `run` mode the ranked list is embedded in the QUERY prompt and the **LLM
oracle picks ONE gap** to implement — the one it judges best relieves the current
bottleneck. The loop's role after that is purely mechanical: re-search the enlarged
design space and gate the result (keep if >2% faster on the suite, else revert every
touched file). In manual mode a human picks instead.

The extractor's ranking (`_rank`: in-suite first, un-elevated first, then category
salience) is only a *suggested ordering* — the oracle is free to choose any gap.

---

## Map of the subsystem

| Path | Role |
|------|------|
| `akt/core/evolve/loop.py` | THE loop — init/run/submit/gate/keep-restore; builds the QUERY, drives the oracle |
| `akt/core/analysis/flexgraph_extract.py` | Investigates the serving stack; **auto-derives the gap frontier + the lowering graph** |
| `akt/core/analysis/flexgraph_generated.json` | **Where the gaps are stored** (generated) |
| `akt/benchmark/adapter.py` | **The gaps/bottleneck contract** the loop shells out to (FROZEN) |
| `akt/benchmark/` | The frozen measurement harness (suites, gates/eval.py, runners/base.py) |
| `akt/core/runners/*.py` | Editable per-kernel design spaces — where an elevated gap becomes a `Knob` |
| `akt/core/evolve/capabilities/*.json` | One manifest per elevated capability ([schema](core/evolve/capabilities/README.md)) |
| `akt/board/` | The dashboard (trajectory, flexibility graph, serving-stack gap frontier) |

**FROZEN** (defines the measurement — the loop/oracle may not edit): everything under
`akt/benchmark/**` and the loop file itself. Everything else, including the kernels and
the runners, is editable — that is the point ("implement the whole stack").
