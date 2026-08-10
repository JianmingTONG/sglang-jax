# AKT on TPU — onboarding brief

For the Claude session taking this repo onto a real TPU host. Read this, then
`akt/README.md` and `akt/RULES.md`. Written 2026-08-10 at commit `26847c1`+.

## What this is

`akt-loop` branch of sglang-jax (private remote `JianmingTONG/sglang-jax-akt`).
The AKT loop is an autonomous capability-elevation campaign: an LLM oracle
(`--oracle opus`) either ELEVATES an existing-but-unexposed kernel choice
(red-link action) or INVENTS a new algorithm/API (frontier/standalone slot,
declared through the flexgraph interface), and a frozen gate keeps it only on
measured, guarded improvement (>2% paired AND absolute, all guards green).
The behavioral contract is codified in **`akt/RULES.md` (8 rules) with an
automated checker** — keep it green through any modification:

```bash
PYTHONPATH=python:. .venv/bin/python akt/core/verify/rules_check.py
```

## Where the campaign stands

- Everything measured so far ran in **GPU-testbench mode** (`--testbench`,
  Pallas-interpret/GPU proxies; only the linear-attention trace runnable).
  Current testbench incumbent: **1.96 ms** geomean after R15, campaign
  trajectory 25.6 → 1.96 ms across 15 rounds (6 keeps).
- Keeps under the current guarded objective: R8 `kda_resident_pipeline_api`
  (−75.6%), R9 `gla_batched_value_tile_output_api` (−9.3%), R12
  `gla_decay_rescaled_chunk_api` (−3.8%), R15 `gla_fused_operand_walk_api`
  (−3.04%). Rejects R10/R11/R13/R14 were killed by the falsified-additivity
  panel and the API-novelty floor — the guards, not the 2% bar, are the binding
  constraint.
- **None of this is TPU evidence.** The v2 objective
  (`model-serving-empirical-dp-v2`) arms fully only on TPU: compiled execution,
  all 15 callsites, bit-exact kv_cache invariant, incumbent output-hash pins,
  and the conditional live Kimi-Linear-48B gate are wired but dormant on GPU.

## Latest changes you inherit (last five commits)

1. **`39098c1` formal workload layer** — `benchmark/workload_spec.py`
   (`WorkloadSpec → BlockSpec → OpSpec`, lowering with frozen-case reuse,
   new-case materialization, explicit coverage gaps) + `benchmark/model_specs.py`
   (Qwen3-8B, Qwen3-30B-A3B, Llama-3.1-8B, DeepSeek-V3, Kimi-Linear,
   all-kernels, testbench siblings). `verify_legacy_roundtrip()` proves the
   three frozen traces reproduce call-for-call; the pinned
   `model_contract_fingerprint` is untouched by default.
2. **`8d42387` loop-adoptable workload expansion** —
   `AKT_EXTENDED_WORKLOADS=<spec-ids>` extends `MODEL_WORKLOADS` at the source
   (every consumer picks it up). Default = byte-identical; adoption changes the
   contract fingerprint and REQUIRES rebaseline; non-frozen calls are excluded
   fail-closed with a report. `all-kernels` is fully adoptable today.
3. **`55d73c5` campaign-diagram viewer skill**
   (`.claude/skills/campaign_diagram_bottleneck_viewer`) — renders every
   workload as a canonical campaign diagram (x=time, compute-util step line,
   fixed bw pipe, one global scale); pages under `akt/board/campaign_diagrams/`.
4. **`26847c1` rule registry** — `akt/RULES.md` + `akt/core/verify/rules_check.py`
   + battery test. R1 unified workload definition; R2 bottleneck must come from
   the campaign-diagram pipeline (measure→trace→verdict→LIMITER); R3 one global
   utilization scale; R4 ELEVATE-before-INVENT serialization; R5 invention only
   via the flexgraph interface; R6 default==incumbent bit-exact; R7 frozen
   fail-closed; R8 guarded KEEP bar.
5. Earlier this window: **`8a9c082` proposal-form serialization** (PHASE line;
   novel proposals inadmissible while measurable red-links are un-attempted;
   below-bar red-links = cheap declared rejects).

## Your first session on the TPU, in order

```bash
# 0. env: python venv with jax[tpu]; repo root; PYTHONPATH=python:.
# 1. verify the tree
PYTHONPATH=python:. .venv/bin/python -m pytest akt/ -q          # 210 tests
PYTHONPATH=python:. .venv/bin/python akt/core/verify/rules_check.py
PYTHONPATH=python:. .venv/bin/python akt/benchmark/gates/dp_verify.py

# 2. rebaseline ON TPU (no --testbench!). This arms every dormant guard and
#    re-pins fingerprints/hashes/plans against compiled TPU execution.
#    Optionally adopt the expanded objective in the same step:
AKT_EXTENDED_WORKLOADS=all-kernels \
  .venv/bin/python akt/core/evolve/loop.py rebaseline --hours 6 --runs 3

# 3. run. The PHASE line will read ELEVATE: all five red-link actions
#    (fused_mlp/megablox buffer_count, fused_moe/v2 toggles) become measurable
#    on TPU and are MANDATORY before any novel proposal (RULES.md R4).
.venv/bin/python akt/core/evolve/loop.py run --rounds 3 --oracle opus
```

Expectations that differ from the GPU testbench: the testbench geomeans do NOT
carry over (objective scope guards prevent mixing them); the 9 tpu-deferred
kernels (RPA, MoE, fused MLP, kv_cache, gmm_v2) become measurable, so the
bottleneck picture will change — trust the regenerated campaign diagram
(`akt/core/analysis/campaign.py --out akt/board/campaign.json`), per R2, not the
GPU-era one. A `gmm_v2.*` capability on TP4 with ≥4 devices triggers the live
Kimi-Linear checkpoint gate (GSM8K ≥ 0.89, throughput ratio ≥ 0.98).

## Operating conventions (learned the hard way)

- **Post-KEEP the tree is left dirty** with loop outputs (board coverage, eval
  archives). Pattern: commit them as `akt: campaign bookkeeping through RN`,
  then repoint `evolve_state.json:incumbent_commit` to the new HEAD (the pin
  must equal HEAD or `run`/`submit` refuse). Precedents: `2b7da35`, `25fcb13`.
- **Deadline**: `deadline_ts` in `evolve_state.json` stops `run` silently at
  0 gated rounds when past — check it before launching; extending it is a
  bookkeeping edit.
- **FROZEN discipline** (RULES R7): `akt/benchmark/**`, the extractor,
  validators, DP, loop, board source, `python/sgl_jax/test/**` are frozen
  during oracle work. Maintainer edits between rounds are fine but re-run the
  battery + rules_check, commit, and repoint the pin.
- **Elision flags stay out of the serving registry** (R6): serving observes the
  final recurrent state; `single_chunk_state_elision`/`zero_state_output_elision`
  are runner-internal only.
- The Kimi checkpoint is named by HF repo, not pinned to a revision — pin it
  before relying on cross-run comparability of the live gate.
- Logs: `akt/optimization_history/.akt_v2_run.log` (orchestration),
  `.oracle.log` (full oracle transcript), `.model_eval.log` (gate),
  `.evolve_eval.json` (machine summary), `evals/` (per-round archives).
  Board: `cd akt/board && python -m http.server 8778`.
