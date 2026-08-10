# AKT capability-elevation loop

AKT exposes TPU kernel choices through the serving stack. The flexibility graph
mines TWO lists into ONE canonical interface under ONE fingerprint: red-link
ACTIONS (`access: "existing"`) — source-proven, already-existing low-level
selection points not exposed to programmers — and FRONTIER SLOTS
(`access: "frontier"`) — mechanically derived flexibilities not yet considered,
one per Pallas-launch-owning function in an in-suite family. The graph tells us
what exists but is not exposed; the frontier tells us what has not been
considered; both arrive in the same action contract, so the oracle does not treat
them as different kinds — it picks one `gap_id` from one catalog.

A round either ELEVATES a red-link action — adding the control-plane plumbing
needed to name it (`KernelControlPolicy`, a production forwarding path, and
planner metadata) — or fills a FRONTIER SLOT with a NOVEL algorithm/variant. The
two forms are SERIALIZED, elevate-first: while any red-link action measurable
under the current campaign remains un-attempted (no keep/reject record under the
active objective), a novel proposal is inadmissible — the QUERY's PHASE line says
ELEVATE and the validators void a `proposed_action` manifest. A red-link the
oracle judges unable to clear the KEEP bar is declared, not skipped: an elevate
manifest with `estimate.expected_relief_pct` at or below the bar is recorded as a
cheap no-eval reject and counts as attempted. Only when every measurable red-link
has been attempted (or none is measurable under the campaign scope) does the
PHASE flip to INVENT and unlock frontier/standalone proposals. For a NOVEL round: the
manifest foreign-keys the slot's `gap_id`, its `proposed_action` COMPLETES the
slot record, the implementation sits strictly behind the slot's prescribed axis
whose default preserves the incumbent algorithm bit-exactly, and the frozen
extractor must rediscover the finding in the regenerated graph with exactly the
declared semantics.

The flexibility graph is the oracle's authorization boundary, not a performance
oracle. The LLM still ranks actions and writes the implementation; the graph limits
elevation to fingerprinted source findings, confines invention to the graph's own
mined frontier slots, and the gate decides with measurements.

## Quick start

Run from the repository root with the project `.venv`:

```bash
# Start in the repository root.

# Refresh source findings and the executable action contract.
PYTHONPATH=python:. .venv/bin/python akt/core/analysis/flexgraph_extract.py

# Fast proof that the DP implementation equals a bounded Cartesian oracle.
PYTHONPATH=python:. .venv/bin/python akt/benchmark/gates/dp_verify.py

# The implementation tree must be clean before establishing a rollback point.
git status --porcelain

# Fresh campaign. The synthetic gate requires compiled TPU execution.
.venv/bin/python akt/core/evolve/loop.py init --hours 6 --target 0.02 --runs 3

# Migrate an older campaign to the current objective.
.venv/bin/python akt/core/evolve/loop.py rebaseline --hours 6 --runs 3

# Autonomous or manual operation.
.venv/bin/python akt/core/evolve/loop.py run --rounds 3 --oracle codex
.venv/bin/python akt/core/evolve/loop.py status
.venv/bin/python akt/core/evolve/loop.py submit --capability <name> --runs 3
```

The active objective scope is `model-serving-empirical-dp-v2`. `init` and
`rebaseline` fail closed unless `jax.default_backend() == "tpu"`,
`PALLAS_INTERPRET` is disabled, all 15 synthetic callsites execute, and every
synthetic correctness and empirical-DP guard passes.

## Frozen objective

[`benchmark/model_workloads.py`](benchmark/model_workloads.py) defines three
deterministic serving traces. Each call has a fixed seed, frozen tensor generator,
frozen reference, and explicit inference phase.

| Workload | Purpose | Calls |
|---|---|---|
| `tiny-linear-serving` | linear-attention prefill envelope | GMM, two GLA, two KDA |
| `tiny-dense-serving` | dense prefill and decode | GMM v2, fused MLP, RPA, KV update |
| `tiny-moe-serving` | MoE prefill and decode | GMM, MoE v1/v2, fused MLP, RPA, KV update |

Together they cover every frozen `KernelCase`. Campaign state pins the model
contract, generated tensors/references, target hardware, and action graph. Each
evaluation also records its measured-cost graph and empirical-panel hashes, but those
timing-derived hashes are evidence for that evaluation rather than cross-round pins.

These traces use production kernels, but they are not downloaded checkpoints. The
separate live-model gate conditionally runs the actual
`moonshotai/Kimi-Linear-48B-A3B-Instruct` checkpoint.

**Formal workload layer** ([`benchmark/workload_spec.py`](benchmark/workload_spec.py) +
[`benchmark/model_specs.py`](benchmark/model_specs.py)): workloads are also expressible
as architecture-shaped specs — `WorkloadSpec` (arch params) → `BlockSpec` (repeated
transformer block) → `OpSpec` (semantic operator: hidden/heads/seq, not case ids) —
with a lowering onto the kernel inventory that reuses frozen cases where shapes match,
materializes new `KernelCase`s from the same runner constructors otherwise, and reports
ops the inventory cannot express (`attn_prefill_full`, fused-MoE/rpa OpSpec wiring) as
explicit coverage gaps instead of dropping them. The library ships Qwen3-8B,
Qwen3-30B-A3B (MoE), and Kimi-Linear (sgl_jax config defaults) plus testbench-scale
siblings; `legacy_specs()` re-expresses the three frozen traces and
`verify_legacy_roundtrip()` proves the lowering reproduces them call-for-call, so the
pinned `contract_fingerprint` is untouched. New workloads join the measured objective
only through an explicit `rebaseline`. CLI: `python akt/benchmark/model_specs.py
--list / --describe <id> / --lower <id> / --verify-legacy`.

**Expanding the measured objective** (`AKT_EXTENDED_WORKLOADS`): the loop's workload
set is no longer fixed at the three tiny traces. Setting
`AKT_EXTENDED_WORKLOADS=all-kernels,qwen3-tiny-testbench` (any spec-library ids)
extends `MODEL_WORKLOADS` AT THE SOURCE — every consumer (model_eval, adapter,
capability_contract, campaign, loop) picks the extras up automatically. The default
(env unset) is byte-identical to the frozen contract; adoption changes
`contract_fingerprint` by design and therefore requires `loop.py rebaseline`.
Fail-closed v1 restriction: an extended workload contributes only calls the frozen
suite can measure — calls lowering to new (materialized) cases are excluded with an
explicit report until the suite contract grows a case-extension path. `all-kernels`
(the complete inventory as one trace) is fully measurable today;
`python akt/benchmark/model_specs.py --registry-coverage` maps every sglang-jax
model implementation to its spec-family coverage (or names it UNCOVERED).

## One round

1. `adapter.py bottleneck` reports the latest model/callsite timing and selected
   plans. `adapter.py gaps` reports the complete executable graph action catalog.
2. The oracle receives both the human-readable report and a structured action
   contract containing version, target stack, red-link actions, frontier slots
   (`frontier_actions`), and one SHA-256 fingerprint over both. The checked-in v2
   graph currently has five source-proven actions.
3. A proposal selects one `gap_id` under the complete graph fingerprint — a
   red-link action (ELEVATE) or a frontier slot (NOVEL); the two sources share one
   contract, so the oracle never treats them differently — but the FORMS are
   SERIALIZED: the PHASE line (from `elevate_remaining`) admits novel proposals
   only after every measurable red-link action has been attempted. Source evidence, axis,
   family, incumbent value, finite candidate domain, semantic sink, and affected
   callsites are derived from that foreign key rather than copied into the
   manifest; a frontier round's `proposed_action` completes the slot record,
   authoring only the source sink, evidence line, and detail.
4. The implementation exposes that existing axis through a production consumer,
   `KernelControlPolicy`, a runner `Knob`, and planner metadata. A newly added backend
   argument is allowed only as forwarding for the exact hardcoded source axis, with
   the incumbent behavior preserved as its default.
5. The extractor regenerates a candidate graph. A selected red-link action must be
   closed or marked programmer-exposed; a selected frontier slot must no longer be
   emitted while its finding is mined under the same `gap_id`; unrelated red-link
   actions and unrelated frontier slots may not disappear, appear, or change
   semantics.
6. Every valid *deployable* local configuration is correctness-checked and timed on
   TPU. Runner-only benchmark knobs are held at their defaults; adding a stable
   `programmer_control` is what expands their existing values into the serving search.
7. [`core/search/model_dp.py`](core/search/model_dp.py) computes the exact layered
   optimum of each complete finite additive model graph. A bounded Cartesian oracle
   independently checks the implementation.
8. The `synthetic_trace_additivity` gate builds a binary whole-plan panel: every stage
   contains the full-DP choice and, when available, its cheapest distinct alternative.
   It executes every valid combination, up to the fail-closed 64-plan limit, and
   correctness-checks every complete trace.
9. Each challenger is timed against the DP plan in balanced paired order. A
   deterministic bootstrap produces a 95% upper bound on selection regret and on
   multi-factor interaction residual. By default, regret must be at most 1% and the
   interaction residual at most 2%; incomplete or inconclusive panels fail.
10. The selected plan must use the new control at a non-default value. A frozen probe
    observes the declared backend argument at the exact model/callsite scope.
    Candidate and incumbent synthetic traces then run in paired alternating order.
11. After the synthetic process releases the TPU, the conditional live Kimi-Linear
    gate runs only for a changed GMM-v2 control on Kimi's production
    EPMoE→GMM-v2 route. Other actions record `not_applicable`; incompatible hardware
    records a topology skip. Neither case imports the model stack or downloads a
    checkpoint.
12. KEEP requires every guard plus both paired and absolute three-trace geomean
    improvement strictly above the campaign threshold (never below 2%). The validated
    candidate graph is installed atomically only for KEEP; rejection restores the
    declared implementation files.

## What the search proves

Local enumeration covers every valid configuration exposed by the finite deployment
space. The DP is the exact optimum of the fingerprinted *additive measured-cost*
graph. The bounded `DP == brute` check verifies the solver, while the empirical binary
panel tests whether the additive choice survives selected whole-trace interactions.

This is not brute force over arbitrary post-modification programs, the full
non-additive Cartesian model space, or external systems, and it is not a global SOTA
claim. The paired synthetic trace and conditional Kimi measurements are the hardware
evidence within their stated scopes.

## Flexibility-graph guardrail

Run:

```bash
PYTHONPATH=python:. .venv/bin/python akt/core/analysis/flexgraph_extract.py
PYTHONPATH=python:. .venv/bin/python akt/benchmark/adapter.py gaps
```

The extractor scans the Pallas kernel sources and installed Pallas-to-Mosaic
lowering registry. It mines two lists into one context: red-link `actions`
(`access: "existing"`, already-existing selection points) and top-level
`frontier_actions` (`access: "frontier"`, not-yet-considered flexibilities) — one
slot per Pallas-launch-owning function in an in-suite family, gap_id
`<family>:enable_<fn>_variant:schedule-toggle`, prescribed axis name, category
`schedule-toggle`, incumbent `false`, domain `[false, true]`, evidence at the
launch site, callsites from the frozen workloads. A slot is where a NEW
algorithm/variant may be introduced. The oracle receives a canonical context
shaped like:

```json
{
  "version": 2,
  "graph_target": {
    "backend": "tpu",
    "scope": "sglang-jax TPU-Pallas compiler and serving stack"
  },
  "actions": [
    {
      "gap_id": "fused_moe/v2:interleave_bt:schedule-toggle",
      "access": "existing",
      "family": "fused_moe/v2",
      "kernel_ids": ["moe_v2"],
      "source_axis": "interleave_bt",
      "source_function": "_fused_ep_moe_kernel",
      "source_sink": {
        "assignments": ["use_gather_bank"],
        "expression_asts": {"use_gather_bank": "<normalized source expression>"}
      },
      "incumbent_value": true,
      "candidate_values": [true, false],
      "category": "schedule-toggle",
      "source_evidence": {
        "path": "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py",
        "line": 342,
        "detail": "default=True"
      },
      "model_callsites": ["tiny-moe-serving/expert-v2"],
      "action_edge": {
        "source": "action:fused_moe/v2:interleave_bt:schedule-toggle",
        "target": "pallas:dma_start"
      }
    }
  ],
  "frontier_actions": [
    {
      "gap_id": "simple_gla:enable_simple_gla_fwd_variant:schedule-toggle",
      "access": "frontier",
      "family": "simple_gla",
      "kernel_ids": ["gla"],
      "source_axis": "enable_simple_gla_fwd_variant",
      "source_function": "simple_gla_fwd",
      "category": "schedule-toggle",
      "incumbent_value": false,
      "candidate_values": [false, true],
      "source_evidence": {
        "path": "python/sgl_jax/srt/kernels/simple_gla/kernel.py",
        "line": 118,
        "detail": "pallas_call launch site"
      },
      "model_callsites": ["tiny-linear-serving/gla-long"],
      "action_edge": {
        "source": "action:simple_gla:enable_simple_gla_fwd_variant:schedule-toggle",
        "target": "pallas:pallas_call"
      }
    }
  ],
  "fingerprint": "<sha256-of-the-complete-canonical-context-both-lists>"
}
```

The red edge target is extractor category context, not proof that the named source
axis data-flows to that lowering primitive; source/sink validation and measurement
provide the enforceable evidence. The
fingerprint covers the complete ordered two-source context — red-link actions and
frontier slots alike — so changing any entry's family, axis, source function/sink,
incumbent/domain, evidence, callsites, or edge changes the manifest namespace.
Campaign state pins the fingerprint; `run` refuses stale graph state.

Only findings that are source-proven to exist, covered by the frozen model gate, open,
bound to an exact live call argument or normalized assignment expression, and not already programmer-exposed receive red
action links. The checked-in graph has 22 visible findings but only five executable v2
actions. Dead or missing tuning tables and numeric-representation toggles remain
analysis-only, as do other opportunities and hidden compiler boundaries; none is
itself authorization to elevate. Invention does not flow from these findings either:
a NOVEL algorithm is authorized only by a mined frontier slot — the manifest
foreign-keys the slot's `gap_id` and its `proposed_action` completes the slot
record — never by an analysis-only finding.

Before evaluation, AKT regenerates the post-implementation graph in a temporary
location and proves that exactly the selected action closes in the source-mined
catalog, with no unrelated action removed or introduced. A NOVEL round must
additionally show (i) the selected frontier slot no longer emitted — the axis now
exists — (ii) the declared finding mined under the same `gap_id`,
programmer-exposed, field-for-field equal to the declaration, and (iii) unrelated
red-link actions and unrelated frontier slots outside the selected family
preserved. After KEEP, that already validated graph and its new fingerprint become
the next round's pinned context.

## API-novelty guardrails (anti-parameter-tweak)

The campaign audit showed every kept round was a schedule/packaging change —
efficient use of the existing abstraction. Two calibrated guardrails now filter
that class out of the KEEP path (`akt/core/analysis/api_metrics.py`, thresholds
in `akt/core/analysis/api_baseline.json`, both frozen; pinned at
init/rebaseline):

1. **Delta-residual substance (two-pass subtraction).** The candidate program N
   (the affected case with the new control at its plan-selected non-default
   value) and the catalog ⋃E (the same case's one-knob population plus every
   other suite API's programs) are lowered under identical settings to operation
   dependency graphs (jaxpr DAGs, recursing into Pallas kernel bodies; graph
   inputs appear as typed ⊥ tokens). Subtraction runs in two passes: an
   INSTANCE COVER strikes whole embedded catalog programs — anchored wherever
   an API's entry micro-op matches, inputs binding to any values, whole-program
   embedding required — so chained/interleaved/tapping compositions of existing
   APIs read as duplicates; then a PER-OP ANCESTRY MATCH (fixpoint WL — one
   Merkle hash per op's full computation history) strikes everything the
   catalog's union signature multiset contains. What remains is Δ, judged on
   its own substance: per-class mass (compute/MEMORY, roofline criterion,
   total classifier with `unclassified` audit) must clear the MEASURED noise
   floor (p95 of known tweaks' spurious residuals), the largest connected Δ
   component must reach the coherence minimum, and the zero-embed self-check
   must hold. Borderline masses are re-decided by a reversed-order exact retry.
   δ = w(Δ)/w(N) is reported as a composition reference only (R ≡ 1−δ; the v1
   R/D scores are removed). Measured separation: a chunk-size tweak leaves
   Δ = 0; the recurrent reference vs the chunked-kernel catalog leaves
   δ ≈ 0.58 with a connected 36-op Δ.

2. **Genericity = upper-layer coverage.** `G_delta(N)` = the fraction of
   relevant frozen model callsites (each upper pattern counted once) whose best
   measured configuration selects the new control at a non-default value with
   local benefit >= delta (default 2%). The cut is the pinned percentile of the
   existing programmer controls' own G, floored at 0.5 (a proxy-host
   calibration that flattens every existing control's coverage cannot make the
   guard vacuous). Lower-layer fan-out is reported separately as
   `implementation_breadth` — it is not genericity.

Both guardrails run inside the gate (`API-NOVELTY GUARD`) from the same measured
`case_search` population the gate already trusts, and apply uniformly to ELEVATE
and NOVEL rounds — an elevated existing axis whose program is reproducible from
the incumbent catalog is a parameter tweak by definition and now fails, which is
the point: KEEPs must change the abstraction, not re-parameterize it.

```bash
# Recalibrate on the current host (init/rebaseline do this automatically):
PYTHONPATH=python:. .venv/bin/python akt/core/analysis/api_metrics.py profile
# Score one candidate config by hand:
PYTHONPATH=python:. .venv/bin/python akt/core/analysis/api_metrics.py score \
  --case gla:seq512_h8 --knob compact_alignment --value true
```

## Capability versus tuning

A valid round can expose one of:

- an explicit backend argument that already existed but lacked stable programmer
  access (`existing-backend-argument`);
- an existing literal/derived low-level axis that must be lifted into an entry
  argument solely for forwarding (`existing-low-level-axis`); or
- a NEW algorithm/kernel variant filling a mined FRONTIER SLOT
  (`novel-algorithm`): the manifest foreign-keys the slot's `gap_id` and its
  `proposed_action` completes the slot record — graph-owned fields must equal the
  slot; the oracle authors only the source sink, evidence line, and detail.
  Neither the entry argument nor the axis may exist at the incumbent commit, the
  incumbent algorithm remains the exact default (output-hash pinned), and graph
  closure requires the slot to vanish and the extractor to mine the new axis
  programmer-exposed with exactly the declared semantics.

All modes may add a serving API field and forwarding code. Invention outside a
mined frontier slot — a `gap_id` that is not a current slot, a variant not mined by
the extractor, a replaced default path, or a declaration drifting from the slot —
fails closed. Changing defaults, widening an already exposed control, or adding
only a benchmark knob is tuning and cannot pass.

The stable API is
[`KernelControlPolicy`](../python/sgl_jax/srt/configs/kernel_control.py). It is shape-
and state-aware rather than checkpoint-name-aware. A server can load JSON controls:

```bash
PYTHONPATH=python .venv/bin/python -m sgl_jax.launch_server \
  --model-path <model> \
  --kernel-control-config /path/to/kernel-controls.json
```

See [`core/evolve/capabilities/README.md`](core/evolve/capabilities/README.md) for the
pending-manifest schema.

## Conditional live Kimi-Linear gate

[`benchmark/live_model_candidates.py`](benchmark/live_model_candidates.py) defines
one direct GMM-v2 view of one real checkpoint using the TP4 nightly launch profile.
A round launches it only when its new capability changes a `gmm_v2.*` policy.
Eligibility
requires a successful synthetic gate, compiled single-host TPU execution, at least
four local TPU devices with stable device IDs, and tensor parallelism 4.

An eligible run launches Kimi-Linear with the selected policy and an incumbent policy,
attests the server-reported policies, and records:

- GSM8K accuracy with threshold 0.89;
- prefill point C1: input 3072, output 1, 8 prompts;
- packed prefill point C8: input 512, output 1, 32 prompts.

All requests must complete and the selected/incumbent input-throughput ratio geomean
must be at least 0.98. Input throughput and median TTFT are retained per point. If the
topology is ineligible, the evaluator imports no model stack, downloads no checkpoint,
and records a clean skip. A skip permits the portable AKT gate to run but supplies no
real-model performance claim; any eligible attempted run must pass to KEEP.

This is a quality/performance guard, not the source of the round's claimed speedup:
the 0.98 floor permits a small Kimi throughput regression and does not establish SOTA.

The policy is derived from `tiny-dense-serving/dense-projection` and applied
route-wide. Current source inspection shows Kimi's default EPMoE path calling
`megablox_gmm_backend.gmm`, which can dispatch compiled TPU GMM-v2. The live gate
attests the selected policy and completed requests, but it does not machine-observe
that a request took GMM-v2 or the changed kernel argument. No checked-in single-host
live candidate covers fused-MoE-v2 or fused MLP today, so those actions are explicitly
outside the live gate's applicability.

Candidate and incumbent synthetic plans, and both live policy variants, currently
run through the same post-edit implementation. The loop preserves the graph-mined
incumbent control value and compares against stored incumbent measurements, but does
not re-execute the prior Git revision or prove semantic/AST equivalence of its default
path. Cross-revision causal attribution therefore remains weaker than the control and
measurement contracts.

The launch profile names the Hugging Face checkpoint repository but does not pin an
immutable model revision. A future upstream checkpoint update would therefore make a
later live run non-identical unless the profile is pinned first.

The live comparison also starts the selected server before the incumbent server in a
fixed order. Separate model startups can therefore contribute order or thermal drift;
the synthetic paired comparison is better balanced than this checkpoint guard.

## Campaign memory

- KEEP stores the selected plans and graph fingerprint as the next incumbent; history
  retains a compact live-gate status when applicable.
- REJECT restores every declared implementation file from `incumbent_commit`.
- The next query includes kept capabilities, rejected estimates, bottlenecks, runtime
  evidence, and the latest structured graph context.
- Objective scopes never mix. Historical `stateful-serving-deployable-v1` and
  `model-serving-dp-real-hw-v1` results cannot seed
  `model-serving-empirical-dp-v2`.

State is stored in `optimization_history/evolve_state.json`; decisions are appended
to `evolve_history.jsonl`; manifests live under `core/evolve/capabilities/`.

## Component map

| Path | Role |
|---|---|
| `core/evolve/loop.py` | oracle driver, graph closure, gates, keep/restore ratchet |
| `core/evolve/action_catalog.py` | structured action context and fingerprint validator |
| `core/evolve/capability_contract.py` | graph foreign keys, prior access, planner contract |
| `core/evolve/exposure.py` | stable API-to-production-to-kernel proof |
| `core/analysis/flexgraph_extract.py` | source and compiler graph extractor |
| `benchmark/model_workloads.py` | immutable three-trace definitions and seeds |
| `benchmark/gates/model_eval.py` | local search, DP, empirical panel, paired TPU gate |
| `core/search/model_dp.py` | exact DP, bounded brute oracle, binary plan panel |
| `benchmark/live_model_candidates.py` | host-neutral Kimi candidate and policy mapping |
| `benchmark/gates/live_model_eval.py` | conditional checkpoint launch and metrics |
| `benchmark/runtime.py` | probe-authenticated model/callsite event collector |
| `core/runners/*.py` | editable kernel design spaces and dispatch mappings |
| `python/sgl_jax/srt/configs/kernel_control.py` | stable programmer control policy |
| `board/` | progress, objective, history, action catalog, flexibility graph |

The benchmark harness, graph extractor and validators, DP, loop, board source, and
maintainer tests are frozen during oracle work. Production kernels, stable controls,
production consumers, and runners are editable only as declared by one pending
manifest.
