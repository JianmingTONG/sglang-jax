# AKT hardening status

The enforced objective scope is `model-serving-empirical-dp-v2`.

## Enforced now

- The flexibility graph is a machine-readable guardrail. Its canonical v2 context
  includes target, action identities, kernel-family mappings, source axes, finite
  candidate domains, call sinks or normalized assignment expressions, evidence,
  callsites, and edges under one SHA-256 fingerprint. The checked-in graph
  exposes five source-proven existing actions. Dead or missing tuning tables,
  numeric-representation toggles, and other findings without an exact live
  sink/argument remain analysis-only context.
- Campaign state pins that fingerprint. Oracle execution and pending manifests fail
  before TPU work when the graph is stale, invalid, empty, or foreign keys do not
  match exactly.
- A round may add API and forwarding plumbing to expose the selected existing axis
  (`existing-backend-argument` / `existing-low-level-axis`), or may add ONE new
  algorithm/variant under the `novel-algorithm` access mode — admissible only
  against a mined FRONTIER SLOT (`access: "frontier"`, the graph's top-level
  `frontier_actions`, under the same fingerprint as the red-link actions): the
  manifest foreign-keys the slot's `gap_id` and its `proposed_action` completes
  the slot record (graph-owned fields equal to the slot; oracle-owned = source
  sink, evidence line, detail), with neither the entry argument nor the axis
  present at the incumbent commit and the incumbent algorithm preserved as the
  exact (output-hash-pinned) default. Closure requires the regenerated graph to
  stop emitting the slot AND to mine the declared finding programmer-exposed,
  field-for-field equal to the declaration. Undeclared invention and declaration
  drift from the slot both fail closed.
- Runner-only benchmark knobs are default-only in the deployment space. Promoting an
  existing knob to a stable programmer control expands the exact graph-owned domain
  without changing the incumbent default.
- The gate proves exact affected callsites, graph `source_axis`, mapped
  `kernel_family`, incumbent access history, runner values/default, production
  forwarding, backend argument observation, and complete file declaration.
- Every valid deployable local configuration is measured and correctness-checked.
  Exact layered DP searches the complete finite measured additive graph; a bounded
  Cartesian oracle independently checks the solver.
- `synthetic_trace_additivity` empirically challenges the DP choice with the complete
  valid binary neighborhood of DP-selected versus cheapest-distinct stage choices,
  up to 64 plans. Complete plans are correctness-checked and timed in balanced pairs.
  Deterministic bootstrap bounds on regret and multi-factor interaction must pass;
  truncation and inconclusive evidence fail closed.
- Candidate and incumbent three-trace plans run in paired alternating order on the
  same compiled TPU. KEEP still requires both paired and absolute geomean improvement
  strictly above 2% or the higher campaign threshold.
- After the synthetic process exits, a compatible single-host TP4 TPU runs the actual
  `moonshotai/Kimi-Linear-48B-A3B-Instruct` checkpoint. GMM-v2 actions use a direct
  Kimi EPMoE→GMM-v2 policy view and require a changed selected control; other actions
  are explicitly not applicable. GSM8K,
  two prefill points, request completion, server-policy attestation, and a 0.98
  input-throughput ratio floor are enforced. Ineligible topology is an explicit
  no-import/no-download skip; an eligible attempt must pass.
- The candidate graph is regenerated before hardware evaluation. KEEP is impossible
  unless the selected action closes without collateral action loss, and only the
  validated graph is installed for the next round.
- Fail-cheap-first: the eval-independent halves of the programmer-exposure and
  capability-contract validators run in a `static_only` pre-check (plus a static
  scan of every manifest-declared edit for references to the frozen probe-evidence
  channel) BEFORE the target-HW eval; a failing round rejects without paying for the
  three-model measurement, and the full validators still run on the real summary.
- Backend-probe reports are authenticated by CODE-OBJECT IDENTITY: only wrappers the
  frozen evaluator registered may record a `verified_backend_probe` event; a direct
  or filename-spoofed call from editable code raises. Registration is frozen-tree
  gated, and the static forgery scan is the second layer (residual below).
- Default-reproduces-incumbent: init/rebaseline/KEEP pin per-callsite output hashes
  of the incumbent plan; every round re-executes that plan on the post-edit tree and
  rejects on any bit-level drift — a default-path semantic change hiding inside the
  family allclose tolerance fails closed.
- Bit-exact invariant: a kernel whose frozen refs declare `BITEXACT_INVARIANT`
  (kv_cache's DMA scatter-copy) must produce bit-identical outputs across its FULL
  research space (runner-only values included); divergence or an incorrect research
  config fails the eval. The suite contract pins the flag so an editable runner
  cannot drop it. fused_mlp is deliberately NOT declared (MXU blocked-summation
  rounding is not structurally order-invariant).
- The DP preflight pins a GOLDEN plan+cost per workload (pure functions of the
  frozen callsite ids), and negative controls prove the correctness comparator can
  fail (wrong output / wrong shape / wrong leaf count rejected; a wrong config is
  excluded from the DP-visible measurements; probe forgery and registration abuse
  raise).
- Red-link exhaustion is a first-class verdict: an EMPTY-but-valid action catalog
  records `space_exhausted` (checked before every round, including after a mid-run
  KEEP closes the last action) and the board shows the exhaustion card (live
  open-action count wins over the recorded flag). Exhaustion covers only the
  red-link list — frontier slots remain actionable — so it no longer stops `run`:
  remaining rounds are frontier/novel-only, bounded by `--rounds` and the campaign
  deadline; `rebaseline` clears the flag when the graph is widened.
- API-NOVELTY GUARD: two calibrated guardrails filter parameter tweaks out of
  KEEP. (1) Directional ISA-grounded redundancy — candidate and incumbent
  programs are lowered to operation dependency graphs (jaxpr DAGs incl. Pallas
  kernel bodies) under identical shape/dtype/compiler settings; the duplicate
  risk `D(N)=max_E R(N,E)` (ancestry-WL maximum-weight dependency-preserving
  match, instruction-count weights first) must fall below the pinned
  leave-one-out percentile of the family's own one-knob parameter-tweak
  population, unless the directional generalization signature holds (low
  `R(N,E*)`, high `R(E*,N)`). (2) Genericity — `G_delta` upper-layer coverage
  over the frozen model callsites (each pattern once; lower-layer fan-out
  reported separately as implementation breadth) must reach the pinned
  percentile of the existing programmer controls' own coverage. Thresholds are
  catalog percentiles profiled from the existing APIs
  (`api_metrics.py profile` -> `api_baseline.json`, both frozen), pinned at
  init/rebaseline by fingerprint, recomputed live by the frozen evaluator at
  gate time from the same measured `case_search` population; missing baseline,
  untraceable programs, or fingerprint mismatch fail closed. Applies uniformly
  to ELEVATE and NOVEL rounds.
- The gate's metric coupling is a self-describing contract (`adapter.py gate`:
  objective name/unit/direction/summary keys) fetched LIVE from the frozen adapter
  at gate time — never trusted from editable campaign state.
- Correctness replay: `restore` refuses a KEPT capability (its code IS the incumbent
  commit; a checkout there is a no-op) and points at git-revert + `rebaseline`, which
  detects reverted keeps (revert commit or vanished keep commit in HEAD's history),
  downgrades their manifests, and taints their history rounds `post-run-invalid`;
  the board's incumbent envelope and headline then stop crediting them. `restore`
  also refuses to run concurrently with a live round.
- Every gated round archives its full eval summary (per-config latency samples,
  paired ratios, certificates) under `optimization_history/evals/` keyed by campaign
  and round, records a paired-ratio noise band the board draws around each attempt,
  and the board adds a research→deployable→valid→measured→selected design-space
  funnel with per-knob attribution.

## Claim boundary

The graph constrains what the LLM may expose; it does not discover the winning
implementation or replace measurement. The LLM remains the proposal oracle.

The DP is the exact optimum of its finite additive measured-cost graph. `DP == brute`
is a bounded solver check. The empirical binary panel tests a selected interaction
neighborhood, but it is not exhaustive search over the full non-additive model space,
arbitrary modified programs, or external SOTA implementations.

The synthetic traces cover every AKT kernel family and prefill/decode phase. The live
gate adds one real Kimi-Linear TP4 checkpoint and two prefill operating points. It
directly maps only GMM-v2 actions today; fused-MoE-v2 and fused-MLP still lack a
single-host direct checkpoint candidate. Policy attestation plus source inspection
provides end-to-end request evidence under the selected policy, not a machine-observed
per-kernel route,
and an ineligible-topology skip carries no live-model claim.

## Remaining limits

1. Broader checkpoint, topology, decode, and traffic coverage remains outside this
   objective.
2. A larger non-additive plan panel or adaptive interaction search would strengthen
   evidence beyond the current fail-closed binary neighborhood.
3. Direct per-kernel tracing in the live Kimi server would strengthen attribution
   beyond policy attestation and completed requests.
4. Candidate and incumbent variants share the post-edit implementation; the loop
   does not re-execute the prior Git revision. The pinned incumbent output hashes
   now prove the default path's OUTPUTS are bit-identical across edits (within a
   compiler version), but cross-revision TIMING attribution remains incomplete.
5. The checked-in historical campaign still requires TPU `rebaseline` before it can
   advance under the v2 objective (the output-hash pin and funnel populate then).
6. The Kimi checkpoint is identified by Hugging Face repository name but is not
   pinned to an immutable model revision, so a future upstream update could change
   the artifact evaluated by a later run.
7. The live gate starts selected then incumbent in a fixed order with separate server
   startups; unlike the synthetic gate, it does not balance startup-order drift.
8. CPython has no true in-process privilege boundary: code that deliberately spoofs
   `co_filename` via `compile` could self-register a probe code object. The static
   scan of manifest-declared edits for runtime-module references is the second
   layer; a forgery must evade both, and the residual is accepted and documented
   rather than claimed away.
9. The bit-exact invariant and incumbent output-hash pins execute only on target
   hardware (the whole v2 gate is TPU-only), so on a non-TPU box they are wired but
   dormant, like every other eval-side guard.
