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
- A round may add API and forwarding plumbing only to expose the selected existing
  axis. `existing-backend-argument` and `existing-low-level-axis` are the only access
  modes; neither authorizes a new kernel algorithm or semantic behavior.
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
   does not re-execute the prior Git revision or prove semantic equivalence of the
   default path, so cross-revision causal attribution remains incomplete.
5. The checked-in historical campaign still requires TPU `rebaseline` before it can
   advance under the v2 objective.
6. The Kimi checkpoint is identified by Hugging Face repository name but is not
   pinned to an immutable model revision, so a future upstream update could change
   the artifact evaluated by a later run.
7. The live gate starts selected then incumbent in a fixed order with separate server
   startups; unlike the synthetic gate, it does not balance startup-order drift.
