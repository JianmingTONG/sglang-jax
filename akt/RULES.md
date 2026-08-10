# AKT loop — behavioral rule registry

Normative rules the loop's behavior MUST satisfy. Every rule names its enforcing
code and its automated compliance check. Any modification to the loop, benchmark,
workloads, or diagnostics must keep `rules_check` green:

```bash
PYTHONPATH=python:. .venv/bin/python akt/core/verify/rules_check.py   # all rules
pytest akt/benchmark/test_rules_check.py                              # same, in the battery
```

A rule change is itself a change to the measurement contract: edit this file and
`rules_check.py` in the same commit, and say why in the commit message.

---

## R1 — Unified workload definition

**RULE:** Every workload in the measured objective MUST be expressible in the
formal representation (`WorkloadSpec → BlockSpec → OpSpec`) and reproduce the
active `MODEL_WORKLOADS` call-for-call when lowered. Hand-authored call tuples
outside the formal layer are not admissible; extended workloads enter ONLY via
the spec library + `AKT_EXTENDED_WORKLOADS` + rebaseline.

- WHY: workloads must generalize to real architectures (Qwen3, Kimi-Linear,
  DeepSeek, all-kernels) without hand-editing the frozen contract; one authority
  for what a workload IS.
- ENFORCED BY: `benchmark/workload_spec.py` (`legacy_specs`,
  `verify_legacy_roundtrip`, `lower`), `benchmark/model_workloads.py`
  (`_extended_workloads`, `validate_workloads`), `benchmark/model_specs.py`.
- CHECK: `rules_check.check_r1_unified_workloads` — legacy roundtrip exact;
  frozen prefix intact; every extended workload re-derivable from the library;
  unknown ids fail closed.

## R2 — Bottleneck obtained via the campaign diagram

**RULE:** The bottleneck handed to the oracle MUST be derived by the
campaign-diagram pipeline — measure (L0 wall time) → trace (op graph) → rate →
verdict (compute/memory/dependency-bound) → recurse to a LIMITER — never by
reading source code alone and never by assertion. The LIMITER must name a
measured callsite of the active workloads, and the same diagnosis must be
renderable as the board's campaign diagram.

- WHY: WHICH is slow must come from execution; WHY may be a traced proxy, but it
  must be the declared, reproducible pipeline — not oracle intuition.
- ENFORCED BY: `core/analysis/campaign.py` (`diagnose`, `_verdict`,
  `render_query_block`), `benchmark/adapter.py bottleneck` (QUERY source),
  `board/build.py::_campaign` + `board/index.html::renderCampaign` (rendering).
- CHECK: `rules_check.check_r2_campaign_diagram_bottleneck` — `diagnose()` runs
  from real inputs; every L0 segment carries a named `latency_source`; the
  LIMITER names an inventory callsite; L1 stage shares declare their estimation
  basis.

## R3 — One global utilization scale (diagram consistency)

**RULE:** Within one diagnosis, all utilizations MUST share one scale (busiest
measured segment = 1.0), so a parent segment equals the duration-weighted mean
of its children. Per-level re-normalization that lets a child read "100%" under
a "~0%" parent is non-compliant; within-segment op-mass composition must be
labeled as composition, not utilization.

- WHY: a breakdown whose averages cannot reproduce the whole is not a breakdown.
- ENFORCED BY: `core/analysis/campaign.py` mass/rate computation; the viewer
  (`.claude/skills/campaign_diagram_bottleneck_viewer`).
- CHECK: `rules_check.check_r3_global_scale` — recomputes L1 child rates of the
  dominant L0 call on the global scale and asserts the duration-weighted child
  average equals the parent within 1%.

## R4 — Serialized proposal forms: ELEVATE before INVENT

**RULE:** A novel proposal (`proposed_action`) is inadmissible while any
red-link action measurable under the current campaign remains un-attempted
(no keep/reject record under the active objective). Sub-bar red-links are
declared (cheap no-eval reject), never silently skipped.

- ENFORCED BY: `core/evolve/loop.py` (`elevate_remaining`,
  `enforce_phase_serialization`, `record_phase_reject`, `phase_block`).
- CHECK: `rules_check.check_r4_serialization` — synthetic-state probe: with an
  un-attempted measurable red-link, a novel manifest raises; after a recorded
  attempt it passes.

## R5 — Invention only through the flexgraph interface

**RULE:** A new algorithm/variant/API is admissible ONLY as a declared canonical
action record (frontier or standalone slot foreign key + `proposed_action`);
after implementation the regenerated graph MUST re-mine the declared finding
programmer-exposed, field-for-field equal. Undeclared invention voids the round.

- ENFORCED BY: `core/evolve/capability_contract.py` (`validate_proposed_action`,
  access mode `novel-algorithm`), `core/evolve/loop.py`
  (`candidate_action_graph_closure`, `_novel_finding_mismatches`),
  `core/evolve/action_catalog.py`.
- CHECK: `rules_check.check_r5_flexgraph_interface` — catalog validates; minable
  categories are the closed set; a colliding/unknown novel declaration is
  rejected.

## R6 — The default path is the incumbent, bit-exactly

**RULE:** Every programmer control's default MUST reproduce the incumbent
behavior; an empty `KernelControlPolicy` is bit-identical to stock. The
incumbent plan re-executed on a post-edit tree MUST reproduce pinned
per-callsite output hashes.

- ENFORCED BY: `srt/configs/kernel_control.py` (registry + typed defaults),
  the gate's incumbent output-hash pin (`core/evolve/loop.py`,
  `benchmark/gates/model_eval.py`).
- CHECK: `rules_check.check_r6_default_is_incumbent` — empty policy resolves to
  the dataclass defaults for every registered family; elision flags stay
  un-registered.

## R7 — Frozen measurement, fail-closed

**RULE:** The benchmark harness, workload contract, references/tolerances,
extractor, validators, DP, and loop are FROZEN during oracle work; any frozen
edit voids the round. Measurement never silently narrows: unmeasured calls are
reported (not dropped), unsupported ops are coverage gaps (not omissions),
unknown inputs raise.

- ENFORCED BY: `core/evolve/loop.py` (`FROZEN`, `_frozen_fingerprint`,
  `revert_worktree`), `benchmark/suites.py` contract validation,
  `benchmark/workload_spec.py` coverage reports.
- CHECK: `rules_check.check_r7_frozen_fail_closed` — FROZEN covers the
  measurement-defining paths; workload lowering accounts for every op
  (measurable + gaps == expanded).

## R8 — KEEP requires measured, guarded improvement

**RULE:** KEEP requires every guard (correctness of every valid config, DP ==
bounded brute, empirical panel not falsified, closure, exposure, contract,
probes, hash pins) AND both paired and absolute geomean improvement strictly
above the campaign bar (floor 2%). A raw win that fails any guard is rejected.

- ENFORCED BY: `core/evolve/loop.py::gate_capability` (`guard_ok` conjunction +
  threshold), `benchmark/gates/model_eval.py`, `core/search/model_dp.py`.
- CHECK: `rules_check.check_r8_keep_bar` — the live gate contract declares the
  objective/direction; the loop enforces the 2% floor; campaign history shows
  no KEEP below the bar and rejected rounds carry guard reasons.

---

Provenance note: R1–R3 codify the workload-generalization and
campaign-diagram-bottleneck requirements (2026-08-09/10); R4–R8 codify
mechanisms that already existed — this registry makes them citable and
checkable in one place.
