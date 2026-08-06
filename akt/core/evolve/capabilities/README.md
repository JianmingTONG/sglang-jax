# Capability manifests

One pending manifest proposes exactly one of:

- **ELEVATE** — exposing one source-proven flexibility already present in the
  generated v2 action graph as a red-link action (`access: "existing"`). The
  graph, not free text, owns the source path, axis, incumbent value, kernel
  family, affected model callsites, and action edge.
- **NOVEL** — inventing one new low-level algorithm, kernel variant, or standalone
  API. The axis NAME is free (no prescribed template). Two forms exist:
  - **Form (A) — VARIANT TOGGLE**: a new boolean/enum axis on an existing kernel
    entry, admissible against a mined per-launch FRONTIER SLOT
    (`access: "frontier"`, the graph's top-level `frontier_actions` list,
    fingerprinted together with the red-link actions): the manifest's `gap_id`
    foreign-keys the slot, its `proposed_action` completes the slot record, the
    incumbent algorithm stays the exact default, and after implementation the
    frozen extractor must stop emitting the slot and rediscover the new axis as a
    mined finding with exactly the declared semantics.
  - **Form (B) — STANDALONE API**: a genuinely new JAX-callable handle — its own
    function, typically its own FILE under `python/sgl_jax/srt/kernels/<family>/`
    — admissible against the family's PERPETUAL `<family>:new_api:standalone`
    slot (the graph's `standalone_frontier_actions` list). That slot never
    closes: introducing one API does not exhaust the family, so the extractor
    keeps emitting it, and graph closure requires its persistence.

## Schema

```json
{
  "name": "moe_v2_interleave_bt",
  "gap_id": "fused_moe/v2:interleave_bt:schedule-toggle",
  "action_graph_fingerprint": "<current-64-hex-fingerprint>",
  "hypothesis": "Shape-scoped BT interleaving reduces expert-v2 stalls.",
  "estimate": {
    "model": "tiny-moe-serving",
    "callsite": "tiny-moe-serving/expert-v2",
    "baseline_share_pct": 18.0,
    "expected_relief_pct": 3.2,
    "reasoning": "The call is 18% of aggregate time and the expected local reduction is 18%."
  },
  "search_dimensions": [
    {
      "control": "moe_v2.interleave_bt",
      "kernel_function": "fused_ep_moe_v2",
      "consumer": "python/sgl_jax/srt/layers/fused_moe.py"
    }
  ],
  "files_touched": [
    "python/sgl_jax/srt/configs/kernel_control.py",
    "python/sgl_jax/srt/layers/fused_moe.py",
    "akt/core/runners/moe_v2.py",
    "akt/core/evolve/capabilities/moe_v2_interleave_bt.json"
  ],
  "status": "pending"
}
```

The top-level fields shown above are exact. `audit_case` and `proposed_action` are
the only optional fields.
`files_touched` must exactly equal every non-bookkeeping edit and must include the
manifest itself. Oracle-frozen harness files are forbidden.

A NOVEL round adds the `proposed_action` object with exactly these fields,
completing the selected frontier slot (the same record shape
`action_catalog_context` emits for a mined action):

```json
"proposed_action": {
  "gap_id": "simple_gla:enable_simple_gla_fwd_variant:schedule-toggle",
  "family": "simple_gla",
  "kernel_ids": ["gla"],
  "source_axis": "enable_simple_gla_fwd_variant",
  "source_function": "simple_gla_fwd",
  "source_sink": {"assignments": ["..."], "expression_asts": {"...": "<ast>"}},
  "incumbent_value": false,
  "candidate_values": [false, true],
  "category": "schedule-toggle",
  "source_evidence": {"path": "python/sgl_jax/srt/kernels/...", "line": 0, "detail": "..."},
  "model_callsites": ["tiny-linear-serving/gla-long"],
  "action_edge": {"source": "action:<gap_id>", "target": "pallas:<primitive>"}
}
```

Form-(A) constraints: `gap_id` must be the id of one current frontier slot
(`<family>:enable_<fn>_variant:schedule-toggle`) and equal the manifest `gap_id`.
Graph-owned fields — `family`, `kernel_ids`, `source_axis`, `source_function`,
`category`, `incumbent_value`, `candidate_values`, `model_callsites`,
`source_evidence.path`, and `action_edge` — must equal the slot field-for-field;
the oracle authors only `source_sink`, `source_evidence.line`, and
`source_evidence.detail`. For a NOVEL round the slot — not the incumbent red-link
list — supplies the graph-owned fields below.

Form-(B) constraints (STANDALONE API): `gap_id` is `<family>:<axis>:schedule-toggle`
with a FREE `<axis>` name (e.g. `impl`, `gla_impl`, `dispatch_to`) that collides
with no existing finding or slot axis of the family; it is deliberately NOT a
per-launch frontier slot id. Requirements:

- the family must carry a perpetual `<family>:new_api:standalone` slot in the
  graph's `standalone_frontier_actions` (only in-suite families with model
  callsites get one);
- `category` must be `"schedule-toggle"` — the shape the frozen extractor mines
  for the dispatch control after implementation;
- `kernel_ids` and `model_callsites` must equal the standalone slot's (the
  callsites are derived from the kernel_ids); the family is the slot's family;
- everything else is ORACLE-OWNED: the axis name, the finite `candidate_values`
  domain (strings allowed, e.g. `["incumbent", "subchunk_v3"]`, and it must
  contain the typed `incumbent_value`), `source_function`, `source_sink` (the
  dispatch assignment, e.g. `fn = <new_handle> if impl == "<api_name>" else
  _incumbent`, in the extractor's mined-sink shape), and `source_evidence` —
  whose `path` may name a NEW file under `python/sgl_jax/srt/kernels/` (it must
  exist in the working tree NOW; it need not exist at the incumbent commit);
- the dispatch control is still deployed like any control: registered under
  `KernelControlPolicy`, forwarded from a production consumer to the backend
  entry, and exposed as a runner Knob — the default `"incumbent"` value must
  reproduce the output-hash-pinned incumbent path bit-exactly;
- prior-inaccessibility still applies: neither the entry argument nor the axis
  name may exist at the incumbent commit (access mode `novel-algorithm`);
- graph closure requires the regenerated graph to MINE the declared dispatch
  finding programmer-exposed, field-for-field equal to `proposed_action`, while
  the perpetual `<family>:new_api:standalone` slot MUST still be present — it
  never closes and its persistence is not an error.

Each search dimension contains exactly three fields:

- `control`: `<kernel-family>.<axis>`. The family must be one of the selected
  action's `kernel_ids`; the key becomes both the runner knob and backend argument.
- `kernel_function`: the backend entry in the graph-derived kernel source. It must
  forward the argument to the graph-derived `source_function` and, for a hardcoded
  literal, to its exact mined sink.
- `consumer`: a production path under `layers/`, `models/`, or `model_executor/`
  that resolves the stable control and forwards it to `kernel_function`.

Do not copy graph-owned fields into the manifest. The validator derives
`source_evidence`, callsites, source axis, default, kernel path/family/argument, and
the complete finite candidate domain. The runner must expose that exact domain. This
avoids two authorities for the same action and prevents a proposal from inventing a
new value or mode.

## Required implementation route

The implementation must create one inspectable path:

`KernelControlPolicy` typed field and registry → production consumer → backend entry
argument → selected source function/sink → runner knob with the same
`programmer_control`.

If the backend argument already existed, the round only exposes it. If the action was
a hardcoded literal, the round may lift that exact literal into an argument whose
default preserves the mined incumbent value. A NOVEL round may add a new kernel
algorithm, variant, or standalone API, but only strictly behind the declared
axis's non-default values: the default path must reproduce the incumbent
bit-exactly (it is output-hash pinned), and graph closure must find the new axis
mined programmer-exposed with exactly the declared semantics — additionally, for
form (A) the per-launch slot must no longer be emitted, while for form (B) the
perpetual standalone slot must still be emitted. An invented behavior that is
anchored to neither a frontier slot nor its family's standalone slot — or that
drifts from its declaration — fails closure.

The frozen evaluator instruments the derived backend entry and accepts only an
observed non-default argument at the exact affected model callsite. Self-reported
runtime events are not evidence.

## Acceptance

A round can be kept only when:

1. The fingerprint matches the current canonical two-source context, and the
   `gap_id` identifies one current executable red-link action (ELEVATE), one
   current frontier slot whose record the `proposed_action` completes (NOVEL
   form A), or a free-named axis authorized by the family's
   `<family>:new_api:standalone` slot (NOVEL form B).
2. The incumbent lacked the stable programmer control. For ELEVATE the selected
   low-level behavior already existed; for NOVEL neither the entry argument nor
   the axis existed at the incumbent commit (access mode `novel-algorithm`).
3. Typed control, consumer, backend, source function/sink, runner, and every affected
   callsite form one verified route.
4. The deployment space is a pure extension: every incumbent configuration remains
   at the new dimension's default, and every advertised value is exercised.
5. Regeneration closes only the selected action; unrelated action semantics remain
   unchanged. For NOVEL, the regenerated graph must mine the declared finding
   under the same `gap_id` programmer-exposed, field-for-field equal to
   `proposed_action`, and must preserve unrelated red-link actions and unrelated
   frontier/standalone slots outside the selected family. Form A additionally
   requires the selected per-launch frontier slot to no longer be emitted (the
   axis now exists); form B requires the perpetual standalone slot to STILL be
   emitted (it never closes).
6. Every deployable local configuration is correct and measured on the target TPU.
7. Exact additive DP agrees with the bounded Cartesian solver check, and the bounded
   empirical whole-trace panel supports its selection.
8. The selected non-default value is observed by the frozen backend probe.
9. Both same-round paired and stored-baseline absolute synthetic geomean improve by
   more than the campaign threshold (at least 2%).
10. When the action changes Kimi-Linear's GMM-v2 route, the same-round live quality,
    completion, policy-attestation, and throughput guards pass. Other actions are
    explicitly `not_applicable`; incompatible hardware is a no-attempt topology skip.

The DP optimum is only for the finite fingerprinted additive graph. The empirical
panel is bounded, and the conditional Kimi result is end-to-end policy/request
evidence rather than a machine-observed per-kernel route or a global SOTA claim.
