# Capability manifests

One pending manifest proposes exposing one source-proven flexibility already present
in the generated v2 action graph. The graph, not free text, owns the source path,
axis, incumbent value, kernel family, affected model callsites, and action edge.

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

The top-level fields shown above are exact. `audit_case` is the only optional field.
`files_touched` must exactly equal every non-bookkeeping edit and must include the
manifest itself. Oracle-frozen harness files are forbidden.

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
default preserves the mined incumbent value. There is no authorization to add a new
kernel algorithm, schedule behavior, path, or unrelated abstraction.

The frozen evaluator instruments the derived backend entry and accepts only an
observed non-default argument at the exact affected model callsite. Self-reported
runtime events are not evidence.

## Acceptance

A round can be kept only when:

1. The fingerprint and `gap_id` identify one current executable red action.
2. The incumbent lacked the stable programmer control, while the selected low-level
   behavior already existed.
3. Typed control, consumer, backend, source function/sink, runner, and every affected
   callsite form one verified route.
4. The deployment space is a pure extension: every incumbent configuration remains
   at the new dimension's default, and every advertised value is exercised.
5. Regeneration closes only the selected action; unrelated action semantics remain
   unchanged.
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
