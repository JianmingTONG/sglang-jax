# Capability manifests

One JSON file per elevated flexibility, `<name>.json`. The oracle (or a human)
writes it after implementing the stack and registering the new search Knob, before
`submit`. The loop reads it to gate, restore, and board the capability.

## Schema

```json
{
  "name": "kda_prefetch_distance",
  "gap": "kda: prefetch distance is fixed in the state propagation stage",
  "hypothesis": "one line: what elevating this gap should let the search do",
  "estimated_relief_pct": 8.0,
  "search_dimension": "one line: the new Knob now enumerated by the runner and how run() plumbs it to the kernel",
  "audit_case": "kda",
  "programmer_controls": [
    {
      "control": "kda.prefetch_distance",
      "config_path": "python/sgl_jax/srt/configs/kernel_control.py",
      "consumer": "python/sgl_jax/srt/layers/attention/linear/kda_backend.py",
      "kernel_argument": "prefetch_distance"
    }
  ],
  "files_touched": [
    "python/sgl_jax/srt/kernels/kda/kda.py",
    "python/sgl_jax/srt/configs/kernel_control.py",
    "python/sgl_jax/srt/layers/attention/linear/kda_backend.py",
    "akt/core/runners/kda.py"
  ],
  "status": "pending"
}
```

## Field notes

- **`gap`** — copy the frontier line you chose from `evolve status` (the `[i] [...]` entry).
- **`estimated_relief_pct`** — your step-(1) estimate of bottleneck relief; the board
  compares estimate vs measured.
- **`search_dimension`** — the contract: the new choice MUST be a new `Knob` axis in
  the kernel's runner `DesignSpace` (`akt/core/runners/<kernel>.py`), and
  `run(inputs, cfg)` must actually plumb that knob into the kernel. The Knob must set
  `programmer_control="<family>.<key>"`. Widening an existing control's values is
  parameter tuning, not a capability-elevation round.
- **`audit_case`** — the kernel this targets; `submit --audit` runs
  `adapter.py space --kernel <it>` to print the enlarged design-space size as evidence
  the new axis is enumerated (searched, not sampled).
- **`programmer_controls`** — the end-to-end exposure proof. Each new runner control
  declares its stable `<family>.<key>` API identifier, the central config definition,
  the production layer/backend that consumes it, and the low-level kernel keyword it
  forwards. The frozen exposure gate verifies all four against source and eval metadata.
- **`files_touched`** — EVERY file the capability changed or created, so `restore` can
  revert the whole add-on on a reject. Kernels are pure Python/Pallas — there is NO
  compiled artifact to list. Do NOT list anything under `akt/benchmark/`, the board
  implementation, `flexgraph_extract.py` miner, `akt/core/evolve/loop.py`, or
  `akt/core/evolve/exposure.py` (FROZEN).
- **`status`** — start `"pending"`; the loop sets `"kept"`/`"rejected"` and adds
  `objective_scope` plus `delta_pct` / `measured_geomean_s` / `reject_reason`. Manifests
  from an older objective remain audit history but are not presented as active KEEP or
  REJECT evidence to the oracle.

## What a capability may / may not touch

- **Editable**: `python/sgl_jax/srt/kernels/**`, the stable programmer API at
  `python/sgl_jax/srt/configs/kernel_control.py`, a production consumer under
  `python/sgl_jax/srt/layers/**` or `model_executor/**`, and
  `akt/core/runners/<kernel>.py`.
- **FROZEN** (voids the round): `akt/benchmark/**` — the suite registry, the gate
  (`gates/eval.py`), the adapter, and `runners/base.py` (the contract + timer +
  correctness-check + search). Also `flexgraph_extract.py`, `akt/core/evolve/loop.py`,
  `exposure.py`, and board source (`build.py`, `index.html`, tests, and vendored chart).
- **Reviewable invariant** (do not game the gate): a capability may add a genuinely new
  control and extend the run mapping and kernel — it must NOT weaken a case's
  `make_inputs` / `reference` / `atol` / `rtol` (the correctness contract).

The loop keeps iff the programmer-exposure gate passes, every runnable case remains
correct, and the measured suite geomean beats the incumbent by more than
`target_improvement` (default 2%). Otherwise it reverts every `files_touched` entry.

The measured objective is `stateful-serving-deployable-v1`: recurrent cases use a
nonzero incoming state and compare both token output and final state. Capability-added
runner knobs without a registered production control are anchored at their default in
this deployment search. They remain useful as research experiments, but cannot select
the incumbent or justify a KEEP.
