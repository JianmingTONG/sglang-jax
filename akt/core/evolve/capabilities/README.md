# Capability manifests

One JSON file per elevated flexibility, `<name>.json`. The oracle (or a human)
writes it after implementing the stack and registering the new search Knob, before
`submit`. The loop reads it to gate, restore, and board the capability.

## Schema

```json
{
  "name": "kv_slices_per_block",
  "gap": "kv_cache: get_best_num_slices_per_block — tuned table is dead code (early-returns 4/page)",
  "hypothesis": "one line: what elevating this gap should let the search do",
  "estimated_relief_pct": 8.0,
  "search_dimension": "one line: the new Knob (name + value set) now enumerated by the kernel's runner DesignSpace, and how run() plumbs it to the kernel",
  "audit_case": "kv_cache",
  "files_touched": [
    "python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py",
    "python/sgl_jax/srt/kernels/update_kv_cache/tuned_block_sizes.py",
    "akt/core/runners/kv_cache.py"
  ],
  "status": "pending"
}
```

## Field notes

- **`gap`** — copy the frontier line you chose from `evolve status` (the `[i] [...]` entry).
- **`estimated_relief_pct`** — your step-(1) estimate of bottleneck relief; the board
  compares estimate vs measured.
- **`search_dimension`** — the contract: the new choice MUST be a `Knob` (new axis, or a
  widened value set) in the kernel's runner `DesignSpace` (`akt/core/runners/<kernel>.py`),
  and `run(inputs, cfg)` must actually plumb that knob into the kernel — so the frozen
  search enumerates it and the timed config == the executed config (cost == execution).
- **`audit_case`** — the kernel this targets; `submit --audit` runs
  `adapter.py space --kernel <it>` to print the enlarged design-space size as evidence
  the new axis is enumerated (searched, not sampled).
- **`files_touched`** — EVERY file the capability changed or created, so `restore` can
  revert the whole add-on on a reject. Kernels are pure Python/Pallas — there is NO
  compiled artifact to list. Do NOT list anything under `akt/benchmark/` or
  `akt/core/evolve/loop.py` (FROZEN — the loop rejects such a manifest).
- **`status`** — start `"pending"`; the loop sets `"kept"`/`"rejected"` and adds
  `delta_pct` / `measured_geomean_s` / `reject_reason`.

## What a capability may / may not touch

- **Editable**: `python/sgl_jax/srt/kernels/**` (plumb the knob through the Pallas kernel)
  and `akt/core/runners/<kernel>.py` (register the Knob + extend the `run()` mapping).
- **FROZEN** (voids the round): `akt/benchmark/**` — the suite registry, the gate
  (`gates/eval.py`), the adapter, and `runners/base.py` (the contract + timer +
  correctness-check + search). Also `akt/core/evolve/loop.py`.
- **Reviewable invariant** (do not game the gate): a capability may ADD knobs / widen
  value sets / extend the run-mapping and the kernel — it must NOT weaken a case's
  `make_inputs` / `reference` / `atol` / `rtol` (the correctness contract).

The loop keeps iff the measured suite geomean beats the incumbent by more than
`target_improvement` (default 2%) with every runnable case still correct; otherwise it
reverts every `files_touched` entry to the incumbent commit.
