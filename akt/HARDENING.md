# AKT hardening roadmap — porting robustness from `maple/akt` (A)

This sglang-jax AKT loop (**B**) was forked from the mature FHE/ORION AKT loop
(**A**, `maple/akt`). B inherited A's *engine* almost verbatim — the same five
subcommands, the same oracle seam (claude/codex, stream-json, watchdog, rate-limit
sleep), the same FROZEN-fingerprint tamper guard, the same void handling, and the same
keep→git-commit incumbent ratchet. What B did **not** inherit is A's downstream
robustness: a hard **gate**, a dedicated **verification harness**, a **convergence /
design-space analysis** layer, and **board integrity**.

This doc records the gap. **Tier 0 (the four confirmed latent bugs) is DONE** (see the
commit that adds this file). **Tiers 1–3 are the planned port**, each item annotated
with its source in A, its value to B, and a rough effort.

> Scope note: A's proofs are about the `vn_go` layout **DP** search; B's "search" is the
> `runners/base.py::search_best` *enumerate → check-correct → time → argmin* autotune.
> Where an A feature is CKKS/FHE-specific it is called out as **not portable**; the rest
> port as the *idea*, re-expressed against B's autotune + real-latency objective.

---

## Tier 0 — confirmed latent bugs (DONE)

| # | Bug (B) | Fix | File |
|---|---------|-----|------|
| 1 | `search_best(cap=256)` silently truncated the larger moe spaces (raw ~750 / ~1620 configs) — "optimal within the space" was false and nothing surfaced it. | cap raised to `4096` (matches `enumerate`'s default → current kernels searched exhaustively) **and** a `truncated` flag now propagates `base → eval → gate`; a truncated search **fails** the gate. | `benchmark/runners/base.py`, `benchmark/gates/eval.py`, `core/evolve/loop.py` |
| 2 | A capability could be KEPT while a kernel it affects was never correctness-checked: `correct=None` (tpu-deferred) passed silently and there was **no missing-case guard**. | `init` snapshots `expected_cases`; the gate now fails if any expected case is **missing** from the eval (a broken runner import that drops a kernel), and records `n_verified` / `n_deferred` so a KEEP with unverified kernels is visible. | `core/evolve/loop.py` |
| 3 | `hw_eval` never deleted `.evolve_eval.json` first, so a **crashed eval re-read the prior round's numbers** as if fresh. | `out_path.unlink(missing_ok=True)` before the eval — a crash now leaves no file and correctly fails the round. | `core/evolve/loop.py` |
| 4 | `build.py` dumped `board.json`/`flexgraph.json` with `allow_nan=False` **without cleaning**, so a single NaN/Inf metric raised and left the board un-rebuilt. | added `_clean()` (coerce non-finite floats → `null`) applied before both dumps. | `board/build.py` |

---

## Tier 1 — gate hardening (the biggest structural gap)

B's gate is a 2-stage funnel (speed + a weak correctness check). A's is a 5-stage funnel
where each stage is a hard reject and the expensive eval runs **last** (fail-cheap-first).
Port, in priority order:

1. **Manifest v2 validation contract.** *A: `loop.py::validate_manifest`.* B's
   `load_manifest` only substring-checks `files_touched` against FROZEN — no schema, no
   required fields, no path-safety, no "is this knob actually a live runner Knob" check. A
   malformed or path-traversing manifest passes. **Port:** a `validate_manifest(m)` that
   checks a name regex, required fields, `_manifest_path` safety (reject absolute / `..` /
   non-normalized), the FROZEN set, and that each declared knob resolves to a real
   `Knob(...)` in `core/runners/`. *Effort: low–med · generic.*

2. **Execution-evidence cross-check.** *A: `loop.py::_execution_evidence`.* B never
   verifies the manifest's declared knob was actually **selected by the autotuner** and
   **changed behaviour** — despite B's own "cost == execution" premise. **Port (light
   analog):** assert the declared knob name appears in the winning `best_config` for at
   least one case, and that the best config differs from the default (else the "new
   flexibility" is a silent no-op that only shuffled timing noise). *Effort: med · generic.*

3. **Per-round optimality certificate.** *A: the per-round `{production_cost, exact_cost,
   modeled_gap, space_fingerprint, certified_global_modeled_optimum}` record.* B keeps/
   rejects on geomean but records no proof the enlarged space was fully searched. **Port:**
   emit one machine-readable record per round `{space_size, n_valid, n_correct, truncated,
   exhausted, best_config, space_fingerprint}` (most fields already exist post-Tier-0) and
   wire it into `evolve_history.jsonl` + the board. *Effort: low–med · generic.*

4. **Fail-cheap-first ordering.** *A: eval runs only after verifier + certificate pass.* B
   runs the expensive `hw_eval` unconditionally every round. **Port:** run the manifest
   validation + a cheap search/exhaustiveness check *before* `hw_eval`, so a broken round
   rejects without paying for the full suite eval. *Effort: low · generic.*

---

## Tier 2 — a real `akt/core/verify/` harness (B has none today)

A ships a dedicated `core/verify/` dir whose only job is to **prove** the search and the
gate, separate from the perf loop. B's correctness lives entirely inside the frozen perf
harness (`check_correct`'s allclose). Create `akt/core/verify/` with:

1. **Brute-vs-search optimality proof.** *A: `dp_verify.py::verify_synthetic`.* Build a toy
   `DesignSpace` whose fastest config is known (or force a synthetic timing), and assert
   `search_best` returns it. Today B's optimality is only a docstring claim. *Effort: low ·
   light-FHE (rewrite for autotune).*

2. **Negative control (the gate can fail).** *A: `layout_verify.py` asserts a deliberately-
   wrong result is rejected.* Feed `check_correct` a mutated/wrong output and assert it
   **rejects** it — proving the allclose gate detects real bugs (guards against a reference
   and a run sharing a bug, or coincidentally-matching shapes). *Effort: low · generic.*

3. **Golden `best_config` regression.** *A: `dp_verify.py::verify_models` pins the DP result
   for the 3 real models.* Pin a per-case golden `best_config` (or golden `search_note`) so a
   runner/objective regression that changes the chosen optimum is caught. *Effort: low ·
   generic.*

4. **Bit-exact identity checks for config-independent transforms.** *A: `input_verify.py`
   asserts a `~1e-9` identity separate from the loose end-to-end gate.* Several B kernels
   have paths that should be bit-exact regardless of the knob (e.g. `fused_mlp`'s per-
   `b_inter` weight re-interleave); assert those tightly instead of under the blanket
   `atol=rtol=2e-2`. *Effort: low · light-FHE.*

5. **DEFAULT-reproduces-incumbent proof.** *A: `verify_prerot` proves ON==OFF within noise
   and OFF has no markers.* Prove that a capability's **default** config reproduces the
   shipped kernel exactly (no silent behavioural drift on the un-elevated path). *Effort:
   med · light-FHE.*

---

## Tier 3 — convergence & board completeness

1. **Convergence / optimum-distance certificate.** *A: `core/analysis/tier_analysis.py`
   emits `optimality_gap_upper_bound == 0.0`.* B's loop has no principled **stopping
   signal** — it can't tell "design space exhausted, stop" from "keep elevating". B already
   computes half of it (`elevated_in_akt` in `flexgraph_extract.py`); extend it into a
   per-run certificate. *Effort: med · light-FHE.*

2. **Supported → Enabled → Selected funnel + utilization ratios.** *A: `tier_analysis.py` /
   `space_breakdown.py`.* B reports only an absolute `space.size()` + `best_config`; it can't
   say whether a newly-elevated knob usefully **enlarged the reachable space** or just
   over-exposed it, nor localize exposed-but-unused flexibility to a specific knob. *Effort:
   med · light-FHE.*

3. **Board correctness-replay (taint reverted KEEPs).** *A: `board/correctness_replay.py`.* A
   later-reverted or `correct=False` KEEP still counts in `kept` and its geomean stays the
   displayed incumbent. Port the tainted-interval state machine so an invalidated KEEP can't
   remain the shown incumbent. *Effort: low–med · generic.*

4. **Self-describing `gate` adapter subcommand.** *A: `adapter.py::cmd_gate` emits
   `objective_name` / `objective_unit` / `lower_is_better`.* B's loop is hardwired to geomean
   latency. A metric-agnostic `gate` contract decouples the objective (useful when the TPU
   run swaps the metric). *Effort: low · generic.*

5. *(nice-to-have)* design-choice **activation timeline**, multi-run **archival**, and a
   run-to-run **noise band** on the board — all `board/build.py` mechanisms in A. *Effort:
   med · generic.*

---

## Explicitly NOT ported

- **`calibrate_weights.py`** — A microbenchmarks CKKS `Rotate`/`Mul`/`Add` to calibrate the
  ROT/MUL/CT cost-model weights. **B optimizes directly on measured latency** (`perf_counter`
  + `block_until_ready`); there is no proxy cost model, so there are no weights to calibrate.
  No B analog needed.
- **`slot_util.py` / `packing_dump.py`** — CKKS ciphertext-slot utilization and per-layer
  packing dumps. Deep-FHE; only a loose "VMEM/VREG/MXU occupancy vs latency" analog would
  ever apply, and it would be a fresh implementation, not a port.
