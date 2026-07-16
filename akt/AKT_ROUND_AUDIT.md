# AKT round audit (R1-R12)

Audit endpoint: `eecce8353a9042b66a7f42ca0e8cd4e3cfa8efb4` (R12), 2026-07-16.

## Verdict

No benchmark tampering or mismatch against the **historical output-only contract** was
found in R1-R12. That contract supplied no incoming recurrent state and did not compare
the returned state, so this audit does **not** establish production-serving correctness
or a valid stateful incumbent. In particular, R8-R11 optimized work that the benchmark
declared unobservable but subsequent decode requires. They are retained as legacy
experiments, not deployable capability wins.

The non-elision controls remain candidates for the corrected stateful search, but their
recorded deltas must not be carried across objective scopes. Current acceptance uses
`stateful-serving-deployable-v1`, starts recurrent cases from nonzero state, compares
both token output and final state, and anchors runner-only controls at defaults. This
host still provides Pallas-interpret rather than TPU latency evidence.

## Legacy round results

| Round | Capability | Recorded geomean | Delta | Historical status |
|---:|---|---:|---:|:---:|
| 1 | `kda_state_block_chunks` | 0.7722 ms | -3.60% | remeasure under current objective |
| 2 | `kda_intra_solve_blocks` | 0.7266 ms | -5.90% | remeasure under current objective |
| 3 | `kda_scalar_intra_solve` | 0.6722 ms | -7.49% | remeasure under current objective |
| 4 | `gla_compact_alignment` | 0.5567 ms | -17.18% | remeasure under current objective |
| 5 | `gla_full_sequence_chunk` | 0.5255 ms | -5.62% | remeasure under current objective |
| 6 | `kda_state_dim_alignment` | 0.5144 ms | -2.10% | remeasure under current objective |
| 7 | `kda_compute_block_chunks` | 0.4993 ms | -2.94% | remeasure under current objective |
| 8 | `gla_single_chunk_state_elision` | 0.3901 ms | -21.87% | output-only, not deployable |
| 9 | `kda_single_chunk_state_elision` | 0.3297 ms | -15.48% | output-only, not deployable |
| 10 | `kda_zero_state_output_elision` | 0.3165 ms | -3.98% | output-only, not deployable |
| 11 | `gla_zero_state_output_elision` | 0.3061 ms | -3.31% | output-only, not deployable |
| 12 | `gla_value_tile_grouping` | 0.2854 ms | -6.75% | remeasure under current objective |

## Integrity evidence

- All 12 round commits' non-bookkeeping paths exactly equal their manifest
  `files_touched` lists.
- `git diff 14a939d..eecce83 -- akt/benchmark python/sgl_jax/test` is empty.
- GLA `_scan_segment`, `_scan_varlen`, and `fused_recurrent_simple_gla` ASTs are
  unchanged from the pre-campaign baseline. KDA `naive.py` is byte-for-byte unchanged.
- Added production/runner lines contain no benchmark IDs, input seeds, timers, sleeps,
  tolerance changes, platform branches, or reference substitutions.
- Every round selected its newly added non-default choice in at least one best config
  under the old objective; this does not make the choice deployable or stateful-safe.
- R12's official full gate reported `all_correct=true`, 6 runnable and 9 deferred
  cases, no truncation, and 5/5 configured native-test invocations passing. Every
  runnable case had `n_correct == n_valid`; GLA checked 56/56 and 72/72 choices, KDA
  checked 132/132 and 94/94 choices, and GMM checked 18/18 and 27/27 choices.

## Independent correctness checks

- Re-ran the final modified GLA/KDA spaces with seed 7: 354 valid configs checked,
  zero failures (128 GLA and 226 KDA).
- Checked packed multi-sequence GLA and KDA, nonzero initial states, returned final
  states, compact alignment, KDA compute/state grouping, scalar solve, and 64/128 state
  alignment. Maximum FP32 error was below `1.8e-4` and within the frozen tolerances.
- Checked BF16 winning/elision and grouped paths. GLA max error was `3.97e-4`; KDA
  output max error was `3.05e-5`.
- Checked the serving-style KDA `use_gate_in_kernel=True` path with packed sequences,
  BF16 inputs, nonzero state, and custom grouping. Output/state max errors were
  `3.05e-5` / `1.79e-4`.
- Checked R12 grouping with `V=256`, per-token state, and group sizes 1/2/4; outputs
  matched the recurrent reference and final states.
- Confirmed both state-elision implementations reject unsupported multi-sequence use.
- The repository Simple GLA native test passed 12/12 independently; R12's official
  gate repeated that check successfully.

## Hardening applied after R12

- Frozen case inventory and exact runner wiring for canonical inputs, references,
  tolerances, output projections, and native tests.
- Frozen native-test tree, local frozen GLA/KDA recurrences, and AST fingerprints for
  reference functions that share editable TPU kernel files.
- Exact manifest-to-worktree scope validation and protection of campaign state/history
  during oracle execution.
- Exact bookkeeping allowlist, frozen board source, and rollback of partial oracle edits
  on rate-limit and invalid-manifest paths.
- Fail-closed behavior for native test errors/timeouts and for any incorrect config
  advertised as valid by a `DesignSpace`.
- Frozen execution-regime declarations and a stable runnable-case set, preventing a
  candidate from improving the geomean by making a broken local case appear
  `tpu-deferred`.
- Restored the pre-existing positional parameter order of `chunk_kda_fwd` and
  `kda_fwd_intra`; adding `intra_block_size` had silently shifted public optional args.
- Added contract/API regression tests; 45 focused AKT/API tests pass. The existing
  Lightning/KDA backend suite adds 22 passes and 11 expected TPU-only skips.
- Replaced the recurrent objective with nonzero incoming state plus observed final state,
  and tagged objective scope so legacy measurements cannot advance the new incumbent.
- Added a stable shape-aware `KernelControlPolicy`, production backend forwarding, and a
  frozen exposure gate. Runner-only experiments no longer determine the deployment
  optimum or close a flexibility gap.
- Re-evaluated the full stateful deployable space: all 6 runnable cases passed every
  advertised configuration (GLA 48/48 and 64/64; KDA 124/124 and 86/86; GMM 18/18 and
  27/27), all 5 native-test invocations passed, and no search was truncated.

## Residual limitations

1. Nine full-suite cases are TPU-deferred. The R1-R12 edits touch GLA/KDA, which are
   runnable and checked here, but Mosaic TPU compilation, VMEM feasibility, and TPU
   latency remain unverified. R5's 2048 tile and R12's grouped blocks especially need a
   real TPU gate before deployment.
2. Acceptance uses three timing samples per config and compares against a prior-round
   measurement. Near-threshold R6, R7, R10, and R11 gains should be remeasured with more
   repetitions or a paired incumbent/candidate protocol before publication.
3. At the R12 audit endpoint, knobs existed only in kernels and AKT runners. The later
   `KernelControlPolicy` work exposes the deployable KDA/GLA controls through serving,
   but the legacy R1-R12 deltas remain invalid for the new objective and require a clean
   `loop.py rebaseline` before further rounds.
4. KDA has no runnable end-to-end native backend test on this host. Its direct kernel,
   packed-state, BF16, and in-kernel-gate paths passed, but an attached-TPU backend test
   remains required.
