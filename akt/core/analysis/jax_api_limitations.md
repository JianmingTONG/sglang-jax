# JAX API limitations evidenced by the AKT campaign

The flexibility graph normally adds choices ON TOP of JAX/Pallas. This file records the
limitations the campaign evidenced IN the JAX API boundary itself, so the graph can report
them instead of silently hiding them (`flexgraph_extract.py` emits each entry below as an
analysis-only `jax-api-gap` finding).

## 1. No single-move row-permutation / de-alignment primitive (PATCHED)

**The limitation.** JAX/jnp offers no primitive that moves each row of an array to a
caller-chosen destination row exactly once. The two expressible formulations both
over-pay:

- `x.at[idx].add(...)` / `.at[idx].set(...)` lowers to XLA scatter — a read-modify-write
  over the destination buffer with accumulate semantics. It initializes and touches every
  destination row (zeros + gather + add), and its int32 index-manipulation stages are
  DMA-bound.
- `jnp.take` / gather materializes a second full buffer in the new order; both operands
  are live simultaneously and the index space must be inverted first.

Kernel authors therefore either eat the RMW scatter, or hand-fuse the permutation into
each kernel — per kernel, not reusable.

**Evidence (incumbent callers eating the scatter).**

- `python/sgl_jax/srt/kernels/simple_gla/simple_gla.py:1312` — `_unalign_output`:
  `o_out = o_out.at[0, gather_idx].add(jnp.where(is_valid[:, None, None], o_aligned[0], 0))`
  restores GLA token order from the aligned staging layout through an accumulate scatter.
- `python/sgl_jax/srt/kernels/kda/kda.py:566` and `:1082` —
  `out = out.at[flat_pos].add(...)` scatters KDA per-chunk results back to packed token
  positions (invalid rows aliased onto `T_alloc - 1`, then sliced off).

**Evidence (measured cost — the campaign limiter).**

- R1 manifest `akt/core/evolve/capabilities/gla_output_subchunk_schedule.json:5`
  (hypothesis): the incumbent output stage "returns the aligned staging layout, leaving
  the caller to restore token order with a read-modify-write scatter over the ALIGNED
  index space, which touches every output row through an accumulate".
- Same manifest, `:11` (estimate): a stage decomposition under the frozen deterministic-XLA
  regime put the aligned-space scatter at 12.044 ms of gla-long's 12.462 ms (96.6%) and
  3.038 ms of gla-short's 3.623 ms (83.9%); at that snapshot it was the campaign's named
  limiter "#26 jit|float32r4, cmp=2%/bw=98%, DMA memory-bound — cut bytes moved / keep
  data resident".
- Current campaign limiter record `akt/board/campaign.json` (L1/L2): the L0-dominant
  callsite `tiny-linear-serving/kda-long` is limited by stage `#47 jit|int32r2` — an int32
  index-manipulation stage, 21.9% of the callsite, `cmp_util ≈ 0.0002`, `bw_util ≈ 0.9998`,
  L2 engine = DMA, verdict memory-bound; sibling int32 stages `#42`/`#72 jit|int32r1`
  carry ~13.1 MB byte mass each at `bw_util ≈ 1.0`. These are the index construction +
  scatter plumbing of exactly the `.at[flat_pos].add` formulation above.

**Evidence (the workaround is not reusable).**

- R1 (`gla_output_subchunk_schedule`, KEEP, +58.67%) removed the scatter for GLA only by
  hand-fusing the permutation into the kernel's own launcher:
  `python/sgl_jax/srt/kernels/simple_gla/simple_gla.py:1150-1156` applies
  `o = o[:, emit_order]` inside `_chunk_fwd_o_pl`. Correct, but expressed as private
  schedule surgery on one kernel — KDA's two scatter sites (above) cannot call it.

**The patch.** `python/sgl_jax/srt/kernels/common/permute.py` now provides
`single_move_permute(rows, dest_index, out_len)` — a reusable JAX-level API for direct
single-move row permutation WITHOUT accumulate semantics: a Pallas kernel (TPU and
interpret-safe) grids over source rows and DMA-copies each row straight to its
destination block via a scalar-prefetched dynamic output index (each destination row is
written exactly once; dropped rows are routed to a trash row that is sliced off), plus a
pure-jnp fallback (`take` on the inverse permutation, computed via `argsort` when only
forward scatter indices are given) for hosts where Pallas cannot lower. Bit-exact
equality with the scatter formulation on GLA-unalign packed/varlen patterns is proven by
`python/sgl_jax/test/test_permute_common.py`. Existing kernels are deliberately NOT
rewired: the API is introduced as an existing handle for future loop rounds to elevate.

> Per-API doc (context → limitation → patch → solution):
> `docs/kernels/api/single_move_permute.md` — index of all boundary APIs:
> `docs/kernels/api/README.md`.

## 2. Secondary limitations surfaced by the testbench (documented, not patched)

Recorded per round in `akt/optimization_history/evals/*.json` (regime `tpu-deferred`,
e.g. `c1786023097_round_001_gla_output_subchunk_schedule.json`) — on this non-TPU host
Pallas lowers through the Triton backend, which cannot express what the TPU (Mosaic)
backend can:

- **Pallas-Triton: no scratch memory** — `NotImplementedError: scratch memory not
  implemented in the Triton backend` (`moe_v1:t32_e8_k2_h2048_i1024`).
- **Pallas-Triton: no dynamic grid bounds** — `NotImplementedError: dynamic grid bounds
  not supported in the Triton backend` (`kv_cache:h8_cache4096_new256`,
  `kv_cache:h16_cache8192_new512`).
- **Backend probes leak host assumptions** — `NotImplementedError: Unsupported
  tpu_version=-1.` and `ValueError: Unsupported TPU device kind: NVIDIA GeForce RTX 5090`
  from Mosaic tuned-table selectors probed off-TPU.
- **Interpret gaps** — the interpret proxy pays a Python-level grid loop per program, so
  it exaggerates the fixed per-program cost that Mosaic amortizes (noted in
  `akt/optimization_history/evolve_history.jsonl` round records); cases that need the
  Triton-missing features cannot even fall back to timing and stay `tpu-deferred`.

These are documented so the graph records them as boundary limitations; no patch is
attempted here.
