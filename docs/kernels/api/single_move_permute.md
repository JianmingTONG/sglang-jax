# `single_move_permute` — single-move row permutation / de-alignment

Module: `python/sgl_jax/srt/kernels/common/permute.py` · status: **patched-api-available**
· gap: `common:single_move_permute:jax-api-gap` (see
`akt/core/analysis/jax_api_limitations.md` §1)

## Context

Linear-attention kernels stage their work in an **aligned** layout — sequences padded and
blocked to chunk boundaries so the Pallas grid is static — and must restore packed
**token order** at the output boundary. That aligned-staging → token-order-restoration
step is a pure row permutation (each aligned row carries exactly one original token
position), and it recurs across the family:

- `python/sgl_jax/srt/kernels/simple_gla/simple_gla.py:1312` — `_unalign_output`:
  `o_out = o_out.at[0, gather_idx].add(jnp.where(is_valid[:, None, None], o_aligned[0], 0))`.
- `python/sgl_jax/srt/kernels/kda/kda.py:566` and `:1082` —
  `out = out.at[flat_pos].add(...)` scatters KDA per-chunk results back to packed token
  positions (invalid rows aliased onto `T_alloc - 1`, then sliced off).

**Campaign limiter numbers.** This formulation has repeatedly been the AKT campaign's
measured limiter:

- R1 stage decomposition (manifest `akt/core/evolve/capabilities/gla_output_subchunk_schedule.json`):
  the aligned-space scatter accounted for **12.044 ms of gla-long's 12.462 ms (96.6%)**
  and **3.038 ms of gla-short's 3.623 ms (83.9%)** — at that snapshot the named limiter
  "#26 jit|float32r4, cmp=2%/bw=98%, DMA memory-bound".
- Current campaign record (`akt/board/campaign.json`, L1/L2): the L0-dominant callsite
  `tiny-linear-serving/kda-long` is limited by stage **#47 jit|int32r2** — an int32
  index-manipulation stage, **21.9% of the callsite**, `cmp_util ≈ 0.0002`,
  `bw_util ≈ 0.9998`, L2 engine = **DMA**, verdict **memory-bound**; sibling int32 stages
  #42/#72 carry **~13.1 MB** byte mass each at `bw_util ≈ 1.0`. These are the index
  construction and scatter plumbing of exactly the `.at[flat_pos].add` formulation above.

## Limitation

JAX/jnp provides **no primitive that moves each row of an array to a caller-chosen
destination row exactly once**. The two expressible formulations both over-pay
(evidence: `akt/core/analysis/jax_api_limitations.md` §1):

- **Accumulate scatter** — `x.at[idx].add(...)` / `.at[idx].set(...)` lowers to an XLA
  scatter: a read-modify-write over the destination buffer with accumulate semantics. It
  zero-initializes and touches every destination row (zeros + gather + add), and its
  int32 index-manipulation stages are DMA-bound — the limiter stages quoted above.
- **Take / gather double-buffer** — `jnp.take` materializes a second full buffer in the
  new order; both operands are live simultaneously, and the index space must be
  **inverted** first (callers hold forward scatter indices).

The pre-existing workaround is not reusable: R1 removed the scatter for GLA only by
hand-fusing the permutation into that kernel's own launcher
(`simple_gla.py:1150-1156`, `o = o[:, emit_order]` inside `_chunk_fwd_o_pl`) — private
schedule surgery on one kernel that KDA's two scatter sites cannot call.

## Patch

The reusable API, `python/sgl_jax/srt/kernels/common/permute.py`:

```python
single_move_permute(rows, dest_index, out_len=None, *, backend="auto", interpret=None)
```

- **Semantics** — `out[dest_index[i]] = rows[i]` for every in-range `dest_index[i]`;
  entries outside `[0, out_len)` (e.g. `-1` varlen padding) drop their row. Precondition
  (both paths): the in-range entries of `dest_index` are a bijection onto
  `0..out_len-1`, as in packed varlen de-alignment.
- **Pallas single-move mechanism** (`backend="pallas"`, TPU and interpret-safe) — the
  kernel grids over source rows with
  `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1)`, so the forward `dest_index`
  vector is **scalar-prefetched and read by the output BlockSpec index map**
  (`out_specs=pl.BlockSpec((1, width), lambda i, dest_ref: (dest_ref[i], 0))`). Each
  source row is DMA-copied straight into its destination block — one move per row, no
  accumulate, no zero-init of the destination. Dropped rows are routed to a trailing
  trash row (`out_len`) that is sliced off.
- **jnp fallback** (`backend="jnp"`, auto-selected off-TPU when not interpreting) —
  `jnp.take` on the inverse permutation, computed via `argsort` since callers hold only
  forward indices.
- **Bit-exactness guarantee** — under the bijection precondition the result equals the
  scatter formulation `zeros.at[dest].set(rows, mode="drop")`
  (`single_move_permute_reference`). Proven, including the GLA-unalign packed/varlen
  patterns, by `python/sgl_jax/test/test_permute_common.py`.

## Solution

- **Adoption by kernels/rounds** — replace an aligned→packed de-alignment scatter
  (`out.at[idx].add/set(rows)`) with `single_move_permute(rows, idx, out_len)`; the
  caller keeps its forward scatter indices (no inversion) and drops the
  zero-init/`jnp.where` masking that the accumulate formulation required. Target sites:
  simple_gla `_unalign_output` and kda's two `.at[flat_pos].add` sites.
- **Elevatable by the loop** — the API is deliberately **not** wired into existing
  kernels; it is introduced as an existing, discoverable handle for the AKT
  capability-elevation loop to adopt per kernel, so each rewiring is its own round with
  its own correctness + >2% gate.
- **Expected effect on the DMA-bound limiter** — each output row is moved once instead
  of read-modify-written: the scatter's zero-init + gather + add byte traffic and the
  int32 index-manipulation stages (the #47/#42/#72-class stages at `bw_util ≈ 1.0`)
  disappear, directly cutting the bytes moved through the DMA engine at the campaign's
  L1/L2 limiter. R1's hand-fused equivalent of the same transformation on GLA alone
  measured **+58.67%**, which bounds the shape of the win available at the remaining
  KDA sites.
