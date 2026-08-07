# JAX/Pallas boundary APIs — per-API docs

This directory documents every JAX/Pallas boundary API this tree adds — reusable
primitives that patch a limitation in the JAX API surface itself, as evidenced by the
AKT campaign and recorded in `akt/core/analysis/jax_api_limitations.md` (each entry
there is emitted by the flexgraph extractor as a `jax-api-gap` finding and rendered on
the AKT board's "JAX/Pallas boundary — limitations & patches" panel).

**Every future added JAX/Pallas API must ship a doc here, strictly organized in this
exact four-section order:**

1. **Context** — where the pattern arises in the serving kernels (concrete callsites),
   with the campaign/limiter numbers that motivated the patch.
2. **Limitation** — the JAX/Pallas API boundary gap itself, citing the
   `akt/core/analysis/jax_api_limitations.md` evidence.
3. **Patch** — the API: signature, mechanism (Pallas path and any fallback), and the
   correctness guarantee with its test path.
4. **Solution** — how kernels/rounds should adopt it, whether/how it is elevatable by
   the AKT loop, and the expected effect on the measured limiter.

## Current entries

| API | Status | Doc |
| --- | --- | --- |
| `single_move_permute` — single-move row permutation / de-alignment (no accumulate scatter, no take double-buffer) | patched-api-available | [single_move_permute.md](single_move_permute.md) |
