# Elevation registry — HierEvo TPU v6e-8 run, 2026-08-10 (sglang-jax track)

Canonical in-repo registration of the **3 controls kept on the `sglang-jax`
track** by the 8-hour HierEvo loop run on a TPU v6e-8 host (2 tracks · 244
campaigns · 2145 rounds; this track ran on chips 0–3 under jax 0.10.2). Each
JSON file is one kept control in ControlRegistry row form (control / family /
knob / default / deployable values) plus its landing provenance: the campaign
and round that kept it, the measured suite improvement, the limiter it
addressed, and — where `api_deduplication.minimize()` fired — the minimized
domain.

| kept control (HierEvo name) | default | deployable values |
|---|---|---|
| `sgl_rpa_v3.d_bkv_sz` | 2048 | 256, 512, 1024, 2048 |
| `sgl_rpa_v3.p_bkv_sz` | 1024 | 1024, 256 (minimized) |
| `sgl_gmm.tk` | 1408 | 128, 1408 |

Full run log (campaign tables, round ledgers, flexibility graphs, campaign
diagrams): the board artifact at
<https://claude.ai/code/artifact/7c92db26-7e50-46d0-8320-2f81a74969fc>.

## In-repo naming map

The `sgl_` prefix is HierEvo-side namespacing and does not appear in this
repository. The controls land in the programmer control plane
`python/sgl_jax/srt/configs/kernel_control.py` (`PROGRAMMER_CONTROL_REGISTRY`,
reachable through `--kernel-control-config`) as:

| HierEvo control | in-repo family.key | kernel argument |
|---|---|---|
| `sgl_rpa_v3.d_bkv_sz` | `rpa_v3.d_bkv_sz` | `ragged_paged_attention(..., d_bkv_sz=)` |
| `sgl_rpa_v3.p_bkv_sz` | `rpa_v3.p_bkv_sz` | `ragged_paged_attention(..., p_bkv_sz=)` |
| `sgl_gmm.tk` | `gmm.tk` | megablox backend `gmm(..., tiling=/v2_tile_info=)` k-tile |

The in-repo defaults are the explicit "unset" sentinel `None`, which preserves
the incumbent selection path (tuned table / heuristic / auto-tiler)
byte-identically; on the campaign shapes that path resolves to the kept
defaults above. Note the `rpa_v3.p_bkv_sz` liveness condition: the prefill
pallas stage only executes when the caller passes a non-None
`chunk_prefill_size`; production does not today, so prefill work runs in the
MIXED stage and the control is dormant until chunk prefill is enabled.

## Provenance

The run VM was **preempted before its local commits could be pushed** — the
only commit that escaped is the in-flight state snapshot on the sglang-jax
side. These registration files were therefore reconstructed post-hoc from the
run's board artifact, whose three independent records (campaign table, round
ledger, flexibility-graph land labels) are cross-consistent for every control.
One commit registers one control, so each control has a single citable
commit ID.
