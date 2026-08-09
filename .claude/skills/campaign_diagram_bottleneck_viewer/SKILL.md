---
name: campaign_diagram_bottleneck_viewer
description: Draw campaign diagrams (campaign_diagram_tools grammar) for all AKT workloads — the three frozen serving traces plus the formal model-spec library (Qwen3, Qwen3-MoE, Kimi-Linear). Use when the user asks to visualize workload bottlenecks, latency profiles, compute/bandwidth utilization, or campaign diagrams for any AKT workload; also after a KEEP round or rebaseline to refresh the per-workload bottleneck views.
---

# Campaign-diagram bottleneck viewer

Renders one self-contained HTML page per AKT workload plus an index, each with a
campaign diagram in the canonical `campaign_diagram_tools` grammar
(https://github.com/jsemer/campaign_diagram_tools, local checkout at
`/home/ubuntu/work/campaign_diagram_tools`).

## Run

From the sglang-jax repo root:

```bash
PYTHONPATH=python:. .venv/bin/python \
    .claude/skills/campaign_diagram_bottleneck_viewer/scripts/draw_campaign_diagram.py \
    --workload all --out-dir akt/board/campaign_diagrams
```

Options:
- `--workload <id>` — one workload (`tiny-linear-serving`, `qwen3-8b`,
  `qwen3-30b-a3b`, `kimi-linear-sgl-default`, `qwen3-tiny-testbench`, ...);
  `all` (default) renders every frozen trace + every model-spec instance.
- `--no-live-timing` — only use latencies already in the freshest gate eval;
  do not time un-measured cases ad hoc. Default is live timing with
  `--iters 5` for cases runnable on this host.
- Output: `<out-dir>/<workload>.html` + `<out-dir>/index.html`. Serve with the
  board (`python -m http.server` in `akt/board/`) or open directly.

## How to read the diagram (the grammar)

- **x = time.** Each kernel call is one rectangle spanning its share of the
  workload's measured total.
- **The solid step line sits at the call's COMPUTE utilization** (y-axis,
  0..1; headroom to 1.52 like the tool's default).
- **The fixed-height grey pipe riding on the step line is 100% of memory
  bandwidth** (0.5 y-units, the tool's `bw_util_scaling` default). The colored
  **fill fraction of the pipe = the call's BANDWIDTH utilization**.
- One **global scale per diagram**: the busiest measured call defines 1.0 for
  both rates, so a parent's utilization equals the duration-weighted mean of
  its children — never mix per-level normalizations.
- **Hatched segments are unmeasured** on the current host (tpu-deferred
  kernels, untimed new shapes): placeholder width, no utilization drawn.
- The amber outline marks the workload's dominant measured call. For the
  fast-suite limiter drill-down (L1 stages, L2 multi-resource engine bands),
  regenerate `akt/board/campaign.json` with
  `PYTHONPATH=python:. .venv/bin/python akt/core/analysis/campaign.py --out akt/board/campaign_diagrams/../campaign.json`
  — the board's "Bottleneck campaign diagrams" panel renders it.

## Caveats (do not silently drop these when reporting)

- Utilizations are **proxies**: op-graph flop/byte masses over measured wall
  time, normalized per diagram — this host has no hardware counters.
- Latency sources are per-segment and named in every tooltip: gate-eval
  medians > live `time_config` > unmeasured.
- Model-spec workloads (qwen3-8b, kimi-linear-...) are **synthetic shape
  envelopes** derived from architecture configs — not checkpoints — and their
  new cases are NOT part of the frozen measured objective until a maintainer
  `rebaseline`. Coverage gaps (e.g. full-attention prefill has no kernel
  family) are listed by `python akt/benchmark/model_specs.py --lower <id>`.
