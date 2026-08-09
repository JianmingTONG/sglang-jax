"""Campaign-diagram bottleneck viewer for AKT workloads.

Renders, for every workload (the three frozen serving traces AND the formal
model-spec library: Qwen3, Qwen3-MoE, Kimi-Linear, testbench siblings), a
self-contained HTML page with a campaign diagram in the canonical
campaign_diagram_tools grammar:

  L0  one rectangle per kernel callsite; x = time (measured where possible),
      solid step line at the call's compute utilization, fixed-height memory
      pipe riding on the line, colored fill fraction of the pipe = bandwidth
      utilization. All utilizations on ONE global scale per diagram (busiest
      measured call = 1.0), so a parent equals the duration-weighted mean of
      its children. Unmeasured calls (tpu-deferred kernels, un-timed new
      shapes) render hatched with no invented utilization.
  L1  compiled stages of the dominant measured callsite; widths estimated
      proportional to op mass (labeled as estimates), same global scale.
  L2  engines of the dominant stage in MULTI-RESOURCE mode: fixed capacity
      bands (MXU/VPU/DMA, grey + black roof), vertical fill fraction of a
      band = that engine's utilization; x spans the stage's single interval.

Usage (from the sglang-jax repo root):
  PYTHONPATH=python:. .venv/bin/python \
      .claude/skills/campaign_diagram_bottleneck_viewer/scripts/draw_campaign_diagram.py \
      [--workload all|<id>] [--out-dir akt/board/campaign_diagrams] \
      [--live-timing/--no-live-timing] [--iters 5]

Latency sources, in order: the freshest gate eval's case_search medians; live
time_config timing for cases runnable on this host (when --live-timing, the
default); otherwise the call is drawn as unmeasured. Every segment's tooltip
names its source.
"""

from __future__ import annotations

import os

os.environ.setdefault("PALLAS_INTERPRET", "1")   # before any kernel import

import argparse
import html
import json
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
for _p in (str(REPO), str(REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from akt.core.analysis import campaign as cd            # noqa: E402  (helpers)
from akt.benchmark.model_workloads import MODEL_WORKLOADS  # noqa: E402

X0, W, GAP = 96, 648, 2
PIPE = 0.5            # bw pipe height in y-units (campaign_diagram_tools default)
YMAX = 1.52           # y-axis headroom (tool default)
MIN_SEG = 14.0        # px floor so tiny/unmeasured segments stay hoverable

CSS = """
:root { --bg:#F3F5F7; --panel:#fff; --ink:#1C242E; --muted:#5A6879; --line:#D7DEE5;
  --accent:#D97014; --accent-ink:#9A4E0C; --chip:#E8EDF1;
  --kda:#2a78d6; --gmm:#eb6834; --gla:#1baf7a; --neut:#8a97a5; }
@media (prefers-color-scheme: dark) { :root { --bg:#12181F; --panel:#1A222C; --ink:#E6EBF0;
  --muted:#93A1B0; --line:#2B3642; --accent:#F08A2C; --accent-ink:#F0A55E; --chip:#232E3A;
  --kda:#3987e5; --gmm:#d95926; --gla:#199e70; --neut:#7c8b9b; } }
body { background:var(--bg); color:var(--ink); font-family:ui-monospace,Menlo,Consolas,monospace;
  margin:2rem auto; max-width:900px; padding:0 1rem; }
h1 { font-size:1.15rem; } a { color:var(--accent-ink); }
figure { background:var(--panel); border:1px solid var(--line); padding:1rem .8rem; margin:1rem 0; overflow-x:auto; }
figcaption { font-size:.75rem; color:var(--muted); padding-top:.5rem; }
svg { min-width:720px; display:block; font-family:inherit; }
.ct { font-size:10.5px; letter-spacing:.06em; fill:var(--muted); }
.cax { font-size:9.5px; fill:var(--muted); } .cn { font-size:10px; fill:var(--ink); }
.cnm { font-size:9px; fill:var(--muted); } .chdr { font-size:11px; fill:var(--ink); font-weight:600; }
.cbase { stroke:var(--line); } .cgrid { stroke:var(--line); stroke-width:.5; stroke-dasharray:2 4; }
.cpipe { fill:var(--chip); stroke:var(--line); stroke-width:.6; }
.croof { stroke:var(--ink); stroke-width:1.4; } .cstep { stroke-width:2.4; }
.cwedge { fill:var(--accent); opacity:.08; }
.chot { fill:none; stroke:var(--accent); stroke-width:1.6; }
.chotlab { font-size:9.5px; letter-spacing:.1em; fill:var(--accent-ink); }
.climit { font-size:10.5px; fill:var(--accent-ink); }
.unm { fill:var(--chip); stroke:var(--muted); stroke-dasharray:3 3; }
.f-kda .cstep{stroke:var(--kda)} .f-kda .cfill{fill:var(--kda)}
.f-gmm .cstep{stroke:var(--gmm)} .f-gmm .cfill{fill:var(--gmm)}
.f-gla .cstep{stroke:var(--gla)} .f-gla .cfill{fill:var(--gla)}
.f-oth .cstep{stroke:var(--neut)} .f-oth .cfill{fill:var(--neut)}
"""

FAMS = {"kda": "f-kda", "gla": "f-gla", "gmm": "f-gmm", "gmm_v2": "f-gmm"}


def esc(text):
    return html.escape(str(text), quote=True)


def fam_class(case_id):
    return FAMS.get(case_id.split(":")[0], "f-oth")


def collect_workloads():
    """{workload_id: (description, [ModelCall...])} for frozen + spec library."""
    out = {}
    for workload in MODEL_WORKLOADS:
        out[workload.model_id] = (workload.description, list(workload.calls))
    try:
        from akt.benchmark.model_specs import library
        from akt.benchmark.workload_spec import lower
        for wid, spec in library().items():
            if spec.family == "legacy":
                continue                      # identical to the frozen entry
            lowered = lower(spec)
            out[wid] = (spec.description + f" [{spec.family}; {spec.provenance[:60]}]",
                        list(lowered.workload.calls))
    except Exception as error:  # noqa: BLE001 — frozen traces still render
        print(f"[viewer] spec library unavailable: {error}")
    return out


def case_objects(calls, live_timing, iters):
    """case_id -> (KernelCase|None, latency_s|None, source)."""
    from akt.benchmark.suites import load_cases
    frozen = {case.case_id: case for case in load_cases("full")}
    table = cd._eval_latency_table()
    new_cases = {}
    try:
        from akt.benchmark.model_specs import library
        from akt.benchmark.workload_spec import materialize_cases
        for spec in library().values():
            if spec.family != "legacy":
                new_cases.update(materialize_cases(spec))
    except Exception:  # noqa: BLE001
        pass

    out = {}
    for call in calls:
        cid = call.case_id
        if cid in out:
            continue
        case = frozen.get(cid) or new_cases.get(cid)
        row = table.get(cid)
        if row:
            out[cid] = (case, row["latency_s"], row["source"])
            continue
        latency = None
        source = "unmeasured"
        if case is not None and live_timing:
            try:
                from akt.benchmark.runners.base import time_config
                timing = time_config(case, case.space.default_config(), iters=iters)
                latency, source = float(timing["median_s"]), f"live time_config iters={iters} (default config)"
            except Exception as error:  # noqa: BLE001 — tpu-deferred etc.
                source = f"unmeasured ({type(error).__name__})"
        out[cid] = (case, latency, source)
    return out


def utilizations(calls, cases):
    """Global-scale cmp/bw utils per measured case (busiest = 1.0)."""
    try:
        operation_graph, classify, _note = cd._graph_tools()
    except cd.CampaignUnavailable as error:
        return {}, f"operation-graph tools unavailable — utilizations omitted ({error})"
    rates = {}
    for call in calls:
        case, latency, _src = cases[call.case_id]
        if case is None or latency is None or call.case_id in rates:
            continue
        try:
            graph = cd._trace_case(case, operation_graph)
            classes = list(classify(list(graph["labels"])))
            flop, byte = cd._masses(graph["labels"], graph["weights"], classes)
            rates[call.case_id] = (flop / latency, byte / latency, graph, classes)
        except Exception as error:  # noqa: BLE001
            print(f"[viewer] trace failed for {call.case_id}: {error}")
    if not rates:
        return {}, "no measured+traceable call"
    cmax = max(r[0] for r in rates.values()) or 1.0
    bmax = max(r[1] for r in rates.values()) or 1.0
    return {cid: (c / cmax, b / bmax, g, cl) for cid, (c, b, g, cl) in rates.items()}, None


def draw_workload(wid, description, calls, live_timing, iters):
    cases = case_objects(calls, live_timing, iters)
    utils, util_note = utilizations(calls, cases)
    measured = [c for c in calls if cases[c.case_id][1] is not None]
    unmeasured = [c for c in calls if cases[c.case_id][1] is None]
    total_s = sum(cases[c.case_id][1] for c in measured) or 1.0

    svg = []
    A = svg.append
    y0, H0 = 224, 150

    def yp(u):
        return y0 - u * H0 / YMAX

    A(f'<svg viewBox="0 0 760 {y0 + 120}" role="img">')
    A(f'<text x="{X0}" y="18" class="chdr">WORKLOAD: {esc(wid)} — synthetic serving trace, not a checkpoint</text>')
    A(f'<text x="{X0}" y="32" class="cnm">{esc(description[:120])}</text>')
    A(f'<text x="{X0}" y="{y0 - H0 - 24}" class="ct">L0 — one kernel call per rectangle · x = time (&#931; measured {total_s*1e3:.2f} ms) · step line = compute util · pipe fill = bw util (global scale)</text>')
    for u, lab in ((0.0, "0"), (0.5, ".5"), (1.0, "1.0")):
        yy = yp(u)
        A(f'<line x1="{X0-4}" y1="{yy:.1f}" x2="{X0+W}" y2="{yy:.1f}" class="{"cbase" if u == 0 else "cgrid"}"/>')
        A(f'<text x="{X0-8}" y="{yy+3:.1f}" text-anchor="end" class="cax">{lab}</text>')
    A(f'<text x="{X0-42}" y="{yp(0.5):.1f}" class="cax" transform="rotate(-90 {X0-42} {yp(0.5):.1f})" text-anchor="middle">compute util</text>')

    unm_px = MIN_SEG * len(unmeasured)
    meas_w = W - GAP * (len(calls) - 1) - unm_px
    x = X0
    dominant = max(measured, key=lambda c: cases[c.case_id][1], default=None)
    for call in calls:
        case, latency, source = cases[call.case_id]
        if latency is None:
            title = f"{call.call_id} — {call.case_id} — UNMEASURED on this host ({source}); no utilization invented"
            A(f'<g><title>{esc(title)}</title><rect x="{x:.1f}" y="{yp(PIPE):.1f}" width="{MIN_SEG}" height="{yp(0)-yp(PIPE):.1f}" class="unm"/></g>')
            x += MIN_SEG + GAP
            continue
        width = max(latency / total_s * meas_w, 3.0)
        util = utils.get(call.case_id)
        cls = fam_class(call.case_id)
        share = latency / total_s
        if util:
            cu, bu = util[0], util[1]
            y_line, y_top = yp(cu), yp(cu + PIPE)
            pipe_px = y_line - y_top
            fill_px = pipe_px * min(bu, 1.0)
            title = (f"{call.call_id} — {call.case_id} — {latency*1e3:.3f} ms ({share*100:.0f}%) · "
                     f"compute line {cu*100:.1f}% · pipe filled {bu*100:.1f}% (global) · {source}")
            A(f'<g class="cs {cls}"><title>{esc(title)}</title>')
            A(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{width:.1f}" height="{pipe_px:.1f}" class="cpipe"/>')
            A(f'<rect x="{x:.1f}" y="{y_line-fill_px:.1f}" width="{width:.1f}" height="{fill_px:.1f}" class="cfill"/>')
            A(f'<line x1="{x:.1f}" y1="{y_line:.1f}" x2="{x+width:.1f}" y2="{y_line:.1f}" class="cstep"/>')
            if call is dominant:
                A(f'<rect x="{x:.1f}" y="{yp(YMAX):.1f}" width="{width:.1f}" height="{yp(0)-yp(YMAX)+4:.1f}" class="chot"/>')
            A("</g>")
            if share > 0.08:
                A(f'<text x="{x+width/2:.1f}" y="{y0+14}" text-anchor="middle" class="cn">{esc(call.call_id[:22])}</text>')
                A(f'<text x="{x+width/2:.1f}" y="{y0+27}" text-anchor="middle" class="cnm">{share*100:.0f}% · cmp {cu*100:.0f}% · bw {bu*100:.0f}%</text>')
        else:
            title = f"{call.call_id} — {call.case_id} — {latency*1e3:.3f} ms ({share*100:.0f}%) · trace unavailable · {source}"
            A(f'<g class="cs {cls}"><title>{esc(title)}</title><rect x="{x:.1f}" y="{yp(0.02):.1f}" width="{width:.1f}" height="{yp(0)-yp(0.02):.1f}" class="cfill"/></g>')
        x += width + GAP
    if dominant is not None:
        A(f'<text x="{X0}" y="{y0+48}" class="climit">DOMINANT MEASURED CALL: {esc(dominant.call_id)} '
          f'({cases[dominant.case_id][1]*1e3:.3f} ms) — see the L1/L2 drill-down for the fast-suite limiter '
          f'in akt/board/campaign.json (regenerate with akt/core/analysis/campaign.py)</text>')
    if util_note:
        A(f'<text x="{X0}" y="{y0+66}" class="cnm">note: {esc(util_note)}</text>')
    if unmeasured:
        A(f'<text x="{X0}" y="{y0+84}" class="cnm">hatched = {len(unmeasured)} unmeasured call(s) on this host (tpu-deferred kernels / un-timed shapes) — widths are placeholders, no utilization drawn</text>')
    A("</svg>")

    caption = ("Grammar (campaign_diagram_tools): x = time; solid step line at the call&#8217;s compute "
               "utilization; the fixed-height pipe (0.5 y-units = 100% bw) rides on the line; its colored "
               "fill fraction is the bw utilization. One global scale per diagram (busiest measured call = 1.0). "
               "Utilizations are op-graph/latency proxies, not hardware counters.")
    return (f"<figure>{''.join(svg)}<figcaption>{caption}</figcaption></figure>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="all")
    parser.add_argument("--out-dir", default="akt/board/campaign_diagrams")
    parser.add_argument("--live-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    workloads = collect_workloads()
    wanted = list(workloads) if args.workload == "all" else [args.workload]
    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for wid in wanted:
        description, calls = workloads[wid]
        print(f"[viewer] {wid}: {len(calls)} calls")
        figure = draw_workload(wid, description, calls, args.live_timing, args.iters)
        page = (f"<!-- generated by campaign_diagram_bottleneck_viewer -->\n"
                f"<title>campaign diagram — {esc(wid)}</title>\n<style>{CSS}</style>\n"
                f"<h1>{esc(wid)}</h1>\n<p><a href='index.html'>&#8592; all workloads</a></p>\n{figure}\n"
                f"<p>generated {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        (out_dir / f"{wid}.html").write_text(page)
        index_rows.append(f"<li><a href='{esc(wid)}.html'>{esc(wid)}</a> — {esc(description[:100])}</li>")
    index = (f"<title>AKT campaign diagrams</title>\n<style>{CSS}</style>\n"
             f"<h1>Campaign diagrams — all workloads</h1>\n<ul>{''.join(index_rows)}</ul>\n"
             f"<p>generated {time.strftime('%Y-%m-%d %H:%M:%S')} · grammar: campaign_diagram_tools; "
             f"see each page&#8217;s caption</p>")
    (out_dir / "index.html").write_text(index)
    print(f"[viewer] wrote {len(wanted)} diagram page(s) + index to {out_dir}")


if __name__ == "__main__":
    main()
