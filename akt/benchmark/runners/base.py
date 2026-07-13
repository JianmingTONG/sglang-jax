"""The kernel-runner CONTRACT — the AI-domain analog of vn_go's layout design space.

Every sglang-jax kernel we tune is wrapped as one or more `KernelCase`s. A case
carries: how to build inputs, how to RUN the Pallas kernel under a given config,
a pure-JAX REFERENCE for correctness, and a `DesignSpace` (the tunable knobs +
their candidate value sets). The AKT loop's "search" is `search_best`: enumerate
the design space, keep the numerically-correct configs, time them on THIS machine,
and return the fastest. A *capability* elevates a NEW `Knob` into a kernel's
`DesignSpace` (mirroring how an FHE capability registers a new layout axis).

Feasibility on this box: these are TPU-Pallas/Mosaic kernels. Some run directly on
GPU, some only under Pallas `interpret=True` (CPU), some are TPU-only. Each case is
PROBED once (`probe_regime`) and profiled in whatever regime runs here; TPU-only
cases are recorded as `tpu-deferred` (wired, but their definitive latency waits for
a TPU). Timing here is `perf_counter` with warmup + `block_until_ready` (honest
this-machine latency); the TPU path swaps in `multiple_iteration_timeit_from_trace`.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import numpy as np


# --------------------------------------------------------------- design space
@dataclass
class Knob:
    """One tunable axis. `elevated_by` names the capability that added it
    (None = part of the kernel's base/shipped design space)."""
    name: str
    values: list
    default: Any
    elevated_by: str | None = None


@dataclass
class DesignSpace:
    knobs: list[Knob]
    valid: Callable[[dict], bool] = field(default=lambda cfg: True)

    def default_config(self) -> dict:
        return {k.name: k.default for k in self.knobs}

    def size(self) -> int:
        n = 1
        for k in self.knobs:
            n *= max(1, len(k.values))
        return n

    def enumerate(self, cap: int = 4096):
        names = [k.name for k in self.knobs]
        vals = [k.values for k in self.knobs]
        seen = 0
        for combo in itertools.product(*vals):
            cfg = dict(zip(names, combo))
            if not self.valid(cfg):
                continue
            yield cfg
            seen += 1
            if seen >= cap:
                return

    def base_space(self) -> "DesignSpace":
        """The design space BEFORE any capability elevation (only base knobs).
        Used at init to report the autotuning-only optimum separately from the
        capability-elevation trajectory."""
        base = [k for k in self.knobs if k.elevated_by is None]
        return DesignSpace(knobs=base, valid=self.valid)


# --------------------------------------------------------------- kernel case
@dataclass
class KernelCase:
    kernel_id: str
    shape_id: str
    make_inputs: Callable[[], dict]          # -> {arg_name: value} (jax arrays + statics)
    run: Callable[[dict, dict], Any]         # (inputs, config) -> output pytree
    reference: Callable[[dict], Any]         # (inputs) -> ground-truth output pytree
    space: DesignSpace
    atol: float = 1e-2
    rtol: float = 2e-2
    check_out: Callable[[Any], Any] = field(default=lambda o: o)  # tensors to compare
    regime_pref: tuple[str, ...] = ("gpu", "cpu-interpret")       # try in this order
    note: str = ""

    @property
    def case_id(self) -> str:
        return f"{self.kernel_id}:{self.shape_id}"


# --------------------------------------------------------------- correctness
def _leaves(o):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(o)]


def check_correct(case: KernelCase, cfg: dict) -> tuple[bool, str]:
    inp = case.make_inputs()
    try:
        out = case.check_out(case.run(inp, cfg))
        ref = case.check_out(case.reference(inp))
    except Exception as e:  # noqa: BLE001
        return False, f"exc {type(e).__name__}: {str(e)[:120]}"
    ol, rl = _leaves(out), _leaves(ref)
    if len(ol) != len(rl):
        return False, f"nleaves {len(ol)}!={len(rl)}"
    worst = 0.0
    for a, b in zip(ol, rl):
        if a.shape != b.shape:
            return False, f"shape {a.shape}!={b.shape}"
        a = a.astype(np.float32); b = b.astype(np.float32)
        if not np.allclose(a, b, atol=case.atol, rtol=case.rtol, equal_nan=False):
            md = float(np.nanmax(np.abs(a - b)))
            return False, f"maxabs {md:.2e} > atol {case.atol:.1e}"
        worst = max(worst, float(np.nanmax(np.abs(a - b))) if a.size else 0.0)
    return True, f"maxabs {worst:.2e}"


# --------------------------------------------------------------- timing
def time_config(case: KernelCase, cfg: dict, iters: int = 30, warmup: int = 3) -> dict:
    """Median/mean wall latency of one config on THIS machine. Compiles on the
    warmup calls; blocks on device each iter. Honest for GPU / CPU-interpret."""
    inp = case.make_inputs()
    for _ in range(warmup):
        jax.block_until_ready(case.run(inp, cfg))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(case.run(inp, cfg))
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_s": ts[len(ts) // 2], "mean_s": sum(ts) / len(ts),
            "min_s": ts[0], "iters": iters}


# --------------------------------------------------------------- search
def search_best(case: KernelCase, space: DesignSpace | None = None,
                iters: int = 20, cap: int = 256) -> dict:
    """Autotune: over `space`, keep correct configs, time them, return the fastest.
    This is the AI-domain analog of vn_go's DP over the layout space. Returns
    {best_config, best_median_s, n_valid, n_correct, n_evaluated, default_median_s}."""
    space = space or case.space
    default = case.space.default_config()
    best_cfg, best_t = None, math.inf
    n_valid = n_correct = n_eval = 0
    default_t = None
    for cfg in space.enumerate(cap=cap):
        n_valid += 1
        ok, _why = check_correct(case, cfg)
        if not ok:
            continue
        n_correct += 1
        n_eval += 1
        t = time_config(case, cfg, iters=iters)["median_s"]
        if cfg == default:
            default_t = t
        if t < best_t:
            best_t, best_cfg = t, cfg
    if default_t is None:  # ensure the incumbent/default is always measured
        ok, _ = check_correct(case, default)
        if ok:
            default_t = time_config(case, default, iters=iters)["median_s"]
    return {"best_config": best_cfg, "best_median_s": best_t if best_cfg else None,
            "default_config": default, "default_median_s": default_t,
            "n_valid": n_valid, "n_correct": n_correct, "n_evaluated": n_eval,
            "space_size": space.size()}


# --------------------------------------------------------------- regime probe
def probe_regime(case: KernelCase) -> tuple[str, str]:
    """Which execution regime runs this case on THIS machine. Returns
    (regime, detail). regime in {gpu, cpu-interpret, tpu-deferred}. The caller is
    responsible for having set JAX_PLATFORMS / PALLAS_INTERPRET before import for
    the interpret regime; here we just test whether the default backend runs it."""
    cfg = case.space.default_config()
    ok, why = check_correct(case, cfg)
    if ok:
        return jax.default_backend(), why
    return "failed", why
