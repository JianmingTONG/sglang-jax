"""FROZEN suite registry — the benchmark-specific knowledge the AKT core consumes.

Porting AKT to another kernel set = replace the runner modules + this registry;
core/ (the loop) and board/ need no changes. Each runner module (in the EDITABLE
akt/core/runners/, so capabilities can register new tuning Knobs) exposes a
top-level `CASES: list[KernelCase]` (see akt/benchmark/runners/base.py — the FROZEN
contract + timer + correctness-check + search). Tolerances live on the cases;
correctness references are the kernels' own pure-JAX `ref_*`/naive impls.

Unlike the FHE port, there are NO hardcoded reference latencies: the incumbent is
MEASURED on this machine at `init` (the shipped/default config profiled over N
runs), per the campaign's baseline definition.

REGIME NOTE (this box: RTX 5090, no TPU). These are TPU-Pallas kernels. Pallas
does not lower to the GPU Triton backend, so only kernels that thread an
`interpret` flag run here (CPU/GPU interpret = functional proxy, NOT TPU-faithful
latency). The rest are `tpu-deferred`: fully wired (inputs, config->tiling map,
pure-JAX reference, design space) with their definitive latency awaiting a TPU.
This registry is part of the FROZEN harness — the loop must not edit it.
"""
from __future__ import annotations

import importlib

# All wired kernels (the "full" suite). Order is display order.
RUNNER_MODULES = [
    "gla", "kda", "gmm",            # interpret-runnable on this box
    "rpa", "moe_v1", "moe_v2", "fused_mlp", "kv_cache",  # tpu-deferred
]
# The gate suite: the kernels that actually execute here (interpret proxy), so the
# loop can run end-to-end on this machine. The TPU-deferred kernels join the gate
# once a TPU is attached (they are already wired + reference-validated).
FAST_SUITE = ["gla", "kda", "gmm"]
FULL_SUITE = RUNNER_MODULES

# Kernels expected to run in Pallas interpret here vs TPU-only (informational; the
# eval gate probes each case at runtime and records the actual regime).
INTERPRET_KERNELS = {"gla", "kda", "gmm"}
TPU_ONLY_KERNELS = {"rpa", "moe_v1", "moe_v2", "fused_mlp", "kv_cache"}


def load_cases(which: str = "fast"):
    mods = FAST_SUITE if which == "fast" else FULL_SUITE
    cases = []
    for m in mods:
        try:
            mod = importlib.import_module(f"akt.core.runners.{m}")
            cs = list(getattr(mod, "CASES", []))
            if not cs:
                print(f"[suites] {m}: no CASES exported — skipped")
            cases += cs
        except Exception as e:  # noqa: BLE001
            print(f"[suites] skip {m}: {type(e).__name__}: {str(e)[:140]}")
    return cases


def design_space_total(which: str = "full") -> int:
    """Sum of per-case design-space sizes (the 'supported' count gate analog)."""
    total = 0
    for c in load_cases(which):
        try:
            total += c.space.size()
        except Exception:  # noqa: BLE001
            pass
    return total
