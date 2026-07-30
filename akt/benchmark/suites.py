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

import functools
import importlib

import jax.numpy as jnp

from akt.benchmark.runners.base import identity_check_out

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


class SuiteContractError(RuntimeError):
    """An editable runner changed a frozen workload/correctness contract."""


# Frozen workload inventory and canonical make_inputs positional arguments. Runners
# may enlarge DesignSpace and change run(), but they may not replace workloads,
# inputs, references, tolerances, output projections, or native-test wiring.
EXPECTED_CASES = {
    "gla": {
        "gla:seq512_h8": (512, 8),
        "gla:seq2048_h8": (2048, 8),
    },
    "kda": {
        "kda:seq128_h2_d64": (128, 2, 64),
        "kda:seq256_h4_d128": (256, 4, 128),
    },
    "gmm": {
        "gmm:m256_k512_n512_g4": (256, 512, 512, 4),
        "gmm:m512_k512_n512_g8": (512, 512, 512, 8),
        "gmm_v2:m512_k1024_n1024_g8": (512, 1024, 1024, 8),
    },
    "rpa": {
        "rpa_v3:d_s2_q4kv2_hd128_p256_ctx512": (2, 4, 2, 128, 256, (512, 384)),
        "rpa_v3:d_s3_q8kv8_hd128_p256_ctx1024": (3, 8, 8, 128, 256, (1024, 768, 512)),
    },
    "moe_v1": {
        "moe_v1:t32_e8_k2_h2048_i1024": (jnp.bfloat16, 2, 8, 2048, 1024, 32),
    },
    "moe_v2": {
        "moe_v2:t32_e8_k2_h512_i512": (jnp.bfloat16, 2, 8, 512, 512, 32),
    },
    "fused_mlp": {
        "fused_mlp:s128_h256_i512": (128, 256, 512),
        "fused_mlp:s256_h512_i512": (256, 512, 512),
    },
    "kv_cache": {
        "kv_cache:h8_cache4096_new256": (8, 4096, 256, 128),
        "kv_cache:h16_cache8192_new512": (16, 8192, 512, 128),
    },
}

# Execution support is part of the frozen workload contract.  An editable runner
# must not be able to hide a broken local case by relabeling it as TPU-only.
EXPECTED_REGIME_PREFS = {
    "gla:seq512_h8": ("gpu", "cpu-interpret"),
    "gla:seq2048_h8": ("gpu", "cpu-interpret"),
    "kda:seq128_h2_d64": ("cpu-interpret",),
    "kda:seq256_h4_d128": ("cpu-interpret",),
    "gmm:m256_k512_n512_g4": ("gpu", "cpu-interpret"),
    "gmm:m512_k512_n512_g8": ("gpu", "cpu-interpret"),
    "gmm_v2:m512_k1024_n1024_g8": ("tpu",),
    "rpa_v3:d_s2_q4kv2_hd128_p256_ctx512": ("tpu-deferred",),
    "rpa_v3:d_s3_q8kv8_hd128_p256_ctx1024": ("tpu-deferred",),
    "moe_v1:t32_e8_k2_h2048_i1024": ("tpu-deferred",),
    "moe_v2:t32_e8_k2_h512_i512": ("tpu-deferred",),
    "fused_mlp:s128_h256_i512": ("tpu",),
    "fused_mlp:s256_h512_i512": ("tpu",),
    "kv_cache:h8_cache4096_new256": ("tpu-deferred",),
    "kv_cache:h16_cache8192_new512": ("tpu-deferred",),
}


def _validate_case_contract(case, refs, expected_input_args: tuple) -> None:
    """Reject any editable-runner attempt to weaken the frozen contract."""
    errors = []
    maker = case.make_inputs
    if not isinstance(maker, functools.partial):
        errors.append("make_inputs is not functools.partial")
    else:
        if maker.func is not refs.make_inputs:
            errors.append("make_inputs does not use the frozen generator")
        if maker.args != expected_input_args or maker.keywords:
            errors.append(
                f"make_inputs args changed: {maker.args!r}/{maker.keywords!r}"
            )
    if case.reference is not refs.reference:
        errors.append("reference is not the frozen reference")
    if case.atol != refs.ATOL or case.rtol != refs.RTOL:
        errors.append(
            f"tolerance changed: atol={case.atol!r}, rtol={case.rtol!r}"
        )
    expected_check = getattr(refs, "check_out", identity_check_out)
    if case.check_out is not expected_check:
        errors.append("check_out is not the frozen output projection")
    if case.native_test != refs.NATIVE_TEST:
        errors.append(f"native_test changed: {case.native_test!r}")
    expected_bitexact = bool(getattr(refs, "BITEXACT_INVARIANT", False))
    if bool(getattr(case, "bitexact_invariant", False)) != expected_bitexact:
        errors.append(
            f"bitexact_invariant changed: expected={expected_bitexact!r}, "
            f"actual={getattr(case, 'bitexact_invariant', False)!r}"
        )
    expected_regime = EXPECTED_REGIME_PREFS.get(case.case_id)
    if case.regime_pref != expected_regime:
        errors.append(
            f"regime_pref changed: expected={expected_regime!r}, "
            f"actual={case.regime_pref!r}"
        )
    if errors:
        raise SuiteContractError(f"{case.case_id}: " + "; ".join(errors))


def load_cases(which: str = "fast"):
    mods = FAST_SUITE if which == "fast" else FULL_SUITE
    cases = []
    for m in mods:
        try:
            # Import and retain the frozen objects before editable runner code runs.
            refs = importlib.import_module(f"akt.benchmark.refs.{m}")
            mod = importlib.import_module(f"akt.core.runners.{m}")
            cs = list(getattr(mod, "CASES", []))
            expected = EXPECTED_CASES[m]
            actual_ids = [c.case_id for c in cs]
            if len(actual_ids) != len(set(actual_ids)):
                raise SuiteContractError(f"{m}: duplicate case id")
            if set(actual_ids) != set(expected):
                raise SuiteContractError(
                    f"{m}: workload set changed; expected={sorted(expected)}, "
                    f"actual={sorted(actual_ids)}"
                )
            for case in cs:
                _validate_case_contract(case, refs, expected[case.case_id])
            cases.extend(cs)
        except Exception as e:  # noqa: BLE001
            if isinstance(e, SuiteContractError):
                raise
            raise SuiteContractError(
                f"{m}: runner/contract import failed: {type(e).__name__}: {str(e)[:140]}"
            ) from e
    return cases


def design_space_total(which: str = "full") -> int:
    """Sum of per-case deployment-space sizes used by the AKT objective."""
    total = 0
    for c in load_cases(which):
        try:
            total += c.space.deployment_space().size()
        except Exception:  # noqa: BLE001
            pass
    return total
