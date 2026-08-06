"""Frozen model-level AKT search and target-hardware acceptance gate.

For each of three immutable serving workloads this gate:

* materializes fixed-seed inputs/weights and frozen correctness references;
* measures every valid deployable configuration of every callsite;
* finds the exact optimum of the finite additive model with a layered dynamic program;
* verifies DP against exhaustive Cartesian search on a bounded version of the same
  model graph;
* executes selected and incumbent synthetic serving-trace bundles in paired order;
* records backend execution events for newly selected capability controls.

No off-target or deferred case contributes to the objective.  The DP certificate is
an exact statement about the fingerprinted additive model; paired model latency is
the independent real-hardware keep/reject measurement.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
for path in (str(REPO), str(REPO / "python")):
    if path not in sys.path:
        sys.path.insert(0, path)

from akt.benchmark.model_workloads import (  # noqa: E402
    MODEL_WORKLOADS,
    TARGET_BACKEND,
    callsite_id,
    contract_fingerprint,
    contract_payload,
    validate_workloads,
)
from akt.benchmark.runtime import (  # noqa: E402
    _register_probe_code,
    _report_probed_backend_execution,
    capture_runtime_events,
    runtime_scope,
)
from akt.benchmark.suites import load_cases  # noqa: E402
from akt.core.evolve.capability_contract import derive_search_dimensions  # noqa: E402
from akt.core.search.model_dp import (  # noqa: E402
    EmpiricalPanel,
    PlanCandidate,
    certify,
    empirical_panel_fingerprint,
    empirical_plan_panel,
    measured_graph_fingerprint,
)


OBJECTIVE_SCOPE = "model-serving-empirical-dp-v2"
EMPIRICAL_SCOPE = "synthetic_trace_additivity"
_CAPABILITY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _geomean(values):
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
        and math.isfinite(value)
    ]
    return math.exp(sum(math.log(value) for value in finite) / len(finite)) if finite else None


def _hardware_record(target_backend: str) -> dict:
    backend = jax.default_backend()
    devices = [
        {
            "platform": device.platform,
            "kind": getattr(device, "device_kind", ""),
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    interpret = os.environ.get("PALLAS_INTERPRET", "0") == "1"
    payload = {
        "backend": backend,
        "target_backend": target_backend,
        "pallas_interpret": interpret,
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
        "local_device_count": int(jax.local_device_count()),
        "global_device_count": int(jax.device_count()),
        "devices": devices,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["ok"] = backend == target_backend and not interpret
    return payload


def _hash_tree(tree) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _compare(case, output, reference) -> tuple[bool, str]:
    try:
        out_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(case.check_out(output))]
        ref_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(case.check_out(reference))]
    except Exception as error:  # noqa: BLE001
        return False, f"projection failed: {type(error).__name__}: {str(error)[:100]}"
    if len(out_leaves) != len(ref_leaves):
        return False, f"nleaves {len(out_leaves)} != {len(ref_leaves)}"
    worst = 0.0
    for actual, expected in zip(out_leaves, ref_leaves):
        if actual.shape != expected.shape:
            return False, f"shape {actual.shape} != {expected.shape}"
        actual = actual.astype(np.float32)
        expected = expected.astype(np.float32)
        if not np.allclose(
            actual,
            expected,
            atol=case.atol,
            rtol=case.rtol,
            equal_nan=False,
        ):
            delta = float(np.nanmax(np.abs(actual - expected)))
            return False, f"maxabs {delta:.2e} > atol {case.atol:.1e}"
        if actual.size:
            worst = max(worst, float(np.nanmax(np.abs(actual - expected))))
    return True, f"maxabs {worst:.2e}"


def _local_timing_protocol(runs: int) -> dict:
    return {
        "clock": "time.perf_counter",
        "device_synchronization": "jax.block_until_ready",
        "warmup_runs": "one correctness execution per configuration",
        "measured_runs": runs,
        "statistic": "median",
        "ordering": "deterministic-round-robin-rotated-reversed",
    }

def _kernel_module_name(relative: str) -> str:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ValueError(f"invalid backend source path {relative!r}")
    parts = list(path.with_suffix("").parts)
    if not parts or parts[0] != "python":
        raise ValueError(f"backend source must be under python/: {relative!r}")
    parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _install_backend_probes(cases, capability: str | None) -> list[dict]:
    """Wrap each declared backend entry before any candidate is compiled.

    The wrapper binds the concrete kernel argument and records it only while the
    frozen evaluator has established a model/callsite scope.  Runner aliases and the
    defining module are both patched, covering direct imports and module-qualified
    calls without trusting editable runner self-reports.
    """

    if not capability:
        return []
    if not _CAPABILITY_RE.fullmatch(capability):
        raise ValueError(f"invalid capability identifier {capability!r}")
    manifest_path = REPO / "akt/core/evolve/capabilities" / f"{capability}.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("name") != capability:
        raise ValueError("runtime probe manifest identity mismatch")
    _action, dimensions, dimension_errors = derive_search_dimensions(manifest, REPO)
    if dimension_errors:
        raise ValueError(
            "runtime probe manifest is invalid: " + "; ".join(dimension_errors)
        )

    grouped: dict[tuple[str, str], list[dict]] = {}
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            raise ValueError("runtime probe dimension must be an object")
        key = (dimension.get("kernel_path"), dimension.get("kernel_function"))
        if not all(isinstance(item, str) and item for item in key):
            raise ValueError(f"runtime probe has invalid backend identity: {key!r}")
        grouped.setdefault(key, []).append(dimension)

    runner_modules = {
        sys.modules[case.run.__module__]
        for case in cases
        if case.run.__module__ in sys.modules
    }
    installed = []
    for (kernel_path, kernel_function), backend_dimensions in grouped.items():
        module = importlib.import_module(_kernel_module_name(kernel_path))
        target = getattr(module, kernel_function, None)
        if not callable(target):
            raise ValueError(f"backend probe target is not callable: {kernel_function}")
        signature = inspect.signature(target)
        for dimension in backend_dimensions:
            argument = dimension.get("kernel_argument")
            if argument not in signature.parameters:
                raise ValueError(
                    f"backend probe argument {argument!r} is absent from {kernel_function}"
                )

        @functools.wraps(target)
        def observed_backend(
            *call_args,
            __target=target,
            __signature=signature,
            __dimensions=tuple(backend_dimensions),
            __backend=kernel_function,
            **call_kwargs,
        ):
            bound = __signature.bind_partial(*call_args, **call_kwargs)
            for dimension in __dimensions:
                argument = dimension["kernel_argument"]
                if argument in bound.arguments:
                    _report_probed_backend_execution(
                        capability=capability,
                        control=dimension["control"],
                        value=bound.arguments[argument],
                        backend=__backend,
                    )
            return __target(*call_args, **call_kwargs)

        # authenticate the wrapper to the frozen reporter by code-object identity
        _register_probe_code(observed_backend.__code__)
        aliases = []
        for candidate_module in runner_modules | {module}:
            for name, value in list(vars(candidate_module).items()):
                if value is target:
                    setattr(candidate_module, name, observed_backend)
                    aliases.append(f"{candidate_module.__name__}.{name}")
        if not aliases:
            raise ValueError(
                f"no model runner dispatches declared backend {kernel_function!r}"
            )
        installed.append(
            {
                "kernel_path": kernel_path,
                "kernel_function": kernel_function,
                "controls": [dimension["control"] for dimension in backend_dimensions],
                "patched_aliases": sorted(aliases),
            }
        )
    return installed


def _clear_runner_dispatch_caches(workload, prepared) -> None:
    """Force the selected-plan run through newly installed Python probes."""

    jax.clear_caches()
    modules = {
        sys.modules[prepared[callsite_id(workload, call)][0].run.__module__]
        for call in workload.calls
    }
    for module in modules:
        for value in vars(module).values():
            clear = getattr(value, "cache_clear", None)
            if callable(clear):
                clear()


def _search_case(case, inputs, reference, runs: int, max_configs: int) -> dict:
    space = case.space.deployment_space()
    configs = list(itertools.islice(space.enumerate(cap=None), max_configs + 1))
    if not configs:
        raise ValueError(f"{case.case_id} has no valid deployable configurations")
    if len(configs) > max_configs:
        raise ValueError(
            f"{case.case_id} has more than {max_configs} valid configurations; "
            "no truncated optimum is accepted"
        )

    bitexact = bool(getattr(case, "bitexact_invariant", False))
    correct_entries = []
    failures = []
    output_hashes: dict[str, list] = {}

    def _execute_config(config):
        """One materialized run + frozen comparison; hashes correct outputs when
        the case declares the bit-exact invariant. Shared by the deployment loop
        and the research-space invariant sweep below."""
        output = case.run(inputs, config)
        jax.block_until_ready(output)
        correct, detail = _compare(case, output, reference)
        if correct and bitexact:
            output_hashes.setdefault(
                _hash_tree(case.check_out(output)), []
            ).append(config)
        return correct, detail

    for index, config in enumerate(configs, 1):
        try:
            correct, detail = _execute_config(config)
        except Exception as error:  # noqa: BLE001
            correct = False
            detail = f"{type(error).__name__}: {str(error)[:120]}"
        if not correct:
            failures.append({"config": config, "reason": detail})
            continue
        correct_entries.append(
            {
                "config": config,
                "latency_samples_s": [],
            }
        )
    if bitexact:
        # The invariant claims the kernel's AXES cannot change the math, so it is
        # checked over the FULL RESEARCH space (runner-only knob values included),
        # not just the deployment space — which collapses runner-only knobs to
        # their defaults and can shrink to a single config (making a deployment-
        # only check vacuous). Research-only configs are hashed and correctness-
        # checked here but never timed and never enter the DP.
        research_configs = list(
            itertools.islice(case.space.enumerate(cap=None), max_configs + 1)
        )
        if len(research_configs) > max_configs:
            raise ValueError(
                f"{case.case_id} research space exceeds {max_configs} configs; "
                "the bit-exact invariant sweep must be exhaustive"
            )
        for config in research_configs:
            if config in configs:
                continue                    # already executed + hashed above
            correct, detail = _execute_config(config)
            if not correct:
                raise ValueError(
                    f"{case.case_id} violates its BITEXACT_INVARIANT contract: "
                    f"research config {config} is not even allclose-correct "
                    f"({detail}) though the axes are declared config-independent"
                )
    if bitexact and len(output_hashes) > 1:
        # The frozen refs contract declares this kernel's knobs config-independent
        # (pure data movement): any bit-level divergence across correct configs is a
        # semantic drift the family allclose tolerance would hide. Fail closed.
        groups = {
            digest[:12]: configs_[:3] for digest, configs_ in output_hashes.items()
        }
        raise ValueError(
            f"{case.case_id} violates its BITEXACT_INVARIANT contract: correct "
            f"configs produced {len(output_hashes)} distinct output hashes: {groups}"
        )

    if runs < 1:
        raise ValueError("local timing runs must be positive")
    n_correct = len(correct_entries)
    for pass_index in range(runs):
        order = list(range(n_correct))
        if order:
            shift = (pass_index // 2) % n_correct
            order = order[shift:] + order[:shift]
            if pass_index % 2:
                order.reverse()
        for entry_index in order:
            entry = correct_entries[entry_index]
            started = time.perf_counter()
            jax.block_until_ready(case.run(inputs, entry["config"]))
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(
                    f"{case.case_id} produced invalid latency {elapsed!r} for "
                    f"{entry['config']}"
                )
            entry["latency_samples_s"].append(elapsed)
        print(
            f"[akt-model-eval] {case.case_id}: timing pass "
            f"{pass_index + 1}/{runs} over {n_correct} correct configs",
            flush=True,
        )
    measurements = []
    for entry in correct_entries:
        measurements.append(
            {
                **entry,
                "latency_s": statistics.median(entry["latency_samples_s"]),
            }
        )
    return {
        "case": case.case_id,
        "space_size": space.size(),
        # the FULL research space (runner-only knob values included) — the funnel's
        # top tier; space_size above is the deployment-space total.
        "research_space_size": case.space.size(),
        "valid_configs": len(configs),
        "correct_configs": len(measurements),
        "all_configs_correct": len(measurements) == len(configs),
        "failures": failures[:16],
        "measurements": measurements,
        "bitexact_invariant": bitexact,
        "bitexact_output_hash": next(iter(output_hashes), None) if bitexact else None,
        "timing_protocol": _local_timing_protocol(runs),
        "knobs": [
            {
                "name": knob.name,
                "values": knob.values,
                "default": knob.default,
                "elevated_by": knob.elevated_by,
                "programmer_control": knob.programmer_control,
            }
            for knob in space.knobs
        ],
    }


def _normalize_plan(workload, raw_plan, case_by_id) -> dict[str, dict]:
    normalized = {}
    for call in workload.calls:
        site = callsite_id(workload, call)
        case = case_by_id[call.case_id]
        config = case.space.deployment_space().default_config()
        provided = (raw_plan or {}).get(site, {})
        unknown = set(provided) - set(config)
        if unknown:
            raise ValueError(f"incumbent plan {site} has unknown controls: {sorted(unknown)}")
        config.update(provided)
        if not case.space.deployment_space().valid(config):
            raise ValueError(f"incumbent plan {site} is no longer valid: {config}")
        normalized[site] = config
    return normalized


def _execute_model(workload, plan, prepared, *, collect_events=False):
    outputs = []
    capture = capture_runtime_events() if collect_events else None
    if capture is None:
        for call in workload.calls:
            site = callsite_id(workload, call)
            case, inputs, _reference = prepared[site]
            with runtime_scope(workload.model_id, site):
                outputs.append(case.run(inputs, plan[site]))
        jax.block_until_ready(tuple(outputs))
        return outputs, []
    with capture as events:
        for call in workload.calls:
            site = callsite_id(workload, call)
            case, inputs, _reference = prepared[site]
            with runtime_scope(workload.model_id, site):
                outputs.append(case.run(inputs, plan[site]))
        jax.block_until_ready(tuple(outputs))
    return outputs, events


def _paired_plan_time(
    workload,
    numerator_plan,
    denominator_plan,
    prepared,
    runs: int,
    *,
    numerator_label: str,
    denominator_label: str,
    warmup: bool = False,
) -> dict:
    """Measure balanced adjacent pairs and report a paired geometric ratio."""

    paired_runs = _balanced_pair_count(runs)
    if warmup:
        _execute_model(workload, denominator_plan, prepared)
        _execute_model(workload, numerator_plan, prepared)

    def measure(plan):
        started = time.perf_counter()
        _execute_model(workload, plan, prepared)
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid complete-plan latency {elapsed!r}")
        return elapsed

    numerator_samples = []
    denominator_samples = []
    pair_orders = []
    for index in range(paired_runs):
        if index % 2:
            numerator_samples.append(measure(numerator_plan))
            denominator_samples.append(measure(denominator_plan))
            pair_orders.append(f"{numerator_label}-{denominator_label}")
        else:
            denominator_samples.append(measure(denominator_plan))
            numerator_samples.append(measure(numerator_plan))
            pair_orders.append(f"{denominator_label}-{numerator_label}")
    log_ratios = [
        math.log(numerator / denominator)
        for numerator, denominator in zip(numerator_samples, denominator_samples)
    ]
    return {
        f"{numerator_label}_s": statistics.median(numerator_samples),
        f"{denominator_label}_s": statistics.median(denominator_samples),
        f"{numerator_label}_samples_s": numerator_samples,
        f"{denominator_label}_samples_s": denominator_samples,
        "paired_log_ratios": log_ratios,
        "ratio": math.exp(statistics.median(log_ratios)),
        "pair_orders": pair_orders,
        "paired_runs": paired_runs,
    }


def _balanced_pair_count(runs: int) -> int:
    if runs < 2:
        raise ValueError("paired runs must be at least 2")
    return runs if runs % 2 == 0 else runs + 1


def _empirical_timing_protocol(
    runs: int,
    bootstrap_samples: int,
    regret_tolerance: float,
    interaction_tolerance: float,
) -> dict:
    return {
        "scope": EMPIRICAL_SCOPE,
        "clock": "time.perf_counter",
        "device_synchronization": "jax.block_until_ready",
        "warmup": "one complete correctness execution per plan",
        "pair_order": "selected/challenger then challenger/selected alternating",
        "paired_runs": _balanced_pair_count(runs),
        "ratio_statistic": "exp(median(log(challenger_s/selected_s)))",
        "confidence": "deterministic nonparametric joint bootstrap",
        "confidence_level": 0.95,
        "bootstrap_samples": bootstrap_samples,
        "regret_tolerance": regret_tolerance,
        "interaction_tolerance": interaction_tolerance,
    }


def _check_model_outputs(workload, outputs, prepared) -> tuple[bool, list[str]]:
    errors = []
    for call, output in zip(workload.calls, outputs):
        site = callsite_id(workload, call)
        case, _inputs, reference = prepared[site]
        correct, detail = _compare(case, output, reference)
        if not correct:
            errors.append(f"{site}: {detail}")
    return not errors, errors


def _empirical_dp_metrics(
    panel: EmpiricalPanel,
    records: list[dict],
    *,
    panel_sha256: str,
    bootstrap_samples: int,
    regret_tolerance: float,
    interaction_tolerance: float,
) -> dict:
    """Compare modeled sums with paired whole-trace measurements.

    This function is intentionally independent of JAX so deterministic synthetic
    tests can exercise the statistical pass/fail boundary.
    """

    if bootstrap_samples < 100:
        raise ValueError("empirical bootstrap_samples must be at least 100")
    selected_id = panel.selected.fingerprint()
    by_id = {record["plan_id"]: record for record in records}
    if selected_id not in by_id:
        raise ValueError("empirical records omit the full DP-selected plan")
    selected_predicted = panel.selected.predicted_cost_s
    if not math.isfinite(selected_predicted) or selected_predicted <= 0:
        raise ValueError("selected modeled latency must be finite and positive")

    errors = []
    challengers = []
    for record in records:
        predicted = float(record["predicted_s"])
        if not math.isfinite(predicted) or predicted <= 0:
            raise ValueError("plan modeled latency must be finite and positive")
        record["predicted_ratio"] = predicted / selected_predicted
        if record["plan_id"] == selected_id:
            record["observed_ratio"] = 1.0
            record["prediction_residual_log"] = 0.0
            continue
        samples = np.asarray(record.get("paired_log_ratios") or [], dtype=np.float64)
        if samples.size < 2 or not np.all(np.isfinite(samples)):
            errors.append(f"plan {record['plan_id'][:12]} lacks finite paired samples")
            continue
        observed_log = float(np.median(samples))
        record["observed_ratio"] = math.exp(observed_log)
        record["prediction_residual_log"] = observed_log - math.log(
            record["predicted_ratio"]
        )
        challengers.append((record, samples))

    measured_records = [item for item in records if "observed_ratio" in item]

    if not challengers:
        if len(records) == 1 and not errors:
            return {
                "ok": True,
                "status": "validated-vacuous-single-plan",
                "errors": [],
                "selection_supported": True,
                "additivity_supported": True,
                "regret": {
                    "point": 0.0,
                    "upper95": 0.0,
                    "tolerance": regret_tolerance,
                },
                "interaction": {
                    "verdict": "vacuous-single-plan",
                    "factor_count": 0,
                    "comparisons": [],
                    "max_abs_residual": 0.0,
                    "max_abs_residual_upper95": 0.0,
                    "tolerance": interaction_tolerance,
                },
            }
        errors.append("empirical panel has no measured challenger")
        return {
            "ok": False,
            "status": "inconclusive",
            "errors": errors,
            "selection_supported": False,
            "additivity_supported": False,
        }

    seed = int(panel_sha256[:16], 16)
    rng = np.random.default_rng(seed)
    bootstrap_ratio_by_id = {selected_id: np.ones(bootstrap_samples, dtype=np.float64)}
    observed_ratio_by_choices = {panel.selected.choices: 1.0}
    record_by_choices = {tuple(record["choices"]): record for record in measured_records}
    for record, samples in challengers:
        indices = rng.integers(0, samples.size, size=(bootstrap_samples, samples.size))
        boot_logs = np.median(samples[indices], axis=1)
        bootstrap_ratio_by_id[record["plan_id"]] = np.exp(boot_logs)
        observed_ratio_by_choices[tuple(record["choices"])] = record["observed_ratio"]

    challenger_ratios = np.column_stack(
        [bootstrap_ratio_by_id[record["plan_id"]] for record, _samples in challengers]
    )
    bootstrap_regret = np.maximum(
        0.0,
        np.max(1.0 / challenger_ratios - 1.0, axis=1),
    )
    point_regret = max(
        0.0,
        max(1.0 / record["observed_ratio"] - 1.0 for record, _samples in challengers),
    )
    regret_upper = float(np.quantile(bootstrap_regret, 0.95))
    selection_supported = regret_upper <= regret_tolerance

    factor_indices = [
        index for index, stage in enumerate(panel.stage_candidates) if len(stage) == 2
    ]
    zero_choices = panel.selected.choices
    singleton_choices = {}
    for factor in factor_indices:
        choices = list(zero_choices)
        choices[factor] = 1
        choices = tuple(choices)
        if choices in record_by_choices:
            singleton_choices[factor] = choices

    interactions = []
    bootstrap_interactions = []
    for choices, record in sorted(record_by_choices.items()):
        enabled = [factor for factor in factor_indices if choices[factor] == 1]
        if len(enabled) < 2 or any(factor not in singleton_choices for factor in enabled):
            continue
        additive_ratio = 1.0 + sum(
            observed_ratio_by_choices[singleton_choices[factor]] - 1.0
            for factor in enabled
        )
        residual = record["observed_ratio"] - additive_ratio
        interactions.append(
            {
                "plan_id": record["plan_id"],
                "choices": list(choices),
                "order": len(enabled),
                "observed_ratio": record["observed_ratio"],
                "additive_reconstruction_ratio": additive_ratio,
                "residual_ratio": residual,
            }
        )
        additive_boot = 1.0 + sum(
            bootstrap_ratio_by_id[record_by_choices[singleton_choices[factor]]["plan_id"]]
            - 1.0
            for factor in enabled
        )
        bootstrap_interactions.append(
            np.abs(bootstrap_ratio_by_id[record["plan_id"]] - additive_boot)
        )

    if len(factor_indices) < 2:
        interaction_point = interaction_upper = 0.0
        additivity_supported = True
        additivity_verdict = "vacuous-single-factor"
    elif not interactions:
        interaction_point = interaction_upper = None
        additivity_supported = False
        additivity_verdict = "inconclusive-no-valid-interaction-plan"
        errors.append("binary panel contains no measurable multi-factor interaction")
    else:
        interaction_point = max(abs(item["residual_ratio"]) for item in interactions)
        interaction_boot = np.max(np.column_stack(bootstrap_interactions), axis=1)
        interaction_upper = float(np.quantile(interaction_boot, 0.95))
        additivity_supported = interaction_upper <= interaction_tolerance
        additivity_verdict = "supported" if additivity_supported else "unsupported"

    if not selection_supported:
        errors.append(
            f"empirical DP regret upper95 {regret_upper:.6f} exceeds "
            f"tolerance {regret_tolerance:.6f}"
        )
    if not additivity_supported and interactions:
        errors.append(
            f"interaction residual upper95 {interaction_upper:.6f} exceeds "
            f"tolerance {interaction_tolerance:.6f}"
        )
    ok = selection_supported and additivity_supported and not errors
    point_falsified = point_regret > regret_tolerance or (
        interaction_point is not None and interaction_point > interaction_tolerance
    )
    return {
        "ok": ok,
        "status": "validated" if ok else ("falsified" if point_falsified else "inconclusive"),
        "errors": errors,
        "selection_supported": selection_supported,
        "additivity_supported": additivity_supported,
        "regret": {
            "point": point_regret,
            "upper95": regret_upper,
            "tolerance": regret_tolerance,
        },
        "interaction": {
            "verdict": additivity_verdict,
            "factor_count": len(factor_indices),
            "comparisons": interactions,
            "max_abs_residual": interaction_point,
            "max_abs_residual_upper95": interaction_upper,
            "tolerance": interaction_tolerance,
        },
    }


def _evaluate_empirical_panel(
    workload,
    panel: EmpiricalPanel,
    prepared,
    *,
    measured_graph_sha256: str,
    runs: int,
    bootstrap_samples: int,
    regret_tolerance: float,
    interaction_tolerance: float,
) -> dict:
    protocol = _empirical_timing_protocol(
        runs,
        bootstrap_samples,
        regret_tolerance,
        interaction_tolerance,
    )
    panel_sha256 = empirical_panel_fingerprint(
        panel,
        timing_protocol=protocol,
        measured_graph_sha256=measured_graph_sha256,
    )
    result = {
        "scope": EMPIRICAL_SCOPE,
        "ok": False,
        "status": "inconclusive",
        "panel_complete": panel.complete,
        "panel_sha256": panel_sha256,
        "measured_graph_sha256": measured_graph_sha256,
        "timing_protocol": protocol,
        "declared_combinations": panel.declared_combinations,
        "valid_plans": len(panel.plans),
        "max_plans": panel.max_plans,
        "selected_plan_id": panel.selected.fingerprint(),
        "plans": [],
        "errors": [],
    }
    if not panel.complete:
        result["errors"].append(panel.error or "empirical plan panel is incomplete")
        return result

    all_correct = True
    records = []
    for index, empirical_plan in enumerate(panel.plans, 1):
        plan = empirical_plan.plan()
        outputs, _events = _execute_model(workload, plan, prepared)
        correct, correctness_errors = _check_model_outputs(workload, outputs, prepared)
        all_correct &= correct
        record = {
            "plan_id": empirical_plan.fingerprint(),
            "choices": list(empirical_plan.choices),
            "plan": plan,
            "predicted_s": empirical_plan.predicted_cost_s,
            "correct": correct,
            "correctness_errors": correctness_errors,
        }
        records.append(record)
        print(
            f"[akt-model-eval] {workload.model_id} empirical plan correctness "
            f"{index}/{len(panel.plans)}: {'ok' if correct else 'FAILED'}",
            flush=True,
        )
    result["all_plans_correct"] = all_correct
    result["plans"] = records
    if not all_correct:
        result["status"] = "falsified"
        result["errors"].append("one or more complete empirical plans failed correctness")
        return result
    selected_plan = panel.selected.plan()
    for index, record in enumerate(records[1:], 1):
        paired = _paired_plan_time(
            workload,
            record["plan"],
            selected_plan,
            prepared,
            runs,
            numerator_label="challenger",
            denominator_label="selected",
        )
        record.update(paired)
        print(
            f"[akt-model-eval] {workload.model_id} empirical paired plan "
            f"{index}/{len(records) - 1}",
            flush=True,
        )
    metrics = _empirical_dp_metrics(
        panel,
        records,
        panel_sha256=panel_sha256,
        bootstrap_samples=bootstrap_samples,
        regret_tolerance=regret_tolerance,
        interaction_tolerance=interaction_tolerance,
    )
    result.update(metrics)
    result["plans"] = records
    return result


def _runtime_evidence(capability, models, case_by_id) -> dict:
    if not capability:
        return {"required": False, "ok": True, "controls": [], "events": []}
    by_control = {}
    all_events = []
    for model in models:
        all_events.extend(model.get("runtime_events") or [])
        plan = model.get("selected_plan") or {}
        workload = next(item for item in MODEL_WORKLOADS if item.model_id == model["model"])
        for call in workload.calls:
            site = callsite_id(workload, call)
            case = case_by_id[call.case_id]
            for knob in case.space.deployment_space().knobs:
                if knob.elevated_by != capability or not knob.programmer_control:
                    continue
                selected = plan[site][knob.name]
                record = by_control.setdefault(
                    knob.programmer_control,
                    {"control": knob.programmer_control, "selections": [], "observed": []},
                )
                record["selections"].append(
                    {
                        "model": workload.model_id,
                        "callsite": site,
                        "knob": knob.name,
                        "default": knob.default,
                        "selected": selected,
                        "nondefault": selected != knob.default,
                    }
                )
    for event in all_events:
        control = event.get("control")
        if control in by_control:
            by_control[control]["observed"].append(event)
    errors = []
    for control, record in by_control.items():
        selected = [entry for entry in record["selections"] if entry["nondefault"]]
        matched = [
            event
            for event in record["observed"]
            if event.get("verified_backend_probe") is True
            and any(
                event.get("model") == entry["model"]
                and event.get("callsite") == entry["callsite"]
                and event.get("value") == entry["selected"]
                and event.get("backend")
                for entry in selected
            )
        ]
        record["matched"] = matched
        if not selected:
            errors.append(f"{control} was not selected at a non-default value")
        elif not matched:
            errors.append(f"{control} had no matching backend execution event")
    if not by_control:
        errors.append(f"capability {capability!r} added no deployable runner control")
    return {
        "required": True,
        "ok": not errors,
        "controls": list(by_control.values()),
        "events": all_events,
        "errors": errors,
    }


def evaluate(args) -> dict:
    hardware = _hardware_record(args.target_backend)
    testbench = bool(args.allow_non_target)
    if testbench and not hardware["ok"] and not hardware["pallas_interpret"]:
        # TESTBENCH: off-target execution always proceeds in Pallas interpret
        # mode so the TPU kernels that support it can run on this host.
        os.environ["PALLAS_INTERPRET"] = "1"
        hardware = _hardware_record(args.target_backend)
    summary = {
        "suite": "three-frozen-models",
        "objective_scope": OBJECTIVE_SCOPE,
        "target_hardware": hardware,
        "target_hardware_ok": hardware["ok"],
        "model_contract": contract_payload(),
        "model_contract_fingerprint": contract_fingerprint(),
        "models": [],
        "all_correct": False,
        "n_deferred": 0,
        "empirical_dp_certificate": {
            "scope": EMPIRICAL_SCOPE,
            "ok": False,
            "status": "deferred",
            "models": [],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not hardware["ok"] and not args.allow_non_target:
        summary["n_deferred"] = sum(len(model.calls) for model in MODEL_WORKLOADS)
        summary["error"] = (
            f"target hardware required: backend={hardware['backend']!r}, "
            f"target={args.target_backend!r}, interpret={hardware['pallas_interpret']}"
        )
        return summary
    if testbench:
        summary["testbench"] = True
        print(
            "[akt-model-eval] TESTBENCH (--allow-non-target): interpret proxy on "
            f"backend={hardware['backend']!r}; results are NOT target-hardware "
            "evidence (target_hardware.ok stays False)",
            flush=True,
        )

    cases = load_cases("full")
    case_by_id = {case.case_id: case for case in cases}
    validate_workloads(case_by_id)
    summary["backend_probes"] = _install_backend_probes(cases, args.capability)
    prepared = {}
    tensor_fingerprints = {}
    deferred_calls: dict[str, str] = {}
    for workload in MODEL_WORKLOADS:
        for call in workload.calls:
            site = callsite_id(workload, call)
            print(f"[akt-model-eval] prepare fixed inputs/reference {site}", flush=True)
            case = case_by_id[call.case_id]
            try:
                inputs = case.make_inputs(seed=call.seed)
                reference = case.reference(inputs)
                jax.block_until_ready(reference)
            except Exception as error:  # noqa: BLE001
                if not testbench:
                    raise
                deferred_calls[site] = (
                    f"prepare raised {type(error).__name__}: {str(error)[:200]}"
                )
                continue
            prepared[site] = (case, inputs, reference)
            tensor_fingerprints[site] = {
                "input_weight_sha256": _hash_tree(inputs),
                "reference_sha256": _hash_tree(case.check_out(reference)),
                "seed": call.seed,
                "phase": call.phase,
            }
    if testbench:
        # Probe every callsite once on the DEFAULT deployable config: a case that
        # raises here cannot run on this host and is DEFERRED (per-call error
        # recorded); a model workload containing any deferred call is excluded.
        for site in sorted(prepared):
            case, inputs, reference = prepared[site]
            config = case.space.deployment_space().default_config()
            print(
                f"[akt-model-eval] testbench default-config probe {site}", flush=True
            )
            try:
                output = case.run(inputs, config)
                jax.block_until_ready(output)
                _compare(case, output, reference)
            except Exception as error:  # noqa: BLE001
                deferred_calls[site] = (
                    f"default-config probe raised {type(error).__name__}: "
                    f"{str(error)[:200]}"
                )
                prepared.pop(site)
                tensor_fingerprints.pop(site, None)
    runnable_workloads = []
    deferred_models = []
    for workload in MODEL_WORKLOADS:
        bad = {
            site: deferred_calls[site]
            for call in workload.calls
            if (site := callsite_id(workload, call)) in deferred_calls
        }
        if bad:
            deferred_models.append(
                {"model": workload.model_id, "deferred_calls": bad}
            )
        else:
            runnable_workloads.append(workload)
    if testbench:
        summary["deferred_models"] = deferred_models
        summary["deferred_calls"] = deferred_calls
        if not runnable_workloads:
            summary["n_deferred"] = len(deferred_calls)
            summary["error"] = (
                "testbench: every model workload contains a deferred call: "
                + "; ".join(sorted(deferred_calls))
            )
            return summary
    elif deferred_models:
        raise AssertionError("deferred calls are impossible outside testbench mode")
    runnable_case_ids = {
        call.case_id for workload in runnable_workloads for call in workload.calls
    }
    summary["tensor_fingerprints"] = tensor_fingerprints
    summary["workload_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "contract": summary["model_contract_fingerprint"],
                "tensors": tensor_fingerprints,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    case_search = {}
    for case_id, case in case_by_id.items():
        if case_id not in runnable_case_ids:
            continue                      # testbench-only: case has no runnable model
        print(f"[akt-model-eval] exhaustive local search {case_id}", flush=True)
        site = next(site for site, item in prepared.items() if item[0].case_id == case_id)
        _case, inputs, reference = prepared[site]
        case_search[case_id] = _search_case(
            case,
            inputs,
            reference,
            args.runs,
            args.max_configs,
        )
    summary["case_search"] = case_search

    incumbent_plans = {}
    if args.incumbent_plans:
        incumbent_plans = json.loads(Path(args.incumbent_plans).read_text())

    model_results = []
    for workload in runnable_workloads:
        print(f"[akt-model-eval] exact DP and bounded brute force {workload.model_id}", flush=True)
        stages = []
        stage_timing_samples = []
        for call in workload.calls:
            site = callsite_id(workload, call)
            measurements = case_search[call.case_id]["measurements"]
            stage = [
                PlanCandidate.from_dict(
                    site,
                    measurement["config"],
                    measurement["latency_s"],
                )
                for measurement in measurements
            ]
            stages.append(stage)
            stage_timing_samples.append(
                {
                    "callsite": site,
                    "measurements": [
                        {
                            "config": measurement["config"],
                            "latency_samples_s": measurement["latency_samples_s"],
                        }
                        for measurement in measurements
                    ],
                }
            )
        certificate = certify(stages)
        local_protocol = _local_timing_protocol(args.runs)
        measured_graph_sha256 = measured_graph_fingerprint(
            stages,
            timing_protocol=local_protocol,
            timing_samples=stage_timing_samples,
        )
        certificate["measured_graph_fingerprint"] = measured_graph_sha256
        certificate["local_timing_protocol"] = local_protocol
        panel = empirical_plan_panel(stages, max_plans=args.empirical_max_plans)
        selected_plan = panel.selected.plan()
        if selected_plan != certificate["full_plan"]:
            raise AssertionError("empirical panel does not contain the full DP plan")
        incumbent_plan = _normalize_plan(
            workload,
            incumbent_plans.get(workload.model_id, selected_plan),
            case_by_id,
        )

        _clear_runner_dispatch_caches(workload, prepared)
        selected_outputs, events = _execute_model(
            workload,
            selected_plan,
            prepared,
            collect_events=bool(args.capability),
        )
        selected_correct, selected_errors = _check_model_outputs(
            workload, selected_outputs, prepared
        )
        # DEFAULT-REPRODUCES-INCUMBENT evidence: execute the incumbent plan once and
        # record per-callsite output hashes (plus the selected plan's, which becomes
        # the pin after a KEEP). The loop pins these in campaign state and rejects a
        # later round whose incumbent-plan outputs drift bit-wise — a default-path
        # semantic change hiding inside the family allclose tolerance fails closed.
        incumbent_outputs, _ = _execute_model(workload, incumbent_plan, prepared)
        incumbent_correct, incumbent_errors = _check_model_outputs(
            workload, incumbent_outputs, prepared
        )

        def _plan_output_hashes(outputs):
            return {
                callsite_id(workload, call): _hash_tree(
                    prepared[callsite_id(workload, call)][0].check_out(output)
                )
                for call, output in zip(workload.calls, outputs)
            }

        incumbent_output_hashes = _plan_output_hashes(incumbent_outputs)
        selected_output_hashes = _plan_output_hashes(selected_outputs)
        empirical = _evaluate_empirical_panel(
            workload,
            panel,
            prepared,
            measured_graph_sha256=measured_graph_sha256,
            runs=args.empirical_runs,
            bootstrap_samples=args.empirical_bootstrap_samples,
            regret_tolerance=args.empirical_regret_tolerance,
            interaction_tolerance=args.empirical_interaction_tolerance,
        )
        paired = _paired_plan_time(
            workload,
            selected_plan,
            incumbent_plan,
            prepared,
            args.runs,
            numerator_label="candidate",
            denominator_label="incumbent",
            warmup=True,
        )
        print(
            f"[akt-model-eval] paired model timing {workload.model_id}: "
            f"candidate={paired['candidate_s'] * 1e3:.3f}ms "
            f"incumbent={paired['incumbent_s'] * 1e3:.3f}ms",
            flush=True,
        )
        model_results.append(
            {
                "model": workload.model_id,
                "description": workload.description,
                "callsites": [callsite_id(workload, call) for call in workload.calls],
                "callsite_cases": {
                    callsite_id(workload, call): call.case_id
                    for call in workload.calls
                },
                "phases": [call.phase for call in workload.calls],
                "selected_plan": selected_plan,
                "incumbent_plan": incumbent_plan,
                "selected_correct": selected_correct,
                "correctness_errors": selected_errors,
                "incumbent_correct": incumbent_correct,
                "incumbent_correctness_errors": incumbent_errors,
                "incumbent_output_hashes": incumbent_output_hashes,
                "selected_output_hashes": selected_output_hashes,
                "certificate": certificate,
                "empirical_dp_certificate": empirical,
                "runtime_events": events,
                **paired,
            }
        )
    summary["models"] = model_results
    empirical_models = [
        {
            "model": model["model"],
            "ok": model["empirical_dp_certificate"].get("ok", False),
            "status": model["empirical_dp_certificate"].get("status", "inconclusive"),
            "panel_sha256": model["empirical_dp_certificate"].get("panel_sha256"),
        }
        for model in model_results
    ]
    summary["empirical_dp_certificate"] = {
        "scope": EMPIRICAL_SCOPE,
        # Testbench: interpret-mode timing noise routinely leaves the bootstrap
        # INCONCLUSIVE at the 1%/2% tolerances. Inconclusive is not falsified —
        # under testbench it is recorded but non-blocking; a FALSIFIED panel
        # (point estimate beyond tolerance) still fails. TPU path unchanged.
        "ok": len(empirical_models) == len(runnable_workloads)
        and (
            all(
                model["status"] != "falsified" for model in empirical_models
            )
            if testbench
            else all(model["ok"] for model in empirical_models)
        ),
        "status": (
            "validated"
            if len(empirical_models) == len(runnable_workloads)
            and all(model["ok"] for model in empirical_models)
            else (
                "falsified"
                if any(model["status"] == "falsified" for model in empirical_models)
                else "inconclusive"
            )
        ),
        "models": empirical_models,
    }
    runtime_evidence = _runtime_evidence(args.capability, model_results, case_by_id)
    summary["runtime_evidence"] = runtime_evidence
    summary["candidate_geomean_s"] = _geomean(
        model["candidate_s"] for model in model_results
    )
    summary["incumbent_geomean_s"] = _geomean(
        model["incumbent_s"] for model in model_results
    )
    summary["paired_ratio_geomean"] = _geomean(model["ratio"] for model in model_results)
    summary["n_choices"] = sum(
        result["valid_configs"] for result in case_search.values()
    )
    summary["n_deferred"] = (
        # testbench: the deferred CALLS (excluded from models above); honest count.
        len(deferred_calls)
        if testbench
        else sum(1 for result in case_search.values() if not result.get("measurements"))
    )
    certificates_ok = all(
        model["certificate"]["certified_additive_optimum"]
        and model["certificate"]["bounded_dp_matches_bruteforce"]
        for model in model_results
    )
    summary["all_correct"] = (
        # testbench relaxes ONLY the hardware and zero-deferred conjuncts; every
        # correctness statement about the RUNNABLE cases/models stays enforced.
        (testbench or (hardware["ok"] and summary["n_deferred"] == 0))
        and all(result["all_configs_correct"] for result in case_search.values())
        and all(model["selected_correct"] for model in model_results)
        and all(model["incumbent_correct"] for model in model_results)
        and certificates_ok
        and summary["empirical_dp_certificate"]["ok"]
        and runtime_evidence["ok"]
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out")
    parser.add_argument("--incumbent-plans")
    parser.add_argument("--capability")
    parser.add_argument("--target-backend", default=TARGET_BACKEND)
    parser.add_argument("--max-configs", type=int, default=100_000)
    parser.add_argument("--empirical-runs", type=int, default=8)
    parser.add_argument("--empirical-max-plans", type=int, default=64)
    parser.add_argument("--empirical-bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--empirical-regret-tolerance", type=float, default=0.01)
    parser.add_argument("--empirical-interaction-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--allow-non-target",
        action="store_true",
        help=(
            "TESTBENCH: proceed off target hardware in Pallas interpret mode; "
            "callsites whose default-config probe raises are deferred and their "
            "model workloads excluded (summary gains testbench/deferred_models; "
            "target_hardware.ok stays False). Set by loop.py --testbench."
        ),
    )
    args = parser.parse_args()
    try:
        summary = evaluate(args)
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        summary = {
            "suite": "three-frozen-models",
            "objective_scope": OBJECTIVE_SCOPE,
            "all_correct": False,
            "target_hardware_ok": False,
            "empirical_dp_certificate": {
                "scope": EMPIRICAL_SCOPE,
                "ok": False,
                "status": "inconclusive",
                "models": [],
            },
            "error": f"{type(error).__name__}: {str(error)[:300]}",
            "models": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    print("AKT_MODEL_EVAL " + json.dumps(summary, default=_json_default, allow_nan=False))
    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, indent=2, default=_json_default, allow_nan=False)
        )
    return summary


if __name__ == "__main__":
    main()
