"""Pure contracts for AKT's conditional Kimi-Linear live-model gate.

The module intentionally imports no JAX, model, HTTP, YAML, or serving code.  It
decides whether the one supported Kimi checkpoint is applicable and eligible
before the evaluator can import those heavy dependencies or resolve a checkpoint.

Kimi is direct evidence only for a changed, capability-elevated GMM-v2 programmer
control.  Initialization, rebaseline, unrelated actions, and unchanged controls
produce an explicit no-attempt ``not_applicable`` record.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

LIVE_MODEL_SCHEMA_VERSION = "akt.live-model-eval.v1"
KIMI_MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
KIMI_CANDIDATE_ID = "kimi-linear-48b-a3b-gmm-v2"
KIMI_PROFILE_PATH = "test/srt/nightly/launch_profiles/kimi-linear-v6e-4.yaml"
KIMI_ROUTE_ID = "kimi-linear/epmoe/megablox-gmm-v2"
KIMI_POLICY_FAMILY = "gmm_v2"
KIMI_POLICY_SOURCE_MODEL = "tiny-dense-serving"
KIMI_POLICY_SOURCE_CALLSITE = "tiny-dense-serving/dense-projection"
KIMI_POLICY_SOURCE_CASE = "gmm_v2:m512_k1024_n1024_g8"


@dataclass(frozen=True)
class QualityMetric:
    name: str
    dataset: str
    threshold: float
    eval_batch_size: int
    generation_config: Mapping[str, Any]
    limit: int | None = None


@dataclass(frozen=True)
class PerformanceMetric:
    name: str
    input_len: int
    output_len: int
    num_prompts: int
    max_concurrency: int
    seed: int = 42


@dataclass(frozen=True)
class KimiLiveModel:
    """The single checkpoint/profile supported by the live gate."""

    candidate_id: str
    model_id: str
    launch_profile: str
    target_backend: str
    min_devices: int
    tensor_parallel_size: int
    route_id: str
    quality: QualityMetric
    performance: tuple[PerformanceMetric, ...]
    min_input_throughput_ratio: float


KIMI_CANDIDATE = KimiLiveModel(
    candidate_id=KIMI_CANDIDATE_ID,
    model_id=KIMI_MODEL_ID,
    launch_profile=KIMI_PROFILE_PATH,
    target_backend="tpu",
    min_devices=4,
    tensor_parallel_size=4,
    route_id=KIMI_ROUTE_ID,
    quality=QualityMetric(
        name="kimi-linear-hybrid",
        dataset="gsm8k",
        threshold=0.89,
        eval_batch_size=64,
        generation_config={"temperature": 0.0, "max_tokens": 2048},
    ),
    performance=(
        PerformanceMetric(
            name="kimi-linear-long-prefill-c1-i3072-o1",
            input_len=3072,
            output_len=1,
            num_prompts=8,
            max_concurrency=1,
        ),
        PerformanceMetric(
            name="kimi-linear-packed-prefill-c8-i512-o1",
            input_len=512,
            output_len=1,
            num_prompts=32,
            max_concurrency=8,
        ),
    ),
    min_input_throughput_ratio=0.98,
)


def json_default(value: Any) -> Any:
    """Convert common metric scalar types without importing their libraries."""

    if is_dataclass(value):
        return asdict(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def json_safe(value: Any) -> Any:
    """Recursively convert metrics to strict JSON, including non-finite sentinels."""

    if is_dataclass(value):
        value = asdict(value)
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe(item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item) for item in value), key=repr)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isnan(numeric):
            return "NaN"
        if math.isinf(numeric):
            return "Infinity" if numeric > 0 else "-Infinity"
        return value if isinstance(value, int) else numeric
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_default,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def candidate_descriptor() -> dict[str, Any]:
    """Return only stable identity and launch fields, not the whole contract."""

    return {
        "candidate_id": KIMI_CANDIDATE.candidate_id,
        "model_id": KIMI_CANDIDATE.model_id,
        "launch_profile": KIMI_CANDIDATE.launch_profile,
        "target_backend": KIMI_CANDIDATE.target_backend,
        "tensor_parallel_size": KIMI_CANDIDATE.tensor_parallel_size,
        "route_id": KIMI_CANDIDATE.route_id,
    }


def candidate_eligibility(synthetic_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the synthetic gate proved compiled single-host TPU4."""

    reasons: list[dict[str, str]] = []
    if not isinstance(synthetic_summary, Mapping):
        synthetic_summary = {}
        reasons.append(
            {"code": "invalid_synthetic_summary", "detail": "summary is not an object"}
        )

    if synthetic_summary.get("all_correct") is not True:
        reasons.append(
            {
                "code": "synthetic_gate_failed",
                "detail": "synthetic all_correct is not true",
            }
        )
    if synthetic_summary.get("n_deferred") != 0:
        reasons.append(
            {
                "code": "synthetic_cases_deferred",
                "detail": (
                    f"synthetic n_deferred is {synthetic_summary.get('n_deferred')!r}, "
                    "not 0"
                ),
            }
        )

    hardware = synthetic_summary.get("target_hardware")
    if not isinstance(hardware, Mapping):
        hardware = {}
        reasons.append(
            {"code": "missing_hardware_record", "detail": "target_hardware is absent"}
        )
    if (
        synthetic_summary.get("target_hardware_ok") is not True
        or hardware.get("ok") is not True
    ):
        reasons.append(
            {
                "code": "target_hardware_not_verified",
                "detail": "both hardware verification flags must be true",
            }
        )
    backend = hardware.get("backend")
    if backend != KIMI_CANDIDATE.target_backend:
        reasons.append(
            {
                "code": "backend_mismatch",
                "detail": (
                    f"requires {KIMI_CANDIDATE.target_backend!r}, observed {backend!r}"
                ),
            }
        )
    if hardware.get("pallas_interpret") is not False:
        reasons.append(
            {
                "code": "interpret_mode",
                "detail": "live evaluation requires compiled Pallas execution",
            }
        )

    raw_devices = hardware.get("devices")
    devices = (
        raw_devices
        if isinstance(raw_devices, Sequence) and not isinstance(raw_devices, str)
        else []
    )
    tpu_devices = [
        device
        for device in devices
        if isinstance(device, Mapping)
        and device.get("platform") == KIMI_CANDIDATE.target_backend
    ]
    local_device_count = hardware.get("local_device_count", len(tpu_devices))
    process_count = hardware.get("process_count", 1)
    if type(process_count) is not int or process_count != 1:
        reasons.append(
            {
                "code": "not_single_host",
                "detail": f"requires process_count=1, observed {process_count!r}",
            }
        )
    if (
        type(local_device_count) is not int
        or local_device_count < KIMI_CANDIDATE.min_devices
        or len(tpu_devices) < KIMI_CANDIDATE.min_devices
    ):
        reasons.append(
            {
                "code": "insufficient_tpu_devices",
                "detail": (
                    f"requires {KIMI_CANDIDATE.min_devices} local TPU devices, "
                    f"observed count={local_device_count!r}, inventory={len(tpu_devices)}"
                ),
            }
        )

    device_indexes = [
        device.get("id")
        for device in tpu_devices[: KIMI_CANDIDATE.tensor_parallel_size]
        if type(device.get("id")) is int
    ]
    if len(device_indexes) != KIMI_CANDIDATE.tensor_parallel_size:
        reasons.append(
            {
                "code": "missing_device_indexes",
                "detail": "TPU inventory lacks a complete integer device selection",
            }
        )

    return {
        "eligible": not reasons,
        "reason_codes": [reason["code"] for reason in reasons],
        "reasons": reasons,
        "requirements": {
            "backend": KIMI_CANDIDATE.target_backend,
            "pallas_interpret": False,
            "single_host": True,
            "min_devices": KIMI_CANDIDATE.min_devices,
            "tensor_parallel_size": KIMI_CANDIDATE.tensor_parallel_size,
        },
        "observed": {
            "backend": backend,
            "pallas_interpret": hardware.get("pallas_interpret"),
            "process_count": process_count,
            "local_device_count": local_device_count,
            "tpu_device_count": len(tpu_devices),
            "device_indexes": device_indexes,
            "hardware_fingerprint": hardware.get("fingerprint"),
        },
    }


def resolve_kimi_policy(
    synthetic_summary: Mapping[str, Any], capability: str | None
) -> dict[str, Any]:
    """Resolve the one direct Kimi route, or return explicit non-applicability.

    Only controls both elevated by ``capability`` and exposed as dotted GMM-v2
    programmer controls enter the server policy.  A direct run requires at least
    one such control to differ between the selected and incumbent plans.
    """

    if not capability:
        return {
            "applicable": False,
            "reason": "no_capability",
            "capability": capability,
        }
    case_search = synthetic_summary.get("case_search")
    case = (
        case_search.get(KIMI_POLICY_SOURCE_CASE)
        if isinstance(case_search, Mapping)
        else None
    )
    knobs = case.get("knobs") if isinstance(case, Mapping) else None
    if not isinstance(knobs, list):
        return {
            "applicable": False,
            "reason": "no_matching_gmm_v2_control",
            "capability": capability,
        }

    prefix = KIMI_POLICY_FAMILY + "."
    matching = [
        knob
        for knob in knobs
        if isinstance(knob, Mapping)
        and knob.get("elevated_by") == capability
        and isinstance(knob.get("name"), str)
        and isinstance(knob.get("programmer_control"), str)
        and knob["programmer_control"].startswith(prefix)
        and len(knob["programmer_control"]) > len(prefix)
    ]
    if not matching:
        return {
            "applicable": False,
            "reason": "no_matching_gmm_v2_control",
            "capability": capability,
        }

    models = synthetic_summary.get("models")
    if not isinstance(models, list):
        raise ValueError("synthetic summary has no model result list")
    source_models = [
        model
        for model in models
        if isinstance(model, Mapping) and model.get("model") == KIMI_POLICY_SOURCE_MODEL
    ]
    if len(source_models) != 1:
        raise ValueError(
            f"expected exactly one {KIMI_POLICY_SOURCE_MODEL!r} model result; "
            f"got {len(source_models)}"
        )
    source_model = source_models[0]
    plans: dict[str, Mapping[str, Any]] = {}
    for label, field_name in (
        ("selected", "selected_plan"),
        ("incumbent", "incumbent_plan"),
    ):
        raw_plan = source_model.get(field_name)
        if not isinstance(raw_plan, Mapping):
            raise ValueError(f"source model has no {field_name}")
        config = raw_plan.get(KIMI_POLICY_SOURCE_CALLSITE)
        if not isinstance(config, Mapping):
            raise ValueError(f"{field_name} is missing {KIMI_POLICY_SOURCE_CALLSITE!r}")
        plans[label] = config

    controls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for knob in matching:
        knob_name = knob["name"]
        programmer_control = knob["programmer_control"]
        control = programmer_control[len(prefix) :]
        if programmer_control in seen:
            raise ValueError(f"duplicate programmer control {programmer_control!r}")
        seen.add(programmer_control)
        for label, config in plans.items():
            if knob_name not in config:
                raise ValueError(f"{label} source plan is missing knob {knob_name!r}")
        controls.append(
            {
                "knob": knob_name,
                "programmer_control": programmer_control,
                "default": knob.get("default"),
                "selected": plans["selected"][knob_name],
                "incumbent": plans["incumbent"][knob_name],
                "changed": (
                    plans["selected"][knob_name] != plans["incumbent"][knob_name]
                ),
            }
        )

    if not any(control["changed"] for control in controls):
        return {
            "applicable": False,
            "reason": "capability_policy_unchanged",
            "capability": capability,
            "controls": controls,
        }

    def policy(label: str) -> dict[str, Any]:
        return {
            KIMI_POLICY_FAMILY: {
                "default": {},
                "rules": [
                    {
                        "when": {},
                        "set": {
                            control["programmer_control"][len(prefix) :]: control[label]
                            for control in controls
                        },
                    }
                ],
            }
        }

    selected = policy("selected")
    incumbent = policy("incumbent")
    return {
        "applicable": True,
        "capability": capability,
        "selected": selected,
        "incumbent": incumbent,
        "hashes": {
            "selected": stable_json_hash(selected),
            "incumbent": stable_json_hash(incumbent),
        },
        "source": {
            "case": KIMI_POLICY_SOURCE_CASE,
            "callsite": KIMI_POLICY_SOURCE_CALLSITE,
            "route_id": KIMI_ROUTE_ID,
        },
        "coverage": {
            "only_programmer_controls": True,
            "controls": controls,
            "capability": capability,
            "capability_applies": True,
            "evaluation_intent": "direct-capability",
        },
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _point_runs(point: Any) -> list[Mapping[str, Any]]:
    if not isinstance(point, Mapping):
        return []
    runs = point.get("runs")
    if isinstance(runs, list):
        return [run for run in runs if isinstance(run, Mapping)]
    return [point] if "completed" in point else []


def _point_metric(point: Any, key: str) -> float | None:
    if isinstance(point, Mapping):
        aggregate = point.get("aggregate")
        if isinstance(aggregate, Mapping):
            value = _finite_number(aggregate.get(key))
            if value is not None:
                return value
    values = [
        value
        for run in _point_runs(point)
        if (value := _finite_number(run.get(key))) is not None
    ]
    return statistics.median(values) if values else None


def _geomean(values: Sequence[float]) -> float | None:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_performance(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the mandatory same-round selected/incumbent comparison."""

    performance = execution.get("performance")
    performance = performance if isinstance(performance, Mapping) else {}
    selected = performance.get("selected")
    incumbent = performance.get("incumbent")
    selected = selected if isinstance(selected, Mapping) else {}
    incumbent = incumbent if isinstance(incumbent, Mapping) else {}

    points: dict[str, Any] = {}
    ratios: list[float] = []
    all_complete = True
    for spec in KIMI_CANDIDATE.performance:
        selected_point = selected.get(spec.name)
        incumbent_point = incumbent.get(spec.name)
        selected_runs = _point_runs(selected_point)
        incumbent_runs = _point_runs(incumbent_point)
        selected_complete = bool(selected_runs) and all(
            run.get("completed") == spec.num_prompts for run in selected_runs
        )
        incumbent_complete = bool(incumbent_runs) and all(
            run.get("completed") == spec.num_prompts for run in incumbent_runs
        )
        all_complete = all_complete and selected_complete and incumbent_complete

        selected_tps = _point_metric(selected_point, "input_throughput")
        incumbent_tps = _point_metric(incumbent_point, "input_throughput")
        ratio = (
            selected_tps / incumbent_tps
            if selected_tps is not None
            and incumbent_tps is not None
            and incumbent_tps > 0
            else None
        )
        if ratio is not None and ratio > 0:
            ratios.append(ratio)
        points[spec.name] = {
            "expected_prompts": spec.num_prompts,
            "selected_runs": len(selected_runs),
            "incumbent_runs": len(incumbent_runs),
            "selected_complete": selected_complete,
            "incumbent_complete": incumbent_complete,
            "selected_input_throughput": selected_tps,
            "incumbent_input_throughput": incumbent_tps,
            "input_throughput_ratio": ratio,
            "selected_median_ttft_ms": _point_metric(selected_point, "median_ttft_ms"),
            "incumbent_median_ttft_ms": _point_metric(
                incumbent_point, "median_ttft_ms"
            ),
        }

    ratio_geomean = (
        _geomean(ratios) if len(ratios) == len(KIMI_CANDIDATE.performance) else None
    )
    return {
        "points": points,
        "all_requests_completed": all_complete,
        "input_throughput_ratio_geomean": ratio_geomean,
        "minimum_input_throughput_ratio": KIMI_CANDIDATE.min_input_throughput_ratio,
        "passed": bool(
            all_complete
            and ratio_geomean is not None
            and ratio_geomean >= KIMI_CANDIDATE.min_input_throughput_ratio
        ),
    }


def _quality_summary(execution: Mapping[str, Any]) -> dict[str, Any]:
    quality = execution.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    score = _finite_number(quality.get("score"))
    return {
        "case": KIMI_CANDIDATE.quality.name,
        "dataset": KIMI_CANDIDATE.quality.dataset,
        "score": score,
        "threshold": KIMI_CANDIDATE.quality.threshold,
        "passed": bool(score is not None and score >= KIMI_CANDIDATE.quality.threshold),
    }


def _policy_attestation(
    execution: Mapping[str, Any], policy_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    reported = execution.get("server_reported_policies")
    reported = reported if isinstance(reported, Mapping) else {}
    selected_ok = reported.get("selected") == policy_bundle.get("selected")
    incumbent_ok = reported.get("incumbent") == policy_bundle.get("incumbent")
    return {
        "selected_matches": selected_ok,
        "incumbent_matches": incumbent_ok,
        "passed": selected_ok and incumbent_ok,
    }


def build_candidate_result(
    eligibility: Mapping[str, Any],
    policy_bundle: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the compact, fail-closed result for one direct live attempt."""

    if not isinstance(execution, Mapping):
        raise ValueError("live candidate executor must return an object")
    quality = _quality_summary(execution)
    performance = summarize_performance(execution)
    attestation = _policy_attestation(execution, policy_bundle)
    failure_reasons = []
    if not quality["passed"]:
        failure_reasons.append("quality_threshold")
    if not performance["passed"]:
        failure_reasons.append("performance_or_completion")
    if not attestation["passed"]:
        failure_reasons.append("server_policy_attestation")
    passed = not failure_reasons
    return {
        "schema_version": LIVE_MODEL_SCHEMA_VERSION,
        "candidate": candidate_descriptor(),
        "status": "passed" if passed else "failed",
        "attempted": True,
        "eligible": True,
        "applicability": "direct-capability",
        "eligibility": dict(eligibility),
        "policies": {
            "selected": policy_bundle["selected"],
            "incumbent": policy_bundle["incumbent"],
        },
        "policy_hashes": dict(policy_bundle["hashes"]),
        "policy_source": dict(policy_bundle["source"]),
        "policy_coverage": dict(policy_bundle["coverage"]),
        "metrics": {
            "quality": quality,
            "performance": performance,
            "policy_attestation": attestation,
        },
        "runtime_coverage": json_safe(execution.get("runtime_coverage", {})),
        "failure_reasons": failure_reasons,
    }


def skipped_candidate_result(
    eligibility: Mapping[str, Any], policy_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a clean no-attempt result for an applicable but ineligible route."""

    return {
        "schema_version": LIVE_MODEL_SCHEMA_VERSION,
        "candidate": candidate_descriptor(),
        "status": "skipped",
        "attempted": False,
        "eligible": False,
        "applicability": "direct-capability",
        "eligibility": dict(eligibility),
        "policy_coverage": dict(policy_bundle["coverage"]),
        "skip_reasons": list(eligibility.get("reason_codes", [])),
    }


def not_applicable_result(resolution: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit no-launch record compatible with the existing loop."""

    reason = str(resolution.get("reason") or "capability_not_applicable")
    eligibility = {
        "eligible": False,
        "reason_codes": [reason],
        "reasons": [
            {
                "code": reason,
                "detail": "the action does not change an elevated GMM-v2 control",
            }
        ],
    }
    return {
        "schema_version": LIVE_MODEL_SCHEMA_VERSION,
        "candidate": candidate_descriptor(),
        "status": "skipped",
        "attempted": False,
        "eligible": False,
        "applicability": "not_applicable",
        "eligibility": eligibility,
        "policy_coverage": {
            "capability": resolution.get("capability"),
            "capability_applies": False,
            "evaluation_intent": "not_applicable",
        },
        "skip_reasons": [reason],
    }


__all__ = [
    "KIMI_CANDIDATE",
    "KIMI_CANDIDATE_ID",
    "KIMI_MODEL_ID",
    "KIMI_POLICY_FAMILY",
    "KIMI_POLICY_SOURCE_CALLSITE",
    "KIMI_POLICY_SOURCE_CASE",
    "KIMI_POLICY_SOURCE_MODEL",
    "KIMI_PROFILE_PATH",
    "KIMI_ROUTE_ID",
    "LIVE_MODEL_SCHEMA_VERSION",
    "KimiLiveModel",
    "PerformanceMetric",
    "QualityMetric",
    "build_candidate_result",
    "candidate_descriptor",
    "candidate_eligibility",
    "json_default",
    "json_safe",
    "not_applicable_result",
    "resolve_kimi_policy",
    "skipped_candidate_result",
    "stable_json_hash",
    "summarize_performance",
]
