"""Conditional Kimi-Linear acceptance gate for AKT.

The evaluator launches the one Kimi checkpoint only when the current capability
changes an elevated GMM-v2 programmer control and the synthetic result proves a
compiled single-host TPU4 target.  All other actions emit a no-attempt record.

CLI contract used by the evolution loop::

    python akt/benchmark/gates/live_model_eval.py \
      --synthetic-summary model_eval.json --out live_model_eval.json \
      [--capability NAME] [--runs N]
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import statistics
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from akt.benchmark.live_model_candidates import (  # noqa: E402
    KIMI_CANDIDATE,
    LIVE_MODEL_SCHEMA_VERSION,
    build_candidate_result,
    candidate_descriptor,
    candidate_eligibility,
    json_default,
    json_safe,
    not_applicable_result,
    resolve_kimi_policy,
    skipped_candidate_result,
    stable_json_hash,
)

SUITE_NAME = "akt-live-model-candidates"
Executor = Callable[..., Mapping[str, Any]]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _failed_candidate(
    eligibility: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
    policy_bundle: Mapping[str, Any] | None = None,
    attempted: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": LIVE_MODEL_SCHEMA_VERSION,
        "candidate": candidate_descriptor(),
        "status": "failed",
        "attempted": attempted,
        "eligible": bool(eligibility.get("eligible")),
        "applicability": "direct-capability",
        "eligibility": dict(eligibility),
        "failure_stage": stage,
        "error": f"{type(error).__name__}: {str(error)[:1000]}",
    }
    if policy_bundle is not None:
        result.update(
            {
                "policies": {
                    "selected": policy_bundle.get("selected"),
                    "incumbent": policy_bundle.get("incumbent"),
                },
                "policy_hashes": dict(policy_bundle.get("hashes", {})),
                "policy_source": dict(policy_bundle.get("source", {})),
                "policy_coverage": dict(policy_bundle.get("coverage", {})),
            }
        )
    return result


def _dry_run_candidate(
    eligibility: Mapping[str, Any], policy_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": LIVE_MODEL_SCHEMA_VERSION,
        "candidate": candidate_descriptor(),
        "status": "dry-run",
        "attempted": False,
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
    }


def evaluate_live_model_candidates(
    synthetic_summary: Mapping[str, Any],
    *,
    capability: str | None = None,
    runs: int = 1,
    executor: Executor | None = None,
    repo: str | Path = REPO,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate the one conditional Kimi route and return a schema document.

    The injectable ``executor`` is called only for a changed direct GMM-v2 policy
    on eligible hardware.  It must measure selected and incumbent policies in the
    same invocation; no prior live result is accepted as a baseline.
    """

    if type(runs) is not int or runs <= 0:
        raise ValueError(f"runs must be a positive integer; got {runs!r}")
    if not isinstance(synthetic_summary, Mapping):
        raise ValueError("synthetic summary must be an object")

    try:
        policy_bundle = resolve_kimi_policy(synthetic_summary, capability)
    except Exception as error:  # noqa: BLE001 - policy mapping fails closed
        eligibility = candidate_eligibility(synthetic_summary)
        candidate_result = _failed_candidate(
            eligibility,
            stage="policy_mapping",
            error=error,
            attempted=False,
        )
    else:
        if policy_bundle["applicable"] is not True:
            candidate_result = not_applicable_result(policy_bundle)
        else:
            eligibility = candidate_eligibility(synthetic_summary)
            if not eligibility["eligible"]:
                candidate_result = skipped_candidate_result(eligibility, policy_bundle)
            elif dry_run:
                candidate_result = _dry_run_candidate(eligibility, policy_bundle)
            else:
                selected_executor = executor or _execute_kimi_candidate
                try:
                    execution = selected_executor(
                        policies=policy_bundle,
                        eligibility=eligibility,
                        runs=runs,
                        repo=Path(repo),
                    )
                    candidate_result = build_candidate_result(
                        eligibility, policy_bundle, execution
                    )
                except Exception as error:  # noqa: BLE001 - live failures fail closed
                    traceback.print_exc()
                    candidate_result = _failed_candidate(
                        eligibility,
                        stage="execution",
                        error=error,
                        policy_bundle=policy_bundle,
                        attempted=True,
                    )

    status = candidate_result["status"]
    return {
        "schema_version": LIVE_MODEL_SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "status": status,
        "all_passed": status == "passed",
        "capability": capability,
        "runs": runs,
        "dry_run": dry_run,
        "synthetic_summary_sha256": stable_json_hash(synthetic_summary),
        "candidates": [candidate_result],
        "timestamp": _utc_now(),
    }


def _remove_option(
    args: Sequence[str], option: str, *, variable_arity: bool = False
) -> list[str]:
    """Remove a server option so the live policy has one source of truth."""

    result: list[str] = []
    index = 0
    while index < len(args):
        token = str(args[index])
        if token.startswith(option + "="):
            index += 1
            continue
        if token != option:
            result.append(token)
            index += 1
            continue
        index += 1
        if variable_arity:
            while index < len(args) and not str(args[index]).startswith("--"):
                index += 1
        elif index < len(args):
            index += 1
    return result


def _server_args_with_policy(
    base_args: Sequence[str], policy: Mapping[str, Any], device_indexes: Sequence[int]
) -> list[str]:
    args = _remove_option(base_args, "--kernel-control-config")
    args = _remove_option(args, "--device-indexes", variable_arity=True)
    args.extend(["--device-indexes", *(str(index) for index in device_indexes)])
    args.extend(
        [
            "--kernel-control-config",
            json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False),
        ]
    )
    return args


def _numeric_medians(runs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = set().union(*(run.keys() for run in runs)) if runs else set()
    aggregate: dict[str, float] = {}
    for key in sorted(keys):
        values = []
        for run in runs:
            value = run.get(key)
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                break
            value = float(value)
            if not math.isfinite(value):
                break
            values.append(value)
        else:
            if values:
                aggregate[key] = statistics.median(values)
    return aggregate


def _execute_kimi_candidate(
    *,
    policies: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    runs: int,
    repo: Path,
) -> Mapping[str, Any]:
    """Launch selected and incumbent Kimi policies in the same evaluation round."""

    if policies["selected"] == policies["incumbent"]:
        raise ValueError("direct Kimi comparison requires distinct policies")

    python_dir = repo / "python"
    test_srt = repo / "test/srt"
    nightly_dir = test_srt / "nightly"
    single_host_dir = nightly_dir / "single_host"
    for path in (repo, python_dir, test_srt, nightly_dir, single_host_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Lazy by design: these can initialize serving/model runtimes or downloads.
    import requests  # noqa: PLC0415
    from accuracy_case_runner import profile_server_spec  # noqa: PLC0415
    from cases import AccuracyCase, PerfCase  # noqa: PLC0415
    from drivers import run_benchmark_for_case, run_eval_for_case  # noqa: PLC0415
    from profiles import load_profile  # noqa: PLC0415

    from sgl_jax.srt.utils import kill_process_tree  # noqa: PLC0415
    from sgl_jax.test.test_utils import (  # noqa: PLC0415
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        popen_launch_server,
    )

    profile = load_profile(repo / KIMI_CANDIDATE.launch_profile)
    if profile.model_path != KIMI_CANDIDATE.model_id:
        raise ValueError(
            f"launch profile model {profile.model_path!r} does not match "
            f"{KIMI_CANDIDATE.model_id!r}"
        )
    if profile.tp_size != KIMI_CANDIDATE.tensor_parallel_size:
        raise ValueError(
            f"launch profile TP={profile.tp_size} does not match candidate "
            f"TP={KIMI_CANDIDATE.tensor_parallel_size}"
        )
    spec = profile_server_spec(profile)
    observed = eligibility.get("observed")
    device_indexes = (
        observed.get("device_indexes") if isinstance(observed, Mapping) else []
    )
    if len(device_indexes) != KIMI_CANDIDATE.tensor_parallel_size:
        raise ValueError("eligible result has no complete TPU device index selection")

    quality_case = AccuracyCase(
        name=KIMI_CANDIDATE.quality.name,
        dataset=KIMI_CANDIDATE.quality.dataset,
        model_id=KIMI_CANDIDATE.model_id,
        eval_batch_size=KIMI_CANDIDATE.quality.eval_batch_size,
        generation_config=dict(KIMI_CANDIDATE.quality.generation_config),
        limit=KIMI_CANDIDATE.quality.limit,
        score_threshold=KIMI_CANDIDATE.quality.threshold,
    )

    def run_variant(
        label: str, policy: Mapping[str, Any], *, run_quality: bool
    ) -> dict[str, Any]:
        other_args = _server_args_with_policy(
            spec["other_args"], policy, device_indexes
        )
        process = None
        try:
            process = popen_launch_server(
                spec["model"],
                spec["base_url"],
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                device="tpu",
            )
            response = requests.get(spec["base_url"] + "/get_server_info", timeout=30)
            response.raise_for_status()
            server_info = response.json()
            reported_policy = server_info.get("kernel_control_config")
            if reported_policy != policy:
                raise RuntimeError(
                    f"{label} server policy mismatch: expected={policy!r}, "
                    f"reported={reported_policy!r}"
                )

            performance: dict[str, Any] = {}
            for metric in KIMI_CANDIDATE.performance:
                case = PerfCase(
                    name=metric.name,
                    input_len=metric.input_len,
                    output_len=metric.output_len,
                    num_prompts=metric.num_prompts,
                    max_concurrency=metric.max_concurrency,
                    seed=metric.seed,
                )
                raw_runs = [
                    run_benchmark_for_case(
                        case,
                        spec["base_url"],
                        spec["model"],
                        profile=False,
                    )
                    for _ in range(runs)
                ]
                performance[metric.name] = {
                    "runs": raw_runs,
                    "aggregate": _numeric_medians(raw_runs),
                }

            quality = None
            if run_quality:
                raw_quality, started_at, finished_at = run_eval_for_case(
                    quality_case, spec["base_url"]
                )
                quality = {
                    "score": (
                        raw_quality.get("score")
                        if isinstance(raw_quality, Mapping)
                        else None
                    ),
                    "duration_seconds": finished_at - started_at,
                }
            return {
                "performance": performance,
                "quality": quality,
                "server_reported_policy": reported_policy,
            }
        finally:
            if process is not None:
                kill_process_tree(process.pid)
                process.wait(timeout=30)

    selected_run = run_variant("selected", policies["selected"], run_quality=True)
    incumbent_run = run_variant("incumbent", policies["incumbent"], run_quality=False)

    def completion(run: Mapping[str, Any]) -> dict[str, list[Any]]:
        completed: dict[str, list[Any]] = {}
        performance = run.get("performance")
        performance = performance if isinstance(performance, Mapping) else {}
        for metric in KIMI_CANDIDATE.performance:
            point = performance.get(metric.name)
            raw_runs = point.get("runs") if isinstance(point, Mapping) else []
            completed[metric.name] = [
                raw.get("completed") for raw in raw_runs if isinstance(raw, Mapping)
            ]
        return completed

    return {
        "quality": selected_run["quality"],
        "performance": {
            "selected": selected_run["performance"],
            "incumbent": incumbent_run["performance"],
        },
        "server_reported_policies": {
            "selected": selected_run["server_reported_policy"],
            "incumbent": incumbent_run["server_reported_policy"],
        },
        "runtime_coverage": {
            "kind": "server-policy-attestation-plus-completed-model-requests",
            "route_id": KIMI_CANDIDATE.route_id,
            "comparison": "same-round-selected-then-incumbent",
            "completed": {
                "selected": completion(selected_run),
                "incumbent": completion(incumbent_run),
            },
            "quality_dataset": KIMI_CANDIDATE.quality.dataset,
            "claim_limit": (
                "Server policy plus end-to-end requests; not a per-kernel trace."
            ),
        },
    }


def _failure_document(
    error: BaseException, *, capability: str | None, runs: int
) -> dict[str, Any]:
    return {
        "schema_version": LIVE_MODEL_SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "status": "failed",
        "all_passed": False,
        "capability": capability,
        "runs": runs,
        "candidates": [],
        "failure_stage": "input_or_orchestration",
        "error": f"{type(error).__name__}: {str(error)[:1000]}",
        "timestamp": _utc_now(),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--capability")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve applicability, eligibility, and policy without a model launch",
    )
    args = parser.parse_args(argv)

    try:
        synthetic_summary = json.loads(Path(args.synthetic_summary).read_text())
        document = evaluate_live_model_candidates(
            synthetic_summary,
            capability=args.capability,
            runs=args.runs,
            dry_run=args.dry_run,
        )
    except Exception as error:  # noqa: BLE001 - always emit a machine result
        traceback.print_exc()
        document = _failure_document(error, capability=args.capability, runs=args.runs)

    document["inputs"] = {
        "synthetic_summary": str(Path(args.synthetic_summary).resolve())
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_document = json_safe(document)
    out_path.write_text(
        json.dumps(
            serializable_document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n"
    )
    print(
        "AKT_LIVE_MODEL_EVAL "
        + json.dumps(serializable_document, allow_nan=False, default=json_default)
    )
    return serializable_document


if __name__ == "__main__":
    main()
