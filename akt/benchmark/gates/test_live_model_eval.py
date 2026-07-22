import json

import pytest

from akt.benchmark.gates.live_model_eval import evaluate_live_model_candidates, main
from akt.benchmark.live_model_candidates import KIMI_CANDIDATE
from akt.benchmark.test_live_model_candidates import gmm_summary


def _fake_execution(
    policies,
    *,
    score=0.91,
    selected_tps=110.0,
    incumbent_tps=100.0,
    reported_policies=None,
):
    selected = {}
    incumbent = {}
    for metric in KIMI_CANDIDATE.performance:
        selected[metric.name] = {
            "runs": [
                {
                    "completed": metric.num_prompts,
                    "input_throughput": selected_tps,
                    "median_ttft_ms": 9.0,
                    "request_rate": float("inf"),
                }
            ]
        }
        incumbent[metric.name] = {
            "runs": [
                {
                    "completed": metric.num_prompts,
                    "input_throughput": incumbent_tps,
                    "median_ttft_ms": 10.0,
                }
            ]
        }
    return {
        "quality": {"score": score},
        "performance": {"selected": selected, "incumbent": incumbent},
        "server_reported_policies": reported_policies
        or {
            "selected": policies["selected"],
            "incumbent": policies["incumbent"],
        },
        "runtime_coverage": {
            "kind": "mock-server-policy-plus-model-requests",
            "requests_executed": True,
        },
    }


@pytest.mark.parametrize("capability", [None, "unrelated_control"])
def test_not_applicable_actions_do_not_invoke_executor(capability):
    def forbidden(**_kwargs):
        raise AssertionError("executor must not run")

    document = evaluate_live_model_candidates(
        gmm_summary(), capability=capability, executor=forbidden
    )

    assert document["status"] == "skipped"
    result = document["candidates"][0]
    assert result["attempted"] is False
    assert result["eligible"] is False
    assert result["applicability"] == "not_applicable"
    assert result["policy_coverage"]["evaluation_intent"] == "not_applicable"


def test_matching_unchanged_control_is_not_applicable_without_launch():
    def forbidden(**_kwargs):
        raise AssertionError("executor must not run")

    document = evaluate_live_model_candidates(
        gmm_summary(selected_buffer_count=3, incumbent_buffer_count=3),
        capability="gmm_v2_buffer_count",
        executor=forbidden,
    )

    result = document["candidates"][0]
    assert result["applicability"] == "not_applicable"
    assert result["skip_reasons"] == ["capability_policy_unchanged"]


@pytest.mark.parametrize(
    ("summary", "reason"),
    [
        (gmm_summary(backend="gpu"), "backend_mismatch"),
        (gmm_summary(all_correct=False), "synthetic_gate_failed"),
    ],
)
def test_ineligible_direct_action_skips_without_invoking_executor(summary, reason):
    def forbidden(**_kwargs):
        raise AssertionError("executor must not run")

    document = evaluate_live_model_candidates(
        summary, capability="gmm_v2_buffer_count", executor=forbidden
    )

    assert document["status"] == "skipped"
    result = document["candidates"][0]
    assert result["attempted"] is False
    assert result["applicability"] == "direct-capability"
    assert reason in result["skip_reasons"]


def test_direct_action_records_compact_quality_performance_and_attestation():
    calls = []

    def fake_executor(**kwargs):
        calls.append(kwargs)
        return _fake_execution(kwargs["policies"])

    document = evaluate_live_model_candidates(
        gmm_summary(),
        capability="gmm_v2_buffer_count",
        runs=3,
        executor=fake_executor,
    )

    assert len(calls) == 1
    assert calls[0]["runs"] == 3
    assert "incumbent_live" not in calls[0]
    assert document["status"] == "passed"
    result = document["candidates"][0]
    assert result["candidate"]["candidate_id"] == KIMI_CANDIDATE.candidate_id
    assert "static_source_route" not in result["candidate"]
    assert "raw_metrics" not in result
    assert result["policy_coverage"]["capability_applies"] is True
    assert result["policy_coverage"]["evaluation_intent"] == "direct-capability"
    assert len(result["policy_hashes"]["selected"]) == 64
    assert result["metrics"]["quality"]["passed"] is True
    performance = result["metrics"]["performance"]
    assert performance["all_requests_completed"] is True
    assert performance["input_throughput_ratio_geomean"] == pytest.approx(1.1)
    assert result["metrics"]["policy_attestation"]["passed"] is True
    assert result["runtime_coverage"]["requests_executed"] is True


@pytest.mark.parametrize(
    ("execution_overrides", "failure_reason"),
    [
        ({"score": 0.5}, "quality_threshold"),
        ({"selected_tps": 90.0}, "performance_or_completion"),
    ],
)
def test_quality_and_performance_fail_closed(execution_overrides, failure_reason):
    def fake_executor(**kwargs):
        return _fake_execution(kwargs["policies"], **execution_overrides)

    document = evaluate_live_model_candidates(
        gmm_summary(),
        capability="gmm_v2_buffer_count",
        executor=fake_executor,
    )

    assert document["status"] == "failed"
    assert failure_reason in document["candidates"][0]["failure_reasons"]


def test_policy_attestation_fails_closed():
    def fake_executor(**kwargs):
        return _fake_execution(
            kwargs["policies"],
            reported_policies={"selected": {}, "incumbent": {}},
        )

    document = evaluate_live_model_candidates(
        gmm_summary(),
        capability="gmm_v2_buffer_count",
        executor=fake_executor,
    )

    result = document["candidates"][0]
    assert result["status"] == "failed"
    assert "server_policy_attestation" in result["failure_reasons"]


def test_executor_failure_is_failed_not_skipped():
    def broken(**_kwargs):
        raise RuntimeError("server launch failed")

    document = evaluate_live_model_candidates(
        gmm_summary(),
        capability="gmm_v2_buffer_count",
        executor=broken,
    )

    result = document["candidates"][0]
    assert result["status"] == "failed"
    assert result["eligible"] is True
    assert result["attempted"] is True
    assert result["failure_stage"] == "execution"
    assert "server launch failed" in result["error"]


def test_dry_run_resolves_direct_policy_without_invoking_executor():
    def forbidden(**_kwargs):
        raise AssertionError("executor must not run")

    document = evaluate_live_model_candidates(
        gmm_summary(),
        capability="gmm_v2_buffer_count",
        executor=forbidden,
        dry_run=True,
    )

    assert document["status"] == "dry-run"
    result = document["candidates"][0]
    assert result["attempted"] is False
    assert result["policies"]["selected"]["gmm_v2"]["rules"]


def test_cli_always_writes_schema_document_on_bad_input(tmp_path):
    bad_summary = tmp_path / "bad.json"
    output = tmp_path / "result.json"
    bad_summary.write_text("not json")

    returned = main(
        [
            "--synthetic-summary",
            str(bad_summary),
            "--out",
            str(output),
            "--runs",
            "2",
        ]
    )
    written = json.loads(output.read_text())

    assert returned["status"] == "failed"
    assert written["schema_version"] == "akt.live-model-eval.v1"
    assert written["status"] == "failed"
    assert written["inputs"] == {"synthetic_summary": str(bad_summary.resolve())}
