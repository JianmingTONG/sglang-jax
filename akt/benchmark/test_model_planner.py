import json
import math
import sys
import types
from types import SimpleNamespace

import pytest

from akt.benchmark import adapter
from akt.benchmark.gates import model_eval
from akt.benchmark.gates.dp_verify import verify_models
from akt.benchmark.runtime import (
    _report_probed_backend_execution,
    capture_runtime_events,
    runtime_scope,
)
from akt.benchmark.model_workloads import (
    MODEL_WORKLOADS,
    callsite_inventory,
    callsites_for_kernel_ids,
    contract_fingerprint,
    validate_workloads,
)
from akt.benchmark.suites import EXPECTED_CASES
from akt.core.search.model_dp import (
    PlanCandidate,
    empirical_plan_panel,
    measured_graph_fingerprint,
    search_bruteforce,
    search_dp,
)


def _expected_cases():
    return {
        case_id
        for family_cases in EXPECTED_CASES.values()
        for case_id in family_cases
    }


def test_three_frozen_models_cover_every_kernel_case_once():
    validate_workloads(_expected_cases())
    assert len(MODEL_WORKLOADS) == 3
    assert len(callsite_inventory()) == len(_expected_cases())
    assert len(contract_fingerprint()) == 64


def test_gap_kernel_mapping_reaches_model_callsites():
    sites = callsites_for_kernel_ids(["gla", "kv_cache"])
    assert any(site.startswith("tiny-linear-serving/") for site in sites)
    assert any(site.startswith("tiny-dense-serving/") for site in sites)
    assert any(site.startswith("tiny-moe-serving/") for site in sites)


def test_dp_matches_brute_force_for_all_three_bounded_models():
    reports = verify_models()
    assert {report["model"] for report in reports} == {
        model.model_id for model in MODEL_WORKLOADS
    }
    assert all(report["bounded_dp_matches_bruteforce"] for report in reports)


def test_additive_dp_matches_cartesian_brute_force():
    stages = [
        [
            PlanCandidate.from_dict("a", {"variant": 0}, 2.0),
            PlanCandidate.from_dict("a", {"variant": 1}, 1.0),
        ],
        [
            PlanCandidate.from_dict("b", {"variant": 0}, 4.0),
            PlanCandidate.from_dict("b", {"variant": 1}, 3.0),
        ],
    ]
    dp = search_dp(stages)
    brute = search_bruteforce(stages)
    assert dp.cost_s == brute.cost_s == 4.0
    assert dp.plan() == brute.plan()


def test_empty_planner_stage_fails_closed():
    with pytest.raises(ValueError, match="no candidates"):
        search_dp([[]])


def test_empirical_panel_fails_closed_without_dropping_selected_plan():
    stages = [
        [
            PlanCandidate.from_dict(f"stage-{index}", {"variant": 0}, 1.0),
            PlanCandidate.from_dict(f"stage-{index}", {"variant": 1}, 2.0),
        ]
        for index in range(7)
    ]
    panel = empirical_plan_panel(stages, max_plans=64)

    assert not panel.complete
    assert panel.declared_combinations == 128
    assert panel.plans == (panel.selected,)
    assert "exceeding the fail-closed limit" in panel.error


def test_measured_graph_fingerprint_includes_cost_samples_and_protocol():
    first = [[PlanCandidate.from_dict("stage", {"variant": 0}, 1.0)]]
    second = [[PlanCandidate.from_dict("stage", {"variant": 0}, 1.1)]]
    base = measured_graph_fingerprint(
        first,
        timing_protocol={"runs": 3},
        timing_samples=[[1.0, 1.1, 0.9]],
    )

    assert base != measured_graph_fingerprint(
        second,
        timing_protocol={"runs": 3},
        timing_samples=[[1.0, 1.1, 0.9]],
    )
    assert base != measured_graph_fingerprint(
        first,
        timing_protocol={"runs": 5},
        timing_samples=[[1.0, 1.1, 0.9]],
    )
    assert base != measured_graph_fingerprint(
        first,
        timing_protocol={"runs": 3},
        timing_samples=[[1.0, 1.2, 0.9]],
    )


def _two_factor_panel_and_records(ratios):
    stages = [
        [
            PlanCandidate.from_dict("a", {"variant": 0}, 1.0),
            PlanCandidate.from_dict("a", {"variant": 1}, 2.0),
        ],
        [
            PlanCandidate.from_dict("b", {"variant": 0}, 1.0),
            PlanCandidate.from_dict("b", {"variant": 1}, 2.0),
        ],
    ]
    panel = empirical_plan_panel(stages, max_plans=4)
    records = []
    for plan in panel.plans:
        ratio = ratios[plan.choices]
        record = {
            "plan_id": plan.fingerprint(),
            "choices": list(plan.choices),
            "predicted_s": plan.predicted_cost_s,
        }
        if plan != panel.selected:
            record["paired_log_ratios"] = [math.log(ratio)] * 4
        records.append(record)
    return panel, records


def test_empirical_metrics_validate_additive_plan():
    panel, records = _two_factor_panel_and_records(
        {(0, 0): 1.0, (0, 1): 1.5, (1, 0): 1.5, (1, 1): 2.0}
    )
    metrics = model_eval._empirical_dp_metrics(
        panel,
        records,
        panel_sha256="a" * 64,
        bootstrap_samples=100,
        regret_tolerance=0.01,
        interaction_tolerance=0.02,
    )

    assert metrics["ok"]
    assert metrics["status"] == "validated"
    assert metrics["regret"]["upper95"] == pytest.approx(0.0)
    assert metrics["interaction"]["max_abs_residual_upper95"] == pytest.approx(0.0)


def test_empirical_metrics_reject_nonadditive_complete_plan_interaction():
    panel, records = _two_factor_panel_and_records(
        {(0, 0): 1.0, (0, 1): 1.5, (1, 0): 1.5, (1, 1): 1.9}
    )
    metrics = model_eval._empirical_dp_metrics(
        panel,
        records,
        panel_sha256="b" * 64,
        bootstrap_samples=100,
        regret_tolerance=0.01,
        interaction_tolerance=0.02,
    )

    assert not metrics["ok"]
    assert metrics["selection_supported"]
    assert not metrics["additivity_supported"]
    assert metrics["status"] == "falsified"
    assert metrics["interaction"]["max_abs_residual"] == pytest.approx(0.1)


def test_single_plan_empirical_certificate_is_explicitly_vacuous():
    stages = [[PlanCandidate.from_dict("only", {"variant": 0}, 1.0)]]
    panel = empirical_plan_panel(stages)
    records = [
        {
            "plan_id": panel.selected.fingerprint(),
            "choices": [0],
            "predicted_s": 1.0,
        }
    ]
    metrics = model_eval._empirical_dp_metrics(
        panel,
        records,
        panel_sha256="c" * 64,
        bootstrap_samples=100,
        regret_tolerance=0.01,
        interaction_tolerance=0.02,
    )

    assert metrics["ok"]
    assert metrics["status"] == "validated-vacuous-single-plan"
    assert metrics["selection_supported"]
    assert metrics["additivity_supported"]


def test_local_search_times_correct_configs_in_balanced_round_robin(monkeypatch):
    class FakeSpace:
        knobs = []

        def deployment_space(self):
            return self

        def enumerate(self, cap=None):
            del cap
            yield {"variant": 0}
            yield {"variant": 1}

        def size(self):
            return 2

    calls = []

    def run(_inputs, config):
        calls.append(config["variant"])
        return 0.0

    case = SimpleNamespace(
        case_id="fake:case",
        space=FakeSpace(),
        run=run,
        check_out=lambda output: output,
        atol=0.0,
        rtol=0.0,
    )
    ticks = iter(float(value) for value in range(8))
    monkeypatch.setattr(model_eval.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(model_eval.time, "perf_counter", lambda: next(ticks))

    result = model_eval._search_case(case, {}, 0.0, runs=2, max_configs=4)

    assert calls[:2] == [0, 1]  # correctness/warmup
    assert calls[2:] == [0, 1, 1, 0]
    assert [item["latency_samples_s"] for item in result["measurements"]] == [
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    assert result["timing_protocol"]["ordering"] == (
        "deterministic-round-robin-rotated-reversed"
    )


def test_complete_plan_pairs_use_deterministic_balanced_order(monkeypatch):
    selected = {"kind": "selected"}
    challenger = {"kind": "challenger"}
    clock = SimpleNamespace(value=0.0)
    execution_order = []

    def execute(_workload, plan, _prepared):
        execution_order.append(plan["kind"])
        clock.value += 1.0 if plan is selected else 2.0
        return [], []

    monkeypatch.setattr(model_eval, "_execute_model", execute)
    monkeypatch.setattr(model_eval.time, "perf_counter", lambda: clock.value)

    result = model_eval._paired_plan_time(
        SimpleNamespace(),
        challenger,
        selected,
        {},
        runs=3,
        numerator_label="challenger",
        denominator_label="selected",
    )

    assert execution_order == [
        "selected",
        "challenger",
        "challenger",
        "selected",
        "selected",
        "challenger",
        "challenger",
        "selected",
    ]
    assert result["selected_samples_s"] == [1.0] * 4
    assert result["challenger_samples_s"] == [2.0] * 4
    assert result["paired_log_ratios"] == pytest.approx([math.log(2.0)] * 4)
    assert result["pair_orders"] == [
        "selected-challenger",
        "challenger-selected",
        "selected-challenger",
        "challenger-selected",
    ]
    assert result["paired_runs"] == 4
    assert result["ratio"] == pytest.approx(2.0)


def test_runtime_event_is_bound_to_the_frozen_model_callsite_scope():
    with capture_runtime_events() as events:
        with runtime_scope("tiny-linear-serving", "tiny-linear-serving/gla-short"):
            _report_probed_backend_execution(
                capability="new_tile",
                control="gla.new_tile",
                value=64,
                backend="chunk_simple_gla_fwd_varlen",
            )
    assert events == [
        {
            "model": "tiny-linear-serving",
            "callsite": "tiny-linear-serving/gla-short",
            "capability": "new_tile",
            "control": "gla.new_tile",
            "value": 64,
            "backend": "chunk_simple_gla_fwd_varlen",
            "verified_backend_probe": True,
        }
    ]


def test_frozen_evaluator_wraps_the_declared_backend_and_binds_its_argument(
    tmp_path, monkeypatch
):
    capability = "new_tile"
    manifest_dir = tmp_path / "akt/core/evolve/capabilities"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / f"{capability}.json").write_text(
        json.dumps(
            {
                "name": capability,
                "search_dimensions": [
                    {
                        "control": "gla.new_tile",
                        "kernel_function": "backend",
                        "consumer": "python/fake_consumer.py",
                    }
                ],
            }
        )
    )

    backend_module = types.ModuleType("fake_backend")

    def backend(value, *, tile):
        return value + tile

    backend.__module__ = "fake_backend"
    backend_module.backend = backend
    runner_module = types.ModuleType("fake_runner")
    runner_module.backend = backend

    def run(_inputs, config):
        return runner_module.backend(1, tile=config["tile"])

    run.__module__ = "fake_runner"
    monkeypatch.setitem(sys.modules, "fake_backend", backend_module)
    monkeypatch.setitem(sys.modules, "fake_runner", runner_module)
    monkeypatch.setattr(model_eval, "REPO", tmp_path)
    monkeypatch.setattr(
        model_eval,
        "derive_search_dimensions",
        lambda _manifest, _repo: (
            {},
            [
                {
                    "control": "gla.new_tile",
                    "kernel_path": "python/fake_backend.py",
                    "kernel_function": "backend",
                    "kernel_argument": "tile",
                }
            ],
            [],
        ),
    )

    installed = model_eval._install_backend_probes(
        [SimpleNamespace(run=run)], capability
    )
    with capture_runtime_events() as events:
        with runtime_scope("model", "model/call"):
            assert runner_module.backend(1, tile=8) == 9

    assert installed[0]["kernel_function"] == "backend"
    assert events[0]["value"] == 8
    assert events[0]["verified_backend_probe"] is True


def test_next_round_bottleneck_uses_retained_plan_after_rejection(
    tmp_path, monkeypatch, capsys
):
    evaluation = tmp_path / "eval.json"
    site = "tiny-linear-serving/gla-short"
    evaluation.write_text(
        json.dumps(
            {
                "gate_decision": "reject",
                "models": [
                    {
                        "model": "tiny-linear-serving",
                        "candidate_s": 0.001,
                        "incumbent_s": 0.002,
                        "ratio": 0.5,
                        "callsites": [site],
                        "selected_plan": {site: {"tile": 2}},
                        "incumbent_plan": {site: {"tile": 1}},
                    }
                ],
                "case_search": {
                    "gla:seq512_h8": {
                        "measurements": [
                            {"config": {"tile": 1}, "latency_s": 0.002},
                            {"config": {"tile": 2}, "latency_s": 0.001},
                        ]
                    }
                },
                "runtime_evidence": {"required": False},
            }
        )
    )
    monkeypatch.setattr(adapter, "EVAL_OUT", evaluation)

    adapter.cmd_bottleneck(None)
    rendered = capsys.readouterr().out

    assert "RETAINED INCUMBENT MODEL tiny-linear-serving: 2.000ms" in rendered
    assert "selected={'tile': 1}" in rendered
