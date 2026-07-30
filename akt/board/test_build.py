import json
from pathlib import Path

import akt.board.build as board_build
from akt.board.build import (
    _apply_incumbent_envelope,
    _compact_empirical_status,
    _compact_live_status,
    _compact_models,
    _control_inventory,
    _flexgraph,
    _kernels_from_eval,
    _trail_record,
)


ROOT = Path(__file__).resolve().parents[2]


def test_incumbent_envelope_starts_at_baseline_and_never_rises():
    trail = [
        {"round": 1, "decision": "keep", "correct": True, "geomean_s": 0.9},
        {"round": 2, "decision": "reject", "correct": True, "geomean_s": 0.7},
        {"round": 3, "decision": "keep", "correct": False, "geomean_s": 0.6},
        {"round": 4, "decision": "keep", "correct": True, "geomean_s": 0.95},
        {"round": 5, "decision": "keep", "correct": True, "geomean_s": 0.8},
    ]

    baseline, incumbent = _apply_incumbent_envelope(
        trail, {"base_search_geomean": 1.0, "incumbent_geomean": 0.8}
    )

    assert baseline == 1.0
    assert incumbent == 0.8
    assert [row["incumbent_after_s"] for row in trail] == [0.9, 0.9, 0.9, 0.9, 0.8]
    assert [row["advances_incumbent"] for row in trail] == [True, False, False, False, True]


def test_post_run_invalid_and_superseded_measurements_do_not_advance_envelope():
    trail = [
        {
            "round": 1,
            "decision": "keep",
            "correct": True,
            "geomean_s": 0.7,
            "performance_status": "post-run-invalid",
        },
        {
            "round": 2,
            "decision": "keep",
            "correct": True,
            "geomean_s": 0.6,
            "performance_status": "superseded-measurement",
        },
        {"round": 3, "decision": "keep", "correct": True, "geomean_s": 0.8},
    ]

    baseline, incumbent = _apply_incumbent_envelope(
        trail, {"base_search_geomean": 1.0}
    )

    assert baseline == 1.0
    assert incumbent == 0.8
    assert [row["incumbent_eligible"] for row in trail] == [False, False, True]
    assert [row["incumbent_after_s"] for row in trail] == [1.0, 1.0, 0.8]


def test_control_inventory_distinguishes_production_and_runner_only_axes():
    inventory = _control_inventory()

    assert "kda.state_block_chunks" in inventory["programmer_controls"]
    assert "gla.output_value_tiles" in inventory["programmer_controls"]
    assert "kda.single_chunk_state_elision" in inventory["runner_only_knobs"]
    assert "gla.zero_state_output_elision" in inventory["runner_only_knobs"]


def test_incumbent_envelope_excludes_measurements_from_an_old_objective():
    trail = [
        {
            "round": 1,
            "decision": "keep",
            "correct": True,
            "geomean_s": 0.5,
            "objective_scope": None,
        },
        {
            "round": 2,
            "decision": "keep",
            "correct": True,
            "geomean_s": 0.8,
            "objective_scope": "stateful-serving-deployable-v1",
        },
    ]

    baseline, incumbent = _apply_incumbent_envelope(
        trail,
        {
            "objective_scope": "stateful-serving-deployable-v1",
            "objective_baseline_geomean": 1.0,
        },
    )

    assert baseline == 1.0
    assert incumbent == 0.8
    assert trail[0]["objective_compatible"] is False
    assert trail[0]["incumbent_eligible"] is False


def test_model_eval_case_search_and_models_have_compact_board_views(
    tmp_path, monkeypatch
):
    evaluation = {
        "target_hardware": {"backend": "tpu"},
        "case_search": {
            "gmm_v2:test": {
                "space_size": 3,
                "valid_configs": 2,
                "correct_configs": 2,
                "all_configs_correct": True,
                "knobs": [{"name": "buffer_count", "default": 3}],
                "measurements": [
                    {"config": {"buffer_count": 3}, "latency_s": 0.003},
                    {"config": {"buffer_count": 4}, "latency_s": 0.002},
                ],
            }
        },
        "models": [{
            "model": "tiny-dense-serving",
            "candidate_s": 0.002,
            "incumbent_s": 0.003,
            "ratio": 2 / 3,
            "selected_correct": True,
            "certificate": {
                "certified_additive_optimum": True,
                "bounded_dp_matches_bruteforce": True,
            },
            "empirical_dp_certificate": {"ok": True, "status": "validated"},
        }],
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(evaluation))
    monkeypatch.setattr(board_build, "EVAL", path)

    kernels, loaded = _kernels_from_eval()

    assert loaded == evaluation
    assert kernels[0]["best_config"] == {"buffer_count": 4}
    assert kernels[0]["default_s"] == 0.003
    assert kernels[0]["best_s"] == 0.002
    assert kernels[0]["correct"] is True
    assert _compact_models(evaluation) == [{
        "model": "tiny-dense-serving",
        "candidate_s": 0.002,
        "incumbent_s": 0.003,
        "ratio": 2 / 3,
        "correct": True,
        "dp_ok": True,
        "empirical": {
            "status": "validated",
            "ok": True,
            "models": 0,
            "passed_models": 0,
        },
    }]


def test_compact_empirical_and_live_kimi_statuses_include_topology_skip():
    empirical = _compact_empirical_status({
        "empirical_dp_certificate": {
            "ok": True,
            "status": "validated",
            "models": [{"ok": True}, {"ok": True}, {"ok": False}],
        }
    })
    not_applicable = _compact_live_status({
        "live_model_gate": {
            "ok": True,
            "eligible_candidates": 0,
            "passed_candidates": 0,
            "not_applicable_candidates": 1,
            "topology_skipped_candidates": 0,
        },
        "live_model_evaluation": {"status": "skipped"},
    })

    assert empirical == {
        "status": "validated",
        "ok": True,
        "models": 3,
        "passed_models": 2,
    }
    assert not_applicable["status"] == "not_applicable"
    assert not_applicable["ok"] is True
    assert not_applicable["not_applicable"] == 1
    assert _compact_live_status({})["status"] == "not_run"
    failed_skip = _compact_live_status({
        "live_model_gate": {
            "ok": False,
            "eligible_candidates": 0,
            "topology_skipped_candidates": 1,
        },
        "live_model_evaluation": {"status": "skipped"},
    })
    assert failed_skip["status"] == "skipped"
    assert failed_skip["ok"] is False

    historical_topology_skip = _compact_live_status({
        "live_model_gate": {
            "ok": True,
            "eligible_candidates": 0,
            "passed_candidates": 0,
            "not_applicable_candidates": 0,
            "topology_skipped_candidates": 1,
        }
    })
    assert historical_topology_skip["status"] == "topology_skipped"
    assert historical_topology_skip["ok"] is True


def test_current_round_history_is_normalized_for_the_compact_board():
    record = {
        "round": 13,
        "capability": "gmm_v2_buffer_count",
        "gap_id": "megablox_gmm_kernel:buffer_count:pipeline-depth",
        "action_snapshot": {"evidence": "python/kernel.py:140"},
        "estimate": {"expected_relief_pct": 3.4},
        "search_dimensions": [{
            "control": "gmm_v2.buffer_count",
            "candidate_values": [3, 4],
        }],
        "improvement_pct": 2.7,
        "accuracy": {
            "tiny-dense-serving": {
                "selected_plan": {
                    "tiny-dense-serving/dense-projection": {"buffer_count": 4}
                }
            }
        },
        "live_model_gate": {
            "ok": True,
            "eligible_candidates": 0,
            "topology_skipped_candidates": 1,
        },
    }

    row = _trail_record(record, {})

    assert row["gap"] == record["gap_id"]
    assert row["source_evidence"] == "python/kernel.py:140"
    assert row["delta_pct"] == 2.7
    assert row["estimated_relief_pct"] == 3.4
    assert row["search_dimension"] == "gmm_v2.buffer_count=[3, 4]"
    assert row["best_configs"] == [{
        "case": "tiny-dense-serving/dense-projection",
        "config": {"buffer_count": 4},
    }]
    assert row["live_kimi"]["status"] == "topology_skipped"


def test_board_uses_generated_action_and_hidden_lowering_edges():
    first = _flexgraph({})
    graph = _flexgraph(
        {}, {"action_graph_fingerprint": first["action_graph_fingerprint"]}
    )
    action_ids = {edge["gap_id"] for edge in graph["action_edges"]}
    expected = {
        gap["gap_id"]
        for gap in graph["gaps"]
        if gap.get("open")
        and gap.get("eligible_for_current_gate")
        and not gap.get("programmer_exposed")
    }

    assert graph["kind"] != "unavailable"
    assert action_ids == expected
    assert graph["hidden_lowering_edges"]
    assert graph["action_context_status"] == "current"
    assert graph["actions_executable_for_campaign"] is True

    html = (ROOT / "akt/board/index.html").read_text()
    assert "fg.action_edges" in html
    assert "fg.hidden_lowering_edges" in html
    assert "fg.gap_edges" not in html


def test_board_marks_an_unbound_or_stale_action_graph_non_executable():
    unbound = _flexgraph({})
    stale = _flexgraph({}, {"action_graph_fingerprint": "0" * 64})

    assert unbound["action_context_status"] == "unbound"
    assert unbound["actions_executable_for_campaign"] is False
    assert stale["action_context_status"] == "stale"
    assert stale["actions_executable_for_campaign"] is False


def test_board_fails_closed_without_v2_edge_tables(monkeypatch):
    monkeypatch.setattr(board_build, "_load_json", lambda _path: {"nodes": [{}]})

    graph = _flexgraph({})

    assert graph["kind"] == "unavailable"
    assert graph["action_edges"] == []
    assert graph["hidden_lowering_edges"] == []


def test_board_keeps_compact_gate_ui_without_latent_audit_redesign():
    html = (ROOT / "akt/board/index.html").read_text()

    assert 'id="mtab"' in html
    assert 'id="gatebadges"' in html
    assert "not_applicable" in html
    assert 'id="hardwareaudit"' not in html
    assert 'id="provenanceaudit"' not in html
    assert "agentic_chart" not in html
    assert "live_model_evaluation" not in html


def test_design_funnel_attributes_selection_to_cases_and_knobs():
    summary = {
        "case_search": {
            "gla:seq512_h8": {
                "research_space_size": 12,
                "space_size": 6,
                "valid_configs": 5,
                "correct_configs": 5,
                "knobs": [
                    {"name": "chunk_size", "values": [64, 128, 256],
                     "programmer_control": "gla.chunk_size", "elevated_by": "cap_a"},
                    {"name": "variant", "values": [0, 1],
                     "programmer_control": None, "elevated_by": None},
                ],
            },
            "fused_mlp:s128_h256_i512": {
                "research_space_size": 4,
                "space_size": 4,
                "valid_configs": 4,
                "correct_configs": 4,
                "knobs": [],
            },
        },
        "models": [
            {
                "model": "tiny-linear-serving",
                "callsite_cases": {
                    "tiny-linear-serving/gla-short": "gla:seq512_h8",
                    "tiny-linear-serving/gla-long": "gla:seq512_h8",
                },
                "selected_plan": {
                    "tiny-linear-serving/gla-short": {"chunk_size": 128},
                    "tiny-linear-serving/gla-long": {"chunk_size": 256},
                },
            },
            {
                "model": "tiny-dense-serving",
                "callsite_cases": {
                    "tiny-dense-serving/dense-mlp": "fused_mlp:s128_h256_i512"
                },
                "selected_plan": {
                    "tiny-dense-serving/dense-mlp": {"b_inter": 128},
                },
            },
        ],
    }
    funnel = board_build._design_funnel(summary)
    by_case = {row["case"]: row for row in funnel["cases"]}
    gla = by_case["gla:seq512_h8"]
    assert (gla["research"], gla["deployable"], gla["valid"]) == (12, 6, 5)
    assert gla["selected"] == 2          # two DISTINCT configs across its callsites
    assert gla["deployable_over_research"] == 0.5
    assert gla["knobs"][0]["deployable_values"] == 3    # programmer-controlled
    assert gla["knobs"][1]["deployable_values"] == 1    # runner-only -> default-anchored
    mlp = by_case["fused_mlp:s128_h256_i512"]
    assert mlp["selected"] == 1
    totals = funnel["totals"]
    assert totals["research"] == 16 and totals["selected"] == 3
    assert board_build._design_funnel({}) == {"cases": [], "totals": {}}
