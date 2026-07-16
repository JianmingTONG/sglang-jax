from akt.board.build import _apply_incumbent_envelope, _control_inventory


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
