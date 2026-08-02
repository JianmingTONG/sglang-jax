import json
import time
from types import SimpleNamespace

from akt.core.evolve import loop
from akt.core.evolve.action_catalog import action_catalog_fingerprint
from akt.core.evolve.loop import _render_oracle_line, _scan_rate_limit
from akt.core.evolve.loop import validate_estimate_threshold
import pytest


def test_json_scalars_and_arrays_do_not_crash_stream_parser():
    for line in ('"json string"', "42", "true", "null", "[]"):
        assert _render_oracle_line(line) == line
        assert _scan_rate_limit(line) is None


def test_plain_codex_output_is_rendered_verbatim():
    assert _render_oracle_line("working on the kernel") == "working on the kernel"


def test_explicit_rate_limit_reset_is_extracted():
    line = (
        '{"type":"rate_limit_event","rate_limit_info":'
        '{"status":"rejected","resetsAt":1234}}'
    )
    assert _scan_rate_limit(line) == 1234.0


def test_result_429_uses_a_future_fallback_reset():
    before = time.time()
    reset = _scan_rate_limit('{"type":"result","api_error_status":429}')
    assert reset is not None
    assert before + 1799 <= reset <= time.time() + 1801


def _legacy_state():
    return {
        "round": 12,
        "deadline_ts": time.time() + 3600,
        "target_improvement": 0.02,
        "objective_name": "geomean_s",
        "objective_unit": "s",
        "incumbent_geomean": 0.001,
        "incumbent_choices": 7244,
    }


def test_query_requires_rebaseline_for_legacy_objective(monkeypatch):
    monkeypatch.setattr(loop, "bottleneck_block", lambda: "== BOTTLENECK test")
    monkeypatch.setattr(loop, "action_space_block", lambda: "== RED-LINK ACTION SPACE test")
    monkeypatch.setattr(
        loop,
        "_cap_summaries",
        lambda: ([{"name": "old_elision", "status": "kept"}], []),
    )

    rendered = loop.query(_legacy_state())

    assert "REBASELINE_REQUIRED" in rendered
    assert loop.OBJECTIVE_SCOPE in rendered
    assert "LEGACY kept" in rendered
    assert "old_elision" in rendered
    assert "KEPT under active objective: (none yet)" in rendered


def test_query_separates_current_and_legacy_capability_evidence(monkeypatch):
    state = _legacy_state() | {"objective_scope": loop.OBJECTIVE_SCOPE}
    monkeypatch.setattr(loop, "bottleneck_block", lambda: "== BOTTLENECK test")
    monkeypatch.setattr(loop, "action_space_block", lambda: "== RED-LINK ACTION SPACE test")
    monkeypatch.setattr(
        loop,
        "_cap_summaries",
        lambda: (
            [
                {"name": "old_elision", "status": "kept"},
                {
                    "name": "new_pipeline_control",
                    "status": "kept",
                    "objective_scope": loop.OBJECTIVE_SCOPE,
                    "delta_pct": 3.0,
                },
            ],
            [],
        ),
    )

    rendered = loop.query(state)

    assert "KEPT under active objective: new_pipeline_control(+3.0%)" in rendered
    assert "LEGACY kept (code is in baseline; old deltas are not evidence): old_elision" in rendered


def test_run_refuses_stale_objective_before_starting_oracle(monkeypatch, capsys):
    monkeypatch.setattr(loop, "load", _legacy_state)
    monkeypatch.setattr(
        loop,
        "_implementation_changes",
        lambda: (_ for _ in ()).throw(AssertionError("dirty check should not run")),
    )

    loop.cmd_run(SimpleNamespace(rounds=1, oracle="codex", oracle_cmd=None))

    assert "objective is stale" in capsys.readouterr().out


def test_rate_limited_attempt_reverts_partial_oracle_edits(monkeypatch):
    state = {
        **_legacy_state(),
        "objective_scope": loop.OBJECTIVE_SCOPE,
        "deadline_ts": time.time() + 1,
        "incumbent_commit": "deadbeef",
        "incumbent_plans": {"m1": {}, "m2": {}, "m3": {}},
        "incumbent_case_spaces": {"case": [{}]},
        "expected_models": ["m1", "m2", "m3"],
        "action_graph_fingerprint": action_catalog_fingerprint(loop.ROOT),
    }
    reverted = []
    monkeypatch.setattr(loop, "load", lambda: state)
    monkeypatch.setattr(loop, "git_head", lambda: "deadbeef")
    monkeypatch.setattr(loop, "_implementation_changes", lambda: [])
    monkeypatch.setattr(loop, "_worktree_changed", lambda: [])
    monkeypatch.setattr(loop, "_frozen_fingerprint", lambda: {})
    monkeypatch.setattr(loop, "_frozen_diff", lambda _pre: [])
    monkeypatch.setattr(
        loop,
        "invoke_oracle",
        lambda _state, _args: (None, time.time() + 60),
    )
    monkeypatch.setattr(loop, "revert_worktree", reverted.append)
    monkeypatch.setattr(loop, "write_status", lambda *_args, **_kwargs: None)

    loop.cmd_run(SimpleNamespace(rounds=1, oracle="codex", oracle_cmd=None))

    assert reverted == ["deadbeef"]


def test_round_requires_current_head_to_match_measured_incumbent(monkeypatch):
    monkeypatch.setattr(loop, "git_head", lambda: "new-clean-commit")

    with pytest.raises(RuntimeError, match="does not match the measured incumbent"):
        loop.validate_incumbent_head({"incumbent_commit": "measured-commit"})


def test_estimate_must_clear_the_campaign_keep_threshold():
    state = {"target_improvement": 0.02}
    validate_estimate_threshold(
        {"estimate": {"expected_relief_pct": 2.1}}, state
    )
    with pytest.raises(ValueError, match="strictly >"):
        validate_estimate_threshold(
            {"estimate": {"expected_relief_pct": 2.0}}, state
        )


def test_pending_manifest_must_select_a_graph_red_link_before_evaluation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(loop, "CAPS", tmp_path)
    manifest = {
        "name": "invented_action",
        "action_graph_fingerprint": action_catalog_fingerprint(loop.ROOT),
        "gap_id": "not-a-red-link",
        "hypothesis": "invented",
        "estimate": {},
        "search_dimensions": [],
        "files_touched": ["python/sgl_jax/srt/kernels/invented.py"],
        "status": "pending",
    }
    (tmp_path / "invented_action.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="executable red-link action"):
        loop.load_manifest("invented_action", require_pending=True)


def test_manifest_names_and_paths_reject_shell_syntax(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "CAPS", tmp_path)
    with pytest.raises(ValueError, match="invalid capability name"):
        loop.load_manifest("bad;name", require_pending=True)

    context = loop.action_contract_block()
    prefix = "== FROZEN FLEXIBILITY-GRAPH ACTION CONTRACT\n"
    action_context = json.loads(context[len(prefix) :])
    action = action_context["actions"][0]
    manifest = {
        "name": "safe_name",
        "gap_id": action["gap_id"],
        "action_graph_fingerprint": action_context["fingerprint"],
        "hypothesis": "test",
        "estimate": {},
        "search_dimensions": [],
        "files_touched": ["python/$(touch_injected).py"],
        "status": "pending",
    }
    (tmp_path / "safe_name.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unsupported characters"):
        loop.load_manifest("safe_name", require_pending=True)


def test_rejected_manifests_are_restart_safe_bookkeeping(tmp_path, monkeypatch):
    relative = "akt/core/evolve/capabilities/rejected_round.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"name": "rejected_round", "status": "rejected"})
    )
    monkeypatch.setattr(loop, "ROOT", tmp_path)

    assert loop._is_bookkeeping(relative) is True


def test_restore_failure_stops_before_rejected_bookkeeping(monkeypatch):
    def failed_checkout(*args, **_kwargs):
        if args[0] == "cat-file":
            return 0, ""
        if args[0] == "ls-tree":
            return 0, "python/sgl_jax/srt/example.py\0"
        return 1, "checkout failed"

    monkeypatch.setattr(loop, "git", failed_checkout)

    with pytest.raises(RuntimeError, match="cannot restore"):
        loop.restore_capability(
            {"files_touched": ["python/sgl_jax/srt/example.py"]}, "deadbeef"
        )


def test_restore_does_not_delete_a_file_when_the_commit_lookup_fails(
    tmp_path, monkeypatch
):
    relative = "python/sgl_jax/srt/example.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("keep me")
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(loop, "git", lambda *_args, **_kwargs: (1, "bad commit"))

    with pytest.raises(RuntimeError, match="cannot resolve incumbent commit"):
        loop.restore_capability({"files_touched": [relative]}, "bad")

    assert path.read_text() == "keep me"


def test_worktree_status_failure_is_not_treated_as_clean(monkeypatch):
    monkeypatch.setattr(loop, "git", lambda *_args, **_kwargs: (1, "status failed"))

    with pytest.raises(RuntimeError, match="cannot inspect Git worktree"):
        loop._worktree_changed()


def test_candidate_graph_closure_rejects_every_collateral_action_change(
    tmp_path, monkeypatch
):
    from akt.core.evolve import action_catalog

    selected = "selected"
    before = {
        selected: {"semantic": "selected"},
        "removed": {"semantic": "stable"},
        "changed": {"semantic": "before"},
    }
    after = {
        "changed": {"semantic": "after"},
        "added": {"semantic": "new"},
    }
    candidate_path = (
        tmp_path / "akt/optimization_history/.candidate_flexgraph.json"
    )
    candidate_path.parent.mkdir(parents=True)

    def write_candidate(*_args, **_kwargs):
        candidate_path.write_text(
            json.dumps(
                {
                    "gaps": [
                        {
                            "gap_id": selected,
                            "open": False,
                            "programmer_exposed": True,
                        }
                    ]
                }
            )
        )
        return 0, ""

    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(loop, "run_argv", write_candidate)
    monkeypatch.setattr(action_catalog, "load_action_catalog", lambda _repo: before)
    monkeypatch.setattr(
        action_catalog, "load_action_graph", lambda _repo: ({}, after)
    )
    monkeypatch.setattr(
        action_catalog,
        "action_catalog_context",
        lambda _repo: {"fingerprint": "candidate"},
    )
    monkeypatch.setattr(
        action_catalog, "action_semantic_record", lambda action: action["semantic"]
    )

    result = loop.candidate_action_graph_closure({"gap_id": selected})

    assert result["ok"] is False
    assert result["unrelated_missing"] == ["removed"]
    assert result["unrelated_added"] == ["added"]
    assert result["unrelated_changed"] == ["changed"]


def test_failed_keep_commit_restores_graph_and_reports_rejection(
    tmp_path, monkeypatch
):
    graph = tmp_path / "akt/core/analysis/flexgraph_generated.json"
    manifest_path = tmp_path / "akt/core/evolve/capabilities/cap.json"
    graph.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    graph.write_text("candidate graph")
    calls = []

    def fake_git(*args, **_kwargs):
        calls.append(args)
        if args[0] == "commit":
            return 1, "commit failed"
        return 0, ""

    monkeypatch.setattr(loop, "git", fake_git)
    monkeypatch.setattr(loop, "git_head", lambda: "old-head")
    result = loop._commit_kept_candidate(
        {
            "name": "cap",
            "files_touched": [
                "akt/core/evolve/capabilities/cap.json",
                "python/kernel.py",
            ],
        },
        manifest_path,
        graph,
        "incumbent graph",
        round_number=1,
        reason="passed",
        previous_head="old-head",
    )

    assert result == {
        "ok": False,
        "head": "old-head",
        "error": "KEEP COMMIT FAILED: commit failed",
    }
    assert graph.read_text() == "incumbent graph"
    assert any(args[:3] == ("reset", "-q", "HEAD") for args in calls)


def test_keep_commit_aborts_if_head_changed_during_evaluation(tmp_path, monkeypatch):
    graph = tmp_path / "graph.json"
    manifest_path = tmp_path / "cap.json"
    graph.write_text("candidate graph")
    monkeypatch.setattr(loop, "git_head", lambda: "different-head")
    monkeypatch.setattr(
        loop,
        "git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Git mutation must not start")
        ),
    )

    result = loop._commit_kept_candidate(
        {"name": "cap", "files_touched": ["cap.json"]},
        manifest_path,
        graph,
        "incumbent graph",
        round_number=1,
        reason="passed",
        previous_head="old-head",
    )

    assert result["ok"] is False
    assert "HEAD changed during evaluation" in result["error"]
    assert graph.read_text() == "incumbent graph"


def test_keep_commit_verifies_the_new_commits_parent(tmp_path, monkeypatch):
    graph = tmp_path / "graph.json"
    manifest_path = tmp_path / "cap.json"
    graph.write_text("candidate graph")
    heads = iter(("old-head", "new-head"))

    def fake_git(*args, **_kwargs):
        if args[0] == "rev-parse":
            return 0, "old-head\n"
        return 0, ""

    monkeypatch.setattr(loop, "git_head", lambda: next(heads))
    monkeypatch.setattr(loop, "git", fake_git)

    result = loop._commit_kept_candidate(
        {"name": "cap", "files_touched": ["cap.json"]},
        manifest_path,
        graph,
        "incumbent graph",
        round_number=1,
        reason="passed",
        previous_head="old-head",
    )

    assert result == {"ok": True, "head": "new-head", "error": None}


def test_failed_keep_cleanup_restores_graph_before_raising(tmp_path, monkeypatch):
    graph = tmp_path / "graph.json"
    manifest_path = tmp_path / "cap.json"
    graph.write_text("candidate graph")

    def fake_git(*args, **_kwargs):
        if args[0] == "commit":
            return 1, "commit failed"
        if args[0] == "reset":
            return 1, "reset failed"
        return 0, ""

    monkeypatch.setattr(loop, "git_head", lambda: "old-head")
    monkeypatch.setattr(loop, "git", fake_git)

    with pytest.raises(RuntimeError, match="cleanup was incomplete"):
        loop._commit_kept_candidate(
            {"name": "cap", "files_touched": ["cap.json"]},
            manifest_path,
            graph,
            "incumbent graph",
            round_number=1,
            reason="passed",
            previous_head="old-head",
        )

    assert graph.read_text() == "incumbent graph"


def test_unresolvable_successful_commit_fails_loudly_after_graph_cleanup(
    tmp_path, monkeypatch
):
    graph = tmp_path / "graph.json"
    manifest_path = tmp_path / "cap.json"
    graph.write_text("candidate graph")
    heads = iter(("old-head", None))
    monkeypatch.setattr(loop, "git_head", lambda: next(heads))
    monkeypatch.setattr(loop, "git", lambda *_args, **_kwargs: (0, ""))

    with pytest.raises(RuntimeError, match="committed HEAD could not be resolved"):
        loop._commit_kept_candidate(
            {"name": "cap", "files_touched": ["cap.json"]},
            manifest_path,
            graph,
            "incumbent graph",
            round_number=1,
            reason="passed",
            previous_head="old-head",
        )

    assert graph.read_text() == "incumbent graph"


def test_oracle_action_contract_is_structured_and_fingerprinted():
    rendered = loop.action_contract_block()
    prefix = "== FROZEN FLEXIBILITY-GRAPH ACTION CONTRACT\n"
    assert rendered.startswith(prefix)
    context = json.loads(rendered[len(prefix) :])
    assert context["version"] == 2
    assert context["fingerprint"] == action_catalog_fingerprint(loop.ROOT)
    assert context["actions"]
    assert all("source_axis" in action for action in context["actions"])
    assert all("source_function" in action for action in context["actions"])
    assert all("incumbent_value" in action for action in context["actions"])


def test_run_refuses_graph_changed_outside_an_accepted_round(monkeypatch, capsys):
    state = {
        **_legacy_state(),
        "objective_scope": loop.OBJECTIVE_SCOPE,
        "incumbent_plans": {"m1": {}, "m2": {}, "m3": {}},
        "incumbent_case_spaces": {"case": [{}]},
        "expected_models": ["m1", "m2", "m3"],
        "action_graph_fingerprint": "0" * 64,
    }
    monkeypatch.setattr(loop, "load", lambda: state)
    monkeypatch.setattr(
        loop,
        "_implementation_changes",
        lambda: (_ for _ in ()).throw(AssertionError("dirty check must not run")),
    )

    loop.cmd_run(SimpleNamespace(rounds=1, oracle="codex", oracle_cmd=None))

    assert "oracle NOT started" in capsys.readouterr().out


def test_live_model_gate_allows_only_clean_topology_skip_or_pass():
    skipped = {
        "status": "skipped",
        "attempted": False,
        "applicability": "direct-capability",
        "eligibility": {"eligible": False},
        "candidate": {"candidate_id": "kimi"},
    }
    assert loop.live_model_gate_evidence({"candidates": [skipped]})["ok"]

    failed = {
        "status": "failed",
        "attempted": True,
        "applicability": "direct-capability",
        "eligibility": {"eligible": True},
        "candidate": {"candidate_id": "kimi"},
    }
    evidence = loop.live_model_gate_evidence({"candidates": [failed]})
    assert evidence["ok"] is False
    assert "did not pass" in evidence["errors"][0]

    passed = failed | {
        "status": "passed",
        "policy_coverage": {
            "capability_applies": True,
            "evaluation_intent": "direct-capability",
        },
    }
    assert loop.live_model_gate_evidence({"candidates": [passed]})["ok"]


def test_live_model_gate_accepts_clean_non_applicability():
    evidence = loop.live_model_gate_evidence(
        {
            "candidates": [
                {
                    "status": "skipped",
                    "attempted": False,
                    "applicability": "not_applicable",
                }
            ]
        }
    )

    assert evidence["ok"] is True
    assert evidence["not_applicable_candidates"] == 1


def test_incumbent_output_drift_detects_bitwise_default_path_change():
    """The default-reproduces-incumbent pin: exact match passes; any drifted,
    missing, or added callsite hash fails; an unpinned campaign checks nothing."""
    pinned = {
        "tiny-linear-serving": {"tiny-linear-serving/gla-short": "a" * 64},
        "tiny-dense-serving": {"tiny-dense-serving/dense-mlp": "b" * 64},
    }
    exact = [
        {
            "model": "tiny-linear-serving",
            "incumbent_output_hashes": {"tiny-linear-serving/gla-short": "a" * 64},
        },
        {
            "model": "tiny-dense-serving",
            "incumbent_output_hashes": {"tiny-dense-serving/dense-mlp": "b" * 64},
        },
    ]
    assert loop._incumbent_output_drift(pinned, exact) == {}

    drifted = [dict(exact[0]), dict(exact[1])]
    drifted[1] = {
        "model": "tiny-dense-serving",
        "incumbent_output_hashes": {"tiny-dense-serving/dense-mlp": "c" * 64},
    }
    assert loop._incumbent_output_drift(pinned, drifted) == {
        "tiny-dense-serving": ["tiny-dense-serving/dense-mlp"]
    }

    missing = [exact[0], {"model": "tiny-dense-serving", "incumbent_output_hashes": {}}]
    assert loop._incumbent_output_drift(pinned, missing) == {
        "tiny-dense-serving": ["tiny-dense-serving/dense-mlp"]
    }

    assert loop._incumbent_output_drift({}, drifted) == {}  # unpinned -> no check


def test_empty_but_valid_catalog_is_convergence_not_error(monkeypatch):
    """Zero open actions must present as a CONVERGED verdict, not a broken graph."""
    import akt.core.evolve.action_catalog as action_catalog

    real_context = action_catalog.action_catalog_context(loop.ROOT)
    assert loop.action_catalog_convergence() == (False, len(real_context["actions"]))

    empty = dict(real_context, actions=[])
    monkeypatch.setattr(
        action_catalog, "action_catalog_context", lambda _repo: empty
    )
    assert loop.action_catalog_convergence() == (True, 0)
    # the oracle contract itself still refuses to run with zero actions
    with pytest.raises(RuntimeError, match="CONVERGED"):
        loop.action_contract_block()


def test_manual_restore_of_a_kept_round_taints_its_history(tmp_path, monkeypatch):
    """A restored KEEP must be downgraded and its rounds marked post-run-invalid,
    so the board's incumbent envelope stops crediting its measurement."""
    hist = tmp_path / "evolve_history.jsonl"
    hist.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"round": 1, "capability": "cap_a", "decision": "keep",
                 "geomean_s": 1.0, "correct": True},
                {"round": 2, "capability": "cap_b", "decision": "reject"},
                {"round": 3, "capability": "cap_a", "decision": "keep",
                 "geomean_s": 0.9, "correct": True},
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(loop, "HIST", hist)

    assert loop._taint_restored_rounds("cap_a") == 2
    records = [json.loads(line) for line in hist.read_text().splitlines()]
    assert [r.get("performance_status") for r in records] == [
        "post-run-invalid", None, "post-run-invalid"
    ]
    assert records[1] == {"round": 2, "capability": "cap_b", "decision": "reject"}
    assert loop._taint_restored_rounds("cap_missing") == 0


def test_gate_contract_is_self_describing_and_matches_the_frozen_literals():
    """The adapter's AKT_GATE contract must mirror the loop's fallback literals
    exactly, so contract-driven and legacy-state gating are byte-identical."""
    import subprocess
    import sys

    out = subprocess.check_output(
        [sys.executable, str(loop.ROOT / "akt/benchmark/adapter.py"), "gate"],
        text=True, cwd=loop.ROOT,
    )
    line = [l for l in out.splitlines() if l.startswith("AKT_GATE ")][-1]
    contract = json.loads(line[len("AKT_GATE "):])
    assert contract["objective_name"] == "paired_model_geomean_s"
    assert contract["objective_unit"] == "s"
    assert contract["lower_is_better"] is True
    assert contract["summary_keys"] == {
        "candidate": "candidate_geomean_s",
        "incumbent": "incumbent_geomean_s",
        "paired_ratio": "paired_ratio_geomean",
    }
    assert contract["objective_scope"] == loop.OBJECTIVE_SCOPE


def test_probe_forgery_scan_rejects_manifest_files_touching_runtime(tmp_path, monkeypatch):
    """Layer 2 of probe authentication: an edited file referencing the frozen
    runtime-evidence channel must fail the static pre-check."""
    (tmp_path / "python").mkdir()
    clean = tmp_path / "python/clean_kernel.py"
    clean.write_text("def kernel():\n    return 1\n")
    dirty = tmp_path / "python/forging_kernel.py"
    dirty.write_text(
        "from akt.benchmark.runtime import _report_probed_backend_execution\n"
    )
    monkeypatch.setattr(loop, "ROOT", tmp_path)

    assert loop._scan_probe_forgery({"files_touched": ["python/clean_kernel.py"]}) == []
    errors = loop._scan_probe_forgery(
        {"files_touched": ["python/clean_kernel.py", "python/forging_kernel.py"]}
    )
    assert len(errors) == 1 and "forging_kernel.py" in errors[0]
    assert loop._scan_probe_forgery({"files_touched": ["missing.py"]}) == []
