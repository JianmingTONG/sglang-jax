import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

from akt.core.evolve import loop
from akt.core.evolve.action_catalog import (
    action_catalog_fingerprint,
    load_frontier_catalog,
)
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
    monkeypatch.setattr(action_catalog, "load_frontier_catalog", lambda _repo: {})
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


def test_empty_but_valid_catalog_flags_exhaustion_but_continues(monkeypatch):
    """Zero open actions exhausts only the mined red-link space: the round may
    continue with a NOVEL-algorithm proposal, so the contract renders (with the
    novel-only note) instead of raising, and _maybe_stop_converged never stops."""
    import akt.core.evolve.action_catalog as action_catalog

    real_context = action_catalog.action_catalog_context(loop.ROOT)
    assert loop.action_catalog_convergence() == (False, len(real_context["actions"]))

    empty = dict(real_context, actions=[])
    monkeypatch.setattr(
        action_catalog, "action_catalog_context", lambda _repo: empty
    )
    assert loop.action_catalog_convergence() == (True, 0)
    block = loop.action_contract_block()
    assert "frontier slots remain available" in block

    saved = {}
    monkeypatch.setattr(loop, "save", saved.update)
    state = {"round": 4, "space_exhausted": False}
    assert loop._maybe_stop_converged(state) is False
    assert state["space_exhausted"] is True
    assert saved["space_exhausted"] is True


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


def _expression_ast(source):
    return ast.dump(ast.parse(source, mode="eval").body, include_attributes=False)


def _frontier_slot():
    """Pick a real frontier slot from the live graph (prefer the simple_gla family)."""
    frontier = load_frontier_catalog(loop.ROOT)
    assert frontier, "the generated graph carries no frontier slots"
    for slot in frontier.values():
        if slot.get("family") == "simple_gla":
            return slot
    return next(iter(frontier.values()))


def _novel_proposed_action():
    slot = _frontier_slot()
    gap_id = slot["gap_id"]
    evidence_path = str(slot["evidence"]).rsplit(":", 1)[0]
    return {
        # Graph-owned fields: copied verbatim from the frontier slot template.
        "gap_id": gap_id,
        "family": slot["family"],
        "kernel_ids": sorted(slot["kernel_ids"]),
        "source_axis": slot["source_axis"],
        "source_function": slot["source_function"],
        "incumbent_value": slot["incumbent_value"],
        "candidate_values": list(slot["candidate_values"]),
        "category": slot["category"],
        "model_callsites": sorted(slot["model_callsites"]),
        "action_edge": dict(slot["action_edge"]),
        # Oracle-owned fields: the implementation's mined sink and evidence line.
        "source_sink": {
            "assignments": ["use_variant_path"],
            "expression_asts": {
                "use_variant_path": _expression_ast(slot["source_axis"])
            },
        },
        "source_evidence": {
            "path": evidence_path,
            "line": 0,
            "detail": "novel schedule variant implemented behind the frontier toggle",
        },
    }


def test_pending_manifest_accepts_a_valid_novel_proposed_action(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "CAPS", tmp_path)
    proposed = _novel_proposed_action()
    callsite = sorted(proposed["model_callsites"])[0]
    manifest = {
        "name": "novel_frontier_variant",
        "gap_id": proposed["gap_id"],
        "action_graph_fingerprint": action_catalog_fingerprint(loop.ROOT),
        "hypothesis": "a frontier-anchored novel variant, declared in the flexgraph interface",
        "estimate": {
            "model": callsite.split("/", 1)[0],
            "callsite": callsite,
            "baseline_share_pct": 25.0,
            "expected_relief_pct": 4.0,
            "reasoning": "The slot's dominant callsite carries this kernel's trace share.",
        },
        "proposed_action": proposed,
        "search_dimensions": [
            {
                "control": f"{sorted(proposed['kernel_ids'])[0]}.{proposed['source_axis']}",
                "kernel_function": proposed["source_function"],
                "consumer": "python/sgl_jax/srt/layers/attention/linear/lightning_backend.py",
            }
        ],
        "files_touched": [
            proposed["source_evidence"]["path"],
            "akt/core/evolve/capabilities/novel_frontier_variant.json",
        ],
        "status": "pending",
    }
    (tmp_path / "novel_frontier_variant.json").write_text(json.dumps(manifest))

    loaded, _path = loop.load_manifest("novel_frontier_variant", require_pending=True)
    assert loaded["proposed_action"]["gap_id"] == proposed["gap_id"]


def test_novel_closure_requires_the_mined_finding_to_match(tmp_path, monkeypatch):
    from akt.core.evolve import action_catalog

    proposed = _novel_proposed_action()
    gap_id = proposed["gap_id"]
    evidence_path = proposed["source_evidence"]["path"]
    manifest = {"gap_id": gap_id, "proposed_action": proposed}
    candidate_path = tmp_path / "akt/optimization_history/.candidate_flexgraph.json"
    candidate_path.parent.mkdir(parents=True)

    mined = {
        "gap_id": gap_id,
        "family": proposed["family"],
        "kernel_ids": list(proposed["kernel_ids"]),
        "axis": proposed["source_axis"],
        "source_axis": proposed["source_axis"],
        "source_function": proposed["source_function"],
        "category": proposed["category"],
        "source_sink": proposed["source_sink"],
        "incumbent_value": proposed["incumbent_value"],
        "candidate_values": list(proposed["candidate_values"]),
        "model_callsites": list(proposed["model_callsites"]),
        "evidence": f"{evidence_path}:1234",
        "open": False,
        "programmer_exposed": True,
    }
    graphs = {"gaps": []}

    def write_candidate(*_args, **_kwargs):
        candidate_path.write_text(json.dumps(graphs))
        return 0, ""

    # The closure reads the frontier twice: the incumbent tree (loop.ROOT, here
    # monkeypatched to tmp_path) has the slot still OPEN, while the regenerated
    # candidate graph (loaded from a separate temporary root) has it closed.
    before_frontier = {
        gap_id: {
            "gap_id": gap_id,
            "family": proposed["family"],
            "evidence": f"{evidence_path}:532",
        }
    }
    after_frontier = {}

    def fake_frontier(repo):
        if repo == tmp_path:
            return dict(before_frontier)
        return dict(after_frontier)

    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(loop, "run_argv", write_candidate)
    monkeypatch.setattr(action_catalog, "load_action_catalog", lambda _repo: {})
    monkeypatch.setattr(action_catalog, "load_action_graph", lambda _repo: ({}, {}))
    monkeypatch.setattr(action_catalog, "load_frontier_catalog", fake_frontier)
    monkeypatch.setattr(
        action_catalog,
        "action_catalog_context",
        lambda _repo: {"fingerprint": "candidate"},
    )

    # 1. Not mined at all -> the invented algorithm was not written down in the
    #    flexgraph interface.
    result = loop.candidate_action_graph_closure(manifest)
    assert result["ok"] is False
    assert any("was not mined into the regenerated graph" in e for e in result["errors"])

    # 2. Mined with drifted semantics -> field-for-field mismatch is fail-closed.
    graphs = {"gaps": [dict(mined, candidate_values=list(proposed["candidate_values"]) + ["extra"])]}
    result = loop.candidate_action_graph_closure(manifest)
    assert result["ok"] is False
    assert any("candidate_values" in e and "mined as" in e for e in result["errors"])

    # 3. Mined exactly as declared and programmer-exposed at birth -> closure holds.
    graphs = {"gaps": [mined]}
    result = loop.candidate_action_graph_closure(manifest)
    assert result["ok"] is True, result["errors"]
    assert result["selected_finding"]["gap_id"] == gap_id

    # 4. Slot still emitted by the regenerated frontier -> the declared axis was
    #    never implemented; the round fails even though the mined finding matches.
    after_frontier[gap_id] = dict(before_frontier[gap_id])
    result = loop.candidate_action_graph_closure(manifest)
    assert result["ok"] is False
    assert any("still open" in e for e in result["errors"])


# ---------------------------------------------------------------- testbench mode


def test_testbench_live_document_is_accepted_as_clean():
    """The synthetic live doc injected under --testbench must pass the frozen
    live gate: not_applicable + no attempt + skipped is a clean skip."""
    doc = loop._testbench_live_document()
    assert doc["schema_version"] == "akt.live-model-eval.v1"
    assert doc["status"] == "skipped"
    evidence = loop.live_model_gate_evidence(doc)
    assert evidence["ok"] is True, evidence["errors"]
    assert evidence["not_applicable_candidates"] == 1
    assert evidence["eligible_candidates"] == 0


def test_testbench_relaxes_exactly_hardware_and_deferred_conjuncts():
    """Only under a testbench campaign, and only the two named conjuncts."""
    failing = {"target_hardware_ok": False, "n_deferred": 2}
    strict = loop._testbench_gate_relaxations(
        failing, {"expected_models": ["m1", "m2", "m3"]}
    )
    assert strict == {
        "testbench": False,
        "required_models": 3,
        "target_hardware_ok": False,
        "no_deferred": False,
    }

    relaxed = loop._testbench_gate_relaxations(
        failing, {"testbench": True, "expected_models": ["tiny-linear-serving"]}
    )
    assert relaxed == {
        "testbench": True,
        "required_models": 1,
        "target_hardware_ok": True,
        "no_deferred": True,
    }

    passing = {"target_hardware_ok": True, "n_deferred": 0}
    on_target = loop._testbench_gate_relaxations(passing, {})
    assert on_target["target_hardware_ok"] is True
    assert on_target["no_deferred"] is True
    assert on_target["required_models"] == 3


def _fake_stream_gate(seen, summary_payload):
    def fake(command, log_path, *, progress_prefixes, capability, timeout=0, env=None):
        seen["command"] = [str(part) for part in command]
        seen["env"] = env
        out = Path(seen["command"][seen["command"].index("--out") + 1])
        out.write_text(json.dumps(summary_payload))
        return 0

    return fake


def test_model_hw_eval_testbench_flags_env_and_synthetic_live_doc(
    tmp_path, monkeypatch
):
    (tmp_path / "akt/optimization_history").mkdir(parents=True)
    seen = {}
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(
        loop, "_stream_gate", _fake_stream_gate(seen, {"testbench": True, "models": []})
    )
    live_calls = []
    monkeypatch.setattr(
        loop,
        "_live_model_hw_eval",
        lambda *args, **kwargs: live_calls.append((args, kwargs)),
    )

    summary = loop.model_hw_eval(3, testbench=True)

    assert "--allow-non-target" in seen["command"]
    assert seen["env"]["PALLAS_INTERPRET"] == "1"
    assert live_calls == []                    # live gate skipped entirely
    assert summary["live_model_evaluation"]["schema_version"] == (
        "akt.live-model-eval.v1"
    )
    assert summary["live_model_evaluation"]["status"] == "skipped"
    assert summary["live_model_gate"]["ok"] is True


def test_model_hw_eval_non_testbench_path_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / "akt/optimization_history").mkdir(parents=True)
    seen = {}
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(
        loop, "_stream_gate", _fake_stream_gate(seen, {"models": []})
    )
    live_doc = {
        "schema_version": "akt.live-model-eval.v1",
        "candidates": [
            {"applicability": "not_applicable", "attempted": False, "status": "skipped"}
        ],
    }
    live_calls = []
    monkeypatch.setattr(
        loop,
        "_live_model_hw_eval",
        lambda *args, **kwargs: (live_calls.append((args, kwargs)), live_doc)[1],
    )

    summary = loop.model_hw_eval(3)

    assert "--allow-non-target" not in seen["command"]
    assert seen["env"]["PALLAS_INTERPRET"] == "0"
    assert len(live_calls) == 1                # strict path still runs the live gate
    assert summary["live_model_evaluation"] == live_doc


def test_query_prints_testbench_banner_only_for_testbench_campaigns(monkeypatch):
    monkeypatch.setattr(loop, "bottleneck_block", lambda: "== BOTTLENECK test")
    monkeypatch.setattr(
        loop, "action_space_block", lambda: "== RED-LINK ACTION SPACE test"
    )
    monkeypatch.setattr(loop, "_cap_summaries", lambda: ([], []))
    state = _legacy_state() | {"objective_scope": loop.OBJECTIVE_SCOPE}

    assert loop.TESTBENCH_BANNER not in loop.query(state)
    banner_rendered = loop.query(state | {"testbench": True})
    assert loop.TESTBENCH_BANNER in banner_rendered
    assert banner_rendered.index(loop.TESTBENCH_BANNER) < banner_rendered.index(
        "CAPABILITY QUERY"
    )


def test_submit_and_run_refuse_a_testbench_flag_mismatch(monkeypatch, capsys):
    state = {
        **_legacy_state(),
        "objective_scope": loop.OBJECTIVE_SCOPE,
        "testbench": True,
        "incumbent_plans": {"m1": {}},
        "incumbent_case_spaces": {"case": [{}]},
        "expected_models": ["m1"],
    }
    monkeypatch.setattr(loop, "load", lambda: state)

    with pytest.raises(RuntimeError, match="testbench flag mismatch"):
        loop.cmd_submit(SimpleNamespace(capability="x", runs=3, audit=False))

    loop.cmd_run(SimpleNamespace(rounds=1, oracle="codex", oracle_cmd=None))
    assert "--testbench flag mismatch" in capsys.readouterr().out


# ------------------------------------------------------- api-novelty history cap


def test_api_novelty_history_drops_oversized_delta_graph():
    small = {"ok": True, "delta": {"graph": {"labels": ["a"], "edges": []}}}
    assert loop._truncate_api_graph(small)["delta"]["graph"] == {
        "labels": ["a"],
        "edges": [],
    }

    big = {
        "ok": False,
        "delta": {"graph": {"labels": ["x" * 128] * 1024, "edges": []}},
    }
    truncated = loop._truncate_api_graph(big)["delta"]["graph"]
    assert truncated["dropped"] is True
    assert "64KB" in truncated["note"]
    assert len(json.dumps(truncated).encode()) < loop._API_GRAPH_HISTORY_CAP

    assert loop._truncate_api_graph({"ok": True}) == {"ok": True}


# ------------------------------------------------- standalone-API novel form (B)


def _standalone_slot(family="simple_gla", kernel_ids=("gla",)):
    """A synthetic perpetual `<family>:new_api:standalone` slot (extractor shape)."""
    from akt.benchmark.model_workloads import callsites_for_kernel_ids

    return {
        "gap_id": f"{family}:new_api:standalone",
        "family": family,
        "kernel_ids": sorted(kernel_ids),
        "source_axis": "new_api",
        "axis": "new_api",
        "category": "standalone-api",
        "incumbent_value": "incumbent",
        "incumbent_value_known": True,
        "candidate_values": ["incumbent"],
        "evidence": "python/sgl_jax/srt/kernels/simple_gla/simple_gla.py:0",
        "detail": "perpetual standalone-API slot",
        "model_callsites": callsites_for_kernel_ids(set(kernel_ids)),
        "access": "frontier",
        "perpetual": True,
    }


def _standalone_proposed_action(slot, axis="gla_impl"):
    """A form-(B) proposal: free-named string-enum dispatch axis, oracle-owned
    function/sink/domain/evidence, family+callsites copied from the slot."""
    gap_id = f"{slot['family']}:{axis}:schedule-toggle"
    return {
        "gap_id": gap_id,
        "family": slot["family"],
        "kernel_ids": sorted(slot["kernel_ids"]),
        "source_axis": axis,
        "source_function": "simple_gla_fwd",
        "source_sink": {
            "assignments": ["kernel_fn"],
            "expression_asts": {
                "kernel_fn": _expression_ast(
                    f"_subchunk_v3 if {axis} == 'subchunk_v3' else _incumbent"
                )
            },
        },
        "incumbent_value": "incumbent",
        "candidate_values": ["incumbent", "subchunk_v3"],
        "category": "schedule-toggle",
        "source_evidence": {
            "path": "python/sgl_jax/srt/kernels/simple_gla/simple_gla.py",
            "line": 0,
            "detail": "free-named dispatch control selecting the standalone API",
        },
        "model_callsites": sorted(slot["model_callsites"]),
        "action_edge": {"source": f"action:{gap_id}", "target": "pallas:dot_general"},
    }


def test_standalone_api_proposal_with_free_axis_is_accepted(monkeypatch):
    """Form (B): a proposal whose gap_id is NOT a per-launch frontier slot is
    accepted when the family carries a `<family>:new_api:standalone` slot, the
    category is schedule-toggle, the axis name is free, and the string domain
    contains the typed incumbent."""
    from akt.core.evolve import capability_contract

    slot = _standalone_slot()
    graph = {"gaps": [], "standalone_frontier_actions": [slot]}
    monkeypatch.setattr(
        capability_contract, "load_action_graph", lambda _repo: (graph, {})
    )
    monkeypatch.setattr(capability_contract, "load_frontier_catalog", lambda _repo: {})

    proposed = _standalone_proposed_action(slot)
    manifest = {"gap_id": proposed["gap_id"], "proposed_action": proposed}
    accepted, errors = capability_contract.validate_proposed_action(
        manifest, loop.ROOT
    )

    assert errors == []
    assert accepted["source_axis"] == "gla_impl"
    assert accepted["candidate_values"] == ["incumbent", "subchunk_v3"]


def test_standalone_free_axis_collision_with_existing_finding_is_rejected(monkeypatch):
    from akt.core.evolve import capability_contract

    slot = _standalone_slot()
    colliding = {
        "gap_id": "simple_gla:gla_impl:shape-pinned-tile",
        "family": "simple_gla",
        "source_axis": "gla_impl",
        "category": "shape-pinned-tile",
    }
    graph = {"gaps": [colliding], "standalone_frontier_actions": [slot]}
    monkeypatch.setattr(
        capability_contract, "load_action_graph", lambda _repo: (graph, {})
    )
    monkeypatch.setattr(capability_contract, "load_frontier_catalog", lambda _repo: {})

    proposed = _standalone_proposed_action(slot)
    manifest = {"gap_id": proposed["gap_id"], "proposed_action": proposed}
    _accepted, errors = capability_contract.validate_proposed_action(
        manifest, loop.ROOT
    )

    assert any(
        "collides with an existing finding/slot axis" in error for error in errors
    )


def test_standalone_closure_requires_mined_dispatch_and_tolerates_perpetual_slot(
    tmp_path, monkeypatch
):
    from akt.core.evolve import action_catalog

    slot = _standalone_slot()
    proposed = _standalone_proposed_action(slot)
    gap_id = proposed["gap_id"]
    manifest = {"gap_id": gap_id, "proposed_action": proposed}
    evidence_path = proposed["source_evidence"]["path"]
    mined = {
        "gap_id": gap_id,
        "family": proposed["family"],
        "kernel_ids": list(proposed["kernel_ids"]),
        "axis": proposed["source_axis"],
        "source_axis": proposed["source_axis"],
        "source_function": proposed["source_function"],
        "category": proposed["category"],
        "source_sink": proposed["source_sink"],
        "incumbent_value": proposed["incumbent_value"],
        "candidate_values": list(proposed["candidate_values"]),
        "model_callsites": list(proposed["model_callsites"]),
        "evidence": f"{evidence_path}:321",
        "open": False,
        "programmer_exposed": True,
    }

    incumbent_graph = tmp_path / "akt/core/analysis/flexgraph_generated.json"
    incumbent_graph.parent.mkdir(parents=True)
    incumbent_graph.write_text(
        json.dumps({"standalone_frontier_actions": [slot]})
    )
    candidate_path = tmp_path / "akt/optimization_history/.candidate_flexgraph.json"
    candidate_path.parent.mkdir(parents=True)
    graphs = {"gaps": [mined], "standalone_frontier_actions": [dict(slot)]}

    def write_candidate(*_args, **_kwargs):
        candidate_path.write_text(json.dumps(graphs))
        return 0, ""

    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(loop, "run_argv", write_candidate)
    monkeypatch.setattr(action_catalog, "load_action_catalog", lambda _repo: {})
    monkeypatch.setattr(action_catalog, "load_action_graph", lambda _repo: ({}, {}))
    monkeypatch.setattr(action_catalog, "load_frontier_catalog", lambda _repo: {})
    monkeypatch.setattr(
        action_catalog,
        "action_catalog_context",
        lambda _repo: {"fingerprint": "candidate"},
    )

    # 1. The perpetual slot persisting in the regenerated graph is NOT an error,
    #    and the mined dispatch finding (programmer-exposed) satisfies closure.
    result = loop.candidate_action_graph_closure(manifest)
    assert result["ok"] is True, result["errors"]
    assert result["standalone_mode"] is True
    assert result["selected_finding"]["gap_id"] == gap_id

    # 2. Dropping the perpetual slot fails closure: the slot never closes.
    graphs = {"gaps": [mined], "standalone_frontier_actions": []}
    result = loop.candidate_action_graph_closure(manifest)
    assert result["ok"] is False
    assert any("never closes" in error for error in result["errors"])

    # 3. The declared dispatch finding must be mined; the slot alone is not
    #    closure evidence.
    graphs = {"gaps": [], "standalone_frontier_actions": [dict(slot)]}
    result = loop.candidate_action_graph_closure(manifest)
    assert result["ok"] is False
    assert any(
        "was not mined into the regenerated graph" in error
        for error in result["errors"]
    )


def test_oracle_prompt_presents_both_novel_forms():
    prompt = loop.ORACLE_PROMPT
    assert "STANDALONE API" in prompt
    assert "a new JAX-callable handle in its own file" in prompt
    assert "free-named dispatch control whose default is the incumbent path" in prompt
    assert "or a variant toggle on an existing entry" in prompt
    assert "the axis name is yours" in prompt
    assert "new_api:standalone" in prompt
    assert "never closes" in prompt


def test_opus_oracle_forces_opus_model_at_max_effort():
    command = loop.ORACLES["opus"]
    assert "--model opus" in command
    assert "--effort max" in command
    assert "--dangerously-skip-permissions" in command
    assert "--output-format stream-json" in command
    assert "--verbose" in command
    assert command.startswith("cat {prompt} | claude -p ")
