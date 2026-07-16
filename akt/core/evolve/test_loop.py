import time
from types import SimpleNamespace

from akt.core.evolve import loop
from akt.core.evolve.loop import _render_oracle_line, _scan_rate_limit


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
    monkeypatch.setattr(loop, "frontier_block", lambda: "== FRONTIER test")
    monkeypatch.setattr(
        loop,
        "_cap_summaries",
        lambda: ([{"name": "old_elision", "status": "kept"}], []),
    )

    rendered = loop.query(_legacy_state())

    assert "REBASELINE_REQUIRED" in rendered
    assert "stateful-serving-deployable-v1" in rendered
    assert "LEGACY kept" in rendered
    assert "old_elision" in rendered
    assert "KEPT under active objective: (none yet)" in rendered


def test_query_separates_current_and_legacy_capability_evidence(monkeypatch):
    state = _legacy_state() | {"objective_scope": loop.OBJECTIVE_SCOPE}
    monkeypatch.setattr(loop, "bottleneck_block", lambda: "== BOTTLENECK test")
    monkeypatch.setattr(loop, "frontier_block", lambda: "== FRONTIER test")
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
    }
    reverted = []
    monkeypatch.setattr(loop, "load", lambda: state)
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
