import time

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
