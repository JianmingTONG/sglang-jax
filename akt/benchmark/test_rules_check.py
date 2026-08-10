"""The AKT rule registry (akt/RULES.md) must hold on every tree state.

Runs the same checker as `akt/core/verify/rules_check.py`; a modification that
violates a rule fails here before it ever reaches a campaign.
"""

import pytest

from akt.core.verify.rules_check import CHECKS, run_all


def test_rule_registry_is_documented():
    from pathlib import Path

    rules_md = Path(__file__).resolve().parents[2] / "akt/RULES.md"
    text = rules_md.read_text()
    for rule_id in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"):
        assert f"## {rule_id} " in text, f"{rule_id} missing from RULES.md"
    # every check function is referenced by name in the registry doc
    for name, check in CHECKS:
        assert check.__name__ in text, (
            f"{check.__name__} not referenced in RULES.md — keep the doc and "
            "the checker in the same commit"
        )


@pytest.mark.parametrize("name,check", CHECKS, ids=[n for n, _c in CHECKS])
def test_rule_holds(name, check):
    ok, evidence = check()
    assert ok, f"{name} violated: {evidence}"


def test_checker_treats_a_crashed_check_as_failure(monkeypatch):
    import akt.core.verify.rules_check as rc

    def boom():
        raise RuntimeError("synthetic crash")

    monkeypatch.setattr(rc, "CHECKS", [("synthetic", boom)])
    results = rc.run_all()
    assert results[0][1] is False and "crashed" in results[0][2]
