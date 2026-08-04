"""Maintainer tests for the API-novelty guardrails (redundancy + genericity)."""

import jax.numpy as jnp
import pytest

from akt.core.analysis.api_metrics import (
    _percentiles,
    duplicate_risk,
    gate_metrics,
    genericity_from_case_search,
    implementation_breadth,
    operation_graph,
    redundancy,
)


def _graph(fn, *args, **kwargs):
    return operation_graph(lambda: fn(*args), **kwargs)


def _base(a, b):
    return jnp.tanh(a @ b) + a


def _superset(a, b):
    return jnp.exp(jnp.sin(_base(a, b))) * 2.0


def test_identical_programs_are_fully_redundant_both_ways():
    a, b = jnp.ones((8, 8)), jnp.ones((8, 8))
    g1, g2 = _graph(_base, a, b), _graph(_base, a, b)
    assert redundancy(g1, g2) == 1.0
    assert redundancy(g2, g1) == 1.0


def test_shape_tweak_of_the_same_program_is_a_duplicate():
    """A tile/size re-parameterization keeps op composition and wiring — the
    rank-abstract labels must match it fully (that IS the parameter-tweak
    signature the guardrail rejects)."""
    g_small = _graph(_base, jnp.ones((8, 8)), jnp.ones((8, 8)))
    g_large = _graph(_base, jnp.ones((32, 32)), jnp.ones((32, 32)))
    assert redundancy(g_small, g_large) == 1.0
    assert redundancy(g_large, g_small) == 1.0


def test_directional_generalization_signature():
    """E reproducible by N while N exceeds E: R(E,N) high, R(N,E) low."""
    a, b = jnp.ones((8, 8)), jnp.ones((8, 8))
    g_e = _graph(_base, a, b)
    g_n = _graph(_superset, a, b)
    r_en = redundancy(g_e, g_n)
    r_ne = redundancy(g_n, g_e)
    assert r_en > 0.9
    assert r_ne < r_en

    result = duplicate_risk(g_n, {"existing": g_e})
    assert result["closest"] == "existing"
    assert result["R_en"] == r_en
    assert result["D"] == r_ne


def test_wiring_change_is_not_matched_by_op_multiset_alone():
    """Same primitive multiset, different dependency wiring => WL must mismatch
    (dependency-preserving, not bag-of-instructions)."""

    def chain(a):
        x = a * a
        y = x + x
        return y * y, y + y

    def rewired(a):
        x = a + a
        y = x * x
        return y + y, y * y

    g1 = _graph(chain, jnp.ones((4, 4)))
    g2 = _graph(rewired, jnp.ones((4, 4)))
    assert redundancy(g1, g2) < 1.0


def test_percentiles_are_order_statistics():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pct = _percentiles(scores)
    assert pct["min"] == 0.1 and pct["max"] == 1.0
    assert pct["p50"] == pytest.approx(0.5, abs=0.11)
    assert pct["p10"] <= pct["p50"] <= pct["p90"]
    assert pct["n"] == 10


def _case_search_fixture():
    return {
        "caseA": {
            "knobs": [
                {"name": "new_knob", "values": [False, True], "default": False}
            ],
            "measurements": [
                {"config": {"new_knob": False}, "latency_s": 1.00},
                {"config": {"new_knob": True}, "latency_s": 0.90},
            ],
        },
        "caseB": {
            "knobs": [
                {"name": "new_knob", "values": [False, True], "default": False}
            ],
            "measurements": [
                {"config": {"new_knob": False}, "latency_s": 1.00},
                {"config": {"new_knob": True}, "latency_s": 0.999},
            ],
        },
    }


def test_genericity_counts_each_upper_pattern_once_with_benefit_floor():
    result = genericity_from_case_search(
        _case_search_fixture(),
        control_knob="new_knob",
        relevant_case_to_callsites={
            "caseA": ["model1/siteA", "model2/siteA2"],
            "caseB": ["model1/siteB"],
        },
        delta=0.02,
    )
    # caseA benefits 10% at both its upper patterns; caseB only 0.1% (< delta).
    assert result["covered_patterns"] == ["model1/siteA", "model2/siteA2"]
    assert result["G"] == pytest.approx(2 / 3)
    assert result["per_pattern"]["model1/siteB"]["benefit"] < 0.02


def test_implementation_breadth_is_reported_separately():
    paths = {
        "fam.knob": [
            {"path": "k.py", "function": "entry"},
            {"path": "k.py", "function": "inner"},
        ]
    }
    assert implementation_breadth(paths) == 2
    assert implementation_breadth({}) == 0


def test_gate_metrics_rejects_a_parameter_tweak_on_a_real_case():
    """End-to-end on the real GLA case: promoting an EXISTING axis value is a
    parameter tweak — its program is reproducible from the incumbent one-knob
    population (D ~= 1) and must fail the redundancy guardrail under any
    plausible calibrated cut."""
    baseline = {
        "wl_iters": 2,
        "weight_mode": "count",
        "delta": 0.02,
        "fingerprint": "test",
        "policy": {
            "redundancy_percentile": 10,
            "generalization_percentile": 50,
            "genericity_percentile": 50,
        },
        "redundancy": {
            "per_family": {"gla": {"p10": 0.6, "p50": 0.9, "n": 10}},
            "global": {"p10": 0.6, "p50": 0.9, "n": 10},
        },
        "genericity": {"global": {"p50": 0.5, "n": 4}},
    }
    case_search = {
        "gla:seq512_h8": {
            "knobs": [
                {"name": "compact_alignment", "values": [False, True], "default": False}
            ],
            "measurements": [
                {"config": {"compact_alignment": False}, "latency_s": 1.0},
                {"config": {"compact_alignment": True}, "latency_s": 0.8},
            ],
        }
    }
    result = gate_metrics(
        baseline=baseline,
        capability_controls=[
            {
                "case_id": "gla:seq512_h8",
                "knob": "compact_alignment",
                "selected_value": True,
            }
        ],
        case_search=case_search,
        relevant_case_to_callsites={
            "gla:seq512_h8": ["tiny-linear-serving/gla-short"]
        },
    )
    assert result["per_case"], result["errors"]
    worst = result["worst_duplicate_risk"]
    assert worst["D"] > 0.9          # reproducible from the incumbent population
    assert worst["redundancy_ok"] is False
    assert result["redundancy_ok"] is False
    assert result["ok"] is False
    # genericity side: full coverage at 20% benefit clears its cut independently
    assert result["genericity_min"] == 1.0
    assert result["genericity_ok"] is True
