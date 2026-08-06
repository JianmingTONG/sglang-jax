"""Maintainer tests for the delta-residual API-novelty guardrail (v2).

Covers the two-pass subtraction (instance cover + per-op ancestry match), the
total roofline taxonomy, coherence (incl. contain-edge union), the zero-embed
self-check, the real-case noise-floor verdict, gate wiring, and the unchanged
v1 genericity / implementation-breadth guardrails.
"""

import jax.numpy as jnp
import pytest

from akt.core.analysis.api_metrics import (
    _percentiles,
    catalog_for_case,
    classify_labels,
    classify_primitive,
    delta_verdict,
    gate_metrics,
    genericity_from_case_search,
    implementation_breadth,
    instance_cover,
    operation_graph,
    subtract_catalog,
    unclassified_primitives,
)

X = jnp.ones((8, 8), jnp.float32)
W = jnp.ones((8, 8), jnp.float32)


def _graph(fn):
    return operation_graph(fn)


def _prog_a():  # tanh(x @ w): dot_general, tanh
    return jnp.tanh(X @ W)


def _prog_b(t=X):  # exp(t) * 2: exp, mul
    return jnp.exp(t) * 2


def _prog_c(t=X):  # cumsum(t) + 1
    return jnp.cumsum(t) + 1


def _catalog(order=("a", "b", "c")):
    built = {
        "a": _graph(_prog_a),
        "b": _graph(lambda: _prog_b()),
        "c": _graph(lambda: _prog_c()),
    }
    return {key: built[key] for key in order}


# ------------------------------------------------------------- Δ subtraction


def test_composition_of_catalog_programs_is_a_full_duplicate():
    """N = c(b(a(x,w))) is pure composition of existing APIs: the instance
    cover strikes all three whole programs and Δ is empty."""

    def n():
        return _prog_c(_prog_b(_prog_a()))

    record = subtract_catalog(_graph(n), _catalog())
    assert record["delta_ops"] == 0
    assert record["mass"] == {"compute": 0, "memory": 0}
    assert sorted(i["program"] for i in record["struck_instances"]) == ["a", "b", "c"]
    assert record["zero_embed"] == 0


def test_intermediate_tap_leaves_only_the_novel_wiring_connected():
    """N taps a's intermediate output (t) around b: both instances are still
    struck; Δ is exactly the new {mul, sin, add} wiring, one connected
    component of size 3."""

    def n():
        t = _prog_a()
        u = _prog_b(t)
        return u * t + jnp.sin(t)

    record = subtract_catalog(_graph(n), _catalog())
    assert sorted(i["program"] for i in record["struck_instances"]) == ["a", "b"]
    assert record["delta_ops"] == 3
    assert sorted(l.split("|")[0] for l in record["graph"]["labels"]) == [
        "add",
        "mul",
        "sin",
    ]
    assert record["coherence"]["components"] == 1
    assert record["coherence"]["largest"] == 3
    assert record["zero_embed"] == 0


def test_interleaved_binding_lets_instances_anchor_on_produced_values():
    """N = b(a(x,w)): b's instance input binds to a's output (not a graph
    input) and both instances are still struck — Δ empty."""

    def n():
        return _prog_b(_prog_a())

    record = subtract_catalog(_graph(n), {"a": _graph(_prog_a), "b": _graph(_prog_b)})
    assert record["delta_ops"] == 0
    assert sorted(i["program"] for i in record["struck_instances"]) == ["a", "b"]
    assert record["zero_embed"] == 0


def test_subtraction_is_deterministic_under_catalog_insertion_order():
    def n():
        t = _prog_a()
        u = _prog_b(t)
        return u * t + jnp.sin(t)

    n_graph = _graph(n)
    first = subtract_catalog(n_graph, _catalog(order=("a", "b", "c")))
    second = subtract_catalog(n_graph, _catalog(order=("c", "b", "a")))
    assert first["delta_ops"] == second["delta_ops"]
    assert sorted(
        (i["program"], i["ops"]) for i in first["struck_instances"]
    ) == sorted((i["program"], i["ops"]) for i in second["struck_instances"])
    assert first["graph"]["labels"] == second["graph"]["labels"]
    assert first["zero_embed"] == second["zero_embed"] == 0


def test_single_op_program_cannot_cover_a_lone_matching_op():
    """Whole-instance soundness: a 1-op catalog program is below the
    min_instance_ops=2 default and must never open an instance cover — only a
    whole multi-op embedding proves reproducibility."""
    y = jnp.ones((8, 8), jnp.float32)
    cat = {"only_exp": _graph(lambda: jnp.exp(X))}

    # The cover skips the 1-op program even on a literally matching op...
    lone = _graph(lambda: jnp.exp(y) + y)
    struck, instances = instance_cover(lone, cat)
    assert struck == set() and instances == []
    # ...while dropping the guard WOULD strike it (the guard is load-bearing).
    struck_lo, instances_lo = instance_cover(lone, cat, min_instance_ops=1)
    assert len(struck_lo) == 1 and instances_lo[0]["program"] == "only_exp"

    # With a produced input the op's ancestry differs from the catalog's exp,
    # so pass 2 cannot match it either: exp ends in Δ.
    produced = _graph(lambda: jnp.exp(y * y) + y)
    record = subtract_catalog(produced, cat)
    assert record["struck_instances"] == [] and record["n_struck"] == 0
    assert "exp" in [l.split("|")[0] for l in record["graph"]["labels"]]
    assert record["zero_embed"] == 0


def test_same_op_multiset_with_different_wiring_is_not_subtracted():
    """Ancestry soundness: (a*a)+(a*a) vs catalog (a+a)*(a+a) share no
    derivation despite overlapping primitive names — Δ stays nonzero."""
    cat = {"prog": _graph(lambda: (X + X) * (X + X))}
    record = subtract_catalog(_graph(lambda: (X * X) + (X * X)), cat)
    assert record["delta_ops"] > 0
    assert record["struck_instances"] == []
    assert record["zero_embed"] == 0


# ---------------------------------------------------------------- taxonomy


def test_taxonomy_is_total_over_primitives():
    assert classify_primitive("dot_general") == "compute"
    for name in (
        "reshape",
        "squeeze",
        "bitcast_convert",
        "gather",
        "transpose",
        "broadcast_in_dim",
    ):
        assert classify_primitive(name) == "memory", name
    for name in ("pjit", "pallas_call", "scan"):
        assert classify_primitive(name) == "container", name
    # Unknown primitives default to compute (fail-substantive) AND are audited.
    assert classify_primitive("frobnicate_v2") == "compute"
    labels = ["frobnicate_v2|float32r2", "add|float32r2", "reshape|float32r1"]
    assert classify_labels(labels) == ["compute", "compute", "memory"]
    assert unclassified_primitives(labels) == ["frobnicate_v2"]


def test_contain_edges_unify_delta_coherence_across_container_boundaries():
    """Two Δ ops connected ONLY by a contain edge form one component: the
    coherence union walks data edges AND container membership."""
    synthetic = {
        "labels": ["cumsum|float32r2", "cumsum|float32r2"],
        "weights": [1.0, 1.0],
        "edges": [],
        "in_tokens": [["⊥float32r2"], ["⊥float32r2"]],
        "contain": [(0, 1)],
    }
    record = subtract_catalog(synthetic, {})
    assert record["delta_ops"] == 2
    assert record["coherence"]["components"] == 1
    assert record["coherence"]["largest"] == 2
    assert record["zero_embed"] == 0


# ------------------------------------------------------------ v1 guardrails


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


# ------------------------------------------------------------ real-case gate


@pytest.fixture(scope="module")
def fast_cases():
    from akt.benchmark.suites import load_cases

    return {case.case_id: case for case in load_cases("fast")}


@pytest.fixture(scope="module")
def gla_catalog(fast_cases):
    return catalog_for_case(fast_cases, "gla:seq512_h8", exclude_knob=None)


def test_promoting_an_existing_axis_value_stays_under_the_noise_floor(
    fast_cases, gla_catalog
):
    """chunk_size=128 with its own program removed from the catalog is still
    reconstructed from the remaining one-knob population: Δ mass sits at or
    below the measured noise floor and the verdict fails."""
    excluded = "gla:seq512_h8:chunk_size=128"
    assert excluded in gla_catalog
    catalog = {pid: g for pid, g in gla_catalog.items() if pid != excluded}
    verdict = delta_verdict(
        fast_cases["gla:seq512_h8"],
        "chunk_size",
        128,
        catalog,
        noise_floor=4,
        coherence_min=3,
        epsilon=2,
    )
    assert verdict["mass_total"] <= 4
    assert verdict["pass"] is False
    assert verdict["zero_embed"] == 0


def test_pure_jax_reference_is_novel_relative_to_the_full_catalog(
    fast_cases, gla_catalog
):
    """Sanity in the accepting direction: the case's pure-JAX reference is a
    genuinely different program — a large, coherent Δ survives subtraction of
    the entire kernel catalog."""
    case = fast_cases["gla:seq512_h8"]
    inputs = case.make_inputs()
    n_graph = operation_graph(lambda: case.reference(inputs))
    record = subtract_catalog(n_graph, gla_catalog)
    assert record["delta_pct"] > 0.3
    assert record["coherence"]["largest"] >= 3
    assert record["zero_embed"] == 0


def test_gate_metrics_rejects_a_parameter_tweak_on_a_real_case():
    """End-to-end wiring: promoting an EXISTING axis value (gla
    compact_alignment=True) is a parameter tweak — Δ mass at/below the pinned
    noise floor fails the redundancy guardrail even though its genericity is
    perfect."""
    baseline = {
        "weight_mode": "count",
        "fingerprint": "test",
        "delta": {
            "noise_floor": {"p95": 4.0, "n": 10},
            "epsilon_ops": 2,
            "coherence_min": 3,
            "noise_percentile": 95,
        },
        "genericity": {"global": {"p50": 0.5, "n": 4}},
        "policy": {"genericity_percentile": 50},
    }
    case_search = {
        "gla:seq512_h8": {
            "knobs": [
                {
                    "name": "compact_alignment",
                    "values": [False, True],
                    "default": False,
                }
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
    assert not result["errors"], result["errors"]
    assert result["redundancy_ok"] is False  # a real tweak: mass <= noise floor
    assert result["genericity_ok"] is True  # 20% benefit at the one callsite
    assert result["ok"] is False
    per_case = result["delta"]["per_case"]
    assert per_case and per_case[0]["pass"] is False
    assert per_case[0]["mass_total"] <= 4.0
    assert per_case[0]["zero_embed"] == 0
    assert result["baseline_fingerprint"] == "test"
