from dataclasses import replace
import functools
import importlib

import jax
import jax.numpy as jnp
import pytest

from akt.benchmark.gates.eval import eval_case, native_verdict_ok
from akt.benchmark.runners.base import DesignSpace, KernelCase, Knob
from akt.benchmark.suites import (
    EXPECTED_CASES,
    SuiteContractError,
    _validate_case_contract,
    load_cases,
)
from akt.core.evolve import loop
from akt.core.evolve.loop import (
    FROZEN,
    _frozen_fingerprint,
    _implementation_changes,
    validate_manifest_scope,
)


def test_full_suite_matches_frozen_case_inventory():
    expected = {
        case_id
        for runner_cases in EXPECTED_CASES.values()
        for case_id in runner_cases
    }
    assert {case.case_id for case in load_cases("full")} == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda case: replace(case, reference=lambda _inp: None),
        lambda case: replace(case, atol=case.atol * 1000),
        lambda case: replace(case, check_out=lambda _out: ()),
        lambda case: replace(case, regime_pref=("tpu",)),
        lambda case: replace(
            case,
            make_inputs=functools.partial(case.make_inputs.func, 16, 1),
        ),
        lambda case: replace(case, native_test=None),
    ],
)
def test_runner_cannot_weaken_frozen_contract(mutate):
    case = next(case for case in load_cases("fast") if case.kernel_id == "gla")
    refs = importlib.import_module("akt.benchmark.refs.gla")
    with pytest.raises(SuiteContractError):
        _validate_case_contract(mutate(case), refs, EXPECTED_CASES["gla"][case.case_id])


def test_native_checks_fail_closed():
    assert native_verdict_ok([])
    assert native_verdict_ok([{"status": "passed"}])
    for status in ("failed", "error", "no-tests"):
        assert not native_verdict_ok([{"status": status}])


def test_case_fails_when_any_advertised_valid_config_is_incorrect():
    case = KernelCase(
        kernel_id="contract_test",
        shape_id="scalar",
        make_inputs=lambda: {"x": jnp.asarray([1.0])},
        run=lambda inp, cfg: inp["x"] + (1.0 if cfg["wrong"] else 0.0),
        reference=lambda inp: inp["x"],
        space=DesignSpace(
            [
                Knob(
                    "wrong",
                    [False, True],
                    default=False,
                    programmer_control="contract_test.wrong",
                )
            ]
        ),
    )
    result = eval_case(case, runs=1, native=False)
    assert result["correct"] is False
    assert result["n_correct"] == 1
    assert result["n_valid"] == 2
    assert result["incorrect_configs"]


def test_local_default_failure_is_not_mislabeled_as_tpu_deferred():
    case = KernelCase(
        kernel_id="contract_test",
        shape_id="broken_local",
        make_inputs=lambda: {"x": jnp.asarray([1.0])},
        run=lambda inp, _cfg: inp["x"] + 1.0,
        reference=lambda inp: inp["x"],
        space=DesignSpace([Knob("unused", [False], default=False)]),
        regime_pref=("cpu-interpret",),
    )

    result = eval_case(case, runs=1, native=False)

    assert result["correct"] is False
    assert result["regime"] != "tpu-deferred"


def test_tpu_only_default_failure_is_deferred_off_tpu():
    case = KernelCase(
        kernel_id="contract_test",
        shape_id="tpu_only",
        make_inputs=lambda: {"x": jnp.asarray([1.0])},
        run=lambda inp, _cfg: inp["x"] + 1.0,
        reference=lambda inp: inp["x"],
        space=DesignSpace([Knob("unused", [False], default=False)]),
        regime_pref=("tpu",),
    )

    result = eval_case(case, runs=1, native=False)

    if jax.default_backend() == "tpu":
        assert result["correct"] is False
    else:
        assert result["correct"] is None
        assert result["regime"] == "tpu-deferred"


def test_runner_only_capability_cannot_determine_deployment_incumbent():
    case = KernelCase(
        kernel_id="contract_test",
        shape_id="runner_only",
        make_inputs=lambda: {"x": jnp.asarray([1.0])},
        run=lambda inp, cfg: inp["x"] + (1.0 if cfg["shortcut"] else 0.0),
        reference=lambda inp: inp["x"],
        space=DesignSpace(
            [
                Knob(
                    "shortcut",
                    [False, True],
                    default=False,
                    elevated_by="output_only_shortcut",
                )
            ]
        ),
    )

    result = eval_case(case, runs=1, native=False)

    assert result["research_space_size"] == 2
    assert result["space_size"] == 1
    assert result["n_valid"] == 1
    assert result["best_config"] == {"shortcut": False}


def test_native_test_tree_and_shared_reference_symbols_are_fingerprinted():
    assert "python/sgl_jax/test/" in FROZEN
    assert "akt/board/index.html" in FROZEN
    keys = _frozen_fingerprint()
    assert any(key.endswith("::frozen-reference-ast") for key in keys)
    assert not any("__pycache__" in key or key.endswith(".pyc") for key in keys)
    assert not any(value.startswith("ERROR:") for value in keys.values())


def test_manifest_scope_must_exactly_match_oracle_edits(monkeypatch):
    manifest = {
        "files_touched": [
            "python/kernel.py",
            "akt/core/evolve/capabilities/example.json",
        ]
    }
    changed = [
        "akt/board/evolve_status.json",
        "python/kernel.py",
        "akt/core/evolve/capabilities/example.json",
    ]
    monkeypatch.setattr(loop, "_worktree_changed", lambda: changed)
    validate_manifest_scope(manifest)

    changed.append("python/undeclared.py")
    with pytest.raises(ValueError, match="undeclared"):
        validate_manifest_scope(manifest)


def test_only_generated_campaign_records_are_bookkeeping(monkeypatch):
    monkeypatch.setattr(
        loop,
        "_worktree_changed",
        lambda: [
            "akt/optimization_history/evolve_state.json",
            "akt/optimization_history/evolve_history.jsonl",
            "akt/board/evolve_status.json",
            "akt/board/index.html",
        ],
    )

    assert _implementation_changes() == ["akt/board/index.html"]
