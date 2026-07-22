import ast
from pathlib import Path

import pytest

from akt.benchmark.live_model_candidates import (
    KIMI_CANDIDATE,
    KIMI_POLICY_SOURCE_CALLSITE,
    candidate_eligibility,
    resolve_kimi_policy,
    stable_json_hash,
)


def synthetic_summary(*, backend="tpu", all_correct=True):
    hardware_ok = backend == "tpu"
    return {
        "all_correct": all_correct,
        "n_deferred": 0 if all_correct else 1,
        "target_hardware_ok": hardware_ok,
        "target_hardware": {
            "ok": hardware_ok,
            "backend": backend,
            "pallas_interpret": False,
            "fingerprint": "tpu-v6e-4-test",
            "devices": [
                {"platform": backend, "kind": "test", "id": index} for index in range(4)
            ],
        },
        "models": [],
        "case_search": {},
    }


def gmm_summary(
    *,
    selected_buffer_count=4,
    incumbent_buffer_count=3,
    backend="tpu",
    all_correct=True,
):
    summary = synthetic_summary(backend=backend, all_correct=all_correct)
    summary["models"].append(
        {
            "model": "tiny-dense-serving",
            "selected_plan": {
                KIMI_POLICY_SOURCE_CALLSITE: {
                    "buffer_count": selected_buffer_count,
                    "other_control": 2,
                    "runner_scratch": True,
                }
            },
            "incumbent_plan": {
                KIMI_POLICY_SOURCE_CALLSITE: {
                    "buffer_count": incumbent_buffer_count,
                    "other_control": 1,
                    "runner_scratch": False,
                }
            },
        }
    )
    summary["case_search"]["gmm_v2:m512_k1024_n1024_g8"] = {
        "knobs": [
            {
                "name": "buffer_count",
                "values": [2, 3, 4],
                "default": 3,
                "programmer_control": "gmm_v2.buffer_count",
                "elevated_by": "gmm_v2_buffer_count",
            },
            {
                "name": "runner_scratch",
                "default": False,
                "programmer_control": None,
                "elevated_by": "gmm_v2_buffer_count",
            },
            {
                "name": "other_control",
                "default": 1,
                "programmer_control": "gmm_v2.other_control",
                "elevated_by": "another_capability",
            },
        ]
    }
    return summary


def test_pure_candidate_module_has_no_runtime_or_model_imports():
    source = Path(__file__).with_name("live_model_candidates.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])

    assert imported.isdisjoint(
        {"jax", "transformers", "huggingface_hub", "sgl_jax", "requests", "yaml"}
    )


def test_direct_policy_contains_only_the_changed_elevated_gmm_control():
    bundle = resolve_kimi_policy(gmm_summary(), "gmm_v2_buffer_count")

    assert bundle["applicable"] is True
    assert bundle["selected"] == {
        "gmm_v2": {
            "default": {},
            "rules": [{"when": {}, "set": {"buffer_count": 4}}],
        }
    }
    assert bundle["incumbent"]["gmm_v2"]["rules"][0]["set"] == {"buffer_count": 3}
    assert bundle["coverage"] == {
        "only_programmer_controls": True,
        "controls": [
            {
                "knob": "buffer_count",
                "programmer_control": "gmm_v2.buffer_count",
                "default": 3,
                "selected": 4,
                "incumbent": 3,
                "changed": True,
            }
        ],
        "capability": "gmm_v2_buffer_count",
        "capability_applies": True,
        "evaluation_intent": "direct-capability",
    }
    assert bundle["hashes"]["selected"] == stable_json_hash(bundle["selected"])
    assert bundle["hashes"]["selected"] != bundle["hashes"]["incumbent"]
    assert KIMI_CANDIDATE.route_id == bundle["source"]["route_id"]


@pytest.mark.parametrize(
    ("summary", "capability", "reason"),
    [
        (gmm_summary(), None, "no_capability"),
        (gmm_summary(), "unrelated_control", "no_matching_gmm_v2_control"),
        (
            gmm_summary(selected_buffer_count=3, incumbent_buffer_count=3),
            "gmm_v2_buffer_count",
            "capability_policy_unchanged",
        ),
    ],
)
def test_non_direct_actions_are_explicitly_not_applicable(summary, capability, reason):
    resolution = resolve_kimi_policy(summary, capability)

    assert resolution["applicable"] is False
    assert resolution["reason"] == reason


def test_matching_control_with_missing_source_plan_fails_closed():
    summary = gmm_summary()
    summary["models"] = []

    with pytest.raises(ValueError, match="expected exactly one"):
        resolve_kimi_policy(summary, "gmm_v2_buffer_count")


def test_eligibility_requires_compiled_single_host_tpu_inventory():
    eligible = candidate_eligibility(gmm_summary())
    assert eligible["eligible"] is True
    assert eligible["observed"]["device_indexes"] == [0, 1, 2, 3]

    gpu = candidate_eligibility(gmm_summary(backend="gpu"))
    assert gpu["eligible"] is False
    assert "backend_mismatch" in gpu["reason_codes"]
    assert "target_hardware_not_verified" in gpu["reason_codes"]

    interpreted = gmm_summary()
    interpreted["target_hardware"]["pallas_interpret"] = True
    result = candidate_eligibility(interpreted)
    assert result["eligible"] is False
    assert "interpret_mode" in result["reason_codes"]
