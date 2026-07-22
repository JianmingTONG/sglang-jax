import ast
import copy
import json
import subprocess
from pathlib import Path

import pytest

import akt.core.evolve.capability_contract as capability_contract
from akt.benchmark.runners.base import DesignSpace, Knob
from akt.core.evolve.action_catalog import (
    ActionCatalogError,
    action_semantic_record,
    action_catalog_context,
    load_action_graph,
)
from akt.core.evolve.capability_contract import (
    _argument_forwarding_path,
    validate_action_reference,
    validate_capability_contract,
    validate_space_extension,
)


ROOT = Path(__file__).resolve().parents[2]


def _expression_ast(source):
    return ast.dump(ast.parse(source, mode="eval").body, include_attributes=False)


def _head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _moe_manifest(name="moe_interleave_control"):
    context = action_catalog_context(ROOT)
    action = next(
        item
        for item in context["actions"]
        if item["gap_id"] == "fused_moe/v2:interleave_bt:schedule-toggle"
    )
    return {
        "name": name,
        "action_graph_fingerprint": context["fingerprint"],
        "gap_id": action["gap_id"],
        "estimate": {
            "model": "tiny-moe-serving",
            "callsite": "tiny-moe-serving/expert-v2",
            "baseline_share_pct": 20.0,
            "expected_relief_pct": 3.0,
            "reasoning": "Existing schedule toggle dominates this frozen callsite.",
        },
        "search_dimensions": [
            {
                "control": "moe_v2.interleave_bt",
                "kernel_function": "fused_ep_moe_v2",
                "consumer": "python/sgl_jax/srt/layers/fused_moe.py",
            }
        ],
    }


def test_unknown_gap_id_cannot_be_replaced_by_free_text():
    evidence = validate_capability_contract(
        {
            "name": "fake_capability",
            "action_graph_fingerprint": action_catalog_context(ROOT)["fingerprint"],
            "gap_id": "made-up-gap",
            "search_dimensions": [],
        },
        {},
        ROOT,
        _head(),
    )
    assert evidence["ok"] is False
    assert any("not an executable red-link action" in error for error in evidence["errors"])


def test_action_catalog_fails_closed_when_a_red_link_is_missing(tmp_path):
    graph = json.loads(
        (ROOT / "akt/core/analysis/flexgraph_generated.json").read_text()
    )
    proven_id = action_catalog_context(ROOT)["actions"][0]["gap_id"]
    graph["action_edges"] = [
        edge for edge in graph["action_edges"] if edge["gap_id"] != proven_id
    ]
    path = tmp_path / "akt/core/analysis/flexgraph_generated.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(graph))

    with pytest.raises(ActionCatalogError, match="cover every executable open finding"):
        load_action_graph(tmp_path)


def test_action_catalog_accepts_only_the_current_contract_version(tmp_path):
    graph = json.loads(
        (ROOT / "akt/core/analysis/flexgraph_generated.json").read_text()
    )
    graph["action_contract_version"] = 1
    path = tmp_path / "akt/core/analysis/flexgraph_generated.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(graph))

    with pytest.raises(ActionCatalogError, match="unsupported action contract"):
        load_action_graph(tmp_path)


def test_known_null_incumbent_is_distinct_from_unknown(tmp_path):
    graph = json.loads(
        (ROOT / "akt/core/analysis/flexgraph_generated.json").read_text()
    )
    gap_id = graph["action_edges"][0]["gap_id"]
    gap = next(item for item in graph["gaps"] if item["gap_id"] == gap_id)
    gap["incumbent_value"] = None
    gap["incumbent_value_known"] = True
    gap["candidate_values"] = [None, 3]
    path = tmp_path / "akt/core/analysis/flexgraph_generated.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(graph))

    _loaded, catalog = load_action_graph(tmp_path)
    assert catalog[gap_id]["incumbent_value"] is None

    gap["incumbent_value_known"] = False
    path.write_text(json.dumps(graph))
    with pytest.raises(ActionCatalogError, match="non-action findings"):
        load_action_graph(tmp_path)


def test_semantic_action_record_ignores_volatile_evidence_only():
    _graph, catalog = load_action_graph(ROOT)
    action = next(iter(catalog.values()))
    moved = copy.deepcopy(action)
    source_path, _line = moved["evidence"].rsplit(":", 1)
    moved["evidence"] = f"{source_path}:9999"
    moved["detail"] = "rewritten presentation text"

    assert action_semantic_record(moved) == action_semantic_record(action)

    moved["incumbent_value"] = float(action["incumbent_value"])
    assert action_semantic_record(moved) != action_semantic_record(action)


def test_compute_tile_observation_cannot_become_an_action(tmp_path):
    graph = json.loads(
        (ROOT / "akt/core/analysis/flexgraph_generated.json").read_text()
    )
    gap_id = graph["action_edges"][0]["gap_id"]
    gap = next(item for item in graph["gaps"] if item["gap_id"] == gap_id)
    gap["category"] = "compute-tile-pinned"
    path = tmp_path / "akt/core/analysis/flexgraph_generated.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(graph))

    with pytest.raises(ActionCatalogError, match="non-action findings"):
        load_action_graph(tmp_path)


def test_action_context_is_canonical_and_excludes_missing_implementations():
    first = action_catalog_context(ROOT)
    second = action_catalog_context(ROOT)

    assert first == second
    assert len(first["fingerprint"]) == 64
    assert len(first["actions"]) == 5
    assert all(
        action["category"] not in {"missing-tuned-table", "dead-tuned-table"}
        for action in first["actions"]
    )
    gmm_v2 = next(
        action
        for action in first["actions"]
        if action["gap_id"] == "megablox_gmm_kernel:buffer_count:pipeline-depth"
    )
    assert gmm_v2["kernel_ids"] == ["gmm_v2"]
    assert gmm_v2["model_callsites"] == ["tiny-dense-serving/dense-projection"]
    assert gmm_v2["source_function"] == "generate_block_specs"
    assert gmm_v2["incumbent_value"] == 3
    assert gmm_v2["candidate_values"] == [2, 3, 4]
    fused_mlp = next(
        action
        for action in first["actions"]
        if action["gap_id"] == "fused_mlp:buffer_count:pipeline-depth"
    )
    assert fused_mlp["source_function"] == "mlp_kernel_main"
    assert fused_mlp["incumbent_value"] == 3
    interleave = next(
        action
        for action in first["actions"]
        if action["gap_id"] == "fused_moe/v2:interleave_bt:schedule-toggle"
    )
    assert interleave["incumbent_value"] is True
    assert interleave["candidate_values"] == [True, False]
    assert interleave["source_sink"]["assignments"] == ["use_gather_bank"]
    assert set(interleave["source_sink"]["expression_asts"]) == {
        "use_gather_bank"
    }
    prefetch = next(
        action
        for action in first["actions"]
        if action["gap_id"]
        == "fused_moe/v2:cross_expert_prefetch_mode:schedule-toggle"
    )
    assert prefetch["candidate_values"] == ["full", "none", "w13"]
    assert prefetch["source_sink"]["assignments"] == [
        "enable_cross_expert_prefetch",
        "full_cross_expert_prefetch",
    ]
    assert set(prefetch["source_sink"]["expression_asts"]) == set(
        prefetch["source_sink"]["assignments"]
    )


def test_static_forwarding_path_follows_import_alias_and_partial(tmp_path):
    source_path = tmp_path / "python/demo/source.py"
    wrapper_path = tmp_path / "python/demo/wrapper.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        """
import functools

def source_axis(*, buffer_count):
    return buffer_count

def public_entry(*, buffer_count):
    return functools.partial(source_axis, buffer_count=buffer_count)
"""
    )
    wrapper_path.write_text(
        """
from demo.source import public_entry as selected_backend

def dispatch(*, buffer_count):
    return selected_backend(buffer_count=buffer_count)
"""
    )

    path = _argument_forwarding_path(
        tmp_path,
        start_path="python/demo/wrapper.py",
        start_function="dispatch",
        start_argument="buffer_count",
        target_path="python/demo/source.py",
        target_function="source_axis",
        target_argument="buffer_count",
    )

    assert [step["function"] for step in path] == [
        "dispatch",
        "public_entry",
        "source_axis",
    ]


def test_static_forwarding_rejects_transforms_and_requires_the_mined_sink(tmp_path):
    source = tmp_path / "python/demo/source.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
def source_axis(*, buffer_count):
    return Buffered(buffer_count=not buffer_count)
"""
    )

    arguments = dict(
        repo=tmp_path,
        start_path="python/demo/source.py",
        start_function="source_axis",
        start_argument="buffer_count",
        target_path="python/demo/source.py",
        target_function="source_axis",
        target_argument="buffer_count",
        target_sink={"callee": "Buffered", "argument": "buffer_count"},
    )
    assert _argument_forwarding_path(**arguments) is None

    source.write_text(
        """
def source_axis(*, buffer_count):
    depth = buffer_count
    return Buffered(buffer_count=depth)
"""
    )
    assert _argument_forwarding_path(**arguments)
    assert _argument_forwarding_path(
        **(arguments | {"target_sink": {"callee": "Other", "argument": "buffer_count"}})
    ) is None


def test_schedule_forwarding_requires_the_mined_assignment_sink(tmp_path):
    source = tmp_path / "python/demo/source.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
def source_axis(*, interleave_bt):
    ignored = interleave_bt
    use_gather_bank = True
    return use_gather_bank
"""
    )
    arguments = dict(
        repo=tmp_path,
        start_path="python/demo/source.py",
        start_function="source_axis",
        start_argument="interleave_bt",
        target_path="python/demo/source.py",
        target_function="source_axis",
        target_argument="interleave_bt",
        target_sink={
            "assignments": ["use_gather_bank"],
            "expression_asts": {
                "use_gather_bank": _expression_ast("interleave_bt and True")
            },
        },
    )
    assert _argument_forwarding_path(**arguments) is None

    source.write_text(
        """
def source_axis(*, interleave_bt):
    use_gather_bank = interleave_bt and True
    return use_gather_bank
"""
    )
    assert _argument_forwarding_path(**arguments)


def test_schedule_forwarding_requires_every_mined_assignment_sink(tmp_path):
    source = tmp_path / "python/demo/source.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
def source_axis(*, mode):
    enabled = mode != "none"
    full = True
    return enabled, full
"""
    )
    arguments = dict(
        repo=tmp_path,
        start_path="python/demo/source.py",
        start_function="source_axis",
        start_argument="mode",
        target_path="python/demo/source.py",
        target_function="source_axis",
        target_argument="mode",
        target_sink={
            "assignments": ["enabled", "full"],
            "expression_asts": {
                "enabled": _expression_ast('mode != "none"'),
                "full": _expression_ast('mode == "full"'),
            },
        },
    )
    assert _argument_forwarding_path(**arguments) is None

    source.write_text(
        """
def source_axis(*, mode):
    enabled = mode != "none"
    full = mode == "full"
    return enabled, full
"""
    )
    assert _argument_forwarding_path(**arguments)

    source.write_text(
        """
def source_axis(*, mode):
    enabled = mode != "none"
    full = mode == "full" and False
    return enabled, full
"""
    )
    assert _argument_forwarding_path(**arguments) is None


def test_manifest_foreign_key_derives_graph_owned_metadata():
    context = action_catalog_context(ROOT)
    action = next(
        item
        for item in context["actions"]
        if item["gap_id"] == "fused_mlp:buffer_count:pipeline-depth"
    )
    manifest = {
        "action_graph_fingerprint": context["fingerprint"],
        "gap_id": action["gap_id"],
    }

    result = validate_action_reference(manifest, ROOT)
    assert result["ok"] is True
    assert result["source_evidence"] == action["source_evidence"]
    assert result["affected_model_callsites"] == action["model_callsites"]

    spoofed = manifest | {
        "source_evidence": {"path": "invented.py", "line": 1},
        "affected_model_callsites": ["invented/callsite"],
    }
    result = validate_action_reference(spoofed, ROOT)
    assert result["ok"] is True
    assert result["source_evidence"] == action["source_evidence"]
    assert result["affected_model_callsites"] == action["model_callsites"]

    stale = copy.deepcopy(manifest)
    stale["action_graph_fingerprint"] = "0" * 64
    result = validate_action_reference(stale, ROOT)
    assert result["ok"] is False
    assert any("action_graph_fingerprint" in error for error in result["errors"])


def test_open_non_model_gap_remains_visible_but_is_not_selectable():
    graph, catalog = load_action_graph(ROOT)
    gap = next(
        item
        for item in graph["gaps"]
        if item["gap_id"] == "grouped_topk/v1:bt:shape-pinned-tile"
    )
    evidence = validate_capability_contract(
        {
            "name": "topk_tile",
            "action_graph_fingerprint": action_catalog_context(ROOT)["fingerprint"],
            "gap_id": gap["gap_id"],
            "search_dimensions": [],
        },
        {},
        ROOT,
        _head(),
    )
    assert gap["open"] is True
    assert gap["eligible_for_current_gate"] is False
    assert gap["gap_id"] not in catalog
    assert gap["action_node"] is None
    assert evidence["ok"] is False
    assert any("not an executable red-link action" in error for error in evidence["errors"])


def test_space_extension_preserves_every_old_config_at_the_new_default():
    manifest = _moe_manifest("moe_new_interleave")
    summary = {
        "case_search": {
            "moe_v2:t32_e8_k2_h512_i512": {
                "knobs": [
                    {
                        "name": "interleave_bt",
                        "default": True,
                        "elevated_by": "moe_new_interleave",
                    }
                ],
                "measurements": [
                    {"config": {"bt": 32, "interleave_bt": True}},
                    {"config": {"bt": 32, "interleave_bt": False}},
                ],
            }
        }
    }
    incumbent = {"moe_v2:t32_e8_k2_h512_i512": [{"bt": 32}]}

    evidence = validate_space_extension(manifest, summary, incumbent)

    assert evidence["ok"] is True
    assert evidence["grown_cases"] == ["moe_v2:t32_e8_k2_h512_i512"]


def test_space_extension_rejects_replacing_instead_of_appending_choices():
    manifest = _moe_manifest("moe_new_interleave")
    summary = {
        "case_search": {
            "moe_v2:t32_e8_k2_h512_i512": {
                "knobs": [
                    {
                        "name": "interleave_bt",
                        "default": True,
                        "elevated_by": "moe_new_interleave",
                    }
                ],
                "measurements": [
                    {"config": {"bt": 64, "interleave_bt": False}}
                ],
            }
        }
    }
    incumbent = {"moe_v2:t32_e8_k2_h512_i512": [{"bt": 32}]}

    evidence = validate_space_extension(manifest, summary, incumbent)

    assert evidence["ok"] is False
    assert any("not a pure extension" in error for error in evidence["errors"])


def test_runner_only_knob_is_default_only_until_programmer_elevation():
    benchmark_only = DesignSpace([Knob("tile", [32, 64, 128], default=64)])
    elevated = DesignSpace(
        [
            Knob(
                "tile",
                [32, 64, 128],
                default=64,
                elevated_by="tile_control",
                programmer_control="demo.tile",
            )
        ]
    )

    assert list(benchmark_only.deployment_space().enumerate(cap=None)) == [{"tile": 64}]
    assert list(elevated.deployment_space().enumerate(cap=None)) == [
        {"tile": 32},
        {"tile": 64},
        {"tile": 128},
    ]
    assert elevated.deployment_space().default_config() == {"tile": 64}


def test_space_extension_promotes_an_existing_default_anchored_knob():
    manifest = _moe_manifest("moe_interleave_control")
    summary = {
        "case_search": {
            "moe_v2:t32_e8_k2_h512_i512": {
                "knobs": [
                    {
                        "name": "interleave_bt",
                        "default": True,
                        "elevated_by": "moe_interleave_control",
                    }
                ],
                "measurements": [
                    {"config": {"interleave_bt": True}},
                    {"config": {"interleave_bt": False}},
                ],
            }
        }
    }
    incumbent = {"moe_v2:t32_e8_k2_h512_i512": [{"interleave_bt": True}]}

    evidence = validate_space_extension(manifest, summary, incumbent)

    assert evidence["ok"] is True
    assert evidence["grown_cases"] == ["moe_v2:t32_e8_k2_h512_i512"]
    assert evidence["promoted_knobs"] == [
        {
            "case": "moe_v2:t32_e8_k2_h512_i512",
            "knob": "interleave_bt",
            "default": True,
        }
    ]


def _existing_moe_action_contract():
    control = "moe_v2.interleave_bt"
    manifest = _moe_manifest()
    summary = {
        "case_search": {
            "moe_v2:t32_e8_k2_h512_i512": {
                "knobs": [
                    {
                        "name": "interleave_bt",
                        "values": [True, False],
                        "default": True,
                        "elevated_by": "moe_interleave_control",
                        "programmer_control": control,
                    }
                ]
            }
        },
        "runtime_evidence": {
            "controls": [
                {
                    "control": control,
                    "matched": [{"backend": "fused_ep_moe_v2"}],
                }
            ]
        },
    }
    return manifest, summary


def test_dimension_is_bound_to_selected_family_axis_and_control():
    manifest, summary = _existing_moe_action_contract()

    evidence = validate_capability_contract(manifest, summary, ROOT, _head())
    assert evidence["ok"] is True, evidence["errors"]
    assert [
        step["function"]
        for step in evidence["source_forwarding_paths"]["moe_v2.interleave_bt"]
    ] == ["fused_ep_moe_v2", "_fused_ep_moe_kernel"]
    assert evidence["access_modes"] == {
        "moe_v2.interleave_bt": "existing-backend-argument"
    }
    derived = evidence["derived_dimensions"][0]
    assert derived["knob"] == derived["kernel_argument"] == "interleave_bt"
    assert derived["default"] is True
    assert derived["kernel_family"] == "moe_v2"
    assert derived["kernel_path"].endswith("fused_moe/v2/kernel.py")

    unrelated = copy.deepcopy(manifest)
    dimension = unrelated["search_dimensions"][0]
    dimension["control"] = "kda.interleave_bt"
    evidence = validate_capability_contract(unrelated, summary, ROOT, _head())

    assert evidence["ok"] is False
    assert any("control family must be mapped" in error for error in evidence["errors"])


def test_runner_domain_must_match_the_source_mined_candidates():
    manifest, summary = _existing_moe_action_contract()
    summary["case_search"]["moe_v2:t32_e8_k2_h512_i512"]["knobs"][0]["values"] = [
        True,
        "invented",
        False,
    ]

    evidence = validate_capability_contract(manifest, summary, ROOT, _head())

    assert evidence["ok"] is False
    assert any(
        "runner values/default do not match the graph-derived dimension" in error
        for error in evidence["errors"]
    )


def test_manifest_cannot_override_the_graph_owned_candidate_domain():
    manifest, summary = _existing_moe_action_contract()
    manifest["search_dimensions"][0]["candidate_values"] = [True, "invented"]

    evidence = validate_capability_contract(manifest, summary, ROOT, _head())

    assert evidence["ok"] is False
    assert any("must contain exactly" in error for error in evidence["errors"])


def test_dimension_cannot_name_an_unrelated_function_in_the_same_source_file():
    manifest, summary = _existing_moe_action_contract()
    manifest["search_dimensions"][0]["kernel_function"] = "ref_moe"
    summary["runtime_evidence"]["controls"][0]["matched"][0]["backend"] = (
        "ref_moe"
    )

    evidence = validate_capability_contract(manifest, summary, ROOT, _head())

    assert evidence["ok"] is False
    assert any(
        "is not an explicit argument" in error or "no static keyword-forwarding path" in error
        for error in evidence["errors"]
    )


def test_backend_default_must_preserve_the_graph_incumbent(monkeypatch):
    manifest, summary = _existing_moe_action_contract()
    original = capability_contract._function_argument_default

    def changed_default(source, function, argument):
        if function == "fused_ep_moe_v2" and argument == "interleave_bt":
            return True, False
        return original(source, function, argument)

    monkeypatch.setattr(
        capability_contract, "_function_argument_default", changed_default
    )

    evidence = validate_capability_contract(manifest, summary, ROOT, _head())

    assert evidence["ok"] is False
    assert any("backend default" in error for error in evidence["errors"])


def test_search_dimension_rejects_redundant_derived_fields():
    manifest, summary = _existing_moe_action_contract()
    manifest["search_dimensions"][0]["access_mode"] = "new-backend-argument"

    evidence = validate_capability_contract(manifest, summary, ROOT, _head())

    assert evidence["ok"] is False
    assert any("must contain exactly" in error for error in evidence["errors"])
