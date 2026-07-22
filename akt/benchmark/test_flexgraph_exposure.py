import ast
from pathlib import Path

from akt.core.analysis.flexgraph_extract import (
    _axis_dependencies,
    _mine_file,
    _resolve_axis,
    build_graph,
    mine_serving_stack,
)


ROOT = Path(__file__).resolve().parents[2]


def test_axis_flow_follows_static_aliases_but_not_array_data_flow():
    tree = ast.parse(
        """
def solve(matrix, block_size=16):
    return matrix

def kernel(v, chunk_size, intra_block_size, scalar_intra_solve):
    BT = chunk_size
    solve_block_size = 1 if scalar_intra_solve else intra_block_size
    solve(v, block_size=intra_block_size)
    V = v.shape[-1]
    BV = 128 if V % 128 == 0 else V
"""
    )
    dependencies = _axis_dependencies(tree)
    named = {
        "chunk_size",
        "intra_block_size",
        "scalar_intra_solve",
    }

    assert _resolve_axis("BT", named, dependencies, "kernel") == {"chunk_size"}
    assert _resolve_axis("solve_block_size", named, dependencies, "kernel") == {
        "intra_block_size",
        "scalar_intra_solve",
    }
    assert _resolve_axis("block_size", named, dependencies, "solve") == {
        "intra_block_size"
    }
    assert _resolve_axis("BV", named, dependencies, "kernel") == set()


def test_live_miner_distinguishes_production_exposure_from_runner_only_axes():
    mined = mine_serving_stack(
        {
            "kernels": ROOT / "python/sgl_jax/srt/kernels",
            "repo": ROOT,
        }
    )
    gaps = {(gap["family"], gap["axis"]): gap for gap in mined["gaps"]}

    assert gaps[("kda", "BT,block_size,chunk_size")]["programmer_exposed"] is True
    assert gaps[("simple_gla", "BT,chunk_size")]["programmer_exposed"] is True
    assert gaps[("simple_gla", "BV")]["programmer_exposed"] is False
    assert gaps[("update_kv_cache", "num_slices_per_block,page_size")]["exposure_level"] == "runner"


def test_ifexp_gap_requires_the_axis_itself_to_be_optional(tmp_path):
    source = """
def kernel(bkv_sz, bkv_csz, sliding_window):
    bkv_sz = min(1024, bkv_sz) if sliding_window is None else sliding_window
    bkv_csz = bkv_csz if bkv_csz is not None else bkv_sz
"""
    path = tmp_path / "kernel.py"
    mined = _mine_file(path, source)
    pinned = {
        finding["axis"]
        for finding in mined["findings"]
        if finding["category"] == "compute-tile-pinned"
    }

    assert pinned == {"bkv_csz"}


def test_hardcoded_pipeline_value_is_bound_to_its_enclosing_function(tmp_path):
    source = """
def backend_entry(x):
    return wrapper(mode=Buffered(buffer_count=3), x=x)
"""
    path = tmp_path / "kernel.py"

    mined = _mine_file(path, source)
    pipeline = next(
        finding
        for finding in mined["findings"]
        if finding["category"] == "pipeline-depth"
    )

    assert pipeline["axis"] == "buffer_count"
    assert pipeline["fn"] == "backend_entry"
    assert pipeline["incumbent_value"] == 3
    assert pipeline["incumbent_value_known"] is True


def test_null_schedule_default_is_known_source_evidence(tmp_path):
    path = tmp_path / "kernel.py"
    mined = _mine_file(path, "def kernel(prefetch_mode=None):\n    return prefetch_mode\n")
    finding = next(
        item
        for item in mined["findings"]
        if item["category"] == "schedule-toggle"
    )

    assert finding["incumbent_value_known"] is True
    assert finding["incumbent_value"] is None


def test_schedule_domain_follows_only_static_parameter_forwarding(tmp_path):
    source = '''
def low_level(*, prefetch_mode="full"):
    enabled = prefetch_mode != "none"
    full = prefetch_mode == "full"
    return enabled, full

def public(*, prefetch_mode="full"):
    if prefetch_mode not in ("none", "full", "w13"):
        raise ValueError(prefetch_mode)
    return low_level(prefetch_mode=prefetch_mode)

def unrelated(*, prefetch_mode="full"):
    if prefetch_mode == "foreign":
        return None
'''
    path = tmp_path / "kernel.py"
    mined = _mine_file(path, source)
    finding = next(
        item
        for item in mined["findings"]
        if item["category"] == "schedule-toggle"
        and item["fn"] == "low_level"
    )

    assert finding["candidate_values"] == ["full", "none", "w13"]
    assert finding["source_sink"]["assignments"] == ["enabled", "full"]
    assert set(finding["source_sink"]["expression_asts"]) == {"enabled", "full"}


def test_missing_table_is_an_observation_not_an_executable_action():
    mined = mine_serving_stack(
        {
            "kernels": ROOT / "python/sgl_jax/srt/kernels",
            "repo": ROOT,
        }
    )
    missing_table = [
        gap for gap in mined["gaps"] if gap["category"] == "missing-tuned-table"
    ]

    assert missing_table
    assert all(gap["existing_low_level_proven"] is False for gap in missing_table)
    assert all(gap["eligible_for_current_gate"] is False for gap in missing_table)


def test_numeric_semantics_switch_is_not_a_schedule_action():
    mined = mine_serving_stack(
        {
            "kernels": ROOT / "python/sgl_jax/srt/kernels",
            "repo": ROOT,
        }
    )
    activation_quant = next(
        gap for gap in mined["gaps"] if gap["axis"] == "enable_act_quant"
    )

    assert activation_quant["existing_low_level_proven"] is False
    assert activation_quant["eligible_for_current_gate"] is False


def test_every_mined_gap_has_a_unique_stable_action_id_and_model_mapping():
    mined = mine_serving_stack(
        {
            "kernels": ROOT / "python/sgl_jax/srt/kernels",
            "repo": ROOT,
        }
    )
    gaps = mined["gaps"]
    ids = [gap["gap_id"] for gap in gaps]
    assert len(ids) == len(set(ids))
    assert all(gap["id"] == gap["gap_id"] for gap in gaps)
    assert all(isinstance(gap["model_callsites"], list) for gap in gaps)
    assert all(isinstance(gap["incumbent_value_known"], bool) for gap in gaps)
    assert all(
        gap["incumbent_value_known"]
        for gap in gaps
        if gap["eligible_for_current_gate"]
    )
    assert any(gap["open"] and not gap["eligible_for_current_gate"] for gap in gaps)


def test_red_links_are_exactly_the_open_actions_consumed_by_the_loop():
    graph = build_graph()
    action_ids = {
        gap["gap_id"]
        for gap in graph["gaps"]
        if gap["open"] and not gap["programmer_exposed"]
        and gap["eligible_for_current_gate"]
    }
    edge_ids = [edge["gap_id"] for edge in graph["action_edges"]]
    nodes = {node["id"]: node for node in graph["nodes"]}

    assert len(edge_ids) == len(set(edge_ids))
    assert graph["graph_target"]["backend"] == "tpu"
    assert graph["graph_target"]["stack"][-1] == "TPU hardware"
    assert set(edge_ids) == action_ids
    assert graph["n_open_action_edges"] == len(action_ids)
    assert "gap_edges" not in graph
    assert graph["hidden_lowering_edges"]
    for edge in graph["action_edges"]:
        action = nodes[edge["source"]]
        assert action["layer"] == "action"
        assert action["nameability"] == "open-action"
        assert action["gap_id"] == edge["gap_id"]
        assert edge["eligible_for_current_gate"] is True
        assert edge["target"].startswith("pallas:")

    missing_table_ids = {
        gap["gap_id"]
        for gap in graph["gaps"]
        if gap["category"] == "missing-tuned-table"
    }
    assert missing_table_ids.isdisjoint(edge_ids)

    uncovered = [
        gap for gap in graph["gaps"]
        if gap["open"] and not gap["eligible_for_current_gate"]
    ]
    assert uncovered
    assert all(gap["action_node"] is None for gap in uncovered)
