import ast
from pathlib import Path

from akt.core.analysis.flexgraph_extract import (
    _axis_dependencies,
    _resolve_axis,
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

    assert gaps[("kda", "solve_block_size")]["programmer_exposed"] is True
    assert gaps[("simple_gla", "BT,chunk_size")]["programmer_exposed"] is True
    assert gaps[("simple_gla", "BV")]["programmer_exposed"] is False
    assert gaps[("ragged_paged_attention", "bkv_sz")]["exposure_level"] == "runner"
