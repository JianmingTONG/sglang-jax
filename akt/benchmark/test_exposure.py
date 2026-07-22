import json
from pathlib import Path

from akt.core.evolve.action_catalog import action_catalog_context
from akt.core.evolve.exposure import validate_programmer_exposure


ROOT = Path(__file__).resolve().parents[2]
GAP_ID = "fused_moe/v2:interleave_bt:schedule-toggle"


def _repo(
    tmp_path: Path,
    *,
    registered: bool = True,
    transformed: bool = False,
    control_default: bool = True,
) -> Path:
    graph = json.loads(
        (ROOT / "akt/core/analysis/flexgraph_generated.json").read_text()
    )
    graph_path = tmp_path / "akt/core/analysis/flexgraph_generated.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(json.dumps(graph))

    config_path = tmp_path / "python/sgl_jax/srt/configs/kernel_control.py"
    config_path.parent.mkdir(parents=True)
    registered_controls = '("interleave_bt",)' if registered else "()"
    config_path.write_text(
        f"""
class MoeControls:
    interleave_bt: bool = {control_default!r}

PROGRAMMER_CONTROL_REGISTRY = {{"moe_v2": {registered_controls}}}
_CONTROL_TYPES = {{"moe_v2": MoeControls}}
"""
    )

    kernel_path = tmp_path / "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py"
    kernel_path.parent.mkdir(parents=True)
    kernel_path.write_text(
        """
def _fused_ep_moe_kernel(*, interleave_bt=True):
    use_gather_bank = interleave_bt and True
    return use_gather_bank

def fused_ep_moe_v2(*, interleave_bt=True):
    return _fused_ep_moe_kernel(interleave_bt=interleave_bt)
"""
    )

    consumer_path = tmp_path / "python/sgl_jax/srt/layers/fused_moe.py"
    consumer_path.parent.mkdir(parents=True)
    forwarded_value = (
        "not controls.interleave_bt" if transformed else "controls.interleave_bt"
    )
    consumer_path.write_text(
        f"""
from sgl_jax.srt.kernels.fused_moe.v2.kernel import fused_ep_moe_v2

def run(policy):
    controls = policy.resolve("moe_v2", None)
    return fused_ep_moe_v2(interleave_bt={forwarded_value})
"""
    )
    return tmp_path


def _manifest(repo: Path):
    context = action_catalog_context(repo)
    return {
        "name": "moe_interleave_control",
        "action_graph_fingerprint": context["fingerprint"],
        "gap_id": GAP_ID,
        "search_dimensions": [
            {
                "control": "moe_v2.interleave_bt",
                "kernel_function": "fused_ep_moe_v2",
                "consumer": "python/sgl_jax/srt/layers/fused_moe.py",
            }
        ],
    }


def _results(programmer_control="moe_v2.interleave_bt"):
    return [
        {
            "case": "moe_v2:shape",
            "knobs": [
                {
                    "name": "interleave_bt",
                    "elevated_by": "moe_interleave_control",
                    "programmer_control": programmer_control,
                }
            ],
        }
    ]


def test_exposure_gate_accepts_registered_typed_production_control(tmp_path):
    repo = _repo(tmp_path)

    evidence = validate_programmer_exposure(_manifest(repo), _results(), repo)

    assert evidence["ok"] is True, evidence["errors"]
    assert evidence["gap_id"] == GAP_ID
    assert evidence["controls"] == ["moe_v2.interleave_bt"]
    assert evidence["runner_controls"] == ["moe_v2.interleave_bt"]
    assert evidence["errors"] == []


def test_exposure_gate_rejects_runner_only_knob(tmp_path):
    repo = _repo(tmp_path)
    manifest = _manifest(repo)
    manifest["search_dimensions"] = []

    evidence = validate_programmer_exposure(manifest, _results(), repo)

    assert evidence["ok"] is False
    assert any("non-empty list" in error for error in evidence["errors"])


def test_exposure_gate_rejects_mismatched_runner_metadata(tmp_path):
    repo = _repo(tmp_path)

    evidence = validate_programmer_exposure(
        _manifest(repo), _results("moe_v2.some_other_control"), repo
    )

    assert evidence["ok"] is False
    assert any("do not exactly match" in error for error in evidence["errors"])


def test_exposure_gate_rejects_unregistered_control(tmp_path):
    repo = _repo(tmp_path, registered=False)

    evidence = validate_programmer_exposure(_manifest(repo), _results(), repo)

    assert evidence["ok"] is False
    assert any(
        "absent from PROGRAMMER_CONTROL_REGISTRY" in error
        for error in evidence["errors"]
    )


def test_exposure_gate_rejects_a_changed_typed_api_default(tmp_path):
    repo = _repo(tmp_path, control_default=False)

    evidence = validate_programmer_exposure(_manifest(repo), _results(), repo)

    assert evidence["ok"] is False
    assert any("typed API default" in error for error in evidence["errors"])


def test_exposure_gate_rejects_transformed_consumer_value(tmp_path):
    repo = _repo(tmp_path, transformed=True)

    evidence = validate_programmer_exposure(_manifest(repo), _results(), repo)

    assert evidence["ok"] is False
    assert any("does not forward" in error for error in evidence["errors"])
