from pathlib import Path

from akt.core.evolve.exposure import validate_programmer_exposure


ROOT = Path(__file__).resolve().parents[2]


def _manifest():
    return {
        "name": "kda_state_block_chunks",
        "files_touched": [
            "python/sgl_jax/srt/configs/kernel_control.py",
            "python/sgl_jax/srt/layers/attention/linear/kda_backend.py",
            "akt/core/runners/kda.py",
        ],
        "programmer_controls": [
            {
                "control": "kda.state_block_chunks",
                "config_path": "python/sgl_jax/srt/configs/kernel_control.py",
                "consumer": "python/sgl_jax/srt/layers/attention/linear/kda_backend.py",
                "kernel_argument": "state_block_chunks",
            }
        ],
    }


def _results(programmer_control="kda.state_block_chunks"):
    return [
        {
            "case": "kda:shape",
            "knobs": [
                {
                    "name": "state_block_chunks",
                    "elevated_by": "kda_state_block_chunks",
                    "programmer_control": programmer_control,
                }
            ],
        }
    ]


def test_exposure_gate_accepts_registered_production_control():
    evidence = validate_programmer_exposure(_manifest(), _results(), ROOT)

    assert evidence == {
        "ok": True,
        "controls": ["kda.state_block_chunks"],
        "runner_controls": ["kda.state_block_chunks"],
        "errors": [],
    }


def test_exposure_gate_rejects_runner_only_knob():
    manifest = _manifest()
    manifest["programmer_controls"] = []

    evidence = validate_programmer_exposure(manifest, _results(), ROOT)

    assert evidence["ok"] is False
    assert "non-empty programmer_controls" in evidence["errors"][0]


def test_exposure_gate_rejects_mismatched_runner_metadata():
    evidence = validate_programmer_exposure(
        _manifest(), _results("kda.some_other_control"), ROOT
    )

    assert evidence["ok"] is False
    assert any("do not exactly match" in error for error in evidence["errors"])


def test_exposure_gate_rejects_internal_output_only_flag():
    manifest = _manifest()
    entry = manifest["programmer_controls"][0]
    entry["control"] = "kda.single_chunk_state_elision"
    entry["kernel_argument"] = "single_chunk_state_elision"

    evidence = validate_programmer_exposure(
        manifest,
        _results("kda.single_chunk_state_elision"),
        ROOT,
    )

    assert evidence["ok"] is False
    assert any("absent from PROGRAMMER_CONTROL_REGISTRY" in error for error in evidence["errors"])
