from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.configs.kernel_control import (
    KernelControlContext,
    KernelControlPolicy,
)


def _context(**overrides):
    values = {
        "sequence_length": 256,
        "num_sequences": 2,
        "num_heads": 4,
        "head_dim": 128,
        "value_dim": 128,
        "has_initial_state": True,
        "output_final_state": True,
        "device_kind": "tpu-v6e",
    }
    values.update(overrides)
    return KernelControlContext(**values)


def test_shape_rules_override_defaults_in_order():
    policy = KernelControlPolicy.from_config(
        {
            "kda": {
                "default": {"compute_block_chunks": 2},
                "rules": [
                    {
                        "when": {"head_dim": 64, "max_sequence_length": 512},
                        "set": {"state_dim_alignment": 64},
                    },
                    {
                        "when": {"sequence_length": 256},
                        "set": {"state_block_chunks": 2},
                    },
                ],
            }
        }
    )

    controls = policy.resolve_kda(_context(head_dim=64))

    assert controls.compute_block_chunks == 2
    assert controls.state_dim_alignment == 64
    assert controls.state_block_chunks == 2


def test_empty_policy_preserves_kernel_defaults():
    controls = KernelControlPolicy().resolve_kda(_context())

    assert controls.as_kernel_kwargs() == {
        "chunk_size": 64,
        "intra_block_size": 16,
        "scalar_intra_solve": False,
        "compute_block_chunks": 1,
        "state_block_chunks": 1,
        "state_dim_alignment": 128,
        # kept capability kda_resident_pipeline_api (R8): dispatch control whose
        # "incumbent" default preserves the output-hash-pinned incumbent path.
        "pipeline_impl": "incumbent",
        "single_chunk_state_elision": False,
        "zero_state_output_elision": False,
    }


def test_output_only_state_elision_is_not_a_programmer_control():
    with pytest.raises(ValueError, match="unknown controls"):
        KernelControlPolicy.from_config(
            {
                "gla": {
                    "chunk_size": 256,
                    "compact_alignment": True,
                    "single_chunk_state_elision": True,
                }
            }
        )


def test_server_args_normalizes_json_control_config_at_startup():
    from sgl_jax.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="test-model",
        kernel_control_config='{"kda":{"state_block_chunks":2}}',
    )

    assert args.kernel_control_config == {
        "kda": {
            "default": {"state_block_chunks": 2},
            "rules": [],
        }
    }


def test_policy_rejects_values_outside_the_verified_design_space():
    with pytest.raises(ValueError, match="must be one of"):
        KernelControlPolicy.from_config({"kda": {"compute_block_chunks": 8}})


def test_policy_rejects_mistyped_shape_rule_at_startup():
    with pytest.raises(ValueError, match="positive integer"):
        KernelControlPolicy.from_config(
            {
                "gla": {
                    "rules": [
                        {
                            "when": {"sequence_length": "1024"},
                            "set": {"chunk_size": 128},
                        }
                    ]
                }
            }
        )


def test_device_kind_rule_uses_the_concrete_jax_device_name():
    policy = KernelControlPolicy.from_config(
        {
            "kda": {
                "rules": [
                    {
                        "when": {"device_kind": jax.devices()[0].device_kind},
                        "set": {"compute_block_chunks": 2},
                    }
                ]
            }
        }
    )

    controls = policy.resolve_kda(
        _context(device_kind=jax.devices()[0].device_kind)
    )

    assert controls.compute_block_chunks == 2


def test_kda_backend_forwards_programmer_controls(monkeypatch):
    from sgl_jax.srt.layers.attention.linear import kda_backend as module

    captured = {}

    def fake_chunk_kda(q, k, v, g, beta, **kwargs):
        captured.update(kwargs)
        return (jnp.zeros_like(v), kwargs["initial_state"], *([None] * 10))

    monkeypatch.setattr(module, "chunk_kda", fake_chunk_kda)
    monkeypatch.setattr(module.jax, "shard_map", lambda fn, **_kwargs: fn)
    backend = module.KDAAttnBackend(
        mesh=None,
        kernel_control={
            "kda": {
                "chunk_size": 64,
                "intra_block_size": 8,
                "scalar_intra_solve": True,
                "compute_block_chunks": 2,
                "state_block_chunks": 2,
                "state_dim_alignment": 64,
            }
        },
    )
    tokens, heads, dim = 128, 2, 64
    array = jnp.zeros((tokens, heads, dim), dtype=jnp.float32)
    initial_state = jnp.zeros((1, heads, dim, dim), dtype=jnp.float32)
    layer = SimpleNamespace(
        A_log=SimpleNamespace(value=jnp.zeros((1, 1, heads, 1))),
        dt_bias=SimpleNamespace(value=jnp.zeros((1, 1, heads, dim))),
        scale=1.0,
    )

    output, final_state = backend._forward_extend(
        array,
        array,
        array,
        array,
        jnp.zeros((tokens, heads)),
        initial_state,
        jnp.array([0, tokens], dtype=jnp.int32),
        layer,
    )

    assert output.shape == array.shape
    assert final_state is initial_state
    assert captured["chunk_size"] == 64
    assert captured["intra_block_size"] == 8
    assert captured["scalar_intra_solve"] is True
    assert captured["compute_block_chunks"] == 2
    assert captured["state_block_chunks"] == 2
    assert captured["state_dim_alignment"] == 64


def test_gla_backend_forwards_programmer_controls(monkeypatch):
    from sgl_jax.srt.layers.attention.linear import lightning_backend as module

    captured = {}

    def fake_simple_gla(q, k, v, **kwargs):
        captured.update(kwargs)
        return jnp.zeros_like(v), kwargs["h0"]

    monkeypatch.setattr(module, "simple_gla_fwd", fake_simple_gla)
    monkeypatch.setattr(module.jax, "shard_map", lambda fn, **_kwargs: fn)
    backend = module.LightningAttnBackend(
        mesh=None,
        kernel_control={
            "gla": {
                "chunk_size": 128,
                "compact_alignment": True,
                "output_value_tiles": 2,
            }
        },
    )
    tokens, heads, dim = 256, 2, 128
    array = jnp.zeros((tokens, heads, dim), dtype=jnp.float32)
    backend.forward_metadata = SimpleNamespace(
        cu_q_lens=jnp.array([0, tokens], dtype=jnp.int32)
    )
    recurrent_buffer = jnp.zeros((2, heads, dim, dim), dtype=jnp.float32)

    output, new_buffer = backend._forward_extend(
        array,
        array,
        array,
        recurrent_buffer,
        jnp.array([1], dtype=jnp.int32),
        jnp.array([True]),
        jnp.zeros((heads,), dtype=jnp.float32),
    )

    assert output.shape == array.shape
    assert new_buffer.shape == recurrent_buffer.shape
    assert captured["chunk_size"] == 128
    assert captured["compact_alignment"] is True
    assert captured["output_value_tiles"] == 2
    assert captured["single_chunk_state_elision"] is False
    assert captured["zero_state_output_elision"] is False
