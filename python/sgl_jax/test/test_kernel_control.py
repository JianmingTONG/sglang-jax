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


def test_empty_policy_preserves_rpa_v3_kernel_defaults():
    controls = KernelControlPolicy().resolve_rpa_v3(_context())

    # None = the kernel's incumbent tuned-table/heuristic block-size selection.
    assert controls.as_kernel_kwargs() == {"d_bkv_sz": None}


def test_rpa_v3_policy_rejects_values_outside_the_certified_domain():
    with pytest.raises(ValueError, match="must be one of"):
        KernelControlPolicy.from_config({"rpa_v3": {"d_bkv_sz": 384}})


def test_rpa_v3_scalar_override_swaps_only_the_kv_tiles():
    from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
        _apply_bkv_sz_override,
    )

    assert _apply_bkv_sz_override(
        {"bq_sz": 32, "bkv_sz": 4096, "bq_csz": 16, "bkv_csz": 512}, 2048
    ) == {"bq_sz": 32, "bkv_sz": 2048, "bq_csz": 16, "bkv_csz": 512}
    # The compute tile falls back to the override when it stops dividing it.
    assert _apply_bkv_sz_override(
        {"bq_sz": 1, "bkv_sz": 4096, "bq_csz": 1, "bkv_csz": 768}, 1024
    ) == {"bq_sz": 1, "bkv_sz": 1024, "bq_csz": 1, "bkv_csz": 1024}
    # The compute tile is clamped down to the override.
    assert _apply_bkv_sz_override(
        {"bq_sz": 1, "bkv_sz": 4096, "bq_csz": 1, "bkv_csz": 4096}, 256
    ) == {"bq_sz": 1, "bkv_sz": 256, "bq_csz": 1, "bkv_csz": 256}


def test_rpa_v3_kernel_rejects_ambiguous_decode_block_size_controls():
    from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
        ragged_paged_attention,
    )

    q = jnp.zeros((8, 4, 128), dtype=jnp.bfloat16)
    kv = jnp.zeros((8, 2, 128), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((4, 16, 4, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([8], dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)
    cu = jnp.array([0, 8], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="mutually exclusive"):
        ragged_paged_attention(
            q,
            kv,
            kv,
            kv_cache,
            kv_lens,
            page_indices,
            cu,
            cu,
            distribution,
            None,
            d_block_sizes=(1, 2048, 1, 2048),
            d_bkv_sz=2048,
        )


def _ensure_optional_serving_deps():
    """Stub serving-only optional deps absent from the device-free test host.

    ``flashattention_backend`` transitively imports the constrained-decoding
    and experts-capture stacks; only the two leaf third-party modules below can
    be missing here, and nothing in these tests executes them. Real modules,
    when installed, are always preferred.
    """
    import sys
    import types

    class _Unavailable:
        pass

    try:
        import llguidance  # noqa: F401
    except ImportError:
        stub = types.ModuleType("llguidance")
        stub.LLMatcher = _Unavailable
        stub.LLTokenizer = _Unavailable
        stub.StructTag = _Unavailable
        stub.LLInterpreter = _Unavailable
        stub.grammar_from = _Unavailable
        sys.modules["llguidance"] = stub
    try:
        import pybase64  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pybase64")
        stub.b64encode = _Unavailable
        stub.b64decode = _Unavailable
        sys.modules["pybase64"] = stub


def _flashattention_backend_call(monkeypatch, kernel_control):
    _ensure_optional_serving_deps()
    from sgl_jax.srt.layers.attention import flashattention_backend as module

    captured = {}

    def fake_rpa_v3(queries, keys, values, kv_cache_fused, *args, **kwargs):
        captured.update(kwargs)
        return jnp.zeros_like(queries), kv_cache_fused

    monkeypatch.setattr(module, "ragged_paged_attention_v3", fake_rpa_v3)
    monkeypatch.setattr(module.jax, "shard_map", lambda fn, **_kwargs: fn)
    tokens, heads, dim = 8, 4, 128
    backend = module.FlashAttention(
        heads,
        2,
        dim,
        page_size=16,
        kernel_control=kernel_control,
    )
    backend.forward_metadata = SimpleNamespace(
        seq_lens=jnp.array([4, 4], dtype=jnp.int32),
        page_indices=jnp.zeros((4,), dtype=jnp.int32),
        cu_q_lens=jnp.array([0, 4, 8], dtype=jnp.int32),
        cu_kv_lens=jnp.array([0, 16, 32], dtype=jnp.int32),
        distribution=jnp.array([0, 2, 2], dtype=jnp.int32),
        custom_mask=None,
        swa_page_indices=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        head_dim=dim,
        scaling=1.0,
        sliding_window_size=None,
        logit_cap=None,
        xai_temperature_len=0,
        softmax_dtype=None,
    )
    array = jnp.zeros((tokens, heads, dim), dtype=jnp.float32)

    output, _ = backend(array, array, array, layer, None, None)

    assert output.shape == (tokens, heads * dim)
    return captured


def test_flashattention_backend_forwards_programmer_controls(monkeypatch):
    captured = _flashattention_backend_call(
        monkeypatch, {"rpa_v3": {"d_bkv_sz": 1024}}
    )

    assert captured["d_bkv_sz"] == 1024


def test_flashattention_backend_defaults_to_incumbent_block_sizes(monkeypatch):
    captured = _flashattention_backend_call(monkeypatch, None)

    assert captured["d_bkv_sz"] is None
