import os

import jax.numpy as jnp
import numpy as np
import pytest


requires_interpret = pytest.mark.skipif(
    os.environ.get("PALLAS_INTERPRET") != "1",
    reason="custom Pallas control numerics require PALLAS_INTERPRET=1 off TPU",
)


@requires_interpret
def test_packed_stateful_kda_custom_controls_match_recurrent_reference():
    from akt.benchmark.refs.kda import make_inputs
    from sgl_jax.srt.kernels.kda import chunk_kda, naive_recurrent_kda

    lengths = (80, 112)
    offsets = np.cumsum((0, *lengths))
    cu_seqlens = jnp.asarray(offsets, jnp.int32)
    inputs = make_inputs(sum(lengths), 2, 64, seed=31)
    initial_state = jnp.asarray(
        np.random.default_rng(31).standard_normal((2, 2, 64, 64)) * 0.01,
        jnp.float32,
    )
    output, final_state, *_ = chunk_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["scale"],
        initial_state,
        True,
        cu_seqlens,
        chunk_size=64,
        intra_block_size=8,
        scalar_intra_solve=False,
        compute_block_chunks=2,
        state_block_chunks=2,
        state_dim_alignment=64,
    )
    expected_outputs = []
    expected_states = []
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        expected_output, expected_state = naive_recurrent_kda(
            inputs["q"][:, start:stop],
            inputs["k"][:, start:stop],
            inputs["v"][:, start:stop],
            inputs["g"][:, start:stop],
            inputs["beta"][:, start:stop],
            scale=inputs["scale"],
            initial_state=initial_state[index : index + 1],
            output_final_state=True,
        )
        expected_outputs.append(expected_output)
        expected_states.append(expected_state)
    expected_output = jnp.concatenate(expected_outputs, axis=1)
    expected_state = jnp.concatenate(expected_states, axis=0)

    np.testing.assert_allclose(output, expected_output, atol=1e-2, rtol=2e-2)
    np.testing.assert_allclose(final_state, expected_state, atol=1e-2, rtol=2e-2)


@requires_interpret
def test_packed_stateful_gla_custom_controls_match_recurrent_reference():
    from akt.benchmark.refs.gla import make_inputs
    from sgl_jax.srt.kernels.simple_gla.native import naive_gla_prefill
    from sgl_jax.srt.kernels.simple_gla.simple_gla import simple_gla_fwd

    lengths = (80, 112)
    cu_seqlens = jnp.asarray(np.cumsum((0, *lengths)), jnp.int32)
    inputs = make_inputs(sum(lengths), 2, seed=37)
    initial_state = jnp.asarray(
        np.random.default_rng(37).standard_normal((2, 2, 128, 128)) * 0.01,
        jnp.float32,
    )
    output, final_state = simple_gla_fwd(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g_gamma=inputs["g_gamma"],
        h0=initial_state,
        use_ht=True,
        cu_seqlens_dev=cu_seqlens,
        chunk_size=64,
        compact_alignment=True,
        output_value_tiles=2,
    )
    expected_output, expected_state = naive_gla_prefill(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g_gamma"],
        initial_state,
        cu_seqlens,
    )

    np.testing.assert_allclose(output, expected_output, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(final_state, expected_state, atol=1e-3, rtol=1e-3)
