import inspect

from sgl_jax.srt.kernels.kda.kda import chunk_kda_fwd, kda_fwd_intra


def test_chunk_kda_fwd_preserves_existing_positional_parameter_order():
    existing = [
        "q",
        "k",
        "v",
        "g",
        "beta",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "use_qk_l2norm_in_kernel",
        "chunk_indices",
        "chunk_size",
        "safe_gate",
        "lower_bound",
        "use_gate_in_kernel",
        "A_log",
        "dt_bias",
        "disable_recompute",
        "return_intermediate_states",
        "cp_context",
        "transpose_state_layout",
    ]
    assert list(inspect.signature(chunk_kda_fwd).parameters)[: len(existing)] == existing


def test_kda_fwd_intra_preserves_existing_positional_parameter_order():
    existing = [
        "q",
        "k",
        "v",
        "gk",
        "beta",
        "scale",
        "cu_seqlens",
        "chunk_size",
        "chunk_indices",
        "safe_gate",
        "disable_recompute",
    ]
    assert list(inspect.signature(kda_fwd_intra).parameters)[: len(existing)] == existing
