"""Bit-exactness tests for the single-move row-permutation common kernel.

Proves that ``single_move_permute`` (both the Pallas single-move kernel in
interpret mode and the pure-jnp argsort/take fallback) is bit-exact with the
incumbent scatter formulation on the packed/varlen patterns the GLA kernel's
unalign step actually uses — the exact API gap digested in
akt/core/analysis/jax_api_limitations.md.

Shapes mirror the frozen gla suite cases (``gla:seq512_h8`` / ``gla:seq2048_h8``:
H=8 heads, K=V=128 head dim, chunk_size=64, packed batch B=1).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.common.permute import (
    single_move_permute,
    single_move_permute_reference,
)
from sgl_jax.srt.kernels.simple_gla.simple_gla import (
    _build_align_gather_idx,
    _compute_t_aligned,
    _unalign_output,
)

_HEADS = 8
_HEAD_DIM = 128
_CHUNK = 64


def _rows(rng, n, trailing=(_HEADS, _HEAD_DIM)):
    return jnp.asarray(
        rng.standard_normal((n, *trailing)) * 0.1, dtype=jnp.float32
    )


def _gla_unalign_pattern(lengths, chunk_size=_CHUNK):
    """Rebuild the exact index pattern simple_gla's unalign step permutes with.

    Returns (dest_index over the ALIGNED row space, T_aligned, cu, aligned_cu,
    T_orig): each aligned row's destination in original packed token order, with
    padding rows marked -1 (dropped).
    """
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32)
    t_orig = int(cu[-1])
    aligned = [(l + chunk_size - 1) // chunk_size * chunk_size for l in lengths]
    aligned_cu = jnp.asarray(
        np.concatenate([[0], np.cumsum(aligned)]), dtype=jnp.int32
    )
    t_aligned = _compute_t_aligned(t_orig, len(lengths), chunk_size)
    gather_idx, is_valid = _build_align_gather_idx(cu, aligned_cu, t_aligned)
    dest_index = jnp.where(is_valid, gather_idx, jnp.int32(-1))
    return dest_index, t_aligned, cu, aligned_cu, t_orig


def _assert_bit_exact(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


# ---------------------------------------------------------------------------
# GLA unalign patterns: single_move_permute == the incumbent scatter formulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["pallas", "jnp"])
def test_gla_seq512_varlen_unalign_bit_exact(backend):
    # gla:seq512_h8 packed shape with ragged per-sequence lengths (real varlen:
    # none divisible by the chunk, so align/unalign genuinely permutes rows).
    lengths = [120, 200, 192]  # sum = 512
    dest, t_aligned, cu, aligned_cu, t_orig = _gla_unalign_pattern(lengths)
    rng = np.random.default_rng(0)
    o_aligned = _rows(rng, t_aligned)

    scatter = _unalign_output(o_aligned[None], cu, aligned_cu, t_orig)[0]
    moved = single_move_permute(
        o_aligned, dest, t_orig, backend=backend, interpret=(backend == "pallas")
    )

    assert moved.shape == (t_orig, _HEADS, _HEAD_DIM)
    _assert_bit_exact(moved, scatter)
    _assert_bit_exact(
        single_move_permute_reference(o_aligned, dest, t_orig), scatter
    )


def test_gla_seq2048_varlen_unalign_bit_exact_jnp():
    # gla:seq2048_h8 packed shape; the wide case runs on the jnp fallback path.
    lengths = [1000, 548, 500]  # sum = 2048
    dest, t_aligned, cu, aligned_cu, t_orig = _gla_unalign_pattern(lengths)
    rng = np.random.default_rng(1)
    o_aligned = _rows(rng, t_aligned)

    scatter = _unalign_output(o_aligned[None], cu, aligned_cu, t_orig)[0]
    moved = single_move_permute(o_aligned, dest, t_orig, backend="jnp")

    assert moved.shape == (2048, _HEADS, _HEAD_DIM)
    _assert_bit_exact(moved, scatter)


def test_gla_single_sequence_alignment_is_identity():
    # A chunk-multiple single sequence (the frozen gla cases use cu=[0, seqlen])
    # aligns to itself: the permutation must reduce to a bit-exact identity.
    lengths = [512]
    dest, t_aligned, _cu, _aligned_cu, t_orig = _gla_unalign_pattern(lengths)
    rng = np.random.default_rng(2)
    o_aligned = _rows(rng, t_aligned)
    for backend in ("pallas", "jnp"):
        moved = single_move_permute(
            o_aligned, dest, t_orig, backend=backend, interpret=True
        )
        _assert_bit_exact(moved, o_aligned[:t_orig])


# ---------------------------------------------------------------------------
# General single-move semantics (both paths vs the scatter reference)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["pallas", "jnp"])
def test_random_bijection_with_dropped_rows(backend):
    rng = np.random.default_rng(3)
    num_rows, out_len = 96, 80
    rows = _rows(rng, num_rows, trailing=(_HEAD_DIM,))
    dest = np.full(num_rows, -1, dtype=np.int32)
    kept = rng.choice(num_rows, size=out_len, replace=False)
    dest[kept] = rng.permutation(out_len)
    dest = jnp.asarray(dest)

    expected = single_move_permute_reference(rows, dest, out_len)
    moved = single_move_permute(
        rows, dest, out_len, backend=backend, interpret=(backend == "pallas")
    )
    _assert_bit_exact(moved, expected)


@pytest.mark.parametrize("backend", ["pallas", "jnp"])
def test_reversal_and_default_out_len(backend):
    rng = np.random.default_rng(4)
    rows = _rows(rng, 16, trailing=(8,))
    dest = jnp.asarray(np.arange(15, -1, -1), dtype=jnp.int32)
    moved = single_move_permute(
        rows, dest, backend=backend, interpret=(backend == "pallas")
    )
    _assert_bit_exact(moved, rows[::-1])


def test_pallas_and_jnp_paths_agree():
    rng = np.random.default_rng(5)
    lengths = [40, 72, 16]  # ragged, chunk 64 -> permuting pattern
    dest, t_aligned, _cu, _aligned_cu, t_orig = _gla_unalign_pattern(lengths)
    rows = _rows(rng, t_aligned)
    via_pallas = single_move_permute(
        rows, dest, t_orig, backend="pallas", interpret=True
    )
    via_jnp = single_move_permute(rows, dest, t_orig, backend="jnp")
    _assert_bit_exact(via_pallas, via_jnp)


def test_input_validation():
    rows = jnp.zeros((4, 8), dtype=jnp.float32)
    with pytest.raises(ValueError):
        single_move_permute(rows, jnp.zeros((3,), dtype=jnp.int32))
    with pytest.raises(ValueError):
        single_move_permute(
            rows, jnp.zeros((4,), dtype=jnp.int32), 4, backend="nope"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
