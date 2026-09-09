"""Greedy-helper checks against library math; no original/model execution."""

import ast
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def greedy():
    # Importing the serving framework would introduce unrelated runtime setup.
    # Execute exactly the candidate method, including its actual JAX operations.
    source = Path(__file__).resolve().parents[2] / "python/sgl_jax/srt/layers/sampler.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sampler"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_greedy_sampling"
    )
    namespace = {"jax": jax, "jnp": jnp, "lax": jax.lax}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return jax.jit(lambda values: namespace["_greedy_sampling"](None, (values, None, None)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", ["edges", "random", "captured_vocab", "singletons"])
def test_greedy_ids_and_logprob_values(greedy, dtype, case):
    rng = np.random.default_rng(624)
    if case == "edges":
        rows = np.array(
            list(itertools.product([-np.inf, -2.0, -0.0, 0.0, 2.0, np.inf, np.nan], repeat=4))
        )
    elif case == "singletons":
        rows = np.array([[0.0], [np.inf], [-np.inf], [np.nan]])
    elif case == "captured_vocab":
        rows = rng.standard_normal((1, 151936))
        # Equal maxima straddle tensor partitions in the serving layout.
        rows[0, 18991] = rows[0, 132944] = 100.0
    else:
        rows = rng.standard_normal((256, 257))
        rows[:, 128] = rows[:, 0]
        rows[::8, 3] = rows[::8, 254] = np.nan
    values = jnp.asarray(rows, dtype=dtype)
    ids, logprobs = greedy(values)
    expected_ids = jnp.argmax(values, axis=-1).flatten()
    expected_logprobs = jax.nn.log_softmax(values, axis=-1)
    assert ids.dtype == expected_ids.dtype
    assert logprobs.dtype == expected_logprobs.dtype == values.dtype
    np.testing.assert_array_equal(np.asarray(ids), np.asarray(expected_ids))
    # NumPy's custom bfloat16 dtype does not recognize matching NaNs in this
    # assertion. FP32 represents every BF16 value exactly, including infinities.
    np.testing.assert_array_equal(
        np.asarray(logprobs).astype(np.float32), np.asarray(expected_logprobs).astype(np.float32)
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_empty_vocabulary_still_rejected(greedy, dtype):
    with pytest.raises(ValueError, match="argmax of an empty sequence"):
        greedy(jnp.empty((1, 0), dtype=dtype))
