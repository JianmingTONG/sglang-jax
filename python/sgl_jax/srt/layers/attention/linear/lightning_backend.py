"""LightningAttnBackend — GLA backend.

DECODE uses ``decode_simple_gla_fused`` (Pallas, in-kernel async DMA
gather/scatter on the recurrent state buffer).

EXTEND uses the baseline ``simple_gla_fwd`` (Pallas) wrapped with JAX
gather/scatter — a fused EXTEND variant existed but was reverted as
slower than the baseline path used here.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.kernel_control import (
    KernelControlContext,
    KernelControlPolicy,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
    get_current_device_kind,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)

# The chunked prefill entry is a hard dependency of this backend: it is the only
# extend path, and its kernel module needs nothing this file does not already
# require. Import it from its defining module so the control it receives is
# statically traceable to the kernel argument.
from sgl_jax.srt.kernels.simple_gla.simple_gla import simple_gla_fwd

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla_fused import decode_simple_gla_fused
except ModuleNotFoundError:
    decode_simple_gla_fused = None

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

_CHUNK_SIZE = 64


def _build_alibi_base_slopes(num_heads: int) -> list[float]:
    """ALiBi base slopes matching the HF BailingMoeV2.5 reference."""

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + _build_alibi_base_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
    )


def _compute_layer_slope(
    layer_id: int,
    num_hidden_layers: int,
    num_heads: int,
    mesh: jax.sharding.Mesh | None = None,
) -> jax.Array:
    """Per-layer slope decay used as ``g_gamma`` by the simple_gla kernels.

    Sharded along the ``tensor`` axis when ``mesh`` is provided, matching
    the ``P("tensor")`` spec the slope is consumed with inside the jitted
    forward.
    """
    base = np.asarray(_build_alibi_base_slopes(num_heads), dtype=np.float32)
    slope_np = -base * (1 - (layer_id - 1) / (num_hidden_layers - 1) + 1e-5)
    if mesh is None:
        return jnp.asarray(slope_np)
    sharding = NamedSharding(mesh, P("tensor"))
    return jax.make_array_from_callback(slope_np.shape, sharding, lambda idx: slope_np[idx])


class LightningAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for GLA (Gated Linear Attention) used by BailingMoeV2.5.

    Per-layer slope (g_gamma) is pre-computed once in __init__ and indexed by
    ``layer.layer_id`` at call time, matching upstream
    ``LightningAttentionBackend.tp_slope`` ownership.
    """

    def __init__(
        self,
        mesh: jax.sharding.Mesh = None,
        chunk_size: int = _CHUNK_SIZE,
        linear_recurrent_layer_ids: list[int] | None = None,
        num_hidden_layers: int | None = None,
        num_heads: int | None = None,
        kernel_control: KernelControlPolicy | dict | str | None = None,
    ):
        """Construct a LightningAttnBackend.

        Args:
            mesh: Required for production forward.
            chunk_size: simple_gla kernel chunk size.
            linear_recurrent_layer_ids: Global layer ids of every Lightning
                attention layer in the model.
            num_hidden_layers: Total transformer layer count, used to scale
                the per-layer slope. Required iff
                ``linear_recurrent_layer_ids`` is provided.
            num_heads: Per-layer head count, used to size the slope vector.
                Required iff ``linear_recurrent_layer_ids`` is provided.
            kernel_control: Programmer-supplied shape-aware low-level controls.
        """
        super().__init__(mesh=mesh)
        self.chunk_size = chunk_size
        self.kernel_control = KernelControlPolicy.from_config(kernel_control)
        if (
            linear_recurrent_layer_ids is not None
            and num_hidden_layers is not None
            and num_heads is not None
        ):
            self.tp_slope = nnx.data(
                {
                    lid: _compute_layer_slope(lid, num_hidden_layers, num_heads, mesh)
                    for lid in linear_recurrent_layer_ids
                }
            )
        else:
            self.tp_slope = nnx.data({})

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixLightningAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        **kwargs,
    ) -> tuple[jax.Array, tuple]:
        md = self.forward_metadata
        # GLA decode is a fused Pallas kernel (in-kernel DMA gather/scatter), so
        # a masked recurrent track scatter cannot be added cleanly. The recurrent
        # extra-buffer is gated OFF for GLA at startup; fail fast if a track
        # boundary somehow reaches this backend.
        if md.recurrent_track_indices is not None:
            raise NotImplementedError(
                "recurrent extra-buffer (--enable-recurrent-extra-buffer) is not "
                "supported with the GLA/Lightning backend"
            )
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer.layer_id)

        try:
            slope = self.tp_slope[layer.layer_id]
        except KeyError:
            raise KeyError(
                f"LightningAttnBackend has no slope for layer_id={layer.layer_id}; "
                f"registered ids: {sorted(self.tp_slope.keys())}. "
                f"Was this backend created via attn_backend_wrapper with a "
                f"non-empty linear_recurrent_layer_ids?"
            ) from None

        if forward_batch.forward_mode == ForwardMode.DECODE:
            output, new_buffer = self._forward_decode(
                q,
                k,
                v,
                recurrent_buffer,
                md.recurrent_indices,
                md.has_initial_state,
                slope,
            )
        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_buffer = self._forward_extend(
                q,
                k,
                v,
                recurrent_buffer,
                md.recurrent_indices,
                md.has_initial_state,
                slope,
            )
        else:
            raise NotImplementedError(
                f"LightningAttnBackend does not support {forward_batch.forward_mode}"
            )

        return output.reshape(output.shape[0], -1), (new_buffer, [])

    @named_scope("lightning_decode")
    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        recurrent_buffer: jax.Array,
        recurrent_indices: jax.Array,
        has_initial_state: jax.Array,
        slope: jnp.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Decode forward via fused Pallas kernel with in-kernel state DMA."""
        if decode_simple_gla_fused is None:
            raise ImportError("simple_gla_fused kernel is required for GLA decode")

        def _decode_fn(q_l, k_l, v_l, gamma, buf_l, idx_l, has_l):
            return decode_simple_gla_fused(
                q_l,
                k_l,
                v_l,
                recurrent_buffer=buf_l,
                recurrent_indices=idx_l,
                has_initial_state=has_l,
                g_gamma=gamma,
                scale=None,
            )

        return jax.shard_map(
            _decode_fn,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),
                P("data", "tensor", None),
                P("data", "tensor", None),
                P("tensor"),
                P("data", "tensor", None, None),
                P("data"),
                P("data"),
            ),
            out_specs=(
                P("data", "tensor", None),
                P("data", "tensor", None, None),
            ),
            check_vma=False,
        )(q, k, v, slope, recurrent_buffer, recurrent_indices, has_initial_state)

    @named_scope("lightning_extend")
    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        recurrent_buffer: jax.Array,
        recurrent_indices: jax.Array,
        has_initial_state: jax.Array,
        slope: jnp.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Extend forward via baseline simple_gla_fwd + JAX gather/scatter."""
        cu_seqlens = self.forward_metadata.cu_q_lens
        chunk_size = self.chunk_size

        def _prefill_fn(q_l, k_l, v_l, gamma, buf_l, idx_l, has_l, cu_l):
            h0 = buf_l[idx_l]
            h0 = jnp.where(has_l[:, None, None, None], h0, 0.0)
            controls = self.kernel_control.resolve_gla(
                KernelControlContext(
                    sequence_length=q_l.shape[0],
                    num_sequences=cu_l.shape[0] - 1,
                    num_heads=q_l.shape[-2],
                    head_dim=q_l.shape[-1],
                    value_dim=v_l.shape[-1],
                    has_initial_state=h0 is not None,
                    output_final_state=True,
                    device_kind=get_current_device_kind(),
                ),
                chunk_size=chunk_size,
            )

            output, ht = simple_gla_fwd(
                q_l[None],
                k_l[None],
                v_l[None],
                g_gamma=gamma,
                h0=h0,
                cu_seqlens_dev=cu_l,
                scale=None,
                use_ht=True,
                chunk_size=controls.chunk_size,
                compact_alignment=controls.compact_alignment,
                single_chunk_state_elision=controls.single_chunk_state_elision,
                zero_state_output_elision=controls.zero_state_output_elision,
                output_value_tiles=controls.output_value_tiles,
                enable__chunk_fwd_o_pl_variant=(
                    controls.enable__chunk_fwd_o_pl_variant
                ),
                enable_chunk_fwd_h_kernel_varlen_variant=(
                    controls.enable_chunk_fwd_h_kernel_varlen_variant
                ),
                output_impl=controls.output_impl,
                chunk_impl=controls.chunk_impl,
            )

            # Skip writing back to dummy slot 0.
            keep_mask = (idx_l == 0).reshape(-1, 1, 1, 1)
            safe_val = jnp.where(keep_mask, buf_l[idx_l], ht)
            new_buf = buf_l.at[idx_l].set(safe_val)
            return output[0], new_buf

        return jax.shard_map(
            _prefill_fn,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),
                P("data", "tensor", None),
                P("data", "tensor", None),
                P("tensor"),
                P("data", "tensor", None, None),
                P("data"),
                P("data"),
                P("data"),
            ),
            out_specs=(
                P("data", "tensor", None),
                P("data", "tensor", None, None),
            ),
            check_vma=False,
        )(q, k, v, slope, recurrent_buffer, recurrent_indices, has_initial_state, cu_seqlens)


__all__ = ["LightningAttnBackend"]
