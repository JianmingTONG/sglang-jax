"""Reusable JAX-level kernel primitives shared across kernel families.

Modules here patch limitations in the JAX/Pallas API boundary itself (see
akt/core/analysis/jax_api_limitations.md) rather than implementing any single
serving kernel. They are standalone handles: existing kernels are not rewired
to them here — consuming them is a per-kernel elevation decision.
"""

from sgl_jax.srt.kernels.common.permute import (  # noqa: F401
    single_move_permute,
    single_move_permute_reference,
)
