"""Formal workload representation for AKT serving traces.

The frozen objective (`model_workloads.MODEL_WORKLOADS`) is a flat tuple of
hand-authored kernel calls. That is exact but not generic: it cannot describe a
new architecture (Qwen3, Kimi-Linear, DeepSeek-V3, ...) without hand-editing the
frozen contract. This module adds the missing formal layer:

    OpSpec        one logical operator instance with SEMANTIC parameters
                  (hidden size, heads, sequence length ...), not case ids
    BlockSpec     a repeated structural unit (one transformer block)
    WorkloadSpec  an architecture-shaped workload: arch params + blocks + phases

and a LOWERING that maps each OpSpec onto the AKT kernel inventory:

    lower(spec)             -> (ModelCall trace, coverage report)
    materialize_cases(spec) -> KernelCase objects for shapes OUTSIDE the frozen
                               inventory (built with the same runner
                               constructors the frozen cases use)

SAFETY CONTRACT (why this file can exist next to a pinned campaign):
  * Nothing here mutates `model_workloads.MODEL_WORKLOADS`, the suites, or any
    frozen case — `contract_fingerprint()` is unchanged by importing or using
    this module. `verify_legacy_roundtrip()` PROVES the formalism reproduces the
    three frozen traces call-for-call.
  * New (non-legacy) workloads and materialized cases join the measured
    objective only through an explicit maintainer `loop.py rebaseline`; until
    then they are description-only: they can be lowered, described, and
    benchmarked ad hoc, but the gate never sees them.

Operator coverage is reported, never silently dropped: an op the kernel
inventory cannot express (e.g. full-attention PREFILL — the inventory has only
the decode ragged-paged-attention kernel) appears in the coverage report as
`unsupported`, which is itself a finding: it names the next kernel family the
AKT loop would need to cover that architecture completely.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from akt.benchmark.model_workloads import (
    MODEL_WORKLOADS,
    ModelCall,
    ModelWorkload,
)


# ------------------------------------------------------------------ spec IR
@dataclass(frozen=True)
class OpSpec:
    """One logical operator instance, in SEMANTIC parameters."""

    op: str                      # operator family, one of OP_LOWERINGS
    params: tuple[tuple[str, Any], ...]  # sorted (name, value) pairs
    phase: str = "prefill"       # prefill | long-prefill | decode
    seed: int = 0
    call_id: str | None = None   # stable call name; derived when omitted

    @staticmethod
    def make(op: str, phase: str = "prefill", seed: int = 0,
             call_id: str | None = None, **params: Any) -> "OpSpec":
        return OpSpec(op=op, params=tuple(sorted(params.items())),
                      phase=phase, seed=seed, call_id=call_id)

    def param(self, name: str) -> Any:
        for key, value in self.params:
            if key == name:
                return value
        raise KeyError(f"{self.op}: missing param {name!r}")


@dataclass(frozen=True)
class BlockSpec:
    """A repeated structural unit — one architecture block."""

    block_id: str
    ops: tuple[OpSpec, ...]
    repeat: int = 1


@dataclass(frozen=True)
class WorkloadSpec:
    """An architecture-shaped serving workload."""

    workload_id: str
    family: str                        # dense | moe | hybrid-linear | legacy
    arch: tuple[tuple[str, Any], ...]  # sorted architecture parameters
    blocks: tuple[BlockSpec, ...]
    description: str = ""
    provenance: str = "hand-authored"  # where arch numbers come from

    @staticmethod
    def make(workload_id: str, family: str, blocks: Iterable[BlockSpec],
             description: str = "", provenance: str = "hand-authored",
             **arch: Any) -> "WorkloadSpec":
        return WorkloadSpec(workload_id=workload_id, family=family,
                            arch=tuple(sorted(arch.items())),
                            blocks=tuple(blocks), description=description,
                            provenance=provenance)


def spec_fingerprint(spec: WorkloadSpec) -> str:
    """Canonical content hash — the spec-level analog of contract_fingerprint."""

    def encode(obj: Any) -> Any:
        if isinstance(obj, (WorkloadSpec, BlockSpec, OpSpec)):
            return {k: encode(v) for k, v in vars(obj).items()}
        if isinstance(obj, tuple):
            return [encode(v) for v in obj]
        return obj

    raw = json.dumps(encode(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


# ------------------------------------------------------------ op lowerings
@dataclass(frozen=True)
class OpLowering:
    """How one operator family maps onto the AKT kernel inventory."""

    kernel_id: str
    runner_module: str                     # akt.core.runners.<name>
    shape_id: Callable[[OpSpec], str]
    factory_args: Callable[[OpSpec], tuple]  # runner make_inputs positional args
    space_args: Callable[[OpSpec], tuple]    # runner _space positional args


def _p(*names: str) -> Callable[[OpSpec], tuple]:
    return lambda op: tuple(op.param(n) for n in names)


OP_LOWERINGS: dict[str, OpLowering] = {
    # dense/grouped matmul (projections use groups=1; MoE expert matmuls use
    # groups=experts) -> megablox gmm
    "matmul_grouped": OpLowering(
        kernel_id="gmm", runner_module="akt.core.runners.gmm",
        shape_id=lambda op: f"m{op.param('m')}_k{op.param('k')}_n{op.param('n')}_g{op.param('groups')}",
        factory_args=_p("m", "k", "n", "groups"),
        space_args=_p("m", "k", "n"),
    ),
    # gated linear attention chunked prefill
    "gla_prefill": OpLowering(
        kernel_id="gla", runner_module="akt.core.runners.gla",
        shape_id=lambda op: f"seq{op.param('seq')}_h{op.param('heads')}",
        factory_args=_p("seq", "heads"),
        space_args=_p("seq", "heads"),
    ),
    # Kimi Delta Attention chunked prefill
    "kda_prefill": OpLowering(
        kernel_id="kda", runner_module="akt.core.runners.kda",
        shape_id=lambda op: f"seq{op.param('seq')}_h{op.param('heads')}_d{op.param('head_dim')}",
        factory_args=_p("seq", "heads", "head_dim"),
        space_args=_p("seq", "head_dim"),
    ),
    # SwiGLU MLP
    "mlp_swiglu": OpLowering(
        kernel_id="fused_mlp", runner_module="akt.core.runners.fused_mlp",
        shape_id=lambda op: f"s{op.param('seq')}_h{op.param('hidden')}_i{op.param('inter')}",
        factory_args=_p("seq", "hidden", "inter"),
        space_args=_p("seq", "inter", "hidden"),
    ),
    # paged KV-cache scatter update (decode)
    "kv_update": OpLowering(
        kernel_id="kv_cache", runner_module="akt.core.runners.kv_cache",
        shape_id=lambda op: f"h{op.param('heads')}_cache{op.param('cache')}_new{op.param('new')}",
        factory_args=lambda op: (op.param("heads"), op.param("cache"), op.param("new"), 128),
        space_args=lambda op: (op.param("heads"), 128),
    ),
}

# Operator families the inventory CANNOT express today. Lowering reports them
# instead of failing — the report is the coverage gap list.
UNSUPPORTED_OPS: dict[str, str] = {
    "attn_prefill_full": (
        "full/flash self-attention PREFILL — the AKT inventory has only the "
        "decode ragged-paged-attention kernel (rpa_v3); a prefill attention "
        "kernel family would have to be added to cover dense-prefill "
        "architectures end to end"
    ),
    "attn_decode_paged": (
        "decode ragged paged attention EXISTS (rpa_v3) but its case factory is "
        "shape-struct-driven; wiring OpSpec params onto rpa_v3's _make_case "
        "struct is deferred (legacy traces reference its two frozen shapes "
        "directly via case_ref)"
    ),
    "moe_ffn": (
        "fused MoE FFN EXISTS (moe_v1 / moe_v2) but the runner factories pin "
        "dtype and use module-level shape constants; OpSpec wiring deferred "
        "(legacy traces reference the frozen shapes via case_ref)"
    ),
}


def _frozen_case_ids() -> set[str]:
    from akt.benchmark.suites import load_cases

    return {case.case_id for case in load_cases("full")}


# ------------------------------------------------------------------ lowering
@dataclass
class LoweredWorkload:
    workload: ModelWorkload
    coverage: list[dict]                 # per-op: supported / case_ref / unsupported
    new_case_ids: list[str] = field(default_factory=list)


def _expand_ops(spec: WorkloadSpec) -> list[tuple[str, OpSpec]]:
    """Blocks × repeat → flat (call_id, op) list. Identical repeated blocks
    collapse to ONE representative instance tagged with the repeat count —
    measuring the same shape N times adds no information, and the repeat count
    is preserved in the call_id for cost extrapolation."""

    flat: list[tuple[str, OpSpec]] = []
    for block in spec.blocks:
        suffix = f"@x{block.repeat}" if block.repeat > 1 else ""
        for index, op in enumerate(block.ops):
            call_id = op.call_id or f"{block.block_id}.{op.op}{index}{suffix}"
            flat.append((call_id, op))
    return flat


def lower(spec: WorkloadSpec) -> LoweredWorkload:
    """Lower a formal spec onto the AKT kernel inventory."""

    frozen = _frozen_case_ids()
    calls: list[ModelCall] = []
    coverage: list[dict] = []
    new_cases: list[str] = []
    for call_id, op in _expand_ops(spec):
        if op.op == "case_ref":                       # direct frozen-case pin
            case_id = op.param("case_id")
            calls.append(ModelCall(call_id, case_id, op.phase, op.seed))
            coverage.append({"call": call_id, "op": op.op, "case": case_id,
                             "status": "supported (frozen case)"})
            continue
        if op.op in UNSUPPORTED_OPS:
            coverage.append({"call": call_id, "op": op.op, "status": "unsupported",
                             "why": UNSUPPORTED_OPS[op.op]})
            continue
        lowering = OP_LOWERINGS.get(op.op)
        if lowering is None:
            raise ValueError(f"unknown operator family {op.op!r}")
        case_id = f"{lowering.kernel_id}:{lowering.shape_id(op)}"
        calls.append(ModelCall(call_id, case_id, op.phase, op.seed))
        status = "supported (frozen case)" if case_id in frozen else "supported (new case)"
        if case_id not in frozen and case_id not in new_cases:
            new_cases.append(case_id)
        coverage.append({"call": call_id, "op": op.op, "case": case_id,
                         "status": status})
    workload = ModelWorkload(model_id=spec.workload_id,
                             description=spec.description,
                             calls=tuple(calls))
    return LoweredWorkload(workload=workload, coverage=coverage,
                           new_case_ids=new_cases)


def materialize_cases(spec: WorkloadSpec) -> dict[str, Any]:
    """Build KernelCase objects for the spec's NEW (non-frozen) shapes.

    Uses the very same runner constructors the frozen cases are built from, so
    a new shape inherits the frozen refs contract (reference, tolerance,
    native test) automatically. Returns {case_id: KernelCase}."""

    import importlib

    from akt.benchmark.runners.base import KernelCase

    lowered = lower(spec)
    wanted = set(lowered.new_case_ids)
    cases: dict[str, Any] = {}
    for call_id, op in _expand_ops(spec):
        lowering = OP_LOWERINGS.get(op.op)
        if lowering is None:
            continue
        case_id = f"{lowering.kernel_id}:{lowering.shape_id(op)}"
        if case_id not in wanted or case_id in cases:
            continue
        runner = importlib.import_module(lowering.runner_module)
        template = runner.CASES[0]
        cases[case_id] = KernelCase(
            kernel_id=lowering.kernel_id,
            shape_id=lowering.shape_id(op),
            make_inputs=functools.partial(runner.make_inputs, *lowering.factory_args(op)),
            run=template.run,
            reference=template.reference,
            space=runner._space(*lowering.space_args(op)),
            atol=template.atol,
            rtol=template.rtol,
            check_out=template.check_out,
            regime_pref=template.regime_pref,
            native_test=template.native_test,
            bitexact_invariant=template.bitexact_invariant,
            note=f"materialized from WorkloadSpec {spec.workload_id!r} "
                 f"({spec.provenance}); NOT part of the frozen objective",
        )
    return cases


# ------------------------------------------------- legacy roundtrip proof
def legacy_specs() -> tuple[WorkloadSpec, ...]:
    """The three frozen traces, re-expressed formally (case_ref pins).

    This is deliberately the weakest expression (direct case pins rather than
    semantic params) because its job is EXACTNESS: proving the formal layer
    reproduces the frozen contract byte-for-byte, not re-deriving it."""

    specs = []
    for workload in MODEL_WORKLOADS:
        ops = tuple(
            OpSpec.make("case_ref", phase=call.phase, seed=call.seed,
                        call_id=call.call_id, case_id=call.case_id)
            for call in workload.calls
        )
        specs.append(WorkloadSpec.make(
            workload_id=workload.model_id, family="legacy",
            blocks=[BlockSpec(block_id="trace", ops=ops)],
            description=workload.description,
            provenance="frozen model_workloads.MODEL_WORKLOADS",
        ))
    return tuple(specs)


def verify_legacy_roundtrip() -> bool:
    """Lowering the legacy specs must reproduce MODEL_WORKLOADS exactly."""

    for spec, frozen_workload in zip(legacy_specs(), MODEL_WORKLOADS):
        lowered = lower(spec)
        if lowered.workload.calls != frozen_workload.calls:
            raise AssertionError(
                f"legacy roundtrip failed for {frozen_workload.model_id}: "
                f"{lowered.workload.calls} != {frozen_workload.calls}"
            )
        if lowered.new_case_ids:
            raise AssertionError(
                f"legacy roundtrip materialized new cases: {lowered.new_case_ids}"
            )
    return True


__all__ = [
    "BlockSpec",
    "LoweredWorkload",
    "OpSpec",
    "OP_LOWERINGS",
    "UNSUPPORTED_OPS",
    "WorkloadSpec",
    "legacy_specs",
    "lower",
    "materialize_cases",
    "spec_fingerprint",
    "verify_legacy_roundtrip",
]
