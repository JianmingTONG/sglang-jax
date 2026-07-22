"""Frozen three-model serving workload contract for AKT.

These are intentionally small, deterministic serving traces.  Every call names one
of the frozen ``KernelCase`` contracts, an explicit input/weight seed, and an
inference phase.  Together the three traces cover the complete AKT kernel inventory
exactly once, preventing a capability from escaping the model objective by moving a
kernel to a deferred or unmeasured bucket.

The actual tensors and correctness references remain owned by
``akt.benchmark.refs``.  The model evaluator fingerprints the generated tensors and
reference outputs at campaign initialization and requires that fingerprint in every
later round.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable


MODEL_CONTRACT_VERSION = 1
TARGET_BACKEND = "tpu"


@dataclass(frozen=True)
class ModelCall:
    """One immutable operation call in a model inference trace."""

    call_id: str
    case_id: str
    phase: str
    seed: int


@dataclass(frozen=True)
class ModelWorkload:
    """A bounded model graph used by both exhaustive verification and DP."""

    model_id: str
    description: str
    calls: tuple[ModelCall, ...]


MODEL_WORKLOADS = (
    ModelWorkload(
        model_id="tiny-linear-serving",
        description="deterministic linear-attention prefill shape envelope",
        calls=(
            ModelCall("input-projection", "gmm:m256_k512_n512_g4", "prefill", 0),
            ModelCall("gla-short", "gla:seq512_h8", "prefill", 0),
            ModelCall("kda-short", "kda:seq128_h2_d64", "prefill", 0),
            ModelCall("gla-long", "gla:seq2048_h8", "long-prefill", 0),
            ModelCall("kda-long", "kda:seq256_h4_d128", "long-prefill", 0),
        ),
    ),
    ModelWorkload(
        model_id="tiny-dense-serving",
        description="deterministic dense transformer prefill and decode trace",
        calls=(
            ModelCall("dense-projection", "gmm_v2:m512_k1024_n1024_g8", "prefill", 0),
            ModelCall("dense-mlp", "fused_mlp:s128_h256_i512", "prefill", 0),
            ModelCall(
                "paged-attention",
                "rpa_v3:d_s2_q4kv2_hd128_p256_ctx512",
                "decode",
                0,
            ),
            ModelCall("cache-update", "kv_cache:h8_cache4096_new256", "decode", 42),
        ),
    ),
    ModelWorkload(
        model_id="tiny-moe-serving",
        description="deterministic mixture-of-experts prefill and decode trace",
        calls=(
            ModelCall("router-projection", "gmm:m512_k512_n512_g8", "prefill", 0),
            ModelCall("expert-v1", "moe_v1:t32_e8_k2_h2048_i1024", "prefill", 1234),
            ModelCall("expert-v2", "moe_v2:t32_e8_k2_h512_i512", "prefill", 1234),
            ModelCall("expert-mlp", "fused_mlp:s256_h512_i512", "prefill", 0),
            ModelCall(
                "paged-attention",
                "rpa_v3:d_s3_q8kv8_hd128_p256_ctx1024",
                "decode",
                0,
            ),
            ModelCall("cache-update", "kv_cache:h16_cache8192_new512", "decode", 42),
        ),
    ),
)


def callsite_id(model: ModelWorkload, call: ModelCall) -> str:
    return f"{model.model_id}/{call.call_id}"


def callsite_inventory() -> dict[str, ModelCall]:
    return {
        callsite_id(model, call): call
        for model in MODEL_WORKLOADS
        for call in model.calls
    }


def callsites_for_kernel_ids(kernel_ids: Iterable[str]) -> list[str]:
    wanted = set(kernel_ids)
    return sorted(
        callsite_id(model, call)
        for model in MODEL_WORKLOADS
        for call in model.calls
        if call.case_id.split(":", 1)[0] in wanted
    )


def contract_payload() -> dict:
    return {
        "version": MODEL_CONTRACT_VERSION,
        "target_backend": TARGET_BACKEND,
        "models": [
            {
                "model_id": model.model_id,
                "description": model.description,
                "calls": [asdict(call) for call in model.calls],
            }
            for model in MODEL_WORKLOADS
        ],
    }


def contract_fingerprint() -> str:
    raw = json.dumps(contract_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_workloads(expected_case_ids: Iterable[str] | None = None) -> None:
    if len(MODEL_WORKLOADS) != 3:
        raise ValueError("AKT requires exactly three frozen model workloads")
    model_ids = [model.model_id for model in MODEL_WORKLOADS]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model workload identifiers must be unique")

    callsites = callsite_inventory()
    expected_calls = sum(len(model.calls) for model in MODEL_WORKLOADS)
    if len(callsites) != expected_calls:
        raise ValueError("model callsite identifiers must be unique within each model")
    for model in MODEL_WORKLOADS:
        if not model.calls:
            raise ValueError(f"{model.model_id} has no calls")
        phases = {call.phase for call in model.calls}
        if not phases <= {"prefill", "long-prefill", "decode"}:
            raise ValueError(f"{model.model_id} has an unknown inference phase: {phases}")
        if any(type(call.seed) is not int or call.seed < 0 for call in model.calls):
            raise ValueError(f"{model.model_id} has an invalid deterministic seed")

    case_ids = [call.case_id for model in MODEL_WORKLOADS for call in model.calls]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("each frozen KernelCase must occur in exactly one model workload")
    if expected_case_ids is not None and set(case_ids) != set(expected_case_ids):
        raise ValueError(
            "model workloads do not cover the complete kernel inventory: "
            f"missing={sorted(set(expected_case_ids) - set(case_ids))}, "
            f"extra={sorted(set(case_ids) - set(expected_case_ids))}"
        )


__all__ = [
    "MODEL_CONTRACT_VERSION",
    "MODEL_WORKLOADS",
    "TARGET_BACKEND",
    "ModelCall",
    "ModelWorkload",
    "callsite_id",
    "callsite_inventory",
    "callsites_for_kernel_ids",
    "contract_fingerprint",
    "contract_payload",
    "validate_workloads",
]
