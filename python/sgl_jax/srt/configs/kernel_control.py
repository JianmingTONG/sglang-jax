"""Programmer-facing controls for performance-critical kernel scheduling.

The kernel functions expose low-level static arguments, but application code should
not have to call those functions directly or hardcode one configuration for every
shape. ``KernelControlPolicy`` is the stable control plane between server/model code
and those arguments.

Configuration is accepted as a Python mapping, a JSON string, or a JSON file. Each
kernel family has optional defaults and ordered shape rules; later matching rules
override earlier ones::

    {
      "kda": {
        "default": {"intra_block_size": 16},
        "rules": [
          {
            "when": {"head_dim": 64, "max_sequence_length": 512},
            "set": {"state_dim_alignment": 64, "state_block_chunks": 2}
          }
        ]
      },
      "gla": {
        "rules": [
          {
            "when": {"min_sequence_length": 1024},
            "set": {"chunk_size": 256, "compact_alignment": true}
          }
        ]
      }
    }

The policy is intentionally independent of JAX so it can be validated at server
startup and inspected by AKT's capability-exposure gate.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


PROGRAMMER_CONTROL_REGISTRY = {
    "kda": (
        "chunk_size",
        "intra_block_size",
        "scalar_intra_solve",
        "compute_block_chunks",
        "state_block_chunks",
        "state_dim_alignment",
    ),
    "gla": (
        "chunk_size",
        "compact_alignment",
        "output_value_tiles",
        "enable__chunk_fwd_o_pl_variant",
        "enable_chunk_fwd_h_kernel_varlen_variant",
    ),
}


@dataclass(frozen=True)
class KernelControlContext:
    """Static kernel-call context used to select and validate controls.

    Shapes are the per-device shapes seen by the kernel after sharding.
    """

    sequence_length: int
    num_sequences: int
    num_heads: int
    head_dim: int
    value_dim: int
    has_initial_state: bool
    output_final_state: bool
    device_kind: str = ""

    def __post_init__(self):
        for name in (
            "sequence_length",
            "num_sequences",
            "num_heads",
            "head_dim",
            "value_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        for name in ("has_initial_state", "output_final_state"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.device_kind, str):
            raise ValueError("device_kind must be a string")


def _require_bool(name: str, value: Any) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean; got {value!r}")


def _require_choice(name: str, value: Any, choices: tuple[Any, ...]) -> None:
    if value not in choices or type(value) is not type(choices[0]):
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")


@dataclass(frozen=True)
class KDAKernelControls:
    chunk_size: int = 64
    intra_block_size: int = 16
    scalar_intra_solve: bool = False
    compute_block_chunks: int = 1
    state_block_chunks: int = 1
    state_dim_alignment: int = 128
    # Internal execution flags, deliberately absent from the programmer registry:
    # serving observes the final recurrent state, so these must remain disabled.
    single_chunk_state_elision: bool = False
    zero_state_output_elision: bool = False

    def validate(self, context: KernelControlContext | None = None) -> None:
        _require_choice("kda.chunk_size", self.chunk_size, (16, 32, 64, 128, 256))
        _require_choice("kda.intra_block_size", self.intra_block_size, (8, 16, 32))
        _require_choice("kda.compute_block_chunks", self.compute_block_chunks, (1, 2))
        _require_choice("kda.state_block_chunks", self.state_block_chunks, (1, 2, 4))
        _require_choice("kda.state_dim_alignment", self.state_dim_alignment, (64, 128))
        _require_bool("kda.scalar_intra_solve", self.scalar_intra_solve)
        _require_bool("kda.single_chunk_state_elision", self.single_chunk_state_elision)
        _require_bool("kda.zero_state_output_elision", self.zero_state_output_elision)
        if self.chunk_size % self.intra_block_size:
            raise ValueError(
                "kda.intra_block_size must divide kda.chunk_size; "
                f"got {self.intra_block_size} and {self.chunk_size}"
            )
        if self.zero_state_output_elision and not self.single_chunk_state_elision:
            raise ValueError(
                "kda.zero_state_output_elision requires single_chunk_state_elision"
            )
        if context is None:
            return
        if self.single_chunk_state_elision and (
            context.num_sequences != 1
            or context.sequence_length != self.chunk_size
            or context.has_initial_state
            or context.output_final_state
        ):
            raise ValueError(
                "kda.single_chunk_state_elision requires one full-sequence chunk, "
                "no initial state, and no requested final state"
            )

    def as_kernel_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GLAKernelControls:
    chunk_size: int = 64
    compact_alignment: bool = False
    # Internal execution flags, deliberately absent from the programmer registry:
    # serving observes the final recurrent state, so these must remain disabled.
    single_chunk_state_elision: bool = False
    zero_state_output_elision: bool = False
    output_value_tiles: int = 1
    enable__chunk_fwd_o_pl_variant: bool = False
    enable_chunk_fwd_h_kernel_varlen_variant: bool = False

    def validate(self, context: KernelControlContext | None = None) -> None:
        _require_choice(
            "gla.chunk_size",
            self.chunk_size,
            (16, 32, 64, 128, 256, 512, 1024, 2048),
        )
        _require_choice("gla.output_value_tiles", self.output_value_tiles, (1, 2, 4, 8))
        _require_bool("gla.compact_alignment", self.compact_alignment)
        _require_bool(
            "gla.enable__chunk_fwd_o_pl_variant", self.enable__chunk_fwd_o_pl_variant
        )
        _require_bool(
            "gla.enable_chunk_fwd_h_kernel_varlen_variant",
            self.enable_chunk_fwd_h_kernel_varlen_variant,
        )
        _require_bool("gla.single_chunk_state_elision", self.single_chunk_state_elision)
        _require_bool("gla.zero_state_output_elision", self.zero_state_output_elision)
        if self.zero_state_output_elision and not self.single_chunk_state_elision:
            raise ValueError(
                "gla.zero_state_output_elision requires single_chunk_state_elision"
            )
        if context is None:
            return
        if context.head_dim % 128 or context.value_dim % 128:
            raise ValueError(
                "simple GLA controls require head_dim and value_dim divisible by 128; "
                f"got {context.head_dim}/{context.value_dim}"
            )
        value_tiles = context.num_heads * (context.value_dim // 128)
        if value_tiles % self.output_value_tiles:
            raise ValueError(
                "gla.output_value_tiles must divide the per-device head/value tile count; "
                f"got group={self.output_value_tiles}, tiles={value_tiles}"
            )
        if self.single_chunk_state_elision and (
            context.num_sequences != 1
            or context.sequence_length != self.chunk_size
            or not self.compact_alignment
            or context.has_initial_state
            or context.output_final_state
        ):
            raise ValueError(
                "gla.single_chunk_state_elision requires compact alignment, one "
                "full-sequence chunk, no initial state, and no requested final state"
            )

    def as_kernel_kwargs(self) -> dict[str, Any]:
        return asdict(self)


_CONTROL_TYPES = {
    "kda": KDAKernelControls,
    "gla": GLAKernelControls,
}
_DEFAULTS = {
    family: {field.name: field.default for field in fields(control_type)}
    for family, control_type in _CONTROL_TYPES.items()
}
_CONTEXT_FIELDS = {field.name for field in fields(KernelControlContext)}


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object; got {type(value).__name__}")
    return dict(value)


def _validate_control_override(
    family: str,
    override: Mapping[str, Any],
    label: str,
    *,
    base: Mapping[str, Any] | None = None,
) -> None:
    unknown = set(override) - set(PROGRAMMER_CONTROL_REGISTRY[family])
    if unknown:
        raise ValueError(f"{label} has unknown controls: {sorted(unknown)}")
    candidate = dict(_DEFAULTS[family])
    candidate.update(base or {})
    candidate.update(override)
    controls = _CONTROL_TYPES[family](**candidate)
    controls.validate()


def _validate_match(match: Mapping[str, Any], label: str) -> None:
    for key, value in match.items():
        base = key
        if key.startswith("min_") or key.startswith("max_"):
            base = key[4:]
            if base not in _CONTEXT_FIELDS - {"device_kind"}:
                raise ValueError(f"{label} has unknown range match {key!r}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label}.{key} must be a positive integer")
        elif key not in _CONTEXT_FIELDS:
            raise ValueError(f"{label} has unknown match key {key!r}")
        elif key == "device_kind":
            if not isinstance(value, str):
                raise ValueError(f"{label}.{key} must be a string")
        elif key in {"has_initial_state", "output_final_state"}:
            if type(value) is not bool:
                raise ValueError(f"{label}.{key} must be boolean")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")


def _matches(match: Mapping[str, Any], context: KernelControlContext) -> bool:
    for key, expected in match.items():
        if key.startswith("min_"):
            if getattr(context, key[4:]) < expected:
                return False
        elif key.startswith("max_"):
            if getattr(context, key[4:]) > expected:
                return False
        elif getattr(context, key) != expected:
            return False
    return True


def _load_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, KernelControlPolicy):
        return value.to_dict()
    if isinstance(value, str):
        source = value.strip()
        if source.startswith("{"):
            value = json.loads(source)
        else:
            value = json.loads(Path(source).expanduser().read_text())
    return _as_mapping(value, "kernel control configuration")


class KernelControlPolicy:
    """Validated defaults and ordered shape rules for production kernels."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        raw = _as_mapping(config or {}, "kernel control configuration")
        unknown_families = set(raw) - set(PROGRAMMER_CONTROL_REGISTRY)
        if unknown_families:
            raise ValueError(f"unknown kernel control families: {sorted(unknown_families)}")

        normalized = {}
        for family, family_value in raw.items():
            family_config = _as_mapping(family_value, family)
            structured = bool({"default", "rules"} & set(family_config))
            if structured:
                extra = set(family_config) - {"default", "rules"}
                if extra:
                    raise ValueError(
                        f"{family} mixes default/rules with controls: {sorted(extra)}"
                    )
                default = _as_mapping(family_config.get("default", {}), f"{family}.default")
                rules_value = family_config.get("rules", [])
            else:
                default = family_config
                rules_value = []
            _validate_control_override(family, default, f"{family}.default")
            if not isinstance(rules_value, list):
                raise ValueError(f"{family}.rules must be an array")
            rules = []
            for index, rule_value in enumerate(rules_value):
                rule = _as_mapping(rule_value, f"{family}.rules[{index}]")
                if set(rule) != {"when", "set"}:
                    raise ValueError(
                        f"{family}.rules[{index}] must contain exactly 'when' and 'set'"
                    )
                match = _as_mapping(rule["when"], f"{family}.rules[{index}].when")
                override = _as_mapping(rule["set"], f"{family}.rules[{index}].set")
                _validate_match(match, f"{family}.rules[{index}].when")
                _validate_control_override(
                    family,
                    override,
                    f"{family}.rules[{index}].set",
                    base=default,
                )
                rules.append({"when": match, "set": override})
            normalized[family] = {"default": default, "rules": rules}
        self._config = normalized

    @classmethod
    def from_config(cls, value: Any = None) -> "KernelControlPolicy":
        if isinstance(value, cls):
            return value
        return cls(_load_mapping(value))

    @classmethod
    def supported_controls(cls, family: str) -> tuple[str, ...]:
        try:
            return PROGRAMMER_CONTROL_REGISTRY[family]
        except KeyError as error:
            raise ValueError(f"unknown kernel family {family!r}") from error

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    def resolve(
        self,
        family: str,
        context: KernelControlContext | Mapping[str, Any],
        *,
        base: Mapping[str, Any] | None = None,
    ) -> KDAKernelControls | GLAKernelControls:
        if family not in _CONTROL_TYPES:
            raise ValueError(f"unknown kernel family {family!r}")
        if not isinstance(context, KernelControlContext):
            context = KernelControlContext(**_as_mapping(context, "kernel context"))
        values = dict(_DEFAULTS[family])
        if base:
            base = _as_mapping(base, f"{family} base controls")
            _validate_control_override(family, base, f"{family} base controls")
            values.update(base)
        family_config = self._config.get(family, {"default": {}, "rules": []})
        values.update(family_config["default"])
        for rule in family_config["rules"]:
            if _matches(rule["when"], context):
                values.update(rule["set"])
        controls = _CONTROL_TYPES[family](**values)
        controls.validate(context)
        return controls

    def resolve_kda(
        self,
        context: KernelControlContext | Mapping[str, Any],
    ) -> KDAKernelControls:
        controls = self.resolve("kda", context)
        assert isinstance(controls, KDAKernelControls)
        return controls

    def resolve_gla(
        self,
        context: KernelControlContext | Mapping[str, Any],
        *,
        chunk_size: int = 64,
    ) -> GLAKernelControls:
        controls = self.resolve("gla", context, base={"chunk_size": chunk_size})
        assert isinstance(controls, GLAKernelControls)
        return controls


def normalize_kernel_control_config(value: Any = None) -> dict[str, Any]:
    """Load and validate a JSON/file/mapping value for storage in ``ServerArgs``."""

    return KernelControlPolicy.from_config(value).to_dict()


__all__ = [
    "GLAKernelControls",
    "KDAKernelControls",
    "KernelControlContext",
    "KernelControlPolicy",
    "PROGRAMMER_CONTROL_REGISTRY",
    "normalize_kernel_control_config",
]
