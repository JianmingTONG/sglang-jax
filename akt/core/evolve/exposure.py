"""Frozen end-to-end programmer-exposure gate for AKT capabilities.

Performance alone does not prove capability elevation. This validator requires a
new runner dimension to map to a registered programmer control and verifies that a
production layer/backend forwards that control to the low-level kernel argument.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


_CONTROL_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_CONTROL_CONFIG = "python/sgl_jax/srt/configs/kernel_control.py"
_PRODUCTION_PREFIXES = (
    "python/sgl_jax/srt/layers/",
    "python/sgl_jax/srt/model_executor/",
)


def _safe_repo_file(repo: Path, relative: str) -> Path | None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    resolved = (repo / path).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError:
        return None
    return resolved


def _control_registry(config_path: Path) -> dict[str, set[str]]:
    tree = ast.parse(config_path.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name)
            and target.id == "PROGRAMMER_CONTROL_REGISTRY"
            for target in targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, dict):
            break
        return {
            str(family): {str(control) for control in controls}
            for family, controls in value.items()
        }
    raise ValueError("PROGRAMMER_CONTROL_REGISTRY is missing or not a literal mapping")


def _control_fields(config_path: Path) -> dict[str, set[str]]:
    """Map registry family names to fields on their typed control objects."""
    tree = ast.parse(config_path.read_text())
    class_fields = {
        node.name: {
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    family_types = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "_CONTROL_TYPES"
            for target in targets
        ) or not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Name)
            ):
                family_types[key.value] = value.id
    return {
        family: class_fields.get(class_name, set())
        for family, class_name in family_types.items()
    }


def _is_family_resolver(call: ast.Call, family: str) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr == f"resolve_{family}":
        return True
    return (
        call.func.attr == "resolve"
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == family
    )


def _consumer_forwards_argument(
    path: Path,
    family: str,
    control_key: str,
    kernel_argument: str,
) -> bool:
    """Prove a resolver-produced control reaches the declared kernel keyword."""
    tree = ast.parse(path.read_text())
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        control_vars = set()
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call) or not _is_family_resolver(value, family):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            control_vars.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        if not control_vars:
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != kernel_argument:
                    continue
                for child in ast.walk(keyword.value):
                    if (
                        isinstance(child, ast.Attribute)
                        and child.attr == control_key
                        and isinstance(child.value, ast.Name)
                        and child.value.id in control_vars
                    ):
                        return True
    return False


def validate_programmer_exposure(
    manifest: dict,
    eval_results: list[dict],
    repo: Path,
) -> dict:
    """Return auditable evidence that a capability reaches programmer level."""

    errors = []
    capability = manifest.get("name")
    declared_entries = manifest.get("programmer_controls")
    if not isinstance(declared_entries, list) or not declared_entries:
        return {
            "ok": False,
            "controls": [],
            "runner_controls": [],
            "errors": ["manifest must declare a non-empty programmer_controls list"],
        }

    runner_controls = set()
    for result in eval_results:
        for knob in result.get("knobs") or []:
            if knob.get("elevated_by") == capability and knob.get("programmer_control"):
                runner_controls.add(knob["programmer_control"])

    declared_controls = []
    touched = set(manifest.get("files_touched") or [])
    config_path = _safe_repo_file(repo, _CONTROL_CONFIG)
    try:
        registry = _control_registry(config_path) if config_path else {}
        control_fields = _control_fields(config_path) if config_path else {}
    except Exception as error:  # noqa: BLE001
        registry = {}
        control_fields = {}
        errors.append(f"cannot inspect programmer control registry: {error}")

    for index, entry in enumerate(declared_entries):
        label = f"programmer_controls[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object")
            continue
        required = {"control", "config_path", "consumer", "kernel_argument"}
        if set(entry) != required:
            errors.append(f"{label} must contain exactly {sorted(required)}")
            continue
        control = entry.get("control")
        if not isinstance(control, str) or not _CONTROL_RE.fullmatch(control):
            errors.append(f"{label}.control is not a family.key identifier")
            continue
        declared_controls.append(control)
        family, key = control.split(".", 1)
        if key not in registry.get(family, set()):
            errors.append(f"{control} is absent from PROGRAMMER_CONTROL_REGISTRY")
        if key not in control_fields.get(family, set()):
            errors.append(f"{control} is absent from its typed kernel control object")

        if entry.get("config_path") != _CONTROL_CONFIG:
            errors.append(
                f"{control} must use the stable {_CONTROL_CONFIG} programmer API"
            )
        elif _CONTROL_CONFIG not in touched:
            errors.append(f"{control} did not change its programmer API definition")

        consumer = entry.get("consumer")
        if not isinstance(consumer, str) or not consumer.startswith(_PRODUCTION_PREFIXES):
            errors.append(
                f"{control} consumer must be a production layer or model_executor path"
            )
            continue
        if consumer not in touched:
            errors.append(f"{control} production consumer was not changed by the capability")
        consumer_path = _safe_repo_file(repo, consumer)
        if consumer_path is None or not consumer_path.is_file():
            errors.append(f"{control} consumer does not exist: {consumer}")
            continue
        kernel_argument = entry.get("kernel_argument")
        if not isinstance(kernel_argument, str) or not kernel_argument.isidentifier():
            errors.append(f"{control} has invalid kernel_argument")
        else:
            try:
                forwarded = _consumer_forwards_argument(
                    consumer_path,
                    family,
                    key,
                    kernel_argument,
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"cannot inspect {control} consumer: {error}")
            else:
                if not forwarded:
                    errors.append(
                        f"{consumer} does not forward {control} to keyword "
                        f"{kernel_argument}"
                    )

    if len(declared_controls) != len(set(declared_controls)):
        errors.append("programmer_controls contains duplicate control identifiers")
    if set(declared_controls) != runner_controls:
        errors.append(
            "declared controls do not exactly match newly elevated runner controls: "
            f"declared={sorted(set(declared_controls))}, runner={sorted(runner_controls)}"
        )

    return {
        "ok": not errors,
        "controls": sorted(set(declared_controls)),
        "runner_controls": sorted(runner_controls),
        "errors": errors,
    }


__all__ = ["validate_programmer_exposure"]
