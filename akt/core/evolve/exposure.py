"""Frozen end-to-end programmer-exposure gate for AKT capabilities.

Performance alone does not prove capability elevation. This validator requires a
new runner dimension to map to a registered programmer control and verifies that a
production layer/backend forwards that control to the low-level kernel argument.
"""

from __future__ import annotations

import ast
from pathlib import Path

from akt.core.evolve.capability_contract import (
    _argument_forwarding_path,
    _control_registry,
    derive_search_dimensions,
    _module_source_path,
    _safe_file,
)

_CONTROL_CONFIG = "python/sgl_jax/srt/configs/kernel_control.py"
_PRODUCTION_PREFIXES = (
    "python/sgl_jax/srt/layers/",
    "python/sgl_jax/srt/model_executor/",
    "python/sgl_jax/srt/models/",
)


def _control_fields(config_path: Path) -> dict[str, dict[str, tuple[bool, object]]]:
    """Map typed control fields to whether they have a literal default and its value."""
    tree = ast.parse(config_path.read_text())
    class_fields = {
        node.name: {
            statement.target.id: (
                True,
                ast.literal_eval(statement.value),
            )
            if statement.value is not None
            else (False, None)
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
    repo: Path,
    kernel_path: str,
    kernel_function: str,
) -> bool:
    """Prove a resolver-produced control reaches the declared kernel keyword."""
    tree = ast.parse(path.read_text())
    imported = {}
    for top_level in tree.body:
        if (
            isinstance(top_level, ast.ImportFrom)
            and top_level.module
            and not top_level.level
        ):
            imported_path = _module_source_path(repo, top_level.module)
            if imported_path is None:
                continue
            for alias in top_level.names:
                if alias.name != "*":
                    imported[alias.asname or alias.name] = (imported_path, alias.name)
    local_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
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
            called = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            for keyword in node.keywords:
                if keyword.arg != kernel_argument:
                    continue
                value = keyword.value
                if not (
                    isinstance(value, ast.Attribute)
                    and value.attr == control_key
                    and isinstance(value.value, ast.Name)
                    and value.value.id in control_vars
                ):
                    continue
                if called in local_functions:
                    start_path, start_function = path, called
                elif called in imported:
                    start_path, start_function = imported[called]
                else:
                    continue
                try:
                    start_path = start_path.resolve().relative_to(repo.resolve())
                except ValueError:
                    continue
                forwarding = _argument_forwarding_path(
                    repo,
                    start_path=start_path,
                    start_function=start_function,
                    start_argument=kernel_argument,
                    target_path=kernel_path,
                    target_function=kernel_function,
                    target_argument=kernel_argument,
                )
                if forwarding is not None:
                    return True
    return False


def validate_programmer_exposure(
    manifest: dict,
    eval_results: list[dict] | dict,
    repo: Path,
) -> dict:
    """Return auditable evidence that a capability reaches programmer level."""

    action_reference, dimensions, errors = derive_search_dimensions(manifest, repo)
    capability = manifest.get("name")

    runner_controls = set()
    if isinstance(eval_results, dict):
        results = list((eval_results.get("case_search") or {}).values())
    else:
        results = eval_results
    for result in results:
        for knob in result.get("knobs") or []:
            if knob.get("elevated_by") == capability and knob.get("programmer_control"):
                runner_controls.add(knob["programmer_control"])

    declared_controls = [dimension["control"] for dimension in dimensions]
    config_path = _safe_file(repo, _CONTROL_CONFIG)
    try:
        registry = (
            _control_registry(config_path.read_text(), required=True)
            if config_path
            else {}
        )
        control_fields = _control_fields(config_path) if config_path else {}
    except Exception as error:  # noqa: BLE001
        registry = {}
        control_fields = {}
        errors.append(f"cannot inspect programmer control registry: {error}")

    for dimension in dimensions:
        control = dimension["control"]
        family, key = control.split(".", 1)
        if key not in registry.get(family, set()):
            errors.append(f"{control} is absent from PROGRAMMER_CONTROL_REGISTRY")
        if key not in control_fields.get(family, set()):
            errors.append(f"{control} is absent from its typed kernel control object")
        else:
            default_known, control_default = control_fields[family][key]
            expected_default = dimension["default"]
            if not default_known or not (
                type(control_default) is type(expected_default)
                and control_default == expected_default
            ):
                errors.append(
                    f"{control} typed API default must preserve the graph incumbent "
                    f"{expected_default!r}"
                )

        consumer = dimension["consumer"]
        if not isinstance(consumer, str) or not consumer.startswith(_PRODUCTION_PREFIXES):
            errors.append(
                f"{control} consumer must be a production layer, model, or model_executor path"
            )
            continue
        consumer_path = _safe_file(repo, consumer)
        if consumer_path is None or not consumer_path.is_file():
            errors.append(f"{control} consumer does not exist: {consumer}")
            continue
        kernel_argument = dimension["kernel_argument"]
        try:
            forwarded = _consumer_forwards_argument(
                consumer_path,
                family,
                key,
                kernel_argument,
                repo,
                dimension["kernel_path"],
                dimension["kernel_function"],
            )
        except Exception as error:  # noqa: BLE001
            errors.append(f"cannot inspect {control} consumer: {error}")
        else:
            if not forwarded:
                errors.append(
                    f"{consumer} does not forward {control} to "
                    f"{dimension['kernel_function']}.{kernel_argument}"
                )

    if set(declared_controls) != runner_controls:
        errors.append(
            "declared controls do not exactly match newly elevated runner controls: "
            f"declared={sorted(set(declared_controls))}, runner={sorted(runner_controls)}"
        )

    return {
        "ok": not errors,
        "gap_id": action_reference.get("gap_id"),
        "source_evidence": action_reference.get("source_evidence"),
        "affected_model_callsites": action_reference.get("affected_model_callsites") or [],
        "controls": sorted(set(declared_controls)),
        "runner_controls": sorted(runner_controls),
        "errors": errors,
    }


__all__ = ["validate_programmer_exposure"]
