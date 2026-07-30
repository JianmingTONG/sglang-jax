"""Frozen proposal contract for source-derived AKT capability elevation."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
from collections import deque

from akt.benchmark.model_workloads import callsite_inventory
from akt.core.evolve.action_catalog import action_catalog_context, load_action_catalog


CONTROL_CONFIG = "python/sgl_jax/srt/configs/kernel_control.py"
_CONTROL_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_DIMENSION_FIELDS = {"control", "kernel_function", "consumer"}
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _safe_file(repo: Path, relative: str) -> Path | None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    resolved = (repo / path).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError:
        return None
    return resolved


def _git_text(repo: Path, commit: str, relative: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else None


def _function_has_argument(source: str, function: str, argument: str) -> bool:
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
    ]
    if not matches:
        raise ValueError(f"function {function!r} was not found")
    for node in matches:
        names = {
            item.arg
            for item in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        }
        if argument in names:
            return True
    return False


def _function_argument_default(
    source: str, function: str, argument: str
) -> tuple[bool, object | None]:
    """Return whether one function argument has a literal default and its value."""

    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one function named {function!r}; got {len(matches)}")
    node = matches[0]
    positional = node.args.posonlyargs + node.args.args
    positional_defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    defaults = {
        item.arg: default
        for item, default in zip(positional, positional_defaults)
    }
    defaults.update(zip((item.arg for item in node.args.kwonlyargs), node.args.kw_defaults))
    if argument not in defaults or defaults[argument] is None:
        return False, None
    try:
        return True, ast.literal_eval(defaults[argument])
    except (ValueError, TypeError):
        return False, None


def _module_source_path(repo: Path, module: str) -> Path | None:
    """Resolve a production absolute import without importing executable code."""

    base = repo / "python" / Path(*module.split("."))
    candidates = (base.with_suffix(".py"), base / "__init__.py")
    return next((path for path in candidates if path.is_file()), None)


def _imported_functions(repo: Path, tree: ast.AST) -> dict[str, tuple[Path, str]]:
    imported = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.ImportFrom) or not node.module or node.level:
            continue
        path = _module_source_path(repo, node.module)
        if path is None:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            imported[alias.asname or alias.name] = (path, alias.name)
    return imported


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_alias(node: ast.AST, names: set[str]) -> bool:
    """Accept only identity aliases; transformed expressions are not forwarding proof."""

    return isinstance(node, ast.Name) and node.id in names


def _target_uses_argument(
    node: ast.AST,
    argument: str,
    sink: dict | None,
) -> bool:
    aliases = {argument}
    changed = True
    while changed:
        changed = False
        for assignment in ast.walk(node):
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            if not _is_alias(assignment.value, aliases):
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    if sink is None:
        return any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in aliases
            for child in ast.walk(node)
        )
    assignment_targets = sink.get("assignments")
    if isinstance(assignment_targets, list):
        expected_expressions = sink.get("expression_asts")
        if not isinstance(expected_expressions, dict):
            return False
        proven = set()
        for assignment in ast.walk(node):
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            named_targets = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            required_targets = named_targets & set(assignment_targets)
            if not required_targets:
                continue
            if any(
                isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in aliases
                for child in ast.walk(assignment.value)
            ):
                expression = ast.dump(assignment.value, include_attributes=False)
                proven.update(
                    target
                    for target in required_targets
                    if expected_expressions.get(target) == expression
                )
        return proven == set(assignment_targets)
    for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
        if _call_name(call.func) != sink.get("callee"):
            continue
        if any(
            keyword.arg == sink.get("argument")
            and _is_alias(keyword.value, aliases)
            for keyword in call.keywords
        ):
            return True
    return False


def _argument_forwarding_path(
    repo: Path,
    *,
    start_path: str | Path,
    start_function: str,
    start_argument: str,
    target_path: str | Path,
    target_function: str,
    target_argument: str,
    target_sink: dict | None = None,
) -> list[dict] | None:
    """Find a static keyword-forwarding path between production functions.

    This deliberately recognizes only a narrow, auditable data flow: an entry
    argument (or a simple local alias of it) must be passed as a named keyword to a
    local/imported function, including ``functools.partial``.  It is sufficient for
    the existing fused-MoE, fused-MLP, and GMM wrapper stacks and fails closed on
    dynamic dispatch or opaque ``**kwargs`` construction.
    """

    start = _safe_file(repo, str(start_path))
    target = _safe_file(repo, str(target_path))
    if start is None or target is None or not start.is_file() or not target.is_file():
        return None
    start, target = start.resolve(), target.resolve()
    queue = deque([(start, start_function, start_argument, [])])
    visited: set[tuple[Path, str, str]] = set()
    parsed: dict[Path, tuple[ast.AST, dict[str, ast.AST], dict[str, tuple[Path, str]]]] = {}

    while queue:
        path, function, argument, prefix = queue.popleft()
        state = (path, function, argument)
        if state in visited or len(prefix) > 24:
            continue
        visited.add(state)
        try:
            if path not in parsed:
                tree = ast.parse(path.read_text())
                functions = {
                    node.name: node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                parsed[path] = (tree, functions, _imported_functions(repo, tree))
            _tree, functions, imports = parsed[path]
        except Exception:  # noqa: BLE001 - malformed/unreadable source is no proof
            continue
        node = functions.get(function)
        if node is None:
            continue
        step = {"path": str(path.relative_to(repo.resolve())), "function": function, "argument": argument}
        trail = [*prefix, step]
        if path == target and function == target_function and argument == target_argument:
            if _function_has_argument(
                path.read_text(), function, argument
            ) and _target_uses_argument(node, argument, target_sink):
                return trail
            continue

        tainted = {argument}
        changed = True
        while changed:
            changed = False
            for assignment in ast.walk(node):
                if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                    continue
                if not _is_alias(assignment.value, tainted):
                    continue
                targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
                for assigned in targets:
                    if isinstance(assigned, ast.Name) and assigned.id not in tainted:
                        tainted.add(assigned.id)
                        changed = True

        local_functions = set(functions)
        for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
            callable_node = call.func
            keywords = call.keywords
            if _call_name(call.func) == "partial" and call.args:
                callable_node = call.args[0]
            callee = _call_name(callable_node)
            if not callee:
                continue
            if callee in local_functions:
                next_path, next_function = path, callee
            elif callee in imports:
                next_path, next_function = imports[callee]
                next_path = next_path.resolve()
            else:
                continue
            for keyword in keywords:
                if keyword.arg and _is_alias(keyword.value, tainted):
                    queue.append((next_path, next_function, keyword.arg, trail))
    return None


def _control_registry(
    source: str | None, *, required: bool = False
) -> dict[str, set[str]]:
    if not source:
        if required:
            raise ValueError("PROGRAMMER_CONTROL_REGISTRY source is missing")
        return {}
    tree = ast.parse(source)
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
    if required:
        raise ValueError("PROGRAMMER_CONTROL_REGISTRY is missing or not a literal mapping")
    return {}


def _source_evidence(gap: dict) -> dict:
    evidence = gap.get("evidence") or ""
    path, separator, line = evidence.rpartition(":")
    if not separator:
        path, line = evidence, "0"
    return {
        "path": path,
        "line": int(line or 0),
        "detail": gap.get("detail") or gap.get("what") or "",
    }


def _axis_names(gap: dict) -> set[str]:
    source_axis = gap.get("source_axis") or gap.get("axis") or ""
    names = {part for part in re.split(r"[,\s]+", source_axis) if part}
    names.update(
        part
        for axis in gap.get("runner_axes") or []
        for part in re.split(r"[,\s]+", str(axis))
        if part
    )
    return names


def _eval_dimensions(eval_summary: dict, capability: str) -> dict[str, list[dict]]:
    by_control: dict[str, list[dict]] = {}
    callsites = callsite_inventory()
    case_search = eval_summary.get("case_search") or {}
    for site, call in callsites.items():
        for knob in (case_search.get(call.case_id) or {}).get("knobs") or []:
            control = knob.get("programmer_control")
            if knob.get("elevated_by") == capability and control:
                by_control.setdefault(control, []).append({"callsite": site, **knob})
    return by_control


def validate_action_reference(manifest: dict, repo: Path) -> dict:
    """Validate a proposal foreign key before any target-hardware evaluation."""

    errors = []
    try:
        context = action_catalog_context(repo)
        catalog = load_action_catalog(repo)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "errors": [f"cannot load red-link action catalog: {error}"]}

    expected_fingerprint = context["fingerprint"]
    if manifest.get("action_graph_fingerprint") != expected_fingerprint:
        errors.append(
            "action_graph_fingerprint must match the canonical oracle action context: "
            f"expected={expected_fingerprint!r}"
        )

    gap_id = manifest.get("gap_id")
    gap = catalog.get(gap_id)
    if gap is None:
        errors.append(
            f"gap_id {gap_id!r} is not an executable red-link action in the generated graph"
        )
        gap = {}

    expected_evidence = _source_evidence(gap) if gap else None
    affected = sorted(set(gap.get("model_callsites") or []))
    return {
        "ok": not errors,
        "gap_id": gap_id,
        "gap": gap,
        "action_edge": gap.get("action_edge") if gap else None,
        "action_graph_fingerprint": expected_fingerprint,
        "source_evidence": expected_evidence,
        "affected_model_callsites": affected,
        "errors": errors,
    }


def _same_typed_value(left, right) -> bool:
    return type(left) is type(right) and left == right


def _same_typed_values(left, right) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(_same_typed_value(a, b) for a, b in zip(left, right))
    )


def derive_search_dimensions(
    manifest: dict,
    repo: Path,
    action_reference: dict | None = None,
) -> tuple[dict, list[dict], list[str]]:
    """Derive all graph-owned dimension fields from the selected action."""

    action_reference = action_reference or validate_action_reference(manifest, repo)
    errors = list(action_reference.get("errors") or [])
    gap = action_reference.get("gap") or {}
    gap_id = action_reference.get("gap_id")
    source_evidence = action_reference.get("source_evidence") or {}
    source_axis = gap.get("source_axis") or gap.get("axis")
    source_function = gap.get("source_function")
    default = gap.get("incumbent_value")
    candidate_values = list(gap.get("candidate_values") or [])
    allowed_families = set(gap.get("kernel_ids") or [])
    allowed_axes = _axis_names(gap)

    dimensions = manifest.get("search_dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        errors.append("search_dimensions must be a non-empty list")
        dimensions = []

    derived = []
    for index, dimension in enumerate(dimensions):
        label = f"search_dimensions[{index}]"
        if not isinstance(dimension, dict) or set(dimension) != _DIMENSION_FIELDS:
            errors.append(f"{label} must contain exactly {sorted(_DIMENSION_FIELDS)}")
            continue

        control = dimension.get("control")
        if not isinstance(control, str) or not _CONTROL_RE.fullmatch(control):
            errors.append(f"{label}.control must be a family.key identifier")
            continue
        family, key = control.split(".", 1)
        if family not in allowed_families:
            errors.append(
                f"{label}.control family must be mapped by {gap_id}: "
                f"expected one of {sorted(allowed_families)}"
            )
        if key not in allowed_axes:
            errors.append(
                f"{label}.control key does not expose the selected source axis; "
                f"allowed axis names={sorted(allowed_axes)}"
            )

        kernel_function = dimension.get("kernel_function")
        if not isinstance(kernel_function, str) or not kernel_function.isidentifier():
            errors.append(f"{label}.kernel_function must be an identifier")
            continue
        consumer = dimension.get("consumer")
        if not isinstance(consumer, str) or not consumer:
            errors.append(f"{label}.consumer must be a repository-relative path")
            continue

        derived.append(
            {
                "label": label,
                "control": control,
                "candidate_values": candidate_values,
                "consumer": consumer,
                "kernel_family": family,
                "knob": key,
                "default": default,
                "kernel_path": source_evidence.get("path"),
                "kernel_function": kernel_function,
                "kernel_argument": key,
                "source_axis": source_axis,
                "source_function": source_function,
                "allowed_axes": allowed_axes,
            }
        )

    controls = [dimension["control"] for dimension in derived]
    if len(controls) != len(set(controls)):
        errors.append("search_dimensions contains duplicate controls")
    return action_reference, derived, errors


def validate_capability_contract(
    manifest: dict,
    eval_summary: dict,
    repo: Path,
    incumbent_commit: str,
    *,
    static_only: bool = False,
) -> dict:
    """Validate gap identity, prior inaccessibility, estimate, and planner dimensions.

    ``static_only=True`` runs every check derivable from the manifest, the source
    tree, and the incumbent git revision alone — skipping only the legs that need
    the target-HW eval summary (elevated-dimension observation and runtime backend
    events). The loop uses it as a fail-cheap pre-check BEFORE the expensive eval;
    the full validation still runs afterwards on the real summary.
    """

    action_reference = validate_action_reference(manifest, repo)
    action_reference, dimensions, errors = derive_search_dimensions(
        manifest, repo, action_reference
    )
    capability = manifest.get("name")
    gap_id = action_reference.get("gap_id")
    gap = action_reference.get("gap") or {}
    action_graph_fingerprint = action_reference.get("action_graph_fingerprint")
    expected_evidence = action_reference.get("source_evidence")
    affected = action_reference.get("affected_model_callsites") or []

    estimate = manifest.get("estimate")
    estimate_required = {"model", "callsite", "baseline_share_pct", "expected_relief_pct", "reasoning"}
    if not isinstance(estimate, dict) or set(estimate) != estimate_required:
        errors.append(f"estimate must contain exactly {sorted(estimate_required)}")
    else:
        if estimate["callsite"] not in affected:
            errors.append("estimate.callsite must be one of affected_model_callsites")
        expected_model = str(estimate["callsite"]).split("/", 1)[0]
        if estimate["model"] != expected_model:
            errors.append("estimate.model does not match estimate.callsite")
        for key in ("baseline_share_pct", "expected_relief_pct"):
            value = estimate.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                errors.append(f"estimate.{key} must be a positive number")
        if not isinstance(estimate.get("reasoning"), str) or not estimate["reasoning"].strip():
            errors.append("estimate.reasoning must explain the bottleneck-relief calculation")

    controls = []
    forwarding_paths: dict[str, list[dict]] = {}
    access_modes: dict[str, str] = {}
    dimension_evidence = []
    old_registry = _control_registry(
        _git_text(repo, incumbent_commit, CONTROL_CONFIG)
    )
    eval_dimensions = _eval_dimensions(eval_summary, str(capability))
    runtime_controls = {
        record.get("control"): record
        for record in (eval_summary.get("runtime_evidence") or {}).get("controls") or []
        if isinstance(record, dict)
    }

    for dimension in dimensions:
        label = dimension["label"]
        control = dimension["control"]
        controls.append(control)
        values = dimension["candidate_values"]
        default = dimension["default"]
        knob_name = dimension["knob"]
        if not static_only:
            observed = eval_dimensions.get(control) or []
            if not observed:
                errors.append(
                    f"{control!r} is not a newly elevated model-planner dimension"
                )
            for knob in observed:
                if knob.get("name") != knob_name:
                    errors.append(f"{control} runner knob does not match {knob_name!r}")
                if not _same_typed_values(
                    knob.get("values"), values
                ) or not _same_typed_value(knob.get("default"), default):
                    errors.append(
                        f"{control} runner values/default do not match the "
                        "graph-derived dimension"
                    )
            observed_sites = {item["callsite"] for item in observed}
            if observed_sites != set(affected):
                errors.append(
                    f"{control} must occur at every and only affected model callsite: "
                    f"missing={sorted(set(affected) - observed_sites)}, "
                    f"extra={sorted(observed_sites - set(affected))}"
                )

        kernel_path = dimension["kernel_path"]
        kernel_function = dimension["kernel_function"]
        kernel_argument = dimension["kernel_argument"]
        path = _safe_file(repo, kernel_path) if isinstance(kernel_path, str) else None
        expected_kernel_path = (expected_evidence or {}).get("path")
        if (
            path is None
            or not path.is_file()
            or not str(kernel_path).startswith("python/sgl_jax/srt/kernels/")
        ):
            errors.append(f"{label}.kernel_path must name a production kernel source")
            continue
        if not static_only:
            runtime_record = runtime_controls.get(control) or {}
            mismatched_backends = [
                event.get("backend")
                for event in runtime_record.get("matched") or []
                if event.get("backend") != kernel_function
            ]
            if mismatched_backends:
                errors.append(
                    f"{control} runtime event named a different backend: "
                    f"{mismatched_backends}"
                )
        old_source = _git_text(repo, incumbent_commit, kernel_path)
        try:
            current_source = path.read_text()
            current_has = _function_has_argument(
                current_source, kernel_function, kernel_argument
            )
            old_has = (
                _function_has_argument(old_source, kernel_function, kernel_argument)
                if old_source is not None
                else False
            )
        except Exception as error:  # noqa: BLE001
            errors.append(f"cannot prove {control} backend entry: {error}")
            continue
        if not current_has:
            errors.append(f"{control} is not an explicit argument of {kernel_function}")
        else:
            try:
                default_known, backend_default = _function_argument_default(
                    current_source, kernel_function, kernel_argument
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"cannot inspect {control} backend default: {error}")
            else:
                if not default_known or not _same_typed_value(backend_default, default):
                    errors.append(
                        f"{control} backend default must preserve the graph incumbent "
                        f"{default!r}"
                    )
        source_function = gap.get("source_function")
        forwarding_path = None
        if current_has and isinstance(source_function, str):
            forwarding_path = _argument_forwarding_path(
                repo,
                start_path=kernel_path,
                start_function=kernel_function,
                start_argument=kernel_argument,
                target_path=expected_kernel_path,
                target_function=source_function,
                target_argument=kernel_argument,
                target_sink=gap.get("source_sink"),
            )
            if forwarding_path is None:
                errors.append(
                    f"{control} has no static keyword-forwarding path from "
                    f"{kernel_function}.{kernel_argument} to the selected graph source "
                    f"{source_function}.{kernel_argument}"
                )
            elif isinstance(control, str):
                forwarding_paths[control] = forwarding_path
                if source_function != kernel_function:
                    source_path = _safe_file(repo, expected_kernel_path)
                    try:
                        source_default_known, source_default = _function_argument_default(
                            source_path.read_text(), source_function, kernel_argument
                        )
                    except Exception as error:  # noqa: BLE001
                        errors.append(f"cannot inspect {control} source default: {error}")
                    else:
                        if not source_default_known or not _same_typed_value(
                            source_default, default
                        ):
                            errors.append(
                                f"{control} source default must preserve the graph "
                                f"incumbent {default!r}"
                            )
        access_mode = (
            "existing-backend-argument" if old_has else "existing-low-level-axis"
        )
        access_modes[control] = access_mode
        if access_mode == "existing-low-level-axis":
            if not gap.get("existing_low_level_proven"):
                errors.append(
                    f"{control} cannot invent a backend behavior from an unproven graph finding"
                )
            allowed_axes = dimension["allowed_axes"]
            if old_source is None or not any(name in old_source for name in allowed_axes):
                errors.append(
                    f"{control} source axis is not present in the incumbent low-level source"
                )

        family, key = control.split(".", 1)
        if key in old_registry.get(family, set()):
            errors.append(
                f"{control} already existed in the incumbent programmer registry; "
                "this is parameter tuning, not new exposure"
            )
        dimension_evidence.append(
            {
                key: dimension[key]
                for key in (
                    "control",
                    "candidate_values",
                    "consumer",
                    "kernel_family",
                    "knob",
                    "default",
                    "kernel_path",
                    "kernel_function",
                    "kernel_argument",
                    "source_axis",
                )
            }
            | {"access_mode": access_mode}
        )

    return {
        "ok": not errors,
        "static_only": static_only,
        "gap_id": gap_id,
        "gap": gap,
        "action_edge": action_reference.get("action_edge"),
        "action_graph_fingerprint": action_graph_fingerprint,
        "source_evidence": expected_evidence,
        "affected_model_callsites": sorted(set(affected)),
        "search_controls": sorted(str(control) for control in set(controls)),
        "derived_dimensions": dimension_evidence,
        "access_modes": access_modes,
        "source_forwarding_paths": forwarding_paths,
        "errors": errors,
    }


def _config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def validate_space_extension(
    manifest: dict,
    eval_summary: dict,
    incumbent_case_spaces: dict[str, list[dict]],
    repo: Path | None = None,
) -> dict:
    """Prove the candidate appends dimensions without deleting incumbent choices."""

    repo = repo or _REPO_ROOT
    action_reference, dimensions, errors = derive_search_dimensions(manifest, repo)
    capability = manifest.get("name")
    inventory = callsite_inventory()
    affected_cases = {
        inventory[site].case_id
        for site in action_reference.get("affected_model_callsites") or []
        if site in inventory
    }
    current_search = eval_summary.get("case_search") or {}
    expected_cases = set(incumbent_case_spaces)
    current_cases = set(current_search)
    if expected_cases != current_cases:
        errors.append(
            "case-space inventory changed: "
            f"missing={sorted(expected_cases - current_cases)}, "
            f"added={sorted(current_cases - expected_cases)}"
        )

    grew = []
    promoted = []
    observed_values: dict[str, set[str]] = {}
    expected_values = {
        dimension["knob"]: {
            _config_key({"value": value})
            for value in dimension["candidate_values"]
        }
        for dimension in dimensions
    }
    for case_id in sorted(expected_cases & current_cases):
        old_configs = incumbent_case_spaces.get(case_id) or []
        result = current_search[case_id]
        current_configs = [
            measurement.get("config") or {}
            for measurement in result.get("measurements") or []
        ]
        new_knobs = {
            knob["name"]: knob
            for knob in result.get("knobs") or []
            if knob.get("elevated_by") == capability
        }
        old_keys = {_config_key(config) for config in old_configs}
        current_keys = {_config_key(config) for config in current_configs}

        if case_id not in affected_cases:
            if new_knobs:
                errors.append(f"{case_id} gained capability knobs but is not an affected callsite")
            if old_keys != current_keys:
                errors.append(f"unaffected case {case_id} changed its valid design space")
            continue
        if not new_knobs:
            errors.append(f"affected case {case_id} has no new capability dimension")
            continue

        promoted_names = {
            name
            for name in new_knobs
            if old_configs and all(name in config for config in old_configs)
        }
        partially_present = {
            name
            for name in new_knobs
            if any(name in config for config in old_configs)
            and name not in promoted_names
        }
        if partially_present:
            errors.append(
                f"{case_id} has inconsistent incumbent presence for promoted knobs: "
                f"{sorted(partially_present)}"
            )
        added_names = set(new_knobs) - promoted_names - partially_present

        for name in sorted(promoted_names):
            default = new_knobs[name].get("default")
            incumbent_values = {
                _config_key({"value": config[name]}) for config in old_configs
            }
            if incumbent_values != {_config_key({"value": default})}:
                errors.append(
                    f"{case_id} can promote existing knob {name!r} only from its "
                    f"preserved incumbent default {default!r}"
                )
            else:
                promoted.append({"case": case_id, "knob": name, "default": default})

        projected = set()
        for config in current_configs:
            projection = dict(config)
            for name in added_names:
                projection.pop(name, None)
            for name in promoted_names:
                projection[name] = new_knobs[name].get("default")
            projected.add(_config_key(projection))
        if projected != old_keys:
            errors.append(
                f"{case_id} is not a pure extension of the incumbent space: "
                f"lost={len(old_keys - projected)}, added-old-projections={len(projected - old_keys)}"
            )

        for old in old_configs:
            default_extension = dict(old)
            default_extension.update(
                {
                    name: new_knobs[name].get("default")
                    for name in added_names
                }
            )
            if _config_key(default_extension) not in current_keys:
                errors.append(
                    f"{case_id} removed incumbent config at new-dimension defaults: {old}"
                )
                break

        if len(current_keys) > len(old_keys):
            grew.append(case_id)
        for name in new_knobs:
            observed_values.setdefault(name, set()).update(
                _config_key({"value": config[name]})
                for config in current_configs
                if name in config
            )

    if not grew:
        errors.append("no affected case gained a valid configuration")
    for knob, values in expected_values.items():
        missing = values - observed_values.get(knob, set())
        if missing:
            errors.append(
                f"runner knob {knob!r} advertises candidate values that occur in no valid "
                f"model configuration: {sorted(missing)}"
            )
    return {
        "ok": not errors,
        "affected_cases": sorted(affected_cases),
        "grown_cases": grew,
        "promoted_knobs": promoted,
        "errors": errors,
    }


__all__ = [
    "_argument_forwarding_path",
    "derive_search_dimensions",
    "validate_action_reference",
    "validate_capability_contract",
    "validate_space_extension",
]
