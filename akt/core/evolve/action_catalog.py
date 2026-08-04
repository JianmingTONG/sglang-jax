"""Canonical graph-linked action catalog consumed by the AKT loop.

An AKT action is not an arbitrary hidden compiler boundary. It is an open,
source-derived flexibility finding exercised by the frozen model gate, with one
stable ``gap_id`` and exactly one red ``action_edge`` in the generated graph. This
module validates that one-to-one contract before the adapter or proposal gate
exposes any action to an oracle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


GRAPH = Path("akt/core/analysis/flexgraph_generated.json")
ACTION_CONTEXT_VERSION = 2

# These finding kinds prove that a selectable behavior is already present below the
# serving API.  In particular, an absent tuned table is an optimization opportunity,
# not an existing low-level flexibility that closes a red link by itself.
_PROVEN_EXISTING_CATEGORIES = frozenset(
    {
        "pipeline-depth",
        "schedule-toggle",
    }
)

# The same two categories are the only shapes the frozen extractor can mine from
# source with a complete sink/incumbent/domain record.  A NOVEL algorithm proposal
# must therefore be written down as one of these interfaces: after implementation the
# regenerated graph has to contain the declared finding, mined by the extractor with
# exactly the declared semantics, or the round fails closure.
MINABLE_INTERFACE_CATEGORIES = _PROVEN_EXISTING_CATEGORIES


class ActionCatalogError(ValueError):
    """The generated graph cannot safely serve as the loop's action space."""


def source_sink_errors(category, source_sink, source_axis) -> list[str]:
    """Category-specific shape rules for a finding's mined sink record.

    Shared by the graph loader and the novel-proposal validator so a declared
    novel action must satisfy exactly the shape the extractor emits.
    """

    if category == "pipeline-depth":
        if (
            not isinstance(source_sink, dict)
            or set(source_sink) != {"callee", "argument"}
            or not all(
                isinstance(source_sink.get(key), str)
                and source_sink[key].isidentifier()
                for key in ("callee", "argument")
            )
            or source_sink["argument"] != source_axis
        ):
            return ["pipeline action has no exact source sink"]
        return []
    if category == "schedule-toggle":
        if (
            not isinstance(source_sink, dict)
            or set(source_sink) != {"assignments", "expression_asts"}
            or not isinstance(source_sink.get("assignments"), list)
            or not source_sink["assignments"]
            or len(source_sink["assignments"])
            != len(set(source_sink["assignments"]))
            or not all(
                isinstance(assignment, str) and assignment.isidentifier()
                for assignment in source_sink["assignments"]
            )
            or not isinstance(source_sink.get("expression_asts"), dict)
            or set(source_sink["expression_asts"])
            != set(source_sink["assignments"])
            or not all(
                isinstance(expression, str) and expression
                for expression in source_sink["expression_asts"].values()
            )
        ):
            return ["schedule action has no exact assignment sinks"]
        return []
    return []


def _proven_existing_action(gap: dict) -> bool:
    """Whether a v2 finding carries complete source-derived authorization evidence."""

    return (
        gap.get("category") in _PROVEN_EXISTING_CATEGORIES
        and gap.get("existing_low_level_proven") is True
        and gap.get("incumbent_value_known") is True
        and "incumbent_value" in gap
    )


def _evidence_parts(gap: dict) -> tuple[str, int]:
    evidence = gap.get("evidence")
    if not isinstance(evidence, str):
        raise ActionCatalogError(f"action {gap.get('gap_id')!r} has invalid evidence")
    path, separator, line = evidence.rpartition(":")
    if not separator or not path:
        raise ActionCatalogError(
            f"action {gap.get('gap_id')!r} evidence must be repo/path.py:line"
        )
    try:
        line_number = int(line)
    except ValueError as error:
        raise ActionCatalogError(
            f"action {gap.get('gap_id')!r} has a non-numeric evidence line"
        ) from error
    return path, line_number


def _canonical_action(gap: dict) -> dict:
    """The exact, JSON-stable action record handed to an oracle/manifest gate."""

    source_path, source_line = _evidence_parts(gap)
    edge = gap["action_edge"]
    return {
        "gap_id": gap["gap_id"],
        "family": gap["family"],
        "kernel_ids": sorted(gap["kernel_ids"]),
        "source_axis": gap.get("source_axis") or gap["axis"],
        "source_function": gap["source_function"],
        "source_sink": gap.get("source_sink"),
        "incumbent_value": gap["incumbent_value"],
        "candidate_values": list(gap["candidate_values"]),
        "category": gap["category"],
        "source_evidence": {
            "path": source_path,
            "line": source_line,
            "detail": gap.get("detail") or gap.get("what") or "",
        },
        "model_callsites": sorted(gap["model_callsites"]),
        "action_edge": {
            "source": edge["source"],
            "target": edge["target"],
        },
    }


def action_semantic_record(action: dict) -> dict:
    """Return the stable semantics used to compare unrelated graph actions.

    Source line numbers, prose, and the derived action-node identifier are omitted so
    an accepted edit may move nearby code without appearing to mutate another action.
    """

    source_path, _source_line = _evidence_parts(action)
    edge = action.get("action_edge")
    if not isinstance(edge, dict) or not isinstance(edge.get("target"), str):
        raise ActionCatalogError(
            f"action {action.get('gap_id')!r} has no stable graph target"
        )
    return {
        "gap_id": action["gap_id"],
        "family": action["family"],
        "kernel_ids": sorted(action["kernel_ids"]),
        "source_path": source_path,
        "source_axis": action.get("source_axis") or action["axis"],
        "source_function": action["source_function"],
        "source_sink": action.get("source_sink"),
        "incumbent_value_type": type(action["incumbent_value"]).__name__,
        "incumbent_value": action["incumbent_value"],
        "candidate_values": list(action["candidate_values"]),
        "category": action["category"],
        "model_callsites": sorted(action["model_callsites"]),
        "graph_target": edge["target"],
    }


def load_action_graph(repo: Path) -> tuple[dict, dict[str, dict]]:
    """Return the validated graph and complete executable action catalog."""

    graph = json.loads((repo / GRAPH).read_text())
    version = graph.get("action_contract_version")
    if version != ACTION_CONTEXT_VERSION:
        raise ActionCatalogError(
            "generated flexibility graph has an unsupported action contract: "
            f"expected={ACTION_CONTEXT_VERSION}, actual={version!r}"
        )
    nodes: dict[str, dict] = {}
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ActionCatalogError("generated flexibility node is missing a string id")
        node_id = node["id"]
        if node_id in nodes:
            raise ActionCatalogError(f"duplicate generated node id {node_id!r}")
        nodes[node_id] = node
    gaps: dict[str, dict] = {}
    for gap in graph.get("gaps") or []:
        if not isinstance(gap, dict):
            raise ActionCatalogError("generated flexibility finding is not an object")
        gap_id = gap.get("gap_id")
        if not isinstance(gap_id, str) or not gap_id:
            raise ActionCatalogError("generated flexibility finding is missing gap_id")
        if gap_id in gaps:
            raise ActionCatalogError(f"duplicate generated gap_id {gap_id!r}")
        gaps[gap_id] = gap
    if not gaps:
        raise ActionCatalogError("generated flexibility graph has no source findings")

    action_ids = {
        gap_id
        for gap_id, gap in gaps.items()
        if gap.get("open") is True
        and gap.get("programmer_exposed") is False
        and gap.get("eligible_for_current_gate") is True
        and _proven_existing_action(gap)
    }
    raw_edges: dict[str, dict] = {}
    for edge in graph.get("action_edges") or []:
        if not isinstance(edge, dict):
            raise ActionCatalogError("action_edges entries must be objects")
        gap_id = edge.get("gap_id")
        if not isinstance(gap_id, str) or not gap_id:
            raise ActionCatalogError("action edge is missing gap_id")
        if gap_id in raw_edges:
            raise ActionCatalogError(f"gap_id {gap_id!r} has multiple action edges")
        source, target = edge.get("source"), edge.get("target")
        if source not in nodes or target not in nodes:
            raise ActionCatalogError(
                f"action edge {gap_id!r} has missing endpoint {source!r}->{target!r}"
            )
        raw_edges[gap_id] = edge

    missing_edges = action_ids - set(raw_edges)
    if missing_edges:
        raise ActionCatalogError(
            "red action edges must cover every executable open finding: "
            f"missing_edges={sorted(missing_edges)}"
        )
    extra_edges = set(raw_edges) - action_ids
    if extra_edges:
        raise ActionCatalogError(
            "red action edges include non-action findings: "
            f"{sorted(extra_edges)}"
        )
    edges = raw_edges
    if graph.get("n_open_action_edges") != len(edges):
        raise ActionCatalogError(
            "n_open_action_edges does not match the validated red action edges"
        )
    action_node_ids = {
        node_id
        for node_id, node in nodes.items()
        if (node.get("layer") == "action" or node.get("nameability") == "open-action")
        and node.get("gap_id") in action_ids
    }
    edge_sources = {edge["source"] for edge in edges.values()}
    if action_node_ids != edge_sources:
        raise ActionCatalogError(
            "action nodes must exactly equal red action-edge sources: "
            f"orphan_nodes={sorted(action_node_ids - edge_sources)}, "
            f"missing_nodes={sorted(edge_sources - action_node_ids)}"
        )
    catalog = {}
    for gap_id in sorted(action_ids):
        gap, edge = gaps[gap_id], edges[gap_id]
        source, target = edge["source"], edge["target"]
        if gap.get("action_node") != source or gap.get("graph_node") != target:
            raise ActionCatalogError(
                f"action edge {gap_id!r} disagrees with its finding endpoints"
            )
        action_node = nodes[source]
        if (
            action_node.get("layer") != "action"
            or action_node.get("nameability") != "open-action"
            or action_node.get("gap_id") != gap_id
        ):
            raise ActionCatalogError(f"{source!r} is not the action node for {gap_id!r}")
        if edge.get("eligible_for_current_gate") is not True:
            raise ActionCatalogError(f"action edge {gap_id!r} has inconsistent eligibility")
        kernel_ids = gap.get("kernel_ids")
        model_callsites = gap.get("model_callsites")
        if not isinstance(model_callsites, list) or not model_callsites:
            raise ActionCatalogError(f"action {gap_id!r} has no model callsite")
        if len(model_callsites) != len(set(model_callsites)):
            raise ActionCatalogError(f"action {gap_id!r} has duplicate model callsites")
        if not isinstance(kernel_ids, list) or not kernel_ids:
            raise ActionCatalogError(f"action {gap_id!r} has no kernel family mapping")
        family = gap.get("family")
        source_axis = gap.get("source_axis") or gap.get("axis")
        source_function = gap.get("source_function")
        if not isinstance(family, str) or not family:
            raise ActionCatalogError(f"action {gap_id!r} has no source family")
        if not isinstance(source_axis, str) or not source_axis:
            raise ActionCatalogError(f"action {gap_id!r} has no source axis")
        if not isinstance(source_function, str) or not source_function.isidentifier():
            raise ActionCatalogError(f"action {gap_id!r} has no source function")
        source_sink = gap.get("source_sink")
        sink_errors = source_sink_errors(
            gap.get("category"), source_sink, source_axis
        )
        if sink_errors:
            raise ActionCatalogError(f"{sink_errors[0]}: {gap_id!r}")
        if gap.get("incumbent_value_known") is not True or "incumbent_value" not in gap:
            raise ActionCatalogError(
                f"action {gap_id!r} has no source-derived incumbent value"
            )
        candidate_values = gap.get("candidate_values")
        if not isinstance(candidate_values, list) or len(candidate_values) < 2:
            raise ActionCatalogError(
                f"action {gap_id!r} has no finite source-derived candidate domain"
            )
        try:
            encoded_values = [
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                for value in candidate_values
            ]
        except (TypeError, ValueError) as error:
            raise ActionCatalogError(
                f"action {gap_id!r} has non-JSON candidate values"
            ) from error
        if len(encoded_values) != len(set(encoded_values)):
            raise ActionCatalogError(f"action {gap_id!r} has duplicate candidate values")
        if not any(
            type(value) is type(gap["incumbent_value"])
            and value == gap["incumbent_value"]
            for value in candidate_values
        ):
            raise ActionCatalogError(
                f"action {gap_id!r} candidate domain omits the typed incumbent"
            )
        _evidence_parts(gap)
        catalog[gap_id] = {
            **gap,
            "kernel_ids": list(kernel_ids),
            "model_callsites": list(model_callsites),
            "source_axis": source_axis,
            "existing_low_level_proven": True,
            "action_edge": dict(edge),
        }

    return graph, catalog


def load_action_catalog(repo: Path) -> dict[str, dict]:
    """Return every red-link action selectable under the frozen model gate."""

    return load_action_graph(repo)[1]


def _validated_frontier(graph: dict) -> dict[str, dict]:
    """Validate the extractor-emitted frontier (novel-slot) records of a graph.

    Frontier slots share the mined-action interface but are not implemented yet:
    they carry no ``source_sink`` and no red ``action_edge`` in the graph. A graph
    without a ``frontier_actions`` array degrades to an empty frontier.
    """

    existing_gap_ids = {
        gap.get("gap_id")
        for gap in graph.get("gaps") or []
        if isinstance(gap, dict) and isinstance(gap.get("gap_id"), str)
    }
    frontier: dict[str, dict] = {}
    for record in graph.get("frontier_actions") or []:
        if not isinstance(record, dict):
            raise ActionCatalogError("frontier action is not an object")
        gap_id = record.get("gap_id")
        if not isinstance(gap_id, str) or not gap_id:
            raise ActionCatalogError("frontier action is missing gap_id")
        if gap_id in frontier:
            raise ActionCatalogError(f"duplicate frontier gap_id {gap_id!r}")
        if gap_id in existing_gap_ids:
            raise ActionCatalogError(
                f"frontier gap_id {gap_id!r} collides with a mined finding"
            )
        category = record.get("category")
        if category not in MINABLE_INTERFACE_CATEGORIES:
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has non-minable category {category!r}"
            )
        family = record.get("family")
        source_axis = record.get("source_axis")
        source_function = record.get("source_function")
        if not isinstance(family, str) or not family:
            raise ActionCatalogError(f"frontier action {gap_id!r} has no source family")
        if not isinstance(source_axis, str) or not source_axis.isidentifier():
            raise ActionCatalogError(f"frontier action {gap_id!r} has no source axis")
        if gap_id != f"{family}:{source_axis}:{category}":
            raise ActionCatalogError(
                f"frontier action {gap_id!r} disagrees with its family/axis/category"
            )
        if not isinstance(source_function, str) or not source_function.isidentifier():
            raise ActionCatalogError(f"frontier action {gap_id!r} has no source function")
        if (
            record.get("incumbent_value_known") is not True
            or "incumbent_value" not in record
        ):
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has no source-derived incumbent value"
            )
        candidate_values = record.get("candidate_values")
        if not isinstance(candidate_values, list) or len(candidate_values) < 2:
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has no finite source-derived candidate domain"
            )
        try:
            encoded_values = [
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                for value in candidate_values
            ]
        except (TypeError, ValueError) as error:
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has non-JSON candidate values"
            ) from error
        if len(encoded_values) != len(set(encoded_values)):
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has duplicate candidate values"
            )
        if not any(
            type(value) is type(record["incumbent_value"])
            and value == record["incumbent_value"]
            for value in candidate_values
        ):
            raise ActionCatalogError(
                f"frontier action {gap_id!r} candidate domain omits the typed incumbent"
            )
        kernel_ids = record.get("kernel_ids")
        if not isinstance(kernel_ids, list) or not kernel_ids:
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has no kernel family mapping"
            )
        model_callsites = record.get("model_callsites")
        if not isinstance(model_callsites, list) or not model_callsites:
            raise ActionCatalogError(f"frontier action {gap_id!r} has no model callsite")
        if len(model_callsites) != len(set(model_callsites)):
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has duplicate model callsites"
            )
        _evidence_parts(record)
        edge = record.get("action_edge")
        if (
            not isinstance(edge, dict)
            or edge.get("source") != f"action:{gap_id}"
            or not isinstance(edge.get("target"), str)
            or not edge["target"].startswith("pallas:")
        ):
            raise ActionCatalogError(
                f"frontier action {gap_id!r} has an invalid action edge"
            )
        frontier[gap_id] = record
    return frontier


def load_frontier_catalog(repo: Path) -> dict[str, dict]:
    """Return the validated not-yet-implemented frontier slots from the graph."""

    graph = json.loads((repo / GRAPH).read_text())
    return _validated_frontier(graph)


def action_catalog_context(repo: Path) -> dict:
    """Return the canonical immutable action context and its content fingerprint."""

    graph, catalog = load_action_graph(repo)
    frontier = _validated_frontier(graph)
    payload = {
        "version": ACTION_CONTEXT_VERSION,
        "graph_target": graph.get("graph_target") or {},
        "actions": [
            {**_canonical_action(catalog[gap_id]), "access": "existing"}
            for gap_id in sorted(catalog)
        ],
        "frontier_actions": [
            {**_canonical_action(frontier[gap_id]), "access": "frontier"}
            for gap_id in sorted(frontier)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        **payload,
        "fingerprint": hashlib.sha256(raw.encode()).hexdigest(),
    }


def action_catalog_fingerprint(repo: Path) -> str:
    """Stable foreign-key namespace for pending capability manifests."""

    return action_catalog_context(repo)["fingerprint"]


__all__ = [
    "ActionCatalogError",
    "ACTION_CONTEXT_VERSION",
    "MINABLE_INTERFACE_CATEGORIES",
    "source_sink_errors",
    "action_semantic_record",
    "action_catalog_context",
    "action_catalog_fingerprint",
    "load_action_catalog",
    "load_action_graph",
    "load_frontier_catalog",
]
