"""Exact additive planner and bounded empirical challenge for AKT model traces.

Every current executable AKT action is a call-local schedule choice.  A model plan is
therefore a finite sequence of independently measured stages, and the exact additive
optimum is the minimum candidate at each stage.  Target-hardware whole-trace timing
separately checks whether that additive choice survives real interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Mapping, Sequence


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class PlanCandidate:
    callsite: str
    config: tuple[tuple[str, object], ...]
    cost_s: float

    @classmethod
    def from_dict(
        cls,
        callsite: str,
        config: Mapping[str, object],
        cost_s: float,
    ) -> "PlanCandidate":
        return cls(callsite, tuple(sorted(config.items())), float(cost_s))

    def config_dict(self) -> dict[str, object]:
        return dict(self.config)


def _candidate_order(candidate: PlanCandidate) -> tuple[float, str]:
    return candidate.cost_s, _json(candidate.config_dict())


@dataclass(frozen=True)
class PlanResult:
    cost_s: float
    candidates: tuple[PlanCandidate, ...]
    candidates_considered: int

    def plan(self) -> dict[str, dict[str, object]]:
        return {
            candidate.callsite: candidate.config_dict()
            for candidate in self.candidates
        }


@dataclass(frozen=True)
class EmpiricalPlan:
    """One complete plan in the selected-versus-alternative binary panel."""

    candidates: tuple[PlanCandidate, ...]
    choices: tuple[int, ...]

    @property
    def predicted_cost_s(self) -> float:
        return sum(candidate.cost_s for candidate in self.candidates)

    def plan(self) -> dict[str, dict[str, object]]:
        return {
            candidate.callsite: candidate.config_dict()
            for candidate in self.candidates
        }

    def fingerprint(self) -> str:
        payload = [
            {
                "callsite": candidate.callsite,
                "config": candidate.config_dict(),
            }
            for candidate in self.candidates
        ]
        return hashlib.sha256(_json(payload).encode()).hexdigest()


@dataclass(frozen=True)
class EmpiricalPanel:
    """Complete binary neighborhood of the additive selection, or a closed failure."""

    selected: EmpiricalPlan
    plans: tuple[EmpiricalPlan, ...]
    stage_candidates: tuple[tuple[PlanCandidate, ...], ...]
    declared_combinations: int
    max_plans: int
    complete: bool
    error: str | None = None


def _validate_stages(stages: Sequence[Sequence[PlanCandidate]]) -> None:
    if not stages:
        raise ValueError("planner requires at least one stage")
    for index, stage in enumerate(stages):
        if not stage:
            raise ValueError(f"planner stage {index} has no candidates")
        for candidate in stage:
            if not math.isfinite(candidate.cost_s) or candidate.cost_s <= 0:
                raise ValueError(
                    f"planner stage {index} has invalid cost {candidate.cost_s!r}"
                )


def search_dp(stages: Sequence[Sequence[PlanCandidate]]) -> PlanResult:
    """Apply the exact layered recurrence for the finite additive model."""

    _validate_stages(stages)
    selected = tuple(min(stage, key=_candidate_order) for stage in stages)
    return PlanResult(
        cost_s=sum(candidate.cost_s for candidate in selected),
        candidates=selected,
        candidates_considered=sum(len(stage) for stage in stages),
    )


def search_bruteforce(stages: Sequence[Sequence[PlanCandidate]]) -> PlanResult:
    """Independent Cartesian oracle used only on a bounded version of each model."""

    _validate_stages(stages)
    combinations = math.prod(len(stage) for stage in stages)
    best = min(
        itertools.product(*stages),
        key=lambda path: (
            sum(candidate.cost_s for candidate in path),
            _json([candidate.config_dict() for candidate in path]),
        ),
    )
    return PlanResult(
        cost_s=sum(candidate.cost_s for candidate in best),
        candidates=tuple(best),
        candidates_considered=combinations,
    )


def bounded_stages(
    stages: Sequence[Sequence[PlanCandidate]],
    *,
    max_per_stage: int = 2,
) -> list[list[PlanCandidate]]:
    """Retain the deterministic best candidates needed for a tractable oracle."""

    _validate_stages(stages)
    if max_per_stage < 1:
        raise ValueError("max_per_stage must be positive")
    return [sorted(stage, key=_candidate_order)[:max_per_stage] for stage in stages]


def empirical_plan_panel(
    stages: Sequence[Sequence[PlanCandidate]],
    *,
    max_plans: int = 64,
) -> EmpiricalPanel:
    """Enumerate the full selected-versus-cheapest-alternative binary panel."""

    if max_plans < 1:
        raise ValueError("max_plans must be positive")
    selected_candidates = search_dp(stages).candidates
    choices_by_stage = []
    for stage, selected in zip(stages, selected_candidates):
        alternatives = sorted(
            (candidate for candidate in stage if candidate.config != selected.config),
            key=_candidate_order,
        )
        choices_by_stage.append(
            (selected, alternatives[0]) if alternatives else (selected,)
        )

    selected = EmpiricalPlan(
        candidates=selected_candidates,
        choices=tuple(0 for _stage in choices_by_stage),
    )
    declared = math.prod(len(stage) for stage in choices_by_stage)
    if declared > max_plans:
        return EmpiricalPanel(
            selected=selected,
            plans=(selected,),
            stage_candidates=tuple(choices_by_stage),
            declared_combinations=declared,
            max_plans=max_plans,
            complete=False,
            error=(
                f"empirical binary panel has {declared} combinations, exceeding "
                f"the fail-closed limit {max_plans}"
            ),
        )

    plans = tuple(
        EmpiricalPlan(
            candidates=tuple(
                choices_by_stage[index][choice]
                for index, choice in enumerate(choice_indices)
            ),
            choices=tuple(choice_indices),
        )
        for choice_indices in itertools.product(
            *(range(len(stage)) for stage in choices_by_stage)
        )
    )
    if not plans or plans[0] != selected:
        raise AssertionError("additive selected plan is absent from empirical panel")
    return EmpiricalPanel(
        selected=selected,
        plans=plans,
        stage_candidates=tuple(choices_by_stage),
        declared_combinations=declared,
        max_plans=max_plans,
        complete=True,
    )


def certify(stages: Sequence[Sequence[PlanCandidate]]) -> dict:
    """Certify the additive optimum and verify the solver on a bounded Cartesian set."""

    full = search_dp(stages)
    bounded = bounded_stages(stages)
    bounded_dp = search_dp(bounded)
    brute = search_bruteforce(bounded)
    match = math.isclose(
        bounded_dp.cost_s,
        brute.cost_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and bounded_dp.plan() == brute.plan()
    return {
        "certified_additive_optimum": True,
        "full_dp_cost_s": full.cost_s,
        "full_plan": full.plan(),
        "full_candidates_considered": full.candidates_considered,
        "bounded_candidate_sizes": [len(stage) for stage in bounded],
        "bounded_combinations": math.prod(len(stage) for stage in bounded),
        "bounded_dp_cost_s": bounded_dp.cost_s,
        "bounded_bruteforce_cost_s": brute.cost_s,
        "bounded_dp_matches_bruteforce": match,
    }


def measured_graph_fingerprint(
    stages: Sequence[Sequence[PlanCandidate]],
    *,
    timing_protocol: Mapping[str, object],
    timing_samples: object | None = None,
) -> str:
    """Fingerprint every measured candidate, sample set, and timing protocol."""

    payload = {
        "timing_protocol": timing_protocol,
        "timing_samples": timing_samples,
        "stages": [
            [
                {
                    "callsite": candidate.callsite,
                    "config": candidate.config_dict(),
                    "cost_s": candidate.cost_s,
                }
                for candidate in stage
            ]
            for stage in stages
        ],
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def empirical_panel_fingerprint(
    panel: EmpiricalPanel,
    *,
    timing_protocol: Mapping[str, object],
    measured_graph_sha256: str,
) -> str:
    payload = {
        "timing_protocol": timing_protocol,
        "measured_graph_sha256": measured_graph_sha256,
        "complete": panel.complete,
        "declared_combinations": panel.declared_combinations,
        "max_plans": panel.max_plans,
        "plans": [plan.fingerprint() for plan in panel.plans],
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


__all__ = [
    "EmpiricalPanel",
    "EmpiricalPlan",
    "PlanCandidate",
    "PlanResult",
    "bounded_stages",
    "certify",
    "empirical_panel_fingerprint",
    "empirical_plan_panel",
    "measured_graph_fingerprint",
    "search_bruteforce",
    "search_dp",
]
