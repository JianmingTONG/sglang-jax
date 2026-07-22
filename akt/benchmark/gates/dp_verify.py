"""Verify AKT's model DP against exhaustive search on all three bounded models.

This fast preflight exercises the planner independently of JAX and target hardware.
The target-hardware evaluator repeats the same DP-vs-brute certificate with bounded
subsets of the real measured candidate tables in every campaign round.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from akt.benchmark.model_workloads import MODEL_WORKLOADS, callsite_id
from akt.core.search.model_dp import PlanCandidate, certify


def _cost(site: str, variant: int) -> float:
    value = int(hashlib.sha256(f"{site}:{variant}".encode()).hexdigest()[:8], 16)
    return 1e-6 * (1 + value % 1000)


def verify_models() -> list[dict]:
    reports = []
    for workload in MODEL_WORKLOADS:
        stages = []
        for call in workload.calls:
            site = callsite_id(workload, call)
            stage = []
            for variant in range(2):
                stage.append(
                    PlanCandidate.from_dict(
                        site,
                        {"bounded_variant": variant},
                        _cost(site, variant),
                    )
                )
            stages.append(stage)
        certificate = certify(stages)
        if not certificate["bounded_dp_matches_bruteforce"]:
            raise AssertionError(f"DP differs from brute force for {workload.model_id}")
        reports.append(
            {
                "model": workload.model_id,
                "calls": len(workload.calls),
                **certificate,
            }
        )
    return reports


def main():
    reports = verify_models()
    for report in reports:
        print(
            f"[akt-dp-verify] {report['model']}: "
            f"DP={report['bounded_dp_cost_s']:.9f} "
            f"brute={report['bounded_bruteforce_cost_s']:.9f} "
            f"combinations={report['bounded_combinations']} MATCH"
        )
    print("AKT_DP_VERIFY " + json.dumps({"ok": True, "models": reports}))


if __name__ == "__main__":
    main()
