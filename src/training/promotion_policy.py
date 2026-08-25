from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from typing import Any


@dataclass(frozen=True)
class PromotionThresholds:
    minimum_f1_improvement: float

    maximum_recall_degradation: float
    maximum_roc_auc_degradation: float
    maximum_brier_score_increase: float


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]
    gates: dict[str, bool]
    evidence: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


def evaluate_promotion_policy(
    *,
    challenger_metrics: dict[
        str,
        float,
    ],
    champion_metrics: (
        dict[str, float] | None
    ),
    thresholds: PromotionThresholds,
) -> PromotionDecision:
    """
    Evaluate explicit classification promotion gates.

    During bootstrap, a valid Challenger is promoted when no Champion
    metrics are available.
    """
    required_metrics = {
        "f1",
        "recall",
        "roc_auc",
        "brier_score",
    }

    missing_challenger_metrics = (
        required_metrics
        - set(challenger_metrics)
    )

    if missing_challenger_metrics:
        raise ValueError(
            "Challenger metrics are missing: "
            f"{sorted(missing_challenger_metrics)}."
        )

    if champion_metrics is None:
        return PromotionDecision(
            promote=True,
            reasons=(
                "No Champion metrics are "
                "available. Bootstrap promotion "
                "is allowed.",
            ),
            gates={
                "bootstrap": True,
            },
            evidence={
                "challenger": (
                    challenger_metrics
                ),
                "champion": None,
                "thresholds": asdict(
                    thresholds
                ),
            },
        )

    missing_champion_metrics = (
        required_metrics
        - set(champion_metrics)
    )

    if missing_champion_metrics:
        raise ValueError(
            "Champion metrics are missing: "
            f"{sorted(missing_champion_metrics)}."
        )

    f1_improvement = (
        challenger_metrics["f1"]
        - champion_metrics["f1"]
    )

    recall_degradation = (
        champion_metrics["recall"]
        - challenger_metrics["recall"]
    )

    roc_auc_degradation = (
        champion_metrics["roc_auc"]
        - challenger_metrics["roc_auc"]
    )

    brier_score_increase = (
        challenger_metrics["brier_score"]
        - champion_metrics["brier_score"]
    )

    gates = {
        "f1_improvement": (
            f1_improvement
            >= thresholds.minimum_f1_improvement
        ),
        "recall_non_regression": (
            recall_degradation
            <= thresholds.maximum_recall_degradation
        ),
        "roc_auc_non_regression": (
            roc_auc_degradation
            <= thresholds.maximum_roc_auc_degradation
        ),
        "brier_score_non_regression": (
            brier_score_increase
            <= thresholds.maximum_brier_score_increase
        ),
    }

    failed_gates = [
        name
        for name, passed
        in gates.items()
        if not passed
    ]

    promote = all(
        gates.values()
    )

    if promote:
        reasons = (
            "Challenger passed all "
            "classification promotion gates.",
        )
    else:
        reasons = (
            "Challenger failed promotion "
            f"gates: {failed_gates}.",
        )

    return PromotionDecision(
        promote=promote,
        reasons=reasons,
        gates=gates,
        evidence={
            "challenger": (
                challenger_metrics
            ),
            "champion": (
                champion_metrics
            ),
            "thresholds": asdict(
                thresholds
            ),
            "deltas": {
                "f1_improvement": (
                    f1_improvement
                ),
                "recall_degradation": (
                    recall_degradation
                ),
                "roc_auc_degradation": (
                    roc_auc_degradation
                ),
                "brier_score_increase": (
                    brier_score_increase
                ),
            },
        },
    )