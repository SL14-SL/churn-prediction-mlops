from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RetrainingAction(StrEnum):
    """Policy action controlling whether automated retraining may proceed."""
    SKIP = "skip"
    TRAIN_CANDIDATE = "train_candidate"
    BLOCK = "block"


@dataclass(frozen=True)
class RetrainingSignals:
    """
    Normalized signals consumed by the retraining decision policy.

    Signal collection is deliberately kept outside this module.
    """

    dataset_version: str | None

    new_training_rows: int
    minimum_training_rows: int

    data_quality_ok: bool
    performance_degraded: bool
    feature_drift_persistent: bool

    cooldown_active: bool
    budget_available: bool

    batch_ids: tuple[str, ...] = ()
    scheduled_retraining_due: bool = False
    days_since_last_training: float | None = None

    performance_window_end: str | None = None
    drift_window_end: str | None = None

    performance_reason: str | None = None
    drift_reason: str | None = None
    data_quality_reason: str | None = None


@dataclass(frozen=True)
class RetrainingDecision:
    """Immutable and reproducibly identified retraining-policy result."""
    action: RetrainingAction
    decision_id: str
    reasons: tuple[str, ...]
    trigger_types: tuple[str, ...]
    evidence: dict[str, Any]

    @property
    def should_train(self) -> bool:
        return self.action == RetrainingAction.TRAIN_CANDIDATE


def _build_decision_id(
    signals: RetrainingSignals,
) -> str:
    """
    Build a stable ID for the same dataset and evaluated monitoring windows.

    Deliberately excludes free-text reasons and wall-clock timestamps.
    """

    identity = {
        "dataset_version": signals.dataset_version,
        "performance_window_end": (
            signals.performance_window_end
        ),
        "drift_window_end": signals.drift_window_end,
        "new_training_rows": signals.new_training_rows,
        "performance_degraded": (
            signals.performance_degraded
        ),
        "feature_drift_persistent": (
            signals.feature_drift_persistent
        ),
        "batch_ids": list(
            signals.batch_ids
        ),
        "scheduled_retraining_due": (
            signals.scheduled_retraining_due
        ),
    }

    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:16]

    return f"retrain-{digest}"


def decide_retraining(
    signals: RetrainingSignals,
) -> RetrainingDecision:
    """
    Decide whether a new Candidate run may be started.

    Precedence:
    1. Block on invalid data or unavailable budget.
    2. Skip during cooldown.
    3. Skip if insufficient new training data is available.
    4. Start a Candidate for persistent performance loss or drift.
    5. New data alone does not trigger retraining.
    """

    decision_id = _build_decision_id(signals)
    evidence = asdict(signals)

    if not signals.data_quality_ok:
        reason = (
            signals.data_quality_reason
            or "Training data quality checks failed."
        )

        return RetrainingDecision(
            action=RetrainingAction.BLOCK,
            decision_id=decision_id,
            reasons=(reason,),
            trigger_types=("data_quality",),
            evidence=evidence,
        )

    if not signals.budget_available:
        return RetrainingDecision(
            action=RetrainingAction.BLOCK,
            decision_id=decision_id,
            reasons=(
                "Retraining budget is currently unavailable.",
            ),
            trigger_types=("budget",),
            evidence=evidence,
        )

    if signals.cooldown_active:
        return RetrainingDecision(
            action=RetrainingAction.SKIP,
            decision_id=decision_id,
            reasons=("Retraining cooldown is active.",),
            trigger_types=(),
            evidence=evidence,
        )

    if (
        signals.new_training_rows
        < signals.minimum_training_rows
    ):
        return RetrainingDecision(
            action=RetrainingAction.SKIP,
            decision_id=decision_id,
            reasons=(
                "Insufficient new training rows: "
                f"{signals.new_training_rows}/"
                f"{signals.minimum_training_rows}.",
            ),
            trigger_types=(),
            evidence=evidence,
        )

    trigger_types: list[str] = []
    reasons: list[str] = []

    if signals.scheduled_retraining_due:
        trigger_types.append(
            "scheduled_refresh",
        )
        reasons.append(
            "Scheduled model refresh interval reached."
        )

    if signals.performance_degraded:
        trigger_types.append("performance_degradation")
        reasons.append(
            signals.performance_reason
            or "Persistent churn model performance degradation detected."
        )

    if signals.feature_drift_persistent:
        trigger_types.append("feature_drift")
        reasons.append(
            signals.drift_reason
            or "Persistent feature drift detected."
        )

    if trigger_types:
        return RetrainingDecision(
            action=RetrainingAction.TRAIN_CANDIDATE,
            decision_id=decision_id,
            reasons=tuple(reasons),
            trigger_types=tuple(trigger_types),
            evidence=evidence,
        )

    return RetrainingDecision(
        action=RetrainingAction.SKIP,
        decision_id=decision_id,
        reasons=(
            "No scheduled refresh, persistent performance degradation "
            "or feature drift detected.",
        ),
        trigger_types=(),
        evidence=evidence,
    )