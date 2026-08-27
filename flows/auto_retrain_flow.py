from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from prefect import (
    flow,
    get_run_logger,
    task,
)

from flows.training_flow import (
    training_pipeline,
)
from src.monitoring.retraining_policy import RetrainingAction

from src.monitoring.retraining_state import (
    decision_was_processed,
    record_successful_retraining,
)
from src.monitoring.trigger import evaluate_retraining
from src.monitoring.monitoring_refresh import refresh_monitoring_signals

def parse_evaluated_at(
    value: str,
) -> datetime:
    parsed = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )

@task(name="Refresh Monitoring Signals")
def task_refresh_monitoring_signals():
    logger = get_run_logger()

    result = (
        refresh_monitoring_signals()
    )

    logger.info(
        "Monitoring signals refreshed | "
        "ground_truth_rows=%s "
        "performance_updated=%s "
        "performance_rows=%s "
        "feature_drift_updated=%s "
        "feature_drift_rows=%s "
        "performance_reason=%s",
        result.ground_truth_rows,
        result.performance_updated,
        result.performance_rows,
        result.feature_drift_updated,
        result.feature_drift_rows,
        result.performance_reason,
    )

    return result

@flow(name="Auto Retrain Decision Flow")
def auto_retrain_flow(
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    
    logger = get_run_logger()

    task_refresh_monitoring_signals()

    decision = evaluate_retraining(
        evaluated_at = evaluated_at
    )

    logger.info(
        "Retraining decision evaluated | "
        "action=%s decision_id=%s "
        "triggers=%s reasons=%s",
        decision.action.value,
        decision.decision_id,
        list(decision.trigger_types),
        list(decision.reasons),
    )

    if (
        decision.action
        == RetrainingAction.BLOCK
    ):
        logger.error(
            "Retraining blocked by policy | "
            "decision_id=%s",
            decision.decision_id,
        )

        return {
            "status": "blocked",
            "decision_id": (
                decision.decision_id
            ),
            "reasons": list(
                decision.reasons
            ),
        }

    if (
        decision.action
        == RetrainingAction.SKIP
    ):
        logger.info(
            "Retraining skipped by policy | "
            "decision_id=%s",
            decision.decision_id,
        )

        return {
            "status": "skipped",
            "decision_id": (
                decision.decision_id
            ),
            "reasons": list(
                decision.reasons
            ),
        }

    if decision_was_processed(
        decision.decision_id
    ):
        logger.info(
            "Retraining decision was already "
            "processed. Skipping duplicate | "
            "decision_id=%s",
            decision.decision_id,
        )

        return {
            "status": "duplicate",
            "decision_id": (
                decision.decision_id
            ),
            "reasons": [
                (
                    "Decision was already "
                    "processed."
                )
            ],
        }

    logger.info(
        "Policy authorized a Candidate run | "
        "decision_id=%s",
        decision.decision_id,
    )

    training_result = (
        training_pipeline(
            force_run=True
        )
    )

    if not isinstance(
        training_result,
        dict,
    ):
        raise RuntimeError(
            "Training pipeline did not return "
            "a result dictionary."
        )

    state = (
        record_successful_retraining(
            decision=decision,
            training_result=(
                training_result
            ),
            simulated_retrained_at=(
                evaluated_at
            ),
        )
    )

    logger.info(
        "Retraining completed and state "
        "persisted | decision_id=%s | "
        "candidate_run_id=%s | "
        "champion_promoted=%s",
        decision.decision_id,
        state["candidate_run_id"],
        state["champion_promoted"],
    )

    return {
        "status": "retrained",
        "decision_id": (
            decision.decision_id
        ),
        "candidate_run_id": (
            state["candidate_run_id"]
        ),
        "champion_promoted": (
            state["champion_promoted"]
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluated-at",
        type=parse_evaluated_at,
        default=None,
    )

    args = parser.parse_args()

    auto_retrain_flow(
        evaluated_at=args.evaluated_at
    )