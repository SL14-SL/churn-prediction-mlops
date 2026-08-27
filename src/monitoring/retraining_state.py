from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.configs.loader import get_path
from src.configs.paths import join_uri
from src.storage.filesystem import file_exists, read_text, write_text

from src.monitoring.retraining_policy import (
    RetrainingDecision,
)


RETRAINING_STATE_FILENAME = (
    "retraining_state.json"
)


def get_retraining_state_path() -> str:
    return join_uri(
        get_path("monitoring"),
        RETRAINING_STATE_FILENAME,
    )


def load_retraining_state() -> dict[str, Any]:
    path = get_retraining_state_path()

    if not file_exists(path):
        return {}

    try:
        payload = json.loads(read_text(path))
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def decision_was_processed(
    decision_id: str,
) -> bool:
    state = load_retraining_state()

    return (
        state.get("last_decision_id")
        == decision_id
    )


def record_successful_retraining(
    *,
    decision: RetrainingDecision,
    training_result: dict[str, Any],
    simulated_retrained_at: datetime | None = None,
) -> dict[str, Any]:
    """
    Persist state only after the training pipeline returned successfully.

    A rejected Candidate still counts as a completed retraining run:
    compute was spent and the same evidence must not trigger it again.
    """

    candidate_run_id = training_result.get(
        "candidate_run_id"
    )

    if not candidate_run_id:
        raise ValueError(
            "Training result does not contain "
            "candidate_run_id."
        )

    simulated_retrained_at_utc = None

    if simulated_retrained_at is not None:
        if simulated_retrained_at.tzinfo is None:
            simulated_retrained_at = (
                simulated_retrained_at.replace(
                    tzinfo=timezone.utc
                )
            )

        simulated_retrained_at_utc = (
            simulated_retrained_at.astimezone(
                timezone.utc
            ).isoformat()
        )

    payload = {
        "schema_version": 1,
        "last_decision_id": (
            decision.decision_id
        ),
        "last_retrained_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "simulated_retrained_at_utc": (
            simulated_retrained_at_utc
        ),
        "action": decision.action.value,
        "trigger_types": list(
            decision.trigger_types
        ),
        "reasons": list(decision.reasons),
        "dataset_version": (
            decision.evidence.get(
                "dataset_version"
            )
        ),
        "performance_window_end": (
            decision.evidence.get(
                "performance_window_end"
            )
        ),
        "drift_window_end": (
            decision.evidence.get(
                "drift_window_end"
            )
        ),
        "candidate_run_id": candidate_run_id,
        "final_refit_run_id": (
            training_result.get(
                "final_refit_run_id"
            )
        ),
        "champion_promoted": bool(
            training_result.get(
                "champion_promoted",
                False,
            )
        ),
        "processed_batch_ids": list(
            decision.evidence.get(
                "batch_ids",
                (),
            )
        )
    }

    write_text(
        get_retraining_state_path(),
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
    )

    return payload