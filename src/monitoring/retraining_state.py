from __future__ import annotations
import fsspec
import pandas as pd
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
RETRAINING_EVENT_HISTORY_FILENAME = (
    "retraining_event_history.parquet"
)


def get_retraining_state_path() -> str:
    return join_uri(
        get_path("monitoring"),
        RETRAINING_STATE_FILENAME,
    )

def get_retraining_event_history_path() -> str:
    return join_uri(
        get_path("monitoring"),
        RETRAINING_EVENT_HISTORY_FILENAME,
    )

def append_retraining_event(
    event: dict[str, Any],
) -> pd.DataFrame:
    """
    Append one completed retraining event.

    The event history is separate from performance history because
    performance windows may be rebuilt from delayed-label batches.
    """
    path = get_retraining_event_history_path()
    new_row = pd.DataFrame([event])

    if file_exists(path):
        with fsspec.open(path, "rb") as file:
            history = pd.read_parquet(file)

        history = pd.concat(
            [history, new_row],
            ignore_index=True,
        )
    else:
        history = new_row

    if "decision_id" in history.columns:
        history = history.drop_duplicates(
            subset=["decision_id"],
            keep="last",
        ).reset_index(drop=True)

    with fsspec.open(path, "wb") as file:
        history.to_parquet(
            file,
            index=False,
        )

    return history
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

    actual_retrained_at_utc = (
        datetime.now(timezone.utc).isoformat()
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
            actual_retrained_at_utc
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

    event_at_utc = (
        simulated_retrained_at_utc
        or actual_retrained_at_utc
    )

    append_retraining_event(
        {
            "event_at_utc": event_at_utc,
            "actual_retrained_at_utc": (
                actual_retrained_at_utc
            ),
            "simulated_retrained_at_utc": (
                simulated_retrained_at_utc
            ),
            "decision_id": (
                decision.decision_id
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
            "candidate_run_id": (
                candidate_run_id
            ),
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
            "trigger_types": json.dumps(
                list(decision.trigger_types)
            ),
            "reasons": json.dumps(
                list(decision.reasons)
            ),
        }
    )

    return payload