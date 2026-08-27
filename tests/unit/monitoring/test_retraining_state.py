import json
from unittest.mock import MagicMock
from datetime import datetime, timezone
import pytest

from src.monitoring import retraining_state
from src.monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingDecision,
)


def decision() -> RetrainingDecision:
    return RetrainingDecision(
        action=(
            RetrainingAction.TRAIN_CANDIDATE
        ),
        decision_id="retrain-test-123",
        reasons=(
            "Persistent degradation.",
        ),
        trigger_types=(
            "performance_degradation",
        ),
        evidence={
            "dataset_version": "batch-v1",
            "batch_ids": (
                "gt-batch-001",
                "gt-batch-002",
            ),
            "performance_window_end": (
                "2026-08-13T00:00:00Z"
            ),
            "drift_window_end": None,
        },
    )


def test_missing_state_returns_empty_dict(
    monkeypatch,
):
    monkeypatch.setattr(
        retraining_state,
        "file_exists",
        MagicMock(return_value=False),
    )

    assert (
        retraining_state.load_retraining_state()
        == {}
    )


def test_decision_was_processed(
    monkeypatch,
):
    monkeypatch.setattr(
        retraining_state,
        "load_retraining_state",
        MagicMock(
            return_value={
                "last_decision_id": (
                    "retrain-test-123"
                )
            }
        ),
    )

    assert retraining_state.decision_was_processed(
        "retrain-test-123"
    )


def test_successful_retraining_is_persisted(
    monkeypatch,
):
    mock_write = MagicMock()
    mock_append_event = MagicMock()

    monkeypatch.setattr(
        retraining_state,
        "get_retraining_state_path",
        MagicMock(
            return_value=(
                "data/monitoring/"
                "retraining_state.json"
            )
        ),
    )
    monkeypatch.setattr(
        retraining_state,
        "write_text",
        mock_write,
    )

    monkeypatch.setattr(
        retraining_state,
        "append_retraining_event",
        mock_append_event,
    )

    result = (
        retraining_state
        .record_successful_retraining(
            decision=decision(),
            training_result={
                "candidate_run_id": "run-123",
                "final_refit_run_id": None,
                "champion_promoted": False,
            },
        )
    )

    assert result["candidate_run_id"] == (
        "run-123"
    )
    assert result["champion_promoted"] is False

    persisted = json.loads(
        mock_write.call_args.args[1]
    )
    assert persisted["last_decision_id"] == (
        "retrain-test-123"
    )
    assert persisted[
        "processed_batch_ids"
    ] == [
        "gt-batch-001",
        "gt-batch-002",
    ]
    mock_append_event.assert_called_once()

    event = (
        mock_append_event.call_args.args[0]
    )

    assert event["decision_id"] == (
        "retrain-test-123"
    )
    assert event["candidate_run_id"] == (
        "run-123"
    )
    assert event["champion_promoted"] is False


def test_result_without_candidate_is_rejected(
    monkeypatch,
):
    monkeypatch.setattr(
        retraining_state,
        "write_text",
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="candidate_run_id",
    ):
        retraining_state.record_successful_retraining(
            decision=decision(),
            training_result={
                "champion_promoted": False,
            },
        )

def test_simulated_retraining_time_is_persisted(
    monkeypatch,
):
    mock_write = MagicMock()

    monkeypatch.setattr(
        retraining_state,
        "get_retraining_state_path",
        MagicMock(
            return_value=(
                "data/monitoring/"
                "retraining_state.json"
            )
        ),
    )
    monkeypatch.setattr(
        retraining_state,
        "write_text",
        mock_write,
    )

    simulated_time = datetime(
        2026,
        8,
        15,
        12,
        0,
        tzinfo=timezone.utc,
    )

    retraining_state.record_successful_retraining(
        decision=decision(),
        training_result={
            "candidate_run_id": "run-123",
            "final_refit_run_id": None,
            "champion_promoted": False,
        },
        simulated_retrained_at=(
            simulated_time
        ),
    )

    persisted = json.loads(
        mock_write.call_args.args[1]
    )

    assert persisted[
        "simulated_retrained_at_utc"
    ] == simulated_time.isoformat()


def test_retraining_event_is_appended(
    monkeypatch,
    tmp_path,
):
    event_path = (
        tmp_path
        / "retraining_event_history.parquet"
    )

    monkeypatch.setattr(
        retraining_state,
        "get_retraining_event_history_path",
        MagicMock(
            return_value=str(event_path)
        ),
    )

    event = {
        "event_at_utc": (
            "2026-08-15T12:00:00+00:00"
        ),
        "decision_id": "retrain-test-123",
        "candidate_run_id": "run-123",
        "champion_promoted": False,
    }

    first = (
        retraining_state
        .append_retraining_event(event)
    )
    second = (
        retraining_state
        .append_retraining_event(
            {
                **event,
                "champion_promoted": True,
            }
        )
    )

    assert len(first) == 1
    assert len(second) == 1
    assert bool(
        second.iloc[0][
            "champion_promoted"
        ]
    ) is True
