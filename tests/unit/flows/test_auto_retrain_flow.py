import pytest

from unittest.mock import MagicMock

from flows import auto_retrain_flow
from src.monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingDecision,
)

@pytest.fixture(autouse=True)
def mock_prefect_run_logger(
    monkeypatch,
):
    logger = MagicMock()

    monkeypatch.setattr(
        auto_retrain_flow,
        "get_run_logger",
        MagicMock(return_value=logger),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "task_refresh_monitoring_signals",
        MagicMock(),
    )

    return logger

def make_decision(
    action: RetrainingAction,
) -> RetrainingDecision:
    return RetrainingDecision(
        action=action,
        decision_id="retrain-test-123",
        reasons=("Test reason.",),
        trigger_types=(
            ("performance_degradation",)
            if action
            == RetrainingAction.TRAIN_CANDIDATE
            else ()
        ),
        evidence={
            "dataset_version": "batch-v1",
        },
    )


def test_skip_does_not_start_training(
    monkeypatch,
):
    mock_training = MagicMock()

    monkeypatch.setattr(
        auto_retrain_flow,
        "evaluate_retraining",
        MagicMock(
            return_value=make_decision(
                RetrainingAction.SKIP
            )
        ),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "training_pipeline",
        mock_training,
    )

    result = (
        auto_retrain_flow.auto_retrain_flow.fn()
    )

    assert result["status"] == "skipped"
    mock_training.assert_not_called()


def test_block_does_not_start_training(
    monkeypatch,
):
    mock_training = MagicMock()

    monkeypatch.setattr(
        auto_retrain_flow,
        "evaluate_retraining",
        MagicMock(
            return_value=make_decision(
                RetrainingAction.BLOCK
            )
        ),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "training_pipeline",
        mock_training,
    )

    result = (
        auto_retrain_flow.auto_retrain_flow.fn()
    )

    assert result["status"] == "blocked"
    mock_training.assert_not_called()


def test_duplicate_does_not_start_training(
    monkeypatch,
):
    mock_training = MagicMock()

    monkeypatch.setattr(
        auto_retrain_flow,
        "evaluate_retraining",
        MagicMock(
            return_value=make_decision(
                RetrainingAction.TRAIN_CANDIDATE
            )
        ),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "decision_was_processed",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "training_pipeline",
        mock_training,
    )

    result = (
        auto_retrain_flow.auto_retrain_flow.fn()
    )

    assert result["status"] == "duplicate"
    mock_training.assert_not_called()


def test_training_success_persists_state(
    monkeypatch,
):
    training_result = {
        "candidate_run_id": "run-123",
        "final_refit_run_id": None,
        "champion_promoted": False,
    }

    mock_training = MagicMock(
        return_value=training_result
    )
    mock_record = MagicMock(
        return_value={
            **training_result,
            "last_decision_id": (
                "retrain-test-123"
            ),
        }
    )

    monkeypatch.setattr(
        auto_retrain_flow,
        "evaluate_retraining",
        MagicMock(
            return_value=make_decision(
                RetrainingAction.TRAIN_CANDIDATE
            )
        ),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "decision_was_processed",
        MagicMock(return_value=False),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "training_pipeline",
        mock_training,
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "record_successful_retraining",
        mock_record,
    )

    result = (
        auto_retrain_flow.auto_retrain_flow.fn()
    )

    mock_training.assert_called_once_with(
        force_run=True
    )
    mock_record.assert_called_once()
    assert result["status"] == "retrained"
    assert result["candidate_run_id"] == (
        "run-123"
    )


def test_failed_training_does_not_persist_state(
    monkeypatch,
):
    mock_record = MagicMock()

    monkeypatch.setattr(
        auto_retrain_flow,
        "evaluate_retraining",
        MagicMock(
            return_value=make_decision(
                RetrainingAction.TRAIN_CANDIDATE
            )
        ),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "decision_was_processed",
        MagicMock(return_value=False),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "training_pipeline",
        MagicMock(
            side_effect=RuntimeError(
                "Training failed."
            )
        ),
    )
    monkeypatch.setattr(
        auto_retrain_flow,
        "record_successful_retraining",
        mock_record,
    )

    with pytest.raises(
        RuntimeError,
        match="Training failed",
    ):
        auto_retrain_flow.auto_retrain_flow.fn()

    mock_record.assert_not_called()