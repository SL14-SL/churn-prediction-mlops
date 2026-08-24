from unittest.mock import (
    MagicMock,
)

import pandas as pd

from src.monitoring import (
    signal_collector,
)


def settings() -> dict:
    return {
        "minimum_new_training_rows": 100,
        "maximum_new_training_rows": (
            100_000
        ),
        "cooldown_hours": 168,
        "scheduled_interval_hours": 168,
        "drift": {
            "lookback_days": 14,
            "consecutive_windows": 2,
        },
        "performance": {
            "consecutive_windows": 2,
            "minimum_samples": 20,
            "min_f1": 0.60,
            "min_recall": 0.65,
            "min_roc_auc": 0.75,
            "max_brier_score": 0.22,
        },
    }


def test_collects_normalized_retraining_signals(
    monkeypatch,
):
    drift_history = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "feature": [
                "Contract",
                "Contract",
            ],
            "drift_detected": [
                True,
                True,
            ],
        }
    )

    performance_history = pd.DataFrame(
        {
            "computed_at": [
                "2026-08-12T00:00:00Z",
                "2026-08-13T00:00:00Z",
            ],
            "n_samples": [
                100,
                120,
            ],
            "f1": [
                0.55,
                0.58,
            ],
            "recall": [
                0.60,
                0.62,
            ],
            "roc_auc": [
                0.70,
                0.72,
            ],
            "brier_score": [
                0.25,
                0.24,
            ],
        }
    )

    monkeypatch.setattr(
        signal_collector,
        "get_retraining_settings",
        MagicMock(
            return_value=settings()
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "get_path",
        MagicMock(
            side_effect=lambda name: {
                "raw_data": "data/raw",
                "monitoring": (
                    "data/monitoring"
                ),
                "models": "models",
            }[name]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_list_files",
        MagicMock(
            return_value=[
                (
                    "data/raw/new_batches/"
                    "train_batch_churn_001.csv"
                ),
            ]
        ),
    )

    mock_batch_reader = MagicMock(
        return_value=(
            150,
            "batch-test-001",
            ("gt-batch-001",),
            True,
            "Ground Truth is valid.",
        )
    )

    monkeypatch.setattr(
        signal_collector,
        "_read_ground_truth_batches",
        mock_batch_reader,
    )
    monkeypatch.setattr(
        signal_collector,
        "_read_parquet_if_available",
        MagicMock(
            side_effect=[
                drift_history,
                performance_history,
            ]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "load_retraining_state",
        MagicMock(
            return_value={}
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_resolve_last_training_at_utc",
        MagicMock(
            return_value=(
                "2026-08-01T01:00:00Z"
            )
        ),
    )

    signals = (
        signal_collector
        .collect_retraining_signals(
            evaluated_at=pd.Timestamp(
                "2026-08-13T01:00:00Z"
            )
        )
    )

    mock_batch_reader.assert_called_once_with(
        [
            (
                "data/raw/new_batches/"
                "train_batch_churn_001.csv"
            ),
        ],
        processed_batch_ids=set(),
    )

    assert signals.batch_ids == (
        "gt-batch-001",
    )
    assert signals.dataset_version == (
        "batch-test-001"
    )
    assert (
        signals.new_training_rows
        == 150
    )
    assert signals.data_quality_ok is True
    assert (
        signals.performance_degraded
        is True
    )
    assert (
        signals.feature_drift_persistent
        is True
    )
    assert signals.cooldown_active is False
    assert signals.budget_available is True
    assert (
        signals.scheduled_retraining_due
        is True
    )
    assert (
        signals.days_since_last_training
        == 12.0
    )


def test_recent_retraining_activates_cooldown():
    state = {
        "last_retrained_at_utc": (
            "2026-08-12T00:00:00Z"
        ),
    }

    result = (
        signal_collector
        ._cooldown_active(
            state,
            evaluated_at=pd.Timestamp(
                "2026-08-13T00:00:00Z"
            ),
            cooldown_hours=168,
        )
    )

    assert result is True


def test_expired_cooldown_is_inactive():
    state = {
        "last_retrained_at_utc": (
            "2026-08-01T00:00:00Z"
        ),
    }

    result = (
        signal_collector
        ._cooldown_active(
            state,
            evaluated_at=pd.Timestamp(
                "2026-08-13T00:00:00Z"
            ),
            cooldown_hours=168,
        )
    )

    assert result is False


def test_rows_above_budget_are_reported(
    monkeypatch,
):
    empty_history = pd.DataFrame()

    monkeypatch.setattr(
        signal_collector,
        "get_retraining_settings",
        MagicMock(
            return_value=settings()
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "get_path",
        MagicMock(
            side_effect=lambda name: {
                "raw_data": "data/raw",
                "monitoring": (
                    "data/monitoring"
                ),
                "models": "models",
            }[name]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_list_files",
        MagicMock(
            return_value=[
                (
                    "data/raw/new_batches/"
                    "train_batch_churn_001.csv"
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_read_ground_truth_batches",
        MagicMock(
            return_value=(
                100_001,
                "batch-too-large",
                ("gt-batch-large",),
                True,
                "Ground Truth is valid.",
            )
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_read_parquet_if_available",
        MagicMock(
            side_effect=[
                empty_history,
                empty_history,
            ]
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "load_retraining_state",
        MagicMock(
            return_value={}
        ),
    )
    monkeypatch.setattr(
        signal_collector,
        "_resolve_last_training_at_utc",
        MagicMock(
            return_value=None
        ),
    )

    signals = (
        signal_collector
        .collect_retraining_signals(
            evaluated_at=pd.Timestamp(
                "2026-08-13T01:00:00Z"
            )
        )
    )

    assert (
        signals.new_training_rows
        == 100_001
    )
    assert (
        signals.budget_available
        is False
    )


def test_processed_batch_rows_are_not_new(
    tmp_path,
    monkeypatch,
):
    batch_path = (
        tmp_path
        / "train_batch_churn_test.csv"
    )

    batch_path.write_text(
        (
            "customerID,Churn\n"
            "1000-AAAAA,Yes\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        signal_collector,
        "validate_train",
        lambda frame: frame,
    )

    first_result = (
        signal_collector
        ._read_ground_truth_batches(
            [str(batch_path)],
            processed_batch_ids=set(),
        )
    )

    first_rows = first_result[0]
    batch_ids = first_result[2]

    assert first_result[3] is True
    assert first_rows == 1
    assert batch_ids

    second_result = (
        signal_collector
        ._read_ground_truth_batches(
            [str(batch_path)],
            processed_batch_ids=set(
                batch_ids
            ),
        )
    )

    assert second_result[0] == 0
    assert second_result[2] == batch_ids
    assert second_result[3] is True


def test_active_release_is_training_fallback(
    monkeypatch,
):
    manifest = MagicMock(
        created_at_utc=(
            "2026-08-01T00:00:00Z"
        )
    )

    monkeypatch.setattr(
        signal_collector,
        "load_active_serving_manifest",
        MagicMock(
            return_value=(
                manifest,
                (
                    "models/serving_releases/"
                    "release-test"
                ),
            )
        ),
    )

    result = (
        signal_collector
        ._resolve_last_training_at_utc(
            {},
            models_path="models",
        )
    )

    assert result == (
        "2026-08-01T00:00:00Z"
    )