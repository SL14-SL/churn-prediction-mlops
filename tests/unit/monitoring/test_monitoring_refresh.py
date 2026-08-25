from unittest.mock import (
    MagicMock,
)

import pandas as pd

from src.monitoring import (
    monitoring_refresh,
)


def labeled_batch(
    *,
    batch_id: str,
    prediction_id: str,
    probability: float,
    churn: str,
    available_at: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label_batch_id": batch_id,
                "prediction_id": (
                    prediction_id
                ),
                "customerID": "1000-AAAAA",
                "churn_probability": (
                    probability
                ),
                "Churn": churn,
                "label_available_at": (
                    available_at
                ),
            }
        ]
    )


def test_builds_performance_history():
    cumulative = pd.concat(
        [
            labeled_batch(
                batch_id="batch-1",
                prediction_id="prediction-1",
                probability=0.90,
                churn="Yes",
                available_at=(
                    "2026-08-24T00:00:00Z"
                ),
            ),
            labeled_batch(
                batch_id="batch-2",
                prediction_id="prediction-2",
                probability=0.10,
                churn="No",
                available_at=(
                    "2026-08-25T00:00:00Z"
                ),
            ),
        ],
        ignore_index=True,
    )

    history = (
        monitoring_refresh
        .build_performance_history(
            cumulative
        )
    )

    assert len(history) == 2
    assert set(
        history["label_batch_id"]
    ) == {
        "batch-1",
        "batch-2",
    }
    assert "f1" in history.columns
    assert "recall" in history.columns
    assert "roc_auc" in history.columns
    assert (
        "brier_score"
        in history.columns
    )


def test_duplicate_prediction_is_removed(
    tmp_path,
):
    first_path = (
        tmp_path / "batch-1.csv"
    )
    second_path = (
        tmp_path / "batch-2.csv"
    )
    output_path = (
        tmp_path / "cumulative.csv"
    )

    first = labeled_batch(
        batch_id="batch-1",
        prediction_id="prediction-1",
        probability=0.80,
        churn="Yes",
        available_at=(
            "2026-08-24T00:00:00Z"
        ),
    )

    second = labeled_batch(
        batch_id="batch-2",
        prediction_id="prediction-1",
        probability=0.90,
        churn="Yes",
        available_at=(
            "2026-08-25T00:00:00Z"
        ),
    )

    first.to_csv(
        first_path,
        index=False,
    )
    second.to_csv(
        second_path,
        index=False,
    )

    cumulative = (
        monitoring_refresh
        .rebuild_cumulative_ground_truth(
            [
                str(first_path),
                str(second_path),
            ],
            output_path=str(
                output_path
            ),
        )
    )

    assert len(cumulative) == 1
    assert (
        cumulative.iloc[0][
            "churn_probability"
        ]
        == 0.90
    )


def test_refresh_without_batches_is_safe(
    monkeypatch,
):
    monkeypatch.setattr(
        monitoring_refresh,
        "get_path",
        MagicMock(
            return_value=(
                "data/monitoring"
            )
        ),
    )
    monkeypatch.setattr(
        monitoring_refresh,
        "_list_files",
        MagicMock(
            return_value=[]
        ),
    )
    monkeypatch.setattr(
        monitoring_refresh,
        "run_feature_drift_check",
        MagicMock(
            return_value=pd.DataFrame()
        ),
    )

    result = (
        monitoring_refresh
        .refresh_monitoring_signals()
    )

    assert (
        result.ground_truth_rows
        == 0
    )
    assert (
        result.performance_updated
        is False
    )
    assert (
        result.feature_drift_updated
        is False
    )