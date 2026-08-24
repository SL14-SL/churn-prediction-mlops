import pandas as pd

from src.monitoring.signal_evaluation import (
    evaluate_performance_degradation,
    evaluate_persistent_feature_drift,
)


def evaluate_performance(
    history: pd.DataFrame,
):
    return (
        evaluate_performance_degradation(
            history,
            consecutive_windows=2,
            minimum_samples=20,
            min_f1=0.60,
            min_recall=0.65,
            min_roc_auc=0.75,
            max_brier_score=0.22,
        )
    )


def test_persistent_feature_drift_triggers():
    history = pd.DataFrame(
        [
            {
                "timestamp": (
                    "2026-08-23T00:00:00Z"
                ),
                "feature": "Contract",
                "drift_detected": True,
            },
            {
                "timestamp": (
                    "2026-08-24T00:00:00Z"
                ),
                "feature": "Contract",
                "drift_detected": True,
            },
        ]
    )

    result = (
        evaluate_persistent_feature_drift(
            history,
            evaluated_at=pd.Timestamp(
                "2026-08-24T12:00:00Z"
            ),
            lookback_days=7,
            consecutive_windows=2,
        )
    )

    assert result.triggered is True
    assert "Contract" in result.reason


def test_single_drift_window_does_not_trigger():
    history = pd.DataFrame(
        [
            {
                "timestamp": (
                    "2026-08-23T00:00:00Z"
                ),
                "feature": "Contract",
                "drift_detected": True,
            },
            {
                "timestamp": (
                    "2026-08-24T00:00:00Z"
                ),
                "feature": "Contract",
                "drift_detected": False,
            },
        ]
    )

    result = (
        evaluate_persistent_feature_drift(
            history,
            evaluated_at=pd.Timestamp(
                "2026-08-24T12:00:00Z"
            ),
            lookback_days=7,
            consecutive_windows=2,
        )
    )

    assert result.triggered is False


def test_persistent_performance_degradation_triggers():
    history = pd.DataFrame(
        [
            {
                "computed_at": (
                    "2026-08-23T00:00:00Z"
                ),
                "n_samples": 100,
                "f1": 0.55,
                "recall": 0.70,
                "roc_auc": 0.80,
                "brier_score": 0.20,
            },
            {
                "computed_at": (
                    "2026-08-24T00:00:00Z"
                ),
                "n_samples": 120,
                "f1": 0.58,
                "recall": 0.68,
                "roc_auc": 0.78,
                "brier_score": 0.21,
            },
        ]
    )

    result = evaluate_performance(
        history
    )

    assert result.triggered is True
    assert "f1" in result.reason


def test_one_bad_window_does_not_trigger():
    history = pd.DataFrame(
        [
            {
                "computed_at": (
                    "2026-08-23T00:00:00Z"
                ),
                "n_samples": 100,
                "f1": 0.55,
                "recall": 0.70,
                "roc_auc": 0.80,
                "brier_score": 0.20,
            },
            {
                "computed_at": (
                    "2026-08-24T00:00:00Z"
                ),
                "n_samples": 120,
                "f1": 0.75,
                "recall": 0.78,
                "roc_auc": 0.85,
                "brier_score": 0.15,
            },
        ]
    )

    result = evaluate_performance(
        history
    )

    assert result.triggered is False


def test_insufficient_samples_are_excluded():
    history = pd.DataFrame(
        [
            {
                "computed_at": (
                    "2026-08-23T00:00:00Z"
                ),
                "n_samples": 10,
                "f1": 0.30,
                "recall": 0.30,
                "roc_auc": 0.50,
                "brier_score": 0.40,
            },
            {
                "computed_at": (
                    "2026-08-24T00:00:00Z"
                ),
                "n_samples": 10,
                "f1": 0.30,
                "recall": 0.30,
                "roc_auc": 0.50,
                "brier_score": 0.40,
            },
        ]
    )

    result = evaluate_performance(
        history
    )

    assert result.triggered is False
    assert (
        "sufficient samples"
        in result.reason
    )


def test_f1_score_legacy_column_is_supported():
    history = pd.DataFrame(
        [
            {
                "timestamp": (
                    "2026-08-23T00:00:00Z"
                ),
                "n_samples": 100,
                "f1_score": 0.50,
                "recall": 0.60,
                "roc_auc": 0.70,
                "brier_score": 0.25,
            },
            {
                "timestamp": (
                    "2026-08-24T00:00:00Z"
                ),
                "n_samples": 100,
                "f1_score": 0.52,
                "recall": 0.61,
                "roc_auc": 0.71,
                "brier_score": 0.24,
            },
        ]
    )

    result = evaluate_performance(
        history
    )

    assert result.triggered is True

    