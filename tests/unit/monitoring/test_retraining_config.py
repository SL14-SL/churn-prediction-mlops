from unittest.mock import (
    MagicMock,
)

from src.monitoring import config


def test_builds_classification_retraining_settings(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "get_monitoring_config",
        MagicMock(
            return_value={
                "performance": {
                    "retrain_thresholds": {
                        "min_f1": 0.61,
                        "min_recall": 0.66,
                        "min_roc_auc": 0.76,
                        "max_brier_score": 0.21,
                    },
                },
                "retraining": {
                    "minimum_new_training_rows": 100,
                    "maximum_new_training_rows": 50_000,
                    "cooldown_hours": 72,
                    "scheduled_interval_hours": 168,
                    "drift": {
                        "lookback_days": 10,
                        "consecutive_windows": 3,
                    },
                    "performance": {
                        "consecutive_windows": 3,
                        "minimum_samples": 30,
                    },
                },
            }
        ),
    )

    settings = (
        config.get_retraining_settings()
    )

    assert (
        settings[
            "minimum_new_training_rows"
        ]
        == 100
    )
    assert (
        settings[
            "maximum_new_training_rows"
        ]
        == 50_000
    )
    assert (
        settings["cooldown_hours"]
        == 72
    )
    assert settings["drift"] == {
        "lookback_days": 10,
        "consecutive_windows": 3,
    }
    assert settings["performance"] == {
        "consecutive_windows": 3,
        "minimum_samples": 30,
        "min_f1": 0.61,
        "min_recall": 0.66,
        "min_roc_auc": 0.76,
        "max_brier_score": 0.21,
    }


def test_retraining_settings_contain_no_regression_metrics(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "get_monitoring_config",
        MagicMock(
            return_value={}
        ),
    )

    settings = (
        config.get_retraining_settings()
    )
    performance = settings[
        "performance"
    ]

    assert "rmse_limit" not in performance
    assert "mae_limit" not in performance
    assert (
        "absolute_bias_limit"
        not in performance
    )