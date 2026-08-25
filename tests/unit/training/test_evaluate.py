from unittest.mock import (
    MagicMock,
)

import pandas as pd

from src.training import evaluate


def test_calculate_model_metrics():
    model = MagicMock()

    model.predict.return_value = [
        0.10,
        0.80,
        0.30,
        0.90,
    ]

    metrics = (
        evaluate.calculate_model_metrics(
            model,
            pd.DataFrame(
                {
                    "feature": [
                        1,
                        2,
                        3,
                        4,
                    ]
                }
            ),
            pd.Series(
                [
                    0,
                    1,
                    0,
                    1,
                ]
            ),
            threshold=0.5,
        )
    )

    assert metrics["f1"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert (
        0.0
        <= metrics["brier_score"]
        <= 1.0
    )


def test_compare_models_uses_promotion_policy(
    monkeypatch,
):
    challenger = MagicMock()
    champion = MagicMock()

    version = MagicMock(
        run_id="champion-run",
    )

    client = MagicMock()
    client.get_model_version_by_alias\
        .return_value = version

    challenger_metrics = {
        "f1": 0.78,
        "recall": 0.80,
        "roc_auc": 0.87,
        "brier_score": 0.14,
    }

    champion_metrics = {
        "f1": 0.75,
        "recall": 0.79,
        "roc_auc": 0.86,
        "brier_score": 0.15,
    }

    promotion_decision = MagicMock(
        promote=True,
        gates={
            "f1_improvement": True,
            "recall_non_regression": True,
            "roc_auc_non_regression": True,
            "brier_score_non_regression": True,
        },
        reasons=(
            "All gates passed.",
        ),
    )

    monkeypatch.setattr(
        evaluate,
        "MlflowClient",
        MagicMock(
            return_value=client
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_load_and_prep_val_data",
        MagicMock(
            return_value=(
                pd.DataFrame(
                    {"feature": [1, 2]}
                ),
                pd.Series([0, 1]),
            )
        ),
    )
    monkeypatch.setattr(
        evaluate.mlflow.pyfunc,
        "load_model",
        MagicMock(
            side_effect=[
                challenger,
                champion,
            ]
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "get_decision_threshold_from_run",
        MagicMock(
            side_effect=[
                0.42,
                0.50,
            ]
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "calculate_model_metrics",
        MagicMock(
            side_effect=[
                challenger_metrics,
                champion_metrics,
            ]
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_generate_and_log_plots",
        MagicMock(),
    )

    policy = MagicMock(
        return_value=(
            promotion_decision
        )
    )

    monkeypatch.setattr(
        evaluate,
        "evaluate_promotion_policy",
        policy,
    )

    promote, metrics = (
        evaluate.compare_models(
            "challenger-run"
        )
    )

    assert promote is True
    assert (
        metrics["challenger_f1"]
        == 0.78
    )
    assert (
        metrics["champion_f1"]
        == 0.75
    )
    assert (
        metrics["promotion_gates"]
        == promotion_decision.gates
    )

    policy.assert_called_once()


def test_compare_models_uses_bootstrap_policy_without_champion(
    monkeypatch,
):
    challenger = MagicMock()

    challenger_metrics = {
        "f1": 0.70,
        "recall": 0.72,
        "roc_auc": 0.80,
        "brier_score": 0.19,
    }

    client = MagicMock()
    client.get_model_version_by_alias\
        .side_effect = RuntimeError(
            "Champion alias not found."
        )

    promotion_decision = MagicMock(
        promote=True,
        gates={
            "bootstrap": True,
        },
        reasons=(
            "Bootstrap promotion.",
        ),
    )

    monkeypatch.setattr(
        evaluate,
        "MlflowClient",
        MagicMock(
            return_value=client
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_load_and_prep_val_data",
        MagicMock(
            return_value=(
                pd.DataFrame(
                    {"feature": [1, 2]}
                ),
                pd.Series([0, 1]),
            )
        ),
    )
    monkeypatch.setattr(
        evaluate.mlflow.pyfunc,
        "load_model",
        MagicMock(
            return_value=challenger
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "get_decision_threshold_from_run",
        MagicMock(
            return_value=0.42
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "calculate_model_metrics",
        MagicMock(
            return_value=(
                challenger_metrics
            )
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_generate_and_log_plots",
        MagicMock(),
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_promotion_policy",
        MagicMock(
            return_value=(
                promotion_decision
            )
        ),
    )

    promote, metrics = (
        evaluate.compare_models(
            "challenger-run"
        )
    )

    assert promote is True
    assert "champion_f1" not in metrics
    assert metrics[
        "promotion_gates"
    ] == {
        "bootstrap": True,
    }