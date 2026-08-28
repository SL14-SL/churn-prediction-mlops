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
        "expected_profit": 1_045.0,
        "realized_profit": 1_040.0,
        "realized_profit_per_action": (
            4.20
        ),
        "intervention_rate": 0.30,
        "intervention_cost": 240.0,
        "intervened_churners": 120.0,
    }

    champion_metrics = {
        "f1": 0.75,
        "recall": 0.79,
        "roc_auc": 0.86,
        "brier_score": 0.15,
        "expected_profit": 1_010.0,
        "realized_profit": 1_000.0,
        "realized_profit_per_action": (
            4.00
        ),
        "intervention_rate": 0.31,
        "intervention_cost": 250.0,
        "intervened_churners": 118.0,
    }

    promotion_decision = MagicMock(
        promote=True,
        gates={
            (
                "business_value_"
                "improvement"
            ): True,
            "f1_non_regression": True,
            "recall_non_regression": True,
            "roc_auc_non_regression": True,
            (
                "brier_score_"
                "non_regression"
            ): True,
        },
        reasons=(
            "All gates passed.",
        ),
        evidence={
            "challenger": (
                challenger_metrics
            ),
            "champion": (
                champion_metrics
            ),
            "deltas": {
                (
                    "realized_profit_"
                    "improvement"
                ): 40.0,
            },
        },
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
    assert (
        metrics[
            "challenger_realized_profit"
        ]
        == 1_040.0
    )
    assert (
        metrics[
            "champion_realized_profit"
        ]
        == 1_000.0
    )
    assert (
        metrics[
            "challenger_expected_profit"
        ]
        == 1_045.0
    )
    assert (
        metrics[
            "promotion_evidence"
        ]
        == promotion_decision.evidence
    )
    policy.assert_called_once()

    policy_call = (
        policy.call_args.kwargs
    )

    assert (
        policy_call[
            "challenger_metrics"
        ]
        == challenger_metrics
    )
    assert (
        policy_call[
            "champion_metrics"
        ]
        == champion_metrics
    )

def test_compare_models_uses_bootstrap_policy_without_champion(
    monkeypatch,
):
    challenger = MagicMock()

    challenger_metrics = {
        "f1": 0.70,
        "recall": 0.72,
        "roc_auc": 0.80,
        "brier_score": 0.19,
        "expected_profit": 920.0,
        "realized_profit": 900.0,
        "realized_profit_per_action": (
            3.50
        ),
        "intervention_rate": 0.28,
        "intervention_cost": 220.0,
        "intervened_churners": 105.0,
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
        evidence={
            "challenger": (
                challenger_metrics
            ),
            "champion": None,
        },
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
    assert (
        metrics[
            "challenger_expected_profit"
        ]
        == 920.0
    )
    assert (
        metrics[
            "challenger_realized_profit"
        ]
        == 900.0
    )
    assert (
        metrics[
            "promotion_evidence"
        ]
        == promotion_decision.evidence
    )