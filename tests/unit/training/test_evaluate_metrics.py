from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.training import evaluate_metrics


def test_get_decision_threshold_from_run():
    run = MagicMock()
    run.data.params = {
        "decision_threshold": "0.42",
    }

    client = MagicMock()
    client.get_run.return_value = run

    evaluate_metrics.MlflowClient = MagicMock(
        return_value=client
    )

    threshold = (
        evaluate_metrics
        .get_decision_threshold_from_run(
            "test-run",
        )
    )

    assert threshold == 0.42

    client.get_run.assert_called_once_with(
        "test-run"
    )


def test_get_decision_threshold_from_run_uses_default():
    run = MagicMock()
    run.data.params = {}

    client = MagicMock()
    client.get_run.return_value = run

    evaluate_metrics.MlflowClient = MagicMock(
        return_value=client
    )

    threshold = (
        evaluate_metrics
        .get_decision_threshold_from_run(
            "test-run",
            default=0.37,
        )
    )

    assert threshold == 0.37


def test_predict_with_threshold_converts_probabilities():
    model = MagicMock()
    model.predict.return_value = np.array(
        [
            0.10,
            0.50,
            0.51,
            0.90,
        ]
    )

    predictions = (
        evaluate_metrics
        .predict_with_threshold(
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
            threshold=0.5,
        )
    )

    assert predictions.tolist() == [
        0,
        1,
        1,
        1,
    ]


def test_predict_with_threshold_preserves_class_predictions():
    model = MagicMock()
    model.predict.return_value = np.array(
        [
            0,
            1,
            0,
        ],
        dtype=int,
    )

    predictions = (
        evaluate_metrics
        .predict_with_threshold(
            model,
            pd.DataFrame(
                {
                    "feature": [
                        1,
                        2,
                        3,
                    ]
                }
            ),
            threshold=0.5,
        )
    )

    assert predictions.tolist() == [
        0,
        1,
        0,
    ]


def test_calculate_model_metrics(
    monkeypatch,
):
    model = MagicMock()

    model.predict.return_value = [
        0.10,
        0.80,
        0.30,
        0.90,
    ]

    business_metrics = {
        "expected_profit": 100.0,
        "realized_profit": 90.0,
        "realized_profit_per_action": 4.5,
        "intervention_rate": 0.5,
        "intervention_cost": 20.0,
        "intervened_churners": 2.0,
    }

    business_mock = MagicMock(
        return_value=business_metrics
    )

    monkeypatch.setattr(
        evaluate_metrics,
        "calculate_model_business_metrics",
        business_mock,
    )

    y_true = pd.Series(
        [
            0,
            1,
            0,
            1,
        ]
    )

    metrics = (
        evaluate_metrics
        .calculate_model_metrics(
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
            y_true,
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

    assert metrics["expected_profit"] == 100.0
    assert metrics["realized_profit"] == 90.0

    business_mock.assert_called_once()

    call = business_mock.call_args.kwargs

    assert call["y_true"].equals(
        y_true
    )
    assert call[
        "probabilities"
    ].tolist() == [
        0.10,
        0.80,
        0.30,
        0.90,
    ]


def test_get_business_evaluation_config(
    monkeypatch,
):
    base_config = MagicMock(
        max_discount_budget=1000.0,
        max_discount_rate=0.5,
    )

    from_config_mock = MagicMock(
        return_value=base_config
    )

    monkeypatch.setattr(
        evaluate_metrics.DecisionConfig,
        "from_config",
        from_config_mock,
    )
    monkeypatch.setattr(
        evaluate_metrics,
        "CFG",
        {
            "decision": {},
        },
    )
    monkeypatch.setattr(
        evaluate_metrics,
        "TRAIN_CFG",
        {
            "promotion": {
                "business_evaluation": {
                    "max_discount_rate": 0.25,
                }
            }
        },
    )

    # dataclasses.replace requires a real dataclass instance.
    decision_config = (
        evaluate_metrics.DecisionConfig(
            customer_value=100.0,
            cost_contact=1.0,
            cost_discount=10.0,
            contact_uplift=0.05,
            discount_uplift=0.10,
            max_discount_budget=1000.0,
            max_discount_rate=0.50,
        )
    )

    from_config_mock.return_value = (
        decision_config
    )

    result = (
        evaluate_metrics
        .get_business_evaluation_config()
    )

    assert result.max_discount_budget == 0.0
    assert result.max_discount_rate == 0.25