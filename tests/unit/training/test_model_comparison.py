from unittest.mock import (
    MagicMock,
)

import pandas as pd

from src.training import model_comparison


def build_metrics(
    *,
    f1: float,
    recall: float,
    roc_auc: float,
    brier_score: float,
    expected_profit: float,
    realized_profit: float,
) -> dict[str, float]:
    """Build a complete model-comparison metric payload."""
    return {
        "f1": f1,
        "recall": recall,
        "roc_auc": roc_auc,
        "brier_score": brier_score,
        "expected_profit": expected_profit,
        "realized_profit": realized_profit,
        "realized_profit_per_action": 4.0,
        "intervention_rate": 0.30,
        "intervention_cost": 240.0,
        "intervened_churners": 100.0,
    }


def install_common_mocks(
    monkeypatch,
    *,
    challenger,
    champion,
    client,
    metric_side_effect,
):
    X_reference = pd.DataFrame(
        {
            "feature": [
                1,
                2,
            ]
        }
    )
    y_reference = pd.Series(
        [
            0,
            1,
        ]
    )

    monkeypatch.setattr(
        model_comparison,
        "MlflowClient",
        MagicMock(
            return_value=client
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "load_and_prepare_validation_data",
        MagicMock(
            return_value=(
                X_reference,
                y_reference,
            )
        ),
    )
    monkeypatch.setattr(
        model_comparison.mlflow.pyfunc,
        "load_model",
        MagicMock(
            side_effect=[
                challenger,
                champion,
            ]
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "get_decision_threshold_from_run",
        MagicMock(
            side_effect=[
                0.42,
                0.50,
            ]
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "calculate_model_metrics",
        MagicMock(
            side_effect=metric_side_effect
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "_generate_and_log_plots",
        MagicMock(),
    )

    return X_reference, y_reference


def test_compare_models_uses_reference_promotion_policy(
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

    challenger_metrics = build_metrics(
        f1=0.78,
        recall=0.80,
        roc_auc=0.87,
        brier_score=0.14,
        expected_profit=1045.0,
        realized_profit=1040.0,
    )
    champion_metrics = build_metrics(
        f1=0.75,
        recall=0.79,
        roc_auc=0.86,
        brier_score=0.15,
        expected_profit=1010.0,
        realized_profit=1000.0,
    )

    install_common_mocks(
        monkeypatch,
        challenger=challenger,
        champion=champion,
        client=client,
        metric_side_effect=[
            challenger_metrics,
            champion_metrics,
        ],
    )

    monkeypatch.setattr(
        model_comparison,
        "load_and_prepare_recent_production_data",
        MagicMock(
            return_value=None
        ),
    )

    promotion_decision = MagicMock(
        promote=True,
        gates={
            "business_value_improvement": True,
            "f1_non_regression": True,
        },
        reasons=(
            "All gates passed.",
        ),
        evidence={
            "challenger": challenger_metrics,
            "champion": champion_metrics,
        },
    )

    policy = MagicMock(
        return_value=promotion_decision
    )

    monkeypatch.setattr(
        model_comparison,
        "evaluate_promotion_policy",
        policy,
    )

    promote, metrics = (
        model_comparison.compare_models(
            "challenger-run"
        )
    )

    assert promote is True
    assert metrics["challenger_f1"] == 0.78
    assert metrics["champion_f1"] == 0.75
    assert (
        metrics["promotion_evaluation_dataset"]
        == "reference_validation"
    )
    assert (
        metrics["challenger_realized_profit"]
        == 1040.0
    )
    assert (
        metrics["champion_realized_profit"]
        == 1000.0
    )

    policy.assert_called_once()

    policy_call = (
        policy.call_args.kwargs
    )

    assert (
        policy_call["challenger_metrics"]
        == challenger_metrics
    )
    assert (
        policy_call["champion_metrics"]
        == champion_metrics
    )


def test_compare_models_uses_bootstrap_policy_without_champion(
    monkeypatch,
):
    challenger = MagicMock()

    client = MagicMock()
    client.get_model_version_by_alias\
        .side_effect = RuntimeError(
            "Champion alias not found."
        )

    challenger_metrics = build_metrics(
        f1=0.70,
        recall=0.72,
        roc_auc=0.80,
        brier_score=0.19,
        expected_profit=920.0,
        realized_profit=900.0,
    )

    X_reference = pd.DataFrame(
        {
            "feature": [
                1,
                2,
            ]
        }
    )
    y_reference = pd.Series(
        [
            0,
            1,
        ]
    )

    monkeypatch.setattr(
        model_comparison,
        "MlflowClient",
        MagicMock(
            return_value=client
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "load_and_prepare_validation_data",
        MagicMock(
            return_value=(
                X_reference,
                y_reference,
            )
        ),
    )
    monkeypatch.setattr(
        model_comparison.mlflow.pyfunc,
        "load_model",
        MagicMock(
            return_value=challenger
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "get_decision_threshold_from_run",
        MagicMock(
            return_value=0.42
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "calculate_model_metrics",
        MagicMock(
            return_value=(
                challenger_metrics
            )
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "_generate_and_log_plots",
        MagicMock(),
    )
    monkeypatch.setattr(
        model_comparison,
        "load_and_prepare_recent_production_data",
        MagicMock(
            return_value=None
        ),
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
            "challenger": challenger_metrics,
            "champion": None,
        },
    )

    monkeypatch.setattr(
        model_comparison,
        "evaluate_promotion_policy",
        MagicMock(
            return_value=promotion_decision
        ),
    )

    promote, metrics = (
        model_comparison.compare_models(
            "challenger-run"
        )
    )

    assert promote is True
    assert "champion_f1" not in metrics
    assert metrics["promotion_gates"] == {
        "bootstrap": True,
    }
    assert (
        metrics["challenger_expected_profit"]
        == 920.0
    )
    assert (
        metrics["challenger_realized_profit"]
        == 900.0
    )


def test_compare_models_uses_recent_data_and_reference_safety(
    monkeypatch,
):
    challenger = MagicMock()
    champion = MagicMock()

    client = MagicMock()
    client.get_model_version_by_alias\
        .return_value = MagicMock(
            run_id="champion-run"
        )

    X_reference = pd.DataFrame(
        {
            "feature": [
                1,
                2,
            ]
        }
    )
    y_reference = pd.Series(
        [
            0,
            1,
        ]
    )
    X_recent = pd.DataFrame(
        {
            "feature": [
                3,
                4,
                5,
            ]
        }
    )
    y_recent = pd.Series(
        [
            1,
            1,
            0,
        ]
    )

    reference_challenger = build_metrics(
        f1=0.74,
        recall=0.77,
        roc_auc=0.85,
        brier_score=0.15,
        expected_profit=1010.0,
        realized_profit=1005.0,
    )
    reference_champion = build_metrics(
        f1=0.75,
        recall=0.78,
        roc_auc=0.85,
        brier_score=0.15,
        expected_profit=1015.0,
        realized_profit=1010.0,
    )
    recent_challenger = build_metrics(
        f1=0.72,
        recall=0.76,
        roc_auc=0.82,
        brier_score=0.17,
        expected_profit=840.0,
        realized_profit=830.0,
    )
    recent_champion = build_metrics(
        f1=0.66,
        recall=0.68,
        roc_auc=0.78,
        brier_score=0.20,
        expected_profit=790.0,
        realized_profit=780.0,
    )

    monkeypatch.setattr(
        model_comparison,
        "MlflowClient",
        MagicMock(
            return_value=client
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "load_and_prepare_validation_data",
        MagicMock(
            return_value=(
                X_reference,
                y_reference,
            )
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "load_and_prepare_recent_production_data",
        MagicMock(
            return_value=(
                X_recent,
                y_recent,
            )
        ),
    )
    monkeypatch.setattr(
        model_comparison.mlflow.pyfunc,
        "load_model",
        MagicMock(
            side_effect=[
                challenger,
                champion,
            ]
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "get_decision_threshold_from_run",
        MagicMock(
            side_effect=[
                0.40,
                0.38,
            ]
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "calculate_model_metrics",
        MagicMock(
            side_effect=[
                reference_challenger,
                reference_champion,
                recent_challenger,
                recent_champion,
            ]
        ),
    )
    monkeypatch.setattr(
        model_comparison,
        "_generate_and_log_plots",
        MagicMock(),
    )

    promotion_decision = MagicMock(
        promote=True,
        gates={
            "business_value_improvement": True,
            "f1_non_regression": True,
            "recall_non_regression": True,
            "roc_auc_non_regression": True,
            "brier_score_non_regression": True,
        },
        reasons=(
            "Recent production gates passed.",
        ),
        evidence={
            "challenger": recent_challenger,
            "champion": recent_champion,
        },
    )

    policy = MagicMock(
        return_value=promotion_decision
    )

    monkeypatch.setattr(
        model_comparison,
        "evaluate_promotion_policy",
        policy,
    )

    reference_safety = MagicMock(
        return_value={
            "reference_f1_non_regression": True,
            "reference_recall_non_regression": True,
            "reference_roc_auc_non_regression": True,
            "reference_brier_non_regression": True,
        }
    )

    monkeypatch.setattr(
        model_comparison,
        "evaluate_reference_safety",
        reference_safety,
    )

    promote, metrics = (
        model_comparison.compare_models(
            "challenger-run"
        )
    )

    assert promote is True
    assert (
        metrics["promotion_evaluation_dataset"]
        == "recent_production"
    )
    assert (
        metrics["challenger_f1"]
        == recent_challenger["f1"]
    )
    assert (
        metrics["champion_f1"]
        == recent_champion["f1"]
    )
    assert (
        metrics["reference_challenger_f1"]
        == reference_challenger["f1"]
    )
    assert (
        metrics["reference_champion_f1"]
        == reference_champion["f1"]
    )

    policy.assert_called_once()

    policy_call = (
        policy.call_args.kwargs
    )

    assert (
        policy_call["challenger_metrics"]
        == recent_challenger
    )
    assert (
        policy_call["champion_metrics"]
        == recent_champion
    )

    reference_safety.assert_called_once_with(
        challenger_metrics=(
            reference_challenger
        ),
        champion_metrics=(
            reference_champion
        ),
        thresholds=(
            policy_call["thresholds"]
        ),
    )