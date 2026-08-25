import pytest

from src.training.promotion_policy import (
    PromotionThresholds,
    evaluate_promotion_policy,
)


def thresholds() -> (
    PromotionThresholds
):
    return PromotionThresholds(
        minimum_f1_improvement=0.005,
        maximum_recall_degradation=0.02,
        maximum_roc_auc_degradation=0.01,
        maximum_brier_score_increase=0.01,
    )


def champion_metrics() -> dict[
    str,
    float,
]:
    return {
        "f1": 0.75,
        "recall": 0.78,
        "roc_auc": 0.85,
        "brier_score": 0.16,
    }


def test_promotes_challenger_that_passes_all_gates():
    decision = (
        evaluate_promotion_policy(
            challenger_metrics={
                "f1": 0.76,
                "recall": 0.77,
                "roc_auc": 0.85,
                "brier_score": 0.16,
            },
            champion_metrics=(
                champion_metrics()
            ),
            thresholds=thresholds(),
        )
    )

    assert decision.promote is True
    assert all(
        decision.gates.values()
    )


def test_rejects_insufficient_f1_improvement():
    decision = (
        evaluate_promotion_policy(
            challenger_metrics={
                "f1": 0.752,
                "recall": 0.78,
                "roc_auc": 0.85,
                "brier_score": 0.16,
            },
            champion_metrics=(
                champion_metrics()
            ),
            thresholds=thresholds(),
        )
    )

    assert decision.promote is False
    assert (
        decision.gates[
            "f1_improvement"
        ]
        is False
    )


def test_rejects_recall_regression():
    decision = (
        evaluate_promotion_policy(
            challenger_metrics={
                "f1": 0.77,
                "recall": 0.74,
                "roc_auc": 0.85,
                "brier_score": 0.16,
            },
            champion_metrics=(
                champion_metrics()
            ),
            thresholds=thresholds(),
        )
    )

    assert decision.promote is False
    assert (
        decision.gates[
            "recall_non_regression"
        ]
        is False
    )


def test_rejects_roc_auc_regression():
    decision = (
        evaluate_promotion_policy(
            challenger_metrics={
                "f1": 0.77,
                "recall": 0.78,
                "roc_auc": 0.82,
                "brier_score": 0.16,
            },
            champion_metrics=(
                champion_metrics()
            ),
            thresholds=thresholds(),
        )
    )

    assert decision.promote is False
    assert (
        decision.gates[
            "roc_auc_non_regression"
        ]
        is False
    )


def test_rejects_brier_score_regression():
    decision = (
        evaluate_promotion_policy(
            challenger_metrics={
                "f1": 0.77,
                "recall": 0.78,
                "roc_auc": 0.85,
                "brier_score": 0.18,
            },
            champion_metrics=(
                champion_metrics()
            ),
            thresholds=thresholds(),
        )
    )

    assert decision.promote is False
    assert (
        decision.gates[
            "brier_score_non_regression"
        ]
        is False
    )


def test_bootstrap_promotes_without_champion():
    decision = (
        evaluate_promotion_policy(
            challenger_metrics={
                "f1": 0.70,
                "recall": 0.72,
                "roc_auc": 0.80,
                "brier_score": 0.19,
            },
            champion_metrics=None,
            thresholds=thresholds(),
        )
    )

    assert decision.promote is True
    assert decision.gates == {
        "bootstrap": True,
    }


def test_missing_challenger_metric_is_rejected():
    with pytest.raises(
        ValueError,
        match="Challenger metrics are missing",
    ):
        evaluate_promotion_policy(
            challenger_metrics={
                "f1": 0.76,
                "recall": 0.77,
                "roc_auc": 0.85,
            },
            champion_metrics=(
                champion_metrics()
            ),
            thresholds=thresholds(),
        )