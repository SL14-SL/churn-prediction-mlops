import mlflow
from mlflow.tracking import MlflowClient
from src.configs.loader import load_config
from src.utils.logger import get_logger
from src.training.promotion_policy import PromotionThresholds, evaluate_promotion_policy


from src.training.dataset import (
    load_and_prepare_recent_production_data,
    load_and_prepare_validation_data,
)
from src.training.evaluate_metrics import (
    get_decision_threshold_from_run,
    calculate_model_metrics,
)
from src.training.evaluate import _generate_and_log_plots



logger = get_logger(__name__)

# Load central configs
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]


def get_promotion_thresholds() -> (
    PromotionThresholds
):
    """Build normalized model-promotion thresholds from training configuration."""
    cfg = TRAIN_CFG.get(
        "promotion",
        {},
    )

    return PromotionThresholds(
        minimum_realized_profit_improvement=float(
            cfg.get(
                "minimum_realized_profit_improvement",
                10.0,
            )
        ),
        maximum_f1_degradation=float(
            cfg.get(
                "maximum_f1_degradation",
                0.02,
            )
        ),
        maximum_recall_degradation=float(
            cfg.get(
                "maximum_recall_degradation",
                0.02,
            )
        ),
        maximum_roc_auc_degradation=float(
            cfg.get(
                "maximum_roc_auc_degradation",
                0.01,
            )
        ),
        maximum_brier_score_increase=float(
            cfg.get(
                "maximum_brier_score_increase",
                0.01,
            )
        ),
    )

def evaluate_reference_safety(
    *,
    challenger_metrics: dict,
    champion_metrics: dict,
    thresholds: PromotionThresholds,
) -> dict[str, bool]:
    """
    Evaluate whether the Champion reference data is safe for model comparison.

    Returns:
        Safety status, reasons and reference-dataset diagnostics.
    """
    reference_cfg = (
        TRAIN_CFG.get(
            "promotion",
            {},
        ).get(
            "reference_safety",
            {},
        )
    )

    maximum_f1_degradation = float(
        reference_cfg.get(
            "maximum_f1_degradation",
            thresholds
            .maximum_f1_degradation,
        )
    )
    maximum_recall_degradation = float(
        reference_cfg.get(
            "maximum_recall_degradation",
            thresholds
            .maximum_recall_degradation,
        )
    )
    maximum_roc_auc_degradation = float(
        reference_cfg.get(
            "maximum_roc_auc_degradation",
            thresholds
            .maximum_roc_auc_degradation,
        )
    )
    maximum_brier_score_increase = float(
        reference_cfg.get(
            "maximum_brier_score_increase",
            thresholds
            .maximum_brier_score_increase,
        )
    )

    return {
        "reference_f1_non_regression": (
            champion_metrics["f1"]
            - challenger_metrics["f1"]
            <= maximum_f1_degradation
        ),
        "reference_recall_non_regression": (
            champion_metrics["recall"]
            - challenger_metrics["recall"]
            <= maximum_recall_degradation
        ),
        "reference_roc_auc_non_regression": (
            champion_metrics["roc_auc"]
            - challenger_metrics["roc_auc"]
            <= maximum_roc_auc_degradation
        ),
        "reference_brier_non_regression": (
            challenger_metrics[
                "brier_score"
            ]
            - champion_metrics[
                "brier_score"
            ]
            <= maximum_brier_score_increase
        ),
    }


   
def compare_models(
    new_run_id: str,
    val_path: str | None = None,
):
    """
    Compare a Candidate with the current Champion across technical and business gates.

    The comparison evaluates predictive quality, calibration, retention economics
    and reference-data safety before applying the configured promotion policy.

    Returns:
        Whether the Candidate is accepted and a complete auditable comparison
        payload.
    """
    del val_path

    client = MlflowClient()
    thresholds = (
        get_promotion_thresholds()
    )

    (
        X_reference,
        y_reference,
    ) = load_and_prepare_validation_data()

    challenger_uri = (
        f"runs:/{new_run_id}/model"
    )

    logger.info(
        "Evaluating Challenger | "
        "run_id=%s",
        new_run_id,
    )

    challenger = (
        mlflow.pyfunc.load_model(
            challenger_uri
        )
    )

    challenger_threshold = (
        get_decision_threshold_from_run(
            new_run_id
        )
    )

    reference_challenger_metrics = (
        calculate_model_metrics(
            challenger,
            X_reference,
            y_reference,
            threshold=(
                challenger_threshold
            ),
        )
    )

    _generate_and_log_plots(
        challenger,
        X_reference,
        y_reference,
        new_run_id,
        threshold=(
            challenger_threshold
        ),
    )

    champion = None
    champion_threshold = None
    reference_champion_metrics = None

    try:
        champion_version = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                "champion",
            )
        )

        champion_threshold = (
            get_decision_threshold_from_run(
                champion_version.run_id
            )
        )

        champion = (
            mlflow.pyfunc.load_model(
                (
                    f"models:/{MODEL_NAME}"
                    "@champion"
                )
            )
        )

        reference_champion_metrics = (
            calculate_model_metrics(
                champion,
                X_reference,
                y_reference,
                threshold=(
                    champion_threshold
                ),
            )
        )

    except Exception as error:
        logger.warning(
            "Champion evaluation unavailable. "
            "Bootstrap promotion policy will "
            "be used | reason=%s",
            error,
        )

    recent_data = None

    if champion is not None:
        recent_data = (
            load_and_prepare_recent_production_data(
                X_reference.columns
            )
        )

    recent_challenger_metrics = None
    recent_champion_metrics = None

    if (
        recent_data is not None
        and champion is not None
        and champion_threshold is not None
    ):
        (
            X_recent,
            y_recent,
        ) = recent_data

        recent_challenger_metrics = (
            calculate_model_metrics(
                challenger,
                X_recent,
                y_recent,
                threshold=(
                    challenger_threshold
                ),
            )
        )

        recent_champion_metrics = (
            calculate_model_metrics(
                champion,
                X_recent,
                y_recent,
                threshold=(
                    champion_threshold
                ),
            )
        )

        primary_challenger_metrics = (
            recent_challenger_metrics
        )
        primary_champion_metrics = (
            recent_champion_metrics
        )
        evaluation_dataset = (
            "recent_production"
        )
    else:
        primary_challenger_metrics = (
            reference_challenger_metrics
        )
        primary_champion_metrics = (
            reference_champion_metrics
        )
        evaluation_dataset = (
            "reference_validation"
        )

    primary_decision = (
        evaluate_promotion_policy(
            challenger_metrics=(
                primary_challenger_metrics
            ),
            champion_metrics=(
                primary_champion_metrics
            ),
            thresholds=thresholds,
        )
    )

    promote = (
        primary_decision.promote
    )
    promotion_gates = dict(
        primary_decision.gates
    )
    promotion_reasons = list(
        primary_decision.reasons
    )
    promotion_evidence = dict(
        primary_decision.evidence
    )

    if (
        evaluation_dataset
        == "recent_production"
        and reference_champion_metrics
        is not None
    ):
        reference_safety_gates = (
            evaluate_reference_safety(
                challenger_metrics=(
                    reference_challenger_metrics
                ),
                champion_metrics=(
                    reference_champion_metrics
                ),
                thresholds=thresholds,
            )
        )

        recent_gates = {
            f"recent_{name}": passed
            for name, passed
            in primary_decision.gates.items()
        }

        promotion_gates = {
            **recent_gates,
            **reference_safety_gates,
        }

        failed_reference_gates = [
            name
            for name, passed
            in reference_safety_gates.items()
            if not passed
        ]

        promote = bool(
            primary_decision.promote
            and all(
                reference_safety_gates.values()
            )
        )

        if failed_reference_gates:
            promotion_reasons.append(
                "Challenger failed reference "
                "safety gates: "
                f"{failed_reference_gates}."
            )

        promotion_evidence = {
            "evaluation_dataset": (
                evaluation_dataset
            ),
            "recent_production": (
                primary_decision.evidence
            ),
            "reference_validation": {
                "challenger": (
                    reference_challenger_metrics
                ),
                "champion": (
                    reference_champion_metrics
                ),
                "safety_gates": (
                    reference_safety_gates
                ),
            },
        }
    else:
        promotion_evidence = (
            primary_decision.evidence
        )

    metrics = {
        "promotion_evaluation_dataset": (
            evaluation_dataset
        ),
        "promotion_gates": (
            promotion_gates
        ),
        "promotion_reasons": (
            promotion_reasons
        ),
        "promotion_evidence": (
            promotion_evidence
        ),
        "challenger_f1": (
            primary_challenger_metrics[
                "f1"
            ]
        ),
        "challenger_recall": (
            primary_challenger_metrics[
                "recall"
            ]
        ),
        "challenger_roc_auc": (
            primary_challenger_metrics[
                "roc_auc"
            ]
        ),
        "challenger_brier_score": (
            primary_challenger_metrics[
                "brier_score"
            ]
        ),
        "challenger_decision_threshold": (
            challenger_threshold
        ),
        "challenger_expected_profit": (
            primary_challenger_metrics[
                "expected_profit"
            ]
        ),
        "challenger_realized_profit": (
            primary_challenger_metrics[
                "realized_profit"
            ]
        ),
        "challenger_profit_per_action": (
            primary_challenger_metrics[
                "realized_profit_per_action"
            ]
        ),
        "challenger_intervention_rate": (
            primary_challenger_metrics[
                "intervention_rate"
            ]
        ),
        "challenger_intervention_cost": (
            primary_challenger_metrics[
                "intervention_cost"
            ]
        ),
        "reference_challenger_f1": (
            reference_challenger_metrics[
                "f1"
            ]
        ),
        "reference_challenger_recall": (
            reference_challenger_metrics[
                "recall"
            ]
        ),
        "reference_challenger_roc_auc": (
            reference_challenger_metrics[
                "roc_auc"
            ]
        ),
        "reference_challenger_brier_score": (
            reference_challenger_metrics[
                "brier_score"
            ]
        ),
        "reference_challenger_realized_profit": (
            reference_challenger_metrics[
                "realized_profit"
            ]
        ),
    }

    if (
        primary_champion_metrics
        is not None
    ):
        metrics.update(
            {
                "champion_f1": (
                    primary_champion_metrics[
                        "f1"
                    ]
                ),
                "champion_recall": (
                    primary_champion_metrics[
                        "recall"
                    ]
                ),
                "champion_roc_auc": (
                    primary_champion_metrics[
                        "roc_auc"
                    ]
                ),
                "champion_brier_score": (
                    primary_champion_metrics[
                        "brier_score"
                    ]
                ),
                "champion_decision_threshold": (
                    champion_threshold
                ),
                "champion_expected_profit": (
                    primary_champion_metrics[
                        "expected_profit"
                    ]
                ),
                "champion_realized_profit": (
                    primary_champion_metrics[
                        "realized_profit"
                    ]
                ),
                "champion_profit_per_action": (
                    primary_champion_metrics[
                        "realized_profit_per_action"
                    ]
                ),
                "champion_intervention_rate": (
                    primary_champion_metrics[
                        "intervention_rate"
                    ]
                ),
                "champion_intervention_cost": (
                    primary_champion_metrics[
                        "intervention_cost"
                    ]
                ),
            }
        )

    if (
        reference_champion_metrics
        is not None
    ):
        metrics.update(
            {
                "reference_champion_f1": (
                    reference_champion_metrics[
                        "f1"
                    ]
                ),
                "reference_champion_recall": (
                    reference_champion_metrics[
                        "recall"
                    ]
                ),
                "reference_champion_roc_auc": (
                    reference_champion_metrics[
                        "roc_auc"
                    ]
                ),
                "reference_champion_brier_score": (
                    reference_champion_metrics[
                        "brier_score"
                    ]
                ),
                "reference_champion_realized_profit": (
                    reference_champion_metrics[
                        "realized_profit"
                    ]
                ),
            }
        )

    if (
        recent_challenger_metrics
        is not None
    ):
        metrics.update(
            {
                "recent_challenger_f1": (
                    recent_challenger_metrics[
                        "f1"
                    ]
                ),
                "recent_challenger_recall": (
                    recent_challenger_metrics[
                        "recall"
                    ]
                ),
                "recent_challenger_roc_auc": (
                    recent_challenger_metrics[
                        "roc_auc"
                    ]
                ),
                "recent_challenger_brier_score": (
                    recent_challenger_metrics[
                        "brier_score"
                    ]
                ),
                "recent_challenger_realized_profit": (
                    recent_challenger_metrics[
                        "realized_profit"
                    ]
                ),
            }
        )

    if (
        recent_champion_metrics
        is not None
    ):
        metrics.update(
            {
                "recent_champion_f1": (
                    recent_champion_metrics[
                        "f1"
                    ]
                ),
                "recent_champion_recall": (
                    recent_champion_metrics[
                        "recall"
                    ]
                ),
                "recent_champion_roc_auc": (
                    recent_champion_metrics[
                        "roc_auc"
                    ]
                ),
                "recent_champion_brier_score": (
                    recent_champion_metrics[
                        "brier_score"
                    ]
                ),
                "recent_champion_realized_profit": (
                    recent_champion_metrics[
                        "realized_profit"
                    ]
                ),
            }
        )

    logger.info(
        "Promotion decision | "
        "dataset=%s promote=%s "
        "gates=%s reasons=%s",
        evaluation_dataset,
        promote,
        promotion_gates,
        promotion_reasons,
    )

    return (
        promote,
        metrics,
    )

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default_run_id"
    compare_models(run_id)