import pandas as pd
from dataclasses import replace
from sklearn.metrics import (
    f1_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from mlflow.tracking import MlflowClient
from src.configs.loader import load_config

from src.inference.decision import DecisionConfig, DecisionEngine
from src.monitoring.performance import compute_business_metrics


# Load central configs
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def get_decision_threshold_from_run(run_id: str, default: float = 0.5) -> float:
    """Return the decision threshold recorded for an MLflow run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    value = run.data.params.get("decision_threshold")

    if value is None:
        return default

    return float(value)

def predict_with_threshold(model, X_val, threshold: float):
    """Convert model probabilities into binary predictions using a threshold."""
    preds = model.predict(X_val)

    # pyfunc returns probabilities for your churn model
    if hasattr(preds, "dtype") and preds.dtype.kind in {"f", "c"}:
        return (preds >= threshold).astype(int)

    return preds


def calculate_model_business_metrics(
    *,
    probabilities,
    y_true,
) -> dict[str, float]:
    """
    Evaluate a model with the existing retention decision policy.

    Realized profit remains simulated because the dataset does not contain
    observed treatment effects.
    """
    decision_config = (
        get_business_evaluation_config()
    )
    decision_engine = (
        DecisionEngine(
            decision_config
        )
    )

    decisions = (
        decision_engine.decide_batch(
            probabilities.tolist()
        )
    )

    evaluation_df = pd.DataFrame(
        {
            "churn": (
                pd.Series(y_true)
                .reset_index(drop=True)
                .astype(int)
            ),
            "churn_probability": (
                probabilities
            ),
            "action": [
                decision["action"]
                for decision in decisions
            ],
            "customer_value": [
                decision[
                    "customer_value"
                ]
                for decision in decisions
            ],
        }
    )

    business_metrics = (
        compute_business_metrics(
            evaluation_df,
            y_true_col="churn",
            y_proba_col=(
                "churn_probability"
            ),
            action_col="action",
            customer_value=(
                decision_config
                .customer_value
            ),
            cost_contact=(
                decision_config
                .cost_contact
            ),
            cost_discount=(
                decision_config
                .cost_discount
            ),
            contact_uplift=(
                decision_config
                .contact_uplift
            ),
            discount_uplift=(
                decision_config
                .discount_uplift
            ),
        )
    )

    return {
        "expected_profit": float(
            business_metrics[
                "expected_profit"
            ]
        ),
        "realized_profit": float(
            business_metrics[
                "realized_profit"
            ]
        ),
        "realized_profit_per_action": (
            float(
                business_metrics[
                    "realized_profit_per_action"
                ]
            )
        ),
        "intervention_rate": float(
            business_metrics[
                "intervention_rate"
            ]
        ),
        "intervention_cost": float(
            business_metrics[
                "total_intervention_cost"
            ]
        ),
        "intervened_churners": float(
            business_metrics[
                "actual_intervened_churners"
            ]
        ),
    }

def calculate_model_metrics(
    model,
    X_val,
    y_val,
    *,
    threshold: float,
) -> dict[str, float]:
    """
    Calculate classification and calibration metrics from probabilities.
    """
    probabilities = model.predict(
        X_val
    )

    probabilities = (
        pd.Series(probabilities)
        .astype(float)
        .to_numpy()
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    business_metrics = (
        calculate_model_business_metrics(
            probabilities=probabilities,
            y_true=y_val,
        )
    )

    classification_metrics = {
        "f1": float(
            f1_score(
                y_val,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_val,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_val,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_val,
                probabilities,
            )
        ),
    }

    return {
        **classification_metrics,
        **business_metrics,
    }

def get_business_evaluation_config() -> (
    DecisionConfig
):
    """
    Build the policy used for model-promotion evaluation.

    Promotion uses a relative discount limit so results scale with the
    validation-set size. The serving API retains its operational batch budget.
    """
    decision_config = (
        DecisionConfig.from_config(
            CFG
        )
    )

    business_cfg = (
        TRAIN_CFG.get(
            "promotion",
            {},
        ).get(
            "business_evaluation",
            {},
        )
    )

    return replace(
        decision_config,
        max_discount_budget=0.0,
        max_discount_rate=float(
            business_cfg.get(
                "max_discount_rate",
                0.20,
            )
        ),
    )
