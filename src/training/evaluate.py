import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    f1_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from mlflow.tracking import MlflowClient
from src.configs.loader import load_config, get_path
from src.utils.logger import get_logger
from src.training.utils import build_drop_columns
from src.training.promotion_policy import PromotionThresholds, evaluate_promotion_policy

logger = get_logger(__name__)

# Load central configs
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]

def _load_and_prep_val_data():
    """Helper to load validation data consistently for Churn."""
    val_path = f"{get_path('splits')}/val.parquet"
    drop_columns = build_drop_columns(TRAIN_CFG)
    # clean_names makes columns lowercase
    target_col = TRAIN_CFG["data"]["target_column"].lower().replace(" ", "_")
    
    val_df = pd.read_parquet(val_path)
    X_val = val_df.drop(columns=drop_columns, errors="ignore")
    # Numeric mapping for metrics
    y_val = (
        val_df[target_col]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )
    return X_val, y_val

def get_decision_threshold_from_run(run_id: str, default: float = 0.5) -> float:
    client = MlflowClient()
    run = client.get_run(run_id)
    value = run.data.params.get("decision_threshold")

    if value is None:
        return default

    return float(value)


def predict_with_threshold(model, X_val, threshold: float):
    preds = model.predict(X_val)

    # pyfunc returns probabilities for your churn model
    if hasattr(preds, "dtype") and preds.dtype.kind in {"f", "c"}:
        return (preds >= threshold).astype(int)

    return preds

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

    return {
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

def get_promotion_thresholds() -> (
    PromotionThresholds
):
    cfg = TRAIN_CFG.get(
        "promotion",
        {},
    )

    return PromotionThresholds(
        minimum_f1_improvement=float(
            cfg.get(
                "minimum_f1_improvement",
                0.005,
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

def _generate_and_log_plots(model, X_val, y_val, run_id, threshold: float = 0.5):
    """Generates evaluation plots and logs them to the specific MLflow run."""
    logger.info(f"Generating evaluation plots for run {run_id}...")
    
    # 1. Predict (Handle pyfunc potential probabilistic output)
    preds = predict_with_threshold(model, X_val, threshold)
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close() # Important to free memory
    
    # 3. Feature Importance Plot (if model supports it)
    # Note: For pyfunc, access to feature_importances_ can be complex.
    # A common workaround is to log feature importance during train.py
    # or to unwrap the model. For simplicity in this blueprint, 
    # we focus on the confusion matrix which is always available.
    # To add feature importance, you'd need framework-specific unwrapping.

    # 4. Log Artifacts to the specific run
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_artifact(cm_path, "evaluation/plots")
        logger.info(f"Confusion matrix plot logged to run {run_id}.")
        
    # Clean up local file
    if os.path.exists(cm_path):
        os.remove(cm_path)

def evaluate_model(model_alias: str = "champion") -> float:
    """
    Evaluates a specific model from the registry on the current validation set.
    Returns the F1-Score.
    """
    X_val, y_val = _load_and_prep_val_data()

    try:
        model_uri = f"models:/{MODEL_NAME}@{model_alias}"
        logger.info(f"Evaluating {model_alias} from registry: {model_uri}")
        
        # Using pyfunc to be framework-agnostic (works for XGB, Sklearn, etc.)
        model = mlflow.pyfunc.load_model(model_uri)
        preds = model.predict(X_val)
        
        # Handle potential probability outputs from pyfunc
        preds = (preds > 0.5).astype(int) if preds.dtype == float else preds
        
        f1 = f1_score(y_val, preds)
        logger.info(f"Model {model_alias} F1-Score: {f1:.4f}")
        return float(f1)
        
    except Exception as e:
        logger.warning(f"Could not evaluate {model_alias}: {e}")
        return None

    
def compare_models(
    new_run_id: str,
    val_path: str | None = None,
):
    """
    Compare Challenger and Champion using explicit promotion gates.

    Returns:
        (promote: bool, metrics: dict)
    """
    del val_path

    client = MlflowClient()
    X_val, y_val = (
        _load_and_prep_val_data()
    )

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

    challenger_metrics = (
        calculate_model_metrics(
            challenger,
            X_val,
            y_val,
            threshold=(
                challenger_threshold
            ),
        )
    )

    _generate_and_log_plots(
        challenger,
        X_val,
        y_val,
        new_run_id,
        threshold=(
            challenger_threshold
        ),
    )

    champion_metrics = None
    champion_threshold = None

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

        champion_metrics = (
            calculate_model_metrics(
                champion,
                X_val,
                y_val,
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

    decision = (
        evaluate_promotion_policy(
            challenger_metrics=(
                challenger_metrics
            ),
            champion_metrics=(
                champion_metrics
            ),
            thresholds=(
                get_promotion_thresholds()
            ),
        )
    )

    metrics = {
        "challenger_f1": (
            challenger_metrics["f1"]
        ),
        "challenger_recall": (
            challenger_metrics["recall"]
        ),
        "challenger_roc_auc": (
            challenger_metrics["roc_auc"]
        ),
        "challenger_brier_score": (
            challenger_metrics[
                "brier_score"
            ]
        ),
        "challenger_decision_threshold": (
            challenger_threshold
        ),
        "promotion_gates": (
            decision.gates
        ),
        "promotion_reasons": list(
            decision.reasons
        ),
    }

    if champion_metrics is not None:
        metrics.update(
            {
                "champion_f1": (
                    champion_metrics["f1"]
                ),
                "champion_recall": (
                    champion_metrics[
                        "recall"
                    ]
                ),
                "champion_roc_auc": (
                    champion_metrics[
                        "roc_auc"
                    ]
                ),
                "champion_brier_score": (
                    champion_metrics[
                        "brier_score"
                    ]
                ),
                "champion_decision_threshold": (
                    champion_threshold
                ),
            }
        )

    logger.info(
        "Promotion decision | "
        "promote=%s gates=%s reasons=%s",
        decision.promote,
        decision.gates,
        list(decision.reasons),
    )

    return (
        decision.promote,
        metrics,
    )

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "default_run_id"
    compare_models(run_id)