import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import replace
from sklearn.metrics import (
    confusion_matrix, 
    f1_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from mlflow.tracking import MlflowClient
from src.configs.loader import load_config, get_path, file_exists
from src.utils.logger import get_logger
from src.training.utils import build_drop_columns
from src.training.promotion_policy import PromotionThresholds, evaluate_promotion_policy

from src.inference.decision import DecisionConfig, DecisionEngine
from src.monitoring.performance import compute_business_metrics
from src.data.features.build_features import build_features


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

def _normalize_column_name(
    value: str,
) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _load_and_prep_recent_production_data(
    reference_columns,
) -> (
    tuple[pd.DataFrame, pd.Series]
    | None
):
    """
    Load the most recent labeled production window.

    Raw customer features are transformed with the same feature
    pipeline and aligned to the reference validation schema.
    """
    promotion_cfg = (
        TRAIN_CFG.get(
            "promotion",
            {},
        )
    )
    recent_cfg = (
        promotion_cfg.get(
            "recent_evaluation",
            {},
        )
    )

    if not bool(
        recent_cfg.get(
            "enabled",
            True,
        )
    ):
        return None

    window_size = int(
        recent_cfg.get(
            "window_size",
            300,
        )
    )
    minimum_samples = int(
        recent_cfg.get(
            "minimum_samples",
            window_size,
        )
    )

    path = (
        f"{get_path('monitoring')}/"
        "cumulative_ground_truth.csv"
    )

    if not file_exists(path):
        logger.info(
            "Recent production evaluation "
            "is unavailable because %s "
            "does not exist.",
            path,
        )
        return None

    dataframe = pd.read_csv(path)

    required_columns = {
        "churn",
        "released_simulation_day",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        logger.warning(
            "Recent production evaluation "
            "is unavailable because columns "
            "are missing: %s",
            sorted(missing_columns),
        )
        return None

    dataframe["churn"] = (
        pd.to_numeric(
            dataframe["churn"],
            errors="coerce",
        )
    )
    dataframe[
        "released_simulation_day"
    ] = pd.to_numeric(
        dataframe[
            "released_simulation_day"
        ],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "churn",
            "released_simulation_day",
        ]
    )

    sort_columns = [
        "released_simulation_day",
    ]

    if (
        "prediction_timestamp"
        in dataframe.columns
    ):
        sort_columns.append(
            "prediction_timestamp"
        )

    dataframe = (
        dataframe
        .sort_values(sort_columns)
        .tail(window_size)
        .reset_index(drop=True)
    )

    if len(dataframe) < minimum_samples:
        logger.info(
            "Recent production evaluation "
            "does not have enough samples | "
            "available=%s required=%s",
            len(dataframe),
            minimum_samples,
        )
        return None

    feature_cfg = (
        TRAIN_CFG.get(
            "features",
            {},
        )
    )
    configured_feature_names = [
        *feature_cfg.get(
            "numeric_columns",
            [],
        ),
        *feature_cfg.get(
            "categorical_columns",
            [],
        ),
    ]

    derived_feature_names = {
        _normalize_column_name(
            feature
        )
        for feature
        in feature_cfg.get(
            "derived_columns",
            [],
        )
    }

    raw_feature_names = [
        feature
        for feature
        in configured_feature_names
        if (
            _normalize_column_name(
                feature
            )
            not in derived_feature_names
        )
    ]

    source_columns = {}

    for column in dataframe.columns:
        normalized = (
            _normalize_column_name(
                column
            )
        )

        if normalized.endswith(
            "_raw"
        ):
            continue

        source_columns.setdefault(
            normalized,
            column,
        )

    missing_features = [
        feature
        for feature in raw_feature_names
        if (
            _normalize_column_name(
                feature
            )
            not in source_columns
        )
    ]

    if missing_features:
        raise ValueError(
            "Recent production data is "
            "missing raw model features: "
            f"{sorted(missing_features)}"
        )

    raw_features = pd.DataFrame(
        {
            _normalize_column_name(
                feature
            ): dataframe[
                source_columns[
                    _normalize_column_name(
                        feature
                    )
                ]
            ].reset_index(drop=True)
            for feature
            in raw_feature_names
        }
    )

    transformed_features = (
        build_features(
            raw_features,
            config=TRAIN_CFG,
        )
    )

    transformed_features = (
        transformed_features.reindex(
            columns=reference_columns,
            fill_value=False,
        )
    )

    target = (
        dataframe["churn"]
        .astype(int)
        .reset_index(drop=True)
    )

    logger.info(
        "Recent production evaluation "
        "loaded | samples=%s features=%s",
        len(target),
        len(
            transformed_features.columns
        ),
    )

    return (
        transformed_features,
        target,
    )


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
    ) = _load_and_prep_val_data()

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
            _load_and_prep_recent_production_data(
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