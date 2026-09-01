import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.inference.decision import DecisionConfig, DecisionEngine
from src.configs.loader import load_config

logger = get_logger(__name__)

CFG = load_config()


def validate_prediction_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize raw prediction input.
    Ensures column names are consistent (lowercase, underscores).
    """
    # 1. Basic Cleaning (match training preprocessing)
    validated_df = input_df.copy()
    validated_df.columns = [
        str(col).lower().replace(" ", "_") 
        for col in validated_df.columns
    ]
    
    return validated_df


def align_features_for_model(
    processed_df: pd.DataFrame,
    model,
    model_type: str,
    feature_schema: dict | None = None,
) -> pd.DataFrame:
    """
    Align inference dataframe to the exact feature structure expected by the model.

    Priority:
    1. feature_schema (preferred, explicit)
    2. fallback to model metadata (MLflow signature)

    Handles:
    - missing columns (filled with 0)
    - column order
    - dtype enforcement
    """
    try:
        df = processed_df.copy()

        # --- 1. Resolve expected feature list ---
        if feature_schema:
            model_features = feature_schema["columns"]
            expected_dtypes = feature_schema.get("dtypes", {})
        else:
            # Fallback to MLflow metadata
            if model_type == "xgboost":
                try:
                    raw_features = model.get_booster().feature_names
                except AttributeError:
                    raw_features = model.metadata.get_input_schema().input_names()
            else:
                raw_features = model.metadata.get_input_schema().input_names()

            model_features = raw_features       #[f.lower() for f in raw_features]
            expected_dtypes = {}

        # --- 2. Normalize column names ---
        # df.columns = [str(col).lower() for col in df.columns]
        df.columns = [str(col) for col in df.columns]

        # --- 3. Add missing columns ---
        missing_cols = set(model_features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        if missing_cols:
            logger.warning(f"Missing features filled with 0: {missing_cols}")

        # --- 4. Remove unexpected columns ---
        extra_cols = set(df.columns) - set(model_features)
        if extra_cols:
            logger.warning(f"Dropping unexpected columns: {extra_cols}")
            df = df.drop(columns=list(extra_cols))

        # --- 5. Enforce column order ---
        df = df[model_features]

        # --- 6. Enforce dtypes (if schema available) ---
        for col, dtype in expected_dtypes.items():
            if col in df:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    logger.warning(f"Failed to cast column {col} to {dtype}")

        return df

    except Exception as e:
        logger.error(f"Feature alignment failed: {e}")
        logger.debug(f"Available columns: {list(processed_df.columns)}")
        raise

def _build_decision_engine():
    """
    Build decision engine using:
    1. Base config (dev/prod.yaml)
    2. Override with model-specific threshold (if available)
    """
    decision_config = DecisionConfig.from_config(CFG)

    return DecisionEngine(decision_config)

def predict_and_decide(
    input_df: pd.DataFrame,
    model,
    raw_input_df: pd.DataFrame | None = None,
    decision_threshold: float = 0.5,
) -> list[dict]:
    """
    Generate churn predictions and translate them into retention decisions.

    Args:
        model: Loaded churn estimator or MLflow model wrapper.
        features: Validated and model-aligned customer features.
        artifacts: Inference metadata including schema and decision settings.

    Returns:
        Customer-level predictions, actions and expected-value estimates.
    """

    decision_engine = _build_decision_engine()

    raw_preds = model.predict(input_df)

    if isinstance(raw_preds, list) and isinstance(raw_preds[0], dict):
        probs = raw_preds[0]["probabilities"]
    else:
        probs = raw_preds

    probs = np.asarray(probs).reshape(-1).astype(float)

    value_df = raw_input_df if raw_input_df is not None else input_df

    customer_values = estimate_customer_values(
        df=value_df,
        default_value=decision_engine.config.customer_value,
    )

    if len(customer_values) != len(probs):
        raise ValueError(
            f"customer_values length ({len(customer_values)}) "
            f"does not match probs length ({len(probs)})"
        )

    results = decision_engine.decide_batch(
        probs=probs.tolist(),
        customer_values=customer_values,
    )

    # Add classification decision based on tuned threshold.
    for result, prob in zip(results, probs, strict=True):
        result["prediction"] = float(prob)
        result["churn_probability"] = float(prob)
        result["churn_prediction"] = int(prob >= decision_threshold)
        result["decision_threshold"] = float(decision_threshold)

    logger.info(f"DEBUG probs shape: {np.array(raw_preds).shape}")
    logger.info(f"DEBUG customer_values: {customer_values}")
    logger.info(f"DEBUG decision_threshold: {decision_threshold}")

    return results


def estimate_customer_values(
    df: pd.DataFrame,
    default_value: float,
) -> list[float]:
    """
    Estimate per-customer value for churn decisions.

    Strategy (Telco demo):
    - base: monthlycharges
    - multiplier: remaining lifetime approximation
    """

    if "monthlycharges" not in df.columns:
        return [default_value] * len(df)

    monthly = pd.to_numeric(df["monthlycharges"], errors="coerce").fillna(0)

    # ---- Remaining lifetime heuristic ----
    if "tenure" in df.columns:
        tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)

        # new customers more valuable (long future)
        remaining_months = np.where(
            tenure < 12,
            12,
            6
        )
    else:
        remaining_months = 6

    values = monthly * remaining_months

    # safety floor (avoid 0-value customers)
    values = values.clip(lower=default_value * 0.25)

    return values.astype(float).tolist()

def apply_prediction_postprocessing(
    predictions: list[float],
) -> list[float]:
    """
    Apply optional domain-specific post-processing for classification.
    Currently, we just ensure the output format is consistent.
    """
    # might want to clip values or apply a threshold
    # But for now,keep the raw model output (0/1 or probability)
    return [float(pred) for pred in predictions]