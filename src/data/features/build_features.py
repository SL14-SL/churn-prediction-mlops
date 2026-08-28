import os
import pandas as pd
from src.configs.loader import file_exists, get_path, load_config
from src.data.features import core
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")
FEATURES_PATH = get_path("features")
VALIDATED_PATH = get_path("validated_data")

def _get_feature_config(config: dict) -> dict:
    return config.get("features", {})

def _apply_step(
    df: pd.DataFrame,
    *,
    step_name: str,
    config: dict,
) -> pd.DataFrame:
    """
    Dispatcher for feature engineering steps defined in YAML.
    """
    feature_cfg = _get_feature_config(config)

    if step_name == "cast_numeric_types":
        cols = feature_cfg.get("cast_to_numeric", [])
        return core.cast_numeric_types(df, cols)

    if step_name == "add_churn_domain_features":
        return core.add_churn_domain_features(df)

    if step_name == "encode_categoricals":
        cols = feature_cfg.get("categorical_columns", [])
        return core.encode_categoricals(df, cols)

    if step_name == "drop_configured":
        cols = feature_cfg.get("drop_columns", [])
        return core.drop_unnecessary_columns(df, columns=cols)

    # Clean column names is always a good final or initial step
    if step_name == "clean_names":
        return core.clean_column_names(df)
    
    if step_name == "cast_ohe_to_bool":
        cols = feature_cfg.get("categorical_columns", [])
        return core.cast_ohe_to_bool(df, categorical_columns=cols)

    raise ValueError(f"Unknown feature step configured: {step_name}")

def build_features(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Main entry point for building features from a dataframe.
    """
    config = config or TRAIN_CFG
    if df.empty:
        return df.copy()

    df = df.copy()
    feature_cfg = _get_feature_config(config)
    
    # Default steps for Churn if nothing is defined in YAML
    enabled_steps = feature_cfg.get(
        "enabled_steps",
        ["cast_numeric_types", "encode_categoricals", "drop_configured", "clean_names"]
    )

    logger.info(f"Building features | rows={len(df)} | steps={enabled_steps}")

    for step_name in enabled_steps:
        df = _apply_step(df, step_name=step_name, config=config)

    return df

def run_feature_pipeline(config: dict | None = None) -> None:
    """
    End-to-end pipeline: Load validated data -> Build features -> Save.
    """
    config = config or TRAIN_CFG
    logger.info(f"Starting feature pipeline. Data source: {VALIDATED_PATH}")

    try:
        # 1. Load validated data 
        train_path = f"{VALIDATED_PATH}/train.parquet"
        if not file_exists(train_path):
            raise FileNotFoundError(f"Validated data not found at {train_path}")
        
        df = pd.read_parquet(train_path)

        # 2. Transform data
        df = build_features(df, config=config)

        # 3. Save feature set
        if not FEATURES_PATH.startswith("gs://"):
            os.makedirs(FEATURES_PATH, exist_ok=True)

        output_file = f"{FEATURES_PATH}/features.parquet"
        df.to_parquet(output_file, index=False)

        logger.info(f"Feature engineering successful. Final shape: {df.shape}")

    except Exception as e:
        logger.error(f"Critical error in run_feature_pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_feature_pipeline()