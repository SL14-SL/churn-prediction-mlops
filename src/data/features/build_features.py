import pandas as pd
from src.configs.loader import load_config
from src.data.features import core
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")

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
