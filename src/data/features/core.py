import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def cast_numeric_types(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts specified columns to numeric, handling common data issues
    like empty strings " " being present in numeric columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # Errors='coerce' replaces " " with NaN, then we fill with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def add_churn_domain_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add validated churn-specific tenure features.

    These features are available at prediction time and do not use
    target information or future data.
    """
    df = df.copy()

    if "tenure" not in df.columns:
        return df

    tenure = pd.to_numeric(
        df["tenure"],
        errors="coerce",
    ).fillna(0.0)

    df["tenure_group"] = pd.cut(
        tenure,
        bins=[
            -1,
            6,
            12,
            24,
            48,
            float("inf"),
        ],
        labels=[
            "0_6_months",
            "7_12_months",
            "13_24_months",
            "25_48_months",
            "49_plus_months",
        ],
    ).astype("string")

    df["is_new_customer"] = (
        tenure <= 12
    ).astype(int)

    return df

def encode_categoricals(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Transforms categorical string columns into numerical dummies.
    """
    df = df.copy()
    # Using drop_first=True to reduce multicollinearity
    # (Especially important for linear models, but good practice for GB too)
    return pd.get_dummies(df, columns=columns, drop_first=True)


def drop_unnecessary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes columns that should not be used as features (IDs, raw targets).
    """
    df = df.copy()
    existing_cols = [c for c in columns if c in df.columns]
    return df.drop(columns=existing_cols)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names for the model: lowercase, no spaces, no special characters.
    """
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") 
        for col in df.columns
    ]
    return df

def cast_ohe_to_bool(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    Identifies all columns generated via OHE from the categorical_columns 
    defined in training.yaml and casts them to bool.
    """
    # Get the list of categorical source columns from the config
    # This comes from your training.yaml -> features -> categorical_columns
    # df = df.copy()
    # for col in df.columns:
    #     # Prüfe auf Integers/Floats, die nur 0 und 1 enthalten
    #     unique_vals = set(df[col].dropna().unique())
    #     if unique_vals.issubset({0, 1, 0.0, 1.0}):
    #         # Wir weisen die Konvertierung explizit neu zu
    #         df[col] = df[col].map({1: True, 0: False, 1.0: True, 0.0: False}).astype(bool)
    # return df

    if not categorical_columns:
        return df

    # Find all columns that start with one of the categorical column names
    # e.g., if 'gender' is in cat_cols, find 'gender_male', 'gender_female'
    ohe_features = [
        col for col in df.columns 
        if any(col.startswith(f"{base_col}_") for base_col in categorical_columns)
    ]

    for col in ohe_features:
        # Cast to bool to satisfy MLflow schema
        df[col] = df[col].astype(bool)
        
    return df