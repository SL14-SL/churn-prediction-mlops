import pandas as pd
from src.configs.loader import load_config, get_path
from src.storage.filesystem import file_exists

from src.utils.logger import get_logger
from src.training.utils import build_drop_columns

from src.data.features.build_features import build_features


logger = get_logger(__name__)

CFG = load_config()
TRAIN_CFG = load_config("training.yaml")

def load_and_prepare_validation_data():
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

def load_and_prepare_recent_production_data(
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
