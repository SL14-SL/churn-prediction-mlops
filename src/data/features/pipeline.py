import os
import pandas as pd
from src.configs.loader import get_path, load_config
from src.storage.filesystem import file_exists
from src.utils.logger import get_logger
from src.data.features import build_features

logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")
FEATURES_PATH = get_path("features")
VALIDATED_PATH = get_path("validated_data")


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