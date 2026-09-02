from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.features import pipeline


def test_build_features_import_is_callable():
    """Ensure the pipeline imports the feature builder function."""
    assert callable(pipeline.build_features)


@patch("src.data.features.pipeline.os.makedirs")
@patch("src.data.features.pipeline.build_features")
@patch("src.data.features.pipeline.pd.read_parquet")
@patch("src.data.features.pipeline.file_exists")
def test_run_feature_pipeline_local_success(
    mock_file_exists,
    mock_read_parquet,
    mock_build_features,
    mock_makedirs,
):
    """Load validated data, build features, and persist them locally."""
    input_df = pd.DataFrame(
        {
            "customerID": ["customer-1"],
            "tenure": [12],
        }
    )
    featured_df = MagicMock()
    featured_df.shape = (1, 2)

    mock_file_exists.return_value = True
    mock_read_parquet.return_value = input_df
    mock_build_features.return_value = featured_df

    config = {
        "data": {
            "target_column": "Churn",
        }
    }

    with (
        patch(
            "src.data.features.pipeline.VALIDATED_PATH",
            "data/validation",
        ),
        patch(
            "src.data.features.pipeline.FEATURES_PATH",
            "data/features",
        ),
    ):
        pipeline.run_feature_pipeline(config=config)

    mock_file_exists.assert_called_once_with(
        "data/validation/train.parquet"
    )
    mock_read_parquet.assert_called_once_with(
        "data/validation/train.parquet"
    )
    mock_build_features.assert_called_once_with(
        input_df,
        config=config,
    )
    mock_makedirs.assert_called_once_with(
        "data/features",
        exist_ok=True,
    )
    featured_df.to_parquet.assert_called_once_with(
        "data/features/features.parquet",
        index=False,
    )


@patch("src.data.features.pipeline.pd.read_parquet")
@patch("src.data.features.pipeline.file_exists")
def test_run_feature_pipeline_missing_validated_data(
    mock_file_exists,
    mock_read_parquet,
):
    """Raise an error when the validated training dataset is missing."""
    mock_file_exists.return_value = False

    with patch(
        "src.data.features.pipeline.VALIDATED_PATH",
        "data/validation",
    ):
        with pytest.raises(
            FileNotFoundError,
            match=(
                "Validated data not found at "
                "data/validation/train.parquet"
            ),
        ):
            pipeline.run_feature_pipeline(
                config={
                    "data": {
                        "target_column": "Churn",
                    }
                }
            )

    mock_read_parquet.assert_not_called()


@patch("src.data.features.pipeline.os.makedirs")
@patch("src.data.features.pipeline.build_features")
@patch("src.data.features.pipeline.pd.read_parquet")
@patch("src.data.features.pipeline.file_exists")
def test_run_feature_pipeline_gcs_output(
    mock_file_exists,
    mock_read_parquet,
    mock_build_features,
    mock_makedirs,
):
    """Persist features to GCS without creating a local directory."""
    input_df = pd.DataFrame(
        {
            "customerID": ["customer-1"],
            "tenure": [12],
        }
    )
    featured_df = MagicMock()
    featured_df.shape = (1, 2)

    mock_file_exists.return_value = True
    mock_read_parquet.return_value = input_df
    mock_build_features.return_value = featured_df

    config = {
        "data": {
            "target_column": "Churn",
        }
    }

    with (
        patch(
            "src.data.features.pipeline.VALIDATED_PATH",
            "gs://test-bucket/validation",
        ),
        patch(
            "src.data.features.pipeline.FEATURES_PATH",
            "gs://test-bucket/features",
        ),
    ):
        pipeline.run_feature_pipeline(config=config)

    mock_file_exists.assert_called_once_with(
        "gs://test-bucket/validation/train.parquet"
    )
    mock_read_parquet.assert_called_once_with(
        "gs://test-bucket/validation/train.parquet"
    )
    mock_build_features.assert_called_once_with(
        input_df,
        config=config,
    )
    mock_makedirs.assert_not_called()
    featured_df.to_parquet.assert_called_once_with(
        "gs://test-bucket/features/features.parquet",
        index=False,
    )