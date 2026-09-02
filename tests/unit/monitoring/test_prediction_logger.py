import pandas as pd

from unittest.mock import patch

from src.monitoring.prediction_logger import log_prediction, normalize_prediction_log_schema


def test_production_prediction_uses_structured_logging_only():
    with (
        patch(
            "src.monitoring.prediction_logger.file_exists"
        ) as mock_file_exists,
        patch(
            "src.monitoring.prediction_logger.pd.DataFrame.to_parquet"
        ) as mock_to_parquet,
    ):
        log_prediction(
            {"customerID": "test-customer"},
            0.75,
            environment="prod",
            model_alias="champion",
            model_version="1",
            model_run_id="run-1",
        )

    mock_file_exists.assert_not_called()
    mock_to_parquet.assert_not_called()
    

def test_normalize_prediction_log_schema_handles_mixed_total_charges():
    """Normalize mixed TotalCharges values to one parquet-safe dtype."""
    df = pd.DataFrame(
        {
            "customerID": [
                "customer-1",
                "customer-2",
            ],
            "TotalCharges": [
                "29.85",
                5067.45,
            ],
            "request_id": [
                "request-1",
                "request-2",
            ],
        }
    )

    result = normalize_prediction_log_schema(df)

    assert str(result["TotalCharges"].dtype) == "string"
    assert result["TotalCharges"].tolist() == [
        "29.85",
        "5067.45",
    ]

def test_normalized_prediction_log_is_parquet_safe(tmp_path):
    """Persist prediction logs containing mixed raw TotalCharges values."""
    df = pd.DataFrame(
        {
            "customerID": [
                "customer-1",
                "customer-2",
            ],
            "TotalCharges": [
                "29.85",
                5067.45,
            ],
            "prediction": [
                0.2,
                0.8,
            ],
        }
    )

    normalized = normalize_prediction_log_schema(df)
    output_path = tmp_path / "inference_log.parquet"

    normalized.to_parquet(output_path, index=False)
    loaded = pd.read_parquet(output_path)

    assert loaded["TotalCharges"].tolist() == [
        "29.85",
        "5067.45",
    ]