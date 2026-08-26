from unittest.mock import patch

from src.monitoring.prediction_logger import log_prediction


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