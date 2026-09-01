from unittest.mock import (
    MagicMock,
)

import numpy as np
import pandas as pd

from src.training import evaluate


def test_evaluate_model_returns_f1(
    monkeypatch,
):
    X_val = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
            ]
        }
    )
    y_val = pd.Series(
        [
            0,
            1,
            0,
            1,
        ]
    )

    model = MagicMock()

    load_validation_mock = MagicMock(
        return_value=(
            X_val,
            y_val,
        )
    )
    predict_mock = MagicMock(
        return_value=np.array(
            [
                0,
                1,
                0,
                1,
            ]
        )
    )

    monkeypatch.setattr(
        evaluate,
        "load_and_prepare_validation_data",
        load_validation_mock,
    )
    monkeypatch.setattr(
        evaluate.mlflow.pyfunc,
        "load_model",
        MagicMock(
            return_value=model
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "predict_with_threshold",
        predict_mock,
    )

    score = evaluate.evaluate_model(
        "champion"
    )

    assert score == 1.0

    evaluate.mlflow.pyfunc.load_model\
        .assert_called_once_with(
            (
                f"models:/"
                f"{evaluate.MODEL_NAME}"
                "@champion"
            )
        )

    predict_mock.assert_called_once_with(
        model,
        X_val,
        threshold=0.5,
    )


def test_evaluate_model_returns_none_when_loading_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        evaluate,
        "load_and_prepare_validation_data",
        MagicMock(
            return_value=(
                pd.DataFrame(
                    {
                        "feature": [
                            1,
                        ]
                    }
                ),
                pd.Series(
                    [
                        0,
                    ]
                ),
            )
        ),
    )
    monkeypatch.setattr(
        evaluate.mlflow.pyfunc,
        "load_model",
        MagicMock(
            side_effect=RuntimeError(
                "Registry unavailable"
            )
        ),
    )

    result = evaluate.evaluate_model(
        "champion"
    )

    assert result is None


def test_generate_and_log_plots(
    monkeypatch,
):
    model = MagicMock()
    X_val = pd.DataFrame(
        {
            "feature": [
                1,
                2,
            ]
        }
    )
    y_val = pd.Series(
        [
            0,
            1,
        ]
    )

    monkeypatch.setattr(
        evaluate,
        "predict_with_threshold",
        MagicMock(
            return_value=np.array(
                [
                    0,
                    1,
                ]
            )
        ),
    )

    run_context = MagicMock()
    run_context.__enter__.return_value = (
        MagicMock()
    )
    run_context.__exit__.return_value = (
        False
    )

    start_run_mock = MagicMock(
        return_value=run_context
    )
    log_artifact_mock = MagicMock()

    monkeypatch.setattr(
        evaluate.mlflow,
        "start_run",
        start_run_mock,
    )
    monkeypatch.setattr(
        evaluate.mlflow,
        "log_artifact",
        log_artifact_mock,
    )

    evaluate._generate_and_log_plots(
        model,
        X_val,
        y_val,
        "test-run",
        threshold=0.42,
    )

    start_run_mock.assert_called_once_with(
        run_id="test-run",
        nested=True,
    )

    log_artifact_mock.assert_called_once()

    artifact_call = (
        log_artifact_mock.call_args
    )

    assert artifact_call.args[1] == (
        "evaluation/plots"
    )
    assert artifact_call.args[0].endswith(
        "confusion_matrix.png"
    )