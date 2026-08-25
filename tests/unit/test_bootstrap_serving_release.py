from unittest.mock import (
    MagicMock,
)

from scripts import (
    bootstrap_serving_release,
)


def test_existing_release_is_not_republished(
    monkeypatch,
):
    monkeypatch.setattr(
        bootstrap_serving_release,
        "load_config",
        MagicMock(
            return_value={
                "model": {
                    "registry_name": (
                        "churn-model"
                    ),
                },
            }
        ),
    )
    monkeypatch.setattr(
        bootstrap_serving_release,
        "get_path",
        MagicMock(
            return_value="models"
        ),
    )

    publish = MagicMock()

    monkeypatch.setattr(
        bootstrap_serving_release,
        "load_active_release_id",
        MagicMock(
            return_value="release-existing"
        ),
    )
    monkeypatch.setattr(
        bootstrap_serving_release,
        "publish_serving_release",
        publish,
    )

    result = (
        bootstrap_serving_release
        .bootstrap_serving_release()
    )

    assert result == {
        "status": "unchanged",
        "release_id": (
            "release-existing"
        ),
    }

    publish.assert_not_called()


def test_bootstraps_existing_champion(
    monkeypatch,
):
    cfg = {
        "model": {
            "registry_name": (
                "churn-model"
            ),
        },
        "tracking": {
            "mlflow_tracking_uri": (
                "http://mlflow:5000"
            ),
        },
    }

    champion_version = MagicMock(
        version="7",
        run_id="run-7",
    )

    run = MagicMock()
    run.data.tags = {
        "model_type": "xgboost",
    }
    run.data.params = {
        "decision_threshold": "0.42",
    }

    client = MagicMock()
    client.get_model_version_by_alias\
        .return_value = (
            champion_version
        )
    client.get_run.return_value = run
    client.download_artifacts\
        .return_value = (
            "/tmp/feature_schema.json"
        )

    manifest = MagicMock(
        release_id="release-7"
    )

    def fake_get_path(name):
        return {
            "models": "models",
            "validated_data": (
                "data/validation"
            ),
        }[name]

    monkeypatch.setattr(
        bootstrap_serving_release,
        "load_config",
        MagicMock(
            return_value=cfg
        ),
    )
    monkeypatch.setattr(
        bootstrap_serving_release,
        "get_path",
        fake_get_path,
    )
    monkeypatch.setattr(
        bootstrap_serving_release,
        "load_active_release_id",
        MagicMock(
            side_effect=FileNotFoundError
        ),
    )
    monkeypatch.setattr(
        bootstrap_serving_release,
        "MlflowClient",
        MagicMock(
            return_value=client
        ),
    )
    monkeypatch.setattr(
        bootstrap_serving_release,
        "build_prediction_probe",
        MagicMock(
            return_value={
                "inputs": [
                    {
                        "customerID": (
                            "1000-AAAAA"
                        ),
                    }
                ]
            }
        ),
    )

    publish = MagicMock(
        return_value=manifest
    )
    deploy = MagicMock(
        return_value={
            "deployment_status": (
                "verified"
            ),
        }
    )

    monkeypatch.setattr(
        bootstrap_serving_release,
        "publish_serving_release",
        publish,
    )
    monkeypatch.setattr(
        bootstrap_serving_release,
        "deploy_and_verify_release",
        deploy,
    )

    result = (
        bootstrap_serving_release
        .bootstrap_serving_release()
    )

    assert result["status"] == (
        "bootstrapped"
    )
    assert result["release_id"] == (
        "release-7"
    )

    call_kwargs = (
        publish.call_args.kwargs
    )

    assert call_kwargs[
        "model_version"
    ] == "7"
    assert call_kwargs[
        "model_run_id"
    ] == "run-7"
    assert call_kwargs[
        "decision_threshold"
    ] == 0.42

    deploy.assert_called_once_with(
        release_id="release-7",
        previous_release_id=None,
    )