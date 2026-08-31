import os
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest


os.environ["PREFECT_API_MODE"] = "ephemeral"
os.environ.pop("PREFECT_API_URL", None)

import flows.training_flow as training_flow
from flows.tasks import (
    registry_tasks,
    serving_tasks
)


@pytest.fixture(autouse=True)
def mock_flow_runtime():
    test_env_cfg = {
        "environment": "test",
        "api": {
            "url": "http://testserver/predict",
        },
        "services": {
            "prefect_api_url": (
                "http://testserver/api"
            ),
        },
    }

    mock_logger = MagicMock()

    with (
        patch(
            "flows.training_flow.ENV_CFG",
            test_env_cfg,
        ),
        patch(
            "flows.training_flow.get_run_logger",
            return_value=mock_logger,
        ),
        patch(
            "flows.tasks.registry_tasks."
            "get_run_logger",
            return_value=mock_logger,
        ),
        patch(
            "flows.tasks.serving_tasks."
            "get_run_logger",
            return_value=mock_logger,
        ),
        patch(
            "flows.tasks.data_tasks.get_run_logger",
            return_value=mock_logger,
        ),
        patch(
            "flows.tasks.training_tasks.get_run_logger",
            return_value=mock_logger,
        ),
    ):
        yield


def test_training_pipeline_stable_system_only_evaluates_champion(
    monkeypatch,
):
    mock_check_drift = MagicMock(
        return_value=False
    )
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock()
    mock_train = MagicMock()
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock()
    mock_publish_release = MagicMock()
    mock_resolve_previous = MagicMock()
    mock_deploy_release = MagicMock()

    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_resolve_previous_release",
        mock_resolve_previous,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "deploy_and_verify_release",
        mock_deploy_release,
    )

    training_flow.training_pipeline.fn(
        force_run=False
    )

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_called_once()

    mock_prepare_data.assert_not_called()
    mock_snapshot_dataset.assert_not_called()
    mock_train.assert_not_called()
    mock_log_dataset_metadata.assert_not_called()
    mock_eval_and_reg.assert_not_called()
    mock_publish_release.assert_not_called()
    mock_resolve_previous.assert_not_called()
    mock_deploy_release.assert_not_called()


def test_training_pipeline_force_run_executes_training_path(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds_test_001",
    }

    registration_result = {
        "promoted": False,
        "alias": "challenger",
        "model_version": "7",
        "model_run_id": "run_123",
        "model_type": "xgboost",
        "decision_threshold": 0.42,
        "metrics": {},
    }

    mock_check_drift = MagicMock(
        return_value=False
    )
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest
    )
    mock_train = MagicMock(
        return_value="run_123"
    )
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(
        return_value=registration_result
    )
    mock_publish_release = MagicMock()
    mock_resolve_previous = MagicMock()
    mock_deploy_release = MagicMock()

    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_resolve_previous_release",
        mock_resolve_previous,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "deploy_and_verify_release",
        mock_deploy_release,
    )

    result = (
        training_flow.training_pipeline.fn(
            force_run=True
        )
    )

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(
        is_drift_run=False
    )
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once()
    mock_log_dataset_metadata.assert_called_once_with(
        "run_123",
        dataset_manifest,
    )
    mock_eval_and_reg.assert_called_once_with(
        "run_123"
    )

    mock_publish_release.assert_not_called()
    mock_resolve_previous.assert_not_called()
    mock_deploy_release.assert_not_called()

    assert result == {
        "run_id": "run_123",
        "candidate_run_id": "run_123",
        "champion_promoted": False,
        "model_version": "7",
        "serving_release_id": None,
        "previous_release_id": None,
        "deployment_status": None,
    }


def test_training_pipeline_drift_with_new_champion_refreshes_api(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds_test_002",
    }

    registration_result = {
        "promoted": True,
        "alias": "champion",
        "model_version": "8",
        "model_run_id": "run_456",
        "model_type": "xgboost",
        "decision_threshold": 0.42,
        "metrics": {
            "challenger_f1": 0.81,
        },
    }

    published_manifest = {
        "release_id": "release-8",
    }

    mock_check_drift = MagicMock(
        return_value=True
    )
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest
    )
    mock_train = MagicMock(
        return_value="run_456"
    )
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(
        return_value=registration_result
    )
    mock_publish_release = MagicMock(
        return_value=published_manifest
    )
    mock_resolve_previous = MagicMock(
        return_value="release-previous"
    )

    mock_deploy_release = MagicMock(
        return_value={
            "deployment_status": "verified",
            "release_id": "release-8",
            "verification": {
                "release_id": "release-8",
            },
            "rolled_back": False,
        }
    )

    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_resolve_previous_release",
        mock_resolve_previous,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "deploy_and_verify_release",
        mock_deploy_release,
    )

    result = (
        training_flow.training_pipeline.fn(
            force_run=False
        )
    )

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(
        is_drift_run=True
    )
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once()
    mock_log_dataset_metadata.assert_called_once_with(
        "run_456",
        dataset_manifest,
    )
    mock_eval_and_reg.assert_called_once_with(
        "run_456"
    )

    mock_publish_release.assert_called_once_with(
        registration_result=(
            registration_result
        ),
        dataset_manifest=dataset_manifest,
    )

    assert result == {
        "run_id": "run_456",
        "candidate_run_id": "run_456",
        "champion_promoted": True,
        "model_version": "8",
        "serving_release_id": "release-8",
        "previous_release_id": (
            "release-previous"
        ),
        "deployment_status": "verified",
    }


def test_training_pipeline_drift_without_new_champion_skips_refresh(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": "ds_test_003",
    }

    registration_result = {
        "promoted": False,
        "alias": "challenger",
        "model_version": "9",
        "model_run_id": "run_789",
        "model_type": "xgboost",
        "decision_threshold": 0.45,
        "metrics": {},
    }

    mock_check_drift = MagicMock(
        return_value=True
    )
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest
    )
    mock_train = MagicMock(
        return_value="run_789"
    )
    mock_log_dataset_metadata = MagicMock()
    mock_eval_and_reg = MagicMock(
        return_value=registration_result
    )
    mock_publish_release = MagicMock()
    mock_resolve_previous = MagicMock()
    mock_deploy_release = MagicMock()

    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_resolve_previous_release",
        mock_resolve_previous,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "deploy_and_verify_release",
        mock_deploy_release,
    )

    result = (
        training_flow.training_pipeline.fn(
            force_run=False
        )
    )

    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(
        is_drift_run=True
    )
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once()
    mock_log_dataset_metadata.assert_called_once_with(
        "run_789",
        dataset_manifest,
    )
    mock_eval_and_reg.assert_called_once_with(
        "run_789"
    )

    mock_publish_release.assert_not_called()
    mock_resolve_previous.assert_not_called()
    mock_deploy_release.assert_not_called()

    assert result == {
        "run_id": "run_789",
        "candidate_run_id": "run_789",
        "champion_promoted": False,
        "model_version": "9",
        "serving_release_id": None,
        "previous_release_id": None,
        "deployment_status": None,
    }


def test_publish_serving_release_task(
    monkeypatch,
):
    registration_result = {
        "promoted": True,
        "alias": "champion",
        "model_version": "8",
        "model_run_id": "run_456",
        "model_type": "xgboost",
        "decision_threshold": 0.42,
        "metrics": {},
    }

    dataset_manifest = {
        "dataset_version": "dataset-1",
        "git_commit": "abc123",
        "effective_config": {
            "environment_config": {
                "environment": "test",
            },
            "training_config": {
                "model": {
                    "type": "xgboost",
                },
            },
        },
    }

    mock_manifest = MagicMock(
        release_id="release-8",
        model_version="8",
    )
    mock_manifest.to_dict.return_value = {
        "release_id": "release-8",
        "model_version": "8",
    }

    mock_publish = MagicMock(
        return_value=mock_manifest
    )
    mock_build_probe = MagicMock(
        return_value={
            "inputs": [
                {
                    "customerID": (
                        "1000-AAAAA"
                    ),
                }
            ],
            "context": {
                "purpose": (
                    "post_deployment_verification"
                ),
            },
        }
    )

    def fake_get_path(
        name,
    ):
        return {
            "models": "models",
            "validated_data": (
                "data/validation"
            ),
        }[name]

    monkeypatch.setattr(
        "flows.tasks.serving_tasks.get_path",
        fake_get_path,
    )
    monkeypatch.setattr(
        "flows.tasks.serving_tasks."
        "build_prediction_probe",
        mock_build_probe,
    )
    monkeypatch.setattr(
        "flows.tasks.serving_tasks."
        "publish_serving_release",
        mock_publish,
    )

    result = (
        serving_tasks
        .task_publish_serving_release
        .fn(
            registration_result=(
                registration_result
            ),
            dataset_manifest=(
                dataset_manifest
            ),
        )
    )

    assert result == {
        "release_id": "release-8",
        "model_version": "8",
    }

    mock_build_probe.assert_called_once_with(
        validated_data_path=(
            "data/validation/train.parquet"
        ),
    )

    mock_publish.assert_called_once()

    call_kwargs = (
        mock_publish.call_args.kwargs
    )

    assert call_kwargs[
        "models_path"
    ] == "models"

    assert call_kwargs[
        "model_name"
    ] == serving_tasks.MODEL_NAME

    assert call_kwargs[
        "model_version"
    ] == "8"

    assert call_kwargs[
        "model_run_id"
    ] == "run_456"

    assert call_kwargs[
        "model_type"
    ] == "xgboost"

    assert call_kwargs[
        "decision_threshold"
    ] == 0.42

    assert call_kwargs[
        "feature_schema_source"
    ] == "models/feature_schema.json"

    assert call_kwargs[
        "dataset_version"
    ] == "dataset-1"

    assert call_kwargs[
        "git_commit"
    ] == "abc123"

    assert call_kwargs[
        "config_hash"
    ] is not None

    assert call_kwargs[
        "prediction_probe_payload"
    ] == mock_build_probe.return_value


def test_publish_serving_release_rejects_non_promoted_model():
    registration_result = {
        "promoted": False,
        "alias": "challenger",
        "model_version": "9",
        "model_run_id": "run_789",
        "model_type": "xgboost",
        "decision_threshold": 0.45,
        "metrics": {},
    }

    with pytest.raises(
        ValueError,
        match="non-promoted model",
    ):
        (
            serving_tasks
            .task_publish_serving_release
            .fn(
                registration_result=(
                    registration_result
                ),
                dataset_manifest={
                    "dataset_version": (
                        "dataset-1"
                    ),
                },
            )
        )

def test_training_pipeline_bootstrap_publishes_and_deploys_initial_release(
    monkeypatch,
):
    dataset_manifest = {
        "dataset_version": (
            "ds-bootstrap-001"
        ),
    }

    registration_result = {
        "promoted": True,
        "alias": "champion",
        "model_version": "1",
        "model_run_id": (
            "run-bootstrap"
        ),
        "model_type": (
            "gradient_boosting"
        ),
        "decision_threshold": 0.38,
        "metrics": {},
    }

    published_manifest = {
        "release_id": (
            "release-bootstrap-v1"
        ),
    }

    deployment_result = {
        "deployment_status": "verified",
        "release_id": (
            "release-bootstrap-v1"
        ),
        "verification": {
            "release_id": (
                "release-bootstrap-v1"
            ),
        },
        "rolled_back": False,
    }

    mock_champion_exists = MagicMock(
        return_value=False
    )
    mock_check_drift = MagicMock(
        return_value=False
    )
    mock_evaluate_champion = MagicMock()
    mock_prepare_data = MagicMock()
    mock_snapshot_dataset = MagicMock(
        return_value=dataset_manifest
    )
    mock_train = MagicMock(
        return_value="run-bootstrap"
    )
    mock_log_dataset_metadata = MagicMock()
    mock_bootstrap_champion = MagicMock(
        return_value=registration_result
    )
    mock_eval_and_reg = MagicMock()
    mock_resolve_previous = MagicMock(
        return_value=None
    )
    mock_publish_release = MagicMock(
        return_value=published_manifest
    )
    mock_deploy_release = MagicMock(
        return_value=deployment_result
    )

    monkeypatch.setattr(
        "flows.training_flow.champion_exists",
        mock_champion_exists,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_evaluate_champion",
        mock_evaluate_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_prepare_data",
        mock_prepare_data,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_snapshot_dataset",
        mock_snapshot_dataset,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_train",
        mock_train,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_log_dataset_metadata",
        mock_log_dataset_metadata,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_bootstrap_champion",
        mock_bootstrap_champion,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_eval_and_reg",
        mock_eval_and_reg,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_resolve_previous_release",
        mock_resolve_previous,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "task_publish_serving_release",
        mock_publish_release,
    )
    monkeypatch.setattr(
        "flows.training_flow."
        "deploy_and_verify_release",
        mock_deploy_release,
    )

    result = (
        training_flow.training_pipeline.fn(
            force_run=True,
            bootstrap=True,
        )
    )

    mock_champion_exists.assert_called_once()
    mock_check_drift.assert_called_once()
    mock_evaluate_champion.assert_not_called()

    mock_prepare_data.assert_called_once_with(
        is_drift_run=False
    )
    mock_snapshot_dataset.assert_called_once()
    mock_train.assert_called_once()
    mock_log_dataset_metadata.assert_called_once_with(
        "run-bootstrap",
        dataset_manifest,
    )

    mock_bootstrap_champion.assert_called_once_with(
        candidate_run_id="run-bootstrap",
    )
    mock_eval_and_reg.assert_not_called()

    mock_resolve_previous.assert_called_once()
    mock_publish_release.assert_called_once_with(
        registration_result=(
            registration_result
        ),
        dataset_manifest=dataset_manifest,
    )
    mock_deploy_release.assert_called_once_with(
        release_id=(
            "release-bootstrap-v1"
        ),
        previous_release_id=None,
    )

    assert result == {
        "run_id": "run-bootstrap",
        "candidate_run_id": (
            "run-bootstrap"
        ),
        "champion_promoted": True,
        "model_version": "1",
        "serving_release_id": (
            "release-bootstrap-v1"
        ),
        "previous_release_id": None,
        "deployment_status": "verified",
    }


def test_training_pipeline_rejects_bootstrap_when_champion_exists(
    monkeypatch,
):
    mock_champion_exists = MagicMock(
        return_value=True
    )
    mock_check_drift = MagicMock()

    monkeypatch.setattr(
        "flows.training_flow.champion_exists",
        mock_champion_exists,
    )
    monkeypatch.setattr(
        "flows.training_flow.task_check_drift",
        mock_check_drift,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Bootstrap rejected: "
            "a Champion already exists"
        ),
    ):
        training_flow.training_pipeline.fn(
            force_run=True,
            bootstrap=True,
        )

    mock_champion_exists.assert_called_once()
    mock_check_drift.assert_not_called()


def test_bootstrap_champion_registers_initial_model(
    monkeypatch,
):
    mock_champion_exists = MagicMock(
        side_effect=[False, False]
    )

    mock_run = MagicMock()
    mock_run.data.params = {
        "model_type": (
            "gradient_boosting"
        ),
        "decision_threshold": "0.38",
    }

    mock_client = MagicMock()
    mock_client.get_run.return_value = (
        mock_run
    )

    mock_client_factory = MagicMock(
        return_value=mock_client
    )

    mock_registered_version = MagicMock()
    mock_registered_version.version = "1"

    mock_register_model = MagicMock(
        return_value=(
            mock_registered_version
        )
    )

    monkeypatch.setattr(
        "flows.tasks.registry_tasks."
        "champion_exists",
        mock_champion_exists,
    )
    monkeypatch.setattr(
        "flows.tasks.registry_tasks."
        "MlflowClient",
        mock_client_factory,
    )
    monkeypatch.setattr(
        "flows.tasks.registry_tasks."
        "register_model",
        mock_register_model,
    )

    result = (
        registry_tasks
        .task_bootstrap_champion
        .fn(
            candidate_run_id=(
                "run-bootstrap"
            ),
        )
    )

    assert (
        mock_champion_exists.call_count
        == 2
    )

    mock_client.get_run.assert_called_once_with(
        "run-bootstrap"
    )
    mock_register_model.assert_called_once_with(
        "run-bootstrap",
        alias="champion",
    )

    assert result == {
        "promoted": True,
        "alias": "champion",
        "model_version": "1",
        "model_run_id": (
            "run-bootstrap"
        ),
        "model_type": (
            "gradient_boosting"
        ),
        "decision_threshold": 0.38,
        "metrics": {},
    }