from unittest.mock import (
    MagicMock,
)

import pytest

from flows import deployment_flow


def test_deploy_and_verify_release_succeeds(
    monkeypatch,
):
    refresh = MagicMock()

    verify = MagicMock(
        return_value={
            "release_id": "release-new",
        }
    )

    rollback = MagicMock()

    monkeypatch.setattr(
        deployment_flow,
        "task_refresh_api",
        refresh,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_verify_serving_release",
        verify,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_rollback_serving_release",
        rollback,
    )

    result = (
        deployment_flow
        .deploy_and_verify_release(
            release_id=(
                "release-new"
            ),
            previous_release_id=(
                "release-old"
            ),
        )
    )

    assert result[
        "deployment_status"
    ] == "verified"

    assert result[
        "rolled_back"
    ] is False

    refresh.assert_called_once_with()

    verify.assert_called_once_with(
        expected_release_id=(
            "release-new"
        ),
    )

    rollback.assert_not_called()


def test_failed_verification_rolls_back(
    monkeypatch,
):
    refresh = MagicMock()

    verify = MagicMock(
        side_effect=[
            RuntimeError(
                "New release is not ready."
            ),
            {
                "release_id": (
                    "release-old"
                ),
            },
        ]
    )

    rollback = MagicMock(
        return_value={
            "release_id": "release-old",
        }
    )

    monkeypatch.setattr(
        deployment_flow,
        "task_refresh_api",
        refresh,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_verify_serving_release",
        verify,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_rollback_serving_release",
        rollback,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Automatic rollback "
            "completed successfully"
        ),
    ):
        (
            deployment_flow
            .deploy_and_verify_release(
                release_id=(
                    "release-new"
                ),
                previous_release_id=(
                    "release-old"
                ),
            )
        )

    refresh.assert_called_once_with()

    rollback.assert_called_once_with(
        previous_release_id=(
            "release-old"
        ),
    )

    assert verify.call_args_list == [
        (
            (),
            {
                "expected_release_id": (
                    "release-new"
                ),
            },
        ),
        (
            (),
            {
                "expected_release_id": (
                    "release-old"
                ),
            },
        ),
    ]


def test_failed_bootstrap_verification_cannot_rollback(
    monkeypatch,
):
    refresh = MagicMock()

    verify = MagicMock(
        side_effect=RuntimeError(
            "Release is not ready."
        )
    )

    rollback = MagicMock()

    monkeypatch.setattr(
        deployment_flow,
        "task_refresh_api",
        refresh,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_verify_serving_release",
        verify,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_rollback_serving_release",
        rollback,
    )

    with pytest.raises(
        RuntimeError,
        match="no previous release",
    ):
        (
            deployment_flow
            .deploy_and_verify_release(
                release_id=(
                    "release-first"
                ),
                previous_release_id=None,
            )
        )

    refresh.assert_called_once_with()
    rollback.assert_not_called()


def test_failed_rollback_is_reported(
    monkeypatch,
):
    refresh = MagicMock()

    verify = MagicMock(
        side_effect=RuntimeError(
            "New release not ready."
        )
    )

    rollback = MagicMock(
        side_effect=RuntimeError(
            "Rollback endpoint failed."
        )
    )

    monkeypatch.setattr(
        deployment_flow,
        "task_refresh_api",
        refresh,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_verify_serving_release",
        verify,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_rollback_serving_release",
        rollback,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "automatic rollback also failed"
        ),
    ):
        (
            deployment_flow
            .deploy_and_verify_release(
                release_id=(
                    "release-new"
                ),
                previous_release_id=(
                    "release-old"
                ),
            )
        )

    refresh.assert_called_once_with()

    rollback.assert_called_once_with(
        previous_release_id=(
            "release-old"
        ),
    )


def test_refresh_failure_also_triggers_rollback(
    monkeypatch,
):
    refresh = MagicMock(
        side_effect=RuntimeError(
            "API reload failed."
        )
    )
    verify = MagicMock()
    rollback = MagicMock(
        return_value={
            "release_id": "release-old",
        }
    )

    monkeypatch.setattr(
        deployment_flow,
        "task_refresh_api",
        refresh,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_verify_serving_release",
        verify,
    )
    monkeypatch.setattr(
        deployment_flow,
        "task_rollback_serving_release",
        rollback,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Automatic rollback "
            "completed successfully"
        ),
    ):
        (
            deployment_flow
            .deploy_and_verify_release(
                release_id=(
                    "release-new"
                ),
                previous_release_id=(
                    "release-old"
                ),
            )
        )

    rollback.assert_called_once_with(
        previous_release_id=(
            "release-old"
        ),
    )

    verify.assert_called_once_with(
        expected_release_id=(
            "release-old"
        ),
    )