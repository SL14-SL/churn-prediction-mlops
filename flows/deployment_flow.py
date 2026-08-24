from flows.tasks.serving_tasks import (
    task_refresh_api,
    task_rollback_serving_release,
    task_verify_serving_release,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def deploy_and_verify_release(
    *,
    release_id: str,
    previous_release_id: str | None,
) -> dict:
    """
    Reload and verify a newly published release.

    If verification fails, restore and verify the previous release.
    """
    try:
        task_refresh_api()

        verification = (
            task_verify_serving_release(
                expected_release_id=(
                    release_id
                ),
            )
        )

        return {
            "deployment_status": (
                "verified"
            ),
            "release_id": release_id,
            "verification": verification,
            "rolled_back": False,
        }

    except Exception as deployment_error:
        logger.exception(
            "Serving release deployment "
            "failed | release_id=%s | "
            "previous_release_id=%s",
            release_id,
            previous_release_id,
        )

        if previous_release_id is None:
            raise RuntimeError(
                "Serving release verification "
                "failed and no previous release "
                "is available for rollback."
            ) from deployment_error

        try:
            rollback_result = (
                task_rollback_serving_release(
                    previous_release_id=(
                        previous_release_id
                    ),
                )
            )

            rollback_verification = (
                task_verify_serving_release(
                    expected_release_id=(
                        previous_release_id
                    ),
                )
            )

        except Exception as rollback_error:
            raise RuntimeError(
                "Serving release verification "
                "failed and automatic rollback "
                "also failed."
            ) from rollback_error

        raise RuntimeError(
            "New serving release failed "
            "verification. Automatic rollback "
            "completed successfully | "
            f"failed_release_id={release_id} | "
            "restored_release_id="
            f"{previous_release_id} | "
            f"rollback_result={rollback_result} | "
            "rollback_verification="
            f"{rollback_verification}"
        ) from deployment_error