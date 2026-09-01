import traceback

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from src.api.dependencies import (
    get_api_key,
)
from src.api.schema import (
    ServingRollbackRequest,
)
from src.api.serving_state import (
    CFG,
    MODEL_NAME,
    MODELS_PATH,
    activate_serving_bundle,
    reload_serving_model,
    require_active_serving_bundle,
)
from src.inference.model_manager import (
    load_serving_bundle_for_release,
)
from src.inference.releases.repository import (
    activate_release_pointer,
    load_active_release_id,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["administration"],
    dependencies=[
        Depends(get_api_key),
    ],
)


@router.post("/reload-model")
def reload_model() -> dict:
    """
    Reload and atomically activate the configured serving release.

    The previous bundle remains active if loading or validation fails.
    """
    try:
        result = reload_serving_model()

    except Exception as error:
        logger.error(
            "Model reload failed: %s",
            traceback.format_exc(),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Model reload failed: "
                f"{error}"
            ),
        ) from error

    return {
        "status": "reloaded",
        **result,
    }


@router.post("/rollback-serving-release")
def rollback_serving_release(
    payload: ServingRollbackRequest,
) -> dict:
    """
    Validate and atomically activate a previously published serving release.

    If activation fails after updating the release pointer, the previous
    pointer is restored before the error is returned.
    """
    previous_bundle = (
        require_active_serving_bundle()
    )
    previous_release_id = (
        previous_bundle.release_id
    )

    stored_release_id = (
        load_active_release_id(
            models_path=MODELS_PATH,
        )
    )

    if (
        stored_release_id
        != previous_release_id
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "In-memory serving release does "
                "not match the active pointer."
            ),
        )

    if (
        payload.release_id
        == previous_release_id
    ):
        return {
            "status": "unchanged",
            "release_id": (
                previous_release_id
            ),
            "previous_release_id": (
                previous_release_id
            ),
        }

    pointer_changed = False

    try:
        candidate_bundle = (
            load_serving_bundle_for_release(
                release_id=(
                    payload.release_id
                ),
                model_name=MODEL_NAME,
                cfg=CFG,
                models_path=MODELS_PATH,
            )
        )

        activate_release_pointer(
            models_path=MODELS_PATH,
            release_id=payload.release_id,
            operation="rollback",
            previous_release_id=(
                previous_release_id
            ),
        )
        pointer_changed = True

        result = activate_serving_bundle(
            candidate_bundle
        )

    except Exception as error:
        if pointer_changed:
            try:
                activate_release_pointer(
                    models_path=MODELS_PATH,
                    release_id=(
                        previous_release_id
                    ),
                    operation=(
                        "rollback_reverted"
                    ),
                    previous_release_id=(
                        payload.release_id
                    ),
                )

            except Exception:
                logger.exception(
                    "CRITICAL: rollback pointer "
                    "could not be restored | "
                    "expected_release_id=%s",
                    previous_release_id,
                )

        logger.exception(
            "Serving release rollback failed | "
            "target_release_id=%s | "
            "previous_release_id=%s",
            payload.release_id,
            previous_release_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Serving release rollback failed. "
                f"Reason: {error}"
            ),
        ) from error

    logger.warning(
        "Serving release rollback completed | "
        "previous_release_id=%s | "
        "active_release_id=%s",
        previous_release_id,
        candidate_bundle.release_id,
    )

    return {
        "status": "rolled_back",
        "previous_release_id": (
            previous_release_id
        ),
        **result,
    }