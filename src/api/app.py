import os
from contextlib import (
    asynccontextmanager,
)

from fastapi import FastAPI
from fastapi_swagger_ui_theme import (
    setup_swagger_ui_theme,
)

from src.api.middleware import (
    serving_monitoring_middleware,
)
from src.api.routers.admin import (
    router as admin_router,
)
from src.api.routers.business import (
    router as business_router,
)
from src.api.routers.health import (
    router as health_router,
)
from src.api.routers.prediction import (
    router as prediction_router,
)
from src.api.serving_state import (
    clear_serving_bundle,
    get_active_serving_bundle,
    reload_serving_model,
    set_data_quality_reference_categories,
)
from src.monitoring.config import (
    get_data_quality_settings,
)
from src.monitoring.data_quality import (
    build_reference_category_cache,
    initialize_data_quality_reference_cache,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Initialize and clean up API serving resources.

    Startup loads data-quality references and the configured serving release.
    Smoke-test mode skips external model and artifact loading.
    """
    if os.getenv(
        "SMOKE_TEST"
    ) == "1":
        logger.info(
            "Smoke test mode enabled. "
            "Skipping model and data-quality loading."
        )

        try:
            yield
        finally:
            clear_serving_bundle()

        return

    reference_frame = (
        initialize_data_quality_reference_cache()
    )

    reference_categories = (
        build_reference_category_cache(
            reference_frame,
            categorical_reference_features=(
                get_data_quality_settings().get(
                    "categorical_reference_features",
                    [],
                )
            ),
        )
    )

    set_data_quality_reference_categories(
        reference_categories
    )

    try:
        reload_serving_model()

        bundle = (
            get_active_serving_bundle()
        )

        if bundle is not None:
            logger.info(
                "Model loaded: %s (version=%s)",
                bundle.model_name,
                bundle.model_version,
            )

    except Exception as error:
        logger.error(
            "Failed to load model from registry: %s",
            error,
        )
        clear_serving_bundle()

    try:
        yield

    finally:
        clear_serving_bundle()
        logger.info(
            "Shutdown: Cleaning up resources."
        )


app = FastAPI(
    title="Churn Prediction API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.middleware(
    "http"
)(
    serving_monitoring_middleware
)

app.include_router(
    health_router
)
app.include_router(
    admin_router
)
app.include_router(
    prediction_router
)
app.include_router(
    business_router
)

setup_swagger_ui_theme(
    app,
    docs_path="/docs",
    title="Churn Prediction API Docs",
)