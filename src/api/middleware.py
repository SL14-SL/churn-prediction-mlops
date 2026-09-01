import time
from collections.abc import Awaitable, Callable

from fastapi import (
    Request,
    Response,
)

from src.monitoring.config import (
    get_serving_settings,
)
from src.monitoring.serving import (
    normalize_path,
    observe_request,
    should_ignore_path,
)


SERVING_CFG = get_serving_settings()


async def serving_monitoring_middleware(
    request: Request,
    call_next: Callable[
        [Request],
        Awaitable[Response],
    ],
) -> Response:
    """
    Record latency, status and exception metrics for serving requests.

    Monitoring and documentation endpoints are excluded according to serving
    configuration to prevent self-observation from distorting API metrics.
    """
    if not SERVING_CFG.get(
        "enabled",
        True,
    ):
        return await call_next(
            request
        )

    raw_path = request.url.path

    if should_ignore_path(
        raw_path,
        SERVING_CFG.get(
            "ignored_paths"
        ),
    ):
        return await call_next(
            request
        )

    method = request.method
    path = normalize_path(
        raw_path,
        SERVING_CFG.get(
            "track_paths"
        ),
    )
    started_at = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(
            request
        )
        status_code = (
            response.status_code
        )
        return response

    finally:
        observe_request(
            method=method,
            path=path,
            status_code=status_code,
            latency_seconds=(
                time.perf_counter()
                - started_at
            ),
        )