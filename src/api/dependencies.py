import os

from fastapi import (
    HTTPException,
    Security,
)
from fastapi.security.api_key import (
    APIKeyHeader,
)
from starlette.status import (
    HTTP_403_FORBIDDEN,
)


API_KEY_NAME = "X-API-KEY"

api_key_header = APIKeyHeader(
    name=API_KEY_NAME,
    auto_error=False,
)


async def get_api_key(
    supplied_api_key: str | None = Security(
        api_key_header
    ),
) -> str:
    """
    Validate the API key supplied in the X-API-KEY header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the supplied key is missing or invalid.
    """
    configured_api_key = os.getenv(
        "API_KEY"
    )

    if (
        configured_api_key
        and supplied_api_key
        == configured_api_key
    ):
        return supplied_api_key

    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )