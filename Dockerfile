# Use a specific slim Python image for a small footprint
FROM python:3.12-slim

# 1. Install uv (Modern Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Environment Variables
# PYTHONDONTWRITEBYTECODE: Prevents .pyc files in volumes
# UV_LINK_MODE: 'copy' avoids issues with hardlinks across different filesystems
# UV_PROJECT_ENVIRONMENT: Fixed location for the virtualenv
# PYTHONPYCACHEPREFIX: Redirects all __pycache__ to /tmp to avoid volume permission errors
# UV_PYTHON_INSTALL_DIR: Force uv to install Python inside /app instead of /home
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/.uv_cache \
    UV_PYTHON_INSTALL_DIR=/app/.python \
    PYTHONPYCACHEPREFIX=/tmp/pycache

WORKDIR /app

# 3. Install system dependencies (as root)
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Security: Create a non-root user
# We create a home directory (-m) and grant broad permissions to /app 
# to prevent UID/GID conflicts with host volumes during development
RUN groupadd -r mlgroup && useradd -m -r -g mlgroup mluser \
    && chown -R mluser:mlgroup /app \
    && chmod 777 /app

# 5. Switch to non-root user
USER mluser

# 6. Install project dependencies
# Copy only metadata first to leverage Docker layer caching
COPY --chown=mluser:mlgroup pyproject.toml uv.lock ./

# IMPORTANT: --no-install-project prevents uv from trying to build your 
# local code as a package during the build phase, avoiding 'egg-info' issues.
RUN uv sync --frozen --no-cache --no-install-project

# 7. Copy application code
# These will be shadowed by your volumes in docker-compose during development
COPY --chown=mluser:mlgroup src/ /app/src/
COPY --chown=mluser:mlgroup configs/ /app/configs/
COPY --chown=mluser:mlgroup flows/ /app/flows/

# 8. Final touches
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/livez || exit 1
  
# --no-sync is CRITICAL: It tells uv NOT to re-verify or re-install the project 
# at startup, which stops it from trying to write to the protected .venv 
# with your host user permissions.
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]