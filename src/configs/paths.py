
from pathlib import Path, PurePosixPath

def get_project_root() -> Path:
    """Find the project root by walking upward until configs/ and src/ exist."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() and (parent / "src").exists():
            return parent
    return current.parents[2]


def join_uri(base: str, *parts: str) -> str:
    """
    Join local or GCS paths without breaking gs:// URIs.
    """
    base = str(base).rstrip("/")
    suffix = "/".join(str(part).strip("/") for part in parts)

    if not suffix:
        return base

    return f"{base}/{suffix}"


def path_name(path: str) -> str:
    """
    Return file name for local or GCS paths.
    """
    return PurePosixPath(str(path)).name


def path_suffix(path: str) -> str:
    """
    Return suffix for local or GCS paths.
    """
    return PurePosixPath(str(path)).suffix.lower()


