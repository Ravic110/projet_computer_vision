"""Path utilities using pathlib."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_assets_dir() -> Path:
    """Return the assets directory."""
    return get_project_root() / "text_detector" / "assets"


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_safe_path(base: Path, filename: str, extension: str) -> Path:
    """Return a safe file path with the given extension."""
    if not filename.endswith(extension):
        filename = f"{filename}{extension}"
    return base / filename
