"""Path utilities using pathlib."""

import os
from pathlib import Path

APP_DIR_NAME = "text-detector"
LOG_FILE_NAME = "text_detector.log"


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_assets_dir() -> Path:
    """Return the packaged assets directory.

    The assets live inside the package so they survive an install; a path
    relative to the repository root only resolves in a source checkout.
    """
    return Path(__file__).parent.parent / "assets"


def get_state_dir() -> Path:
    """Return the directory for the app's own state files.

    Follows XDG_STATE_HOME so the log has one predictable home, instead of
    landing in whatever directory the app happened to be launched from.
    """
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / APP_DIR_NAME


def get_log_path() -> Path:
    """Return the full path of the application log file."""
    return get_state_dir() / LOG_FILE_NAME
