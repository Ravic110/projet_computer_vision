"""Settings persistence for the text detection app."""

from __future__ import annotations

import json
from pathlib import Path

from .config import AppSettings


class SettingsManager:
    """Manages loading and saving application settings to a JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".config" / "text-detector" / "settings.json"
        self._path = path

    def save(self, settings: AppSettings) -> None:
        """Save settings to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "default_language": settings.default_language,
            "default_confidence": settings.default_confidence,
            "gpu_enabled": settings.gpu_enabled,
            "preprocess_enabled": settings.preprocess_enabled,
            "frame_skip": settings.frame_skip,
            "ocr_max_width": settings.ocr_max_width,
            "paragraph_merge": settings.paragraph_merge,
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> AppSettings:
        """Load settings from JSON file, return defaults if missing or corrupted."""
        if not self._path.exists():
            return AppSettings()
        try:
            with open(self._path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return AppSettings()
            valid_fields = {
                "default_language",
                "default_confidence",
                "gpu_enabled",
                "preprocess_enabled",
                "frame_skip",
                "ocr_max_width",
                "paragraph_merge",
            }
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return AppSettings(**filtered)
        except (json.JSONDecodeError, TypeError, ValueError):
            return AppSettings()

    def reset(self) -> None:
        """Delete the settings file to reset to defaults."""
        if self._path.exists():
            self._path.unlink()
