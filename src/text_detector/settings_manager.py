"""Settings persistence for the text detection app."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import AppSettings


class SettingsManager:
    """Manages loading and saving application settings to a JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".config" / "text-detector" / "settings.json"
        self._path = path

    def save(self, settings: AppSettings) -> None:
        """Save settings to the JSON file, replacing it atomically.

        The data is written to a temporary file next to the target and then
        renamed over it, so an interrupted write cannot leave a truncated
        file that would silently reset the user's settings on next launch.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "languages": list(settings.languages),
            "default_confidence": settings.default_confidence,
            "gpu_enabled": settings.gpu_enabled,
            "preprocess_enabled": settings.preprocess_enabled,
            "frame_skip": settings.frame_skip,
            "ocr_max_width": settings.ocr_max_width,
            "paragraph_merge": settings.paragraph_merge,
        }
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

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
                "languages",
                "default_confidence",
                "gpu_enabled",
                "preprocess_enabled",
                "frame_skip",
                "ocr_max_width",
                "paragraph_merge",
            }
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            filtered = self._migrate_languages(data, filtered)
            return AppSettings(**filtered)
        except (json.JSONDecodeError, TypeError, ValueError):
            return AppSettings()

    @staticmethod
    def _migrate_languages(data: dict, filtered: dict) -> dict:
        """Accept the pre-multi-language shape, where one code was stored.

        Files written before multi-language support carry
        "default_language": "fr"; newer ones carry "languages": ["fr", "en"].
        """
        if "languages" in filtered:
            if not isinstance(filtered["languages"], list) or not filtered["languages"]:
                del filtered["languages"]
            return filtered

        legacy = data.get("default_language")
        if isinstance(legacy, str) and legacy:
            filtered["languages"] = [legacy]
        return filtered

    def reset(self) -> None:
        """Delete the settings file to reset to defaults."""
        if self._path.exists():
            self._path.unlink()
