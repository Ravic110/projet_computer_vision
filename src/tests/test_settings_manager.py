"""Tests for settings_manager module."""

import json
import tempfile
from pathlib import Path

from text_detector.config import AppSettings
from text_detector.settings_manager import SettingsManager


class TestSettingsManager:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            manager = SettingsManager(path)
            settings = AppSettings(
                default_language="fr",
                default_confidence=0.5,
                gpu_enabled=True,
                frame_skip=10,
                ocr_max_width=1024,
                paragraph_merge=True,
            )
            manager.save(settings)
            loaded = manager.load()
            assert loaded.default_language == "fr"
            assert loaded.default_confidence == 0.5
            assert loaded.gpu_enabled is True
            assert loaded.frame_skip == 10
            assert loaded.ocr_max_width == 1024
            assert loaded.paragraph_merge is True

    def test_load_missing_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            manager = SettingsManager(path)
            settings = manager.load()
            assert settings.default_language == "en"
            assert settings.default_confidence == 0.25

    def test_load_corrupted_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("not valid json")
            manager = SettingsManager(path)
            settings = manager.load()
            assert settings.default_language == "en"

    def test_load_invalid_type_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("42")
            manager = SettingsManager(path)
            settings = manager.load()
            assert settings.default_language == "en"

    def test_load_filters_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            data = {"default_language": "de", "unknown_key": "value"}
            path.write_text(json.dumps(data))
            manager = SettingsManager(path)
            settings = manager.load()
            assert settings.default_language == "de"

    def test_reset_deletes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            manager = SettingsManager(path)
            settings = AppSettings(default_language="fr")
            manager.save(settings)
            assert path.exists()
            manager.reset()
            assert not path.exists()
            defaults = manager.load()
            assert defaults.default_language == "en"

    def test_default_path_creation(self):
        manager = SettingsManager()
        settings = AppSettings(default_language="es")
        manager.save(settings)
        assert manager._path.exists()
        manager.reset()
