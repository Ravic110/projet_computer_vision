"""Tests for settings_manager module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from text_detector.config import AppSettings
from text_detector.settings_manager import SettingsManager


class TestSettingsManager:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            manager = SettingsManager(path)
            settings = AppSettings(
                languages=["fr"],
                default_confidence=0.5,
                gpu_enabled=True,
                frame_skip=10,
                ocr_max_width=1024,
                paragraph_merge=True,
            )
            manager.save(settings)
            loaded = manager.load()
            assert loaded.languages == ["fr"]
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
            assert settings.languages == ["en"]
            assert settings.default_confidence == 0.25

    def test_load_corrupted_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("not valid json")
            manager = SettingsManager(path)
            settings = manager.load()
            assert settings.languages == ["en"]

    def test_load_invalid_type_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("42")
            manager = SettingsManager(path)
            settings = manager.load()
            assert settings.languages == ["en"]

    def test_load_filters_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            data = {"languages": ["de"], "unknown_key": "value"}
            path.write_text(json.dumps(data))
            manager = SettingsManager(path)
            settings = manager.load()
            assert settings.languages == ["de"]

    def test_reset_deletes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            manager = SettingsManager(path)
            settings = AppSettings(languages=["fr"])
            manager.save(settings)
            assert path.exists()
            manager.reset()
            assert not path.exists()
            defaults = manager.load()
            assert defaults.languages == ["en"]

    def test_default_path_creation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        manager = SettingsManager()
        assert manager._path == tmp_path / ".config" / "text-detector" / "settings.json"
        manager.save(AppSettings(languages=["es"]))
        assert manager._path.exists()

    def test_failed_write_leaves_previous_settings_intact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            manager = SettingsManager(path)
            manager.save(AppSettings(languages=["fr"]))

            with (
                patch("json.dump", side_effect=OSError("disk full")),
                pytest.raises(OSError),
            ):
                manager.save(AppSettings(languages=["de"]))

            assert manager.load().languages == ["fr"]

    def test_failed_write_leaves_no_temp_file_behind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            manager = SettingsManager(path)
            manager.save(AppSettings(languages=["fr"]))

            with (
                patch("json.dump", side_effect=OSError("disk full")),
                pytest.raises(OSError),
            ):
                manager.save(AppSettings(languages=["de"]))

            assert sorted(p.name for p in Path(tmpdir).iterdir()) == ["settings.json"]

    def test_languages_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            manager = SettingsManager(path)
            manager.save(AppSettings(languages=["fr", "en"]))
            assert manager.load().languages == ["fr", "en"]

    def test_load_migrates_legacy_default_language(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(json.dumps({"default_language": "de", "frame_skip": 20}))
            settings = SettingsManager(path).load()
            assert settings.languages == ["de"]
            assert settings.frame_skip == 20

    def test_languages_wins_over_legacy_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(json.dumps({"default_language": "de", "languages": ["fr"]}))
            assert SettingsManager(path).load().languages == ["fr"]

    def test_load_falls_back_on_unusable_languages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(json.dumps({"languages": ["zz"]}))
            assert SettingsManager(path).load().languages == ["en"]

    def test_load_falls_back_on_empty_languages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(json.dumps({"languages": []}))
            assert SettingsManager(path).load().languages == ["en"]
