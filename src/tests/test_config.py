"""Tests for configuration module."""

import pytest

from text_detector.config import AppSettings, ThemeColors


def test_theme_colors_defaults() -> None:
    theme = ThemeColors()
    assert theme.background == "#F0F0F0"
    assert theme.primary == "#4A90E2"
    assert theme.success == "#5AE3B1"
    assert theme.danger == "#CA3074"


def test_app_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.default_language == "en"
    assert settings.default_confidence == 0.25
    assert settings.frame_skip == 15
    assert settings.max_history == 100
    assert settings.gpu_enabled is False
    assert settings.ocr_max_width == 800
    assert settings.paragraph_merge is False


def test_app_settings_validate_language() -> None:
    settings = AppSettings()
    assert settings.validate_language("en") is True
    assert settings.validate_language("fr") is True
    assert settings.validate_language("zz") is False


def test_app_settings_invalid_min_confidence() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        AppSettings(min_confidence=-0.1)


def test_app_settings_invalid_max_confidence() -> None:
    with pytest.raises(ValueError, match="max_confidence"):
        AppSettings(max_confidence=1.5)


def test_app_settings_default_out_of_range() -> None:
    with pytest.raises(ValueError, match="default_confidence"):
        AppSettings(min_confidence=0.5, max_confidence=0.8, default_confidence=0.1)


def test_app_settings_invalid_frame_skip() -> None:
    with pytest.raises(ValueError, match="frame_skip"):
        AppSettings(frame_skip=0)


def test_app_settings_invalid_max_history() -> None:
    with pytest.raises(ValueError, match="max_history"):
        AppSettings(max_history=0)
