"""Tests for configuration module."""

import pytest

from text_detector.config import AppSettings, ThemeColors


def test_theme_colors_defaults() -> None:
    theme = ThemeColors()
    assert theme.background == "#1E1E2E"
    assert theme.accent == "#89B4FA"
    assert theme.success == "#A6E3A1"
    assert theme.danger == "#F38BA8"


def test_app_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.languages == ["en"]
    assert settings.default_confidence == 0.25
    assert settings.frame_skip == 15
    assert settings.max_history == 100
    assert settings.gpu_enabled is False
    assert settings.ocr_max_width == 800
    assert settings.paragraph_merge is False


def test_app_settings_rejects_empty_languages() -> None:
    with pytest.raises(ValueError, match="languages"):
        AppSettings(languages=[])


def test_app_settings_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="languages"):
        AppSettings(languages=["en", "zz"])


def test_app_settings_accepts_several_languages() -> None:
    settings = AppSettings(languages=["fr", "en"])
    assert settings.languages == ["fr", "en"]


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


class TestRevalidation:
    """AppSettings is mutated field by field at runtime, so the checks in
    __post_init__ must stay reachable after construction."""

    def test_validate_accepts_a_sound_mutation(self) -> None:
        settings = AppSettings()
        settings.languages = ["fr"]
        settings.frame_skip = 3
        settings.validate()

    def test_validate_rejects_a_zero_frame_skip(self) -> None:
        settings = AppSettings()
        settings.frame_skip = 0
        with pytest.raises(ValueError, match="frame_skip"):
            settings.validate()

    def test_validate_rejects_an_empty_language_list(self) -> None:
        settings = AppSettings()
        settings.languages = []
        with pytest.raises(ValueError, match="languages"):
            settings.validate()

    def test_validate_rejects_a_confidence_outside_the_slider_range(self) -> None:
        settings = AppSettings()
        settings.default_confidence = 5.0
        with pytest.raises(ValueError, match="default_confidence"):
            settings.validate()


def test_detect_max_width_defaults_below_the_recognition_width() -> None:
    # Detection costs ~95% of a pass and scales with pixel count, so it
    # runs on a smaller frame than recognition.
    settings = AppSettings()
    assert settings.detect_max_width < settings.ocr_max_width


def test_detect_max_width_must_stay_usable() -> None:
    with pytest.raises(ValueError, match="detect_max_width"):
        AppSettings(detect_max_width=32)
