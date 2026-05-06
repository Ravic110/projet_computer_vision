"""Centralized configuration for the text detection app."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThemeColors:
    """Color theme constants for the GUI."""
    background: str = "#F0F0F0"
    white: str = "#FFFFFF"
    frame_bg: str = "#FFFFFF"
    result_frame_bg: str = "#DDEEFF"
    text_output_bg: str = "#F7F7F7"
    title_bg: str = "#63BFF3"
    title_fg: str = "#FFFFFF"
    primary: str = "#4A90E2"
    primary_active: str = "#3A70B2"
    success: str = "#5AE3B1"
    success_active: str = "#4AA57C"
    danger: str = "#CA3074"
    danger_active: str = "#A02458"
    warning: str = "#FFA500"
    warning_active: str = "#CC8400"
    neutral: str = "#888888"
    neutral_active: str = "#666666"
    about: str = "#6D6DFF"
    about_active: str = "#5757D0"


@dataclass
class AppSettings:
    """Application settings with validation."""
    available_languages: list[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "it", "pt"]
    )
    default_language: str = "en"
    min_confidence: float = 0.05
    max_confidence: float = 0.8
    default_confidence: float = 0.25
    confidence_resolution: float = 0.05
    frame_skip: int = 5
    max_history: int = 100
    gpu_enabled: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.max_confidence <= 1.0):
            raise ValueError("max_confidence must be between 0.0 and 1.0")
        if not (self.min_confidence <= self.default_confidence <= self.max_confidence):
            raise ValueError("default_confidence must be between min and max confidence")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if self.max_history < 1:
            raise ValueError("max_history must be >= 1")

    def validate_language(self, lang: str) -> bool:
        """Check if a language code is available."""
        return lang in self.available_languages


THEME = ThemeColors()
SETTINGS = AppSettings()
