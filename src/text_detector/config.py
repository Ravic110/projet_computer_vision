"""Centralized configuration for the text detection app."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThemeColors:
    """Dark theme color palette for the GUI."""

    background: str = "#1E1E2E"
    surface: str = "#2A2A3C"
    surface_light: str = "#33334A"
    frame_bg: str = "#2A2A3C"
    result_frame_bg: str = "#181825"
    text_output_bg: str = "#1A1A2E"
    title_bg: str = "#1E1E2E"
    title_fg: str = "#CDD6F4"
    accent: str = "#89B4FA"
    accent_active: str = "#74C7EC"
    success: str = "#A6E3A1"
    success_active: str = "#94E2D5"
    danger: str = "#F38BA8"
    danger_active: str = "#F2CDCD"
    warning: str = "#F9E2AF"
    warning_active: str = "#FAE3B0"
    neutral: str = "#6C7086"
    neutral_active: str = "#585B70"
    about: str = "#CBA6F7"
    about_active: str = "#B4BEFE"
    button_fg: str = "#1E1E2E"
    text_fg: str = "#CDD6F4"
    text_muted: str = "#A6ADC8"
    border: str = "#45475A"
    scrollbar_bg: str = "#313244"
    scrollbar_fg: str = "#6C7086"
    status_ready: str = "#A6E3A1"
    status_busy: str = "#F9E2AF"
    status_error: str = "#F38BA8"


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
    frame_skip: int = 15
    max_history: int = 100
    gpu_enabled: bool = False
    ocr_max_width: int = 800
    paragraph_merge: bool = False

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
