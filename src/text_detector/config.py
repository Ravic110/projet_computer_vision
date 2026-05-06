"""Centralized configuration for the text detection app."""

import tkinter as tk
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


class ThemeableButton:
    """Factory for creating styled buttons with hover effects."""

    def __init__(self, master, text: str, command, bg: str, active_bg: str,
                 font: str = "Arial", font_size: int = 11, width: int = 14,
                 side: str = "top", padx: int = 0, pady: int = 4,
                 fill: str = "x", expand: bool = False):
        self.bg = bg
        self.active_bg = active_bg
        self.normal_fg = THEME.button_fg
        self.command = command

        self.btn = tk.Button(
            master,
            text=text,
            command=command,
            font=(font, font_size, "bold"),
            bg=bg,
            fg=self.normal_fg,
            activebackground=active_bg,
            activeforeground=self.normal_fg,
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=8,
            cursor="hand2",
            width=width,
        )
        self.btn.pack(side=side, fill=fill, expand=expand, padx=padx, pady=pady)

        self.btn.bind("<Enter>", self._on_enter)
        self.btn.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event=None) -> None:
        self.btn.config(bg=self.active_bg, fg=THEME.background)

    def _on_leave(self, _event=None) -> None:
        self.btn.config(bg=self.bg, fg=self.normal_fg)

    def config(self, **kwargs) -> None:
        self.btn.config(**kwargs)


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
