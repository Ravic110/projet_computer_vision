"""Tests for widgets module."""

import tkinter as tk

import pytest

from text_detector.config import THEME
from text_detector.widgets import StatusLED, ThemeableButton


@pytest.fixture
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


class TestStatusLED:
    def test_init(self, root):
        led = StatusLED(root, color=THEME.status_ready, size=10)
        assert led.color == THEME.status_ready
        assert led.size == 10

    def test_set_color(self, root):
        led = StatusLED(root, size=10)
        led.set_color(THEME.status_error)
        assert led.color == THEME.status_error

    def test_pack(self, root):
        led = StatusLED(root, size=10)
        led.pack()
        root.update()
        assert led.canvas.winfo_width() > 0


class TestThemeableButton:
    def test_init(self, root):
        clicked = []
        btn = ThemeableButton(
            root, "Click me", lambda: clicked.append(True),
            bg=THEME.accent, active_bg=THEME.accent_active,
        )
        assert btn.btn.cget("text") == "Click me"

    def test_click(self, root):
        clicked = []
        btn = ThemeableButton(
            root, "Click", lambda: clicked.append(True),
            bg=THEME.accent, active_bg=THEME.accent_active,
        )
        btn.btn.invoke()
        assert clicked == [True]

    def test_hover_colors(self, root):
        btn = ThemeableButton(
            root, "Hover", lambda: None,
            bg=THEME.accent, active_bg=THEME.accent_active,
        )
        btn._on_enter()
        assert btn.btn.cget("bg") == THEME.accent_active
        btn._on_leave()
        assert btn.btn.cget("bg") == THEME.accent
