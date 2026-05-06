"""Tests for TextRecognitionApp UI improvements."""

import tkinter as tk
from unittest.mock import patch

import pytest

from text_detector.text_detector import TextRecognitionApp


@pytest.fixture
def app():
    """Create a TextRecognitionApp instance with mocked root."""
    root = tk.Tk()
    root.withdraw()
    app = TextRecognitionApp(root)
    yield app
    root.destroy()


class TestCopyToClipboard:
    def test_copy_to_clipboard_with_text(self, app):
        app.detected_text = [
            ((0, 0, 100, 50), "Hello", 0.95),
            ((0, 60, 100, 110), "World", 0.88),
        ]
        app._copy_to_clipboard()
        clipboard_content = app.root.clipboard_get()
        assert clipboard_content == "Hello\nWorld"
        assert "Copied to clipboard" in app.status_label.cget("text")

    def test_copy_to_clipboard_no_text(self, app):
        app.detected_text = []
        app._copy_to_clipboard()
        assert "No text to copy" in app.status_label.cget("text")


class TestKeyboardShortcuts:
    def test_bind_keyboard_shortcuts_calls_bind(self, app):
        with patch.object(app.root, 'bind') as mock_bind:
            app._bind_keyboard_shortcuts()
            assert mock_bind.call_count == 6

    def test_toggle_capture_starts_when_stopped(self, app):
        app.capture_active = False
        with patch.object(app, 'start_capture') as mock_start:
            app._toggle_capture()
            mock_start.assert_called_once()

    def test_toggle_capture_stops_when_running(self, app):
        app.capture_active = True
        with patch.object(app, 'stop_capture') as mock_stop:
            app._toggle_capture()
            mock_stop.assert_called_once()


class TestToggleCapture:
    def test_toggle_capture_starts_when_stopped(self, app):
        app.capture_active = False
        with patch.object(app, 'start_capture') as mock_start:
            app._toggle_capture()
            mock_start.assert_called_once()

    def test_toggle_capture_stops_when_running(self, app):
        app.capture_active = True
        with patch.object(app, 'stop_capture') as mock_stop:
            app._toggle_capture()
            mock_stop.assert_called_once()
