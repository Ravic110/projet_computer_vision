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
            assert mock_bind.call_count == 7

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


class TestFrameSkipSlider:
    def test_frame_skip_changed_updates_label(self, app):
        app.frame_skip_var.set(5)
        app._frame_skip_changed("5")
        assert app.frame_skip_label.cget("text") == "Every 5 frames"

    def test_frame_skip_changed_updates_setting(self, app):
        app.frame_skip_var.set(10)
        app._frame_skip_changed("10")
        from text_detector.config import SETTINGS
        assert SETTINGS.frame_skip == 10

    def test_frame_skip_var_initialized_from_settings(self, app):
        from text_detector.config import SETTINGS
        assert app.frame_skip_var.get() == SETTINGS.frame_skip

    def test_frame_skip_scale_has_correct_range(self, app):
        assert app.frame_skip_scale.cget("from") == 1
        assert app.frame_skip_scale.cget("to") == 60


class TestPasteImageFromClipboard:
    def test_paste_image_creates_context_menu(self, app):
        assert hasattr(app, 'image_context_menu')
        assert hasattr(app, '_paste_image_from_clipboard')

    def test_paste_image_no_image_in_clipboard(self, app):
        with (
            patch('PIL.ImageGrab.grabclipboard', return_value=None),
            patch('tkinter.messagebox.showinfo') as mock_msg,
        ):
            app._paste_image_from_clipboard()
            mock_msg.assert_called_once()

    def test_paste_image_with_valid_image(self, app):
        from PIL import Image
        mock_image = Image.new('RGB', (100, 100), color='red')
        with (
            patch('PIL.ImageGrab.grabclipboard', return_value=mock_image),
            patch.object(app, '_process_current_frame') as mock_process,
            patch.object(app, 'stop_capture') as mock_stop,
        ):
            app._paste_image_from_clipboard()
            mock_stop.assert_called_once()
            mock_process.assert_called_once()
            assert app.current_frame is not None

    def test_paste_image_invalid_content(self, app):
        with (
            patch('PIL.ImageGrab.grabclipboard', return_value="not an image"),
            patch('tkinter.messagebox.showerror') as mock_msg,
        ):
            app._paste_image_from_clipboard()
            mock_msg.assert_called_once()

    def test_ctrl_v_bound_in_shortcuts(self, app):
        bindings = []
        def mock_bind(sequence, func):
            bindings.append(sequence)
        with patch.object(app.root, 'bind', mock_bind):
            app._bind_keyboard_shortcuts()
            assert "<Control-v>" in bindings


class TestHistoryPanel:
    def test_add_to_history_includes_timestamp(self, app):
        app.detected_text = [((0, 0, 100, 50), "Test", 0.9)]
        app._add_to_history()
        assert len(app.history) == 1
        assert "timestamp" in app.history[0]
        assert "text" in app.history[0]
        assert "confidence" in app.history[0]

    def test_refresh_history_display_populates_listbox(self, app):
        app.history = [
            {"text": "Hello", "confidence": 0.95, "timestamp": "10:00:00"},
            {"text": "World", "confidence": 0.88, "timestamp": "10:00:01"},
        ]
        app._refresh_history_display()
        assert app.history_listbox.size() == 2

    def test_filter_history(self, app):
        app.history = [
            {"text": "Hello World", "confidence": 0.95, "timestamp": "10:00:00"},
            {"text": "Goodbye", "confidence": 0.88, "timestamp": "10:00:01"},
        ]
        app.history_search.delete(0, tk.END)
        app.history_search.insert(0, "Hello")
        app._filter_history()
        assert app.history_listbox.size() == 1

    def test_clear_history(self, app):
        app.history = [{"text": "Test", "confidence": 0.9, "timestamp": "10:00:00"}]
        app._clear_history()
        assert len(app.history) == 0
        assert app.history_listbox.size() == 0

    def test_copy_history_item(self, app):
        app.history = [{"text": "Secret", "confidence": 0.9, "timestamp": "10:00:00"}]
        app._refresh_history_display()
        app.history_listbox.selection_set(0)
        app._copy_history_item()
        clipboard_content = app.root.clipboard_get()
        assert clipboard_content == "Secret"


class TestROISelection:
    def test_toggle_roi_mode_activates(self, app):
        assert not app.roi_mode
        app._toggle_roi_mode()
        assert app.roi_mode
        app._toggle_roi_mode()
        assert not app.roi_mode

    def test_clear_roi(self, app):
        app.roi = (10, 10, 100, 100)
        app._clear_roi()
        assert app.roi is None

    def test_get_cropped_frame_without_roi(self, app):
        import numpy as np
        app.current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        app.roi = None
        result = app._get_cropped_frame()
        assert result.shape == (100, 100, 3)

    def test_get_cropped_frame_with_roi(self, app):
        import numpy as np
        app.current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        app.roi = (10, 20, 50, 60)
        result = app._get_cropped_frame()
        assert result.shape == (40, 40, 3)

    def test_get_cropped_frame_no_current_frame(self, app):
        app.current_frame = None
        result = app._get_cropped_frame()
        assert result is None
