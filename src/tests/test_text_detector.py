"""Tests for TextRecognitionApp UI improvements."""

import contextlib
import tkinter as tk
from unittest.mock import patch

import numpy as np
import pytest

from text_detector.layout import COMPACT, LARGE, MEDIUM, MIN_WINDOW
from text_detector.ocr_engine import OCREngine
from text_detector.text_detector import TextRecognitionApp


@pytest.fixture
def app():
    """Create a TextRecognitionApp instance with a hidden root window."""
    root = tk.Tk()
    root.withdraw()
    # Never load a real OCR model in tests: it costs seconds and a download.
    with patch.object(OCREngine, "preload"):
        app = TextRecognitionApp(root)
    yield app
    # Without this the OCR worker and capture threads outlive every test,
    # leaving hundreds of them alive by the end of the suite.
    app.stop_capture()
    app.engine.shutdown()
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
        with patch.object(app.root, "bind") as mock_bind:
            app._bind_keyboard_shortcuts()
            assert mock_bind.call_count == 8

    def test_toggle_capture_starts_when_stopped(self, app):
        app.capture_active = False
        with patch.object(app, "start_capture") as mock_start:
            app._toggle_capture()
            mock_start.assert_called_once()

    def test_toggle_capture_stops_when_running(self, app):
        app.capture_active = True
        with patch.object(app, "stop_capture") as mock_stop:
            app._toggle_capture()
            mock_stop.assert_called_once()


class TestToggleCapture:
    def test_toggle_capture_starts_when_stopped(self, app):
        app.capture_active = False
        with patch.object(app, "start_capture") as mock_start:
            app._toggle_capture()
            mock_start.assert_called_once()

    def test_toggle_capture_stops_when_running(self, app):
        app.capture_active = True
        with patch.object(app, "stop_capture") as mock_stop:
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
        assert hasattr(app, "image_context_menu")
        assert hasattr(app, "_paste_image_from_clipboard")

    def test_paste_image_no_image_in_clipboard(self, app):
        with (
            patch("PIL.ImageGrab.grabclipboard", return_value=None),
            patch("tkinter.messagebox.showinfo") as mock_msg,
        ):
            app._paste_image_from_clipboard()
            mock_msg.assert_called_once()

    def test_paste_image_with_valid_image(self, app):
        from PIL import Image

        mock_image = Image.new("RGB", (100, 100), color="red")
        with (
            patch("PIL.ImageGrab.grabclipboard", return_value=mock_image),
            patch.object(app, "_process_current_frame") as mock_process,
            patch.object(app, "stop_capture") as mock_stop,
        ):
            app._paste_image_from_clipboard()
            mock_stop.assert_called_once()
            mock_process.assert_called_once()
            assert app.current_frame is not None

    def test_paste_image_invalid_content(self, app):
        with (
            patch("PIL.ImageGrab.grabclipboard", return_value="not an image"),
            patch("tkinter.messagebox.showerror") as mock_msg,
        ):
            app._paste_image_from_clipboard()
            mock_msg.assert_called_once()

    def test_ctrl_v_bound_in_shortcuts(self, app):
        bindings = []

        def mock_bind(sequence, func):
            bindings.append(sequence)

        with patch.object(app.root, "bind", mock_bind):
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


class TestHistoryFiltering:
    def _seed(self, app):
        app.history = [
            {"text": "Alpha", "confidence": 0.9, "timestamp": "10:00:00"},
            {"text": "Bravo", "confidence": 0.8, "timestamp": "10:00:01"},
            {"text": "Charlie", "confidence": 0.7, "timestamp": "10:00:02"},
        ]

    def _search(self, app, query):
        app.history_search.delete(0, tk.END)
        app.history_search.insert(0, query)
        app._filter_history()

    def test_placeholder_is_not_treated_as_a_query(self, app):
        self._seed(app)
        app._refresh_history_display()
        assert app.history_listbox.size() == 3

    def test_refresh_preserves_active_filter(self, app):
        self._seed(app)
        self._search(app, "Charlie")
        assert app.history_listbox.size() == 1

        app.history.append({"text": "Delta", "confidence": 0.6, "timestamp": "10:00:03"})
        app._refresh_history_display()
        assert app.history_listbox.size() == 1
        assert "Charlie" in app.history_listbox.get(0)

    def test_copy_history_item_uses_filtered_index(self, app):
        self._seed(app)
        self._search(app, "Charlie")
        app.history_listbox.selection_set(0)
        app._copy_history_item()
        assert app.root.clipboard_get() == "Charlie"

    def test_copy_history_item_unfiltered_still_correct(self, app):
        self._seed(app)
        app._refresh_history_display()
        app.history_listbox.selection_set(1)
        app._copy_history_item()
        assert app.root.clipboard_get() == "Bravo"

    def test_add_to_history_keeps_filter(self, app):
        self._seed(app)
        self._search(app, "Alpha")
        app.detected_text = [((0, 0, 10, 10), "Zulu", 0.9)]
        app._add_to_history()
        assert app.history_listbox.size() == 1
        assert "Alpha" in app.history_listbox.get(0)

    def test_clear_history_empties_view(self, app):
        self._seed(app)
        self._search(app, "Alpha")
        app._clear_history()
        assert app.history_listbox.size() == 0
        app.history_listbox.selection_clear(0, tk.END)
        app._copy_history_item()


class TestShortcutGuard:
    def test_shortcut_skipped_when_entry_focused(self, app):
        called = []
        handler = app._shortcut(lambda: called.append(True))
        with patch.object(app.root, "focus_get", return_value=app.history_search):
            assert handler() is None
        assert called == []

    def test_shortcut_runs_when_entry_not_focused(self, app):
        called = []
        handler = app._shortcut(lambda: called.append(True))
        with patch.object(app.root, "focus_get", return_value=app.image_label):
            assert handler() == "break"
        assert called == [True]

    def test_shortcut_runs_when_focus_is_unknown(self, app):
        called = []
        handler = app._shortcut(lambda: called.append(True))
        with patch.object(app.root, "focus_get", return_value=None):
            handler()
        assert called == [True]

    def test_every_shortcut_is_guarded(self, app):
        bound = {}
        with patch.object(app.root, "bind", lambda seq, fn: bound.__setitem__(seq, fn)):
            app._bind_keyboard_shortcuts()
        assert len(bound) == 8

        targets = [
            "load_image",
            "save_results",
            "_copy_to_clipboard",
            "_paste_image_from_clipboard",
            "_toggle_capture",
            "_reset_settings",
            "clear_results",
            "_toggle_picker_mode",
        ]
        with (
            patch.object(app.root, "focus_get", return_value=app.history_search),
            contextlib.ExitStack() as stack,
        ):
            mocks = [stack.enter_context(patch.object(app, name)) for name in targets]
            for handler in bound.values():
                handler()
            assert all(not m.called for m in mocks)

        with (
            patch.object(app.root, "focus_get", return_value=app.image_label),
            contextlib.ExitStack() as stack,
        ):
            mocks = [stack.enter_context(patch.object(app, name)) for name in targets]
            for handler in bound.values():
                handler()
            assert all(m.called for m in mocks)


class TestDisplayGeometry:
    def test_show_image_records_geometry(self, app):
        app._show_image(np.zeros((50, 80, 3), dtype=np.uint8))
        assert app._display_geometry is not None
        assert (app._display_geometry.width, app._display_geometry.height) == (80, 50)

    def test_show_image_downscales_to_fit_widget(self, app):
        with (
            patch.object(app.image_label, "winfo_width", return_value=400),
            patch.object(app.image_label, "winfo_height", return_value=400),
        ):
            app._show_image(np.zeros((800, 800, 3), dtype=np.uint8))
        assert app._display_geometry.scale == 0.5
        assert app._display_geometry.width == 400


class TestROIMapping:
    def _prepare(self, app, frame_w=100, frame_h=100, widget=400):
        from text_detector.image_processor import compute_display_geometry

        app.current_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        app._display_geometry = compute_display_geometry(frame_w, frame_h, widget, widget)
        return app._display_geometry

    def _event(self, x, y):
        event = tk.Event()
        event.x = x
        event.y = y
        return event

    def test_release_maps_widget_coords_to_frame_coords(self, app):
        geo = self._prepare(app)
        assert (geo.offset_x, geo.offset_y) == (150, 150)
        app._roi_start = (170, 170)
        app._on_roi_release(self._event(230, 240))
        assert app.roi == (20, 20, 80, 90)

    def test_release_clamps_drag_beyond_image(self, app):
        self._prepare(app)
        app._roi_start = (0, 0)
        app._on_roi_release(self._event(399, 399))
        assert app.roi == (0, 0, 100, 100)

    def test_release_on_padding_only_keeps_roi_unset(self, app):
        self._prepare(app)
        app._roi_start = (0, 0)
        app._on_roi_release(self._event(60, 60))
        assert app.roi is None

    def test_release_accounts_for_downscaled_display(self, app):
        self._prepare(app, frame_w=800, frame_h=800, widget=400)
        app._roi_start = (100, 100)
        app._on_roi_release(self._event(300, 300))
        assert app.roi == (200, 200, 600, 600)

    def test_cropped_frame_matches_selected_region(self, app):
        self._prepare(app)
        app.current_frame[20:90, 20:80] = 255
        app._roi_start = (170, 170)
        app._on_roi_release(self._event(230, 240))
        crop = app._get_cropped_frame()
        assert crop.shape == (70, 60, 3)
        assert np.all(crop == 255)

    def test_release_without_frame_still_exits_roi_mode(self, app):
        app.current_frame = None
        app.roi_mode = False
        app._toggle_roi_mode()
        app._roi_start = (10, 10)
        app._on_roi_release(self._event(50, 50))
        assert app._roi_start is None
        assert not app.roi_mode

    def test_release_exits_roi_mode_after_selection(self, app):
        self._prepare(app)
        app.roi_mode = False
        app._toggle_roi_mode()
        app._roi_start = (170, 170)
        app._on_roi_release(self._event(230, 240))
        assert not app.roi_mode

    def test_toggle_roi_mode_button_colour_survives_hover(self, app):
        from text_detector.config import THEME

        app._toggle_roi_mode()
        app.roi_btn._on_enter()
        app.roi_btn._on_leave()
        assert app.roi_btn.btn.cget("bg") == THEME.success
        app._toggle_roi_mode()
        app.roi_btn._on_leave()
        assert app.roi_btn.btn.cget("bg") == THEME.accent

    def test_clear_results_resets_roi(self, app):
        self._prepare(app)
        app.roi = (10, 10, 50, 50)
        app.clear_results()
        assert app.roi is None

    def test_clear_results_exits_roi_mode(self, app):
        app._toggle_roi_mode()
        app.clear_results()
        assert not app.roi_mode
        assert app._roi_start is None

    def test_roi_drag_draws_preview_without_changing_roi(self, app):
        self._prepare(app)
        app._roi_start = (170, 170)
        app._on_roi_drag(self._event(230, 240))
        assert app.roi is None
        assert app.image_label.image is not None


class TestDetectionOffsets:
    def test_detections_are_translated_back_to_full_frame(self, app):
        from text_detector.ocr_engine import DetectionResult

        app.current_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        app.roi = (50, 60, 150, 160)
        app._roi_at_submit = (50, 60, 150, 160)
        bbox = [[10.0, 10.0], [40.0, 10.0], [40.0, 30.0], [10.0, 30.0]]
        app.ocr_result = DetectionResult(detections=[(bbox, "Hi", 0.9)], languages=["en"])
        app._apply_ocr_result()
        assert app.detected_text[0][0][0] == [60.0, 70.0]
        assert app.detected_text[0][0][2] == [90.0, 90.0]

    def test_detections_untouched_without_roi(self, app):
        from text_detector.ocr_engine import DetectionResult

        app.current_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        app.roi = None
        app._roi_at_submit = None
        bbox = [[10.0, 10.0], [40.0, 10.0], [40.0, 30.0], [10.0, 30.0]]
        app.ocr_result = DetectionResult(detections=[(bbox, "Hi", 0.9)], languages=["en"])
        app._apply_ocr_result()
        assert app.detected_text[0][0][0] == [10.0, 10.0]

    def test_process_current_frame_snapshots_roi(self, app):
        app.current_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        app.roi = (10, 20, 110, 120)
        with patch.object(app.engine, "detect_text_async", return_value=True):
            app._process_current_frame()
        assert app._roi_at_submit == (10, 20, 110, 120)

    def test_shortcut_uses_event_widget_over_focus(self, app):
        called = []
        handler = app._shortcut(lambda: called.append(True))
        event = tk.Event()
        event.widget = app.history_search
        with patch.object(app.root, "focus_get", return_value=app.image_label):
            assert handler(event) is None
        assert called == []

    def test_shortcut_runs_when_event_widget_is_not_a_text_input(self, app):
        called = []
        handler = app._shortcut(lambda: called.append(True))
        event = tk.Event()
        event.widget = app.image_label
        with patch.object(app.root, "focus_get", return_value=app.history_search):
            assert handler(event) == "break"
        assert called == [True]


class TestResetSettings:
    def test_reset_restores_frame_skip_slider(self, app):
        from text_detector.config import AppSettings

        app.frame_skip_var.set(42)
        app._reset_settings()
        assert app.frame_skip_var.get() == AppSettings().frame_skip

    def test_reset_updates_frame_skip_label(self, app):
        from text_detector.config import AppSettings

        app.frame_skip_var.set(42)
        app._reset_settings()
        assert app.frame_skip_label.cget("text") == f"Every {AppSettings().frame_skip} frames"

    def test_reset_is_not_undone_by_the_next_save(self, app):
        from text_detector.config import SETTINGS, AppSettings

        app.frame_skip_var.set(42)
        app._reset_settings()
        app._save_settings()
        assert SETTINGS.frame_skip == AppSettings().frame_skip

    def test_reset_reuses_the_ocr_engine(self, app):
        import threading

        engine = app.engine
        before = threading.active_count()
        app._reset_settings()
        assert app.engine is engine
        assert threading.active_count() == before

    def test_reset_clears_the_model_cache(self, app):
        with patch.object(app.engine, "clear_cache") as clear_cache:
            app._reset_settings()
        clear_cache.assert_called_once()


class TestSettingsWriteDebounce:
    def test_burst_of_changes_writes_once(self, app):
        with patch.object(app._settings_manager, "save") as save:
            for value in range(5, 80):
                app._threshold_changed(str(value / 100))
            assert save.call_count == 0
            app._flush_settings_write()
            assert save.call_count == 1

    def test_settings_object_is_updated_immediately(self, app):
        from text_detector.config import SETTINGS

        with patch.object(app._settings_manager, "save"):
            app.frame_skip_var.set(7)
            app._frame_skip_changed("7")
        assert SETTINGS.frame_skip == 7

    def test_pending_write_is_flushed_on_close(self, app):
        with (
            patch.object(app._settings_manager, "save") as save,
            patch.object(app.root, "destroy"),
            patch.object(app.engine, "shutdown"),
        ):
            app._threshold_changed("0.5")
            assert save.call_count == 0
            app.on_closing()
            assert save.call_count == 1

    def test_flush_without_pending_write_is_harmless(self, app):
        with patch.object(app._settings_manager, "save") as save:
            app._flush_settings_write()
            app._flush_settings_write()
        assert save.call_count == 2


class TestLanguageSelection:
    def test_one_checkbutton_per_available_language(self, app):
        from text_detector.config import SETTINGS

        assert sorted(app.language_vars) == sorted(SETTINGS.available_languages)

    def test_selection_starts_from_settings(self, app):
        from text_detector.config import SETTINGS

        assert app.current_languages == list(SETTINGS.languages)
        for code, var in app.language_vars.items():
            assert var.get() is (code in SETTINGS.languages)

    def test_enabling_a_language_adds_it(self, app):
        from text_detector.config import SETTINGS

        with patch.object(app._settings_manager, "save"):
            app.language_vars["fr"].set(True)
            app._languages_changed()
        assert "fr" in app.current_languages
        assert "fr" in SETTINGS.languages

    def test_selection_keeps_the_available_order(self, app):
        from text_detector.config import SETTINGS

        with patch.object(app._settings_manager, "save"):
            app.language_vars["fr"].set(True)
            app.language_vars["de"].set(True)
            app._languages_changed()
        expected = [c for c in SETTINGS.available_languages if c in ("en", "fr", "de")]
        assert app.current_languages == expected

    def test_unchecking_the_last_language_is_refused(self, app):
        with patch.object(app._settings_manager, "save"):
            for code in list(app.language_vars):
                if code != "en":
                    app.language_vars[code].set(False)
            app._languages_changed()
            assert app.current_languages == ["en"]

            app.language_vars["en"].set(False)
            app._languages_changed()

        assert app.current_languages == ["en"]
        assert app.language_vars["en"].get() is True
        assert "at least one" in app.status_label.cget("text").lower()

    def test_changing_languages_keeps_the_model_cache(self, app):
        with (
            patch.object(app._settings_manager, "save"),
            patch.object(app.engine, "clear_cache") as clear_cache,
        ):
            app.language_vars["fr"].set(True)
            app._languages_changed()
        clear_cache.assert_not_called()

    def test_detection_uses_every_selected_language(self, app):
        import numpy as np

        app.current_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch.object(app.engine, "detect_text_async", return_value=True) as detect:
            app.current_languages = ["fr", "en"]
            app._process_current_frame()
        assert detect.call_args.kwargs["languages"] == ["fr", "en"]

    def test_reset_restores_the_default_language_selection(self, app):
        from text_detector.config import AppSettings

        with patch.object(app._settings_manager, "save"):
            app.language_vars["fr"].set(True)
            app._languages_changed()
            app._reset_settings()
        assert app.current_languages == AppSettings().languages
        assert app.language_vars["fr"].get() is False


class TestParagraphMode:
    def test_toggle_updates_the_setting(self, app):
        from text_detector.config import SETTINGS

        with patch.object(app._settings_manager, "save"):
            app.paragraph_var.set(True)
            app._paragraph_changed()
        assert SETTINGS.paragraph_merge is True

    def test_confidence_slider_is_disabled_in_paragraph_mode(self, app):
        with patch.object(app._settings_manager, "save"):
            app.paragraph_var.set(True)
            app._paragraph_changed()
            assert str(app.threshold_scale.cget("state")) == "disabled"

            app.paragraph_var.set(False)
            app._paragraph_changed()
            assert str(app.threshold_scale.cget("state")) == "normal"


class TestOcrMaxWidth:
    def test_slider_starts_from_settings(self, app):
        from text_detector.config import SETTINGS

        assert app.ocr_width_var.get() == SETTINGS.ocr_max_width

    def test_change_updates_the_setting_and_label(self, app):
        from text_detector.config import SETTINGS

        with patch.object(app._settings_manager, "save"):
            app.ocr_width_var.set(1200)
            app._ocr_width_changed("1200")
        assert SETTINGS.ocr_max_width == 1200
        assert "1200" in app.ocr_width_label.cget("text")


class TestMissingConfidenceRendering:
    def _para_detection(self):
        return ([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]], "un paragraphe", None)

    def test_text_output_shows_a_dash(self, app):
        app.detected_text = [self._para_detection()]
        app._update_text_output()
        app.text_output.config(state="normal")
        content = app.text_output.get("1.0", tk.END)
        assert "un paragraphe (—)" in content

    def test_history_stores_no_confidence(self, app):
        app.detected_text = [self._para_detection()]
        app._add_to_history()
        assert app.history[0]["confidence"] is None
        assert "—" in app.history_listbox.get(0)

    def test_history_records_every_language(self, app):
        app.current_languages = ["fr", "en"]
        app.detected_text = [self._para_detection()]
        app._add_to_history()
        assert app.history[0]["language"] == "fr+en"

    def test_txt_export_shows_a_dash(self, app, tmp_path):
        app.detected_text = [self._para_detection()]
        path = tmp_path / "out.txt"
        app._save_txt(path)
        assert "un paragraphe (confidence: —)" in path.read_text()

    def test_csv_export_leaves_the_cell_empty(self, app, tmp_path):
        import csv

        app.current_languages = ["fr"]
        app.detected_text = [self._para_detection()]
        path = tmp_path / "out.csv"
        app._save_csv(path)
        rows = list(csv.reader(path.open()))
        assert rows[1] == ["un paragraphe", "", "fr"]

    def test_json_export_uses_null(self, app, tmp_path):
        import json as json_mod

        app.current_languages = ["fr", "en"]
        app.detected_text = [self._para_detection()]
        path = tmp_path / "out.json"
        app._save_json(path)
        data = json_mod.loads(path.read_text())
        assert data[0]["confidence"] is None
        assert data[0]["language"] == "fr+en"


class _StubGrabber:
    """Frame source that hands out canned frames to update_frame."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    def read(self):
        return self._frames.pop(0) if self._frames else None

    def stop(self):
        self.stopped = True

    @property
    def is_running(self):
        return not self.stopped


class TestFramePreviewCadence:
    """The preview must stay smooth while OCR stays throttled."""

    def _run_capture(self, app, ticks, frame_skip):
        from text_detector.config import SETTINGS

        original = SETTINGS.frame_skip
        SETTINGS.frame_skip = frame_skip
        app.capture_active = True
        app._grabber = _StubGrabber([np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(ticks)])
        try:
            with (
                patch.object(app.root, "after"),
                patch.object(app, "_render_current_frame") as render,
                patch.object(app, "_process_current_frame") as process,
            ):
                for _ in range(ticks):
                    app.update_frame()
            return render, process
        finally:
            SETTINGS.frame_skip = original

    def test_every_captured_frame_is_displayed(self, app):
        render, _process = self._run_capture(app, ticks=6, frame_skip=3)
        assert render.call_count == 6

    def test_ocr_runs_only_on_every_nth_frame(self, app):
        _render, process = self._run_capture(app, ticks=6, frame_skip=3)
        assert process.call_count == 2


class TestOcrResultDelivery:
    """OCR results must reach the GUI on the Tk thread, never from the worker."""

    def test_submission_registers_no_cross_thread_callback(self, app):
        app.current_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch.object(app.engine, "detect_text_async", return_value=True) as detect:
            app._process_current_frame()
        assert detect.call_args.kwargs.get("callback") is None

    def test_finished_results_are_applied_from_the_main_loop(self, app):
        from text_detector.ocr_engine import DetectionResult

        app.current_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
        result = DetectionResult(detections=[(bbox, "Hi", 0.9)], languages=["en"])
        with (
            patch.object(app.engine, "poll_result", return_value=result),
            patch.object(app.root, "after"),
        ):
            app.update_frame()
        assert [text for _bbox, text, _conf in app.detected_text] == ["Hi"]

    def test_idle_main_loop_applies_nothing(self, app):
        app.detected_text = []
        with (
            patch.object(app.engine, "poll_result", return_value=None),
            patch.object(app, "_apply_ocr_result") as apply_result,
            patch.object(app.root, "after"),
        ):
            app.update_frame()
        apply_result.assert_not_called()


class TestStatusRecovery:
    def test_status_leaves_the_busy_state_when_the_frame_is_gone(self, app):
        from text_detector.config import THEME
        from text_detector.ocr_engine import DetectionResult

        app.current_frame = None
        app.ocr_result = DetectionResult(detections=[], languages=["en"])
        app.status_led.set_color(THEME.status_busy)
        app._set_status("Processing...", THEME.status_busy)
        app._apply_ocr_result()
        assert app.status_led.color != THEME.status_busy
        assert "Processing" not in app.status_label.cget("text")


class TestEngineSettingsWiring:
    """The GUI must configure the engine through its public surface."""

    def test_gpu_toggle_reaches_the_engine_settings(self, app):
        with patch.object(app._settings_manager, "save"):
            app.gpu_var.set(True)
            app._gpu_changed()
        assert app.engine.settings.gpu_enabled is True

    def test_preprocess_toggle_reaches_the_engine_settings(self, app):
        with patch.object(app._settings_manager, "save"):
            app.preprocess_var.set(False)
            app._preprocess_changed()
        assert app.engine.settings.preprocess_enabled is False


class TestCaptureLifecycle:
    def test_start_capture_reports_a_camera_that_will_not_open(self, app):
        with (
            patch("text_detector.text_detector.FrameGrabber") as grabber_cls,
            patch("tkinter.messagebox.showerror") as error_box,
        ):
            grabber_cls.return_value.start.side_effect = RuntimeError("Unable to open webcam.")
            app.start_capture()
        error_box.assert_called_once()
        assert app.capture_active is False

    def test_stop_capture_stops_the_grabber(self, app):
        grabber = _StubGrabber([])
        app._grabber = grabber
        app.capture_active = True
        app.stop_capture()
        assert grabber.stopped is True
        assert app.capture_active is False
        assert app._grabber is None


class TestVersionDisplay:
    """The version shown in the UI must not drift from the package."""

    def test_sidebar_shows_the_package_version(self, app):
        from text_detector import __version__

        assert __version__ in app.version_label.cget("text")

    def test_about_dialog_shows_the_package_version(self, app):
        from unittest.mock import patch as _patch

        from text_detector import __version__

        with _patch("tkinter.messagebox.showinfo") as info:
            app.show_about()
        assert __version__ in info.call_args.args[1]


class TestColorPicker:
    def _prepare(self, app, bgr=(0, 0, 255), frame_w=100, frame_h=100, widget=400):
        from text_detector.image_processor import compute_display_geometry

        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        frame[:, :] = bgr
        app.current_frame = frame
        app._display_geometry = compute_display_geometry(frame_w, frame_h, widget, widget)
        return app._display_geometry

    def _event(self, x, y):
        event = tk.Event()
        event.x = x
        event.y = y
        return event

    def test_picker_starts_inactive(self, app):
        assert not app.picker_mode
        assert app.color_sample is None

    def test_toggle_activates_and_deactivates(self, app):
        self._prepare(app)
        app._toggle_picker_mode()
        assert app.picker_mode
        app._toggle_picker_mode()
        assert not app.picker_mode

    def test_picker_without_frame_stays_inactive(self, app):
        app.current_frame = None
        app._toggle_picker_mode()
        assert not app.picker_mode
        assert "No image" in app.status_label.cget("text")

    def test_activating_picker_cancels_roi_mode(self, app):
        self._prepare(app)
        app._toggle_roi_mode()
        assert app.roi_mode
        app._toggle_picker_mode()
        assert app.picker_mode
        assert not app.roi_mode

    def test_activating_roi_cancels_picker_mode(self, app):
        self._prepare(app)
        app._toggle_picker_mode()
        assert app.picker_mode
        app._toggle_roi_mode()
        assert app.roi_mode
        assert not app.picker_mode

    def test_click_reads_the_colour_under_the_cursor(self, app):
        self._prepare(app, bgr=(0, 0, 255))
        app._toggle_picker_mode()
        app._on_pick_click(self._event(200, 200))
        assert app.color_sample is not None
        assert app.color_sample.rgb == (255, 0, 0)
        assert app.color_sample.name == "red"
        assert app.color_swatch.cget("bg") == "#ff0000"
        assert "red" in app.color_name_label.cget("text")
        assert "#ff0000" in app.color_value_label.cget("text")

    def test_click_leaves_picker_mode(self, app):
        self._prepare(app)
        app._toggle_picker_mode()
        app._on_pick_click(self._event(200, 200))
        assert not app.picker_mode

    def test_click_samples_the_raw_frame_not_the_drawn_boxes(self, app):
        self._prepare(app, bgr=(0, 0, 255))
        app.detected_text = [
            ([[0, 0], [100, 0], [100, 100], [0, 100]], "hello", 0.9),
        ]
        app._toggle_picker_mode()
        app._on_pick_click(self._event(200, 200))
        assert app.color_sample.rgb == (255, 0, 0)

    def test_click_without_frame_is_ignored(self, app):
        app.current_frame = None
        app._display_geometry = None
        app._on_pick_click(self._event(200, 200))
        assert app.color_sample is None

    def test_copy_colour_puts_the_hex_on_the_clipboard(self, app):
        self._prepare(app, bgr=(0, 0, 255))
        app._toggle_picker_mode()
        app._on_pick_click(self._event(200, 200))
        app._copy_color()
        assert app.root.clipboard_get() == "#ff0000"
        assert "#ff0000" in app.status_label.cget("text")

    def test_copy_colour_without_a_sample(self, app):
        app.color_sample = None
        app._copy_color()
        assert "No color" in app.status_label.cget("text")

    def test_clear_results_resets_the_swatch(self, app):
        self._prepare(app, bgr=(0, 0, 255))
        app._toggle_picker_mode()
        app._on_pick_click(self._event(200, 200))
        app.clear_results()
        assert app.color_sample is None
        assert not app.picker_mode
        assert app.color_name_label.cget("text") == "—"


class TestResponsiveLayout:
    def _panes(self, app):
        return [str(pane) for pane in app.main_paned.panes()]

    def test_window_declares_a_minimum_size(self, app):
        assert tuple(app.root.minsize()) == MIN_WINDOW

    def test_large_layout_places_text_beside_image(self, app):
        app._apply_layout(LARGE)
        assert app.center_paned.cget("orient") == "horizontal"
        assert str(app.sidebar_container) in self._panes(app)

    def test_medium_layout_stacks_text_under_image(self, app):
        app._apply_layout(MEDIUM)
        assert app.center_paned.cget("orient") == "vertical"
        assert str(app.sidebar_container) in self._panes(app)

    def test_compact_layout_hides_the_sidebar(self, app):
        app._apply_layout(COMPACT)
        assert app.center_paned.cget("orient") == "vertical"
        assert str(app.sidebar_container) not in self._panes(app)

    def test_leaving_compact_restores_the_sidebar_first(self, app):
        app._apply_layout(COMPACT)
        app._apply_layout(LARGE)
        assert self._panes(app)[0] == str(app.sidebar_container)

    def test_toggle_hides_and_restores_the_sidebar(self, app):
        app._apply_layout(LARGE)
        app._toggle_sidebar()
        assert str(app.sidebar_container) not in self._panes(app)
        app._toggle_sidebar()
        assert self._panes(app)[0] == str(app.sidebar_container)

    def test_sidebar_content_is_scrollable(self, app):
        app.root.update_idletasks()
        assert app.sidebar.winfo_reqheight() > 0
        assert app.sidebar_canvas.cget("scrollregion") != ""

    def test_resize_applies_the_new_breakpoint(self, app):
        app._apply_layout(LARGE)
        with patch.object(app.root, "winfo_width", return_value=700):
            app._handle_resize()
        assert app._breakpoint == COMPACT
        assert str(app.sidebar_container) not in self._panes(app)

    def test_resize_redraws_the_current_frame(self, app):
        with (
            patch.object(app, "_render_current_frame") as render,
            patch.object(app.root, "winfo_width", return_value=1200),
        ):
            app._handle_resize()
        render.assert_called_once()

    def test_configure_events_are_debounced(self, app):
        event = tk.Event()
        event.widget = app.root
        app._on_root_configure(event)
        first = app._resize_job
        assert first is not None
        app._on_root_configure(event)
        assert app._resize_job != first

    def test_configure_from_a_child_widget_is_ignored(self, app):
        event = tk.Event()
        event.widget = app.image_label
        app._on_root_configure(event)
        assert app._resize_job is None

    def test_image_pane_absorbs_the_extra_space(self, app):
        # Tk stretches the last pane by default, which starves the image.
        assert app.main_paned.panecget(app.sidebar_container, "stretch") == "never"
        assert app.main_paned.panecget(app.center_paned, "stretch") == "always"
        assert app.center_paned.panecget(app.image_container, "stretch") == "always"
        assert app.center_paned.panecget(app.text_panel, "stretch") == "never"

    def test_text_pane_is_sized_by_width_when_beside_the_image(self, app):
        app._apply_layout(LARGE)
        assert app.center_paned.panecget(app.text_panel, "width") > 0
        assert not app.center_paned.panecget(app.text_panel, "height")

    def test_text_pane_is_sized_by_height_when_under_the_image(self, app):
        app._apply_layout(MEDIUM)
        assert app.center_paned.panecget(app.text_panel, "height") > 0
        assert not app.center_paned.panecget(app.text_panel, "width")

    def test_restored_sidebar_still_does_not_stretch(self, app):
        app._apply_layout(COMPACT)
        app._apply_layout(LARGE)
        assert app.main_paned.panecget(app.sidebar_container, "stretch") == "never"

    def test_scrollbar_shows_only_while_the_column_overflows(self, app):
        app._on_sidebar_scrolled(0.0, 0.5)
        assert app.sidebar_scrollbar.winfo_manager() == "pack"
        app._on_sidebar_scrolled(0.0, 1.0)
        assert app.sidebar_scrollbar.winfo_manager() == ""

    def test_scrollbar_is_packed_before_the_canvas(self, app):
        # Packed after it, the canvas has already claimed the whole cavity
        # and the scrollbar is allocated no width at all.
        app._on_sidebar_scrolled(0.0, 0.5)
        assert app.sidebar_container.pack_slaves()[0] is app.sidebar_scrollbar


class TestModelPreload:
    def test_app_preloads_the_model_at_startup(self):
        # Loading inside the first detection makes it look seconds slower
        # than the rest; the app is idle at startup, so pay it there.
        root = tk.Tk()
        root.withdraw()
        with patch.object(OCREngine, "preload") as preload:
            app = TextRecognitionApp(root)
        try:
            preload.assert_called_once()
        finally:
            app.stop_capture()
            app.engine.shutdown()
            root.destroy()
