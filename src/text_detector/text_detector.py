"""GUI application for real-time text recognition."""

import contextlib
import csv
import json
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from text_detector.config import SETTINGS, THEME
from text_detector.image_processor import (
    NO_CONFIDENCE,
    DisplayGeometry,
    bgr_to_rgb,
    compute_display_geometry,
    draw_boxes_with_colors,
    format_confidence,
    offset_detections,
    widget_rect_to_frame_roi,
)
from text_detector.ocr_engine import DetectionResult, OCREngine
from text_detector.settings_manager import SettingsManager
from text_detector.utils.logging_setup import get_logger
from text_detector.utils.path_helpers import get_assets_dir
from text_detector.widgets import StatusLED, ThemeableButton

logger = get_logger("gui")

HISTORY_SEARCH_PLACEHOLDER = "Search history..."
# Slider drags emit an event per pixel; coalesce the writes they trigger.
SETTINGS_WRITE_DELAY_MS = 400


class TextRecognitionApp:
    """A GUI application for real-time text recognition using EasyOCR."""

    def __init__(self, main: tk.Tk) -> None:
        self.cap: cv2.VideoCapture | None = None
        self.root = main
        self.root.title("Text Detection App")
        self.root.geometry("1200x720")
        self.root.resizable(True, True)
        self.root.configure(bg=THEME.background)

        self._settings_manager = SettingsManager()
        loaded = self._settings_manager.load()
        SETTINGS.languages = loaded.languages
        SETTINGS.default_confidence = loaded.default_confidence
        SETTINGS.gpu_enabled = loaded.gpu_enabled
        SETTINGS.preprocess_enabled = loaded.preprocess_enabled
        SETTINGS.frame_skip = loaded.frame_skip
        SETTINGS.ocr_max_width = loaded.ocr_max_width
        SETTINGS.paragraph_merge = loaded.paragraph_merge

        self.capture_active = False
        self.frame_counter = 0
        self.detected_text: list[tuple] = []
        self.current_frame: np.ndarray | None = None
        self.language_vars = {
            code: tk.BooleanVar(value=code in SETTINGS.languages)
            for code in SETTINGS.available_languages
        }
        self.threshold_var = tk.DoubleVar(value=SETTINGS.default_confidence)
        self.frame_skip_var = tk.IntVar(value=SETTINGS.frame_skip)
        self.gpu_var = tk.BooleanVar(value=SETTINGS.gpu_enabled)
        self.preprocess_var = tk.BooleanVar(value=SETTINGS.preprocess_enabled)
        self.paragraph_var = tk.BooleanVar(value=SETTINGS.paragraph_merge)
        self.ocr_width_var = tk.IntVar(value=SETTINGS.ocr_max_width)
        self.history: list[dict[str, str | float | None]] = []
        self._history_view: list[int] = []
        self.current_languages: list[str] = list(SETTINGS.languages)
        self.roi_mode = False
        self.roi: tuple[int, int, int, int] | None = None
        self._roi_start: tuple[int, int] | None = None
        self._roi_at_submit: tuple[int, int, int, int] | None = None
        self._display_geometry: DisplayGeometry | None = None
        self._settings_write_job: str | None = None

        self.engine = OCREngine(SETTINGS)
        self.ocr_result: DetectionResult | None = None
        self.ocr_lock = threading.Lock()

        self._create_widgets()
        self._configure_icon()
        self._bind_keyboard_shortcuts()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    # ── Initialisation ──────────────────────────────────────────────

    def _configure_icon(self) -> None:
        icon_path = get_assets_dir() / "icon.ico"
        if icon_path.exists():
            with contextlib.suppress(tk.TclError):
                self.root.iconbitmap(str(icon_path))

    def _create_widgets(self) -> None:
        self._create_main_layout()
        self._create_sidebar()
        self._create_image_area()
        self._create_text_panel()
        self._create_status_bar()

    def _create_main_layout(self) -> None:
        self.main_paned = tk.PanedWindow(
            self.root,
            orient="horizontal",
            bg=THEME.background,
            sashwidth=4,
            sashrelief="flat",
        )
        self.main_paned.pack(fill="both", expand=True, padx=8, pady=8)

        self.sidebar = tk.Frame(
            self.main_paned,
            bg=THEME.surface,
            relief="flat",
            padx=12,
            pady=12,
        )

        self.center_paned = tk.PanedWindow(
            self.main_paned,
            orient="horizontal",
            bg=THEME.background,
            sashwidth=4,
            sashrelief="flat",
        )
        self.main_paned.add(self.sidebar, width=240, minsize=200)
        self.main_paned.add(self.center_paned)

    # ── Sidebar ─────────────────────────────────────────────────────

    def _create_sidebar(self) -> None:
        self._create_sidebar_title()

        tk.Frame(self.sidebar, bg=THEME.border, height=1).pack(fill="x", pady=(8, 12))

        self._create_sidebar_section_label("Actions")
        self._create_action_buttons()

        tk.Frame(self.sidebar, bg=THEME.border, height=1).pack(fill="x", pady=(12, 8))

        self._create_sidebar_section_label("Settings")
        self._create_language_selector()
        self._create_threshold_slider()
        self._create_frame_skip_slider()
        self._create_ocr_width_slider()
        self._create_gpu_toggle()

    def _create_sidebar_title(self) -> None:
        tk.Label(
            self.sidebar,
            text="Text Detection",
            font=("Arial", 18, "bold"),
            bg=THEME.surface,
            fg=THEME.accent,
        ).pack(anchor="w", padx=4, pady=(0, 4))

        tk.Label(
            self.sidebar,
            text="v2.0",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        ).pack(anchor="w", padx=6)

    def _create_sidebar_section_label(self, text: str) -> None:
        tk.Label(
            self.sidebar,
            text=text.upper(),
            font=("Arial", 9, "bold"),
            bg=THEME.surface,
            fg=THEME.text_muted,
        ).pack(anchor="w", padx=4, pady=(0, 6))

    def _create_action_buttons(self) -> None:
        self.start_btn = ThemeableButton(
            self.sidebar,
            "▶ Start Webcam",
            self.start_capture,
            bg=THEME.success,
            active_bg=THEME.success_active,
            font="Arial",
            font_size=11,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Start capturing from webcam",
        )
        self.stop_btn = ThemeableButton(
            self.sidebar,
            "■ Stop Capture",
            self.stop_capture,
            bg=THEME.danger,
            active_bg=THEME.danger_active,
            font="Arial",
            font_size=11,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Stop webcam capture",
        )
        self.load_btn = ThemeableButton(
            self.sidebar,
            "📁 Load Image",
            self.load_image,
            bg=THEME.accent,
            active_bg=THEME.accent_active,
            font="Arial",
            font_size=11,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Load an image file (Ctrl+O)",
        )
        self.save_btn = ThemeableButton(
            self.sidebar,
            "💾 Save Results",
            self.save_results,
            bg=THEME.warning,
            active_bg=THEME.warning_active,
            font="Arial",
            font_size=11,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Save detected text (Ctrl+S)",
        )
        self.clear_btn = ThemeableButton(
            self.sidebar,
            "✕ Clear",
            self.clear_results,
            bg=THEME.neutral,
            active_bg=THEME.neutral_active,
            font="Arial",
            font_size=11,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Clear all results (Esc)",
        )

        self.roi_btn = ThemeableButton(
            self.sidebar,
            "▣ Select ROI",
            self._toggle_roi_mode,
            bg=THEME.accent,
            active_bg=THEME.accent_active,
            font="Arial",
            font_size=11,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Select region of interest for OCR",
        )

        self.clear_roi_btn = ThemeableButton(
            self.sidebar,
            "✕ Clear ROI",
            self._clear_roi,
            bg=THEME.neutral,
            active_bg=THEME.neutral_active,
            font="Arial",
            font_size=10,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Clear ROI selection",
        )

        tk.Frame(self.sidebar, bg=THEME.border, height=1).pack(fill="x", pady=(8, 4))

        self.about_btn = ThemeableButton(
            self.sidebar,
            "ℹ About",
            self.show_about,
            bg=THEME.surface_light,
            active_bg=THEME.about,
            font="Arial",
            font_size=10,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Show about dialog",
        )

        self.reset_btn = ThemeableButton(
            self.sidebar,
            "↺ Reset Settings",
            self._reset_settings,
            bg=THEME.surface_light,
            active_bg=THEME.warning,
            font="Arial",
            font_size=10,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Reset all settings to defaults (Ctrl+R)",
        )

    def _create_language_selector(self) -> None:
        """Build the language grid. EasyOCR reads several languages at once."""
        frame = tk.Frame(self.sidebar, bg=THEME.surface)
        frame.pack(fill="x", pady=2)

        header = tk.Frame(frame, bg=THEME.surface)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Languages",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        ).pack(side="left", padx=(4, 0))

        self.language_summary = tk.Label(
            header,
            text="+".join(self.current_languages),
            font=("Arial", 9, "bold"),
            bg=THEME.surface,
            fg=THEME.accent,
        )
        self.language_summary.pack(side="right", padx=4)

        grid = tk.Frame(frame, bg=THEME.surface)
        grid.pack(fill="x", padx=4, pady=(2, 0))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.language_checks: dict[str, tk.Checkbutton] = {}
        for index, code in enumerate(SETTINGS.available_languages):
            check = tk.Checkbutton(
                grid,
                text=code.upper(),
                variable=self.language_vars[code],
                command=self._languages_changed,
                bg=THEME.surface,
                fg=THEME.text_fg,
                activebackground=THEME.surface,
                activeforeground=THEME.accent,
                selectcolor=THEME.surface_light,
                highlightthickness=0,
                anchor="w",
                font=("Arial", 9),
            )
            check.grid(row=index // 2, column=index % 2, sticky="w")
            self.language_checks[code] = check

    def _create_threshold_slider(self) -> None:
        frame = tk.Frame(self.sidebar, bg=THEME.surface)
        frame.pack(fill="x", pady=6)

        tk.Label(
            frame,
            text="Confidence",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        ).pack(side="left", padx=(4, 0))

        self.threshold_label = tk.Label(
            frame,
            text=f"{SETTINGS.default_confidence:.2f}",
            font=("Arial", 9, "bold"),
            bg=THEME.surface,
            fg=THEME.accent,
        )
        self.threshold_label.pack(side="right", padx=(4, 4))

        self.threshold_scale = tk.Scale(
            frame,
            variable=self.threshold_var,
            from_=SETTINGS.min_confidence,
            to=SETTINGS.max_confidence,
            resolution=SETTINGS.confidence_resolution,
            orient="horizontal",
            length=160,
            bg=THEME.surface,
            fg=THEME.text_fg,
            highlightthickness=0,
            activebackground=THEME.accent,
            troughcolor=THEME.surface_light,
            sliderrelief="flat",
            borderwidth=0,
            command=self._threshold_changed,
        )
        self.threshold_scale.pack(fill="x", padx=4)

    def _create_frame_skip_slider(self) -> None:
        frame = tk.Frame(self.sidebar, bg=THEME.surface)
        frame.pack(fill="x", pady=6)

        tk.Label(
            frame,
            text="OCR Frequency",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        ).pack(side="left", padx=(4, 0))

        self.frame_skip_label = tk.Label(
            frame,
            text=f"Every {SETTINGS.frame_skip} frames",
            font=("Arial", 9, "bold"),
            bg=THEME.surface,
            fg=THEME.accent,
        )
        self.frame_skip_label.pack(side="right", padx=(4, 4))

        self.frame_skip_scale = tk.Scale(
            frame,
            variable=self.frame_skip_var,
            from_=1,
            to=60,
            resolution=1,
            orient="horizontal",
            length=160,
            bg=THEME.surface,
            fg=THEME.text_fg,
            highlightthickness=0,
            activebackground=THEME.accent,
            troughcolor=THEME.surface_light,
            sliderrelief="flat",
            borderwidth=0,
            command=self._frame_skip_changed,
        )
        self.frame_skip_scale.pack(fill="x", padx=4)

    def _create_ocr_width_slider(self) -> None:
        frame = tk.Frame(self.sidebar, bg=THEME.surface)
        frame.pack(fill="x", pady=6)

        tk.Label(
            frame,
            text="OCR Detail",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        ).pack(side="left", padx=(4, 0))

        self.ocr_width_label = tk.Label(
            frame,
            text=f"{SETTINGS.ocr_max_width} px",
            font=("Arial", 9, "bold"),
            bg=THEME.surface,
            fg=THEME.accent,
        )
        self.ocr_width_label.pack(side="right", padx=(4, 4))

        self.ocr_width_scale = tk.Scale(
            frame,
            variable=self.ocr_width_var,
            from_=400,
            to=2000,
            resolution=100,
            orient="horizontal",
            length=160,
            bg=THEME.surface,
            fg=THEME.text_fg,
            highlightthickness=0,
            activebackground=THEME.accent,
            troughcolor=THEME.surface_light,
            sliderrelief="flat",
            borderwidth=0,
            command=self._ocr_width_changed,
        )
        self.ocr_width_scale.pack(fill="x", padx=4)

    def _create_gpu_toggle(self) -> None:
        frame = tk.Frame(self.sidebar, bg=THEME.surface)
        frame.pack(fill="x", pady=6)

        self.gpu_check = tk.Checkbutton(
            frame,
            text="GPU Acceleration",
            variable=self.gpu_var,
            command=self._gpu_changed,
            bg=THEME.surface,
            fg=THEME.text_fg,
            activebackground=THEME.surface,
            selectcolor=THEME.surface_light,
            highlightthickness=0,
            font=("Arial", 9),
        )
        self.gpu_check.pack(side="left", padx=4)

        self.preprocess_check = tk.Checkbutton(
            frame,
            text="Preprocessing",
            variable=self.preprocess_var,
            command=self._preprocess_changed,
            bg=THEME.surface,
            fg=THEME.text_fg,
            activebackground=THEME.surface,
            selectcolor=THEME.surface_light,
            highlightthickness=0,
            font=("Arial", 9),
        )
        self.preprocess_check.pack(side="left", padx=4)

        self.paragraph_check = tk.Checkbutton(
            self.sidebar,
            text="Merge into paragraphs",
            variable=self.paragraph_var,
            command=self._paragraph_changed,
            bg=THEME.surface,
            fg=THEME.text_fg,
            activebackground=THEME.surface,
            selectcolor=THEME.surface_light,
            highlightthickness=0,
            anchor="w",
            font=("Arial", 9),
        )
        self.paragraph_check.pack(fill="x", padx=4)

    # ── Image area ──────────────────────────────────────────────────

    def _create_image_area(self) -> None:
        self.image_container = tk.Frame(
            self.center_paned,
            bg=THEME.result_frame_bg,
            bd=2,
            relief="flat",
        )
        self.center_paned.add(self.image_container)

        # Border and internal padding are zeroed so that widget coordinates
        # map onto the displayed image with a plain offset (see _show_image).
        self.image_label = tk.Label(
            self.image_container,
            bg=THEME.result_frame_bg,
            fg=THEME.text_muted,
            font=("Arial", 14),
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
        )
        self.image_label.pack(fill="both", expand=True, padx=8, pady=8)
        self.image_label.config(text="No image loaded")

        self._create_image_context_menu()

        self.image_border = tk.Frame(
            self.image_container,
            bg=THEME.border,
            height=2,
        )

    def _create_image_context_menu(self) -> None:
        self.image_context_menu = tk.Menu(self.root, tearoff=0)
        self.image_context_menu.add_command(
            label="Paste image from clipboard",
            command=self._paste_image_from_clipboard,
        )
        self.image_label.bind("<Button-3>", self._show_image_context_menu)
        self.image_label.bind("<Control-v>", lambda _: self._paste_image_from_clipboard())

    def _show_image_context_menu(self, event: tk.Event) -> None:
        self.image_context_menu.post(event.x_root, event.y_root)

    def _paste_image_from_clipboard(self) -> None:
        try:
            from PIL import ImageGrab

            pil_image = ImageGrab.grabclipboard()
            if pil_image is None:
                messagebox.showinfo("Paste Image", "No image found in clipboard.")
                return
            if not isinstance(pil_image, Image.Image):
                messagebox.showerror("Paste Image", "Clipboard content is not an image.")
                return
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            self.stop_capture()
            self.current_frame = frame
            self._process_current_frame()
            self._set_status("Image pasted from clipboard", THEME.accent)
            self.status_led.set_color(THEME.status_ready)
            logger.info("Image pasted from clipboard")
        except Exception as e:
            messagebox.showerror("Paste Image", f"Failed to paste image: {e}")
            logger.exception("Failed to paste image from clipboard")

    # ── Text panel ──────────────────────────────────────────────────

    def _create_text_panel(self) -> None:
        self.text_panel = tk.Frame(
            self.center_paned,
            bg=THEME.surface,
            width=280,
        )
        self.center_paned.add(self.text_panel)

        self.text_header = tk.Frame(self.text_panel, bg=THEME.accent)
        self.text_header.pack(fill="x")

        tk.Label(
            self.text_header,
            text=" Detected Text",
            font=("Arial", 11, "bold"),
            bg=THEME.accent,
            fg=THEME.button_fg,
            padx=10,
            pady=8,
        ).pack(side="left", anchor="w")

        self.copy_btn = ThemeableButton(
            self.text_header,
            "📋 Copy Text",
            self._copy_to_clipboard,
            bg=THEME.accent_active,
            active_bg=THEME.accent,
            font="Arial",
            font_size=9,
            width=12,
            side="right",
            padx=4,
            pady=4,
            fill="none",
            tooltip="Copy detected text to clipboard (Ctrl+C)",
        )

        self.notebook = ttk.Notebook(
            self.text_panel,
        )
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self._create_current_tab()
        self._create_history_tab()

    def _create_current_tab(self) -> None:
        self.current_frame_tab = tk.Frame(self.notebook, bg=THEME.text_output_bg)
        self.notebook.add(self.current_frame_tab, text="Current")

        self.text_scrollbar = tk.Scrollbar(
            self.current_frame_tab,
            orient="vertical",
            bg=THEME.scrollbar_bg,
            troughcolor=THEME.scrollbar_bg,
            activebackground=THEME.scrollbar_fg,
            highlightthickness=0,
            borderwidth=0,
        )
        self.text_scrollbar.pack(side="right", fill="y")

        self.text_output = tk.Text(
            self.current_frame_tab,
            wrap="word",
            bg=THEME.text_output_bg,
            fg=THEME.text_fg,
            font=("Consolas", 10),
            yscrollcommand=self.text_scrollbar.set,
            insertbackground=THEME.accent,
            selectbackground=THEME.accent,
            selectforeground=THEME.button_fg,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=8,
        )
        self.text_output.pack(side="left", fill="both", expand=True)
        self.text_scrollbar.config(command=self.text_output.yview)
        self.text_output.config(state="disabled")

    def _create_history_tab(self) -> None:
        self.history_tab = tk.Frame(self.notebook, bg=THEME.text_output_bg)
        self.notebook.add(self.history_tab, text="History")

        self.history_search = tk.Entry(
            self.history_tab,
            bg=THEME.surface_light,
            fg=THEME.text_fg,
            insertbackground=THEME.accent,
            relief="flat",
            font=("Arial", 9),
        )
        self.history_search.pack(fill="x", padx=4, pady=(4, 2))
        self.history_search.bind("<KeyRelease>", self._filter_history)
        self.history_search.insert(0, HISTORY_SEARCH_PLACEHOLDER)
        self.history_search.bind("<FocusIn>", self._on_history_search_focus)

        self.history_listbox_frame = tk.Frame(self.history_tab, bg=THEME.text_output_bg)
        self.history_listbox_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.history_scrollbar = tk.Scrollbar(
            self.history_listbox_frame,
            orient="vertical",
            bg=THEME.scrollbar_bg,
            troughcolor=THEME.scrollbar_bg,
            activebackground=THEME.scrollbar_fg,
            highlightthickness=0,
            borderwidth=0,
        )
        self.history_scrollbar.pack(side="right", fill="y")

        self.history_listbox = tk.Listbox(
            self.history_listbox_frame,
            bg=THEME.text_output_bg,
            fg=THEME.text_fg,
            font=("Consolas", 9),
            yscrollcommand=self.history_scrollbar.set,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            selectbackground=THEME.accent,
            selectforeground=THEME.button_fg,
        )
        self.history_listbox.pack(side="left", fill="both", expand=True)
        self.history_scrollbar.config(command=self.history_listbox.yview)
        self.history_listbox.bind("<Double-Button-1>", self._copy_history_item)

        self.clear_history_btn = ThemeableButton(
            self.history_tab,
            "Clear History",
            self._clear_history,
            bg=THEME.neutral,
            active_bg=THEME.neutral_active,
            font="Arial",
            font_size=9,
            fill="x",
            padx=4,
            pady=2,
        )

    # ── Status bar ──────────────────────────────────────────────────

    def _create_status_bar(self) -> None:
        self.status_frame = tk.Frame(
            self.root,
            bg=THEME.surface,
            bd=1,
            relief="flat",
        )
        self.status_frame.pack(side="bottom", fill="x")

        self.status_led = StatusLED(self.status_frame, size=10)
        self.status_led.canvas.pack(side="left", padx=(10, 6), pady=4)

        self.status_label = tk.Label(
            self.status_frame,
            text="Ready",
            font=("Arial", 10),
            bg=THEME.surface,
            fg=THEME.text_muted,
            anchor="w",
        )
        self.status_label.pack(side="left", fill="x", expand=True, pady=4)

        self.fps_label = tk.Label(
            self.status_frame,
            text="",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        )
        self.fps_label.pack(side="right", padx=10, pady=4)

    # ── Keyboard shortcuts ──────────────────────────────────────────

    def _is_text_input(self, widget: object) -> bool:
        """Report whether a widget is one that swallows plain keystrokes."""
        return isinstance(widget, tk.Entry | tk.Text | ttk.Entry | ttk.Combobox)

    def _is_text_input_focused(self, event: tk.Event | None = None) -> bool:
        """Report whether keystrokes are currently going into a text field.

        Prefers the widget the event came from: root-level key bindings fire
        after the focused widget's own bindings, so event.widget identifies
        the real target without depending on window-manager focus.
        """
        if event is not None and getattr(event, "widget", None) is not None:
            return self._is_text_input(event.widget)
        with contextlib.suppress(KeyError, tk.TclError):
            return self._is_text_input(self.root.focus_get())
        return False

    def _shortcut(self, action):
        """Wrap a shortcut so it never fires while the user is typing.

        Shortcuts are bound on the root window, so plain keys such as
        <space> and <Escape> would otherwise reach the history search box.
        """

        def handler(event=None):
            if self._is_text_input_focused(event):
                return None
            action()
            return "break"

        return handler

    def _bind_keyboard_shortcuts(self) -> None:
        self.root.bind("<Control-o>", self._shortcut(lambda: self.load_image()))
        self.root.bind("<Control-s>", self._shortcut(lambda: self.save_results()))
        self.root.bind("<Control-c>", self._shortcut(lambda: self._copy_to_clipboard()))
        self.root.bind("<Control-v>", self._shortcut(lambda: self._paste_image_from_clipboard()))
        self.root.bind("<space>", self._shortcut(lambda: self._toggle_capture()))
        self.root.bind("<Control-r>", self._shortcut(lambda: self._reset_settings()))
        self.root.bind("<Escape>", self._shortcut(lambda: self.clear_results()))

    def _toggle_capture(self) -> None:
        if self.capture_active:
            self.stop_capture()
        else:
            self.start_capture()

    def _copy_to_clipboard(self) -> None:
        if not self.detected_text:
            self._set_status("No text to copy", THEME.status_error)
            return
        text_only = "\n".join(text for _bbox, text, _confidence in self.detected_text)
        self.root.clipboard_clear()
        self.root.clipboard_append(text_only)
        self._set_status("Copied to clipboard", THEME.accent)
        logger.info("Text copied to clipboard")

    # ── Event handlers ──────────────────────────────────────────────

    def _save_settings(self) -> None:
        """Apply the UI values to SETTINGS now, write them to disk shortly after.

        SETTINGS is what the OCR engine reads, so it is updated immediately.
        The disk write is deferred and coalesced: dragging a slider emits one
        event per pixel of travel, which would otherwise rewrite the file
        dozens of times per gesture.
        """
        SETTINGS.languages = list(self.current_languages)
        SETTINGS.default_confidence = self.threshold_var.get()
        SETTINGS.frame_skip = self.frame_skip_var.get()
        SETTINGS.gpu_enabled = self.gpu_var.get()
        SETTINGS.preprocess_enabled = self.preprocess_var.get()
        self._schedule_settings_write()

    def _schedule_settings_write(self) -> None:
        """(Re)arm the deferred write, cancelling any pending one."""
        self._cancel_settings_write()
        self._settings_write_job = self.root.after(
            SETTINGS_WRITE_DELAY_MS, self._flush_settings_write
        )

    def _cancel_settings_write(self) -> None:
        if self._settings_write_job is not None:
            with contextlib.suppress(tk.TclError, ValueError):
                self.root.after_cancel(self._settings_write_job)
            self._settings_write_job = None

    def _flush_settings_write(self) -> None:
        """Write the settings to disk now, cancelling any deferred write."""
        self._cancel_settings_write()
        self._settings_manager.save(SETTINGS)

    def _selected_languages(self) -> list[str]:
        """Return the ticked languages, in the order they are offered."""
        return [code for code in SETTINGS.available_languages if self.language_vars[code].get()]

    def _languages_changed(self) -> None:
        """Apply the language grid, refusing an empty selection."""
        selected = self._selected_languages()
        if not selected:
            # EasyOCR needs at least one language; put the last one back.
            for code in self.current_languages:
                self.language_vars[code].set(True)
            self._set_status("Keep at least one language selected", THEME.status_error)
            return

        self.current_languages = selected
        self.language_summary.config(text="+".join(selected))
        # No cache clearing: readers are keyed by language combination, so a
        # combination used before is reused instead of being reloaded.
        self._set_status(f"Languages: {'+'.join(selected)}", THEME.accent)
        logger.info("Languages changed to %s", selected)
        self._save_settings()

    def _paragraph_changed(self) -> None:
        """Toggle paragraph mode and the controls it makes meaningless."""
        enabled = self.paragraph_var.get()
        SETTINGS.paragraph_merge = enabled
        # Paragraph mode returns merged blocks with no confidence, so there
        # is nothing left for the confidence threshold to filter.
        self.threshold_scale.config(state="disabled" if enabled else "normal")
        status = "enabled" if enabled else "disabled"
        self._set_status(f"Paragraph merge {status}", THEME.accent)
        logger.info("Paragraph merge %s", status)
        self._save_settings()

    def _ocr_width_changed(self, value: str) -> None:
        width = int(float(value))
        self.ocr_width_label.config(text=f"{width} px")
        SETTINGS.ocr_max_width = width
        self._save_settings()

    def _threshold_changed(self, value: str) -> None:
        self.threshold_label.config(text=f"{float(value):.2f}")
        self._save_settings()

    def _frame_skip_changed(self, value: str) -> None:
        skip = int(float(value))
        self.frame_skip_label.config(text=f"Every {skip} frames")
        self._save_settings()

    def _gpu_changed(self) -> None:
        self.engine.clear_cache()
        self.engine._settings.gpu_enabled = self.gpu_var.get()
        status = "enabled" if self.gpu_var.get() else "disabled"
        self._set_status(f"GPU {status}", THEME.accent)
        logger.info("GPU %s", status)
        self._save_settings()

    def _preprocess_changed(self) -> None:
        self.engine._settings.preprocess_enabled = self.preprocess_var.get()
        status = "enabled" if self.preprocess_var.get() else "disabled"
        self._set_status(f"Preprocessing {status}", THEME.accent)
        logger.info("Preprocessing %s", status)
        self._save_settings()

    def _reset_settings(self) -> None:
        """Reset settings to defaults and reload the UI to match."""
        self._cancel_settings_write()
        self._settings_manager.reset()
        defaults = self._settings_manager.load()
        SETTINGS.languages = list(defaults.languages)
        SETTINGS.default_confidence = defaults.default_confidence
        SETTINGS.gpu_enabled = defaults.gpu_enabled
        SETTINGS.preprocess_enabled = defaults.preprocess_enabled
        SETTINGS.frame_skip = defaults.frame_skip
        SETTINGS.ocr_max_width = defaults.ocr_max_width
        SETTINGS.paragraph_merge = defaults.paragraph_merge

        # Every widget bound to a setting must follow, or the next save
        # writes the stale value straight back over the reset.
        for code, var in self.language_vars.items():
            var.set(code in SETTINGS.languages)
        self.language_summary.config(text="+".join(SETTINGS.languages))
        self.threshold_var.set(SETTINGS.default_confidence)
        self.frame_skip_var.set(SETTINGS.frame_skip)
        self.gpu_var.set(SETTINGS.gpu_enabled)
        self.preprocess_var.set(SETTINGS.preprocess_enabled)
        self.paragraph_var.set(SETTINGS.paragraph_merge)
        self.ocr_width_var.set(SETTINGS.ocr_max_width)
        self.current_languages = list(SETTINGS.languages)
        self.threshold_label.config(text=f"{SETTINGS.default_confidence:.2f}")
        self.ocr_width_label.config(text=f"{SETTINGS.ocr_max_width} px")
        self.threshold_scale.config(state="disabled" if SETTINGS.paragraph_merge else "normal")
        self.frame_skip_label.config(text=f"Every {SETTINGS.frame_skip} frames")

        # The engine already holds this same SETTINGS object, so it only
        # needs its readers dropped; replacing it would strand the running
        # worker thread and its cached models.
        self.engine.clear_cache()
        self._set_status("Settings reset to defaults", THEME.neutral)
        logger.info("Settings reset to defaults")

    def start_capture(self) -> None:
        if self.capture_active:
            return
        self.capture_active = True
        self.frame_counter = 0
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Unable to open webcam.")
            self._set_status("Webcam active", THEME.status_ready)
            self.status_led.set_color(THEME.status_ready)
            logger.info("Webcam capture started")
        except Exception as exc:
            self.capture_active = False
            messagebox.showerror("Webcam Error", str(exc))
            self._set_status("Webcam error", THEME.status_error)
            self.status_led.set_color(THEME.status_error)
            logger.error("Webcam error: %s", exc)

    def stop_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.capture_active = False
        self._set_status("Capture stopped", THEME.status_error)
        self.status_led.set_color(THEME.status_ready)
        logger.info("Capture stopped")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*")],
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Unable to open selected image.")
            return

        self.stop_capture()
        self.current_frame = frame
        self._process_current_frame()
        self._set_status(f"Loaded: {Path(path).name}", THEME.accent)
        self.status_led.set_color(THEME.status_ready)
        logger.info("Loaded image: %s", path)

    def _process_current_frame(self) -> None:
        frame = self._get_cropped_frame()
        if frame is None:
            return

        if self.engine.is_busy:
            return

        # Remember which ROI this frame was cropped with: the user may
        # change it before the asynchronous result comes back.
        self._roi_at_submit = self.roi

        def _on_result(result: DetectionResult) -> None:
            with self.ocr_lock:
                self.ocr_result = result
            self.root.after(0, self._apply_ocr_result)

        queued = self.engine.detect_text_async(
            frame,
            languages=list(self.current_languages),
            threshold=self.threshold_var.get(),
            callback=_on_result,
        )
        if queued:
            self._set_status("Processing...", THEME.status_busy)
            self.status_led.set_color(THEME.status_busy)

    def _apply_ocr_result(self) -> None:
        with self.ocr_lock:
            if self.ocr_result is None or not self.ocr_result.success:
                self._set_status("OCR failed", THEME.status_error)
                self.status_led.set_color(THEME.status_error)
                return
            detections = self.ocr_result.detections

        # Detections are relative to the cropped ROI; move them back into
        # full-frame coordinates so boxes land on the right pixels.
        if self._roi_at_submit is not None:
            detections = offset_detections(
                detections, self._roi_at_submit[0], self._roi_at_submit[1]
            )
        self.detected_text = detections

        if self.current_frame is None:
            return
        self._render_current_frame()
        self._update_text_output()
        self._add_to_history()
        self._set_status("Detection complete", THEME.status_ready)
        self.status_led.set_color(THEME.status_ready)

    def update_frame(self) -> None:
        if self.capture_active and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.frame_counter += 1
                if self.frame_counter % SETTINGS.frame_skip == 0:
                    self._show_image(frame)
                    self._process_current_frame()
        self.root.after(33, self.update_frame)

    def _show_image(self, frame: np.ndarray) -> None:
        """Display a full frame, scaled to fit, and record its placement.

        The recorded geometry is what lets pointer coordinates be converted
        back into frame coordinates for ROI selection.
        """
        pil_image = Image.fromarray(bgr_to_rgb(frame))
        geometry = compute_display_geometry(
            pil_image.width,
            pil_image.height,
            self.image_label.winfo_width(),
            self.image_label.winfo_height(),
        )
        if (geometry.width, geometry.height) != (pil_image.width, pil_image.height):
            # LANCZOS costs ~21 ms on a 720p frame, too much for the 33 ms
            # capture loop; still images can afford the better filter.
            resample = (
                Image.Resampling.BILINEAR if self.capture_active else Image.Resampling.LANCZOS
            )
            pil_image = pil_image.resize((geometry.width, geometry.height), resample)
        self._display_geometry = geometry

        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=tk_image, text="")
        self.image_label.image = tk_image  # type: ignore[attr-defined]

    def _render_current_frame(self) -> None:
        """Redraw the current frame with its detection boxes, if any."""
        if self.current_frame is None:
            return
        if self.detected_text:
            self._show_image(draw_boxes_with_colors(self.current_frame, self.detected_text))
        else:
            self._show_image(self.current_frame)

    def _update_text_output(self) -> None:
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        if self.detected_text:
            for _bbox, text, confidence in self.detected_text:
                self.text_output.insert(
                    tk.END, f"{text} ({format_confidence(confidence)})\n"
                )
        else:
            self.text_output.insert(tk.END, "No text detected.\n")
        self.text_output.config(state="disabled")

    def _add_to_history(self) -> None:
        for _bbox, text, confidence in self.detected_text:
            self.history.append(
                {
                    "text": text,
                    "confidence": None if confidence is None else round(confidence, 2),
                    "language": "+".join(self.current_languages),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
            )
        if len(self.history) > SETTINGS.max_history:
            self.history = self.history[-SETTINGS.max_history :]
        self._refresh_history_display()

    def _current_search_query(self) -> str:
        """Return the active history filter, ignoring the placeholder text."""
        query = self.history_search.get()
        if query == HISTORY_SEARCH_PLACEHOLDER:
            return ""
        return query.lower()

    def _refresh_history_display(self) -> None:
        """Repopulate the history listbox, honouring the active search filter."""
        query = self._current_search_query()
        self._history_view = []
        self.history_listbox.delete(0, tk.END)
        for index, entry in enumerate(self.history):
            text = str(entry["text"])
            if query and query not in text.lower():
                continue
            self._history_view.append(index)
            confidence = entry["confidence"]
            shown = NO_CONFIDENCE if confidence is None else confidence
            line = f"[{entry['timestamp']}] {text} ({shown})"
            language = entry.get("language")
            if language:
                line = f"{line} · {language}"
            self.history_listbox.insert(tk.END, line)

    def _filter_history(self, _event=None) -> None:
        self._refresh_history_display()

    def _on_history_search_focus(self, _event=None) -> None:
        if self.history_search.get() == HISTORY_SEARCH_PLACEHOLDER:
            self.history_search.delete(0, tk.END)

    def _copy_history_item(self, _event=None) -> None:
        sel = self.history_listbox.curselection()
        if not sel:
            return
        row = sel[0]
        if row >= len(self._history_view):
            return
        text = self.history[self._history_view[row]]["text"]
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Copied to clipboard", THEME.status_ready)

    def _clear_history(self) -> None:
        self.history.clear()
        self._refresh_history_display()
        self._set_status("History cleared", THEME.neutral)

    def save_results(self) -> None:
        if not self.detected_text:
            messagebox.showwarning("Warning", "No detected text to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save detected text",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
            ],
            initialfile="detected_text_results",
        )
        if not save_path:
            return

        path = Path(save_path)
        suffix = path.suffix.lower()

        try:
            if suffix == ".txt":
                self._save_txt(path)
            elif suffix == ".csv":
                self._save_csv(path)
            elif suffix == ".json":
                self._save_json(path)
            else:
                self._save_txt(path)

            messagebox.showinfo("Saved", f"Results saved to {save_path}")
            logger.info("Results saved to %s", save_path)
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
            logger.error("Save error: %s", exc)

    def _save_txt(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for _bbox, text, confidence in self.detected_text:
                f.write(f"{text} (confidence: {format_confidence(confidence)})\n")

    def _save_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "confidence", "language"])
            for _bbox, text, confidence in self.detected_text:
                writer.writerow(
                    [
                        text,
                        "" if confidence is None else f"{confidence:.2f}",
                        "+".join(self.current_languages),
                    ]
                )

    def _save_json(self, path: Path) -> None:
        data = [
            {
                "text": text,
                "confidence": None if confidence is None else round(confidence, 2),
                "language": "+".join(self.current_languages),
            }
            for _bbox, text, confidence in self.detected_text
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_results(self) -> None:
        self.stop_capture()
        if self.roi_mode:
            self._toggle_roi_mode()
        self.current_frame = None
        self.detected_text = []
        self.ocr_result = None
        self.roi = None
        self._roi_start = None
        self._roi_at_submit = None
        self._display_geometry = None
        self.image_label.config(image="", text="No image loaded")  # type: ignore[arg-type]
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, "Ready. Load an image or start the webcam.")
        self.text_output.config(state="disabled")
        self._set_status("Cleared", THEME.neutral)
        self.status_led.set_color(THEME.status_ready)

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Text Detection App v2.0\n\n"
            "Modern dark-themed interface.\n"
            "Developed using OpenCV, EasyOCR, and Tkinter.\n\n"
            "Features: threaded OCR, multi-language, GPU support,\n"
            "export to TXT/CSV/JSON, detection history.",
        )

    def on_closing(self) -> None:
        self._save_settings()
        self._flush_settings_write()
        self.capture_active = False
        if self.cap is not None:
            self.cap.release()
        self.engine.shutdown()
        self.root.destroy()
        logger.info("Application closed")

    def _toggle_roi_mode(self) -> None:
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            self.roi_btn.config(text="✓ ROI Active")
            self.roi_btn.set_base_bg(THEME.success, THEME.success_active)
            self._set_status("ROI mode: click and drag on image", THEME.accent)
            self.image_label.bind("<Button-1>", self._on_roi_click)
            self.image_label.bind("<B1-Motion>", self._on_roi_drag)
            self.image_label.bind("<ButtonRelease-1>", self._on_roi_release)
        else:
            self.roi_btn.config(text="▣ Select ROI")
            self.roi_btn.set_base_bg(THEME.accent, THEME.accent_active)
            self.image_label.unbind("<Button-1>")
            self.image_label.unbind("<B1-Motion>")
            self.image_label.unbind("<ButtonRelease-1>")

    def _widget_rect_to_roi(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        min_size: int = 10,
    ) -> tuple[int, int, int, int] | None:
        """Map a drag in image_label coordinates onto the current frame."""
        if self.current_frame is None or self._display_geometry is None:
            return None
        height, width = self.current_frame.shape[:2]
        return widget_rect_to_frame_roi(
            start, end, self._display_geometry, width, height, min_size=min_size
        )

    def _on_roi_click(self, event: tk.Event) -> None:
        self._roi_start = (event.x, event.y)

    def _on_roi_drag(self, event: tk.Event) -> None:
        """Draw a rubber band over the frame while the selection is dragged."""
        if self._roi_start is None or self.current_frame is None:
            return
        preview = self._widget_rect_to_roi(self._roi_start, (event.x, event.y), min_size=1)
        if preview is None:
            return
        frame = (
            draw_boxes_with_colors(self.current_frame, self.detected_text)
            if self.detected_text
            else self.current_frame.copy()
        )
        cv2.rectangle(frame, (preview[0], preview[1]), (preview[2], preview[3]), (250, 180, 137), 2)
        self._show_image(frame)

    def _on_roi_release(self, event: tk.Event) -> None:
        start, self._roi_start = self._roi_start, None

        if start is not None and self.current_frame is not None:
            roi = self._widget_rect_to_roi(start, (event.x, event.y))
            if roi is None:
                self._set_status("Selection too small - ROI unchanged", THEME.status_error)
            else:
                self.roi = roi
                self._set_status(
                    f"ROI set: ({roi[0]}, {roi[1]}) to ({roi[2]}, {roi[3]})",
                    THEME.status_ready,
                )
            self._render_current_frame()

        if self.roi_mode:
            self._toggle_roi_mode()

    def _clear_roi(self) -> None:
        self.roi = None
        self._set_status("ROI cleared", THEME.neutral)

    def _get_cropped_frame(self) -> np.ndarray | None:
        if self.current_frame is None:
            return None
        if self.roi:
            height, width = self.current_frame.shape[:2]
            x1 = max(0, min(self.roi[0], width))
            y1 = max(0, min(self.roi[1], height))
            x2 = max(x1, min(self.roi[2], width))
            y2 = max(y1, min(self.roi[3], height))
            crop = self.current_frame[y1:y2, x1:x2]
            if crop.size == 0:
                # The ROI no longer intersects the frame (e.g. a smaller
                # image was loaded); drop it rather than crop to nothing.
                self.roi = None
                self._set_status("ROI outside image - cleared", THEME.status_error)
                return self.current_frame
            return crop
        return self.current_frame

    # ── Helpers ─────────────────────────────────────────────────────

    def _set_status(self, text: str, bg_color: str) -> None:
        self.status_label.config(text=text, fg=bg_color)
