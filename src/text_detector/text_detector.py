"""GUI application for real-time text recognition."""

import contextlib
import csv
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from text_detector import __version__
from text_detector.capture import FrameGrabber
from text_detector.color_detector import ColorSample, sample_color
from text_detector.config import SETTINGS, THEME, AppSettings
from text_detector.image_processor import (
    NO_CONFIDENCE,
    DisplayGeometry,
    bgr_to_rgb,
    compute_display_geometry,
    draw_boxes_with_colors,
    format_confidence,
    offset_detections,
    resize_frame_for_display,
    widget_rect_to_frame_roi,
    widget_to_frame_point,
)
from text_detector.layout import (
    COMPACT,
    LARGE,
    MIN_WINDOW,
    breakpoint_for,
    initial_window_size,
    scale_px,
)
from text_detector.ocr_engine import DetectionResult, OCREngine
from text_detector.settings_manager import SettingsManager
from text_detector.utils.logging_setup import get_logger
from text_detector.utils.path_helpers import get_assets_dir
from text_detector.widgets import StatusLED, ThemeableButton

logger = get_logger("gui")

HISTORY_SEARCH_PLACEHOLDER = "Search history..."
EMPTY_COLOR = "\u2014"
# Slider drags emit an event per pixel; coalesce the writes they trigger.
SETTINGS_WRITE_DELAY_MS = 400
# Dragging a window edge emits a Configure event per pixel; coalesce them.
RESIZE_DEBOUNCE_MS = 120
# Size the text panel keeps when the image takes the remaining space.
TEXT_PANEL_WIDTH = 300
TEXT_PANEL_HEIGHT = 220


def _apply_to_global_settings(loaded: AppSettings) -> None:
    """Copy persisted values onto the SETTINGS singleton, in place.

    The OCR engine holds a reference to that one object, so it is updated
    rather than replaced; a fresh instance would leave the engine reading
    the old values.
    """
    SETTINGS.languages = list(loaded.languages)
    SETTINGS.default_confidence = loaded.default_confidence
    SETTINGS.gpu_enabled = loaded.gpu_enabled
    SETTINGS.preprocess_enabled = loaded.preprocess_enabled
    SETTINGS.frame_skip = loaded.frame_skip
    SETTINGS.ocr_max_width = loaded.ocr_max_width
    SETTINGS.paragraph_merge = loaded.paragraph_merge
    SETTINGS.validate()


class TextRecognitionApp:
    """A GUI application for real-time text recognition using EasyOCR."""

    def __init__(self, main: tk.Tk) -> None:
        self._grabber: FrameGrabber | None = None
        self.root = main
        self.root.title("Text Detection App")
        self._scaling = float(self.root.tk.call("tk", "scaling"))
        self._sidebar_width = scale_px(240, self._scaling)
        self._sidebar_min_width = scale_px(200, self._scaling)
        window_width, window_height = initial_window_size(
            self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        )
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(*MIN_WINDOW)
        self.root.resizable(True, True)
        self.root.configure(bg=THEME.background)

        self._settings_manager = SettingsManager()
        _apply_to_global_settings(self._settings_manager.load())

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
        self.picker_mode = False
        self.color_sample: ColorSample | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self._roi_start: tuple[int, int] | None = None
        self._roi_at_submit: tuple[int, int, int, int] | None = None
        self._display_geometry: DisplayGeometry | None = None
        self._settings_write_job: str | None = None
        self._breakpoint: str | None = None
        self._sidebar_visible = True
        self._resize_job: str | None = None

        self.engine = OCREngine(SETTINGS)
        # Loading the model takes seconds. Start it now, while the window is
        # still being built and nobody is waiting on a detection, instead of
        # letting the first frame pay for it.
        self.engine.preload()
        self.ocr_result: DetectionResult | None = None

        self._create_widgets()
        self._configure_icon()
        self._bind_keyboard_shortcuts()
        self._apply_layout(breakpoint_for(window_width))
        self.root.bind("<Configure>", self._on_root_configure)

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

        self._create_scrollable_sidebar()

        self.center_paned = tk.PanedWindow(
            self.main_paned,
            orient="horizontal",
            bg=THEME.background,
            sashwidth=4,
            sashrelief="flat",
        )
        self.main_paned.add(
            self.sidebar_container,
            width=self._sidebar_width,
            minsize=self._sidebar_min_width,
            stretch="never",
        )
        self.main_paned.add(self.center_paned, stretch="always")

    def _create_scrollable_sidebar(self) -> None:
        """Put the sidebar on a canvas so its content can scroll.

        The settings column is taller than a short screen, and without this
        the last sliders are simply unreachable. self.sidebar stays the frame
        every other builder packs into, so nothing else has to change; the
        widget added to the paned window is sidebar_container.
        """
        self.sidebar_container = tk.Frame(self.main_paned, bg=THEME.surface)

        self.sidebar_scrollbar = tk.Scrollbar(
            self.sidebar_container,
            orient="vertical",
            bg=THEME.scrollbar_bg,
            troughcolor=THEME.scrollbar_bg,
            activebackground=THEME.scrollbar_fg,
            highlightthickness=0,
            borderwidth=0,
        )

        self.sidebar_canvas = tk.Canvas(
            self.sidebar_container,
            bg=THEME.surface,
            highlightthickness=0,
            borderwidth=0,
            yscrollcommand=self._on_sidebar_scrolled,
        )
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_scrollbar.config(command=self.sidebar_canvas.yview)

        self.sidebar = tk.Frame(
            self.sidebar_canvas,
            bg=THEME.surface,
            relief="flat",
            padx=12,
            pady=12,
        )
        self._sidebar_window = self.sidebar_canvas.create_window(
            (0, 0), window=self.sidebar, anchor="nw"
        )
        self.sidebar.bind("<Configure>", self._on_sidebar_content_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)

    def _on_sidebar_content_configure(self, _event: tk.Event) -> None:
        self.sidebar_canvas.config(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event: tk.Event) -> None:
        # Keep the inner frame as wide as the canvas, otherwise the widgets
        # packed with fill="x" only take their requested width.
        self.sidebar_canvas.itemconfigure(self._sidebar_window, width=event.width)

    def _on_sidebar_scrolled(self, first: float | str, last: float | str) -> None:
        """Show the scrollbar only while the column overflows.

        Tk hands the fractions over as strings, hence the conversion.
        """
        top, bottom = float(first), float(last)
        if top <= 0.0 and bottom >= 1.0:
            self.sidebar_scrollbar.pack_forget()
        elif not self.sidebar_scrollbar.winfo_ismapped():
            # Packed after the canvas, which fills the cavity, the scrollbar
            # would be allocated no width at all.
            self.sidebar_scrollbar.pack(side="right", fill="y", before=self.sidebar_canvas)
        self.sidebar_scrollbar.set(top, bottom)

    def _bind_sidebar_wheel(self, widget: tk.Misc) -> None:
        """Bind the wheel on the sidebar and every widget inside it."""
        widget.bind("<MouseWheel>", self._on_sidebar_wheel)
        widget.bind("<Button-4>", self._on_sidebar_wheel)
        widget.bind("<Button-5>", self._on_sidebar_wheel)
        for child in widget.winfo_children():
            self._bind_sidebar_wheel(child)

    def _on_sidebar_wheel(self, event: tk.Event) -> str:
        # X11 reports the wheel as buttons 4 and 5, every other platform as
        # a delta on <MouseWheel>.
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self.sidebar_canvas.yview_scroll(step, "units")
        return "break"

    # ── Sidebar ─────────────────────────────────────────────────────

    def _create_sidebar(self) -> None:
        self._create_sidebar_title()

        tk.Frame(self.sidebar, bg=THEME.border, height=1).pack(fill="x", pady=(8, 12))

        self._create_sidebar_section_label("Actions")
        self._create_action_buttons()

        tk.Frame(self.sidebar, bg=THEME.border, height=1).pack(fill="x", pady=(12, 8))

        self._create_sidebar_section_label("Color")
        self._create_color_readout()

        tk.Frame(self.sidebar, bg=THEME.border, height=1).pack(fill="x", pady=(12, 8))

        self._create_sidebar_section_label("Settings")
        self._create_language_selector()
        self._create_threshold_slider()
        self._create_frame_skip_slider()
        self._create_ocr_width_slider()
        self._create_gpu_toggle()

        self._bind_sidebar_wheel(self.sidebar)

    def _create_sidebar_title(self) -> None:
        tk.Label(
            self.sidebar,
            text="Text Detection",
            font=("Arial", 18, "bold"),
            bg=THEME.surface,
            fg=THEME.accent,
        ).pack(anchor="w", padx=4, pady=(0, 4))

        self.version_label = tk.Label(
            self.sidebar,
            text=f"v{__version__}",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        )
        self.version_label.pack(anchor="w", padx=6)

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

        self.picker_btn = ThemeableButton(
            self.sidebar,
            "🎨 Pick Color",
            self._toggle_picker_mode,
            bg=THEME.about,
            active_bg=THEME.about_active,
            font="Arial",
            font_size=11,
            fill="x",
            expand=True,
            pady=4,
            tooltip="Sample the color under a click (Ctrl+P)",
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

    def _create_color_readout(self) -> None:
        """Build the swatch and labels the color picker writes into."""
        frame = tk.Frame(self.sidebar, bg=THEME.surface)
        frame.pack(fill="x", pady=2)

        self.color_swatch = tk.Canvas(
            frame,
            width=scale_px(28, self._scaling),
            height=scale_px(28, self._scaling),
            bg=THEME.surface_light,
            highlightthickness=1,
            highlightbackground=THEME.border,
            cursor="hand2",
        )
        self.color_swatch.pack(side="left", padx=(4, 8))
        self.color_swatch.bind("<Button-1>", lambda _event: self._copy_color())

        labels = tk.Frame(frame, bg=THEME.surface)
        labels.pack(side="left", fill="x", expand=True)

        self.color_name_label = tk.Label(
            labels,
            text=EMPTY_COLOR,
            font=("Arial", 10, "bold"),
            bg=THEME.surface,
            fg=THEME.text_fg,
            anchor="w",
        )
        self.color_name_label.pack(fill="x")

        self.color_value_label = tk.Label(
            labels,
            text=EMPTY_COLOR,
            font=("Courier", 8),
            bg=THEME.surface,
            fg=THEME.text_muted,
            anchor="w",
        )
        self.color_value_label.pack(fill="x")

        self.color_hsv_label = tk.Label(
            labels,
            text=EMPTY_COLOR,
            font=("Courier", 8),
            bg=THEME.surface,
            fg=THEME.text_muted,
            anchor="w",
        )
        self.color_hsv_label.pack(fill="x")

    def _reset_color_readout(self) -> None:
        """Return the swatch and its labels to their empty state."""
        self.color_sample = None
        self.color_swatch.config(bg=THEME.surface_light)
        self.color_name_label.config(text=EMPTY_COLOR)
        self.color_value_label.config(text=EMPTY_COLOR)
        self.color_hsv_label.config(text=EMPTY_COLOR)

    def _update_color_readout(self, sample: ColorSample) -> None:
        """Show one reading in the sidebar."""
        red, green, blue = sample.rgb
        hue, saturation, value = sample.hsv
        self.color_swatch.config(bg=sample.hex)
        self.color_name_label.config(text=sample.name)
        self.color_value_label.config(text=f"{sample.hex}  RGB({red}, {green}, {blue})")
        self.color_hsv_label.config(text=f"HSV({hue}°, {saturation}%, {value}%)")

    def _copy_color(self) -> None:
        if self.color_sample is None:
            self._set_status("No color sampled yet", THEME.status_error)
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.color_sample.hex)
        self._set_status(f"Copied {self.color_sample.hex}", THEME.accent)
        logger.info("Color %s copied to clipboard", self.color_sample.hex)

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
        # Tk stretches the last pane by default, which would leave the
        # image a sliver while the text panel took the whole window.
        self.center_paned.add(
            self.image_container,
            stretch="always",
            minsize=scale_px(160, self._scaling),
        )

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
            width=scale_px(280, self._scaling),
        )
        self.center_paned.add(
            self.text_panel,
            stretch="never",
            minsize=scale_px(120, self._scaling),
        )

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

        self.sidebar_toggle_btn = ThemeableButton(
            self.status_frame,
            "☰",
            self._toggle_sidebar,
            bg=THEME.surface_light,
            active_bg=THEME.accent,
            font="Arial",
            font_size=9,
            width=2,
            side="left",
            padx=6,
            pady=2,
            fill="none",
            tooltip="Show or hide the settings sidebar",
        )

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

    # ── Responsive layout ───────────────────────────────────────────

    def _apply_layout(self, layout: str) -> None:
        """Rearrange the panes for one of the three window sizes.

        Nothing is rebuilt: the centre pane only changes orientation, and
        the sidebar is removed from the paned window and put back.
        """
        self._breakpoint = layout
        side_by_side = layout == LARGE
        self.center_paned.config(orient="horizontal" if side_by_side else "vertical")
        # A pane is sized along the split axis only; the other dimension is
        # cleared so the pane fills it.
        self.center_paned.paneconfigure(
            self.text_panel,
            width=scale_px(TEXT_PANEL_WIDTH, self._scaling) if side_by_side else "",
            height="" if side_by_side else scale_px(TEXT_PANEL_HEIGHT, self._scaling),
        )
        self._set_sidebar_visible(layout != COMPACT)

    def _set_sidebar_visible(self, visible: bool) -> None:
        if visible == self._sidebar_visible:
            return
        if visible:
            self.main_paned.add(
                self.sidebar_container,
                before=self.center_paned,
                width=self._sidebar_width,
                minsize=self._sidebar_min_width,
                stretch="never",
            )
        else:
            self.main_paned.forget(self.sidebar_container)
        self._sidebar_visible = visible

    def _toggle_sidebar(self) -> None:
        self._set_sidebar_visible(not self._sidebar_visible)

    def _on_root_configure(self, event: tk.Event) -> None:
        """Coalesce the burst of events a window drag emits.

        Child widgets report their own resizes through the same binding, so
        only the root window's events are acted on.
        """
        if event.widget is not self.root:
            return
        self._cancel_resize()
        self._resize_job = self.root.after(RESIZE_DEBOUNCE_MS, self._handle_resize)

    def _cancel_resize(self) -> None:
        if self._resize_job is not None:
            with contextlib.suppress(tk.TclError):
                self.root.after_cancel(self._resize_job)
            self._resize_job = None

    def _handle_resize(self) -> None:
        self._resize_job = None
        layout = breakpoint_for(self.root.winfo_width())
        if layout != self._breakpoint:
            self._apply_layout(layout)
        # A still image keeps the scale it was drawn at, so it has to be
        # redrawn against the new pane size.
        self._render_current_frame()

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
        self.root.bind("<Control-p>", self._shortcut(lambda: self._toggle_picker_mode()))
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
        SETTINGS.validate()
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
        # The engine holds this same SETTINGS object, so _save_settings is
        # what reconfigures it; only the loaded readers must be dropped.
        self.engine.clear_cache()
        status = "enabled" if self.gpu_var.get() else "disabled"
        self._set_status(f"GPU {status}", THEME.accent)
        logger.info("GPU %s", status)
        self._save_settings()

    def _preprocess_changed(self) -> None:
        status = "enabled" if self.preprocess_var.get() else "disabled"
        self._set_status(f"Preprocessing {status}", THEME.accent)
        logger.info("Preprocessing %s", status)
        self._save_settings()

    def _reset_settings(self) -> None:
        """Reset settings to defaults and reload the UI to match."""
        self._cancel_settings_write()
        self._settings_manager.reset()
        _apply_to_global_settings(self._settings_manager.load())

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
        self.frame_counter = 0
        grabber = FrameGrabber()
        try:
            # start() opens the camera synchronously, so a missing or busy
            # device is reported here rather than from the capture thread.
            grabber.start()
        except Exception as exc:
            messagebox.showerror("Webcam Error", str(exc))
            self._set_status("Webcam error", THEME.status_error)
            self.status_led.set_color(THEME.status_error)
            logger.error("Webcam error: %s", exc)
            return

        self._grabber = grabber
        self.capture_active = True
        self._set_status("Webcam active", THEME.status_ready)
        self.status_led.set_color(THEME.status_ready)
        logger.info("Webcam capture started")

    def stop_capture(self) -> None:
        grabber, self._grabber = self._grabber, None
        if grabber is not None:
            grabber.stop()
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

        # No callback: the result is collected by _drain_ocr_results on the
        # Tk thread, so nothing in the worker ever touches a widget.
        queued = self.engine.detect_text_async(
            frame,
            languages=list(self.current_languages),
            threshold=self.threshold_var.get(),
        )
        if queued:
            self._set_status("Processing...", THEME.status_busy)
            self.status_led.set_color(THEME.status_busy)

    def _apply_ocr_result(self) -> None:
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
            # The frame was cleared while OCR ran; there is nothing to draw,
            # but the busy indicator must still be released.
            self._set_status("Detection discarded", THEME.neutral)
            self.status_led.set_color(THEME.status_ready)
            return
        self._render_current_frame()
        self._update_text_output()
        self._add_to_history()
        self._set_status("Detection complete", THEME.status_ready)
        self.status_led.set_color(THEME.status_ready)

    def update_frame(self) -> None:
        """Drive the preview and collect OCR results, on the Tk thread.

        The preview is refreshed on every tick so the video stays smooth;
        only the OCR pass is throttled by frame_skip, since it is what
        costs time.
        """
        self._drain_ocr_results()
        if self.capture_active and self._grabber is not None:
            frame = self._grabber.read()
            if frame is not None:
                self.current_frame = frame
                self.frame_counter += 1
                # Redraw with the boxes from the last completed pass, so
                # they stay visible between OCR runs instead of flashing.
                self._render_current_frame()
                if self.frame_counter % SETTINGS.frame_skip == 0:
                    self._process_current_frame()
        self.root.after(33, self.update_frame)

    def _drain_ocr_results(self) -> None:
        """Apply any OCR result the worker has finished.

        Tk is not thread-safe, so the worker never calls into the GUI --
        not even through after(). It publishes to the engine's queue and
        the main loop picks the result up here.
        """
        result = self.engine.poll_result()
        if result is None:
            return
        self.ocr_result = result
        self._apply_ocr_result()

    def _show_image(self, frame: np.ndarray) -> None:
        """Display a full frame, scaled to fit, and record its placement.

        The recorded geometry is what lets pointer coordinates be converted
        back into frame coordinates for ROI selection.
        """
        height, width = frame.shape[:2]
        geometry = compute_display_geometry(
            width,
            height,
            self.image_label.winfo_width(),
            self.image_label.winfo_height(),
        )
        # Scale before converting to PIL: PIL's resize costs ~19 ms on a
        # 720p frame against ~1 ms for OpenCV's, and the whole preview tick
        # has to fit in 33 ms. Still images can afford the better filter.
        scaled = resize_frame_for_display(frame, geometry, smooth=not self.capture_active)
        pil_image = Image.fromarray(bgr_to_rgb(scaled))
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
                self.text_output.insert(tk.END, f"{text} ({format_confidence(confidence)})\n")
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
        if self.picker_mode:
            self._toggle_picker_mode()
        self._reset_color_readout()
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
            f"Text Detection App v{__version__}\n\n"
            "Modern dark-themed interface.\n"
            "Developed using OpenCV, EasyOCR, and Tkinter.\n\n"
            "Features: threaded capture and OCR, multi-language, GPU support,\n"
            "ROI selection, export to TXT/CSV/JSON, searchable history.",
        )

    def on_closing(self) -> None:
        self._save_settings()
        self._flush_settings_write()
        self._cancel_resize()
        self.stop_capture()
        self.engine.shutdown()
        self.root.destroy()
        logger.info("Application closed")

    def _toggle_roi_mode(self) -> None:
        if not self.roi_mode and self.picker_mode:
            # Both modes claim <Button-1> on the image label, so only one
            # of them can be live at a time.
            self._toggle_picker_mode()

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

    def _toggle_picker_mode(self) -> None:
        if not self.picker_mode:
            if self.current_frame is None:
                self._set_status(
                    "No image to sample - load one or start the webcam",
                    THEME.status_error,
                )
                return
            if self.roi_mode:
                self._toggle_roi_mode()

        self.picker_mode = not self.picker_mode
        if self.picker_mode:
            self.picker_btn.config(text="✓ Picker Active")
            self.picker_btn.set_base_bg(THEME.success, THEME.success_active)
            self._set_status("Picker mode: click on image", THEME.accent)
            self.image_label.bind("<Button-1>", self._on_pick_click)
        else:
            self.picker_btn.config(text="🎨 Pick Color")
            self.picker_btn.set_base_bg(THEME.about, THEME.about_active)
            self.image_label.unbind("<Button-1>")

    def _on_pick_click(self, event: tk.Event) -> None:
        """Read the color under a click and report it in the sidebar."""
        if self.current_frame is None or self._display_geometry is None:
            return

        height, width = self.current_frame.shape[:2]
        x, y = widget_to_frame_point(event.x, event.y, self._display_geometry, width, height)
        # Sampled from the stored frame, never from the annotated copy on
        # screen: a click on a detection box would otherwise report the
        # color of the box rather than of the image.
        sample = sample_color(self.current_frame, x, y)
        if sample is None:
            return

        self.color_sample = sample
        self._update_color_readout(sample)
        self._set_status(f"{sample.name} {sample.hex}", THEME.status_ready)
        logger.info("Sampled color %s at (%d, %d)", sample.hex, x, y)

        if self.picker_mode:
            self._toggle_picker_mode()

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
