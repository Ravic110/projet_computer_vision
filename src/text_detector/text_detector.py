"""GUI application for real-time text recognition."""

import contextlib
import csv
import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from text_detector.config import SETTINGS, THEME
from text_detector.image_processor import bgr_to_rgb, draw_boxes_with_colors
from text_detector.ocr_engine import DetectionResult, OCREngine
from text_detector.settings_manager import SettingsManager
from text_detector.utils.logging_setup import get_logger
from text_detector.utils.path_helpers import get_assets_dir
from text_detector.widgets import StatusLED, ThemeableButton

logger = get_logger("gui")


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
        SETTINGS.default_language = loaded.default_language
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
        self.language_var = tk.StringVar(value=SETTINGS.default_language)
        self.threshold_var = tk.DoubleVar(value=SETTINGS.default_confidence)
        self.frame_skip_var = tk.IntVar(value=SETTINGS.frame_skip)
        self.gpu_var = tk.BooleanVar(value=SETTINGS.gpu_enabled)
        self.preprocess_var = tk.BooleanVar(value=SETTINGS.preprocess_enabled)
        self.history: list[dict[str, str | float]] = []
        self.current_language: str = SETTINGS.default_language

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
        frame = tk.Frame(self.sidebar, bg=THEME.surface)
        frame.pack(fill="x", pady=2)

        tk.Label(
            frame,
            text="Language",
            font=("Arial", 9),
            bg=THEME.surface,
            fg=THEME.text_muted,
        ).pack(side="left", padx=(4, 0))

        self.language_menu = tk.OptionMenu(
            frame,
            self.language_var,
            *SETTINGS.available_languages,
            command=self._language_changed,  # type: ignore[arg-type]
        )
        self.language_menu.config(
            font=("Arial", 10),
            bg=THEME.surface_light,
            fg=THEME.text_fg,
            activebackground=THEME.accent,
            activeforeground=THEME.button_fg,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.language_menu["menu"].config(
            bg=THEME.surface_light,
            fg=THEME.text_fg,
            activebackground=THEME.accent,
            activeforeground=THEME.button_fg,
        )
        self.language_menu.pack(side="right", fill="x", expand=True, padx=4)

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

    # ── Image area ──────────────────────────────────────────────────

    def _create_image_area(self) -> None:
        self.image_container = tk.Frame(
            self.center_paned,
            bg=THEME.result_frame_bg,
            bd=2,
            relief="flat",
        )
        self.center_paned.add(self.image_container)

        self.image_label = tk.Label(
            self.image_container,
            bg=THEME.result_frame_bg,
            fg=THEME.text_muted,
            font=("Arial", 14),
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

        self.text_wrapper = tk.Frame(self.text_panel, bg=THEME.text_output_bg)
        self.text_wrapper.pack(fill="both", expand=True, padx=8, pady=8)

        self.text_scrollbar = tk.Scrollbar(
            self.text_wrapper,
            orient="vertical",
            bg=THEME.scrollbar_bg,
            troughcolor=THEME.scrollbar_bg,
            activebackground=THEME.scrollbar_fg,
            highlightthickness=0,
            borderwidth=0,
        )
        self.text_scrollbar.pack(side="right", fill="y")

        self.text_output = tk.Text(
            self.text_wrapper,
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

    def _bind_keyboard_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _: self.load_image())
        self.root.bind("<Control-s>", lambda _: self.save_results())
        self.root.bind("<Control-c>", lambda _: self._copy_to_clipboard())
        self.root.bind("<Control-v>", lambda _: self._paste_image_from_clipboard())
        self.root.bind("<space>", lambda _: self._toggle_capture())
        self.root.bind("<Control-r>", lambda _: self._reset_settings())
        self.root.bind("<Escape>", lambda _: self.clear_results())

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
        """Persist current settings to disk."""
        SETTINGS.default_language = self.language_var.get()
        SETTINGS.default_confidence = self.threshold_var.get()
        SETTINGS.frame_skip = self.frame_skip_var.get()
        SETTINGS.gpu_enabled = self.gpu_var.get()
        SETTINGS.preprocess_enabled = self.preprocess_var.get()
        self._settings_manager.save(SETTINGS)

    def _language_changed(self, value: str) -> None:
        self.language_var.set(value)
        self.current_language = value
        self.engine.clear_cache()
        self._set_status(f"Language: {value}", THEME.accent)
        logger.info("Language changed to %s", value)
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
        """Reset settings to defaults and reload UI."""
        self._settings_manager.reset()
        defaults = self._settings_manager.load()
        SETTINGS.default_language = defaults.default_language
        SETTINGS.default_confidence = defaults.default_confidence
        SETTINGS.gpu_enabled = defaults.gpu_enabled
        SETTINGS.preprocess_enabled = defaults.preprocess_enabled
        SETTINGS.frame_skip = defaults.frame_skip
        SETTINGS.ocr_max_width = defaults.ocr_max_width
        SETTINGS.paragraph_merge = defaults.paragraph_merge
        self.language_var.set(SETTINGS.default_language)
        self.threshold_var.set(SETTINGS.default_confidence)
        self.gpu_var.set(SETTINGS.gpu_enabled)
        self.preprocess_var.set(SETTINGS.preprocess_enabled)
        self.current_language = SETTINGS.default_language
        self.threshold_label.config(text=f"{SETTINGS.default_confidence:.2f}")
        self.engine = OCREngine(SETTINGS)
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
        if self.current_frame is None:
            return

        if self.engine.is_busy:
            return

        def _on_result(result: DetectionResult) -> None:
            with self.ocr_lock:
                self.ocr_result = result
            self.root.after(0, self._apply_ocr_result)

        queued = self.engine.detect_text_async(
            self.current_frame,
            languages=[self.current_language],
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
            self.detected_text = self.ocr_result.detections

        if self.current_frame is None:
            return
        frame = draw_boxes_with_colors(self.current_frame.copy(), self.detected_text)
        self._show_image(frame)
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
        image = bgr_to_rgb(frame)
        pil_image = Image.fromarray(image)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=tk_image, text="")
        self.image_label.image = tk_image  # type: ignore[attr-defined]

    def _update_text_output(self) -> None:
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        if self.detected_text:
            for _bbox, text, confidence in self.detected_text:
                self.text_output.insert(tk.END, f"{text} ({confidence:.2f})\n")
        else:
            self.text_output.insert(tk.END, "No text detected.\n")
        self.text_output.config(state="disabled")

    def _add_to_history(self) -> None:
        for _bbox, text, confidence in self.detected_text:
            self.history.append(
                {
                    "text": text,
                    "confidence": round(confidence, 2),
                    "language": self.current_language,
                }
            )
        if len(self.history) > SETTINGS.max_history:
            self.history = self.history[-SETTINGS.max_history :]

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
                f.write(f"{text} (confidence: {confidence:.2f})\n")

    def _save_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "confidence", "language"])
            for _bbox, text, confidence in self.detected_text:
                writer.writerow([text, f"{confidence:.2f}", self.current_language])

    def _save_json(self, path: Path) -> None:
        data = [
            {"text": text, "confidence": round(confidence, 2), "language": self.current_language}
            for _bbox, text, confidence in self.detected_text
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_results(self) -> None:
        self.stop_capture()
        self.current_frame = None
        self.detected_text = []
        self.ocr_result = None
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
        self.capture_active = False
        if self.cap is not None:
            self.cap.release()
        self.engine.shutdown()
        self.root.destroy()
        logger.info("Application closed")

    # ── Helpers ─────────────────────────────────────────────────────

    def _set_status(self, text: str, bg_color: str) -> None:
        self.status_label.config(text=text, fg=bg_color)
