"""GUI application for real-time text recognition."""

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
from text_detector.utils.logging_setup import get_logger
from text_detector.utils.path_helpers import get_assets_dir

logger = get_logger("gui")


class TextRecognitionApp:
    """A GUI application for real-time text recognition using EasyOCR."""

    def __init__(self, main: tk.Tk) -> None:
        self.cap: cv2.VideoCapture | None = None
        self.root = main
        self.root.title("Text Recognition App")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        self.root.configure(bg=THEME.background)

        self.capture_active = False
        self.frame_counter = 0
        self.detected_text: list[tuple] = []
        self.current_frame: np.ndarray | None = None
        self.language_var = tk.StringVar(value=SETTINGS.default_language)
        self.threshold_var = tk.DoubleVar(value=SETTINGS.default_confidence)
        self.gpu_var = tk.BooleanVar(value=SETTINGS.gpu_enabled)
        self.history: list[dict[str, str | float]] = []
        self.current_language: str = SETTINGS.default_language

        self.engine = OCREngine(SETTINGS)
        self.ocr_result: DetectionResult | None = None
        self.ocr_lock = threading.Lock()

        self._create_widgets()
        self._configure_icon()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def _configure_icon(self) -> None:
        icon_path = get_assets_dir() / "icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except tk.TclError:
                pass

    def _create_widgets(self) -> None:
        self.main_frame = tk.Frame(self.root, bg=THEME.frame_bg, bd=2, relief="groove")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.title_label = tk.Label(
            self.main_frame,
            text="Text Recognition",
            font=("Arial", 24, "bold"),
            bg=THEME.title_bg,
            fg=THEME.title_fg,
            pady=10,
        )
        self.title_label.pack(fill="x")

        self._create_control_frame()
        self._create_settings_frame()
        self._create_result_frame()

        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10, "italic"),
            bg=THEME.title_bg,
            fg=THEME.title_fg,
            relief="ridge",
            anchor="w",
        )
        self.status_bar.pack(side="bottom", fill="x")

    def _create_control_frame(self) -> None:
        self.control_frame = tk.Frame(self.main_frame, bg=THEME.background)
        self.control_frame.pack(fill="x", pady=10)

        buttons = [
            ("Start Webcam", self.start_capture, THEME.success, THEME.success_active),
            ("Stop Capture", self.stop_capture, THEME.danger, THEME.danger_active),
            ("Load Image", self.load_image, THEME.primary, THEME.primary_active),
            ("Save Results", self.save_results, THEME.warning, THEME.warning_active),
            ("Clear", self.clear_results, THEME.neutral, THEME.neutral_active),
        ]

        for text, cmd, bg, active in buttons:
            btn = tk.Button(
                self.control_frame,
                text=text,
                command=cmd,
                font=("Arial", 12, "bold"),
                bg=bg,
                fg=THEME.white,
                activebackground=active,
                relief="ridge",
                padx=10,
                pady=5,
            )
            btn.pack(side="left", padx=5)

        about_btn = tk.Button(
            self.control_frame,
            text="About",
            command=self.show_about,
            font=("Arial", 12, "bold"),
            bg=THEME.about,
            fg=THEME.white,
            activebackground=THEME.about_active,
            relief="ridge",
            padx=10,
            pady=5,
        )
        about_btn.pack(side="right", padx=5)

    def _create_settings_frame(self) -> None:
        self.settings_frame = tk.Frame(self.main_frame, bg=THEME.background)
        self.settings_frame.pack(fill="x", pady=5)

        tk.Label(self.settings_frame, text="OCR Language:", bg=THEME.background).pack(side="left", padx=(5, 0))
        self.language_menu = tk.OptionMenu(
            self.settings_frame,
            self.language_var,
            *SETTINGS.available_languages,
            command=self._language_changed,
        )
        self.language_menu.pack(side="left", padx=5)

        tk.Label(self.settings_frame, text="Confidence threshold:", bg=THEME.background).pack(side="left", padx=(10, 0))
        self.threshold_scale = tk.Scale(
            self.settings_frame,
            variable=self.threshold_var,
            from_=SETTINGS.min_confidence,
            to=SETTINGS.max_confidence,
            resolution=SETTINGS.confidence_resolution,
            orient="horizontal",
            length=220,
            bg=THEME.background,
        )
        self.threshold_scale.pack(side="left", padx=5)

        self.gpu_check = tk.Checkbutton(
            self.settings_frame,
            text="Use GPU",
            variable=self.gpu_var,
            command=self._gpu_changed,
            bg=THEME.background,
        )
        self.gpu_check.pack(side="left", padx=10)

    def _create_result_frame(self) -> None:
        self.result_frame = tk.Frame(self.main_frame, bg=THEME.result_frame_bg, bd=2, relief="sunken")
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = tk.Label(self.result_frame, bg=THEME.result_frame_bg)
        self.image_label.pack(side="left", fill="both", expand=True)

        self.text_frame = tk.Frame(self.result_frame, bg=THEME.white, bd=1, relief="sunken")
        self.text_frame.pack(side="right", fill="y", padx=5, pady=5)

        self.text_title = tk.Label(
            self.text_frame,
            text="Detected Text",
            font=("Arial", 14, "bold"),
            bg=THEME.primary,
            fg=THEME.white,
            padx=10,
            pady=10,
        )
        self.text_title.pack(fill="x")

        self.text_output = tk.Text(self.text_frame, wrap="word", width=35, bg=THEME.text_output_bg)
        self.text_output.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.text_output.config(state="disabled")

    def _language_changed(self, value: str) -> None:
        self.language_var.set(value)
        self.current_language = value
        self.engine.clear_cache()
        self.status_bar.config(text=f"Language set to {value}", bg=THEME.title_bg)
        logger.info("Language changed to %s", value)

    def _gpu_changed(self) -> None:
        self.engine.clear_cache()
        self.engine._settings.gpu_enabled = self.gpu_var.get()
        status = "enabled" if self.gpu_var.get() else "disabled"
        self.status_bar.config(text=f"GPU {status} (model will reload)", bg=THEME.title_bg)
        logger.info("GPU %s", status)

    def start_capture(self) -> None:
        if self.capture_active:
            return
        self.capture_active = True
        self.frame_counter = 0
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Unable to open webcam.")
            self.status_bar.config(text="Webcam capture started", bg=THEME.success)
            logger.info("Webcam capture started")
        except Exception as exc:
            self.capture_active = False
            messagebox.showerror("Webcam Error", str(exc))
            self.status_bar.config(text="Failed to start webcam", bg=THEME.danger)
            logger.error("Webcam error: %s", exc)

    def stop_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.capture_active = False
        self.status_bar.config(text="Capture stopped", bg=THEME.danger)
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
        self.status_bar.config(text=f"Loaded image: {Path(path).name}", bg=THEME.primary)
        logger.info("Loaded image: %s", path)

    def _process_current_frame(self) -> None:
        if self.current_frame is None:
            return

        def _on_result(result: DetectionResult) -> None:
            with self.ocr_lock:
                self.ocr_result = result
            self.root.after(0, self._apply_ocr_result)

        self.engine.detect_text_async(
            self.current_frame,
            languages=[self.current_language],
            threshold=self.threshold_var.get(),
            callback=_on_result,
        )
        self.status_bar.config(text="Processing...", bg=THEME.warning)

    def _apply_ocr_result(self) -> None:
        with self.ocr_lock:
            if self.ocr_result is None or not self.ocr_result.success:
                self.status_bar.config(text="OCR failed", bg=THEME.danger)
                return
            self.detected_text = self.ocr_result.detections

        frame = draw_boxes_with_colors(self.current_frame.copy(), self.detected_text)
        self._show_image(frame)
        self._update_text_output()
        self._add_to_history()
        self.status_bar.config(text="Detection complete", bg=THEME.success)

    def update_frame(self) -> None:
        if self.capture_active and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.frame_counter += 1
                if self.frame_counter % SETTINGS.frame_skip == 0:
                    self._process_current_frame()

        self.root.after(50, self.update_frame)

    def _show_image(self, frame: np.ndarray) -> None:
        image = bgr_to_rgb(frame)
        pil_image = Image.fromarray(image)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

    def _update_text_output(self) -> None:
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        if self.detected_text:
            for bbox, text, confidence in self.detected_text:
                self.text_output.insert(tk.END, f"{text} (confidence: {confidence:.2f})\n")
        else:
            self.text_output.insert(tk.END, "No text detected yet.\n")
        self.text_output.config(state="disabled")

    def _add_to_history(self) -> None:
        for bbox, text, confidence in self.detected_text:
            self.history.append(
                {"text": text, "confidence": round(confidence, 2), "language": self.current_language}
            )
        if len(self.history) > SETTINGS.max_history:
            self.history = self.history[-SETTINGS.max_history:]

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
            for bbox, text, confidence in self.detected_text:
                f.write(f"{text} (confidence: {confidence:.2f})\n")

    def _save_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "confidence", "language"])
            for bbox, text, confidence in self.detected_text:
                writer.writerow([text, f"{confidence:.2f}", self.current_language])

    def _save_json(self, path: Path) -> None:
        data = [
            {"text": text, "confidence": round(confidence, 2), "language": self.current_language}
            for bbox, text, confidence in self.detected_text
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_results(self) -> None:
        self.stop_capture()
        self.current_frame = None
        self.detected_text = []
        self.ocr_result = None
        self.image_label.config(image=None)
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, "Ready. Load an image or start the webcam.")
        self.text_output.config(state="disabled")
        self.status_bar.config(text="Cleared", bg=THEME.neutral)

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Text Recognition App v2.0\n\n"
            "Developed using OpenCV, EasyOCR, and Tkinter.\n"
            "Load images or use your webcam to detect text in real time.\n"
            "Use the language selector, threshold slider, and GPU toggle to improve accuracy.",
        )

    def on_closing(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.engine.clear_cache()
        self.root.destroy()
        logger.info("Application closed")