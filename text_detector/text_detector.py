import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np


class TextRecognitionApp:
    """A GUI application for real-time text recognition using EasyOCR."""

    def __init__(self, main):
        self.cap = None
        self.root = main
        self.root.title("Text Recognition App")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#F0F0F0")

        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.icon_path = os.path.join(self.assets_dir, "icon.ico")
        self.results_file = os.path.join(self.assets_dir, "detected_text_results.txt")

        self.capture_active = False
        self.frame_counter = 0
        self.detected_text = []
        self.current_frame = None
        self.language_var = tk.StringVar(value="en")
        self.threshold_var = tk.DoubleVar(value=0.25)
        self.available_languages = ["en", "fr", "de", "es", "it", "pt"]

        self._create_widgets()
        self._configure_icon()
        self.reader = easyocr.Reader([self.language_var.get()], gpu=False)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def _configure_icon(self):
        if os.path.exists(self.icon_path):
            try:
                self.root.iconbitmap(self.icon_path)
            except tk.TclError:
                pass

    def _create_widgets(self):
        self.main_frame = tk.Frame(self.root, bg="#FFFFFF", bd=2, relief="groove")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.title_label = tk.Label(
            self.main_frame,
            text="Text Recognition",
            font=("Arial", 24, "bold"),
            bg="#63BFF3",
            fg="#FFFFFF",
            pady=10,
        )
        self.title_label.pack(fill="x")

        self.control_frame = tk.Frame(self.main_frame, bg="#F0F0F0")
        self.control_frame.pack(fill="x", pady=10)

        self.start_button = tk.Button(
            self.control_frame,
            text="Start Webcam",
            command=self.start_capture,
            font=("Arial", 12, "bold"),
            bg="#5AE3B1",
            fg="#FFFFFF",
            activebackground="#4AA57C",
            relief="ridge",
            padx=10,
            pady=5,
        )
        self.start_button.pack(side="left", padx=5)

        self.stop_button = tk.Button(
            self.control_frame,
            text="Stop Capture",
            command=self.stop_capture,
            font=("Arial", 12, "bold"),
            bg="#CA3074",
            fg="#FFFFFF",
            activebackground="#A02458",
            relief="ridge",
            padx=10,
            pady=5,
        )
        self.stop_button.pack(side="left", padx=5)

        self.load_button = tk.Button(
            self.control_frame,
            text="Load Image",
            command=self.load_image,
            font=("Arial", 12, "bold"),
            bg="#4A90E2",
            fg="#FFFFFF",
            activebackground="#3A70B2",
            relief="ridge",
            padx=10,
            pady=5,
        )
        self.load_button.pack(side="left", padx=5)

        self.save_button = tk.Button(
            self.control_frame,
            text="Save Results",
            command=self.save_results,
            font=("Arial", 12, "bold"),
            bg="#FFA500",
            fg="#FFFFFF",
            activebackground="#CC8400",
            relief="ridge",
            padx=10,
            pady=5,
        )
        self.save_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(
            self.control_frame,
            text="Clear",
            command=self.clear_results,
            font=("Arial", 12, "bold"),
            bg="#888888",
            fg="#FFFFFF",
            activebackground="#666666",
            relief="ridge",
            padx=10,
            pady=5,
        )
        self.clear_button.pack(side="left", padx=5)

        self.about_button = tk.Button(
            self.control_frame,
            text="About",
            command=self.show_about,
            font=("Arial", 12, "bold"),
            bg="#6D6DFF",
            fg="#FFFFFF",
            activebackground="#5757D0",
            relief="ridge",
            padx=10,
            pady=5,
        )
        self.about_button.pack(side="right", padx=5)

        self.settings_frame = tk.Frame(self.main_frame, bg="#F0F0F0")
        self.settings_frame.pack(fill="x", pady=5)

        tk.Label(self.settings_frame, text="OCR Language:", bg="#F0F0F0").pack(side="left", padx=(5, 0))
        self.language_menu = tk.OptionMenu(
            self.settings_frame, self.language_var, *self.available_languages, command=self._language_changed
        )
        self.language_menu.pack(side="left", padx=5)

        tk.Label(self.settings_frame, text="Confidence threshold:", bg="#F0F0F0").pack(side="left", padx=(10, 0))
        self.threshold_scale = tk.Scale(
            self.settings_frame,
            variable=self.threshold_var,
            from_=0.05,
            to=0.8,
            resolution=0.05,
            orient="horizontal",
            length=220,
            bg="#F0F0F0",
        )
        self.threshold_scale.pack(side="left", padx=5)

        self.result_frame = tk.Frame(self.main_frame, bg="#DDEEFF", bd=2, relief="sunken")
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = tk.Label(self.result_frame, bg="#DDEEFF")
        self.image_label.pack(side="left", fill="both", expand=True)

        self.text_frame = tk.Frame(self.result_frame, bg="#FFFFFF", bd=1, relief="sunken")
        self.text_frame.pack(side="right", fill="y", padx=5, pady=5)

        self.text_title = tk.Label(
            self.text_frame,
            text="Detected Text",
            font=("Arial", 14, "bold"),
            bg="#4A90E2",
            fg="#FFFFFF",
            padx=10,
            pady=10,
        )
        self.text_title.pack(fill="x")

        self.text_output = tk.Text(self.text_frame, wrap="word", width=35, bg="#F7F7F7")
        self.text_output.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.text_output.config(state="disabled")

        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10, "italic"),
            bg="#63BFF3",
            fg="#FFFFFF",
            relief="ridge",
            anchor="w",
        )
        self.status_bar.pack(side="bottom", fill="x")

    def _language_changed(self, value):
        self.language_var.set(value)
        self.reader = easyocr.Reader([value], gpu=False)
        self.status_bar.config(text=f"Language set to {value}", bg="#63BFF3")

    def start_capture(self):
        if self.capture_active:
            return
        self.capture_active = True
        self.frame_counter = 0
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Unable to open webcam.")
            self.status_bar.config(text="Webcam capture started", bg="#5AE3B1")
        except Exception as exc:
            self.capture_active = False
            messagebox.showerror("Webcam Error", str(exc))
            self.status_bar.config(text="Failed to start webcam", bg="#CA3074")

    def stop_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.capture_active = False
        self.status_bar.config(text="Capture stopped", bg="#CA3074")

    def load_image(self):
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
        self.status_bar.config(text=f"Loaded image: {os.path.basename(path)}", bg="#4A90E2")

    def _process_current_frame(self):
        if self.current_frame is None:
            return

        self.detected_text = self.filter_text(self.reader.readtext(self.current_frame))
        frame = self.draw_boxes_with_colors(self.current_frame.copy(), self.detected_text)
        self._show_image(frame)
        self._update_text_output()

    def update_frame(self):
        if self.capture_active and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.frame_counter += 1
                if self.frame_counter % 5 == 0:
                    self.detected_text = self.filter_text(self.reader.readtext(frame))
                    self._update_text_output()

                frame = self.draw_boxes_with_colors(frame.copy(), self.detected_text)
                self._show_image(frame)

        self.root.after(50, self.update_frame)

    def filter_text(self, text_data):
        threshold = self.threshold_var.get()
        return [item for item in text_data if item[2] >= threshold]

    def draw_boxes_with_colors(self, frame, detections):
        for bbox, text, confidence in detections:
            points = [(int(pt[0]), int(pt[1])) for pt in bbox]
            x1, y1 = max(0, points[0][0]), max(0, points[0][1])
            x2, y2 = max(0, points[2][0]), max(0, points[2][1])
            roi = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
            avg_color = cv2.mean(roi)[:3] if roi is not None else (0, 0, 0)

            cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                frame,
                f"{text} ({confidence:.2f})",
                (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    def _show_image(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

    def _update_text_output(self):
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        if self.detected_text:
            for bbox, text, confidence in self.detected_text:
                self.text_output.insert(tk.END, f"{text} (confidence: {confidence:.2f})\n")
        else:
            self.text_output.insert(tk.END, "No text detected yet.\n")
        self.text_output.config(state="disabled")

    def save_results(self):
        if not self.detected_text:
            messagebox.showwarning("Warning", "No detected text to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save detected text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="detected_text_results.txt",
        )
        if not save_path:
            return

        with open(save_path, "w", encoding="utf-8") as f:
            for bbox, text, confidence in self.detected_text:
                f.write(f"{text} (confidence: {confidence:.2f})\n")

        messagebox.showinfo("Saved", f"Results saved to {save_path}")

    def clear_results(self):
        self.stop_capture()
        self.current_frame = None
        self.detected_text = []
        self.image_label.config(image=None)
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, "Ready. Load an image or start the webcam.")
        self.text_output.config(state="disabled")
        self.status_bar.config(text="Cleared", bg="#888888")

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Text Recognition App\n\nDeveloped using OpenCV, EasyOCR, and Tkinter.\n"
            "Load images or use your webcam to detect text in real time.\n"
            "Use the language selector and threshold slider to improve accuracy.",
        )

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()
