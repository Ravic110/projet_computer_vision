import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


class TextRecognitionApp:
    def __init__(self, main):
        self.cap = None
        self.root = main
        self.root.title("Text Recognition")

        # Interface utilisateur
        self.main_frame = tk.Frame(self.root, bg="white", padx=10, pady=10)
        self.main_frame.pack()

        self.title_label = tk.Label(self.main_frame, text="Text Recognition", font=("Arial", 18), bg="#63BFF3")
        self.title_label.pack()

        self.espacement = tk.Frame(self.main_frame, pady=8, padx=5, bg="white")
        self.espacement.pack()

        self.start_button = tk.Button(
            self.espacement, text="Start", command=self.start_capture, padx=12, pady=7, bd=3,
            font=("Arial", 10, "bold"), background="#5AE3B1", relief="groove"
        )
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(
            self.espacement, text="Stop", command=self.stop_capture, padx=12, pady=7, bd=3,
            font=("Arial", 10, "bold"), relief="groove", background="#CA3074"
        )
        self.stop_button.pack(side=tk.RIGHT)

        self.video_source = tk.StringVar()
        self.video_source.set("Webcam")

        self.source_select = tk.OptionMenu(self.main_frame, self.video_source, "Webcam", "Choose File...")
        self.source_select.pack()

        self.result_frame = tk.Frame(self.main_frame, bg="#5AE3B1", padx=4, pady=5)
        self.result_frame.pack()

        self.image_label = tk.Label(self.result_frame)
        self.image_label.pack()

        self.state = tk.Frame(self.main_frame, pady=8, padx=5, bg="white")
        self.state.pack()
        self.status_bar = tk.Label(self.state, text="Start capture", font=("Arial", 12), bg="#5AE3B1")
        self.status_bar.pack()

        # Initialisation de la reconnaissance
        self.reader = easyocr.Reader(['en', 'fr'], gpu=True)
        self.threshold = 0.25

        self.frame_counter = 0
        self.detected_text = []

        self.update_frame()

        # Fermeture de la fenêtre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_video_source(self):
        try:
            if self.video_source.get() == "Webcam":
                self.cap = cv2.VideoCapture(0)
            elif self.video_source.get() == "Choose File...":
                file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
                if file_path:
                    self.cap = cv2.VideoCapture(file_path)
                else:
                    raise Exception("No file selected.")
            if not self.cap or not self.cap.isOpened():
                raise Exception("Unable to open video source.")
        except Exception as e:
            print(f"Error: {e}")
            self.status_bar.config(text=f"Error: {e}", background="#CA3074")

    def start_capture(self):
        self.get_video_source()
        self.status_bar.config(text="Capturing started", background="#5AE3B1")
        self.result_frame.config(background="#5AE3B1")

    def stop_capture(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.status_bar.config(text="Capturing stopped", background="#CA3074")
        self.result_frame.config(background="#CA3074")
        self.image_label.config(image=None)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret:
                self.frame_counter += 1
                if self.frame_counter % 5 == 0:  # Process every 5th frame
                    text = self.reader.readtext(frame)
                    self.detected_text = self.filter_text(text)

                frame = self.draw_boxes_with_colors(frame, self.detected_text)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                self.image_label.config(image=image)
                self.image_label.image = image

        self.root.after(10, self.update_frame)

    def filter_text(self, text):
        return [t for t in text if t[2] > self.threshold]

    def draw_boxes_with_colors(self, frame, text):
        for detection in text:
            bbox, detected_text, confidence = detection
            bbox = [(int(pt[0]), int(pt[1])) for pt in bbox]

            # Calculate color of the text region
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            roi = frame[y1:y2, x1:x2]  # Region of interest
            avg_color = cv2.mean(roi)[:3]  # Average color (BGR)

            # Draw the bounding box
            cv2.polylines(frame, [np.array(bbox, dtype=np.int32)], isClosed=True, color=(0, 150, 0), thickness=2)
            cv2.putText(
                frame, f"{detected_text} ({int(avg_color[2])},{int(avg_color[1])},{int(avg_color[0])})",
                bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
        return frame

    def on_closing(self):
        print("Closing the application")
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()
