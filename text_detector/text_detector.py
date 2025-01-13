import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os  # Pour gérer les chemins
import numpy as np


class TextRecognitionApp:
    def __init__(self, main):
        self.cap = None
        self.root = main
        self.root.title("Text Recognition App")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        self.root.configure(bg="#F0F0F0")

        # Gestion des chemins
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.icon_path = os.path.join(self.assets_dir, "icon.ico")
        self.results_file = os.path.join(self.assets_dir, "detected_text_results.txt")

        # Vérifier si les fichiers nécessaires existent
        if not os.path.exists(self.icon_path):
            messagebox.showerror("Error", "Icon file not found in the 'assets' folder.")
            self.root.destroy()
            return

        # Ajout d'une icône personnalisée
        self.root.iconbitmap(self.icon_path)

        # Cadre principal
        self.main_frame = tk.Frame(self.root, bg="#FFFFFF", bd=2, relief="groove")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Titre
        self.title_label = tk.Label(
            self.main_frame, text="Text Recognition", font=("Arial", 24, "bold"), bg="#63BFF3", fg="#FFFFFF", pady=10
        )
        self.title_label.pack(fill="x")

        # Boutons de contrôle
        self.control_frame = tk.Frame(self.main_frame, bg="#F0F0F0")
        self.control_frame.pack(fill="x", pady=10)

        self.start_button = tk.Button(
            self.control_frame, text="Start Capture", command=self.start_capture, font=("Arial", 12, "bold"),
            bg="#5AE3B1", fg="#FFFFFF", activebackground="#4AA57C", relief="ridge", padx=10, pady=5
        )
        self.start_button.pack(side="left", padx=5)

        self.stop_button = tk.Button(
            self.control_frame, text="Stop Capture", command=self.stop_capture, font=("Arial", 12, "bold"),
            bg="#CA3074", fg="#FFFFFF", activebackground="#A02458", relief="ridge", padx=10, pady=5
        )
        self.stop_button.pack(side="left", padx=5)

        self.save_button = tk.Button(
            self.control_frame, text="Save Results", command=self.save_results, font=("Arial", 12, "bold"),
            bg="#FFA500", fg="#FFFFFF", activebackground="#CC8400", relief="ridge", padx=10, pady=5
        )
        self.save_button.pack(side="left", padx=5)

        self.about_button = tk.Button(
            self.control_frame, text="About", command=self.show_about, font=("Arial", 12, "bold"),
            bg="#4A90E2", fg="#FFFFFF", activebackground="#3A70B2", relief="ridge", padx=10, pady=5
        )
        self.about_button.pack(side="right", padx=5)

        # Vidéo et résultats
        self.result_frame = tk.Frame(self.main_frame, bg="#DDEEFF", bd=2, relief="sunken")
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = tk.Label(self.result_frame, bg="#DDEEFF")
        self.image_label.pack(fill="both", expand=True)

        # Barre de statut
        self.status_bar = tk.Label(
            self.root, text="Ready", font=("Arial", 10, "italic"), bg="#63BFF3", fg="#FFFFFF", relief="ridge", anchor="w"
        )
        self.status_bar.pack(side="bottom", fill="x")

        # Initialisation EasyOCR
        self.reader = easyocr.Reader(['en', 'fr'], gpu=True)
        self.threshold = 0.25
        self.frame_counter = 0
        self.detected_text = []

        # Mise à jour des frames
        self.update_frame()

        # Gestion de la fermeture
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_capture(self):
        self.get_video_source()
        self.status_bar.config(text="Capturing started", bg="#5AE3B1")

    def stop_capture(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.status_bar.config(text="Capture stopped", bg="#CA3074")
        self.image_label.config(image=None)

    def get_video_source(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                raise Exception("Unable to open video source.")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")
            self.status_bar.config(text="Error opening video source", bg="#CA3074")

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

            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            roi = frame[y1:y2, x1:x2]
            avg_color = cv2.mean(roi)[:3]

            cv2.polylines(frame, [np.array(bbox, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                frame, f"{detected_text} ({int(avg_color[2])},{int(avg_color[1])},{int(avg_color[0])})",
                bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        return frame

    def save_results(self):
        if self.detected_text:
            with open(self.results_file, "w") as f:
                for detection in self.detected_text:
                    f.write(f"{detection[1]} (Confidence: {detection[2]:.2f})\n")
            messagebox.showinfo("Success", f"Results saved successfully in {self.results_file}.")
        else:
            messagebox.showwarning("Warning", "No results to save.")

    def show_about(self):
        messagebox.showinfo(
            "About", "Text Recognition App\n\nDeveloped using OpenCV, EasyOCR, and Tkinter.\n"
                     "Detect and recognize text in real-time with added color detection."
        )

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()
