import cv2
import easyocr
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageOps


class TextRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance de Texte")

        # Interface
        self.main_frame = tk.Frame(self.root, bg="white", padx=10, pady=10)
        self.main_frame.pack()

        self.title_label = tk.Label(self.main_frame, text="Reconnaissance de Texte", font=("Helvetica", 18, "bold"), bg="lightblue")
        self.title_label.pack()

        # Source vidéo
        self.source_frame = tk.Frame(self.main_frame, bg="white", padx=10, pady=10)
        self.source_frame.pack()

        self.source_label = tk.Label(self.source_frame, text="Source vidéo:", font=("Arial", 12), bg="white")
        self.source_label.pack(side=tk.LEFT)

        self.video_source = tk.StringVar()
        self.video_source.set("Webcam")

        self.source_select = tk.OptionMenu(self.source_frame, self.video_source, "Webcam", "Video.mp4")
        self.source_select.pack(side=tk.RIGHT)

        # Langue de reconnaissance
        self.lang_frame = tk.Frame(self.main_frame, bg="white", padx=10, pady=10)
        self.lang_frame.pack()

        self.lang_label = tk.Label(self.lang_frame, text="Langue de reconnaissance:", font=("Arial", 12), bg="white")
        self.lang_label.pack(side=tk.LEFT)

        self.language_var = tk.StringVar()
        self.language_var.set("en")

        self.language_select = tk.OptionMenu(self.lang_frame, self.language_var, "en", "fr")
        self.language_select.pack(side=tk.RIGHT)

        # Boutons d'action
        self.start_button = tk.Button(self.main_frame, text="Démarrer la capture", command=self.start_capture,
                                      font=("Arial", 12))
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.rounded_corners(self.start_button)

        self.stop_button = tk.Button(self.main_frame, text="Arrêter la capture", command=self.stop_capture,
                                     font=("Arial", 12))
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.rounded_corners(self.stop_button)

        # Résultats
        self.result_frame = tk.Frame(self.main_frame, bg="grey", padx=10, pady=10)
        self.result_frame.pack()

        self.image_label = tk.Label(self.result_frame)
        self.image_label.pack()

        # Barre de statut
        self.status_bar = tk.Label(self.main_frame, text="", font=("Arial", 12), bg="white")
        self.status_bar.pack()

        # Recognition initialization
        self.reader = easyocr.Reader([self.language_var.get()], gpu=True)
        self.threshold = 0.25

        self.get_video_source()

        self.update_frame()

        # Window closing management
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_video_source(self):
        try:
            if self.video_source.get() == "Webcam":
                self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture(self.video_source.get())
        except cv2.error as e:
            self.status_bar.config(text=f"Erreur: {e}")

    def start_capture(self):
        self.get_video_source()
        self.status_bar.config(text="Capture vidéo démarrée")

    def rounded_corners(self, widget):
        # Crée une image avec des coins arrondis
        img = Image.new("RGBA", (100, 30), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, 100, 30), radius=5, fill="lightblue")

        # Convertit l'image Pillow en PhotoImage Tkinter
        img_tk = ImageTk.PhotoImage(img)

        # Applique l'image avec des coins arrondis au widget
        widget.config(image=img_tk, compound="center", borderwidth=0)
        widget.img_tk = img_tk
    def stop_capture(self):
        self.cap.release()
        self.status_bar.config(text="Capture vidéo arrêtée")

    def update_frame(self):
        ret, frame = self.cap.read()

        if ret:
            # Recognition processing
            text_detections = self.reader.readtext(frame)
            text_detections = self.filter_text(text_detections)

            # Draw bounding boxes and text
            frame = self.draw_boxes(frame, text_detections)

            # Frame conversion for Tkinter display
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            # Show image
            self.image_label.config(image=image)
            self.image_label.image = image

        self.root.after(10, self.update_frame)

    def filter_text(self, text):
        return [t for t in text if t[2] > self.threshold]

    def draw_boxes(self, frame, text):
        for detection in text:
            bbox, detected_text, confidence = detection
            bbox = [(int(pt[0]), int(pt[1])) for pt in bbox]
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
            cv2.putText(frame, detected_text, bbox[0], cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        return frame

    def on_closing(self):
        print("Fermeture de l'application")
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()
