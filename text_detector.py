import cv2
import easyocr
import tkinter as tk
from PIL import Image, ImageTk


class TextRecognitionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance Texte")

        #  interface
        self.main_frame = tk.Frame(self.root, bg="white", padx=10, pady=10)
        self.main_frame.pack()

        self.title_label = tk.Label(self.main_frame, text="Text Recognition", font=("Arial", 18), bg="white")
        self.title_label.pack()

        self.video_source = tk.StringVar()
        self.video_source.set("Webcam")

        self.source_select = tk.OptionMenu(self.main_frame, self.video_source, "Webcam", "Video.mp4")
        self.source_select.pack()

        self.result_frame = tk.Frame(self.main_frame, bg="grey", padx=10, pady=10)
        self.result_frame.pack()

        self.image_label = tk.Label(self.result_frame)
        self.image_label.pack()

        # Recognition initialization
        self.reader = easyocr.Reader(['en', 'fr'], gpu=True)
        self.threshold = 0.25

        self.get_video_source()

        self.update_frame()

        # Window closing management
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_video_source(self):
        if self.video_source.get() == "Webcam":
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.video_source.get())

    def update_frame(self):
        ret, frame = self.cap.read()

        if ret:
            # Recognition processing
            text = self.reader.readtext(frame)
            text = self.filter_text(text)

            # Draw bounding boxes and text
            frame = self.draw_boxes(frame, text)

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
            bbox = [(int(pt[0]), int(pt[1]))
            for pt in bbox]
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
            cv2.putText(frame, detected_text, bbox[0], cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        return frame

    def on_closing(self):
        print("Closing the application")
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()