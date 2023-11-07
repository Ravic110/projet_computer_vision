import cv2
import easyocr
import tkinter as tk
from PIL import Image, ImageTk

class TextRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Recognition")

        self.reader = easyocr.Reader(['en', 'fr'], gpu=True)
        self.cap = cv2.VideoCapture(0)
        self.threshold = 0.25

        self.label = tk.Label(root)
        self.label.pack()

        self.running = False

        self.create_ui()

    def create_ui(self):
        start_button = tk.Button(self.root, text="Start", command=self.start_capture)
        stop_button = tk.Button(self.root, text="Stop", command=self.stop_capture)

        start_button.pack()
        stop_button.pack()

    def start_capture(self):
        self.running = True
        self.update_frame()

    def stop_capture(self):
        self.running = False

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                text_ = self.reader.readtext(frame)
                text_ = filter(lambda t: t[2] > self.threshold, text_)

                for t in text_:
                    bbox, text, score = t
                    bbox = [(int(x[0]), int(x[1])) for x in bbox]
                    cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 5)
                    cv2.putText(frame, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)

                self.label.img = img
                self.label.config(image=img)

            self.label.after(10, self.update_frame)
        else:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()
