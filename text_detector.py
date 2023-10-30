import cv2
import easyocr
import tkinter as tk
from PIL import Image, ImageTk

# Initialize the text reader
reader = easyocr.Reader(['en', 'fr'], gpu=True)

# Create a Tkinter window
root = tk.Tk()
root.title("Text Recognition")

# Open a connection to the webcam (you can specify the camera index)
cap = cv2.VideoCapture(0)

threshold = 0.25

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Detect text in the frame
        text_ = reader.readtext(frame)

        # Filter out low-scoring detections
        text_ = filter(lambda t: t[2] > threshold, text_)

        # Draw bounding boxes around the detections
        for t in text_:
            bbox, text, score = t
            bbox = [(int(x[0]), int(x[1])) for x in bbox]
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 5)
            cv2.putText(frame, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

        # Convert the OpenCV frame to a format compatible with Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        # Update the label with the new frame
        label.img = img
        label.config(image=img)
        label.after(10, update_frame)

label = tk.Label(root)
label.pack()

update_frame()

root.mainloop()

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
