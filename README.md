# Text Detection App

A real-time text recognition application using computer vision. It captures video from your webcam, detects text in the frames, and displays the recognized text with bounding boxes and color information.

## Prerequisites

- Python 3.11.9 (download from [python.org](https://www.python.org/downloads))

## Installation

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:

```bash
python text_detector/text_detector.py
```

- Click "Start Webcam" to begin video capture and text detection.
- Click "Stop Capture" to stop.
- Click "Load Image" to detect text from a saved image file.
- Click "Save Results" to save detected text to a file.
- Click "Clear" to reset the viewer.
- Use the language selector and confidence slider to improve accuracy.
- Click "About" for information.

## Features

- Real-time text detection using EasyOCR
- Load image files or use the webcam
- GUI built with Tkinter
- Language selection for OCR models
- Confidence threshold control for filtering results
- Text output panel with detected phrases
- Save detected text to a `.txt` file

## Dependencies

- easyocr==1.7.1
- numpy==1.24.3
- opencv-python==4.10.0.84
- pillow==11.0.0