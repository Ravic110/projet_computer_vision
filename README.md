# Text Detection App

A real-time text recognition application using computer vision. It captures video from your webcam, detects text in the frames, and displays the recognized text with bounding boxes and color information.

## Prerequisites

- Python >= 3.11

## Installation

1. Clone or download this repository.
2. Install the package:

   ```bash
   pip install -e .
   ```

3. (Optional) Install dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

## Usage

Run the application:

```bash
python -m text_detector
```

Or via Makefile:

```bash
make run
```

### Controls

- **Start Webcam** - Begin video capture and text detection
- **Stop Capture** - Stop webcam capture
- **Load Image** - Detect text from a saved image file
- **Save Results** - Export detected text (TXT, CSV, or JSON)
- **Clear** - Reset the viewer
- **About** - Application information

### Settings

- **Language selector** - Choose OCR language model (en, fr, de, es, it, pt)
- **Confidence threshold** - Filter results by confidence (0.05 - 0.80)
- **Use GPU** - Enable GPU acceleration for faster detection

## Development

```bash
make install    # Install package
make dev        # Install with dev dependencies
make test       # Run tests with coverage
make lint       # Run linter
make format     # Format code
make clean      # Remove build artifacts
```

## Features

- Real-time text detection using EasyOCR
- Load image files or use the webcam
- GUI built with Tkinter
- Multi-language OCR support
- Confidence threshold control
- GPU acceleration toggle
- Export to TXT, CSV, or JSON
- Detection history tracking

## Architecture

```
src/
├── text_detector/
│   ├── __init__.py          # Package version
│   ├── __main__.py          # Entry point
│   ├── config.py            # Configuration & theme
│   ├── ocr_engine.py        # OCR engine with caching
│   ├── image_processor.py   # Image processing functions
│   ├── text_detector.py     # Tkinter GUI
│   └── utils/
│       ├── logging_setup.py # Logging configuration
│       └── path_helpers.py  # Path utilities
└── tests/                   # Unit tests
```

## Dependencies

- easyocr >= 1.7.1
- numpy >= 1.24.3
- opencv-python >= 4.10.0.84
- pillow >= 11.0.0

## License

MIT License - see LICENSE file for details.
