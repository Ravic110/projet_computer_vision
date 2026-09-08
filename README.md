# Text Detection App

A real-time text recognition application using computer vision. It captures video from your webcam, detects text in the frames, and displays the recognized text with bounding boxes. A color picker reads the color under any point of the image.

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
- **Select ROI** - Drag a rectangle on the image to restrict OCR to that region
- **Clear ROI** - Run OCR on the whole frame again
- **Pick Color** - Click a point on the image to read its color
- **Save Results** - Export detected text (TXT, CSV, or JSON)
- **Clear** - Reset the viewer
- **Reset Settings** - Restore every setting to its default
- **About** - Application information

Right-click the image (or press Ctrl+V) to paste an image from the clipboard.

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Space`  | Start / stop the webcam |
| `Ctrl+O` | Load an image |
| `Ctrl+S` | Save results |
| `Ctrl+C` | Copy detected text |
| `Ctrl+V` | Paste an image from the clipboard |
| `Ctrl+P` | Toggle the color picker |
| `Ctrl+R` | Reset settings |
| `Esc`    | Clear results |

Shortcuts are ignored while the history search box has focus.

### Responsive layout

The window opens at a size derived from the screen and never smaller than
600x450. On a multi-monitor desktop it stays on one monitor: Tk reports the
bounding box of every display, so the usable width is capped at what a 16:9
monitor of that height would be.

The layout then follows the window width:

| Window width | Layout |
|--------------|--------|
| 1150 px and up | sidebar, image and text side by side |
| 750-1150 px | sidebar on the left, text panel under the image |
| under 750 px | sidebar hidden, text panel under the image |

The image always takes the space left over; the text panel keeps a fixed
size. The `☰` button in the status bar shows or hides the sidebar at any
width, and the sidebar scrolls when the settings are taller than the window.
Dimensions written in pixels follow the display's DPI scaling.

### Detection speed

A pass runs in two stages. Locating the text costs about 95% of the work and
scales with pixel count, while reading it is cheap, so detection runs on a
frame downscaled to `detect_max_width` (480 px) and the characters are then
read at the full **OCR Detail** width. On a 720p frame this takes a pass from
roughly 3.3 s to 1.15 s on a 2-core CPU.

The trade is detection sensitivity, not character accuracy: text too small to
be found at the detection width is lost, because recognition cannot recover a
box that was never located. 480 px keeps the text intact on the sample frames
used to tune it; 400 px starts dropping short words. Raising it recovers
sensitivity at a steep price -- detecting at 800 px costs about 1.9 s more per
pass.

The OCR model is loaded at startup rather than during the first detection,
which used to make that first pass roughly 2.2 s slower than the rest.

### Panels

- **Current** - Text detected in the frame on screen, with confidence
- **History** - The last 100 detections, searchable; double-click a row to copy it

### Color picker

Press **Pick Color** (or `Ctrl+P`), then click anywhere on the image. The sidebar
shows the color under the cursor: its nearest CSS name, hex, RGB and HSV values.
Click the swatch to copy the hex code. The reading is the average of a 5x5 pixel
window, so webcam noise does not skew it, and it is taken from the raw frame -
clicking on a detection box reports the image color, not the box color.

The picker and ROI selection both use a click on the image, so activating one
turns the other off.

### Settings

- **Languages** - Tick any combination of en, fr, de, es, it, pt; EasyOCR reads them together
- **Confidence** - Filter results by confidence (0.05 - 0.80)
- **OCR Frequency** - Run OCR every N captured frames (1 - 60); the preview stays smooth regardless
- **OCR Detail** - Width the frame is downscaled to before characters are read (400 - 2000 px)
- **GPU Acceleration** - Use the GPU for detection
- **Preprocessing** - Denoise the frame before OCR (a bilateral filter, ~12 ms)
- **Merge into paragraphs** - Group words into blocks; EasyOCR reports no confidence in this mode, so the confidence filter is disabled

Settings are saved to `~/.config/text-detector/settings.json` and restored on
the next launch. The log is written to `$XDG_STATE_HOME/text-detector/`
(`~/.local/state/text-detector/` by default).

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
- Load image files, paste from the clipboard, or use the webcam
- GUI built with Tkinter
- Threaded capture and OCR: neither blocks the interface
- Region-of-interest selection
- Multi-language OCR (several languages at once)
- Confidence threshold and paragraph-merge modes
- GPU acceleration toggle
- Export to TXT, CSV, or JSON
- Searchable detection history
- Settings persisted between sessions

## Architecture

```
src/
├── text_detector/
│   ├── __init__.py           # Package version
│   ├── __main__.py           # Entry point
│   ├── config.py             # Theme colours and validated settings
│   ├── capture.py            # Background webcam capture thread
│   ├── ocr_engine.py         # OCR worker thread with model caching
│   ├── image_processor.py    # Pure image / geometry functions
│   ├── settings_manager.py   # Settings persistence (atomic JSON)
│   ├── widgets.py            # Reusable themed widgets
│   ├── text_detector.py      # Tkinter GUI
│   ├── assets/               # Application icon
│   └── utils/
│       ├── logging_setup.py  # Logging configuration
│       └── path_helpers.py   # Project, asset and state paths
└── tests/                    # Unit tests
```

### Threading model

Tkinter is not thread-safe, so no background thread ever touches a widget:

- `FrameGrabber` reads the webcam on its own thread and keeps only the
  latest frame, so a slow camera cannot stall the interface.
- `OCREngine` runs detection on a single worker thread with a one-slot work
  queue; frames submitted while it is busy are dropped rather than queued.
- The Tk main loop refreshes the preview every tick and collects finished
  OCR results with `OCREngine.poll_result()`.

## Dependencies

- easyocr >= 1.7.1
- numpy >= 1.24.3
- opencv-python >= 4.10.0.84
- pillow >= 11.0.0

## License

MIT License - see LICENSE file for details.
