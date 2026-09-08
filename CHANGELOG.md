# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `color_detector` module: samples the color around a point of a frame and
  names it from the 148 CSS/X11 colors Pillow already ships, matched in
  CIELAB so the name matches what the eye sees rather than what RGB distance
  suggests
- Color picker in the GUI: "Pick Color" (`Ctrl+P`) arms a single click on the
  image, and the sidebar reports the nearest CSS name, hex, RGB and HSV;
  clicking the swatch copies the hex code
- The picker reads a 5x5 average of the stored frame, so webcam noise and the
  drawn detection boxes cannot skew the result

- `layout` module: window sizing, breakpoints and DPI pixel scaling as pure,
  tested functions
- Responsive layout with three breakpoints: below 1150 px the text panel moves
  under the image, and below 750 px the sidebar is hidden behind a `☰` button
  in the status bar
- The sidebar scrolls, so the settings stay reachable on a short screen
- The window opens at a size derived from the screen, with a 600x450 minimum,
  and stays on one monitor of a multi-monitor desktop

### Changed
- The picker and ROI selection are mutually exclusive: both bind a click on
  the image, so activating one cancels the other
- The image pane now absorbs the space left over instead of the text panel:
  Tk stretches the last pane by default, which left the image a sliver on a
  wide screen and 41 px tall once the panels were stacked
- Dimensions written in pixels (pane widths, the color swatch) follow the
  display's DPI scaling; fonts already did

### Performance
- OCR runs as two stages: detection on a frame downscaled to
  `AppSettings.detect_max_width` (480 px), recognition at the full
  `ocr_max_width`. `reader.detect()` is ~95% of a pass and scales with pixel
  count, while `reader.recognize()` costs ~150 ms. Measured on 720p frames:
  3388 -> 1134 ms (large text), 3170 -> 1161 ms (small text),
  3329 -> 1116 ms (noisy)
- `OCREngine.preload()` loads the model at startup on its own thread instead
  of inside the first detection, taking ~2.2 s off it. It deliberately avoids
  the work queue, which would mark the engine busy and drop a frame submitted
  straight away
- `preprocess_for_ocr()` uses a bilateral filter rather than non-local means:
  same recognition on a noisy 800px frame for 12 ms against 249 ms. Narrowing
  the non-local means search window was measured and rejected -- it loses the
  accuracy benefit while still costing 77-162 ms

### Changed
- Detection boxes are grouped differently, because the detector now works at
  a lower resolution: words that used to be reported separately may be merged
  and vice versa. The extracted text is unaffected, but the drawn rectangles
  and history rows are segmented differently
- On a heavily noised frame one character's case can be lost against the old
  pipeline (`OpenCV` read as `OpenCv`). Recovering it means detecting at full
  resolution, which costs about 1.9 s per pass

### Fixed
- Resizing the window redraws the current still image, which used to keep the
  scale it was first drawn at
- The sidebar scrollbar is packed before its canvas, without which the canvas
  claims the whole width and the scrollbar is never allocated any

## [2.1.0] - 2026-09-07

### Added
- `capture.FrameGrabber`: webcam reads run on their own thread, so a slow or
  disconnected camera no longer stalls the interface
- `image_processor.resize_frame_for_display()`: display scaling as a pure,
  tested function
- `OCREngine.poll_result()` and `OCREngine.settings`: the GUI collects results
  and configures the engine through a public, thread-safe surface
- `AppSettings.validate()`: the invariants checked at construction are now
  re-checked after the settings are mutated at runtime
- `path_helpers.get_state_dir()` / `get_log_path()`: the log has one fixed
  home under `XDG_STATE_HOME` instead of the current working directory

### Changed
- The preview now refreshes on every captured frame; `frame_skip` throttles
  only the OCR pass, which is what actually costs time. Detection boxes stay
  drawn between passes instead of flashing for a single frame.
- OCR results are delivered on the Tk main loop rather than pushed from the
  worker thread through `after()`, which was not thread-safe
- `OCREngine.shutdown()` wakes an idle worker immediately instead of waiting
  out its one-second poll timeout
- Preview frames are scaled with OpenCV instead of PIL before being handed to
  Tk: a 720p tick went from 23.4 ms to 9.7 ms, against a 33 ms budget
- The test fixture now shuts the OCR engine down, so the suite no longer
  leaks a worker thread per test (runtime dropped from ~40 s to ~25 s)
- Application assets moved inside the package, so the icon survives an install
- `requirements.txt` defers to `pyproject.toml` instead of duplicating it

### Fixed
- `StatusLED` no longer stacks a new canvas item on every status change
- The busy indicator is released when a frame is cleared while OCR is running
- The GUI no longer reaches into `OCREngine._settings`
- Only `text_detector` is packaged; `tests` was being installed as a
  top-level package
- `ruff format` compliance, which was failing CI

### Removed
- `compute_avg_color`, `ensure_dir`, `get_safe_path` and
  `AppSettings.validate_language`: production code that only tests referenced

## [2.0.0] - 2026-05-06

### Added
- Modular architecture (config, ocr_engine, image_processor, utils)
- Threading for OCR to prevent GUI freezing
- Multi-language support with model caching
- GPU toggle option in settings
- Export to CSV and JSON formats
- Detection history tracking
- Unit tests with pytest
- Logging configuration
- pyproject.toml for modern packaging
- Makefile for development commands
- Type hints throughout codebase
- Docstrings (Google style)

### Changed
- Replaced os.path with pathlib
- Unfrozen dependency versions in requirements.txt
- Centralized theme colors in config module
- Refactored GUI to use configuration constants

### Removed
- Monolithic single-file architecture
- BOM character from requirements.txt

## [1.0.0] - Previous version
- Initial single-file implementation
- Basic text detection with EasyOCR
- Tkinter GUI with webcam and image support
