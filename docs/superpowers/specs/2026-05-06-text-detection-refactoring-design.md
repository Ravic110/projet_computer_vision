# Text Detection App Refactoring Design

## Overview

Complete refactoring of the Text Detection App to improve architecture, code quality, performance, testing, and developer experience. The monolithic 365-line file will be decomposed into focused modules with clear responsibilities.

## Architecture

### New Directory Structure

```
projet_computer_vision/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Makefile
├── logging.conf
├── src/
│   ├── text_detector/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── config.py
│   │   ├── ocr_engine.py
│   │   ├── image_processor.py
│   │   ├── text_detector.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging_setup.py
│   │       └── path_helpers.py
│   └── tests/
│       ├── __init__.py
│       ├── test_ocr_engine.py
│       ├── test_image_processor.py
│       ├── test_config.py
│       └── test_utils.py
└── text_detector/assets/             ← backward compat (symlink)
```

### Module Responsibilities

**`config.py`** - Centralized configuration: theme colors, default settings, available languages, paths
**`ocr_engine.py`** - EasyOCR wrapper with model caching, multi-language support, GPU toggle, thread-safe operations
**`image_processor.py`** - Pure functions: filter_text, draw_boxes, compute_avg_color, BGR to RGB conversion
**`text_detector.py`** - Tkinter GUI only, delegates to ocr_engine and image_processor, uses threading for OCR
**`utils/logging_setup.py`** - Logging configuration from logging.conf
**`utils/path_helpers.py`** - pathlib-based path utilities

### Data Flow

1. GUI receives input (webcam frame or image file)
2. GUI submits frame to OCR engine via thread pool
3. OCR engine returns detections (bbox, text, confidence)
4. GUI calls image_processor to draw boxes
5. GUI updates display and text panel
6. User can export results (TXT, CSV, JSON)

## Error Handling

- All OCR operations wrapped in try/except with proper logging
- Thread-safe error propagation to GUI via queue
- Graceful degradation when webcam disconnects
- Validation on all user inputs (confidence threshold, language selection)

## Testing Strategy

- pytest for unit tests
- Mock EasyOCR for tests (no GPU/model download needed)
- Test image_processor pure functions with sample data
- Test config loading and validation
- Test path helpers
- Minimum 80% coverage target

## Key Decisions

1. **Threading**: Use `threading.Thread` with `queue.Queue` for OCR to avoid blocking GUI
2. **Model caching**: Cache Reader instances by language tuple to avoid reload
3. **pathlib**: Use Path objects throughout, convert to str only for library calls
4. **Logging**: Configure via logging.conf file, use module-level loggers
5. **Dependencies**: Use `>=` in requirements.txt for flexibility, pin in pyproject.toml for reproducibility
6. **Backward compatibility**: Keep `text_detector/` at root as symlink for existing users
7. **Export formats**: TXT (existing), CSV (new), JSON (new) via same export interface
8. **History**: Simple in-memory list with max 100 entries, accessible via GUI tab

## File Changes

### New Files
- `src/text_detector/__init__.py` - Package init with version
- `src/text_detector/__main__.py` - Entry point
- `src/text_detector/config.py` - Configuration
- `src/text_detector/ocr_engine.py` - OCR engine
- `src/text_detector/image_processor.py` - Image processing
- `src/text_detector/utils/__init__.py` - Utils package
- `src/text_detector/utils/logging_setup.py` - Logging setup
- `src/text_detector/utils/path_helpers.py` - Path helpers
- `src/tests/__init__.py` - Tests package
- `src/tests/test_*.py` - Test files
- `pyproject.toml` - Project config
- `Makefile` - Dev commands
- `logging.conf` - Logging config
- `CHANGELOG.md` - Version history
- `LICENSE` - MIT License

### Modified Files
- `README.md` - Updated structure, installation, features
- `requirements.txt` - Unfrozen versions
- `.gitignore` - Updated patterns

### Removed Files
- `text_detector/text_detector.py` (replaced by modular src/)
