# Text Detection App Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the monolithic text detection app into a modular, tested, well-documented application with threading, multi-language support, and modern tooling.

**Architecture:** Decompose single 365-line file into focused modules (config, ocr_engine, image_processor, gui, utils) under src/text_detector/, with pytest tests under src/tests/, and modern project tooling (pyproject.toml, Makefile, logging).

**Tech Stack:** Python 3.12, Tkinter, EasyOCR, OpenCV, Pillow, pytest, pathlib, threading, logging

---

### Task 1: Project Foundation (pyproject.toml, requirements.txt, Makefile, logging.conf)

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `logging.conf`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "text-detector"
version = "2.0.0"
description = "A real-time text recognition application using computer vision"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "easyocr>=1.7.1",
    "numpy>=1.24.3",
    "opencv-python>=4.10.0.84",
    "pillow>=11.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1.0",
]

[project.scripts]
text-detector = "text_detector.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["src/tests"]
pythonpath = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]
```

- [ ] **Step 2: Create Makefile**

```makefile
.PHONY: install dev test lint format typecheck run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest src/tests -v --cov=text_detector --cov-report=term-missing

lint:
	python -m ruff check src/

format:
	python -m ruff format src/

run:
	python -m text_detector

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
```

- [ ] **Step 3: Create logging.conf**

```ini
[loggers]
keys=root,text_detector

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=standard

[logger_root]
level=INFO
handlers=consoleHandler

[logger_text_detector]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=text_detector
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=standard
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=standard
args=('text_detector.log', 'a')

[formatter_standard]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

- [ ] **Step 4: Update requirements.txt (remove BOM, unfreeze versions)**

```txt
easyocr>=1.7.1
numpy>=1.24.3
opencv-python>=4.10.0.84
pillow>=11.0.0
```

- [ ] **Step 5: Update .gitignore**

```
.idea
.venv
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
text_detector.log
detected_text_results.txt
*.egg
.mypy_cache/
.ruff_cache/
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml Makefile logging.conf requirements.txt .gitignore
git commit -m "chore: add project foundation (pyproject.toml, Makefile, logging, unfrozen deps)"
```

---

### Task 2: Utils Module (path_helpers, logging_setup) + Tests

**Files:**
- Create: `src/text_detector/__init__.py`
- Create: `src/text_detector/utils/__init__.py`
- Create: `src/text_detector/utils/path_helpers.py`
- Create: `src/text_detector/utils/logging_setup.py`
- Create: `src/tests/__init__.py`
- Create: `src/tests/test_utils.py`

- [ ] **Step 1: Create src/text_detector/__init__.py**

```python
"""Text Detection App - Real-time text recognition using computer vision."""

__version__ = "2.0.0"
```

- [ ] **Step 2: Create src/text_detector/utils/__init__.py**

```python
"""Utility modules for text detection."""
```

- [ ] **Step 3: Create src/text_detector/utils/path_helpers.py**

```python
"""Path utilities using pathlib."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_assets_dir() -> Path:
    """Return the assets directory."""
    return get_project_root() / "text_detector" / "assets"


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_safe_path(base: Path, filename: str, extension: str) -> Path:
    """Return a safe file path with the given extension."""
    if not filename.endswith(extension):
        filename = f"{filename}{extension}"
    return base / filename
```

- [ ] **Step 4: Create src/text_detector/utils/logging_setup.py**

```python
"""Logging configuration setup."""

import logging
import logging.config
from pathlib import Path


def setup_logging(config_path: Path | None = None) -> None:
    """Configure logging from logging.conf file.

    Args:
        config_path: Path to logging configuration file.
            Defaults to logging.conf in project root.
    """
    if config_path is None:
        from text_detector.utils.path_helpers import get_project_root
        config_path = get_project_root() / "logging.conf"

    if config_path.exists():
        logging.config.fileConfig(config_path, disable_existing_loggers=False)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the text_detector prefix.

    Args:
        name: Module name (e.g., 'ocr_engine', 'gui').

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"text_detector.{name}")
```

- [ ] **Step 5: Write tests for utils (src/tests/test_utils.py)**

```python
"""Tests for utility modules."""

from pathlib import Path
import tempfile

from text_detector.utils.path_helpers import get_assets_dir, ensure_dir, get_safe_path


def test_get_assets_dir_returns_path() -> None:
    result = get_assets_dir()
    assert isinstance(result, Path)
    assert result.name == "assets"


def test_ensure_dir_creates_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "new" / "nested" / "dir"
        assert not target.exists()
        ensure_dir(target)
        assert target.exists()
        assert target.is_dir()


def test_ensure_dir_existing_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "existing"
        target.mkdir()
        result = ensure_dir(target)
        assert result.exists()


def test_get_safe_path_adds_extension() -> None:
    base = Path("/tmp/test")
    result = get_safe_path(base, "output", ".txt")
    assert result.name == "output.txt"


def test_get_safe_path_keeps_existing_extension() -> None:
    base = Path("/tmp/test")
    result = get_safe_path(base, "output.txt", ".txt")
    assert result.name == "output.txt"
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest src/tests/test_utils.py -v
```
Expected: 5 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/text_detector/__init__.py src/text_detector/utils/ src/tests/test_utils.py
git commit -m "feat: add utils module (path_helpers, logging_setup) with tests"
```

---

### Task 3: Configuration Module + Tests

**Files:**
- Create: `src/text_detector/config.py`
- Modify: `src/tests/test_utils.py` (add config tests)

- [ ] **Step 1: Create src/text_detector/config.py**

```python
"""Centralized configuration for the text detection app."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThemeColors:
    """Color theme constants for the GUI."""
    background: str = "#F0F0F0"
    white: str = "#FFFFFF"
    frame_bg: str = "#FFFFFF"
    result_frame_bg: str = "#DDEEFF"
    text_output_bg: str = "#F7F7F7"
    title_bg: str = "#63BFF3"
    title_fg: str = "#FFFFFF"
    primary: str = "#4A90E2"
    primary_active: str = "#3A70B2"
    success: str = "#5AE3B1"
    success_active: str = "#4AA57C"
    danger: str = "#CA3074"
    danger_active: str = "#A02458"
    warning: str = "#FFA500"
    warning_active: str = "#CC8400"
    neutral: str = "#888888"
    neutral_active: str = "#666666"
    about: str = "#6D6DFF"
    about_active: str = "#5757D0"


@dataclass
class AppSettings:
    """Application settings with validation."""
    available_languages: list[str] = field(default_factory=lambda: ["en", "fr", "de", "es", "it", "pt"])
    default_language: str = "en"
    min_confidence: float = 0.05
    max_confidence: float = 0.8
    default_confidence: float = 0.25
    confidence_resolution: float = 0.05
    frame_skip: int = 5
    max_history: int = 100
    gpu_enabled: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.max_confidence <= 1.0):
            raise ValueError("max_confidence must be between 0.0 and 1.0")
        if not (self.min_confidence <= self.default_confidence <= self.max_confidence):
            raise ValueError("default_confidence must be between min and max confidence")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if self.max_history < 1:
            raise ValueError("max_history must be >= 1")

    def validate_language(self, lang: str) -> bool:
        """Check if a language code is available."""
        return lang in self.available_languages


THEME = ThemeColors()
SETTINGS = AppSettings()
```

- [ ] **Step 2: Create src/tests/test_config.py**

```python
"""Tests for configuration module."""

import pytest

from text_detector.config import AppSettings, ThemeColors


def test_theme_colors_defaults() -> None:
    theme = ThemeColors()
    assert theme.background == "#F0F0F0"
    assert theme.primary == "#4A90E2"
    assert theme.success == "#5AE3B1"
    assert theme.danger == "#CA3074"


def test_app_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.default_language == "en"
    assert settings.default_confidence == 0.25
    assert settings.frame_skip == 5
    assert settings.max_history == 100
    assert settings.gpu_enabled is False


def test_app_settings_validate_language() -> None:
    settings = AppSettings()
    assert settings.validate_language("en") is True
    assert settings.validate_language("fr") is True
    assert settings.validate_language("zz") is False


def test_app_settings_invalid_min_confidence() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        AppSettings(min_confidence=-0.1)


def test_app_settings_invalid_max_confidence() -> None:
    with pytest.raises(ValueError, match="max_confidence"):
        AppSettings(max_confidence=1.5)


def test_app_settings_default_out_of_range() -> None:
    with pytest.raises(ValueError, match="default_confidence"):
        AppSettings(min_confidence=0.5, max_confidence=0.8, default_confidence=0.1)


def test_app_settings_invalid_frame_skip() -> None:
    with pytest.raises(ValueError, match="frame_skip"):
        AppSettings(frame_skip=0)


def test_app_settings_invalid_max_history() -> None:
    with pytest.raises(ValueError, match="max_history"):
        AppSettings(max_history=0)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest src/tests/test_config.py -v
```
Expected: 8 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/text_detector/config.py src/tests/test_config.py
git commit -m "feat: add config module with ThemeColors and validated AppSettings"
```

---

### Task 4: Image Processor Module + Tests

**Files:**
- Create: `src/text_detector/image_processor.py`
- Create: `src/tests/test_image_processor.py`

- [ ] **Step 1: Create src/text_detector/image_processor.py**

```python
"""Image processing functions for text detection visualization."""

from typing import Any

import cv2
import numpy as np


Detection = tuple[list[list[float]], str, float]


def filter_text(
    detections: list[Detection],
    threshold: float,
) -> list[Detection]:
    """Filter detections by confidence threshold.

    Args:
        detections: List of (bbox, text, confidence) tuples from EasyOCR.
        threshold: Minimum confidence value (0.0-1.0).

    Returns:
        Filtered list containing only detections above threshold.
    """
    return [item for item in detections if item[2] >= threshold]


def compute_avg_color(frame: np.ndarray, bbox: list[list[float]]) -> tuple[float, float, float]:
    """Compute average color within a bounding box region.

    Args:
        frame: OpenCV image array (BGR).
        bbox: Bounding box coordinates from EasyOCR.

    Returns:
        Average (B, G, R) color tuple, or (0, 0, 0) if region is invalid.
    """
    points = [(int(pt[0]), int(pt[1])) for pt in bbox]
    x1 = max(0, points[0][0])
    y1 = max(0, points[0][1])
    x2 = max(0, points[2][0])
    y2 = max(0, points[2][1])

    if y2 <= y1 or x2 <= x1:
        return (0.0, 0.0, 0.0)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return (0.0, 0.0, 0.0)

    return cv2.mean(roi)[:3]


def draw_boxes_with_colors(
    frame: np.ndarray,
    detections: list[Detection],
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw bounding boxes and text labels on a frame.

    Args:
        frame: OpenCV image array (BGR). Will be copied.
        detections: List of (bbox, text, confidence) tuples.
        box_color: BGR color for bounding box lines.
        text_color: BGR color for text labels.

    Returns:
        New frame with bounding boxes and labels drawn.
    """
    result = frame.copy()

    for bbox, text, confidence in detections:
        points = [(int(pt[0]), int(pt[1])) for pt in bbox]
        pts = np.array(points, dtype=np.int32)

        cv2.polylines(result, [pts], isClosed=True, color=box_color, thickness=2)
        cv2.putText(
            result,
            f"{text} ({confidence:.2f})",
            (points[0][0], points[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA,
        )

    return result


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB.

    Args:
        frame: OpenCV image array (BGR).

    Returns:
        RGB image array.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

- [ ] **Step 2: Create src/tests/test_image_processor.py**

```python
"""Tests for image processing module."""

import numpy as np

from text_detector.image_processor import filter_text, compute_avg_color, draw_boxes_with_colors, bgr_to_rgb


def _make_detection(text: str, confidence: float) -> tuple:
    """Helper to create a mock detection tuple."""
    bbox = [[10.0, 10.0], [100.0, 10.0], [100.0, 30.0], [10.0, 30.0]]
    return (bbox, text, confidence)


def test_filter_text_above_threshold() -> None:
    detections = [_make_detection("hello", 0.8), _make_detection("world", 0.9)]
    result = filter_text(detections, threshold=0.5)
    assert len(result) == 2


def test_filter_text_below_threshold() -> None:
    detections = [_make_detection("low", 0.1), _make_detection("high", 0.7)]
    result = filter_text(detections, threshold=0.5)
    assert len(result) == 1
    assert result[0][1] == "high"


def test_filter_text_empty() -> None:
    result = filter_text([], threshold=0.5)
    assert result == []


def test_filter_text_exact_threshold() -> None:
    detections = [_make_detection("exact", 0.5)]
    result = filter_text(detections, threshold=0.5)
    assert len(result) == 1


def test_compute_avg_color_uniform() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # Blue channel
    bbox = [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]]
    result = compute_avg_color(frame, bbox)
    assert result[0] == 255.0  # Blue


def test_compute_avg_color_invalid_bbox() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [[50.0, 50.0], [10.0, 50.0], [10.0, 10.0], [50.0, 10.0]]
    result = compute_avg_color(frame, bbox)
    assert result == (0.0, 0.0, 0.0)


def test_draw_boxes_returns_same_shape() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [_make_detection("test", 0.9)]
    result = draw_boxes_with_colors(frame, detections)
    assert result.shape == frame.shape


def test_draw_boxes_does_not_mutate_input() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    detections = [_make_detection("test", 0.9)]
    draw_boxes_with_colors(frame, detections)
    np.testing.assert_array_equal(frame, original)


def test_bgr_to_rgb_conversion() -> None:
    frame = np.array([[[0, 0, 255]]], dtype=np.uint8)  # BGR: Red
    result = bgr_to_rgb(frame)
    assert result[0, 0, 0] == 255  # RGB: Red channel
    assert result[0, 0, 2] == 0    # RGB: Blue channel
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest src/tests/test_image_processor.py -v
```
Expected: 9 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/text_detector/image_processor.py src/tests/test_image_processor.py
git commit -m "feat: add image_processor module with pure functions and tests"
```

---

### Task 5: OCR Engine Module + Tests

**Files:**
- Create: `src/text_detector/ocr_engine.py`
- Create: `src/tests/test_ocr_engine.py`

- [ ] **Step 1: Create src/text_detector/ocr_engine.py**

```python
"""OCR engine wrapper for EasyOCR with model caching and threading support."""

import queue
import threading
from typing import Any, Callable

from text_detector.config import AppSettings
from text_detector.utils.logging_setup import get_logger

logger = get_logger("ocr_engine")


class DetectionResult:
    """Container for OCR detection results."""

    def __init__(
        self,
        detections: list[tuple[list[list[float]], str, float]],
        languages: list[str],
        success: bool = True,
        error: str | None = None,
    ) -> None:
        self.detections = detections
        self.languages = languages
        self.success = success
        self.error = error


class OCREngine:
    """Thread-safe OCR engine with model caching.

    Manages EasyOCR reader instances, caching them by language tuple
    to avoid repeated model loading. Supports asynchronous processing
    via a result queue.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or AppSettings()
        self._cache: dict[tuple[str, ...], Any] = {}
        self._lock = threading.Lock()
        self._result_queue: queue.Queue[DetectionResult] = queue.Queue()

    def _get_reader(self, languages: list[str]) -> Any:
        """Get or create a cached EasyOCR reader.

        Args:
            languages: List of language codes.

        Returns:
            EasyOCR Reader instance.
        """
        import easyocr

        key = tuple(sorted(languages))

        with self._lock:
            if key not in self._cache:
                logger.info("Loading OCR model for languages: %s", languages)
                self._cache[key] = easyocr.Reader(
                    list(key),
                    gpu=self._settings.gpu_enabled,
                )
                logger.info("OCR model loaded successfully")

        return self._cache[key]

    def detect_text(
        self,
        frame,
        languages: list[str] | None = None,
        threshold: float | None = None,
    ) -> DetectionResult:
        """Run OCR on a frame and return filtered detections.

        Args:
            frame: Image array (numpy) to process.
            languages: Language codes for OCR. Uses default if None.
            threshold: Confidence threshold for filtering. Uses default if None.

        Returns:
            DetectionResult with detections and metadata.
        """
        langs = languages or [self._settings.default_language]
        conf_threshold = threshold if threshold is not None else self._settings.default_confidence

        try:
            reader = self._get_reader(langs)
            raw_results = reader.readtext(frame)
            detections = [
                item for item in raw_results if item[2] >= conf_threshold
            ]
            return DetectionResult(detections=detections, languages=langs)
        except Exception as e:
            logger.error("OCR detection failed: %s", e)
            return DetectionResult(
                detections=[],
                languages=langs,
                success=False,
                error=str(e),
            )

    def detect_text_async(
        self,
        frame,
        languages: list[str] | None = None,
        threshold: float | None = None,
        callback: Callable[[DetectionResult], None] | None = None,
    ) -> None:
        """Run OCR in a background thread.

        Args:
            frame: Image array to process.
            languages: Language codes for OCR.
            threshold: Confidence threshold.
            callback: Function called with DetectionResult when complete.
        """
        def _worker() -> None:
            result = self.detect_text(frame, languages, threshold)
            if callback:
                callback(result)
            self._result_queue.put(result)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def get_result(self, timeout: float = 30.0) -> DetectionResult | None:
        """Get the next result from the async queue.

        Args:
            timeout: Seconds to wait for a result.

        Returns:
            DetectionResult or None if timeout.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_cache(self) -> None:
        """Clear the model cache to free memory."""
        with self._lock:
            self._cache.clear()
            logger.info("OCR model cache cleared")

    @property
    def cache_size(self) -> int:
        """Return number of cached models."""
        return len(self._cache)
```

- [ ] **Step 2: Create src/tests/test_ocr_engine.py**

```python
"""Tests for OCR engine module."""

import queue
from unittest.mock import MagicMock, patch

import pytest

from text_detector.config import AppSettings
from text_detector.ocr_engine import DetectionResult, OCREngine


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(default_language="en", default_confidence=0.3)


@pytest.fixture
def engine(settings: AppSettings) -> OCREngine:
    return OCREngine(settings)


def test_detection_result_creation() -> None:
    result = DetectionResult(
        detections=[([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "test", 0.9)],
        languages=["en"],
    )
    assert result.success is True
    assert result.error is None
    assert len(result.detections) == 1


def test_detection_result_error() -> None:
    result = DetectionResult(detections=[], languages=["en"], success=False, error="failed")
    assert result.success is False
    assert result.error == "failed"


def test_ocr_engine_init_with_settings(settings: AppSettings) -> None:
    engine = OCREngine(settings)
    assert engine.cache_size == 0


def test_ocr_engine_detect_text_success(engine: OCREngine) -> None:
    frame = MagicMock()
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "hello", 0.9),
        ([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "low", 0.1),
    ]

    with patch("easyocr.Reader", return_value=mock_reader):
        result = engine.detect_text(frame, languages=["en"], threshold=0.5)

    assert result.success is True
    assert len(result.detections) == 1
    assert result.detections[0][1] == "hello"


def test_ocr_engine_detect_text_failure(engine: OCREngine) -> None:
    frame = MagicMock()

    with patch("easyocr.Reader", side_effect=RuntimeError("model error")):
        result = engine.detect_text(frame, languages=["en"])

    assert result.success is False
    assert result.error == "model error"
    assert result.detections == []


def test_ocr_engine_model_caching(engine: OCREngine) -> None:
    mock_reader = MagicMock()

    with patch("easyocr.Reader", return_value=mock_reader):
        engine.detect_text(MagicMock(), languages=["en"])
        engine.detect_text(MagicMock(), languages=["en"])

    assert engine.cache_size == 1
    assert mock_reader.readtext.call_count == 2


def test_ocr_engine_cache_clearing(engine: OCREngine) -> None:
    mock_reader = MagicMock()

    with patch("easyocr.Reader", return_value=mock_reader):
        engine.detect_text(MagicMock(), languages=["en"])

    assert engine.cache_size == 1
    engine.clear_cache()
    assert engine.cache_size == 0


def test_ocr_engine_detect_text_async(engine: OCREngine) -> None:
    frame = MagicMock()
    callback = MagicMock()
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []

    with patch("easyocr.Reader", return_value=mock_reader):
        engine.detect_text_async(frame, languages=["en"], callback=callback)

    result = engine.get_result(timeout=5.0)
    assert result is not None
    assert result.success is True
    callback.assert_called_once_with(result)


def test_ocr_engine_get_result_timeout(engine: OCREngine) -> None:
    result = engine.get_result(timeout=0.1)
    assert result is None
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest src/tests/test_ocr_engine.py -v
```
Expected: 9 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/text_detector/ocr_engine.py src/tests/test_ocr_engine.py
git commit -m "feat: add OCR engine with model caching, threading, and tests"
```

---

### Task 6: GUI Module (Refactored) + Entry Point

**Files:**
- Create: `src/text_detector/text_detector.py`
- Create: `src/text_detector/__main__.py`

- [ ] **Step 1: Create src/text_detector/text_detector.py**

```python
"""GUI application for real-time text recognition."""

import csv
import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from text_detector.config import SETTINGS, THEME
from text_detector.image_processor import bgr_to_rgb, draw_boxes_with_colors, filter_text
from text_detector.ocr_engine import DetectionResult, OCREngine
from text_detector.utils.logging_setup import get_logger, setup_logging
from text_detector.utils.path_helpers import get_assets_dir

logger = get_logger("gui")


class TextRecognitionApp:
    """A GUI application for real-time text recognition using EasyOCR."""

    def __init__(self, main: tk.Tk) -> None:
        self.cap: cv2.VideoCapture | None = None
        self.root = main
        self.root.title("Text Recognition App")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        self.root.configure(bg=THEME.background)

        self.capture_active = False
        self.frame_counter = 0
        self.detected_text: list[tuple] = []
        self.current_frame: np.ndarray | None = None
        self.language_var = tk.StringVar(value=SETTINGS.default_language)
        self.threshold_var = tk.DoubleVar(value=SETTINGS.default_confidence)
        self.gpu_var = tk.BooleanVar(value=SETTINGS.gpu_enabled)
        self.history: list[dict[str, str | float]] = []
        self.current_language: str = SETTINGS.default_language

        self.engine = OCREngine(SETTINGS)
        self.ocr_result: DetectionResult | None = None
        self.ocr_lock = threading.Lock()

        self._create_widgets()
        self._configure_icon()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def _configure_icon(self) -> None:
        icon_path = get_assets_dir() / "icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except tk.TclError:
                pass

    def _create_widgets(self) -> None:
        self.main_frame = tk.Frame(self.root, bg=THEME.frame_bg, bd=2, relief="groove")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.title_label = tk.Label(
            self.main_frame,
            text="Text Recognition",
            font=("Arial", 24, "bold"),
            bg=THEME.title_bg,
            fg=THEME.title_fg,
            pady=10,
        )
        self.title_label.pack(fill="x")

        self._create_control_frame()
        self._create_settings_frame()
        self._create_result_frame()

        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10, "italic"),
            bg=THEME.title_bg,
            fg=THEME.title_fg,
            relief="ridge",
            anchor="w",
        )
        self.status_bar.pack(side="bottom", fill="x")

    def _create_control_frame(self) -> None:
        self.control_frame = tk.Frame(self.main_frame, bg=THEME.background)
        self.control_frame.pack(fill="x", pady=10)

        buttons = [
            ("Start Webcam", self.start_capture, THEME.success, THEME.success_active),
            ("Stop Capture", self.stop_capture, THEME.danger, THEME.danger_active),
            ("Load Image", self.load_image, THEME.primary, THEME.primary_active),
            ("Save Results", self.save_results, THEME.warning, THEME.warning_active),
            ("Clear", self.clear_results, THEME.neutral, THEME.neutral_active),
        ]

        for text, cmd, bg, active in buttons:
            btn = tk.Button(
                self.control_frame,
                text=text,
                command=cmd,
                font=("Arial", 12, "bold"),
                bg=bg,
                fg=THEME.white,
                activebackground=active,
                relief="ridge",
                padx=10,
                pady=5,
            )
            btn.pack(side="left", padx=5)

        about_btn = tk.Button(
            self.control_frame,
            text="About",
            command=self.show_about,
            font=("Arial", 12, "bold"),
            bg=THEME.about,
            fg=THEME.white,
            activebackground=THEME.about_active,
            relief="ridge",
            padx=10,
            pady=5,
        )
        about_btn.pack(side="right", padx=5)

    def _create_settings_frame(self) -> None:
        self.settings_frame = tk.Frame(self.main_frame, bg=THEME.background)
        self.settings_frame.pack(fill="x", pady=5)

        tk.Label(self.settings_frame, text="OCR Language:", bg=THEME.background).pack(side="left", padx=(5, 0))
        self.language_menu = tk.OptionMenu(
            self.settings_frame,
            self.language_var,
            *SETTINGS.available_languages,
            command=self._language_changed,
        )
        self.language_menu.pack(side="left", padx=5)

        tk.Label(self.settings_frame, text="Confidence threshold:", bg=THEME.background).pack(side="left", padx=(10, 0))
        self.threshold_scale = tk.Scale(
            self.settings_frame,
            variable=self.threshold_var,
            from_=SETTINGS.min_confidence,
            to=SETTINGS.max_confidence,
            resolution=SETTINGS.confidence_resolution,
            orient="horizontal",
            length=220,
            bg=THEME.background,
        )
        self.threshold_scale.pack(side="left", padx=5)

        self.gpu_check = tk.Checkbutton(
            self.settings_frame,
            text="Use GPU",
            variable=self.gpu_var,
            command=self._gpu_changed,
            bg=THEME.background,
        )
        self.gpu_check.pack(side="left", padx=10)

    def _create_result_frame(self) -> None:
        self.result_frame = tk.Frame(self.main_frame, bg=THEME.result_frame_bg, bd=2, relief="sunken")
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = tk.Label(self.result_frame, bg=THEME.result_frame_bg)
        self.image_label.pack(side="left", fill="both", expand=True)

        self.text_frame = tk.Frame(self.result_frame, bg=THEME.white, bd=1, relief="sunken")
        self.text_frame.pack(side="right", fill="y", padx=5, pady=5)

        self.text_title = tk.Label(
            self.text_frame,
            text="Detected Text",
            font=("Arial", 14, "bold"),
            bg=THEME.primary,
            fg=THEME.white,
            padx=10,
            pady=10,
        )
        self.text_title.pack(fill="x")

        self.text_output = tk.Text(self.text_frame, wrap="word", width=35, bg=THEME.text_output_bg)
        self.text_output.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.text_output.config(state="disabled")

    def _language_changed(self, value: str) -> None:
        self.language_var.set(value)
        self.current_language = value
        self.engine.clear_cache()
        self.status_bar.config(text=f"Language set to {value}", bg=THEME.title_bg)
        logger.info("Language changed to %s", value)

    def _gpu_changed(self) -> None:
        self.engine.clear_cache()
        self.engine._settings.gpu_enabled = self.gpu_var.get()
        status = "enabled" if self.gpu_var.get() else "disabled"
        self.status_bar.config(text=f"GPU {status} (model will reload)", bg=THEME.title_bg)
        logger.info("GPU %s", status)

    def start_capture(self) -> None:
        if self.capture_active:
            return
        self.capture_active = True
        self.frame_counter = 0
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Unable to open webcam.")
            self.status_bar.config(text="Webcam capture started", bg=THEME.success)
            logger.info("Webcam capture started")
        except Exception as exc:
            self.capture_active = False
            messagebox.showerror("Webcam Error", str(exc))
            self.status_bar.config(text="Failed to start webcam", bg=THEME.danger)
            logger.error("Webcam error: %s", exc)

    def stop_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.capture_active = False
        self.status_bar.config(text="Capture stopped", bg=THEME.danger)
        logger.info("Capture stopped")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*")],
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Unable to open selected image.")
            return

        self.stop_capture()
        self.current_frame = frame
        self._process_current_frame()
        self.status_bar.config(text=f"Loaded image: {Path(path).name}", bg=THEME.primary)
        logger.info("Loaded image: %s", path)

    def _process_current_frame(self) -> None:
        if self.current_frame is None:
            return

        def _on_result(result: DetectionResult) -> None:
            with self.ocr_lock:
                self.ocr_result = result
            self.root.after(0, self._apply_ocr_result)

        self.engine.detect_text_async(
            self.current_frame,
            languages=[self.current_language],
            threshold=self.threshold_var.get(),
            callback=_on_result,
        )
        self.status_bar.config(text="Processing...", bg=THEME.warning)

    def _apply_ocr_result(self) -> None:
        with self.ocr_lock:
            if self.ocr_result is None or not self.ocr_result.success:
                self.status_bar.config(text="OCR failed", bg=THEME.danger)
                return
            self.detected_text = self.ocr_result.detections

        frame = draw_boxes_with_colors(self.current_frame.copy(), self.detected_text)
        self._show_image(frame)
        self._update_text_output()
        self._add_to_history()
        self.status_bar.config(text="Detection complete", bg=THEME.success)

    def update_frame(self) -> None:
        if self.capture_active and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.frame_counter += 1
                if self.frame_counter % SETTINGS.frame_skip == 0:
                    self._process_current_frame()

        self.root.after(50, self.update_frame)

    def _show_image(self, frame: np.ndarray) -> None:
        image = bgr_to_rgb(frame)
        pil_image = Image.fromarray(image)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

    def _update_text_output(self) -> None:
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        if self.detected_text:
            for bbox, text, confidence in self.detected_text:
                self.text_output.insert(tk.END, f"{text} (confidence: {confidence:.2f})\n")
        else:
            self.text_output.insert(tk.END, "No text detected yet.\n")
        self.text_output.config(state="disabled")

    def _add_to_history(self) -> None:
        for bbox, text, confidence in self.detected_text:
            self.history.append(
                {"text": text, "confidence": round(confidence, 2), "language": self.current_language}
            )
        if len(self.history) > SETTINGS.max_history:
            self.history = self.history[-SETTINGS.max_history:]

    def save_results(self) -> None:
        if not self.detected_text:
            messagebox.showwarning("Warning", "No detected text to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save detected text",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
            ],
            initialfile="detected_text_results",
        )
        if not save_path:
            return

        path = Path(save_path)
        suffix = path.suffix.lower()

        try:
            if suffix == ".txt":
                self._save_txt(path)
            elif suffix == ".csv":
                self._save_csv(path)
            elif suffix == ".json":
                self._save_json(path)
            else:
                self._save_txt(path)

            messagebox.showinfo("Saved", f"Results saved to {save_path}")
            logger.info("Results saved to %s", save_path)
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
            logger.error("Save error: %s", exc)

    def _save_txt(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for bbox, text, confidence in self.detected_text:
                f.write(f"{text} (confidence: {confidence:.2f})\n")

    def _save_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "confidence", "language"])
            for bbox, text, confidence in self.detected_text:
                writer.writerow([text, f"{confidence:.2f}", self.current_language])

    def _save_json(self, path: Path) -> None:
        data = [
            {"text": text, "confidence": round(confidence, 2), "language": self.current_language}
            for bbox, text, confidence in self.detected_text
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear_results(self) -> None:
        self.stop_capture()
        self.current_frame = None
        self.detected_text = []
        self.ocr_result = None
        self.image_label.config(image=None)
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, "Ready. Load an image or start the webcam.")
        self.text_output.config(state="disabled")
        self.status_bar.config(text="Cleared", bg=THEME.neutral)

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Text Recognition App v2.0\n\n"
            "Developed using OpenCV, EasyOCR, and Tkinter.\n"
            "Load images or use your webcam to detect text in real time.\n"
            "Use the language selector, threshold slider, and GPU toggle to improve accuracy.",
        )

    def on_closing(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.engine.clear_cache()
        self.root.destroy()
        logger.info("Application closed")
```

- [ ] **Step 2: Create src/text_detector/__main__.py**

```python
"""Entry point for the text detection application."""

import tkinter as tk

from text_detector.text_detector import TextRecognitionApp
from text_detector.utils.logging_setup import setup_logging


def main() -> None:
    """Run the text recognition application."""
    setup_logging()
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/text_detector/text_detector.py src/text_detector/__main__.py
git commit -m "feat: add refactored GUI with threading, multi-export, GPU toggle, and entry point"
```

---

### Task 7: Documentation (README, CHANGELOG, LICENSE)

**Files:**
- Modify: `README.md`
- Create: `CHANGELOG.md`
- Create: `LICENSE`

- [ ] **Step 1: Update README.md**

```markdown
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
```

- [ ] **Step 2: Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
```

- [ ] **Step 3: Create LICENSE**

```
MIT License

Copyright (c) 2026 Text Detection App Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md LICENSE
git commit -m "docs: update README, add CHANGELOG and LICENSE"
```

---

### Task 8: Run All Tests, Lint, and Final Cleanup

- [ ] **Step 1: Run all tests**

```bash
python -m pytest src/tests -v --cov=text_detector --cov-report=term-missing
```
Expected: 31 tests PASS, coverage >= 80%

- [ ] **Step 2: Run linter**

```bash
python -m ruff check src/
```
Expected: No errors

- [ ] **Step 3: Format code**

```bash
python -m ruff format src/
```

- [ ] **Step 4: Create backward-compat symlink**

```bash
ln -sf ../src/text_detector text_detector
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: run tests, lint, format, and create backward-compat symlink"
```

---

## Self-Review

### 1. Spec Coverage Check

| Spec Requirement | Task |
|---|---|
| Modular architecture | Tasks 1-6 |
| Type hints | All tasks (code includes them) |
| Docstrings | All tasks (code includes them) |
| Threading for OCR | Tasks 5, 6 |
| Logging | Tasks 1, 2, 6 |
| pathlib | Tasks 2, 6, 7 |
| pyproject.toml | Task 1 |
| Tests (pytest) | Tasks 2-5 |
| Makefile | Task 1 |
| GPU option | Tasks 3, 6 |
| Multi-language support | Tasks 3, 5, 6 |
| Export CSV/JSON | Task 6 |
| History tracking | Tasks 3, 6 |
| Progress feedback | Task 6 ("Processing..." status) |
| Theme colors as constants | Task 3, 6 |
| Unfrozen requirements | Task 1 |
| CHANGELOG | Task 7 |
| LICENSE | Task 7 |

All spec requirements covered. No gaps.

### 2. Placeholder Scan

No TBD, TODO, or placeholder patterns found. All code blocks contain complete implementations and tests.

### 3. Type Consistency

- `DetectionResult` defined in Task 5, used in Task 6 - consistent
- `AppSettings` defined in Task 3, used in Tasks 5, 6 - consistent
- `THEME` defined in Task 3, used in Task 6 - consistent
- `SETTINGS` defined in Task 3, used in Tasks 5, 6 - consistent
- All function signatures match their usage
- Import paths are consistent across all tasks
