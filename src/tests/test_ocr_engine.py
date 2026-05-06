"""Tests for OCR engine module."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from text_detector.config import AppSettings
from text_detector.ocr_engine import DetectionResult, OCREngine


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(default_language="en", default_confidence=0.3)


@pytest.fixture
def engine(settings: AppSettings) -> Generator[OCREngine, None, None]:
    eng = OCREngine(settings)
    yield eng
    eng.shutdown()


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
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]], "hello", 0.9),
        ([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]], "low", 0.1),
    ]

    with patch.object(engine, "_get_reader", return_value=mock_reader):
        result = engine.detect_text(frame, languages=["en"], threshold=0.5)

    assert result.success is True
    assert len(result.detections) == 1
    assert result.detections[0][1] == "hello"


def test_ocr_engine_detect_text_failure(engine: OCREngine) -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    with patch.object(engine, "_get_reader", side_effect=RuntimeError("model error")):
        result = engine.detect_text(frame, languages=["en"])

    assert result.success is False
    assert result.error == "model error"
    assert result.detections == []


def test_ocr_engine_model_caching(engine: OCREngine) -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mock_reader = MagicMock()

    with patch("easyocr.Reader", return_value=mock_reader):
        engine.detect_text(frame, languages=["en"])
        engine.detect_text(frame, languages=["en"])

    assert engine.cache_size == 1
    assert mock_reader.readtext.call_count == 2


def test_ocr_engine_cache_clearing(engine: OCREngine) -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mock_reader = MagicMock()

    with patch("easyocr.Reader", return_value=mock_reader):
        engine.detect_text(frame, languages=["en"])

    assert engine.cache_size == 1
    engine.clear_cache()
    assert engine.cache_size == 0


def test_ocr_engine_detect_text_async_returns_true(settings: AppSettings) -> None:
    """Test that async submission succeeds when worker is free."""
    engine = OCREngine(settings)
    frame = MagicMock()

    result = engine.detect_text_async(frame, languages=["en"])
    assert result is True
    assert engine.is_busy is True
    engine.shutdown()


def test_ocr_engine_detect_text_async_drops_when_busy(settings: AppSettings) -> None:
    """Test that async submission is dropped when worker is busy."""
    engine = OCREngine(settings)
    frame = MagicMock()

    # Queue maxsize is 1, so first submission should succeed
    assert engine.detect_text_async(frame, languages=["en"]) is True
    # Second submission should be dropped (queue full or worker busy)
    assert engine.detect_text_async(frame, languages=["en"]) is False
    engine.shutdown()


def test_ocr_engine_get_result_timeout(engine: OCREngine) -> None:
    result = engine.get_result(timeout=0.1)
    assert result is None
