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
