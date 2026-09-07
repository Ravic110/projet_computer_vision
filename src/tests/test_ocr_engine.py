"""Tests for OCR engine module."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from text_detector.config import AppSettings
from text_detector.ocr_engine import DetectionResult, OCREngine


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(languages=["en"], default_confidence=0.3)


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


def _drain_one(engine: OCREngine) -> None:
    """Submit one frame and wait for the worker to finish it."""
    import time

    engine.detect_text_async(MagicMock(), languages=["en"], callback=lambda _r: None)
    deadline = time.monotonic() + 5.0
    while engine.is_busy and time.monotonic() < deadline:
        time.sleep(0.005)


def test_result_queue_does_not_grow_without_bound(settings: AppSettings) -> None:
    from text_detector.ocr_engine import _MAX_PENDING_RESULTS

    engine = OCREngine(settings)
    stub = DetectionResult(detections=[], languages=["en"])
    with patch.object(engine, "detect_text", return_value=stub):
        for _ in range(_MAX_PENDING_RESULTS + 15):
            _drain_one(engine)

    assert engine._result_queue.qsize() <= _MAX_PENDING_RESULTS
    engine.shutdown()


def test_result_queue_keeps_the_most_recent_results(settings: AppSettings) -> None:
    from text_detector.ocr_engine import _MAX_PENDING_RESULTS

    engine = OCREngine(settings)
    for index in range(_MAX_PENDING_RESULTS + 5):
        stub = DetectionResult(detections=[], languages=[f"lang{index}"])
        with patch.object(engine, "detect_text", return_value=stub):
            _drain_one(engine)

    newest = _MAX_PENDING_RESULTS + 4
    results = []
    while engine._result_queue.qsize():
        result = engine.get_result(timeout=0.1)
        assert result is not None
        results.append(result)
    assert results[-1].languages == [f"lang{newest}"]
    engine.shutdown()


def _paragraph_output() -> list:
    """What easyocr returns with paragraph=True: [bbox, text], no confidence."""
    return [[[[10, 10], [90, 10], [90, 30], [10, 30]], "un paragraphe entier"]]


def test_paragraph_mode_yields_detections_without_confidence() -> None:
    import numpy as np

    engine = OCREngine(AppSettings(languages=["fr"], paragraph_merge=True))
    reader = MagicMock()
    reader.readtext.return_value = _paragraph_output()

    with patch.object(engine, "_get_reader", return_value=reader):
        result = engine.detect_text(np.zeros((100, 100, 3), dtype=np.uint8), languages=["fr"])

    assert result.success is True
    assert len(result.detections) == 1
    bbox, text, confidence = result.detections[0]
    assert text == "un paragraphe entier"
    assert confidence is None
    assert len(bbox) == 4
    engine.shutdown()


def test_paragraph_mode_ignores_the_confidence_threshold() -> None:
    import numpy as np

    engine = OCREngine(AppSettings(languages=["fr"], paragraph_merge=True))
    reader = MagicMock()
    reader.readtext.return_value = _paragraph_output()

    with patch.object(engine, "_get_reader", return_value=reader):
        result = engine.detect_text(
            np.zeros((100, 100, 3), dtype=np.uint8), languages=["fr"], threshold=0.99
        )

    assert len(result.detections) == 1
    engine.shutdown()


def test_paragraph_boxes_are_scaled_like_normal_detections() -> None:
    import numpy as np

    engine = OCREngine(AppSettings(languages=["fr"], paragraph_merge=True, ocr_max_width=50))
    reader = MagicMock()
    reader.readtext.return_value = _paragraph_output()

    with patch.object(engine, "_get_reader", return_value=reader):
        result = engine.detect_text(np.zeros((100, 100, 3), dtype=np.uint8), languages=["fr"])

    assert result.detections[0][0][0] == [20.0, 20.0]  # 10 / (50/100)
    engine.shutdown()


def test_reader_cache_is_keyed_by_language_combination(engine: OCREngine) -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    with patch("easyocr.Reader", return_value=MagicMock()) as reader_factory:
        engine.detect_text(frame, languages=["en"])
        engine.detect_text(frame, languages=["fr", "en"])
        engine.detect_text(frame, languages=["en", "fr"])  # same set, different order
        engine.detect_text(frame, languages=["en"])

    assert reader_factory.call_count == 2
    assert engine.cache_size == 2


def test_engine_falls_back_to_the_settings_languages() -> None:
    import numpy as np

    engine = OCREngine(AppSettings(languages=["fr", "de"]))
    reader = MagicMock()
    reader.readtext.return_value = []
    with patch.object(engine, "_get_reader", return_value=reader) as get_reader:
        result = engine.detect_text(np.zeros((10, 10, 3), dtype=np.uint8))

    get_reader.assert_called_once_with(["fr", "de"])
    assert result.languages == ["fr", "de"]
    engine.shutdown()
