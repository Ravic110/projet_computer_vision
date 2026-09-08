"""Tests for OCR engine module."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from text_detector.config import AppSettings
from text_detector.ocr_engine import DetectionResult, OCREngine


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(languages=["en"], default_confidence=0.3)


def _stub_two_stage(reader: MagicMock, results: list, boxes: list | None = None) -> MagicMock:
    """Wire a mock reader for the detect() then recognize() pipeline.

    One box is reported by default so recognition is reached at all; an
    empty list makes detect_text return early with no detections.
    """
    reader.detect.return_value = ([[[0, 10, 0, 10]] if boxes is None else boxes], [[]])
    reader.recognize.return_value = results
    return reader


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
    mock_reader = _stub_two_stage(
        MagicMock(),
        [
            ([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]], "hello", 0.9),
            ([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]], "low", 0.1),
        ],
    )

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
    mock_reader = _stub_two_stage(MagicMock(), [])

    with patch("easyocr.Reader", return_value=mock_reader):
        engine.detect_text(frame, languages=["en"])
        engine.detect_text(frame, languages=["en"])

    assert engine.cache_size == 1
    assert mock_reader.detect.call_count == 2


def test_ocr_engine_cache_clearing(engine: OCREngine) -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mock_reader = _stub_two_stage(MagicMock(), [])

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
    reader = _stub_two_stage(MagicMock(), _paragraph_output())

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
    reader = _stub_two_stage(MagicMock(), _paragraph_output())

    with patch.object(engine, "_get_reader", return_value=reader):
        result = engine.detect_text(
            np.zeros((100, 100, 3), dtype=np.uint8), languages=["fr"], threshold=0.99
        )

    assert len(result.detections) == 1
    engine.shutdown()


def test_paragraph_boxes_are_scaled_like_normal_detections() -> None:
    import numpy as np

    engine = OCREngine(AppSettings(languages=["fr"], paragraph_merge=True, ocr_max_width=50))
    reader = _stub_two_stage(MagicMock(), _paragraph_output())

    with patch.object(engine, "_get_reader", return_value=reader):
        result = engine.detect_text(np.zeros((100, 100, 3), dtype=np.uint8), languages=["fr"])

    assert result.detections[0][0][0] == [20.0, 20.0]  # 10 / (50/100)
    engine.shutdown()


def test_reader_cache_is_keyed_by_language_combination(engine: OCREngine) -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    with patch("easyocr.Reader", return_value=_stub_two_stage(MagicMock(), [])) as reader_factory:
        engine.detect_text(frame, languages=["en"])
        engine.detect_text(frame, languages=["fr", "en"])
        engine.detect_text(frame, languages=["en", "fr"])  # same set, different order
        engine.detect_text(frame, languages=["en"])

    assert reader_factory.call_count == 2
    assert engine.cache_size == 2


def test_engine_falls_back_to_the_settings_languages() -> None:
    import numpy as np

    engine = OCREngine(AppSettings(languages=["fr", "de"]))
    reader = _stub_two_stage(MagicMock(), [], boxes=[])
    with patch.object(engine, "_get_reader", return_value=reader) as get_reader:
        result = engine.detect_text(np.zeros((10, 10, 3), dtype=np.uint8))

    get_reader.assert_called_once_with(["fr", "de"])
    assert result.languages == ["fr", "de"]
    engine.shutdown()


def test_poll_result_returns_none_when_nothing_is_ready(engine: OCREngine) -> None:
    assert engine.poll_result() is None


def test_poll_result_returns_a_finished_result(settings: AppSettings) -> None:
    engine = OCREngine(settings)
    try:
        published = DetectionResult(detections=[], languages=["en"])
        engine._publish_result(published)
        assert engine.poll_result() is published
        assert engine.poll_result() is None
    finally:
        engine.shutdown()


def test_settings_are_readable_without_touching_privates(settings: AppSettings) -> None:
    engine = OCREngine(settings)
    try:
        assert engine.settings is settings
    finally:
        engine.shutdown()


def test_shutdown_returns_promptly(settings: AppSettings) -> None:
    """Shutdown must wake the idle worker, not wait out its poll timeout."""
    import time

    engine = OCREngine(settings)
    start = time.monotonic()
    engine.shutdown()
    assert time.monotonic() - start < 0.3


class TestPreload:
    """Loading the model ahead of the first frame.

    The model takes seconds to load, and doing it inside the first
    detection makes that detection look far slower than the rest.
    """

    def test_preload_loads_the_model_for_the_current_languages(self, settings: AppSettings) -> None:
        engine = OCREngine(settings)
        try:
            with patch("easyocr.Reader", return_value=MagicMock()) as factory:
                engine.preload().join(timeout=10)
            assert factory.call_count == 1
            assert factory.call_args.args[0] == ["en"]
            assert engine.cache_size == 1
        finally:
            engine.shutdown()

    def test_preload_leaves_the_engine_free_to_accept_a_frame(self, settings: AppSettings) -> None:
        # Running the load on the work queue would make the engine report
        # itself busy, and the GUI drops frames submitted while it is.
        engine = OCREngine(settings)
        try:
            with patch("easyocr.Reader", return_value=MagicMock()):
                thread = engine.preload()
                assert engine.is_busy is False
                thread.join(timeout=10)
        finally:
            engine.shutdown()

    def test_a_failed_preload_does_not_escape(self, settings: AppSettings) -> None:
        engine = OCREngine(settings)
        try:
            with patch("easyocr.Reader", side_effect=RuntimeError("model missing")):
                engine.preload().join(timeout=10)
            assert engine.cache_size == 0
        finally:
            engine.shutdown()

    def test_preload_twice_loads_once(self, settings: AppSettings) -> None:
        engine = OCREngine(settings)
        try:
            with patch("easyocr.Reader", return_value=MagicMock()) as factory:
                engine.preload().join(timeout=10)
                engine.preload().join(timeout=10)
            assert factory.call_count == 1
        finally:
            engine.shutdown()


class TestTwoStageDetection:
    """Detection on a small frame, recognition on a larger one.

    reader.detect() costs about 95% of a pass and scales with pixel count,
    while reader.recognize() is cheap. Splitting them buys the detector's
    speed at low resolution without reading characters at that resolution.
    """

    def _frame(self):
        import numpy as np

        return np.zeros((900, 1600, 3), dtype=np.uint8)

    def _reader(self, boxes=None, free=None, results=None):
        reader = MagicMock()
        reader.detect.return_value = (
            [boxes if boxes is not None else [[10, 50, 20, 40]]],
            [free if free is not None else []],
        )
        reader.recognize.return_value = (
            results
            if results is not None
            else [([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], "hi", 0.9)]
        )
        return reader

    def _configure(self, engine, detect=400, recognise=800, preprocess=False):
        engine.settings.ocr_max_width = recognise
        engine.settings.detect_max_width = detect
        engine.settings.preprocess_enabled = preprocess

    def test_detection_runs_on_a_smaller_frame_than_recognition(self, engine: OCREngine) -> None:
        self._configure(engine)
        reader = self._reader()
        with patch.object(engine, "_get_reader", return_value=reader):
            engine.detect_text(self._frame(), threshold=0.0)
        assert reader.detect.call_args.args[0].shape[1] == 400
        assert reader.recognize.call_args.args[0].shape[1] == 800

    def test_boxes_move_from_detection_space_to_recognition_space(self, engine: OCREngine) -> None:
        self._configure(engine)
        reader = self._reader(boxes=[[10, 50, 20, 40]])
        with patch.object(engine, "_get_reader", return_value=reader):
            engine.detect_text(self._frame(), threshold=0.0)
        assert reader.recognize.call_args.kwargs["horizontal_list"] == [[20, 100, 40, 80]]

    def test_rotated_boxes_move_too(self, engine: OCREngine) -> None:
        self._configure(engine)
        reader = self._reader(boxes=[], free=[[[10, 20], [30, 20], [30, 40], [10, 40]]])
        with patch.object(engine, "_get_reader", return_value=reader):
            engine.detect_text(self._frame(), threshold=0.0)
        assert reader.recognize.call_args.kwargs["free_list"] == [
            [[20, 40], [60, 40], [60, 80], [20, 80]]
        ]

    def test_nothing_detected_skips_recognition(self, engine: OCREngine) -> None:
        self._configure(engine)
        reader = self._reader(boxes=[], free=[])
        with patch.object(engine, "_get_reader", return_value=reader):
            result = engine.detect_text(self._frame(), threshold=0.0)
        reader.recognize.assert_not_called()
        assert result.success is True
        assert result.detections == []

    def test_results_come_back_in_full_frame_coordinates(self, engine: OCREngine) -> None:
        self._configure(engine)
        # The 1600px frame is halved for recognition, so a box read at
        # x=10 sits at x=20 in the frame the caller passed in.
        reader = self._reader(
            results=[([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], "hi", 0.9)]
        )
        with patch.object(engine, "_get_reader", return_value=reader):
            result = engine.detect_text(self._frame(), threshold=0.0)
        assert result.detections[0][0] == [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]

    def test_paragraph_mode_reaches_recognize(self, engine: OCREngine) -> None:
        self._configure(engine)
        engine.settings.paragraph_merge = True
        reader = self._reader(results=[([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "hi")])
        with patch.object(engine, "_get_reader", return_value=reader):
            result = engine.detect_text(self._frame(), threshold=0.0)
        assert reader.recognize.call_args.kwargs["paragraph"] is True
        assert result.detections[0][2] is None

    def test_denoising_is_paid_once_and_shared_by_both_stages(self, engine: OCREngine) -> None:
        # Denoising the recognition frame and deriving the detection frame
        # from it costs one pass instead of two.
        self._configure(engine, preprocess=True)
        reader = self._reader()
        with (
            patch.object(engine, "_get_reader", return_value=reader),
            patch("text_detector.ocr_engine.preprocess_for_ocr", side_effect=lambda f: f) as pre,
        ):
            engine.detect_text(self._frame(), threshold=0.0)
        assert pre.call_count == 1

    def test_a_detection_width_above_the_recognition_width_is_ignored(
        self, engine: OCREngine
    ) -> None:
        # Detecting at a higher resolution than recognition would cost more
        # for no benefit; both stages then share one frame.
        self._configure(engine, detect=1200, recognise=800)
        reader = self._reader()
        with patch.object(engine, "_get_reader", return_value=reader):
            engine.detect_text(self._frame(), threshold=0.0)
        assert reader.detect.call_args.args[0].shape[1] == 800
        assert reader.recognize.call_args.args[0].shape[1] == 800
