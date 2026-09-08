"""OCR engine wrapper for EasyOCR with model caching and threading support."""

import contextlib
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from text_detector.config import AppSettings
from text_detector.image_processor import (
    filter_text,
    preprocess_for_ocr,
    resize_frame_for_ocr,
    scale_detections,
    scale_free_boxes,
    scale_horizontal_boxes,
)
from text_detector.utils.logging_setup import get_logger

logger = get_logger("ocr_engine")

_MAX_CACHE_SIZE = 2
# Pushed onto the work queue to wake an idle worker at shutdown. A work
# item is always a tuple, so None is unambiguous as the sentinel.
_STOP = None
_MAX_PENDING_RESULTS = 16

BBox = list[list[float]]
Detection = tuple[BBox, str, float | None]


def _normalise_results(raw_results: list) -> list[Detection]:
    """Bring EasyOCR output to a single (bbox, text, confidence) shape.

    In paragraph mode EasyOCR merges word boxes into blocks and returns
    [bbox, text] with no confidence at all, so the third slot is filled
    with None rather than an invented score.
    """
    detections: list[Detection] = []
    for item in raw_results:
        bbox, text = item[0], item[1]
        confidence = item[2] if len(item) > 2 else None
        detections.append((bbox, text, confidence))
    return detections


@dataclass
class DetectionResult:
    """Container for OCR detection results."""

    detections: list[Detection]
    languages: list[str]
    success: bool = True
    error: str | None = None


class OCREngine:
    """Thread-safe OCR engine with model caching.

    Uses a single worker thread with a bounded queue to prevent
    thread explosion and memory exhaustion. Only processes one
    frame at a time; excess frames are dropped.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or AppSettings()
        self._cache: dict[tuple[str, ...], Any] = {}
        self._lock = threading.Lock()
        self._result_queue: queue.Queue[DetectionResult] = queue.Queue(maxsize=_MAX_PENDING_RESULTS)
        # Carries either a work item or the _STOP sentinel.
        self._work_queue: queue.Queue[tuple | None] = queue.Queue(maxsize=1)
        self._worker_thread: threading.Thread | None = None
        self._preload_thread: threading.Thread | None = None
        self._running = True
        self._is_processing = False
        self._start_worker()

    def _start_worker(self) -> None:
        """Start the single OCR worker thread."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="ocr-worker"
        )
        self._worker_thread.start()

    def preload(self) -> threading.Thread:
        """Load the model for the current languages ahead of the first frame.

        Loading costs seconds, and paying it inside the first detection
        makes that detection look far slower than every one after it. The
        load runs on its own thread rather than through the work queue, so
        the engine never reports itself busy and the GUI does not drop a
        frame submitted straight away.

        Calling this again while a load is in flight, or after one has
        finished, does not start a second one: _get_reader caches per
        language combination.

        Returns:
            The thread doing the loading, already started.
        """
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return self._preload_thread

        self._preload_thread = threading.Thread(
            target=self._preload_model, daemon=True, name="ocr-preload"
        )
        self._preload_thread.start()
        return self._preload_thread

    def _preload_model(self) -> None:
        try:
            self._get_reader(list(self._settings.languages))
        except Exception as e:
            # A failed preload must not take the app down; the first real
            # detection reports the failure through its own result.
            logger.warning("OCR model preload failed: %s", e)

    def _worker_loop(self) -> None:
        """Process OCR requests one at a time from the work queue."""
        while self._running:
            try:
                item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _STOP:  # shutdown requested
                self._work_queue.task_done()
                break

            frame, languages, threshold, callback = item
            self._is_processing = True
            try:
                result = self.detect_text(frame, languages, threshold)
            except Exception as e:
                logger.error("OCR worker error: %s", e)
                result = DetectionResult(
                    detections=[],
                    languages=languages or list(self._settings.languages),
                    success=False,
                    error=str(e),
                )
            finally:
                self._is_processing = False

            if callback:
                callback(result)
            self._publish_result(result)
            self._work_queue.task_done()

    def _publish_result(self, result: DetectionResult) -> None:
        """Queue a result for poll_result(), discarding the oldest if full.

        A caller that only uses the callback never drains this queue, so it
        must stay bounded rather than grow for the life of the process.
        """
        while True:
            try:
                self._result_queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:  # pragma: no cover - drained concurrently
                    continue

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest cached model if cache exceeds max size."""
        while len(self._cache) > _MAX_CACHE_SIZE:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.info("Evicted cached model: %s", oldest_key)

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
                self._evict_cache_if_needed()
                logger.info("Loading OCR model for languages: %s", languages)
                self._cache[key] = easyocr.Reader(
                    list(key),
                    gpu=self._settings.gpu_enabled,
                )
                logger.info("OCR model loaded successfully")

        return self._cache[key]

    def detect_text(
        self,
        frame: np.ndarray,
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
        langs = languages or list(self._settings.languages)
        conf_threshold = threshold if threshold is not None else self._settings.default_confidence

        try:
            reader = self._get_reader(langs)

            recognise_frame, recognise_scale = resize_frame_for_ocr(
                frame, self._settings.ocr_max_width
            )
            if self._settings.preprocess_enabled:
                recognise_frame = preprocess_for_ocr(recognise_frame)

            # Locating the text costs around 95% of a pass and scales with
            # pixel count, while reading it is cheap. Detection therefore
            # runs on a smaller copy, and characters are still read at the
            # full requested detail. Deriving that copy from the frame above
            # means any denoising is paid once and serves both stages.
            detect_frame, detect_scale = resize_frame_for_ocr(
                recognise_frame, self._settings.detect_max_width
            )
            horizontal_list, free_list = reader.detect(detect_frame)
            horizontal, free = horizontal_list[0], free_list[0]

            if not len(horizontal) and not len(free):
                return DetectionResult(detections=[], languages=langs)

            height, width = recognise_frame.shape[:2]
            factor = 1.0 / detect_scale
            raw_results = reader.recognize(
                recognise_frame,
                horizontal_list=scale_horizontal_boxes(horizontal, factor, width, height),
                free_list=scale_free_boxes(free, factor, width, height),
                paragraph=self._settings.paragraph_merge,
            )
            detections = _normalise_results(raw_results)
            detections = filter_text(detections, conf_threshold)
            detections = scale_detections(detections, 1.0 / recognise_scale)
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
        frame: np.ndarray,
        languages: list[str] | None = None,
        threshold: float | None = None,
        callback: Callable[[DetectionResult], None] | None = None,
    ) -> bool:
        """Submit a frame for async OCR processing.

        Only one frame is processed at a time. If the worker is busy,
        the new frame is dropped (non-blocking).

        Args:
            frame: Image array to process.
            languages: Language codes for OCR.
            threshold: Confidence threshold.
            callback: Function called with DetectionResult when complete.

        Returns:
            True if the frame was queued, False if dropped (worker busy).
        """
        try:
            self._work_queue.put_nowait((frame, languages, threshold, callback))
            return True
        except queue.Full:
            return False

    @property
    def is_busy(self) -> bool:
        """Return True if the OCR worker is currently processing."""
        return self._is_processing or not self._work_queue.empty()

    def poll_result(self) -> DetectionResult | None:
        """Return a finished result without blocking, or None if none is ready.

        This is what the Tk main loop uses: the worker thread must never
        touch Tk itself, so results are collected here instead of being
        pushed from the worker.
        """
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

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

    def shutdown(self) -> None:
        """Shut down the worker thread and clear the cache."""
        self._running = False
        # Wake the worker now instead of letting it sit out its poll
        # timeout; closing the app should not stall for a second.
        with contextlib.suppress(queue.Full):
            self._work_queue.put_nowait(_STOP)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self.clear_cache()
        logger.info("OCR engine shut down")

    @property
    def settings(self) -> AppSettings:
        """The settings object this engine reads on every detection.

        Callers mutate it in place to reconfigure the engine; replacing the
        engine would strand its worker thread and cached models.
        """
        return self._settings

    @property
    def cache_size(self) -> int:
        """Return number of cached models."""
        return len(self._cache)
