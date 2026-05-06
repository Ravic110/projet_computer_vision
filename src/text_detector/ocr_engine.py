"""OCR engine wrapper for EasyOCR with model caching and threading support."""

import queue
import threading
from collections.abc import Callable
from typing import Any

from text_detector.config import AppSettings
from text_detector.image_processor import resize_frame_for_ocr, scale_detections
from text_detector.utils.logging_setup import get_logger

logger = get_logger("ocr_engine")

_MAX_CACHE_SIZE = 2


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

    Uses a single worker thread with a bounded queue to prevent
    thread explosion and memory exhaustion. Only processes one
    frame at a time; excess frames are dropped.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or AppSettings()
        self._cache: dict[tuple[str, ...], Any] = {}
        self._lock = threading.Lock()
        self._result_queue: queue.Queue[DetectionResult] = queue.Queue()
        self._work_queue: queue.Queue[tuple] = queue.Queue(maxsize=1)
        self._worker_thread: threading.Thread | None = None
        self._running = True
        self._is_processing = False
        self._start_worker()

    def _start_worker(self) -> None:
        """Start the single OCR worker thread."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="ocr-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Process OCR requests one at a time from the work queue."""
        while self._running:
            try:
                item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frame, languages, threshold, callback = item
            self._is_processing = True
            try:
                result = self.detect_text(frame, languages, threshold)
            except Exception as e:
                logger.error("OCR worker error: %s", e)
                result = DetectionResult(
                    detections=[],
                    languages=languages or [self._settings.default_language],
                    success=False,
                    error=str(e),
                )
            finally:
                self._is_processing = False

            if callback:
                callback(result)
            self._result_queue.put(result)
            self._work_queue.task_done()

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
            ocr_frame, scale = resize_frame_for_ocr(frame, self._settings.ocr_max_width)
            raw_results = reader.readtext(
                ocr_frame,
                paragraph=self._settings.paragraph_merge,
            )
            detections = [item for item in raw_results if item[2] >= conf_threshold]
            detections = scale_detections(detections, 1.0 / scale)
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
        """Shut down the worker thread and clear cache."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self.clear_cache()
        logger.info("OCR engine shut down")

    @property
    def cache_size(self) -> int:
        """Return number of cached models."""
        return len(self._cache)
