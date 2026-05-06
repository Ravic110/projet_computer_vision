"""OCR engine wrapper for EasyOCR with model caching and threading support."""

import queue
import threading
from collections.abc import Callable
from typing import Any

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
