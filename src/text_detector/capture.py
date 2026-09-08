"""Background webcam capture.

Reading from a camera blocks for as long as the device takes to deliver a
frame. Doing that inside the Tk main loop freezes the whole interface, so
the read runs on its own thread and the GUI only ever picks up the frame
that is already waiting.
"""

import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from text_detector.utils.logging_setup import get_logger

logger = get_logger("capture")

# How long to wait before retrying after a failed read, so a disconnected
# camera cannot spin the thread at full speed.
_RETRY_DELAY_S = 0.05


class FrameGrabber:
    """Serves the most recent webcam frame to the GUI thread.

    Only the latest frame is kept: a backlog would make the preview lag
    further behind reality the longer the app runs.

    Args:
        source: Camera index or path passed to the opener.
        opener: Factory returning a cv2.VideoCapture-like object. Injected
            so tests can run without a camera.
    """

    def __init__(
        self,
        source: int | str = 0,
        opener: Callable[..., Any] = cv2.VideoCapture,
    ) -> None:
        self._source = source
        self._opener = opener
        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Open the camera and begin capturing.

        Opening happens here, on the caller's thread, so a camera that is
        missing or already in use is reported straight away instead of
        failing silently inside the worker.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        if self._running:
            return

        capture = self._opener(self._source)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise RuntimeError("Unable to open webcam.")

        self._capture = capture
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-grabber")
        self._thread.start()
        logger.info("Frame grabber started on source %s", self._source)

    def _loop(self) -> None:
        """Keep the latest frame up to date until stopped."""
        while self._running:
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok:
                time.sleep(_RETRY_DELAY_S)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        """Return the most recent frame, or None if none has arrived yet."""
        with self._lock:
            return self._frame

    def stop(self) -> None:
        """Stop capturing and release the camera. Safe to call twice."""
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
            logger.info("Frame grabber stopped")
        with self._lock:
            self._frame = None

    @property
    def is_running(self) -> bool:
        """Return True while the capture thread is active."""
        return self._running
