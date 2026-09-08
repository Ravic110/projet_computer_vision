"""Tests for the background frame grabber."""

import numpy as np
import pytest

from text_detector.capture import FrameGrabber


class FakeCapture:
    """Stand-in for cv2.VideoCapture that serves a fixed list of frames."""

    def __init__(self, frames: list, opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True
        self._opened = False


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    """Poll a predicate until it holds, so tests never sleep blindly."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_start_raises_when_the_camera_will_not_open() -> None:
    grabber = FrameGrabber(opener=lambda _source: FakeCapture([], opened=False))
    with pytest.raises(RuntimeError, match="webcam"):
        grabber.start()
    assert grabber.is_running is False


def test_read_returns_none_before_the_first_frame_arrives() -> None:
    grabber = FrameGrabber(opener=lambda _source: FakeCapture([]))
    grabber.start()
    try:
        assert grabber.read() is None
    finally:
        grabber.stop()


def test_read_returns_a_captured_frame() -> None:
    capture = FakeCapture([_frame(7)])
    grabber = FrameGrabber(opener=lambda _source: capture)
    grabber.start()
    try:
        assert _wait_for(lambda: grabber.read() is not None)
        frame = grabber.read()
        assert frame is not None
        assert frame[0][0][0] == 7
    finally:
        grabber.stop()


def test_read_keeps_only_the_most_recent_frame() -> None:
    capture = FakeCapture([_frame(1), _frame(2), _frame(3)])
    grabber = FrameGrabber(opener=lambda _source: capture)
    grabber.start()
    try:

        def latest_is_third() -> bool:
            frame = grabber.read()
            return frame is not None and bool(frame[0][0][0] == 3)

        assert _wait_for(latest_is_third)
    finally:
        grabber.stop()


def test_stop_releases_the_capture() -> None:
    capture = FakeCapture([_frame(1)])
    grabber = FrameGrabber(opener=lambda _source: capture)
    grabber.start()
    grabber.stop()
    assert capture.released is True
    assert grabber.is_running is False


def test_stop_is_safe_before_start() -> None:
    grabber = FrameGrabber(opener=lambda _source: FakeCapture([]))
    grabber.stop()
    assert grabber.is_running is False


def test_start_is_idempotent() -> None:
    opened = []

    def opener(source):
        opened.append(source)
        return FakeCapture([_frame(1)])

    grabber = FrameGrabber(opener=opener)
    grabber.start()
    try:
        grabber.start()
        assert len(opened) == 1
    finally:
        grabber.stop()
