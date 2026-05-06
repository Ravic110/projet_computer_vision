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
