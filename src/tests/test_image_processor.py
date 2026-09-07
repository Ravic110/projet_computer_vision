"""Tests for image processing module."""

import numpy as np

from text_detector.image_processor import (
    Detection,
    bgr_to_rgb,
    compute_avg_color,
    compute_display_geometry,
    draw_boxes_with_colors,
    filter_text,
    format_confidence,
    offset_detections,
    preprocess_for_ocr,
    resize_frame_for_ocr,
    scale_detections,
    widget_rect_to_frame_roi,
    widget_to_frame_point,
)


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
    assert result[0, 0, 2] == 0  # RGB: Blue channel


def test_resize_frame_for_ocr_no_resize_needed() -> None:
    frame = np.zeros((100, 400, 3), dtype=np.uint8)
    result, scale = resize_frame_for_ocr(frame, max_width=800)
    assert scale == 1.0
    assert result.shape == frame.shape


def test_resize_frame_for_ocr_resizes_large_frame() -> None:
    frame = np.zeros((600, 1600, 3), dtype=np.uint8)
    result, scale = resize_frame_for_ocr(frame, max_width=800)
    assert result.shape[1] == 800
    assert scale < 1.0


def test_scale_detections_no_scale() -> None:
    dets = [_make_detection("test", 0.9)]
    result = scale_detections(dets, scale=1.0)
    assert result == dets


def test_scale_detections_applies_factor() -> None:
    dets = [_make_detection("test", 0.9)]
    result = scale_detections(dets, scale=2.0)
    assert result[0][0][0][0] == 20.0  # 10.0 * 2.0
    assert result[0][1] == "test"


def test_preprocess_for_ocr_returns_3channel_bgr() -> None:
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_for_ocr(frame)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_preprocess_for_ocr_handles_grayscale() -> None:
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = preprocess_for_ocr(gray)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_preprocess_for_ocr_denoises_image() -> None:
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = preprocess_for_ocr(frame, strength=15)
    assert result is not None
    assert result.shape == frame.shape


def test_preprocess_for_ocr_default_strength() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = preprocess_for_ocr(frame)
    assert result.shape == (50, 50, 3)


# ── Display geometry / ROI mapping ──────────────────────────────────


def test_compute_display_geometry_fits_without_upscaling() -> None:
    geo = compute_display_geometry(100, 50, widget_width=400, widget_height=400)
    assert geo.scale == 1.0
    assert (geo.width, geo.height) == (100, 50)
    assert (geo.offset_x, geo.offset_y) == (150, 175)


def test_compute_display_geometry_downscales_large_frame() -> None:
    geo = compute_display_geometry(1000, 500, widget_width=500, widget_height=500)
    assert geo.scale == 0.5
    assert (geo.width, geo.height) == (500, 250)
    assert (geo.offset_x, geo.offset_y) == (0, 125)


def test_compute_display_geometry_limited_by_height() -> None:
    geo = compute_display_geometry(400, 800, widget_width=800, widget_height=400)
    assert geo.scale == 0.5
    assert (geo.width, geo.height) == (200, 400)
    assert geo.offset_x == 300


def test_compute_display_geometry_unrealized_widget() -> None:
    geo = compute_display_geometry(640, 480, widget_width=1, widget_height=1)
    assert geo.scale == 1.0
    assert (geo.offset_x, geo.offset_y) == (0, 0)
    assert (geo.width, geo.height) == (640, 480)


def test_widget_to_frame_point_removes_offset() -> None:
    geo = compute_display_geometry(100, 50, widget_width=400, widget_height=400)
    assert widget_to_frame_point(150, 175, geo, 100, 50) == (0, 0)
    assert widget_to_frame_point(200, 195, geo, 100, 50) == (50, 20)


def test_widget_to_frame_point_applies_scale() -> None:
    geo = compute_display_geometry(1000, 500, widget_width=500, widget_height=500)
    assert widget_to_frame_point(100, 125, geo, 1000, 500) == (200, 0)


def test_widget_to_frame_point_clamps_to_frame() -> None:
    geo = compute_display_geometry(100, 50, widget_width=400, widget_height=400)
    assert widget_to_frame_point(0, 0, geo, 100, 50) == (0, 0)
    assert widget_to_frame_point(399, 399, geo, 100, 50) == (100, 50)


def test_widget_rect_to_frame_roi_maps_selection() -> None:
    geo = compute_display_geometry(100, 100, widget_width=300, widget_height=300)
    roi = widget_rect_to_frame_roi((120, 120), (180, 190), geo, 100, 100)
    assert roi == (20, 20, 80, 90)


def test_widget_rect_to_frame_roi_normalizes_reverse_drag() -> None:
    geo = compute_display_geometry(100, 100, widget_width=300, widget_height=300)
    roi = widget_rect_to_frame_roi((180, 190), (120, 120), geo, 100, 100)
    assert roi == (20, 20, 80, 90)


def test_widget_rect_to_frame_roi_clamps_out_of_bounds_drag() -> None:
    geo = compute_display_geometry(100, 100, widget_width=300, widget_height=300)
    roi = widget_rect_to_frame_roi((0, 0), (500, 500), geo, 100, 100)
    assert roi == (0, 0, 100, 100)


def test_widget_rect_to_frame_roi_rejects_tiny_selection() -> None:
    geo = compute_display_geometry(100, 100, widget_width=300, widget_height=300)
    assert widget_rect_to_frame_roi((120, 120), (124, 124), geo, 100, 100) is None


def test_widget_rect_to_frame_roi_rejects_selection_on_padding() -> None:
    geo = compute_display_geometry(100, 100, widget_width=300, widget_height=300)
    assert widget_rect_to_frame_roi((0, 0), (60, 60), geo, 100, 100) is None


def test_offset_detections_shifts_boxes() -> None:
    dets = [_make_detection("test", 0.9)]
    result = offset_detections(dets, 5, 7)
    assert result[0][0][0] == [15.0, 17.0]
    assert result[0][0][2] == [105.0, 37.0]
    assert result[0][1] == "test"
    assert result[0][2] == 0.9


def test_offset_detections_zero_offset_is_identity() -> None:
    dets = [_make_detection("test", 0.9)]
    assert offset_detections(dets, 0, 0) == dets


def test_offset_detections_does_not_mutate_input() -> None:
    dets = [_make_detection("test", 0.9)]
    offset_detections(dets, 10, 10)
    assert dets[0][0][0] == [10.0, 10.0]


# ── Detections without a confidence (paragraph mode) ────────────────


def _paragraph_detection() -> list[Detection]:
    """A paragraph-mode detection: a box and text, but no confidence."""
    return [(_make_detection("para", 0.9)[0], "para", None)]


def test_filter_text_keeps_detections_without_confidence() -> None:
    detections = _paragraph_detection()
    result = filter_text(detections, threshold=0.99)
    assert len(result) == 1


def test_draw_boxes_labels_detections_without_confidence() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = _paragraph_detection()
    result = draw_boxes_with_colors(frame, detections)
    assert result.shape == frame.shape


def test_scale_detections_preserves_missing_confidence() -> None:
    detections = _paragraph_detection()
    result = scale_detections(detections, scale=2.0)
    assert result[0][2] is None
    assert result[0][0][0][0] == 20.0


def test_offset_detections_preserves_missing_confidence() -> None:
    detections = _paragraph_detection()
    result = offset_detections(detections, 5, 5)
    assert result[0][2] is None


def test_format_confidence_renders_a_value() -> None:
    assert format_confidence(0.876) == "0.88"


def test_format_confidence_renders_absence() -> None:
    assert format_confidence(None) == "—"
