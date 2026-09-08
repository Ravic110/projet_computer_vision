"""Tests for image processing module."""

import numpy as np

from text_detector.image_processor import (
    Detection,
    bgr_to_rgb,
    compute_display_geometry,
    draw_boxes_with_colors,
    filter_text,
    format_confidence,
    offset_detections,
    preprocess_for_ocr,
    resize_frame_for_display,
    resize_frame_for_ocr,
    scale_detections,
    scale_free_boxes,
    scale_horizontal_boxes,
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


class TestResizeForDisplay:
    def test_frame_is_resized_to_the_display_geometry(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        geometry = compute_display_geometry(1280, 720, 900, 600)
        resized = resize_frame_for_display(frame, geometry)
        assert resized.shape[:2] == (geometry.height, geometry.width)

    def test_a_frame_that_already_fits_is_returned_untouched(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        geometry = compute_display_geometry(100, 100, 900, 600)
        assert resize_frame_for_display(frame, geometry) is frame

    def test_smooth_and_fast_paths_agree_on_the_output_size(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        geometry = compute_display_geometry(640, 480, 320, 240)
        fast = resize_frame_for_display(frame, geometry, smooth=False)
        smooth = resize_frame_for_display(frame, geometry, smooth=True)
        assert fast.shape == smooth.shape == (geometry.height, geometry.width, 3)


class TestScaleDetectionBoxes:
    """Boxes found on a small frame have to move onto a larger one.

    EasyOCR reports horizontal boxes as [x_min, x_max, y_min, y_max] and
    rotated ones as four-point polygons, so the two need separate handling.
    """

    def test_horizontal_boxes_scale_and_stay_ints(self) -> None:
        boxes = scale_horizontal_boxes([[10, 50, 20, 40]], 2.0, 1000, 1000)
        assert boxes == [[20, 100, 40, 80]]
        assert all(isinstance(value, int) for value in boxes[0])

    def test_horizontal_boxes_are_clipped_to_the_frame(self) -> None:
        boxes = scale_horizontal_boxes([[-5, 900, -10, 700]], 2.0, 800, 600)
        assert boxes == [[0, 800, 0, 600]]

    def test_free_polygons_scale_point_by_point(self) -> None:
        polygons = scale_free_boxes([[[10, 20], [30, 20], [30, 40], [10, 40]]], 2.0, 1000, 1000)
        assert polygons == [[[20, 40], [60, 40], [60, 80], [20, 80]]]

    def test_free_polygons_are_clipped_to_the_frame(self) -> None:
        polygons = scale_free_boxes([[[-4, -4], [900, 700]]], 2.0, 800, 600)
        assert polygons == [[[0, 0], [800, 600]]]

    def test_a_factor_of_one_leaves_boxes_alone(self) -> None:
        assert scale_horizontal_boxes([[1, 2, 3, 4]], 1.0, 100, 100) == [[1, 2, 3, 4]]
        assert scale_free_boxes([[[1, 2]]], 1.0, 100, 100) == [[[1, 2]]]

    def test_empty_input(self) -> None:
        assert scale_horizontal_boxes([], 2.0, 10, 10) == []
        assert scale_free_boxes([], 2.0, 10, 10) == []

    def test_numpy_integers_are_accepted(self) -> None:
        # EasyOCR returns np.int32 coordinates.
        boxes = scale_horizontal_boxes(
            [[np.int32(10), np.int32(50), np.int32(20), np.int32(40)]], 2.0, 1000, 1000
        )
        assert boxes == [[20, 100, 40, 80]]
        assert all(isinstance(value, int) for value in boxes[0])


def test_preprocess_keeps_edges_sharp_while_cutting_noise() -> None:
    """The property that makes a denoiser usable for text.

    Text is edges. A filter that removes noise by softening them costs
    more accuracy than the noise did, so the contract is: less noise in
    flat areas, undiminished contrast across a boundary.
    """
    rng = np.random.default_rng(0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, 50:] = 255
    noisy = np.clip(frame.astype(np.int16) + rng.normal(0, 25, frame.shape), 0, 255).astype(
        np.uint8
    )

    result = preprocess_for_ocr(noisy)

    assert result[:, 10:40, 0].std() < noisy[:, 10:40, 0].std()
    assert int(result[50, 60, 0]) - int(result[50, 40, 0]) > 200
