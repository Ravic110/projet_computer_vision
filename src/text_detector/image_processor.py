"""Image processing functions for text detection visualization."""

from dataclasses import dataclass

import cv2
import numpy as np

# Confidence is None for paragraph-mode detections: EasyOCR merges word
# boxes into blocks and does not report a confidence for the result.
Detection = tuple[list[list[float]], str, float | None]
NO_CONFIDENCE = "\u2014"


def format_confidence(confidence: float | None) -> str:
    """Render a confidence for display, or a dash when there is none.

    Args:
        confidence: Confidence value, or None for a paragraph-mode detection.

    Returns:
        Two-decimal string, or an em dash.
    """
    return NO_CONFIDENCE if confidence is None else f"{confidence:.2f}"


def resize_frame_for_ocr(
    frame: np.ndarray,
    max_width: int = 800,
) -> tuple[np.ndarray, float]:
    """Resize a frame for faster OCR processing.

    If the frame width exceeds max_width, it is scaled down proportionally.
    Small frames are left unchanged.

    Args:
        frame: OpenCV image array (BGR).
        max_width: Maximum width for OCR input.

    Returns:
        Tuple of (resized frame, scale factor).
        Scale factor is 1.0 if no resizing occurred.
    """
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0

    scale = max_width / width
    new_width = max_width
    new_height = int(height * scale)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_detections(
    detections: list[Detection],
    scale: float,
) -> list[Detection]:
    """Scale bounding box coordinates back to original frame size.

    Args:
        detections: List of (bbox, text, confidence) tuples from EasyOCR.
        scale: The inverse of the resize scale (1/resize_factor).

    Returns:
        List of detections with scaled bounding boxes.
    """
    if scale == 1.0:
        return detections

    return [
        (
            [[pt[0] * scale, pt[1] * scale] for pt in bbox],
            text,
            confidence,
        )
        for bbox, text, confidence in detections
    ]


def filter_text(
    detections: list[Detection],
    threshold: float,
) -> list[Detection]:
    """Filter detections by confidence threshold.

    Detections without a confidence (paragraph mode) are always kept:
    there is no score to compare against the threshold.

    Args:
        detections: List of (bbox, text, confidence) tuples from EasyOCR.
        threshold: Minimum confidence value (0.0-1.0).

    Returns:
        Filtered list containing only detections above threshold.
    """
    return [item for item in detections if item[2] is None or item[2] >= threshold]


def compute_avg_color(frame: np.ndarray, bbox: list[list[float]]) -> tuple[float, float, float]:
    """Compute average color within a bounding box region.

    Args:
        frame: OpenCV image array (BGR).
        bbox: Bounding box coordinates from EasyOCR.

    Returns:
        Average (B, G, R) color tuple, or (0, 0, 0) if region is invalid.
    """
    points = [(int(pt[0]), int(pt[1])) for pt in bbox]
    x1 = max(0, points[0][0])
    y1 = max(0, points[0][1])
    x2 = max(0, points[2][0])
    y2 = max(0, points[2][1])

    if y2 <= y1 or x2 <= x1:
        return (0.0, 0.0, 0.0)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return (0.0, 0.0, 0.0)

    return cv2.mean(roi)[:3]


def draw_boxes_with_colors(
    frame: np.ndarray,
    detections: list[Detection],
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw bounding boxes and text labels on a frame.

    Args:
        frame: OpenCV image array (BGR). Will be copied.
        detections: List of (bbox, text, confidence) tuples.
        box_color: BGR color for bounding box lines.
        text_color: BGR color for text labels.

    Returns:
        New frame with bounding boxes and labels drawn.
    """
    result = frame.copy()

    for bbox, text, confidence in detections:
        points = [(int(pt[0]), int(pt[1])) for pt in bbox]
        pts = np.array(points, dtype=np.int32)

        cv2.polylines(result, [pts], isClosed=True, color=box_color, thickness=2)
        cv2.putText(
            result,
            f"{text} ({format_confidence(confidence)})",
            (points[0][0], points[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA,
        )

    return result


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB.

    Args:
        frame: OpenCV image array (BGR).

    Returns:
        RGB image array.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preprocess_for_ocr(
    frame: np.ndarray,
    strength: int = 10,
) -> np.ndarray:
    """Preprocess image for improved OCR accuracy.

    Converts to grayscale, applies non-local means denoising, then
    converts back to 3-channel BGR for EasyOCR compatibility.

    Args:
        frame: OpenCV image array (BGR or grayscale).
        strength: Denoising intensity (higher = stronger). Default 10.

    Returns:
        3-channel BGR image array with reduced noise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


@dataclass(frozen=True)
class DisplayGeometry:
    """Placement of a frame inside the widget that displays it.

    A frame is scaled down to fit the widget (never up) and centered,
    so widget coordinates and frame coordinates differ by both an
    offset and a scale factor.

    Attributes:
        offset_x: Left edge of the displayed image, in widget pixels.
        offset_y: Top edge of the displayed image, in widget pixels.
        scale: Displayed size divided by frame size (<= 1.0).
        width: Displayed image width, in widget pixels.
        height: Displayed image height, in widget pixels.
    """

    offset_x: int
    offset_y: int
    scale: float
    width: int
    height: int


def compute_display_geometry(
    frame_width: int,
    frame_height: int,
    widget_width: int,
    widget_height: int,
) -> DisplayGeometry:
    """Compute how a frame is laid out inside its display widget.

    The frame is scaled down to fit the widget while preserving aspect
    ratio, then centered. Frames smaller than the widget are not scaled up.
    An unrealized widget (width or height <= 1) falls back to a 1:1,
    top-left placement so coordinates stay usable.

    Args:
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        widget_width: Widget width in pixels.
        widget_height: Widget height in pixels.

    Returns:
        The resulting DisplayGeometry.
    """
    if widget_width <= 1 or widget_height <= 1 or frame_width <= 0 or frame_height <= 0:
        return DisplayGeometry(0, 0, 1.0, frame_width, frame_height)

    scale = min(1.0, widget_width / frame_width, widget_height / frame_height)
    width = max(1, round(frame_width * scale))
    height = max(1, round(frame_height * scale))
    return DisplayGeometry(
        offset_x=(widget_width - width) // 2,
        offset_y=(widget_height - height) // 2,
        scale=scale,
        width=width,
        height=height,
    )


def widget_to_frame_point(
    x: int,
    y: int,
    geometry: DisplayGeometry,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """Convert a widget coordinate to a frame coordinate.

    The result is clamped to the frame bounds, so clicks on the padding
    around the image map to the nearest edge instead of an invalid index.

    Args:
        x: Widget x coordinate.
        y: Widget y coordinate.
        geometry: Layout of the frame inside the widget.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Tuple of (x, y) in frame coordinates, clamped to [0, size].
    """
    scale = geometry.scale if geometry.scale > 0 else 1.0
    frame_x = int((x - geometry.offset_x) / scale)
    frame_y = int((y - geometry.offset_y) / scale)
    return (
        max(0, min(frame_width, frame_x)),
        max(0, min(frame_height, frame_y)),
    )


def widget_rect_to_frame_roi(
    start: tuple[int, int],
    end: tuple[int, int],
    geometry: DisplayGeometry,
    frame_width: int,
    frame_height: int,
    min_size: int = 10,
) -> tuple[int, int, int, int] | None:
    """Convert a dragged widget rectangle into a frame-space ROI.

    Handles drags in any direction, clamps the result to the frame, and
    rejects selections that are too small to be worth running OCR on.

    Args:
        start: Drag start point, in widget coordinates.
        end: Drag end point, in widget coordinates.
        geometry: Layout of the frame inside the widget.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        min_size: Minimum accepted width and height, in frame pixels.

    Returns:
        Tuple of (x1, y1, x2, y2) in frame coordinates, or None if the
        selection is degenerate.
    """
    x1, y1 = widget_to_frame_point(start[0], start[1], geometry, frame_width, frame_height)
    x2, y2 = widget_to_frame_point(end[0], end[1], geometry, frame_width, frame_height)

    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)

    if right - left < min_size or bottom - top < min_size:
        return None

    return (left, top, right, bottom)


def offset_detections(
    detections: list[Detection],
    dx: int,
    dy: int,
) -> list[Detection]:
    """Translate detection boxes by a fixed offset.

    Used to move boxes found in a cropped ROI back into full-frame
    coordinates before drawing them.

    Args:
        detections: List of (bbox, text, confidence) tuples.
        dx: Horizontal offset in pixels.
        dy: Vertical offset in pixels.

    Returns:
        List of detections with translated bounding boxes.
    """
    if dx == 0 and dy == 0:
        return detections

    return [
        (
            [[pt[0] + dx, pt[1] + dy] for pt in bbox],
            text,
            confidence,
        )
        for bbox, text, confidence in detections
    ]
