"""Image processing functions for text detection visualization."""

import cv2
import numpy as np

Detection = tuple[list[list[float]], str, float]


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

    Args:
        detections: List of (bbox, text, confidence) tuples from EasyOCR.
        threshold: Minimum confidence value (0.0-1.0).

    Returns:
        Filtered list containing only detections above threshold.
    """
    return [item for item in detections if item[2] >= threshold]


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
            f"{text} ({confidence:.2f})",
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
