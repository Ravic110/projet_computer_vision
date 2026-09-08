"""Colour sampling and CSS colour naming.

Kept free of Tkinter and of any I/O so the picker logic can be tested
without opening a window, the same way capture and OCR are isolated from
the GUI.
"""

import colorsys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import ImageColor

DEFAULT_WINDOW = 5

RGB = tuple[int, int, int]


def _build_css_table() -> dict[str, RGB]:
    """Read the CSS/X11 colour names Pillow already ships.

    Sorted by name so that aliases sharing one RGB value (aqua/cyan,
    fuchsia/magenta, the grey/gray pairs) are always resolved to the same
    name: the nearest-neighbour search below keeps the first match.

    Returns:
        Mapping of colour name to (r, g, b).
    """
    table: dict[str, RGB] = {}
    for name, value in sorted(ImageColor.colormap.items()):
        # Every entry is a "#rrggbb" string, but getrgb also knows the
        # RGBA forms, so the alpha channel is dropped for the type checker.
        red, green, blue = ImageColor.getrgb(value)[:3]
        table[name] = (red, green, blue)
    return table


CSS_COLORS: dict[str, RGB] = _build_css_table()

_CSS_NAMES: tuple[str, ...] = tuple(CSS_COLORS)
_CSS_LAB: np.ndarray = cv2.cvtColor(
    np.array([list(CSS_COLORS.values())], dtype=np.uint8), cv2.COLOR_RGB2LAB
)[0].astype(np.float64)


@dataclass(frozen=True)
class ColorSample:
    """One colour reading, in every form the interface shows.

    Attributes:
        rgb: Sampled colour as (r, g, b), 0-255.
        name: Nearest CSS colour name.
        hex: Lowercase "#rrggbb" string.
        hsv: (hue 0-360, saturation 0-100, value 0-100).
    """

    rgb: RGB
    name: str
    hex: str
    hsv: tuple[int, int, int]


def nearest_css_name(rgb: RGB) -> str:
    """Find the CSS colour name closest to an RGB value.

    The search runs in CIELAB rather than RGB: RGB distance is
    perceptually misleading, so a muted colour easily lands on a name no
    one would have chosen for it. The 148-entry table is converted once at
    import, leaving one 148-row subtraction per call.

    Args:
        rgb: Colour as (r, g, b), 0-255.

    Returns:
        The matching CSS colour name.
    """
    lab = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0][0].astype(
        np.float64
    )
    distances = np.sum((_CSS_LAB - lab) ** 2, axis=1)
    return _CSS_NAMES[int(np.argmin(distances))]


def rgb_to_hex(rgb: RGB) -> str:
    """Render an RGB value as a lowercase "#rrggbb" string.

    Args:
        rgb: Colour as (r, g, b), 0-255.

    Returns:
        The hex string.
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def rgb_to_hsv_degrees(rgb: RGB) -> tuple[int, int, int]:
    """Convert RGB to HSV in the units people read.

    OpenCV reports hue on 0-179 to fit a byte; this returns the
    conventional 0-360 degrees with saturation and value as percentages.

    Args:
        rgb: Colour as (r, g, b), 0-255.

    Returns:
        Tuple of (hue 0-360, saturation 0-100, value 0-100).
    """
    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
    return (round(hue * 360), round(saturation * 100), round(value * 100))


def sample_color(
    frame: np.ndarray | None,
    x: int,
    y: int,
    window: int = DEFAULT_WINDOW,
) -> ColorSample | None:
    """Read the colour around a point of a frame.

    A single pixel of a webcam frame is noisy and compression-damaged, so
    the reading is the mean of a small square window, cropped rather than
    padded at the borders. Coordinates outside the frame are clamped, so a
    click on the padding around the displayed image reads its nearest edge
    instead of failing.

    Args:
        frame: OpenCV image array (BGR, or 2-D grayscale), or None.
        x: Column in frame coordinates.
        y: Row in frame coordinates.
        window: Side of the averaging square, in pixels.

    Returns:
        The reading, or None when there is no frame to read.
    """
    if frame is None or frame.size == 0:
        return None

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    height, width = frame.shape[:2]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    half = max(1, window) // 2
    patch = frame[
        max(0, y - half) : min(height, y + half + 1),
        max(0, x - half) : min(width, x + half + 1),
    ]

    blue, green, red = (int(channel) for channel in np.rint(patch.mean(axis=(0, 1))))
    rgb = (red, green, blue)
    return ColorSample(
        rgb=rgb,
        name=nearest_css_name(rgb),
        hex=rgb_to_hex(rgb),
        hsv=rgb_to_hsv_degrees(rgb),
    )
