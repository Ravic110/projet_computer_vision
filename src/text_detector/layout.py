"""Rules that adapt the interface to the screen it runs on.

Kept as pure functions so the breakpoints and the sizing arithmetic can be
tested without opening a window, the same way the colour and OCR logic is
isolated from the GUI.
"""

# Below MEDIUM_MIN_WIDTH the sidebar is hidden behind a toggle; below
# LARGE_MIN_WIDTH the text panel moves under the image instead of beside it.
COMPACT = "compact"
MEDIUM = "medium"
LARGE = "large"

MEDIUM_MIN_WIDTH = 750
LARGE_MIN_WIDTH = 1150

MIN_WINDOW = (600, 450)
PREFERRED_WINDOW = (1280, 800)

# Fraction of the screen the window claims before the bounds are applied.
SCREEN_FRACTION = 0.9

# Tk's "tk scaling" value on a 96 dpi display; anything above this is a
# denser screen and pixel dimensions have to grow with it.
BASE_SCALING = 4 / 3

# A monitor is assumed no wider than 16:9 for its height. Used to keep the
# window on one screen of a multi-monitor desktop.
MONITOR_ASPECT = 16 / 9


def breakpoint_for(width: int) -> str:
    """Pick the layout that suits a window width.

    Args:
        width: Window width in pixels.

    Returns:
        One of COMPACT, MEDIUM or LARGE.
    """
    if width >= LARGE_MIN_WIDTH:
        return LARGE
    if width >= MEDIUM_MIN_WIDTH:
        return MEDIUM
    return COMPACT


def initial_window_size(screen_width: int, screen_height: int) -> tuple[int, int]:
    """Choose the size the window opens at.

    Tk reports the bounding box of every attached display, so a laptop with
    a second monitor reads as one very wide screen. The usable width is
    therefore capped at what a 16:9 monitor of this height would be, which
    keeps the window from opening across a bezel.

    A screen smaller than MIN_WINDOW gets MIN_WINDOW anyway: a window too
    small to hold the layout is worse than one slightly larger than the
    screen.

    Args:
        screen_width: Screen width reported by Tk, in pixels.
        screen_height: Screen height reported by Tk, in pixels.

    Returns:
        Tuple of (width, height) in pixels.
    """
    usable_width = min(screen_width, max(round(screen_height * MONITOR_ASPECT), MIN_WINDOW[0]))
    width = max(MIN_WINDOW[0], min(PREFERRED_WINDOW[0], round(usable_width * SCREEN_FRACTION)))
    height = max(MIN_WINDOW[1], min(PREFERRED_WINDOW[1], round(screen_height * SCREEN_FRACTION)))
    return width, height


def scale_px(pixels: int, tk_scaling: float) -> int:
    """Scale a pixel dimension to the display's density.

    Fonts are declared in points and Tk already scales them; dimensions
    written in pixels do not follow, so widths and sizes pass through here.

    Args:
        pixels: Dimension authored for a 96 dpi display.
        tk_scaling: Value of Tk's "tk scaling", pixels per point.

    Returns:
        The dimension for this display, at least 1.
    """
    if tk_scaling <= 0:
        return pixels
    return max(1, round(pixels * tk_scaling / BASE_SCALING))
