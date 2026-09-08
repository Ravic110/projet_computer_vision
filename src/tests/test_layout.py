"""Tests for the responsive layout rules."""

import pytest

from text_detector.layout import (
    COMPACT,
    LARGE,
    LARGE_MIN_WIDTH,
    MEDIUM,
    MEDIUM_MIN_WIDTH,
    MIN_WINDOW,
    PREFERRED_WINDOW,
    breakpoint_for,
    initial_window_size,
    scale_px,
)


class TestBreakpointFor:
    def test_boundaries(self) -> None:
        assert breakpoint_for(MEDIUM_MIN_WIDTH - 1) == COMPACT
        assert breakpoint_for(MEDIUM_MIN_WIDTH) == MEDIUM
        assert breakpoint_for(LARGE_MIN_WIDTH - 1) == MEDIUM
        assert breakpoint_for(LARGE_MIN_WIDTH) == LARGE

    def test_extremes(self) -> None:
        assert breakpoint_for(0) == COMPACT
        assert breakpoint_for(3840) == LARGE


class TestInitialWindowSize:
    @pytest.mark.parametrize(
        "screen",
        [(1024, 600), (1280, 1024), (1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)],
    )
    def test_fits_on_the_screen(self, screen) -> None:
        width, height = initial_window_size(*screen)
        assert width <= screen[0]
        assert height <= screen[1]

    def test_multi_monitor_desktop_stays_on_one_monitor(self) -> None:
        # Tk reports the bounding box of every screen; a 1920x1080 laptop
        # beside a 1280x1024 display reads as 3200x1080.
        width, _height = initial_window_size(3200, 1080)
        assert width <= 1920

    def test_never_smaller_than_the_minimum(self) -> None:
        assert initial_window_size(320, 240) == MIN_WINDOW
        width, height = initial_window_size(800, 600)
        assert width >= MIN_WINDOW[0]
        assert height >= MIN_WINDOW[1]

    def test_never_larger_than_the_preferred_size(self) -> None:
        width, height = initial_window_size(3840, 2160)
        assert (width, height) == PREFERRED_WINDOW

    def test_leaves_room_for_the_task_bar(self) -> None:
        _width, height = initial_window_size(1366, 768)
        assert height < 768


class TestScalePx:
    def test_base_scaling_is_a_no_op(self) -> None:
        assert scale_px(240, 4 / 3) == 240

    def test_hidpi_scaling_grows_the_value(self) -> None:
        assert scale_px(240, 8 / 3) == 480
        assert scale_px(240, 2.0) == 360

    def test_low_scaling_shrinks_the_value(self) -> None:
        assert scale_px(240, 1.0) == 180

    def test_never_returns_zero(self) -> None:
        assert scale_px(1, 0.1) == 1

    def test_non_positive_scaling_is_ignored(self) -> None:
        assert scale_px(240, 0.0) == 240
        assert scale_px(240, -1.0) == 240
