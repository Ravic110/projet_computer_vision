"""Tests for the colour sampling module."""

import numpy as np

from text_detector.color_detector import (
    CSS_COLORS,
    ColorSample,
    nearest_css_name,
    rgb_to_hex,
    rgb_to_hsv_degrees,
    sample_color,
)


def _solid_frame(bgr: tuple[int, int, int], size: int = 10) -> np.ndarray:
    """Build a uniform BGR frame of the given colour."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


class TestCssTable:
    def test_table_holds_the_css_names(self) -> None:
        assert len(CSS_COLORS) >= 140
        assert CSS_COLORS["red"] == (255, 0, 0)
        assert CSS_COLORS["black"] == (0, 0, 0)

    def test_names_are_sorted(self) -> None:
        assert list(CSS_COLORS) == sorted(CSS_COLORS)


class TestNearestCssName:
    def test_exact_value_returns_its_own_name(self) -> None:
        assert nearest_css_name((255, 0, 0)) == "red"
        assert nearest_css_name((0, 0, 0)) == "black"
        assert nearest_css_name((255, 255, 255)) == "white"
        assert nearest_css_name((255, 165, 0)) == "orange"

    def test_alias_ties_resolve_alphabetically(self) -> None:
        # aqua/cyan and fuchsia/magenta share the same RGB value; the
        # first name in alphabetical order must always win.
        assert nearest_css_name((0, 255, 255)) == "aqua"
        assert nearest_css_name((255, 0, 255)) == "fuchsia"

    def test_near_miss_resolves_to_the_expected_neighbour(self) -> None:
        assert nearest_css_name((250, 5, 5)) == "red"
        assert nearest_css_name((3, 3, 3)) == "black"

    def test_every_table_entry_resolves_to_a_known_name(self) -> None:
        for rgb in CSS_COLORS.values():
            assert nearest_css_name(rgb) in CSS_COLORS


class TestRgbToHex:
    def test_formats_lowercase_six_digit_hex(self) -> None:
        assert rgb_to_hex((0, 0, 0)) == "#000000"
        assert rgb_to_hex((255, 255, 255)) == "#ffffff"
        assert rgb_to_hex((18, 52, 86)) == "#123456"


class TestRgbToHsvDegrees:
    def test_primaries(self) -> None:
        assert rgb_to_hsv_degrees((255, 0, 0)) == (0, 100, 100)
        assert rgb_to_hsv_degrees((0, 255, 0)) == (120, 100, 100)
        assert rgb_to_hsv_degrees((0, 0, 255)) == (240, 100, 100)

    def test_achromatic(self) -> None:
        assert rgb_to_hsv_degrees((0, 0, 0)) == (0, 0, 0)
        assert rgb_to_hsv_degrees((255, 255, 255)) == (0, 0, 100)


class TestSampleColor:
    def test_uniform_patch_returns_that_colour(self) -> None:
        sample = sample_color(_solid_frame((0, 0, 255)), 5, 5)
        assert isinstance(sample, ColorSample)
        assert sample.rgb == (255, 0, 0)
        assert sample.name == "red"
        assert sample.hex == "#ff0000"
        assert sample.hsv == (0, 100, 100)

    def test_patch_straddling_two_colours_returns_the_mean(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, :] = (200, 200, 200)
        sample = sample_color(frame, 0, 0, window=5)
        assert sample is not None
        assert sample.rgb == (100, 100, 100)

    def test_window_is_cropped_at_the_borders(self) -> None:
        frame = _solid_frame((0, 0, 0))
        frame[0, 0] = (255, 255, 255)
        # A 3x3 window at the corner covers 4 pixels, one of them white.
        sample = sample_color(frame, 0, 0, window=3)
        assert sample is not None
        assert sample.rgb == (64, 64, 64)

    def test_single_pixel_window(self) -> None:
        frame = _solid_frame((0, 0, 0))
        frame[7, 3] = (0, 255, 0)
        sample = sample_color(frame, 3, 7, window=1)
        assert sample is not None
        assert sample.rgb == (0, 255, 0)

    def test_coordinates_are_clamped_to_the_frame(self) -> None:
        frame = _solid_frame((255, 255, 255))
        for x, y in [(0, 0), (9, 9), (10, 10), (500, 500), (-5, -5)]:
            sample = sample_color(frame, x, y)
            assert sample is not None
            assert sample.rgb == (255, 255, 255)

    def test_grayscale_frame_is_supported(self) -> None:
        frame = np.full((10, 10), 128, dtype=np.uint8)
        sample = sample_color(frame, 5, 5)
        assert sample is not None
        assert sample.rgb == (128, 128, 128)

    def test_empty_frame_returns_none(self) -> None:
        assert sample_color(np.zeros((0, 0, 3), dtype=np.uint8), 0, 0) is None

    def test_missing_frame_returns_none(self) -> None:
        assert sample_color(None, 0, 0) is None
