"""Tests for utility modules."""

from pathlib import Path
import tempfile
from unittest.mock import patch

from text_detector.utils.path_helpers import get_assets_dir, ensure_dir, get_safe_path, get_project_root
from text_detector.utils.logging_setup import setup_logging, get_logger


def test_get_project_root_returns_path() -> None:
    result = get_project_root()
    assert isinstance(result, Path)
    assert (result / "src").exists()


def test_get_assets_dir_returns_path() -> None:
    result = get_assets_dir()
    assert isinstance(result, Path)
    assert result.name == "assets"


def test_ensure_dir_creates_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "new" / "nested" / "dir"
        assert not target.exists()
        ensure_dir(target)
        assert target.exists()
        assert target.is_dir()


def test_ensure_dir_existing_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "existing"
        target.mkdir()
        result = ensure_dir(target)
        assert result.exists()
        assert result == target


def test_get_safe_path_adds_extension() -> None:
    base = Path("/tmp/test")
    result = get_safe_path(base, "output", ".txt")
    assert result.name == "output.txt"


def test_get_safe_path_keeps_existing_extension() -> None:
    base = Path("/tmp/test")
    result = get_safe_path(base, "output.txt", ".txt")
    assert result.name == "output.txt"


def test_get_safe_path_mismatched_extension() -> None:
    base = Path("/tmp/test")
    result = get_safe_path(base, "output.txt", ".csv")
    assert result.name == "output.txt.csv"


def test_setup_logging_fallback(caplog) -> None:
    with patch("text_detector.utils.logging_setup.Path.exists", return_value=False):
        setup_logging()
    logger = get_logger("test_module")
    assert logger.name == "text_detector.test_module"


def test_get_logger_prefix() -> None:
    logger = get_logger("ocr_engine")
    assert logger.name == "text_detector.ocr_engine"
