"""Tests for utility modules."""

from pathlib import Path
from unittest.mock import patch

from text_detector.utils.logging_setup import get_logger, setup_logging
from text_detector.utils.path_helpers import (
    get_assets_dir,
    get_project_root,
)


def test_get_project_root_returns_path() -> None:
    result = get_project_root()
    assert isinstance(result, Path)
    assert (result / "src").exists()


def test_get_assets_dir_returns_path() -> None:
    result = get_assets_dir()
    assert isinstance(result, Path)
    assert result.name == "assets"


def test_assets_live_inside_the_package() -> None:
    """The icon must ship with the package, not sit beside the repository."""
    import text_detector

    assets = get_assets_dir()
    assert assets.parent == Path(text_detector.__file__).parent
    assert (assets / "icon.ico").exists()


def test_setup_logging_fallback(caplog) -> None:
    with patch("text_detector.utils.logging_setup.Path.exists", return_value=False):
        setup_logging()
    logger = get_logger("test_module")
    assert logger.name == "text_detector.test_module"


def test_get_logger_prefix() -> None:
    logger = get_logger("ocr_engine")
    assert logger.name == "text_detector.ocr_engine"


def test_get_log_path_honours_xdg_state_home(tmp_path, monkeypatch) -> None:
    from text_detector.utils.path_helpers import get_log_path

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert get_log_path() == tmp_path / "text-detector" / "text_detector.log"


def test_setup_logging_does_not_write_into_the_working_directory(tmp_path, monkeypatch) -> None:
    """The log belongs in a fixed place, not wherever the app was launched."""
    state = tmp_path / "state"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.chdir(cwd)

    setup_logging()
    get_logger("probe").info("hello")

    assert not (cwd / "text_detector.log").exists()
    assert (state / "text-detector" / "text_detector.log").exists()
