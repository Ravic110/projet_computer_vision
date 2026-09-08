"""Logging configuration setup."""

import logging
import logging.config
from pathlib import Path


def setup_logging(config_path: Path | None = None) -> None:
    """Configure logging from logging.conf file.

    Args:
        config_path: Path to logging configuration file.
            Defaults to logging.conf in project root.
    """
    from text_detector.utils.path_helpers import get_log_path, get_project_root

    if config_path is None:
        config_path = get_project_root() / "logging.conf"

    if config_path.exists():
        log_path = get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # The handler's filename is injected rather than hardcoded in the
        # .conf, which can only express a path relative to the CWD.
        logging.config.fileConfig(
            config_path,
            defaults={"logfilename": str(log_path)},
            disable_existing_loggers=False,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the text_detector prefix.

    Args:
        name: Module name (e.g., 'ocr_engine', 'gui').

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"text_detector.{name}")
