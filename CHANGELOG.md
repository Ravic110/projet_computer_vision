# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-05-06

### Added
- Modular architecture (config, ocr_engine, image_processor, utils)
- Threading for OCR to prevent GUI freezing
- Multi-language support with model caching
- GPU toggle option in settings
- Export to CSV and JSON formats
- Detection history tracking
- Unit tests with pytest
- Logging configuration
- pyproject.toml for modern packaging
- Makefile for development commands
- Type hints throughout codebase
- Docstrings (Google style)

### Changed
- Replaced os.path with pathlib
- Unfrozen dependency versions in requirements.txt
- Centralized theme colors in config module
- Refactored GUI to use configuration constants

### Removed
- Monolithic single-file architecture
- BOM character from requirements.txt

## [1.0.0] - Previous version
- Initial single-file implementation
- Basic text detection with EasyOCR
- Tkinter GUI with webcam and image support
