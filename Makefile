.PHONY: install dev test lint format typecheck run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest src/tests -v --cov=text_detector --cov-report=term-missing

lint:
	python -m ruff check src/

format:
	python -m ruff format src/

run:
	python -m text_detector

typecheck:
	python -m mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
