.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=disttune --cov-report=term-missing

lint:
	ruff check src/ scripts/ tests/
	mypy src/disttune/

format:
	ruff format src/ scripts/ tests/

clean:
	rm -rf outputs/ mlruns/ wandb/ __pycache__ .pytest_cache .mypy_cache
	find . -name "*.pyc" -delete
