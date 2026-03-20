PY := python
RUFF := ruff
PYTEST := pytest

.PHONY: help fmt lint lint-fix test check

help:
	@echo "Targets:"
	@echo "  fmt       - Ruff format"
	@echo "  lint      - Ruff lint (no fixes)"
	@echo "  lint-fix  - Ruff lint with auto-fix"
	@echo "  test      - Run pytest"
	@echo "  check     - Format + lint + tests"

fmt:
	$(RUFF) format .

lint:
	$(RUFF) check .

lint-fix:
	$(RUFF) check . --fix

test:
	$(PYTEST) -v

check: fmt lint test

run:
	uv run run