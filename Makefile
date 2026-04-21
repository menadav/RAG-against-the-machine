PYTHON := uv run python
SRC_DIR := src

install:
	uv sync --all-groups 

run:
	$(PYTHON) -m student index

debug:
	$(PYTHON) -m pdb -m student index

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + [cite: 7]

lint:
	uv run flake8 $(SRC_DIR) 
	uv run mypy $(SRC_DIR) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs 

lint-strict:
	@echo "Running strict linting..."
	uv run flake8 $(SRC_DIR) 
	uv run mypy --strict $(SRC_DIR)


