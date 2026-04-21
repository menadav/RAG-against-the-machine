PYTHON = uv run python
SRC_DIR = src
GOINFRE_PATH := /goinfre/$(USER)/rag_project

export XDG_CACHE_HOME := $(GOINFRE_PATH)/.cache
export UV_CACHE_DIR := $(GOINFRE_PATH)/.cache/uv
export UV_PROJECT_ENVIRONMENT := $(GOINFRE_PATH)/.venv
export HF_HOME := $(GOINFRE_PATH)/.cache/huggingface

install:
	@mkdir -p $(UV_CACHE_DIR)
	@mkdir -p $(GOINFRE_PATH)
	uv sync --all-groups
	@ln -sfn $(GOINFRE_PATH)/.venv .venv

run:
	$(PYTHON) -m src index --max_chunk_size 2000

debug:
	$(PYTHON) -m pdb -m src index

clean:
	@rm -rf .venv
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf $(GOINFRE_PATH)
	@find . -type d -name "__pycache__" -exec rm -rf {} + [cite: 7]

lint:
	@uv run flake8 $(SRC_DIR) 
	@uv run mypy $(SRC_DIR) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs 

lint-strict:
	@echo "Running strict linting..."
	@uv run flake8 $(SRC_DIR) 
	@uv run mypy --strict $(SRC_DIR)


