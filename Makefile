# Makefile for UAL Adapter development

.PHONY: help install install-dev clean test lint format build docker-build docker-run docs

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := ual_adapter
DOCKER_IMAGE := ual-adapter
DOCKER_TAG := latest

# Default target
help:
	@echo "UAL Adapter Development Commands"
	@echo "================================"
	@echo "install      : Install package in development mode"
	@echo "install-dev  : Install with all development dependencies"
	@echo "test         : Run all tests"
	@echo "test-cov     : Run tests with coverage report"
	@echo "test-fast    : Run tests excluding slow tests"
	@echo "lint         : Run all linters"
	@echo "format       : Format code with black and isort"
	@echo "clean        : Remove build artifacts and cache"
	@echo "build        : Build distribution packages"
	@echo "docker-build : Build Docker image"
	@echo "docker-run   : Run Docker container"
	@echo "docs         : Build documentation"
	@echo "release      : Create a new release"

# Installation
install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev,docs]
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow"

test-integration:
	pytest tests/ -v -m integration

# Code quality
lint:
	black --check $(PROJECT_NAME) tests
	isort --check-only $(PROJECT_NAME) tests
	flake8 $(PROJECT_NAME) tests
	mypy $(PROJECT_NAME)

format:
	black $(PROJECT_NAME) tests examples
	isort $(PROJECT_NAME) tests examples

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Building
build: clean
	$(PYTHON) -m build

# Docker
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run -it --rm \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/adapters:/app/adapters \
		-v $(PWD)/outputs:/app/outputs \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		bash

docker-test:
	docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) pytest tests/

# Documentation
docs:
	cd docs && $(MAKE) clean && $(MAKE) html
	@echo "Documentation built in docs/_build/html/"

serve-docs:
	cd docs/_build/html && python -m http.server

# Release
release:
	@echo "Creating release..."
	@read -p "Version number (current: $$(grep version pyproject.toml | head -1 | cut -d'"' -f2)): " version; \
	if [ -n "$$version" ]; then \
		sed -i "s/version = \".*\"/version = \"$$version\"/" pyproject.toml; \
		sed -i "s/__version__ = \".*\"/__version__ = \"$$version\"/" $(PROJECT_NAME)/__init__.py; \
		git add pyproject.toml $(PROJECT_NAME)/__init__.py; \
		git commit -m "Bump version to $$version"; \
		git tag -a "v$$version" -m "Release version $$version"; \
		echo "Release $$version created. Push with: git push && git push --tags"; \
	fi

# Development shortcuts
dev: install-dev
	@echo "Development environment ready!"

check: lint test
	@echo "All checks passed!"

# Performance testing
benchmark:
	python examples/benchmark.py

profile:
	python -m cProfile -o profile.stats examples/complete_example.py
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# GPU testing
test-gpu:
	pytest tests/ -v -m gpu --gpu

# Continuous Integration simulation
ci: clean lint test build
	@echo "CI pipeline complete!"
