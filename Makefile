.PHONY: install format lint test run clean docker-build docker-up docker-down docs docs-serve bump-patch bump-minor bump-major

# Install dependencies (server + tooling) into a uv-managed virtualenv
install:
	uv sync

# Format code
format:
	uv run ruff format app cli tests

# Run linting
lint:
	uv run ruff check app cli tests
	uv run mypy app cli

# Run tests
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=app --cov-report=term-missing --cov-report=xml

# Run the application
run:
	uv run uvicorn app.main:app --reload

# Clean up compiled Python files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -f coverage.xml

# Docker commands
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# Documentation (built with Zensical)
docs:
	uv run zensical build

docs-serve:
	uv run zensical serve

# Version bumping
bump-patch:
	uv run bumpver update --patch

bump-minor:
	uv run bumpver update --minor

bump-major:
	uv run bumpver update --major
