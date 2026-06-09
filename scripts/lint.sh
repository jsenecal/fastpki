#!/bin/bash
set -e

# Run formatting and linting
echo "Running ruff format..."
ruff format app cli tests

echo "Running ruff check..."
ruff check app cli tests

echo "Running mypy..."
mypy app cli

echo "All linting checks passed!"
