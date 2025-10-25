#!/bin/bash
# scripts/check_code.sh

echo "Running code quality checks..."

# Format with black
echo "Formatting code with black..."
black src/ tests/ --line-length 100

# Lint with flake8
echo "Linting with flake8..."
flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

# Run tests
echo "Running tests..."
pytest tests/ -v

echo "✓ All checks passed!"
