.PHONY: help install install-dev test test-cov lint format clean build upload docs serve-docs

# Default target
help:
	@echo "GrainStat Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup:"
	@echo "  install      Install package in current environment"
	@echo "  install-dev  Install package with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  test-fast    Run tests excluding slow tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Build & Release:"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI (requires credentials)"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo ""
	@echo "Examples:"
	@echo "  example      Run basic example"
	@echo "  demo         Run demonstration with sample data"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"
	pre-commit install

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=grainstat --cov-report=html --cov-report=term-missing -v

test-fast:
	pytest tests/ -v -m "not slow"

# Code quality targets
lint:
	flake8 grainstat/ tests/
	black --check grainstat/ tests/
	isort --check-only grainstat/ tests/

format:
	black grainstat/ tests/
	isort grainstat/ tests/

type-check:
	mypy grainstat/

# Build and release targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

upload-test: build
	python -m twine upload --repository testpypi dist/*

# Documentation targets
docs:
	@echo "Building documentation..."
	@echo "Note: This requires sphinx to be installed"
	@echo "Run: pip install sphinx sphinx-rtd-theme"
	# cd docs && make html

serve-docs:
	@echo "Serving documentation..."
	@echo "Note: This requires sphinx to be installed"
	# cd docs && make livehtml

# Example targets
example:
	@echo "Running basic example..."
	python examples/basic_usage.py

demo:
	@echo "Running demonstration..."
	@echo "Note: This requires sample images to be available"
	# python examples/demo.py

# Development helpers
setup-dev: install-dev
	@echo "Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make test' to verify installation"
	@echo "2. Run 'make example' to try basic functionality"
	@echo "3. Run 'make lint' before committing changes"

check: lint type-check test
	@echo "All checks passed!"

# CI targets
ci-test:
	pytest tests/ --cov=grainstat --cov-report=xml -v

ci-lint:
	flake8 grainstat/ tests/ --output-file=flake8.txt
	black --check grainstat/ tests/
	isort --check-only grainstat/ tests/

# Package info
info:
	@echo "Package Information"
	@echo "==================="
	@python -c "import grainstat; print(f'Version: {grainstat.__version__}')"
	@python -c "import grainstat; print(f'Author: {grainstat.__author__}')"
	@echo ""
	@echo "Dependencies:"
	@pip list | grep -E "(numpy|scipy|scikit-image|matplotlib|pandas|Pillow|seaborn)"

# Create sample data for testing
create-sample:
	@echo "Creating sample data..."
	python -c "
import numpy as np
from PIL import Image
import os

# Create sample directory
os.makedirs('sample_data', exist_ok=True)

# Create synthetic microstructure images
for i in range(3):
    # Generate synthetic grain structure
    image = np.zeros((200, 200))

    # Add random circular grains
    np.random.seed(i * 42)
    for _ in range(20):
        x = np.random.randint(20, 180)
        y = np.random.randint(20, 180)
        r = np.random.randint(5, 15)

        yy, xx = np.ogrid[:200, :200]
        mask = (xx - x)**2 + (yy - y)**2 <= r**2
        image[mask] = 1.0

    # Add noise
    image += np.random.normal(0, 0.05, image.shape)
    image = np.clip(image, 0, 1)

    # Save as TIFF
    img_pil = Image.fromarray((image * 255).astype(np.uint8))
    img_pil.save(f'sample_data/sample_{i+1}.tif')

print('Created 3 sample microstructure images in sample_data/')
"

# Version management
bump-patch:
	@echo "Bumping patch version..."
	# This would typically use bump2version or similar
	@echo "Note: Manual version update required in __init__.py and pyproject.toml"

bump-minor:
	@echo "Bumping minor version..."
	@echo "Note: Manual version update required in __init__.py and pyproject.toml"

bump-major:
	@echo "Bumping major version..."
	@echo "Note: Manual version update required in __init__.py and pyproject.toml"

# Performance profiling
profile:
	@echo "Running performance profiling..."
	python -m cProfile -o profile.stats examples/basic_usage.py
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling (requires memory_profiler)
memory-profile:
	@echo "Running memory profiling..."
	@echo "Note: Install memory_profiler with: pip install memory_profiler"
	# mprof run examples/basic_usage.py
	# mprof plot

# Security check
security:
	@echo "Running security checks..."
	@echo "Note: Install safety with: pip install safety"
	# safety check

# All quality checks
quality: lint type-check test
	@echo "All quality checks completed!"