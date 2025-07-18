Testing
=======

This page covers how to run tests, write new tests, and understand the testing framework used in GrainStat.

Overview
--------

GrainStat uses a comprehensive testing strategy to ensure reliability and correctness:

- **Unit tests** for individual components
- **Integration tests** for complete workflows
- **Performance tests** for speed and memory usage
- **Regression tests** to prevent breaking changes
- **Property-based tests** for algorithmic validation

Test Framework
--------------

We use **pytest** as our primary testing framework, along with several plugins:

- ``pytest`` - Core testing framework
- ``pytest-cov`` - Coverage reporting
- ``pytest-xvfb`` - Headless display for GUI tests
- ``pytest-benchmark`` - Performance benchmarking
- ``hypothesis`` - Property-based testing

Running Tests
-------------

Basic Test Execution
~~~~~~~~~~~~~~~~~~~~

Run the complete test suite:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with verbose output
   pytest -v

   # Run specific test file
   pytest tests/test_grainstat.py

   # Run specific test method
   pytest tests/test_grainstat.py::TestImageLoader::test_load_synthetic_image

Test Categories
~~~~~~~~~~~~~~~

We use pytest markers to categorize tests:

.. code-block:: bash

   # Run only unit tests (fast)
   pytest -m unit

   # Run only integration tests
   pytest -m integration

   # Skip slow tests
   pytest -m "not slow"

   # Run only performance tests
   pytest -m performance

Available Markers
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - Description
   * - ``unit``
     - Fast unit tests for individual functions
   * - ``integration``
     - Tests that use multiple components together
   * - ``slow``
     - Tests that take more than 5 seconds
   * - ``performance``
     - Benchmark and performance tests
   * - ``gpu``
     - Tests requiring GPU acceleration
   * - ``large_memory``
     - Tests requiring >2GB RAM

Coverage Reports
~~~~~~~~~~~~~~~~

Generate test coverage reports:

.. code-block:: bash

   # Run tests with coverage
   pytest --cov=grainstat

   # Generate HTML coverage report
   pytest --cov=grainstat --cov-report=html

   # Generate XML coverage report (for CI)
   pytest --cov=grainstat --cov-report=xml

   # Check coverage threshold
   pytest --cov=grainstat --cov-fail-under=80

Parallel Testing
~~~~~~~~~~~~~~~~

Run tests in parallel for faster execution:

.. code-block:: bash

   # Install pytest-xdist first
   pip install pytest-xdist

   # Run tests in parallel
   pytest -n auto

   # Use specific number of workers
   pytest -n 4

Test Structure
--------------

Directory Organization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── conftest.py                  # Pytest configuration and fixtures
   ├── test_grainstat.py           # Main integration tests
   │
   ├── unit/                       # Unit tests
   │   ├── test_image_io.py
   │   ├── test_preprocessing.py
   │   ├── test_segmentation.py
   │   ├── test_morphology.py
   │   ├── test_properties.py
   │   ├── test_metrics.py
   │   ├── test_statistics.py
   │   └── test_plugins.py
   │
   ├── integration/                # Integration tests
   │   ├── test_full_workflow.py
   │   ├── test_batch_processing.py
   │   ├── test_cli.py
   │   └── test_export.py
   │
   ├── performance/                # Performance tests
   │   ├── test_speed.py
   │   ├── test_memory.py
   │   └── test_scalability.py
   │
   ├── data/                       # Test data
   │   ├── synthetic/
   │   ├── real_samples/
   │   └── expected_results/
   │
   └── utils/                      # Test utilities
       ├── image_generators.py
       ├── assertions.py
       └── fixtures.py

Writing Tests
-------------

Basic Test Structure
~~~~~~~~~~~~~~~~~~~~

Here's a template for writing new tests:

.. code-block:: python

   import pytest
   import numpy as np
   from grainstat.core.metrics import MetricsCalculator

   class TestMetricsCalculator:
       """Test the MetricsCalculator class."""

       def setup_method(self):
           """Set up test fixtures before each test method."""
           self.calculator = MetricsCalculator()

           # Create test data
           self.test_properties = {
               1: {
                   'area_um2': 100.0,
                   'perimeter_um': 35.45,
                   'major_axis_um': 11.28,
                   'minor_axis_um': 11.28,
                   'eccentricity': 0.0,
                   'solidity': 1.0,
                   'bbox_um': (0, 0, 11.28, 11.28),
                   'convex_area_um2': 100.0
               }
           }

       def test_ecd_calculation(self):
           """Test equivalent circular diameter calculation."""
           metrics = self.calculator.calculate_derived_metrics(self.test_properties)

           # Expected ECD for area of 100
           expected_ecd = 2 * np.sqrt(100 / np.pi)
           actual_ecd = metrics[1]['ecd_um']

           assert abs(actual_ecd - expected_ecd) < 0.01

       def test_aspect_ratio_calculation(self):
           """Test aspect ratio calculation."""
           metrics = self.calculator.calculate_derived_metrics(self.test_properties)

           # For a circle, aspect ratio should be 1
           assert metrics[1]['aspect_ratio'] == 1.0

       @pytest.mark.parametrize("area,perimeter,expected_shape_factor", [
           (100, 35.45, 1.0),      # Perfect circle
           (100, 50.0, 0.503),     # Less circular
           (64, 32.0, 0.785),      # Square-like
       ])
       def test_shape_factor_calculation(self, area, perimeter, expected_shape_factor):
           """Test shape factor calculation with various inputs."""
           test_props = {1: {
               'area_um2': area,
               'perimeter_um': perimeter,
               'major_axis_um': 10.0,
               'minor_axis_um': 10.0,
               'eccentricity': 0.0,
               'solidity': 1.0,
               'bbox_um': (0, 0, 10, 10),
               'convex_area_um2': area
           }}

           metrics = self.calculator.calculate_derived_metrics(test_props)

           assert abs(metrics[1]['shape_factor'] - expected_shape_factor) < 0.01

Test Fixtures
~~~~~~~~~~~~~

Use pytest fixtures for reusable test data:

.. code-block:: python

   import pytest
   import numpy as np
   from PIL import Image
   import tempfile

   @pytest.fixture
   def synthetic_microstructure():
       """Create a synthetic microstructure for testing."""

       image = np.zeros((200, 200))
       np.random.seed(42)  # Reproducible results

       # Add circular grains
       for i in range(10):
           x = np.random.randint(30, 170)
           y = np.random.randint(30, 170)
           r = np.random.randint(8, 15)

           yy, xx = np.ogrid[:200, :200]
           mask = (xx - x)**2 + (yy - y)**2 <= r**2
           image[mask] = 1.0

       return image

   @pytest.fixture
   def temporary_image_file(synthetic_microstructure):
       """Create a temporary image file for testing."""

       with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
           img_pil = Image.fromarray((synthetic_microstructure * 255).astype(np.uint8))
           img_pil.save(tmp.name)

           yield tmp.name

           # Cleanup
           import os
           os.unlink(tmp.name)

   @pytest.fixture
   def expected_grain_properties():
       """Expected properties for synthetic microstructure."""

       return {
           'grain_count': 10,
           'mean_ecd_range': (8.0, 15.0),
           'total_area_range': (500, 1500)
       }

Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~~

Use Hypothesis for property-based testing:

.. code-block:: python

   import pytest
   from hypothesis import given, strategies as st
   from grainstat.core.metrics import MetricsCalculator

   class TestMetricsProperties:
       """Property-based tests for metrics calculations."""

       @given(
           area=st.floats(min_value=1.0, max_value=10000.0),
           perimeter=st.floats(min_value=1.0, max_value=1000.0)
       )
       def test_shape_factor_bounds(self, area, perimeter):
           """Shape factor should always be between 0 and 1."""

           shape_factor = (4 * np.pi * area) / (perimeter ** 2)

           assert 0 <= shape_factor <= 1

       @given(
           major_axis=st.floats(min_value=1.0, max_value=100.0),
           minor_axis=st.floats(min_value=0.1, max_value=100.0)
       )
       def test_aspect_ratio_properties(self, major_axis, minor_axis):
           """Aspect ratio should always be >= 1."""

           # Ensure major >= minor
           if minor_axis > major_axis:
               major_axis, minor_axis = minor_axis, major_axis

           aspect_ratio = major_axis / minor_axis

           assert aspect_ratio >= 1.0

Mock Objects
~~~~~~~~~~~~

Use mocks to isolate components under test:

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch, MagicMock
   from grainstat.main import GrainAnalyzer

   class TestGrainAnalyzer:
       """Test GrainAnalyzer with mocked dependencies."""

       def test_analyze_workflow_with_mocks(self):
           """Test the analyze workflow using mocks."""

           # Create analyzer
           analyzer = GrainAnalyzer()

           # Mock the image loader
           mock_image = np.random.rand(100, 100)
           analyzer.loader.load_image = Mock(return_value=mock_image)

           # Mock the segmentation
           mock_labeled = np.random.randint(0, 10, (100, 100))
           analyzer.segmenter.watershed_segmentation = Mock(return_value=mock_labeled)

           # Mock the properties calculation
           mock_properties = {1: {'area_um2': 100, 'perimeter_um': 35}}
           analyzer.properties.calculate_properties = Mock(return_value=mock_properties)

           # Run analysis
           results = analyzer.analyze('fake_image.tif', scale=0.5)

           # Verify mocks were called
           analyzer.loader.load_image.assert_called_once_with('fake_image.tif')
           analyzer.properties.calculate_properties.assert_called_once()

           # Verify results structure
           assert 'properties' in results
           assert 'metrics' in results
           assert 'statistics' in results

Performance Testing
-------------------

Benchmark Tests
~~~~~~~~~~~~~~~

Use pytest-benchmark for performance testing:

.. code-block:: python

   import pytest
   from grainstat import GrainAnalyzer

   class TestPerformance:
       """Performance and benchmark tests."""

       def test_analysis_speed(self, benchmark, synthetic_microstructure):
           """Benchmark the analysis speed."""

           analyzer = GrainAnalyzer()

           # Benchmark the analysis
           result = benchmark(
               analyzer.analyze,
               synthetic_microstructure,
               scale=0.5
           )

           # Verify it completed successfully
           assert 'statistics' in result

       @pytest.mark.performance
       def test_large_image_processing(self):
           """Test processing of large images."""

           # Create large synthetic image
           large_image = np.random.rand(2000, 2000)

           analyzer = GrainAnalyzer()

           import time
           start_time = time.time()

           results = analyzer.analyze(large_image, scale=0.1)

           end_time = time.time()
           processing_time = end_time - start_time

           # Should complete within reasonable time
           assert processing_time < 60  # 60 seconds
           assert results['statistics']['grain_count'] > 0

Memory Testing
~~~~~~~~~~~~~~

Test memory usage patterns:

.. code-block:: python

   import pytest
   import psutil
   import os
   from grainstat import GrainAnalyzer

   class TestMemoryUsage:
       """Test memory usage patterns."""

       @pytest.mark.large_memory
       def test_memory_efficiency(self):
           """Test that memory usage stays reasonable."""

           process = psutil.Process(os.getpid())
           initial_memory = process.memory_info().rss

           analyzer = GrainAnalyzer()

           # Process multiple images
           for i in range(10):
               image = np.random.rand(1000, 1000)
               results = analyzer.analyze(image, scale=0.5)

               current_memory = process.memory_info().rss
               memory_increase = current_memory - initial_memory

               # Memory should not grow excessively
               assert memory_increase < 500 * 1024 * 1024  # 500 MB

Integration Tests
-----------------

End-to-End Workflow Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

Test complete workflows:

.. code-block:: python

   import pytest
   import tempfile
   from pathlib import Path
   from grainstat import GrainAnalyzer
   from grainstat.processing.batch import BatchProcessor

   class TestIntegration:
       """Integration tests for complete workflows."""

       def test_full_analysis_workflow(self, temporary_image_file):
           """Test complete analysis workflow."""

           analyzer = GrainAnalyzer()

           # Run full analysis
           results = analyzer.analyze(
               temporary_image_file,
               scale=0.5,
               min_area=50
           )

           # Verify complete results structure
           assert 'properties' in results
           assert 'metrics' in results
           assert 'statistics' in results

           # Verify statistics
           stats = results['statistics']
           assert 'grain_count' in stats
           assert 'ecd_statistics' in stats
           assert stats['grain_count'] > 0

           # Test export functionality
           with tempfile.TemporaryDirectory() as tmp_dir:
               csv_path = Path(tmp_dir) / 'grains.csv'
               json_path = Path(tmp_dir) / 'analysis.json'
               report_path = Path(tmp_dir) / 'report.html'

               analyzer.export_csv(str(csv_path))
               analyzer.export_json(str(json_path))
               analyzer.generate_report(str(report_path))

               # Verify files were created
               assert csv_path.exists()
               assert json_path.exists()
               assert report_path.exists()

               # Verify file contents
               import pandas as pd
               df = pd.read_csv(csv_path)
               assert len(df) == stats['grain_count']

       def test_batch_processing_workflow(self):
           """Test batch processing workflow."""

           with tempfile.TemporaryDirectory() as input_dir:
               with tempfile.TemporaryDirectory() as output_dir:

                   # Create test images
                   for i in range(3):
                       image = np.random.rand(100, 100)
                       img_pil = Image.fromarray((image * 255).astype(np.uint8))
                       img_pil.save(Path(input_dir) / f'test_{i}.tif')

                   # Run batch processing
                   processor = BatchProcessor(n_workers=2)
                   results = processor.process_directory(
                       input_dir=input_dir,
                       output_dir=output_dir,
                       scale=0.5
                   )

                   # Verify batch results
                   assert results['total_images'] == 3
                   assert results['successful'] >= 0
                   assert results['failed'] >= 0
                   assert results['successful'] + results['failed'] == 3

                   # Verify output files
                   output_path = Path(output_dir)
                   assert (output_path / 'batch_summary.csv').exists()
                   assert (output_path / 'batch_summary.json').exists()

Custom Assertions
-----------------

Create domain-specific assertions:

.. code-block:: python

   def assert_valid_grain_properties(properties):
       """Assert that grain properties are valid."""

       assert isinstance(properties, dict)
       assert len(properties) > 0

       for grain_id, props in properties.items():
           assert isinstance(grain_id, int)
           assert grain_id > 0

           # Required properties
           required_props = ['area_um2', 'perimeter_um', 'ecd_um']
           for prop in required_props:
               assert prop in props
               assert props[prop] > 0

           # Logical constraints
           assert props['aspect_ratio'] >= 1.0
           assert 0 <= props['shape_factor'] <= 1.0
           assert 0 <= props['eccentricity'] <= 1.0

   def assert_valid_statistics(statistics):
       """Assert that statistics are valid."""

       assert 'grain_count' in statistics
       assert statistics['grain_count'] > 0

       if 'ecd_statistics' in statistics:
           ecd_stats = statistics['ecd_statistics']
           assert ecd_stats['mean'] > 0
           assert ecd_stats['std'] >= 0
           assert ecd_stats['min'] <= ecd_stats['max']

Test Data Management
--------------------

Synthetic Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate reproducible test data:

.. code-block:: python

   class SyntheticImageGenerator:
       """Generate synthetic microstructure images for testing."""

       def __init__(self, seed=42):
           np.random.seed(seed)

       def generate_circular_grains(self, image_size=(200, 200),
                                   num_grains=20, grain_size_range=(5, 15)):
           """Generate image with circular grains."""

           image = np.zeros(image_size)

           for _ in range(num_grains):
               x = np.random.randint(grain_size_range[1],
                                   image_size[1] - grain_size_range[1])
               y = np.random.randint(grain_size_range[1],
                                   image_size[0] - grain_size_range[1])
               r = np.random.randint(*grain_size_range)

               yy, xx = np.ogrid[:image_size[0], :image_size[1]]
               mask = (xx - x)**2 + (yy - y)**2 <= r**2
               image[mask] = 1.0

           return image

Continuous Integration
----------------------

GitHub Actions Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example CI configuration:

.. code-block:: yaml

   # .github/workflows/test.yml
   name: Tests

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, windows-latest, macos-latest]
           python-version: [3.8, 3.9, 3.10, 3.11]

       steps:
       - uses: actions/checkout@v3

       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: ${{ matrix.python-version }}

       - name: Install dependencies
         run: |
           pip install -e ".[dev,all]"

       - name: Run tests
         run: |
           pytest --cov=grainstat --cov-report=xml

       - name: Upload coverage
         uses: codecov/codecov-action@v3

Test Best Practices
-------------------

Writing Good Tests
~~~~~~~~~~~~~~~~~~

1. **Test one thing at a time**
2. **Use descriptive test names**
3. **Follow the AAA pattern**: Arrange, Act, Assert
4. **Make tests independent** and reproducible
5. **Use appropriate test data** (synthetic when possible)
6. **Mock external dependencies**
7. **Test edge cases** and error conditions
8. **Keep tests fast** (unit tests < 1 second)

Test Maintenance
~~~~~~~~~~~~~~~~

- **Run tests regularly** during development
- **Update tests** when functionality changes
- **Remove obsolete tests** when features are removed
- **Refactor tests** to reduce duplication
- **Monitor test coverage** and aim for >80%

See Also
--------

- :doc:`contributing` - How to contribute tests
- :doc:`architecture` - Understanding the codebase structure
- `pytest documentation <https://docs.pytest.org/>`_ - Official pytest docs
- `Hypothesis documentation <https://hypothesis.readthedocs.io/>`_ - Property-based testing