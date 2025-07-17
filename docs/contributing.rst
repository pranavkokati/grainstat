Contributing Guide
=================

We welcome contributions to GrainStat! This guide will help you get started with contributing code, documentation, or reporting issues.

Types of Contributions
----------------------

We appreciate all types of contributions:

- 🐛 **Bug reports and fixes**
- 💡 **Feature requests and implementations**
- 📚 **Documentation improvements**
- 🧪 **Test cases and examples**
- 🔧 **Performance optimizations**
- 🎨 **User interface improvements**

Getting Started
---------------

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

.. code-block:: bash

   git clone https://github.com/yourusername/grainstat.git
   cd grainstat

3. **Set up development environment**:

.. code-block:: bash

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode with all dependencies
   pip install -e ".[dev,all]"

   # Install pre-commit hooks
   pre-commit install

4. **Verify installation**:

.. code-block:: bash

   # Run tests
   pytest

   # Check code style
   make lint

   # Run example
   python examples/basic_usage.py

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Create a feature branch**:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. **Make your changes** following our coding standards
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Run the test suite**:

.. code-block:: bash

   make test

6. **Check code style**:

.. code-block:: bash

   make lint

7. **Commit your changes**:

.. code-block:: bash

   git add .
   git commit -m "Add your feature description"

8. **Push to your fork**:

.. code-block:: bash

   git push origin feature/your-feature-name

9. **Create a Pull Request** on GitHub

Coding Standards
----------------

Code Style
~~~~~~~~~~

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks with:

.. code-block:: bash

   make lint
   make format  # Auto-format code

Type Hints
~~~~~~~~~~

Use type hints for all public functions:

.. code-block:: python

   from typing import Dict, List, Optional, Tuple
   import numpy as np

   def analyze_grains(image: np.ndarray, scale: float,
                     min_area: int = 50) -> Dict[str, Any]:
       """Analyze grains in the given image.

       Args:
           image: Input grayscale image
           scale: Micrometers per pixel
           min_area: Minimum grain area in pixels

       Returns:
           Dictionary containing analysis results
       """
       pass

Documentation Strings
~~~~~~~~~~~~~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def calculate_grain_metrics(properties: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
       """Calculate derived grain metrics from basic properties.

       This function computes advanced grain characteristics like equivalent
       circular diameter, aspect ratio, and shape factors from basic
       geometric properties.

       Args:
           properties: Dictionary mapping grain IDs to their basic properties.
               Each property dict should contain 'area_um2', 'perimeter_um',
               'major_axis_um', 'minor_axis_um', etc.

       Returns:
           Dictionary mapping grain IDs to extended metrics including all
           original properties plus derived measurements.

       Raises:
           ValueError: If properties dict is empty or missing required keys.

       Example:
           >>> properties = {1: {'area_um2': 100, 'perimeter_um': 35.4}}
           >>> metrics = calculate_grain_metrics(properties)
           >>> print(metrics[1]['ecd_um'])
           11.28
       """
       pass

Testing Guidelines
------------------

Writing Tests
~~~~~~~~~~~~~

We use pytest for testing. Write tests for all new functionality:

.. code-block:: python

   import pytest
   import numpy as np
   from grainstat.core.metrics import MetricsCalculator

   class TestMetricsCalculator:

       def setup_method(self):
           """Set up test fixtures."""
           self.calculator = MetricsCalculator()

           # Create test data
           self.test_properties = {
               1: {
                   'area_um2': 100.0,
                   'perimeter_um': 35.45,
                   'major_axis_um': 11.28,
                   'minor_axis_um': 11.28,
                   'eccentricity': 0.0,
                   'solidity': 1.0
               }
           }

       def test_ecd_calculation(self):
           """Test equivalent circular diameter calculation."""
           metrics = self.calculator.calculate_derived_metrics(self.test_properties)

           expected_ecd = 2 * np.sqrt(100 / np.pi)
           assert abs(metrics[1]['ecd_um'] - expected_ecd) < 0.01

       def test_aspect_ratio_calculation(self):
           """Test aspect ratio calculation."""
           metrics = self.calculator.calculate_derived_metrics(self.test_properties)

           assert metrics[1]['aspect_ratio'] == 1.0

       @pytest.mark.parametrize("area,perimeter,expected", [
           (100, 35.45, 1.0),  # Perfect circle
           (100, 50, 0.5),     # Less circular
           (100, 25, 2.0),     # More circular than possible (edge case)
       ])
       def test_shape_factor_calculation(self, area, perimeter, expected):
           """Test shape factor calculation with various inputs."""
           test_props = {1: {'area_um2': area, 'perimeter_um': perimeter}}
           # Add other required properties...

           metrics = self.calculator.calculate_derived_metrics(test_props)

           assert abs(metrics[1]['shape_factor'] - expected) < 0.1

Test Categories
~~~~~~~~~~~~~~~

Use pytest markers to categorize tests:

.. code-block:: python

   import pytest

   @pytest.mark.unit
   def test_basic_calculation():
       """Unit test for basic calculation."""
       pass

   @pytest.mark.integration
   def test_full_workflow():
       """Integration test for complete workflow."""
       pass

   @pytest.mark.slow
   def test_large_image_processing():
       """Slow test that processes large images."""
       pass

Run specific test categories:

.. code-block:: bash

   # Run only unit tests
   pytest -m unit

   # Run all except slow tests
   pytest -m "not slow"

   # Run with coverage
   pytest --cov=grainstat

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~~

We use Sphinx for documentation:

.. code-block:: bash

   # Install documentation dependencies
   pip install -r docs/requirements.txt

   # Build HTML documentation
   cd docs
   make html

   # View documentation
   open _build/html/index.html

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

Documentation uses reStructuredText (RST) format:

.. code-block:: rst

   Section Title
   =============

   Subsection Title
   ----------------

   **Bold text** and *italic text*.

   Code blocks:

   .. code-block:: python

      import grainstat
      analyzer = grainstat.GrainAnalyzer()

   Math equations:

   .. math::

      \text{ECD} = 2\sqrt{\frac{A}{\pi}}

Adding Examples
~~~~~~~~~~~~~~~

Add examples to the ``examples/`` directory:

.. code-block:: python

   """
   Example: Custom Feature Development

   This example demonstrates how to create custom grain features
   for specialized analysis applications.
   """

   from grainstat import GrainAnalyzer, feature

   @feature
   def custom_metric(region):
       """Calculate a custom grain metric."""
       return region.area_um2 / region.perimeter_um

   def main():
       analyzer = GrainAnalyzer()
       results = analyzer.analyze("sample.tif", scale=0.5)

       # Custom features are automatically included
       for grain_id, grain in results['metrics'].items():
           print(f"Grain {grain_id}: {grain['custom_metric']:.3f}")

   if __name__ == "__main__":
       main()

Pull Request Guidelines
-----------------------

Creating Good Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clear title and description**:
   - Use descriptive title: "Add watershed segmentation optimization"
   - Explain what the PR does and why
   - Reference related issues: "Fixes #123"

2. **Keep changes focused**:
   - One feature/fix per PR
   - Avoid unrelated changes
   - Keep PRs reasonably sized

3. **Include tests**:
   - Add tests for new functionality
   - Ensure all tests pass
   - Maintain or improve test coverage

4. **Update documentation**:
   - Update docstrings for modified functions
   - Add examples for new features
   - Update user guides if needed

5. **Follow checklist**:

.. code-block:: markdown

   ## Pull Request Checklist

   - [ ] Code follows project style guidelines
   - [ ] Tests pass locally (`make test`)
   - [ ] New tests added for new functionality
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated (for significant changes)
   - [ ] No breaking changes (or properly documented)

Review Process
~~~~~~~~~~~~~~

All pull requests go through code review:

1. **Automated checks** run (tests, linting, documentation)
2. **Maintainer review** for code quality and design
3. **Discussion** if changes are needed
4. **Approval and merge** when ready

Responding to Review Feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Address all review comments
- Ask questions if feedback is unclear
- Make requested changes in new commits
- Don't force-push during review (loses context)

Reporting Issues
----------------

Bug Reports
~~~~~~~~~~~

Use our bug report template:

.. code-block:: markdown

   **Bug Description**
   Clear description of the bug.

   **To Reproduce**
   Steps to reproduce the behavior:
   1. Load image 'sample.tif'
   2. Run analysis with scale=0.5
   3. See error

   **Expected Behavior**
   What you expected to happen.

   **Screenshots/Output**
   Error messages or relevant output.

   **Environment:**
   - OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
   - Python version: [e.g., 3.9.7]
   - GrainStat version: [e.g., 1.0.0]
   - Other relevant packages and versions

Feature Requests
~~~~~~~~~~~~~~~~

Use our feature request template:

.. code-block:: markdown

   **Feature Description**
   Clear description of the desired feature.

   **Use Case**
   Explain why this feature would be useful.

   **Proposed Solution**
   Describe how you envision this working.

   **Alternatives Considered**
   Any alternative approaches you've considered.

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We follow a code of conduct to ensure a welcoming environment:

- **Be respectful** and inclusive
- **Be constructive** in feedback and discussions
- **Be patient** with newcomers
- **Focus on the issue**, not the person

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions and reviews

Recognition
-----------

Contributors are recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation** for major features

Release Process
---------------

For Maintainers
~~~~~~~~~~~~~~~

1. **Version bump** in ``__init__.py`` and ``pyproject.toml``
2. **Update CHANGELOG.md** with new features and fixes
3. **Create release tag**:

.. code-block:: bash

   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0

4. **Build and upload** to PyPI:

.. code-block:: bash

   make clean
   make build
   make upload

5. **Create GitHub release** with release notes

Getting Help
------------

If you need help contributing:

1. **Check existing issues** and discussions
2. **Read this guide** and other documentation
3. **Ask questions** in GitHub Discussions
4. **Join our community** and connect with other contributors

Thank you for contributing to GrainStat! Your contributions help make grain analysis more accessible and powerful for the materials science community.