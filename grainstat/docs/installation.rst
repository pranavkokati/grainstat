Installation Guide
==================

This guide covers different ways to install GrainStat and its dependencies.

Requirements
------------

**Python Version**
   GrainStat requires Python 3.8 or higher.

**Operating Systems**
   - Windows 10/11
   - macOS 10.15+ (Catalina or later)
   - Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)

**Hardware Requirements**
   - Minimum: 4 GB RAM, 1 GB disk space
   - Recommended: 8+ GB RAM for large image processing
   - Optional: Multi-core CPU for batch processing acceleration

Basic Installation
------------------

Install from PyPI
~~~~~~~~~~~~~~~~~~

The easiest way to install GrainStat is using pip:

.. code-block:: bash

   pip install grainstat

This installs GrainStat with its core dependencies.

Install with Optional Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For additional functionality, install with optional dependencies:

.. code-block:: bash

   # All optional features
   pip install grainstat[all]

   # PDF report generation
   pip install grainstat[pdf]

   # Interactive Jupyter widgets
   pip install grainstat[interactive]

   # Development tools
   pip install grainstat[dev]

Development Installation
------------------------

For developers who want to contribute or modify GrainStat:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/materialslab/grainstat.git
   cd grainstat

   # Install in development mode
   pip install -e ".[dev,all]"

   # Set up pre-commit hooks
   pre-commit install

Alternative Installation Methods
--------------------------------

Using Conda
~~~~~~~~~~~~

If you prefer conda:

.. code-block:: bash

   # Create a new environment
   conda create -n grainstat python=3.11
   conda activate grainstat

   # Install dependencies
   conda install numpy scipy scikit-image matplotlib pandas pillow seaborn

   # Install GrainStat
   pip install grainstat

Using Poetry
~~~~~~~~~~~~~

For Python project management with Poetry:

.. code-block:: bash

   # Add to your project
   poetry add grainstat

   # With optional dependencies
   poetry add grainstat[all]

Docker Installation
~~~~~~~~~~~~~~~~~~~

Run GrainStat in a Docker container:

.. code-block:: bash

   # Pull the image
   docker pull materialslab/grainstat:latest

   # Run interactively
   docker run -it -v $(pwd):/workspace materialslab/grainstat:latest

Dependency Details
------------------

Core Dependencies
~~~~~~~~~~~~~~~~~

These are automatically installed with GrainStat:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Purpose
   * - numpy
     - ≥1.20.0
     - Numerical computations and array operations
   * - scipy
     - ≥1.7.0
     - Scientific computing and image processing
   * - scikit-image
     - ≥0.18.0
     - Image processing and computer vision
   * - matplotlib
     - ≥3.3.0
     - Plotting and visualization
   * - pandas
     - ≥1.3.0
     - Data manipulation and export
   * - Pillow
     - ≥8.0.0
     - Image I/O and format support
   * - seaborn
     - ≥0.11.0
     - Enhanced statistical visualization

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Purpose
   * - reportlab
     - ≥3.5.0
     - PDF report generation
   * - ipywidgets
     - ≥7.0
     - Interactive Jupyter widgets
   * - jupyter
     - ≥1.0
     - Jupyter notebook integration

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Package
     - Version
     - Purpose
   * - pytest
     - ≥6.0
     - Testing framework
   * - pytest-cov
     - ≥2.0
     - Coverage reporting
   * - black
     - ≥21.0
     - Code formatting
   * - flake8
     - ≥3.8
     - Code linting
   * - mypy
     - ≥0.900
     - Type checking
   * - pre-commit
     - ≥2.0.0
     - Git hooks

Verification
------------

Verify Installation
~~~~~~~~~~~~~~~~~~~

Test that GrainStat is installed correctly:

.. code-block:: python

   import grainstat
   print(f"GrainStat version: {grainstat.__version__}")

   # Test basic functionality
   from grainstat import GrainAnalyzer
   analyzer = GrainAnalyzer()
   print("Installation successful!")

Run Tests
~~~~~~~~~

If you installed the development version, run the test suite:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=grainstat

   # Run specific test module
   pytest tests/test_grainstat.py

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Verify the CLI is working:

.. code-block:: bash

   grainstat --help
   grainstat version

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'grainstat'**
   - Ensure you're using the correct Python environment
   - Try reinstalling: ``pip uninstall grainstat && pip install grainstat``

**Segmentation fault or crashes**
   - Update your graphics drivers
   - Try running without display: ``export MPLBACKEND=Agg``
   - Install with conda if using pip fails

**Memory errors with large images**
   - Reduce image size or use batch processing
   - Increase system memory or use swap space
   - Process images in smaller chunks

**ImportError with optional dependencies**
   - Install optional features: ``pip install grainstat[all]``
   - Check specific package installation: ``pip show reportlab``

**Permission errors on Windows**
   - Run command prompt as administrator
   - Use user installation: ``pip install --user grainstat``

**M1 Mac compatibility issues**
   - Use conda for better ARM64 support:

   .. code-block:: bash

      conda install -c conda-forge numpy scipy scikit-image
      pip install grainstat

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. **Check the FAQ**: Common solutions in our documentation
2. **Search Issues**: https://github.com/materialslab/grainstat/issues
3. **Create New Issue**: Include error messages and system info
4. **Discussions**: https://github.com/materialslab/grainstat/discussions

System Information
~~~~~~~~~~~~~~~~~~

When reporting issues, include:

.. code-block:: python

   import sys
   import platform
   import grainstat
   import numpy as np
   import scipy
   import skimage

   print(f"Python: {sys.version}")
   print(f"Platform: {platform.platform()}")
   print(f"GrainStat: {grainstat.__version__}")
   print(f"NumPy: {np.__version__}")
   print(f"SciPy: {scipy.__version__}")
   print(f"scikit-image: {skimage.__version__}")

Upgrading
---------

Upgrade to Latest Version
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade grainstat

Check for Updates
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip list --outdated | grep grainstat

Uninstallation
--------------

To remove GrainStat:

.. code-block:: bash

   pip uninstall grainstat

This removes GrainStat but keeps its dependencies. To remove dependencies:

.. code-block:: bash

   pip uninstall grainstat numpy scipy scikit-image matplotlib pandas pillow seaborn

Performance Optimization
-------------------------

For optimal performance:

**NumPy Configuration**
   Ensure NumPy uses optimized BLAS libraries:

   .. code-block:: python

      import numpy as np
      np.show_config()

**Parallel Processing**
   Install with multiprocessing support and use multiple cores:

   .. code-block:: bash

      export OMP_NUM_THREADS=4
      grainstat batch input/ output/ --workers 4

**Memory Management**
   For large datasets, monitor memory usage:

   .. code-block:: python

      import psutil
      print(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")

Next Steps
----------

After installation:

1. :doc:`quickstart` - Learn basic usage
2. :doc:`tutorials/index` - Follow step-by-step guides
3. :doc:`examples` - See real-world applications
4. :doc:`api` - Explore the full API reference