GrainStat Documentation
=======================

.. image:: https://img.shields.io/pypi/v/grainstat.svg
   :target: https://pypi.org/project/grainstat/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/grainstat.svg
   :target: https://pypi.org/project/grainstat/
   :alt: Python Versions

.. image:: https://img.shields.io/github/license/materialslab/grainstat.svg
   :target: https://github.com/materialslab/grainstat/blob/main/LICENSE
   :alt: License

.. image:: https://readthedocs.org/projects/grainstat/badge/?version=latest
   :target: https://grainstat.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**Professional grain size analysis for materials science**

GrainStat is a comprehensive Python package for analyzing grain microstructures in materials science. It provides robust tools for image processing, grain segmentation, statistical analysis, and report generation that are essential for materials characterization.

.. grid:: 2

    .. grid-item-card:: 🚀 Quick Start
        :link: quickstart
        :link-type: doc

        Get up and running with GrainStat in minutes. Learn basic usage patterns and see immediate results.

    .. grid-item-card:: 📖 Tutorials
        :link: tutorials/index
        :link-type: doc

        Step-by-step guides covering everything from basic analysis to advanced features.

    .. grid-item-card:: 📚 API Reference
        :link: api
        :link-type: doc

        Complete documentation of all classes, functions, and methods in GrainStat.

    .. grid-item-card:: 💡 Examples
        :link: examples
        :link-type: doc

        Real-world examples and use cases demonstrating GrainStat's capabilities.

Key Features
------------

🔬 **Comprehensive Image Processing**
   - Support for TIFF, PNG, JPEG, BMP, and multi-page TIFF files
   - Spatial calibration for converting pixel measurements to physical units
   - Advanced preprocessing filters (Gaussian smoothing, CLAHE contrast enhancement)

📊 **Advanced Grain Analysis**
   - Multiple segmentation methods (Otsu, adaptive, watershed)
   - Complete shape characterization (area, perimeter, eccentricity, solidity)
   - Derived metrics (ECD, aspect ratio, shape factor, compactness)
   - ASTM E112 grain size number calculation

📈 **Statistical Analysis**
   - Complete distribution statistics (mean, median, std, percentiles)
   - Population-level metrics and size uniformity analysis
   - Distribution fitting (normal, lognormal, gamma, Weibull)
   - Spatial distribution analysis

🎨 **Professional Visualization**
   - Size distribution histograms and cumulative plots
   - Grain overlay visualization with boundaries and labels
   - Interactive grain viewer with click-to-inspect functionality
   - Multi-panel summary figures

📄 **Export & Reporting**
   - CSV/Excel export for grain data
   - Professional HTML/Markdown/PDF reports
   - JSON export for complete analysis results
   - Batch processing summaries

🔧 **Advanced Features**
   - Plugin system for custom feature calculations
   - Batch processing with parallel execution
   - Command-line interface for automated workflows
   - Extensible architecture for research applications

Installation
------------

Install GrainStat using pip:

.. code-block:: bash

   # Basic installation
   pip install grainstat

   # With all optional dependencies
   pip install grainstat[all]

   # For development
   pip install grainstat[dev]

Quick Example
-------------

.. code-block:: python

   from grainstat import GrainAnalyzer

   # Initialize analyzer
   analyzer = GrainAnalyzer()

   # Analyze a microstructure image
   results = analyzer.analyze(
       image_path="microstructure.tif",
       scale=0.5,  # micrometers per pixel
       min_area=50  # minimum grain area in pixels
   )

   # Export results
   analyzer.export_csv("grain_data.csv")
   analyzer.generate_report("analysis_report.html")

   # Generate plots
   analyzer.plot_histogram(save_path="size_distribution.png")
   analyzer.plot_overlay(save_path="grain_overlay.png")

Command Line Usage
------------------

GrainStat includes a powerful command-line interface:

.. code-block:: bash

   # Single image analysis
   grainstat analyze image.tif --scale 0.2 --export-csv results.csv

   # Batch processing
   grainstat batch input_folder/ output_folder/ --scale 0.2 --workers 4

   # Interactive viewer
   grainstat interactive image.tif --scale 0.2

Key Formulas
------------

GrainStat implements standard materials science formulas:

**Equivalent Circular Diameter (ECD)**

.. math::

   \text{ECD} = 2\sqrt{\frac{A}{\pi}}

**ASTM E112 Grain Size Number**

.. math::

   G = -6.6438 \times \log_2(L) - 3.293

where :math:`L` is the mean lineal intercept length in mm.

**Shape Factor (Circularity)**

.. math::

   \phi = \frac{4\pi A}{P^2}

**Aspect Ratio**

.. math::

   \text{AR} = \frac{a}{b}

where :math:`a` and :math:`b` are the major and minor axis lengths.

Applications
------------

GrainStat is designed for materials scientists, metallurgists, and researchers working with:

- **Optical microscopy** of polished and etched samples
- **SEM imaging** of microstructures
- **Quality control** in materials processing
- **Research** on grain growth, recrystallization, and phase transformations
- **Standards compliance** (ASTM E112, ISO 643)

Support
-------

- 📚 **Documentation**: https://grainstat.readthedocs.io/
- 🐛 **Issue Tracker**: https://github.com/materialslab/grainstat/issues
- 💬 **Discussions**: https://github.com/materialslab/grainstat/discussions
- 📧 **Email**: contact@materialslab.com

Citation
--------

If you use GrainStat in your research, please cite:

.. code-block:: bibtex

   @software{grainstat2024,
     title={GrainStat: Professional grain size analysis for materials science},
     author={Materials Science Lab},
     year={2024},
     url={https://github.com/materialslab/grainstat}
   }

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index
   advanced
   cli
   plugins

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   modules

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing
   architecture
   testing

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   license
   glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`