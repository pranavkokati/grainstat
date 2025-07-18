Changelog
=========

All notable changes to GrainStat are documented here. The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Version 1.0.0 (2024-07-17)
---------------------------

**Initial Release** 🎉

This is the first stable release of GrainStat, providing a complete grain analysis solution for materials science.

Added
~~~~~

**Core Analysis Engine**
   - Complete grain analysis pipeline from image loading to statistical reporting
   - Multi-format image support (TIFF, PNG, JPEG, BMP, multi-page TIFF)
   - Spatial calibration system for pixel-to-physical unit conversion
   - Advanced preprocessing with Gaussian smoothing and CLAHE contrast enhancement

**Segmentation Capabilities**
   - Otsu global thresholding for uniform illumination
   - Adaptive local thresholding for uneven lighting conditions
   - Watershed segmentation for separating touching grains
   - Morphological operations for noise reduction and grain separation

**Comprehensive Measurements**
   - Basic grain properties: area, perimeter, centroid, bounding box, orientation
   - Shape descriptors: eccentricity, solidity, aspect ratio, equivalent circular diameter
   - Derived metrics: shape factor, compactness, elongation, convexity, sphericity
   - ASTM E112 grain size number calculation with proper formulation
   - Size classification system (ultrafine, fine, medium, coarse, very coarse)

**Statistical Analysis**
   - Complete distribution statistics (mean, median, std dev, percentiles, skewness, kurtosis)
   - Population-level metrics and size uniformity analysis
   - Distribution fitting capabilities (normal, lognormal, gamma, Weibull)
   - Spatial distribution analysis with nearest neighbor calculations

**Visualization Suite**
   - Size distribution histograms with customizable binning
   - Cumulative distribution plots with percentile markers
   - Grain overlay visualization with boundaries and centroids
   - Shape analysis scatter plots for multi-parameter correlation
   - Interactive grain viewer with click-to-inspect functionality
   - Multi-panel summary figures for comprehensive overviews

**Export and Reporting**
   - CSV export with all grain measurements and metadata
   - JSON export for complete analysis results with proper serialization
   - Excel export with multiple sheets for data and statistics
   - Professional HTML reports with embedded plots and statistics tables
   - Markdown reports for integration with documentation systems
   - PDF reports (optional with reportlab dependency)

**Advanced Features**
   - Plugin system for custom feature calculations with decorator syntax
   - Batch processing with parallel execution using multiprocessing
   - Command-line interface with comprehensive argument parsing
   - Interactive analysis tools for detailed grain inspection
   - Condition comparison capabilities for experimental studies

**Technical Implementation**
   - Modular architecture with clean separation of concerns
   - Type hints throughout codebase for better IDE support
   - Comprehensive error handling with informative messages
   - Memory-efficient processing for large images
   - Cross-platform compatibility (Windows, macOS, Linux)
   - Extensive test suite with unit and integration tests

**Documentation**
   - Complete Read the Docs documentation site
   - Step-by-step tutorials for all skill levels
   - Comprehensive API reference with examples
   - Real-world usage examples and case studies
   - Contributing guidelines and development setup

**Command Line Interface**
   - ``grainstat analyze`` for single image analysis
   - ``grainstat batch`` for parallel processing of multiple images
   - ``grainstat interactive`` for launching the interactive viewer
   - ``grainstat compare`` for condition comparison studies
   - ``grainstat version`` for version and system information

Key Formulas Implemented
~~~~~~~~~~~~~~~~~~~~~~~~

- **Equivalent Circular Diameter**: ECD = 2√(A/π)
- **ASTM E112 Grain Size Number**: G = -6.6438 × log₂(L) - 3.293
- **Shape Factor (Circularity)**: φ = 4πA/P²
- **Aspect Ratio**: AR = a/b (major/minor axis lengths)
- **Compactness**: C = P²/(4πA)
- **Various sphericity formulations** for 2D approximation of 3D measures

Dependencies
~~~~~~~~~~~~

**Core Dependencies**
   - numpy>=1.20.0 (numerical computations)
   - scipy>=1.7.0 (scientific computing)
   - scikit-image>=0.18.0 (image processing)
   - matplotlib>=3.3.0 (plotting and visualization)
   - pandas>=1.3.0 (data manipulation)
   - Pillow>=8.0.0 (image I/O)
   - seaborn>=0.11.0 (enhanced visualization)

**Optional Dependencies**
   - reportlab>=3.5.0 (PDF report generation)
   - ipywidgets>=7.0 (Jupyter integration)
   - jupyter>=1.0 (interactive notebooks)

**Development Dependencies**
   - pytest>=6.0 (testing framework)
   - black>=21.0 (code formatting)
   - flake8>=3.8 (linting)
   - mypy>=0.900 (type checking)
   - pre-commit>=2.0.0 (git hooks)

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Memory Usage**: Optimized for images up to 4K resolution
- **Processing Speed**: ~1-5 seconds per image depending on complexity
- **Parallel Processing**: Scales with available CPU cores
- **Grain Detection**: Accurate for grains >10 pixels in area
- **Scale Range**: Supports 0.001 to 100 μm/pixel scales

Known Limitations
~~~~~~~~~~~~~~~~~

- 2D analysis only (3D support planned for future versions)
- Limited to single-phase analysis (multi-phase requires custom features)
- GPU acceleration not included in initial release
- Machine learning features planned for future releases

Migration Notes
~~~~~~~~~~~~~~~

This is the initial release, so no migration is required.

Breaking Changes
~~~~~~~~~~~~~~~~

None (initial release).

Deprecated Features
~~~~~~~~~~~~~~~~~~~

None (initial release).

Security Updates
~~~~~~~~~~~~~~~~

None (initial release).

Planned for Next Releases
-------------------------

Version 1.1.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

**Planned Additions**
   - GPU acceleration for large image processing
   - Machine learning-based grain boundary detection
   - Enhanced statistical analysis tools
   - Additional export formats (HDF5, MATLAB)
   - Performance optimizations

Version 1.2.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

**Planned Additions**
   - 3D grain analysis capabilities
   - Multi-phase material support
   - Integration with crystallographic analysis tools
   - REST API for web integration
   - Cloud processing capabilities

Version 2.0.0 (Future)
~~~~~~~~~~~~~~~~~~~~~~

**Major Features Under Consideration**
   - Complete machine learning integration
   - Real-time analysis capabilities
   - Integration with materials databases
   - Advanced visualization (VR/AR support)
   - Predictive modeling capabilities

Contributing to Releases
------------------------

We welcome contributions for future releases:

**How to Contribute**
   1. Check the `GitHub Issues <https://github.com/materialslab/grainstat/issues>`_ for planned features
   2. Read our :doc:`contributing` guide
   3. Submit pull requests with new features or improvements
   4. Report bugs and suggest enhancements

**Release Process**
   1. Features are developed in feature branches
   2. All changes require code review and testing
   3. Releases follow semantic versioning
   4. Release notes document all changes
   5. Backward compatibility is maintained when possible

**Beta Testing**
   Join our beta testing program to try new features early:

   .. code-block:: bash

      pip install grainstat[beta]

Release Schedule
---------------

**Regular Releases**
   - **Patch releases** (bug fixes): As needed
   - **Minor releases** (new features): Quarterly
   - **Major releases** (breaking changes): Annually

**Long Term Support**
   - LTS versions will be designated for enterprise users
   - LTS versions receive extended support and security updates
   - Version 1.0.0 is the first LTS candidate

**End of Life Policy**
   - Versions are supported for 2 years after release
   - Security updates provided for 1 year after feature support ends
   - Migration guides provided for major version transitions

Changelog Format
---------------

Each entry includes:

**Version Number and Date**
   Following semantic versioning (MAJOR.MINOR.PATCH)

**Change Categories**
   - **Added**: New features
   - **Changed**: Changes in existing functionality
   - **Deprecated**: Soon-to-be removed features
   - **Removed**: Now removed features
   - **Fixed**: Bug fixes
   - **Security**: Security improvements

**Impact Assessment**
   - **Breaking Changes**: Changes that require user action
   - **Migration Notes**: How to adapt existing code
   - **Performance Impact**: Speed/memory usage changes

Getting Notified of Releases
----------------------------

**GitHub Releases**
   Watch the repository on GitHub to get notified of new releases

**PyPI Updates**
   Use tools like ``pip-check`` to monitor for updates:

   .. code-block:: bash

      pip install pip-check
      pip-check grainstat

**Mailing List**
   Subscribe to our announcement mailing list for major releases

**RSS Feed**
   Follow our releases RSS feed: ``https://github.com/materialslab/grainstat/releases.atom``

See Also
--------

- :doc:`installation` - How to install and upgrade GrainStat
- :doc:`contributing` - How to contribute to future releases
- `GitHub Releases <https://github.com/materialslab/grainstat/releases>`_ - Official release page
- `PyPI Project Page <https://pypi.org/project/grainstat/>`_ - Package index