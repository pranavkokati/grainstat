# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-07-17

### Added

#### Core Features
- **Complete grain analysis pipeline** from image loading to statistical reporting
- **Multi-format image support**: TIFF, PNG, JPEG, BMP, multi-page TIFF
- **Spatial calibration** system for converting pixel measurements to physical units
- **Advanced preprocessing** with Gaussian smoothing and CLAHE contrast enhancement
- **Multiple segmentation methods**: Otsu thresholding, adaptive thresholding, watershed
- **Comprehensive morphological operations** for noise reduction and grain separation

#### Analysis Capabilities
- **Basic grain properties**: area, perimeter, centroid, bounding box, orientation
- **Shape descriptors**: eccentricity, solidity, aspect ratio, equivalent circular diameter
- **Derived metrics**: shape factor, compactness, elongation, convexity, sphericity
- **ASTM E112 grain size number** calculation with proper formulation
- **Size classification**: ultrafine, fine, medium, coarse, very coarse categories
- **Complete statistical analysis**: mean, median, std dev, percentiles, skewness, kurtosis

#### Visualization Suite
- **Size distribution histograms** with customizable binning
- **Cumulative distribution plots** with percentile markers
- **Grain overlay visualization** with boundaries and centroids
- **Shape analysis scatter plots** for multi-parameter correlation
- **Interactive grain viewer** with click-to-inspect functionality
- **Multi-panel summary figures** for comprehensive overviews

#### Export & Reporting
- **CSV export** with all grain measurements and metadata
- **JSON export** for complete analysis results with proper serialization
- **Excel export** with multiple sheets for data and statistics
- **Professional HTML reports** with embedded plots and statistics tables
- **Markdown reports** for integration with documentation systems
- **PDF reports** (optional with reportlab dependency)

#### Advanced Features
- **Plugin system** for custom feature calculations with decorator syntax
- **Batch processing** with parallel execution using multiprocessing
- **Command-line interface** with comprehensive argument parsing
- **Interactive analysis tools** for detailed grain inspection
- **Condition comparison** capabilities for experimental studies

#### Technical Implementation
- **Modular architecture** with clean separation of concerns
- **Type hints** throughout codebase for better IDE support
- **Comprehensive error handling** with informative messages
- **Memory-efficient processing** for large images
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Extensive test suite** with unit and integration tests

### Core Modules

#### Image Processing (`grainstat.core`)
- `image_io.py`: Multi-format image loading with automatic conversion
- `preprocessing.py`: Gaussian smoothing, CLAHE, noise reduction
- `segmentation.py`: Otsu, adaptive, watershed, and region growing methods
- `morphology.py`: Opening, closing, hole filling, object separation
- `properties.py`: RegionProps calculation with scale conversion
- `metrics.py`: Derived measurements and shape analysis
- `statistics.py`: Population statistics and distribution fitting

#### Visualization (`grainstat.visualization`)
- `plots.py`: Comprehensive plotting suite with publication-quality figures
- `interactive.py`: Interactive viewer with grain selection and property display

#### Export (`grainstat.export`)
- `data_export.py`: Multiple format export with proper data serialization
- `reports.py`: Professional report generation with embedded visualizations

#### Processing (`grainstat.processing`)
- `batch.py`: Parallel batch processing with progress tracking

#### Plugins (`grainstat.plugins`)
- `base.py`: Plugin architecture with feature decorator and registration system

### Key Formulas Implemented

- **Equivalent Circular Diameter**: `ECD = 2√(A/π)`
- **ASTM E112 Grain Size Number**: `G = -6.6438 × log₂(L) - 3.293`
- **Shape Factor (Circularity)**: `φ = 4πA/P²`
- **Aspect Ratio**: `AR = a/b` (major/minor axis lengths)
- **Compactness**: `C = P²/(4πA)`
- **Sphericity**: Various formulations for 2D approximation of 3D measure

### Dependencies
- numpy>=1.20.0
- scipy>=1.7.0
- scikit-image>=0.18.0
- matplotlib>=3.3.0
- pandas>=1.3.0
- Pillow>=8.0.0
- seaborn>=0.11.0

### Optional Dependencies
- reportlab>=3.5.0 (PDF reports)
- ipywidgets>=7.0 (Jupyter integration)
- jupyter>=1.0 (interactive notebooks)

### Development Tools
- pytest>=6.0 (testing framework)
- black>=21.0 (code formatting)
- flake8>=3.8 (linting)
- mypy>=0.900 (type checking)
- pre-commit>=2.0.0 (git hooks)

### Command Line Interface
- `grainstat analyze` - Single image analysis
- `grainstat batch` - Batch processing
- `grainstat interactive` - Interactive viewer
- `grainstat compare` - Condition comparison
- `grainstat version` - Version information

### Example Usage

```python
from grainstat import GrainAnalyzer

analyzer = GrainAnalyzer()
results = analyzer.analyze("microstructure.tif", scale=0.5)

# Export results
analyzer.export_csv("grain_data.csv")
analyzer.generate_report("report.html")

# Generate visualizations
analyzer.plot_histogram(save_path="size_dist.png")
analyzer.plot_overlay(save_path="overlay.png")
```

### Documentation
- Comprehensive README with examples and formulas
- Inline code documentation with docstrings
- Type hints for all public APIs
- Example scripts demonstrating key features

### Testing
- Unit tests for all core modules
- Integration tests for end-to-end workflows
- Synthetic image generation for reproducible testing
- Coverage reporting and continuous integration ready

### Performance
- Efficient memory usage for large images
- Parallel processing capabilities
- Optimized algorithms for grain boundary detection
- Minimal external dependencies for core functionality

---

## [Unreleased]

### Planned Features
- GPU acceleration for large image processing
- Machine learning-based grain boundary detection
- 3D grain analysis capabilities
- Additional statistical distribution models
- Enhanced visualization options
- REST API for web integration
- Integration with materials databases

### Known Issues
- None at initial release

---

**Note**: This is the initial release of GrainStat. The version follows semantic versioning principles where:
- MAJOR version changes indicate incompatible API changes
- MINOR version changes add functionality in a backwards compatible manner  
- PATCH version changes include backwards compatible bug fixes