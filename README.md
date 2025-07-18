# GrainStat

**Professional grain size analysis for materials science**

GrainStat is a comprehensive Python package for analyzing grain microstructures in materials science. It provides robust tools for image processing, grain segmentation, statistical analysis, and report generation that are essential for materials characterization.

## Features

### 🔬 **Image Processing & Segmentation**
- Support for TIFF, PNG, JPEG, BMP, and multi-page TIFF files
- Spatial calibration for converting pixel measurements to physical units
- Advanced preprocessing filters (Gaussian smoothing, CLAHE contrast enhancement)
- Multiple thresholding methods (Otsu, adaptive)
- Watershed segmentation for separating touching grains
- Morphological operations for noise reduction

### 📊 **Comprehensive Grain Analysis**
- **Basic properties**: Area, perimeter, centroid, bounding box
- **Shape descriptors**: Eccentricity, solidity, aspect ratio, orientation
- **Derived metrics**: Equivalent circular diameter (ECD), shape factor, compactness
- **ASTM E112 grain size number** calculation
- **Size classification**: Ultrafine, fine, medium, coarse, very coarse

### 📈 **Statistical Analysis**
- Complete distribution statistics (mean, median, std, percentiles)
- Skewness and kurtosis analysis
- Population-level metrics and size uniformity
- Spatial distribution analysis
- Distribution fitting (normal, lognormal, gamma, Weibull)

### 🎨 **Visualization Suite**
- Size distribution histograms
- Cumulative distribution plots
- Grain overlay visualization with boundaries and labels
- Shape analysis scatter plots
- Multi-panel summary figures
- Interactive grain viewer with click-to-inspect functionality

### 📄 **Export & Reporting**
- CSV/Excel export for grain data
- JSON export for complete analysis results
- Professional HTML/Markdown/PDF reports
- Batch processing summaries
- Grain boundary coordinate export

### 🔧 **Advanced Features**
- **Plugin system** for custom feature calculations
- **Batch processing** with parallel execution
- **Command-line interface** for automated workflows
- **Interactive viewer** for detailed grain inspection
- **Condition comparison** for experimental analysis

## Installation

```bash
# Basic installation
pip install grainstat

# With all optional dependencies
pip install grainstat[all]

# For development
pip install grainstat[dev]
```

### Requirements
- Python 3.8+
- NumPy, SciPy, scikit-image
- Matplotlib, Pandas, Pillow
- Seaborn (for enhanced plotting)

## Quick Start

### Basic Analysis

```python
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
```

### Command Line Usage

```bash
# Single image analysis
grainstat analyze image.tif --scale 0.2 --export-csv results.csv --report report.html

# Batch processing
grainstat batch input_folder/ output_folder/ --scale 0.2 --workers 4

# Interactive viewer
grainstat interactive image.tif --scale 0.2

# Compare different conditions
grainstat compare --conditions condition1:path1 condition2:path2 output_dir/
```

### Batch Processing

```python
from grainstat.processing.batch import BatchProcessor

processor = BatchProcessor(n_workers=4)

results = processor.process_directory(
    input_dir="sample_images/",
    output_dir="batch_results/",
    scale=0.5,
    pattern="*.tif"
)
```

### Custom Features with Plugins

```python
from grainstat import feature

@feature
def custom_roundness(region):
    """Calculate custom roundness metric"""
    area = region.area
    perimeter = region.perimeter
    return (4 * 3.14159 * area) / (perimeter ** 2)

@feature(name="grain_complexity")
def calculate_complexity(region):
    """Multi-parameter complexity measure"""
    return {
        'shape_complexity': region.eccentricity * (2 - region.solidity),
        'size_factor': region.area / region.convex_area
    }
```

## Analysis Workflow

1. **Image Loading**: Automatic format detection and conversion to grayscale
2. **Preprocessing**: Gaussian smoothing and contrast enhancement (CLAHE)
3. **Segmentation**: Otsu or adaptive thresholding followed by watershed
4. **Morphological Cleaning**: Opening, closing, and small object removal
5. **Property Calculation**: Comprehensive geometric and shape analysis
6. **Statistical Analysis**: Population statistics and distribution fitting
7. **Visualization**: Multiple plot types for data exploration
8. **Export**: Professional reports and data files

## Key Formulas

### Equivalent Circular Diameter (ECD)
```
ECD = 2√(A/π)
```

### ASTM E112 Grain Size Number
```
G = -6.6438 × log₂(L) - 3.293
```
where L is the mean lineal intercept length in mm.

### Shape Factor (Circularity)
```
φ = 4πA/P²
```

### Aspect Ratio
```
AR = a/b
```
where a and b are the major and minor axis lengths.

## Output Examples

### Statistical Summary
```
GRAIN ANALYSIS SUMMARY
======================
Total grains detected: 1,247
Mean ECD: 12.3 μm
Median ECD: 10.8 μm
ASTM Grain Size Number: 8.2

Size Class Distribution:
  Fine: 423 grains (33.9%)
  Medium: 681 grains (54.6%)
  Coarse: 143 grains (11.5%)
```

### Exported Data
The CSV output includes all measured properties:
- `grain_id`, `ecd_um`, `area_um2`, `perimeter_um`
- `aspect_ratio`, `shape_factor`, `eccentricity`, `solidity`
- `major_axis_um`, `minor_axis_um`, `orientation`
- Custom plugin features

## Applications

GrainStat is designed for materials scientists, metallurgists, and researchers working with:

- **Optical microscopy** of polished and etched samples
- **SEM imaging** of microstructures
- **Quality control** in materials processing
- **Research** on grain growth, recrystallization, and phase transformations
- **Standards compliance** (ASTM E112, ISO 643)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GrainStat in your research, please cite:

```bibtex
@software{grainstat2024,
  title={GrainStat: Professional grain size analysis for materials science},
  author={Materials Science Lab},
  year={2024},
  url={https://github.com/materialslab/grainstat}
}
```

## Support

- 📚 [Documentation](https://grainstat.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/materialslab/grainstat/issues)
- 💬 [Discussions](https://github.com/materialslab/grainstat/discussions)

---

**Made with ❤️ for the materials science community**
