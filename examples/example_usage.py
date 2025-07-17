"""
Basic usage example for GrainStat

This script demonstrates the basic workflow for analyzing grain microstructures
using the GrainStat package.
"""

import os
import numpy as np
from pathlib import Path

# Import GrainStat
from grainstat import GrainAnalyzer, feature


def main():
    # Example image path (replace with your actual image)
    image_path = "sample_microstructure.tif"

    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"Example image not found: {image_path}")
        print("Please provide a microstructure image file.")
        return

    # Create analyzer instance
    analyzer = GrainAnalyzer()

    # Define analysis parameters
    scale = 0.5  # micrometers per pixel
    min_area = 50  # minimum grain area in pixels

    print(f"Analyzing image: {image_path}")
    print(f"Scale: {scale} μm/pixel")

    # Perform grain analysis
    results = analyzer.analyze(
        image_path=image_path,
        scale=scale,
        min_area=min_area,
        gaussian_sigma=1.0,
        threshold_method='otsu',
        use_watershed=True,
        morphology_radius=2
    )

    # Print analysis summary
    print_results_summary(results)

    # Export results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Export grain data to CSV
    csv_path = output_dir / "grain_data.csv"
    analyzer.export_csv(str(csv_path))
    print(f"Grain data exported to: {csv_path}")

    # Export complete results to JSON
    json_path = output_dir / "complete_analysis.json"
    analyzer.export_json(str(json_path))
    print(f"Complete analysis exported to: {json_path}")

    # Generate plots
    hist_path = output_dir / "size_histogram.png"
    analyzer.plot_histogram(save_path=str(hist_path))
    print(f"Size histogram saved to: {hist_path}")

    overlay_path = output_dir / "grain_overlay.png"
    analyzer.plot_overlay(save_path=str(overlay_path))
    print(f"Grain overlay saved to: {overlay_path}")

    cdf_path = output_dir / "cumulative_distribution.png"
    analyzer.plot_cumulative_distribution(save_path=str(cdf_path))
    print(f"Cumulative distribution saved to: {cdf_path}")

    # Generate comprehensive HTML report
    report_path = output_dir / "analysis_report.html"
    analyzer.generate_report(str(report_path), format_type='html')
    print(f"HTML report generated: {report_path}")

    print("\nAnalysis complete!")


def print_results_summary(results):
    """Print a summary of the analysis results"""

    print("\n" + "=" * 60)
    print("GRAIN ANALYSIS SUMMARY")
    print("=" * 60)

    statistics = results.get('statistics', {})

    # Basic counts
    grain_count = statistics.get('grain_count', 0)
    print(f"Total grains detected: {grain_count}")

    # ECD statistics
    ecd_stats = statistics.get('ecd_statistics', {})
    if ecd_stats:
        print(f"\nEquivalent Circular Diameter (ECD) Statistics:")
        print(f"  Mean:      {ecd_stats.get('mean', 0):.2f} μm")
        print(f"  Median:    {ecd_stats.get('median', 0):.2f} μm")
        print(f"  Std Dev:   {ecd_stats.get('std', 0):.2f} μm")
        print(f"  Range:     {ecd_stats.get('min', 0):.2f} - {ecd_stats.get('max', 0):.2f} μm")
        print(f"  25th %ile: {ecd_stats.get('q25', 0):.2f} μm")
        print(f"  75th %ile: {ecd_stats.get('q75', 0):.2f} μm")

    # ASTM grain size
    astm_stats = statistics.get('astm_grain_size', {})
    if astm_stats and 'grain_size_number' in astm_stats:
        astm_number = astm_stats['grain_size_number']
        if astm_number is not None:
            print(f"\nASTM E112 Grain Size Number: {astm_number:.1f}")

    # Size class distribution
    size_dist = statistics.get('size_class_distribution', {})
    if size_dist:
        print(f"\nGrain Size Classification:")
        total_grains = sum(size_dist.values())
        for size_class, count in size_dist.items():
            if count > 0:
                percentage = (count / total_grains) * 100
                print(f"  {size_class:12}: {count:4d} grains ({percentage:5.1f}%)")

    # Area statistics
    area_stats = statistics.get('area_statistics', {})
    if area_stats:
        print(f"\nArea Statistics:")
        print(f"  Mean area: {area_stats.get('mean', 0):.2f} μm²")
        total_area = statistics.get('total_grain_area_um2', 0)
        if total_area > 0:
            print(f"  Total area: {total_area:.2f} μm²")

    print("=" * 60)


# Example of creating custom features
@feature
def grain_elongation_category(region):
    """Categorize grains by elongation"""
    aspect_ratio = getattr(region, 'aspect_ratio', 1.0)

    if aspect_ratio < 1.2:
        return 'equiaxed'
    elif aspect_ratio < 1.5:
        return 'slightly_elongated'
    elif aspect_ratio < 2.0:
        return 'elongated'
    else:
        return 'very_elongated'


@feature(name="texture_parameter")
def calculate_texture_parameter(region):
    """Calculate a custom texture parameter"""
    orientation = getattr(region, 'orientation', 0)
    eccentricity = getattr(region, 'eccentricity', 0)

    # Normalize orientation to 0-1 range
    normalized_orientation = abs(orientation) / (np.pi / 2)

    # Combine orientation and shape
    texture_param = normalized_orientation * eccentricity

    return texture_param


def demonstrate_batch_processing():
    """Example of batch processing multiple images"""

    from grainstat.processing.batch import BatchProcessor

    # Initialize batch processor
    processor = BatchProcessor(n_workers=2)  # Use 2 CPU cores

    input_directory = "sample_images/"
    output_directory = "batch_results/"

    if not os.path.exists(input_directory):
        print(f"Batch demo skipped: {input_directory} not found")
        return

    print(f"\nRunning batch analysis on {input_directory}")

    # Process all TIFF images in the directory
    results = processor.process_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        pattern="*.tif*",
        scale=0.5,  # μm/pixel
        analysis_params={
            'min_area': 50,
            'gaussian_sigma': 1.0,
            'threshold_method': 'otsu',
            'use_watershed': True,
            'morphology_radius': 2
        },
        save_individual_results=True,
        generate_summary=True
    )

    print(f"Batch processing complete!")
    print(f"Processed {results['successful']} images successfully")
    print(f"Results saved to: {output_directory}")


def demonstrate_interactive_viewer():
    """Example of using the interactive viewer"""

    from grainstat.visualization.interactive import InteractiveViewer

    image_path = "sample_microstructure.tif"

    if not os.path.exists(image_path):
        print("Interactive demo skipped: sample image not found")
        return

    # Analyze image first
    analyzer = GrainAnalyzer()
    analyzer.analyze(image_path, scale=0.5)

    # Launch interactive viewer
    viewer = InteractiveViewer(
        analyzer.image,
        analyzer.labeled_image,
        analyzer.grain_metrics
    )

    print("Launching interactive grain viewer...")
    print("Click on grains to see their properties!")
    viewer.show_interactive()


if __name__ == "__main__":
    main()

    # Uncomment to try other features:
    # demonstrate_batch_processing()
    # demonstrate_interactive_viewer()