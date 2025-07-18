import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from .main import GrainAnalyzer
from .processing.batch import BatchProcessor
from .visualization.interactive import InteractiveViewer


def main():
    parser = argparse.ArgumentParser(
        description='GrainStat: Professional grain size analysis for materials science',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image analysis
  grainstat analyze image.tif --scale 0.2 --export-csv results.csv

  # Batch processing
  grainstat batch input_folder/ output_folder/ --scale 0.2 --workers 4

  # Interactive viewer
  grainstat interactive image.tif --scale 0.2

  # Generate report
  grainstat analyze image.tif --scale 0.2 --report report.html
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Single image analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single image')
    analyze_parser.add_argument('image', help='Path to image file')
    analyze_parser.add_argument('--scale', type=float, default=1.0,
                                help='Scale factor (micrometers per pixel)')
    analyze_parser.add_argument('--min-area', type=int, default=50,
                                help='Minimum grain area in pixels')
    analyze_parser.add_argument('--gaussian-sigma', type=float, default=1.0,
                                help='Gaussian smoothing sigma')
    analyze_parser.add_argument('--threshold-method', choices=['otsu', 'adaptive'],
                                default='otsu', help='Thresholding method')
    analyze_parser.add_argument('--no-watershed', action='store_true',
                                help='Disable watershed segmentation')
    analyze_parser.add_argument('--morphology-radius', type=int, default=2,
                                help='Morphological operations radius')
    analyze_parser.add_argument('--export-csv', type=str,
                                help='Export grain data to CSV file')
    analyze_parser.add_argument('--export-json', type=str,
                                help='Export analysis results to JSON file')
    analyze_parser.add_argument('--report', type=str,
                                help='Generate HTML report')
    analyze_parser.add_argument('--plot-histogram', type=str,
                                help='Save histogram plot to file')
    analyze_parser.add_argument('--plot-overlay', type=str,
                                help='Save overlay plot to file')
    analyze_parser.add_argument('--plot-cumulative', type=str,
                                help='Save cumulative distribution plot to file')

    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('input_dir', help='Input directory containing images')
    batch_parser.add_argument('output_dir', help='Output directory for results')
    batch_parser.add_argument('--pattern', default='*.tif*',
                              help='File pattern to match (default: *.tif*)')
    batch_parser.add_argument('--scale', type=float, default=1.0,
                              help='Scale factor (micrometers per pixel)')
    batch_parser.add_argument('--min-area', type=int, default=50,
                              help='Minimum grain area in pixels')
    batch_parser.add_argument('--gaussian-sigma', type=float, default=1.0,
                              help='Gaussian smoothing sigma')
    batch_parser.add_argument('--threshold-method', choices=['otsu', 'adaptive'],
                              default='otsu', help='Thresholding method')
    batch_parser.add_argument('--no-watershed', action='store_true',
                              help='Disable watershed segmentation')
    batch_parser.add_argument('--morphology-radius', type=int, default=2,
                              help='Morphological operations radius')
    batch_parser.add_argument('--workers', type=int, default=None,
                              help='Number of parallel workers')
    batch_parser.add_argument('--no-individual-results', action='store_true',
                              help='Skip saving individual image results')
    batch_parser.add_argument('--no-summary', action='store_true',
                              help='Skip generating batch summary')

    # Interactive viewer command
    interactive_parser = subparsers.add_parser('interactive',
                                               help='Launch interactive grain viewer')
    interactive_parser.add_argument('image', help='Path to image file')
    interactive_parser.add_argument('--scale', type=float, default=1.0,
                                    help='Scale factor (micrometers per pixel)')
    interactive_parser.add_argument('--min-area', type=int, default=50,
                                    help='Minimum grain area in pixels')
    interactive_parser.add_argument('--gaussian-sigma', type=float, default=1.0,
                                    help='Gaussian smoothing sigma')
    interactive_parser.add_argument('--threshold-method', choices=['otsu', 'adaptive'],
                                    default='otsu', help='Thresholding method')
    interactive_parser.add_argument('--no-watershed', action='store_true',
                                    help='Disable watershed segmentation')
    interactive_parser.add_argument('--morphology-radius', type=int, default=2,
                                    help='Morphological operations radius')

    # Compare conditions command
    compare_parser = subparsers.add_parser('compare',
                                           help='Compare grain analysis between conditions')
    compare_parser.add_argument('--conditions', nargs='+', required=True,
                                help='Condition directories in format name:path')
    compare_parser.add_argument('output_dir', help='Output directory for comparison')
    compare_parser.add_argument('--scale', type=float, default=1.0,
                                help='Scale factor (micrometers per pixel)')

    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'analyze':
            run_analyze(args)
        elif args.command == 'batch':
            run_batch(args)
        elif args.command == 'interactive':
            run_interactive(args)
        elif args.command == 'compare':
            run_compare(args)
        elif args.command == 'version':
            run_version()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_analyze(args):
    print(f"Analyzing image: {args.image}")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    analyzer = GrainAnalyzer()

    # Set up analysis parameters
    analysis_params = {
        'scale': args.scale,
        'min_area': args.min_area,
        'gaussian_sigma': args.gaussian_sigma,
        'threshold_method': args.threshold_method,
        'use_watershed': not args.no_watershed,
        'morphology_radius': args.morphology_radius
    }

    # Run analysis
    results = analyzer.analyze(args.image, **analysis_params)

    # Print summary
    print_analysis_summary(results)

    # Export data
    if args.export_csv:
        analyzer.export_csv(args.export_csv)
        print(f"Grain data exported to: {args.export_csv}")

    if args.export_json:
        analyzer.export_json(args.export_json)
        print(f"Analysis results exported to: {args.export_json}")

    if args.report:
        analyzer.generate_report(args.report, 'html')
        print(f"HTML report generated: {args.report}")

    # Generate plots
    if args.plot_histogram:
        analyzer.plot_histogram(save_path=args.plot_histogram)
        print(f"Histogram saved to: {args.plot_histogram}")

    if args.plot_overlay:
        analyzer.plot_overlay(save_path=args.plot_overlay)
        print(f"Overlay plot saved to: {args.plot_overlay}")

    if args.plot_cumulative:
        analyzer.plot_cumulative_distribution(save_path=args.plot_cumulative)
        print(f"Cumulative distribution saved to: {args.plot_cumulative}")


def run_batch(args):
    print(f"Batch processing: {args.input_dir} -> {args.output_dir}")

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    processor = BatchProcessor(n_workers=args.workers)

    # Set up analysis parameters
    analysis_params = {
        'min_area': args.min_area,
        'gaussian_sigma': args.gaussian_sigma,
        'threshold_method': args.threshold_method,
        'use_watershed': not args.no_watershed,
        'morphology_radius': args.morphology_radius
    }

    # Run batch processing
    results = processor.process_directory(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        scale=args.scale,
        analysis_params=analysis_params,
        save_individual_results=not args.no_individual_results,
        generate_summary=not args.no_summary
    )

    # Print batch summary
    print_batch_summary(results)


def run_interactive(args):
    print(f"Launching interactive viewer for: {args.image}")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    analyzer = GrainAnalyzer()

    # Set up analysis parameters
    analysis_params = {
        'scale': args.scale,
        'min_area': args.min_area,
        'gaussian_sigma': args.gaussian_sigma,
        'threshold_method': args.threshold_method,
        'use_watershed': not args.no_watershed,
        'morphology_radius': args.morphology_radius
    }

    # Run analysis
    print("Analyzing image...")
    results = analyzer.analyze(args.image, **analysis_params)

    print_analysis_summary(results)

    # Launch interactive viewer
    viewer = InteractiveViewer(
        analyzer.image,
        analyzer.labeled_image,
        analyzer.grain_metrics
    )

    print("Launching interactive viewer...")
    viewer.show_interactive()


def run_compare(args):
    print("Comparing conditions...")

    # Parse condition arguments
    conditions = {}
    for condition in args.conditions:
        if ':' not in condition:
            raise ValueError(f"Condition must be in format name:path, got: {condition}")

        name, path = condition.split(':', 1)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Condition directory not found: {path}")

        conditions[name] = path

    processor = BatchProcessor()

    results = processor.compare_conditions(
        conditions,
        args.output_dir,
        scale=args.scale
    )

    print(f"Comparison results saved to: {args.output_dir}")
    print(f"Processed {len(conditions)} conditions:")
    for name in conditions.keys():
        print(f"  - {name}")


def run_version():
    from . import __version__, __author__
    print(f"GrainStat version {__version__}")
    print(f"Author: {__author__}")
    print("Professional grain size analysis for materials science")


def print_analysis_summary(results: Dict[str, Any]):
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)

    if 'statistics' in results:
        stats = results['statistics']

        if 'grain_count' in stats:
            print(f"Total grains detected: {stats['grain_count']}")

        if 'ecd_statistics' in stats:
            ecd_stats = stats['ecd_statistics']
            print(f"\nGrain Size Statistics (ECD):")
            print(f"  Mean:   {ecd_stats.get('mean', 0):.2f} μm")
            print(f"  Median: {ecd_stats.get('median', 0):.2f} μm")
            print(f"  Std:    {ecd_stats.get('std', 0):.2f} μm")
            print(f"  Range:  {ecd_stats.get('min', 0):.2f} - {ecd_stats.get('max', 0):.2f} μm")

        if 'astm_grain_size' in stats:
            astm = stats['astm_grain_size']
            if 'grain_size_number' in astm and astm['grain_size_number'] is not None:
                print(f"\nASTM Grain Size Number: {astm['grain_size_number']:.1f}")

        if 'size_class_distribution' in stats:
            size_dist = stats['size_class_distribution']
            print(f"\nSize Class Distribution:")
            for size_class, count in size_dist.items():
                if count > 0:
                    print(f"  {size_class}: {count} grains")

    print("=" * 50)


def print_batch_summary(results: Dict[str, Any]):
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)

    print(f"Total images: {results['total_images']}")
    print(f"Successfully processed: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success_rate']:.1%}")

    if 'results' in results and results['results']:
        # Calculate some aggregate statistics
        all_grain_counts = []
        all_mean_ecds = []

        for result in results['results']:
            if 'statistics' in result:
                stats = result['statistics']
                if 'grain_count' in stats:
                    all_grain_counts.append(stats['grain_count'])
                if 'ecd_statistics' in stats and 'mean' in stats['ecd_statistics']:
                    all_mean_ecds.append(stats['ecd_statistics']['mean'])

        if all_grain_counts:
            import numpy as np
            print(f"\nAcross all images:")
            print(f"  Total grains: {sum(all_grain_counts)}")
            print(f"  Mean grains per image: {np.mean(all_grain_counts):.1f}")

        if all_mean_ecds:
            print(f"  Average ECD across images: {np.mean(all_mean_ecds):.2f} μm")

    print("=" * 50)


# Entry point for setuptools
def cli_main():
    main()


if __name__ == '__main__':
    main()