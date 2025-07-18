import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .core.image_io import ImageLoader
from .core.preprocessing import ImagePreprocessor
from .core.segmentation import GrainSegmenter
from .core.morphology import MorphologyProcessor
from .core.properties import PropertyCalculator
from .core.metrics import MetricsCalculator
from .core.statistics import StatisticsCalculator
from .visualization.plots import PlotGenerator
from .export.data_export import DataExporter
from .export.reports import ReportGenerator
from .plugins.base import PluginManager


class GrainAnalyzer:
    def __init__(self):
        self.loader = ImageLoader()
        self.preprocessor = ImagePreprocessor()
        self.segmenter = GrainSegmenter()
        self.morphology = MorphologyProcessor()
        self.properties = PropertyCalculator()
        self.metrics = MetricsCalculator()
        self.statistics = StatisticsCalculator()
        self.plotter = PlotGenerator()
        self.exporter = DataExporter()
        self.reporter = ReportGenerator()
        self.plugin_manager = PluginManager()

        self.image = None
        self.scale = 1.0
        self.binary_image = None
        self.labeled_image = None
        self.grain_properties = None
        self.grain_metrics = None
        self.stats = None

    def analyze(self, image_path: str, scale: float = 1.0,
                min_area: int = 50, gaussian_sigma: float = 1.0,
                threshold_method: str = 'otsu', use_watershed: bool = True,
                morphology_radius: int = 2) -> Dict[str, Any]:

        self.scale = scale

        # Load and preprocess image
        self.image = self.loader.load_image(image_path)
        processed_image = self.preprocessor.apply_filters(
            self.image, sigma=gaussian_sigma
        )

        # Segment grains
        if threshold_method == 'otsu':
            self.binary_image = self.segmenter.otsu_threshold(processed_image)
        else:
            self.binary_image = self.segmenter.adaptive_threshold(processed_image)

        # Apply morphological operations
        self.binary_image = self.morphology.clean_binary(
            self.binary_image, radius=morphology_radius, min_area=min_area
        )

        # Watershed segmentation if requested
        if use_watershed:
            self.labeled_image = self.segmenter.watershed_segmentation(self.binary_image)
        else:
            self.labeled_image = self.segmenter.simple_labeling(self.binary_image)

        # Calculate properties and metrics
        self.grain_properties = self.properties.calculate_properties(
            self.labeled_image, self.scale
        )
        self.grain_metrics = self.metrics.calculate_derived_metrics(
            self.grain_properties
        )

        # Apply custom features from plugins
        custom_features = self.plugin_manager.apply_features(self.grain_properties)
        if custom_features:
            for grain_id, features in custom_features.items():
                self.grain_metrics[grain_id].update(features)

        # Calculate statistics
        self.stats = self.statistics.calculate_statistics(self.grain_metrics)

        return {
            'properties': self.grain_properties,
            'metrics': self.grain_metrics,
            'statistics': self.stats
        }

    def plot_histogram(self, bins: int = 30, save_path: Optional[str] = None):
        if self.grain_metrics is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        return self.plotter.plot_histogram(self.grain_metrics, bins, save_path)

    def plot_cumulative_distribution(self, save_path: Optional[str] = None):
        if self.grain_metrics is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        return self.plotter.plot_cumulative_distribution(self.grain_metrics, save_path)

    def plot_overlay(self, save_path: Optional[str] = None):
        if self.image is None or self.labeled_image is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        return self.plotter.plot_overlay(
            self.image, self.labeled_image, self.grain_properties, save_path
        )

    def export_csv(self, filepath: str):
        if self.grain_metrics is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        self.exporter.export_csv(self.grain_metrics, filepath)

    def export_json(self, filepath: str):
        if self.grain_metrics is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        self.exporter.export_json(self.grain_metrics, self.stats, filepath)

    def generate_report(self, output_path: str, format_type: str = 'html'):
        if self.grain_metrics is None or self.stats is None:
            raise ValueError("No analysis results available. Run analyze() first.")

        return self.reporter.generate_report(
            self.grain_metrics, self.stats, output_path, format_type
        )