import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns


class PlotGenerator:

    def __init__(self, style: str = 'seaborn-v0_8'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_histogram(self, grain_metrics: Dict[int, Dict[str, Any]],
                       bins: int = 30, save_path: Optional[str] = None,
                       metric: str = 'ecd_um') -> plt.Figure:

        values = [grain[metric] for grain in grain_metrics.values()
                  if np.isfinite(grain[metric])]

        if not values:
            raise ValueError(f"No finite values found for metric: {metric}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(values, bins=bins, alpha=0.7, color=self.colors[0],
                 edgecolor='black', linewidth=0.5)
        ax1.set_xlabel(self._get_metric_label(metric))
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of {self._get_metric_label(metric)}')
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax1.legend()

        # Q-Q plot
        stats.probplot(values, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot: {self._get_metric_label(metric)}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cumulative_distribution(self, grain_metrics: Dict[int, Dict[str, Any]],
                                     save_path: Optional[str] = None,
                                     metric: str = 'ecd_um') -> plt.Figure:

        values = [grain[metric] for grain in grain_metrics.values()
                  if np.isfinite(grain[metric])]

        if not values:
            raise ValueError(f"No finite values found for metric: {metric}")

        values_sorted = np.sort(values)
        cumulative = np.arange(1, len(values_sorted) + 1) / len(values_sorted)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(values_sorted, cumulative, linewidth=2, color=self.colors[1])
        ax.set_xlabel(self._get_metric_label(metric))
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Cumulative Distribution: {self._get_metric_label(metric)}')
        ax.grid(True, alpha=0.3)

        # Add percentile lines
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            value = np.percentile(values, p)
            ax.axvline(value, color='red', linestyle=':', alpha=0.7,
                       label=f'P{p}: {value:.2f}')

        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_overlay(self, original_image: np.ndarray, labeled_image: np.ndarray,
                     grain_properties: Dict[int, Dict[str, Any]],
                     save_path: Optional[str] = None,
                     show_centroids: bool = True, show_labels: bool = False) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(10, 8))

        # Show original image
        ax.imshow(original_image, cmap='gray', alpha=0.8)

        # Create colormap for grain boundaries
        unique_labels = np.unique(labeled_image)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background

        # Create overlay with grain boundaries
        boundary_image = np.zeros_like(labeled_image, dtype=float)
        from skimage import morphology

        for label in unique_labels:
            grain_mask = labeled_image == label
            boundary = grain_mask ^ morphology.erosion(grain_mask)
            boundary_image[boundary] = label

        # Create custom colormap
        cmap = ListedColormap(['none'] + [self.colors[i % len(self.colors)]
                                          for i in range(len(unique_labels))])

        ax.imshow(boundary_image, cmap=cmap, alpha=0.6)

        # Add centroids and labels if requested
        if show_centroids or show_labels:
            for grain_id, props in grain_properties.items():
                centroid = props['centroid_px']

                if show_centroids:
                    ax.plot(centroid[1], centroid[0], 'r+', markersize=8, markeredgewidth=2)

                if show_labels:
                    ax.text(centroid[1], centroid[0], str(grain_id),
                            color='white', fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))

        ax.set_title(f'Grain Overlay (n={len(grain_properties)} grains)')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_scatter_matrix(self, grain_metrics: Dict[int, Dict[str, Any]],
                            metrics: List[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:

        if metrics is None:
            metrics = ['ecd_um', 'aspect_ratio', 'shape_factor', 'area_um2']

        # Prepare data
        data = {}
        for metric in metrics:
            values = [grain[metric] for grain in grain_metrics.values()
                      if np.isfinite(grain[metric])]
            data[self._get_metric_label(metric)] = values

        # Create DataFrame for easier plotting
        import pandas as pd
        df = pd.DataFrame(data)

        # Create scatter matrix
        fig = plt.figure(figsize=(12, 10))
        axes = pd.plotting.scatter_matrix(df, alpha=0.6, figsize=(12, 10),
                                          diagonal='hist', color=self.colors[0])

        plt.suptitle('Grain Metrics Correlation Matrix', y=0.95)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_size_distribution_comparison(self, grain_metrics: Dict[int, Dict[str, Any]],
                                          save_path: Optional[str] = None) -> plt.Figure:

        ecds = [grain['ecd_um'] for grain in grain_metrics.values()
                if np.isfinite(grain['ecd_um'])]

        if not ecds:
            raise ValueError("No finite ECD values found")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Linear scale histogram
        ax1.hist(ecds, bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax1.set_xlabel('ECD (μm)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Size Distribution (Linear Scale)')
        ax1.grid(True, alpha=0.3)

        # Log scale histogram
        ax2.hist(ecds, bins=np.logspace(np.log10(min(ecds)), np.log10(max(ecds)), 30),
                 alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.set_xlabel('ECD (μm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Size Distribution (Log Scale)')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        # Box plot
        ax3.boxplot(ecds, vert=True)
        ax3.set_ylabel('ECD (μm)')
        ax3.set_title('Size Distribution Box Plot')
        ax3.grid(True, alpha=0.3)

        # Violin plot
        ax4.violinplot(ecds, vert=True)
        ax4.set_ylabel('ECD (μm)')
        ax4.set_title('Size Distribution Violin Plot')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_shape_analysis(self, grain_metrics: Dict[int, Dict[str, Any]],
                            save_path: Optional[str] = None) -> plt.Figure:

        aspect_ratios = [grain['aspect_ratio'] for grain in grain_metrics.values()
                         if np.isfinite(grain['aspect_ratio'])]
        shape_factors = [grain['shape_factor'] for grain in grain_metrics.values()
                         if np.isfinite(grain['shape_factor'])]
        ecds = [grain['ecd_um'] for grain in grain_metrics.values()
                if np.isfinite(grain['ecd_um'])]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Aspect ratio vs shape factor
        ax1.scatter(aspect_ratios, shape_factors, alpha=0.6, color=self.colors[0])
        ax1.set_xlabel('Aspect Ratio')
        ax1.set_ylabel('Shape Factor')
        ax1.set_title('Shape Factor vs Aspect Ratio')
        ax1.grid(True, alpha=0.3)

        # ECD vs aspect ratio
        ax2.scatter(ecds, aspect_ratios, alpha=0.6, color=self.colors[1])
        ax2.set_xlabel('ECD (μm)')
        ax2.set_ylabel('Aspect Ratio')
        ax2.set_title('Aspect Ratio vs Size')
        ax2.grid(True, alpha=0.3)

        # Shape factor distribution
        ax3.hist(shape_factors, bins=30, alpha=0.7, color=self.colors[2], edgecolor='black')
        ax3.set_xlabel('Shape Factor')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Shape Factor Distribution')
        ax3.grid(True, alpha=0.3)

        # Aspect ratio distribution
        ax4.hist(aspect_ratios, bins=30, alpha=0.7, color=self.colors[3], edgecolor='black')
        ax4.set_xlabel('Aspect Ratio')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Aspect Ratio Distribution')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _get_metric_label(self, metric: str) -> str:
        labels = {
            'ecd_um': 'ECD (μm)',
            'area_um2': 'Area (μm²)',
            'aspect_ratio': 'Aspect Ratio',
            'shape_factor': 'Shape Factor',
            'perimeter_um': 'Perimeter (μm)',
            'major_axis_um': 'Major Axis (μm)',
            'minor_axis_um': 'Minor Axis (μm)',
            'roundness': 'Roundness',
            'compactness': 'Compactness'
        }
        return labels.get(metric, metric.replace('_', ' ').title())

    def create_multi_panel_summary(self, grain_metrics: Dict[int, Dict[str, Any]],
                                   statistics: Dict[str, Any],
                                   save_path: Optional[str] = None) -> plt.Figure:

        fig = plt.figure(figsize=(16, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Size distribution histogram
        ax1 = fig.add_subplot(gs[0, :2])
        ecds = [grain['ecd_um'] for grain in grain_metrics.values()
                if np.isfinite(grain['ecd_um'])]
        ax1.hist(ecds, bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax1.set_xlabel('ECD (μm)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Grain Size Distribution')
        ax1.grid(True, alpha=0.3)

        # Cumulative distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        values_sorted = np.sort(ecds)
        cumulative = np.arange(1, len(values_sorted) + 1) / len(values_sorted)
        ax2.plot(values_sorted, cumulative, linewidth=2, color=self.colors[1])
        ax2.set_xlabel('ECD (μm)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Size Distribution')
        ax2.grid(True, alpha=0.3)

        # Shape analysis
        ax3 = fig.add_subplot(gs[1, :2])
        aspect_ratios = [grain['aspect_ratio'] for grain in grain_metrics.values()
                         if np.isfinite(grain['aspect_ratio'])]
        shape_factors = [grain['shape_factor'] for grain in grain_metrics.values()
                         if np.isfinite(grain['shape_factor'])]
        ax3.scatter(aspect_ratios, shape_factors, alpha=0.6, color=self.colors[2])
        ax3.set_xlabel('Aspect Ratio')
        ax3.set_ylabel('Shape Factor')
        ax3.set_title('Shape Analysis')
        ax3.grid(True, alpha=0.3)

        # Statistics table
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.axis('off')

        # Create statistics text
        stats_text = self._format_statistics_text(statistics)
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        # Size class distribution
        ax5 = fig.add_subplot(gs[2, :])
        if 'size_class_distribution' in statistics:
            size_classes = statistics['size_class_distribution']
            classes = list(size_classes.keys())
            counts = list(size_classes.values())

            bars = ax5.bar(classes, counts, color=self.colors[:len(classes)])
            ax5.set_xlabel('Size Class')
            ax5.set_ylabel('Count')
            ax5.set_title('Grain Size Classification')

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                             str(count), ha='center', va='bottom')

        plt.suptitle(f'Grain Analysis Summary (n={len(grain_metrics)} grains)',
                     fontsize=16, y=0.95)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _format_statistics_text(self, statistics: Dict[str, Any]) -> str:
        lines = ['GRAIN STATISTICS', '=' * 20, '']

        if 'grain_count' in statistics:
            lines.append(f"Total Grains: {statistics['grain_count']}")

        if 'ecd_statistics' in statistics:
            ecd_stats = statistics['ecd_statistics']
            lines.extend([
                '',
                'ECD Statistics (μm):',
                f"  Mean:   {ecd_stats.get('mean', 0):.2f}",
                f"  Median: {ecd_stats.get('median', 0):.2f}",
                f"  Std:    {ecd_stats.get('std', 0):.2f}",
                f"  Range:  {ecd_stats.get('min', 0):.2f} - {ecd_stats.get('max', 0):.2f}",
            ])

        if 'astm_grain_size' in statistics:
            astm = statistics['astm_grain_size']
            if 'grain_size_number' in astm:
                lines.extend([
                    '',
                    f"ASTM Grain Size: {astm['grain_size_number']:.1f}"
                ])

        return '\n'.join(lines)