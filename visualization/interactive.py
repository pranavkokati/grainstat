import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from typing import Dict, Any, Optional
from skimage import measure


class InteractiveViewer:

    def __init__(self, original_image: np.ndarray, labeled_image: np.ndarray,
                 grain_metrics: Dict[int, Dict[str, Any]]):
        self.original_image = original_image
        self.labeled_image = labeled_image
        self.grain_metrics = grain_metrics

        self.fig = None
        self.ax = None
        self.cursor = None
        self.selected_grain = None

    def show_interactive(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # Display original image
        self.ax.imshow(self.original_image, cmap='gray', alpha=0.8)

        # Overlay grain boundaries
        self._overlay_boundaries()

        # Set up interactive features
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Add instructions
        self.ax.set_title('Interactive Grain Viewer\nClick on grain to view properties, Press "h" for help')

        # Create text area for grain info
        self.info_text = self.fig.text(0.02, 0.02, '', fontsize=10,
                                       verticalalignment='bottom',
                                       bbox=dict(boxstyle="round,pad=0.5",
                                                 facecolor='white', alpha=0.8))

        plt.show()

    def _overlay_boundaries(self):
        # Create boundary overlay
        from skimage import morphology
        from matplotlib.colors import ListedColormap

        boundary_image = np.zeros_like(self.labeled_image, dtype=float)
        unique_labels = np.unique(self.labeled_image)
        unique_labels = unique_labels[unique_labels != 0]

        for label in unique_labels:
            grain_mask = self.labeled_image == label
            boundary = grain_mask ^ morphology.erosion(grain_mask)
            boundary_image[boundary] = label

        # Create colormap
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        cmap = ListedColormap(['none'] + [colors[i % len(colors)]
                                          for i in range(len(unique_labels))])

        self.ax.imshow(boundary_image, cmap=cmap, alpha=0.5)

        # Add centroids
        for grain_id, metrics in self.grain_metrics.items():
            centroid = metrics['centroid_px']
            self.ax.plot(centroid[1], centroid[0], 'r+', markersize=6, alpha=0.7)

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Get clicked coordinates
        x, y = int(event.xdata), int(event.ydata)

        # Find which grain was clicked
        if (0 <= y < self.labeled_image.shape[0] and
                0 <= x < self.labeled_image.shape[1]):

            grain_id = self.labeled_image[y, x]

            if grain_id > 0 and grain_id in self.grain_metrics:
                self.selected_grain = grain_id
                self._display_grain_info(grain_id)
                self._highlight_grain(grain_id)

    def _on_key_press(self, event):
        if event.key == 'h':
            self._show_help()
        elif event.key == 'c':
            self._clear_highlights()
        elif event.key == 's' and self.selected_grain:
            self._save_grain_info()
        elif event.key == 'a':
            self._show_all_grain_info()

    def _display_grain_info(self, grain_id: int):
        metrics = self.grain_metrics[grain_id]

        info_lines = [
            f"GRAIN {grain_id}",
            "=" * 15,
            f"ECD: {metrics['ecd_um']:.2f} μm",
            f"Area: {metrics['area_um2']:.2f} μm²",
            f"Aspect Ratio: {metrics['aspect_ratio']:.2f}",
            f"Shape Factor: {metrics['shape_factor']:.3f}",
            f"Perimeter: {metrics['perimeter_um']:.2f} μm",
            f"Eccentricity: {metrics['eccentricity']:.3f}",
            f"Solidity: {metrics['solidity']:.3f}",
            "",
            "Press 'h' for help",
            "Press 'c' to clear",
            "Press 's' to save info"
        ]

        self.info_text.set_text('\n'.join(info_lines))
        self.fig.canvas.draw()

    def _highlight_grain(self, grain_id: int):
        # Clear previous highlights
        self._clear_highlights()

        # Highlight selected grain
        grain_mask = self.labeled_image == grain_id

        # Create highlight overlay
        highlight = np.zeros_like(self.labeled_image, dtype=float)
        highlight[grain_mask] = 1

        self.highlight_overlay = self.ax.imshow(highlight, cmap='Reds', alpha=0.4)

        # Add grain boundary
        from skimage import morphology
        boundary = grain_mask ^ morphology.erosion(grain_mask)
        boundary_coords = np.where(boundary)
        self.ax.plot(boundary_coords[1], boundary_coords[0], 'r-', linewidth=2)

        self.fig.canvas.draw()

    def _clear_highlights(self):
        # Remove highlight overlays
        for child in self.ax.get_children():
            if hasattr(child, 'get_cmap') and child.get_cmap().name == 'Reds':
                child.remove()

        # Remove boundary lines
        lines_to_remove = []
        for child in self.ax.get_children():
            if hasattr(child, 'get_color') and child.get_color() == 'r':
                lines_to_remove.append(child)

        for line in lines_to_remove:
            line.remove()

        self.fig.canvas.draw()

    def _show_help(self):
        help_text = """
        INTERACTIVE GRAIN VIEWER HELP
        =============================

        Mouse Controls:
        - Click on a grain to select and view properties

        Keyboard Controls:
        - 'h': Show this help
        - 'c': Clear all highlights
        - 's': Save selected grain info to file
        - 'a': Show summary of all grains

        The red crosshairs show grain centroids.
        Click anywhere on a grain to see its properties.
        """

        # Create help window
        help_fig, help_ax = plt.subplots(figsize=(8, 6))
        help_ax.text(0.05, 0.95, help_text, transform=help_ax.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace')
        help_ax.axis('off')
        help_ax.set_title('Help - Interactive Grain Viewer')
        plt.show()

    def _save_grain_info(self):
        if not self.selected_grain:
            print("No grain selected")
            return

        metrics = self.grain_metrics[self.selected_grain]
        filename = f"grain_{self.selected_grain}_info.txt"

        with open(filename, 'w') as f:
            f.write(f"Grain {self.selected_grain} Analysis Results\n")
            f.write("=" * 40 + "\n\n")

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'um' in key:
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value:.6f}\n")
                elif isinstance(value, tuple) and len(value) == 2:
                    f.write(f"{key}: ({value[0]:.2f}, {value[1]:.2f})\n")

        print(f"Grain info saved to {filename}")

    def _show_all_grain_info(self):
        # Create summary window
        summary_fig, summary_ax = plt.subplots(figsize=(10, 8))

        # Calculate summary statistics
        ecds = [grain['ecd_um'] for grain in self.grain_metrics.values()]
        areas = [grain['area_um2'] for grain in self.grain_metrics.values()]
        aspect_ratios = [grain['aspect_ratio'] for grain in self.grain_metrics.values()]

        summary_text = f"""
        GRAIN ANALYSIS SUMMARY
        =====================

        Total Grains: {len(self.grain_metrics)}

        ECD Statistics (μm):
          Mean:   {np.mean(ecds):.2f}
          Median: {np.median(ecds):.2f}
          Std:    {np.std(ecds):.2f}
          Min:    {np.min(ecds):.2f}
          Max:    {np.max(ecds):.2f}

        Area Statistics (μm²):
          Mean:   {np.mean(areas):.2f}
          Median: {np.median(areas):.2f}
          Total:  {np.sum(areas):.2f}

        Aspect Ratio Statistics:
          Mean:   {np.mean(aspect_ratios):.2f}
          Median: {np.median(aspect_ratios):.2f}

        Top 5 Largest Grains (by ECD):
        """

        # Add top 5 largest grains
        sorted_grains = sorted(self.grain_metrics.items(),
                               key=lambda x: x[1]['ecd_um'], reverse=True)

        for i, (grain_id, metrics) in enumerate(sorted_grains[:5]):
            summary_text += f"  {i + 1}. Grain {grain_id}: {metrics['ecd_um']:.2f} μm\n"

        summary_ax.text(0.05, 0.95, summary_text, transform=summary_ax.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        summary_ax.axis('off')
        summary_ax.set_title('All Grains Summary')
        plt.show()


class GrainComparisonViewer:

    def __init__(self, image1: np.ndarray, image2: np.ndarray,
                 metrics1: Dict[int, Dict[str, Any]],
                 metrics2: Dict[int, Dict[str, Any]],
                 labels: tuple = ("Image 1", "Image 2")):
        self.image1 = image1
        self.image2 = image2
        self.metrics1 = metrics1
        self.metrics2 = metrics2
        self.labels = labels

    def show_comparison(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Show images
        ax1.imshow(self.image1, cmap='gray')
        ax1.set_title(f'{self.labels[0]} (n={len(self.metrics1)} grains)')
        ax1.axis('off')

        ax2.imshow(self.image2, cmap='gray')
        ax2.set_title(f'{self.labels[1]} (n={len(self.metrics2)} grains)')
        ax2.axis('off')

        # Compare size distributions
        ecds1 = [grain['ecd_um'] for grain in self.metrics1.values()]
        ecds2 = [grain['ecd_um'] for grain in self.metrics2.values()]

        ax3.hist(ecds1, bins=30, alpha=0.7, label=self.labels[0], color='blue')
        ax3.hist(ecds2, bins=30, alpha=0.7, label=self.labels[1], color='red')
        ax3.set_xlabel('ECD (μm)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Size Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Compare statistics
        stats_text = self._create_comparison_text(ecds1, ecds2)
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')
        ax4.set_title('Statistical Comparison')

        plt.tight_layout()
        plt.show()

    def _create_comparison_text(self, ecds1, ecds2):
        from scipy import stats

        # Statistical comparison
        t_stat, p_value = stats.ttest_ind(ecds1, ecds2)

        text = f"""
        STATISTICAL COMPARISON
        =====================

        {self.labels[0]}:
          Count: {len(ecds1)}
          Mean:  {np.mean(ecds1):.2f} μm
          Std:   {np.std(ecds1):.2f} μm

        {self.labels[1]}:
          Count: {len(ecds2)}
          Mean:  {np.mean(ecds2):.2f} μm
          Std:   {np.std(ecds2):.2f} μm

        T-test Results:
          t-statistic: {t_stat:.3f}
          p-value:     {p_value:.6f}

        Significance: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)
        """

        return text