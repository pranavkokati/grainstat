import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from pathlib import Path

from grainstat.main import GrainAnalyzer
from grainstat.core.image_io import ImageLoader
from grainstat.core.preprocessing import ImagePreprocessor
from grainstat.core.segmentation import GrainSegmenter
from grainstat.core.morphology import MorphologyProcessor
from grainstat.core.properties import PropertyCalculator
from grainstat.core.metrics import MetricsCalculator
from grainstat.core.statistics import StatisticsCalculator
from grainstat.plugins.base import PluginManager, feature
from grainstat.processing.batch import BatchProcessor


class TestImageLoader:

    def test_supported_formats(self):
        loader = ImageLoader()
        expected_formats = {'.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp'}
        assert loader.SUPPORTED_FORMATS == expected_formats

    def test_load_synthetic_image(self):
        # Create a synthetic test image
        test_image = np.random.rand(100, 100).astype(np.float64)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            from PIL import Image
            img_pil = Image.fromarray((test_image * 255).astype(np.uint8))
            img_pil.save(tmp_file.name)

            loader = ImageLoader()
            loaded_image = loader.load_image(tmp_file.name)

            assert loaded_image.shape == (100, 100)
            assert 0 <= loaded_image.min() <= loaded_image.max() <= 1

            os.unlink(tmp_file.name)

    def test_file_not_found(self):
        loader = ImageLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_image("nonexistent_file.tif")

    def test_unsupported_format(self):
        loader = ImageLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load_image("test.xyz")


class TestImagePreprocessor:

    def test_gaussian_smooth(self):
        preprocessor = ImagePreprocessor()
        test_image = np.random.rand(50, 50)

        smoothed = preprocessor.gaussian_smooth(test_image, sigma=1.0)

        assert smoothed.shape == test_image.shape
        assert np.all(smoothed >= 0)
        assert np.all(smoothed <= 1)

    def test_clahe_enhancement(self):
        preprocessor = ImagePreprocessor()
        test_image = np.random.rand(50, 50)

        enhanced = preprocessor.clahe_enhancement(test_image)

        assert enhanced.shape == test_image.shape
        assert np.all(enhanced >= 0)
        assert np.all(enhanced <= 1)

    def test_apply_filters(self):
        preprocessor = ImagePreprocessor()
        test_image = np.random.rand(50, 50)

        processed = preprocessor.apply_filters(test_image, sigma=1.0, use_clahe=True)

        assert processed.shape == test_image.shape


class TestGrainSegmenter:

    def test_otsu_threshold(self):
        segmenter = GrainSegmenter()
        # Create test image with clear bimodal distribution
        test_image = np.zeros((50, 50))
        test_image[:25, :] = 0.2  # Dark region
        test_image[25:, :] = 0.8  # Bright region

        binary = segmenter.otsu_threshold(test_image)

        assert binary.shape == test_image.shape
        assert binary.dtype == bool
        assert np.any(binary)  # Should have some True values

    def test_simple_labeling(self):
        segmenter = GrainSegmenter()
        # Create binary image with distinct regions
        binary_image = np.zeros((50, 50), dtype=bool)
        binary_image[10:20, 10:20] = True  # First region
        binary_image[30:40, 30:40] = True  # Second region

        labeled = segmenter.simple_labeling(binary_image)

        assert labeled.shape == binary_image.shape
        assert labeled.max() >= 2  # Should have at least 2 labels


class TestMorphologyProcessor:

    def test_clean_binary(self):
        processor = MorphologyProcessor()
        # Create noisy binary image
        binary_image = np.random.rand(50, 50) > 0.5

        cleaned = processor.clean_binary(binary_image, radius=1, min_area=10)

        assert cleaned.shape == binary_image.shape
        assert cleaned.dtype == bool

    def test_opening_closing(self):
        processor = MorphologyProcessor()
        binary_image = np.random.rand(50, 50) > 0.5

        opened = processor.opening(binary_image, radius=1)
        closed = processor.closing(binary_image, radius=1)

        assert opened.shape == binary_image.shape
        assert closed.shape == binary_image.shape


class TestPropertyCalculator:

    def test_calculate_properties(self):
        calculator = PropertyCalculator()

        # Create simple labeled image
        labeled_image = np.zeros((50, 50), dtype=int)
        labeled_image[10:20, 10:20] = 1  # 10x10 square
        labeled_image[30:35, 30:40] = 2  # 5x10 rectangle

        properties = calculator.calculate_properties(labeled_image, scale=0.5)

        assert len(properties) == 2
        assert 1 in properties
        assert 2 in properties

        # Check grain 1 (square)
        grain1 = properties[1]
        assert grain1['area_px'] == 100  # 10x10
        assert grain1['area_um2'] == 25  # 100 * 0.5^2

        # Check grain 2 (rectangle)
        grain2 = properties[2]
        assert grain2['area_px'] == 50  # 5x10
        assert grain2['area_um2'] == 12.5  # 50 * 0.5^2


class TestMetricsCalculator:

    def test_calculate_derived_metrics(self):
        calculator = MetricsCalculator()

        # Mock properties for a circular grain
        properties = {
            1: {
                'area_um2': 100.0,
                'perimeter_um': 35.45,  # Approximately 2π√(100/π)
                'major_axis_um': 11.28,
                'minor_axis_um': 11.28,
                'eccentricity': 0.0,
                'solidity': 1.0,
                'bbox_um': (0, 0, 11.28, 11.28),
                'convex_area_um2': 100.0
            }
        }

        metrics = calculator.calculate_derived_metrics(properties)

        assert 1 in metrics
        grain_metrics = metrics[1]

        # Check ECD calculation
        expected_ecd = 2 * np.sqrt(100 / np.pi)
        assert abs(grain_metrics['ecd_um'] - expected_ecd) < 0.1

        # Check aspect ratio
        assert grain_metrics['aspect_ratio'] == 1.0

        # Check shape factor (should be close to 1 for circle)
        assert 0.9 < grain_metrics['shape_factor'] < 1.1


class TestStatisticsCalculator:

    def test_calculate_statistics(self):
        calculator = StatisticsCalculator()

        # Mock grain metrics
        grain_metrics = {
            1: {'ecd_um': 10.0, 'area_um2': 78.5, 'aspect_ratio': 1.0, 'shape_factor': 1.0},
            2: {'ecd_um': 15.0, 'area_um2': 176.7, 'aspect_ratio': 1.2, 'shape_factor': 0.9},
            3: {'ecd_um': 8.0, 'area_um2': 50.3, 'aspect_ratio': 1.5, 'shape_factor': 0.8}
        }

        statistics = calculator.calculate_statistics(grain_metrics)

        assert 'grain_count' in statistics
        assert statistics['grain_count'] == 3

        assert 'ecd_statistics' in statistics
        ecd_stats = statistics['ecd_statistics']
        assert 'mean' in ecd_stats
        assert 'median' in ecd_stats
        assert 'std' in ecd_stats

        # Check mean calculation
        expected_mean = (10.0 + 15.0 + 8.0) / 3
        assert abs(ecd_stats['mean'] - expected_mean) < 0.01


class TestPluginSystem:

    def test_feature_decorator(self):
        @feature
        def test_feature(region):
            return region.area * 2

        # Check that feature was registered
        from grainstat.plugins.base import _plugin_manager
        assert 'test_feature' in _plugin_manager.registered_features

    def test_feature_with_name(self):
        @feature(name="custom_name")
        def another_feature(region):
            return region.perimeter / 2

        from grainstat.plugins.base import _plugin_manager
        assert 'custom_name' in _plugin_manager.registered_features

    def test_plugin_manager_apply_features(self):
        manager = PluginManager()

        # Register a simple feature
        def simple_feature(region):
            return region.area * 3

        manager.register_feature('simple_feature', simple_feature)

        # Mock grain properties
        properties = {
            1: {'area': 100, 'perimeter': 40}
        }

        custom_features = manager.apply_features(properties)

        assert 1 in custom_features
        assert 'simple_feature' in custom_features[1]
        assert custom_features[1]['simple_feature'] == 300


class TestGrainAnalyzer:

    @patch('grainstat.main.ImageLoader')
    @patch('grainstat.main.ImagePreprocessor')
    @patch('grainstat.main.GrainSegmenter')
    def test_analyze_workflow(self, mock_segmenter, mock_preprocessor, mock_loader):
        # Mock the components
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.load_image.return_value = np.random.rand(50, 50)

        mock_preprocessor_instance = Mock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_preprocessor_instance.apply_filters.return_value = np.random.rand(50, 50)

        mock_segmenter_instance = Mock()
        mock_segmenter.return_value = mock_segmenter_instance
        mock_segmenter_instance.otsu_threshold.return_value = np.random.rand(50, 50) > 0.5

        # Create labeled image with one grain
        labeled_image = np.zeros((50, 50), dtype=int)
        labeled_image[10:20, 10:20] = 1
        mock_segmenter_instance.simple_labeling.return_value = labeled_image

        analyzer = GrainAnalyzer()

        with tempfile.NamedTemporaryFile(suffix='.png') as tmp_file:
            results = analyzer.analyze(tmp_file.name, scale=0.5)

        assert 'properties' in results
        assert 'metrics' in results
        assert 'statistics' in results


class TestBatchProcessor:

    def test_find_image_files(self):
        processor = BatchProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            Path(tmp_dir, "image1.tif").touch()
            Path(tmp_dir, "image2.tiff").touch()
            Path(tmp_dir, "image3.png").touch()
            Path(tmp_dir, "not_image.txt").touch()

            files = processor._find_image_files(tmp_dir, "*.tif*")

            assert len(files) == 2
            assert any("image1.tif" in f for f in files)
            assert any("image2.tiff" in f for f in files)


class TestIntegration:

    def test_end_to_end_synthetic_image(self):
        # Create synthetic microstructure image
        image = np.zeros((100, 100))

        # Add circular grains
        y, x = np.ogrid[:100, :100]

        # Grain 1
        mask1 = (x - 30) ** 2 + (y - 30) ** 2 <= 10 ** 2
        image[mask1] = 1.0

        # Grain 2
        mask2 = (x - 70) ** 2 + (y - 70) ** 2 <= 8 ** 2
        image[mask2] = 1.0

        # Add some noise
        image += np.random.normal(0, 0.1, image.shape)
        image = np.clip(image, 0, 1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            img_pil.save(tmp_file.name)

            try:
                analyzer = GrainAnalyzer()
                results = analyzer.analyze(
                    tmp_file.name,
                    scale=1.0,
                    min_area=20,
                    gaussian_sigma=0.5
                )

                # Should detect approximately 2 grains
                assert 'statistics' in results
                grain_count = results['statistics'].get('grain_count', 0)
                assert 1 <= grain_count <= 4  # Allow some tolerance for segmentation

                # Check that basic metrics are calculated
                assert 'metrics' in results
                assert len(results['metrics']) >= 1

                # Check first grain has expected properties
                first_grain = list(results['metrics'].values())[0]
                assert 'ecd_um' in first_grain
                assert 'area_um2' in first_grain
                assert 'aspect_ratio' in first_grain

            finally:
                os.unlink(tmp_file.name)


def test_custom_feature_creation():
    """Test dynamic feature creation utilities"""
    from grainstat.plugins.base import create_ratio_feature, create_classification_feature

    # Test ratio feature
    ratio_func = create_ratio_feature('area_um2', 'perimeter_um', 'test_ratio')

    # Mock region
    class MockRegion:
        def __init__(self):
            self.area_um2 = 100
            self.perimeter_um = 20

    region = MockRegion()
    result = ratio_func(region)
    assert result == 5.0  # 100/20

    # Test classification feature
    classifier = create_classification_feature(
        'ecd_um',
        [10, 20],
        ['small', 'medium', 'large'],
        'size_category'
    )

    region.ecd_um = 15
    result = classifier(region)
    assert result == 'medium'


if __name__ == "__main__":
    pytest.main([__file__])