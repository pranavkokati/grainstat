import numpy as np
from scipy import ndimage
from skimage import filters, segmentation, measure, morphology
from skimage.feature import peak_local_maxima
from typing import Tuple


class GrainSegmenter:

    def __init__(self):
        pass

    def otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        threshold_value = filters.threshold_otsu(image)
        return image > threshold_value

    def adaptive_threshold(self, image: np.ndarray, block_size: int = 35,
                           offset: float = 0.01) -> np.ndarray:
        threshold_value = filters.threshold_local(
            image, block_size=block_size, offset=offset
        )
        return image > threshold_value

    def multi_otsu_threshold(self, image: np.ndarray, classes: int = 3) -> np.ndarray:
        thresholds = filters.threshold_multiotsu(image, classes=classes)
        # Return binary mask for the brightest class (assuming grains are bright)
        return image > thresholds[-1]

    def watershed_segmentation(self, binary_image: np.ndarray) -> np.ndarray:
        # Distance transform
        distance = ndimage.distance_transform_edt(binary_image)

        # Find local maxima as markers
        local_maxima = peak_local_maxima(
            distance, min_distance=5, threshold_abs=0.3 * distance.max()
        )

        # Create markers array
        markers = np.zeros_like(distance, dtype=int)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1

        # Watershed segmentation
        labels = segmentation.watershed(-distance, markers, mask=binary_image)

        return labels

    def simple_labeling(self, binary_image: np.ndarray) -> np.ndarray:
        return measure.label(binary_image, connectivity=2)

    def random_walker_segmentation(self, image: np.ndarray,
                                   markers: np.ndarray) -> np.ndarray:
        return segmentation.random_walker(image, markers)

    def felzenszwalb_segmentation(self, image: np.ndarray, scale: float = 100,
                                  sigma: float = 0.5, min_size: int = 50) -> np.ndarray:
        return segmentation.felzenszwalb(
            image, scale=scale, sigma=sigma, min_size=min_size
        )

    def grow_regions(self, binary_image: np.ndarray, seed_points: np.ndarray,
                     iterations: int = 10) -> np.ndarray:
        # Simple region growing from seed points
        labels = np.zeros_like(binary_image, dtype=int)

        # Initialize with seed points
        for i, (y, x) in enumerate(seed_points):
            if binary_image[y, x]:
                labels[y, x] = i + 1

        # Grow regions iteratively
        for _ in range(iterations):
            dilated = morphology.dilation(labels > 0)
            labels = np.where(dilated & binary_image & (labels == 0),
                              morphology.dilation(labels), labels)

        return labels

    def calculate_optimal_threshold(self, image: np.ndarray,
                                    method: str = 'otsu') -> float:
        if method == 'otsu':
            return filters.threshold_otsu(image)
        elif method == 'li':
            return filters.threshold_li(image)
        elif method == 'triangle':
            return filters.threshold_triangle(image)
        elif method == 'yen':
            return filters.threshold_yen(image)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

    def edge_based_segmentation(self, image: np.ndarray) -> np.ndarray:
        # Use Canny edge detection followed by morphological operations
        edges = filters.sobel(image)
        threshold = filters.threshold_otsu(edges)
        binary_edges = edges > threshold

        # Fill holes and clean up
        filled = ndimage.binary_fill_holes(binary_edges)
        return morphology.remove_small_objects(filled, min_size=50)