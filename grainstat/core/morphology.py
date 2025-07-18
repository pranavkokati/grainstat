import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from skimage.morphology import disk, square, diamond


class MorphologyProcessor:

    def __init__(self):
        pass

    def clean_binary(self, binary_image: np.ndarray, radius: int = 2,
                     min_area: int = 50) -> np.ndarray:
        # Apply opening and closing operations
        cleaned = self.opening(binary_image, radius)
        cleaned = self.closing(cleaned, radius)

        # Remove small objects
        cleaned = self.remove_small_objects(cleaned, min_area)

        # Fill holes
        cleaned = self.fill_holes(cleaned)

        return cleaned

    def opening(self, binary_image: np.ndarray, radius: int,
                shape: str = 'disk') -> np.ndarray:
        selem = self._get_structuring_element(radius, shape)
        return morphology.opening(binary_image, selem)

    def closing(self, binary_image: np.ndarray, radius: int,
                shape: str = 'disk') -> np.ndarray:
        selem = self._get_structuring_element(radius, shape)
        return morphology.closing(binary_image, selem)

    def erosion(self, binary_image: np.ndarray, radius: int,
                shape: str = 'disk') -> np.ndarray:
        selem = self._get_structuring_element(radius, shape)
        return morphology.erosion(binary_image, selem)

    def dilation(self, binary_image: np.ndarray, radius: int,
                 shape: str = 'disk') -> np.ndarray:
        selem = self._get_structuring_element(radius, shape)
        return morphology.dilation(binary_image, selem)

    def remove_small_objects(self, binary_image: np.ndarray,
                             min_area: int) -> np.ndarray:
        return morphology.remove_small_objects(binary_image, min_size=min_area)

    def remove_small_holes(self, binary_image: np.ndarray,
                           max_hole_size: int) -> np.ndarray:
        return morphology.remove_small_holes(binary_image, area_threshold=max_hole_size)

    def fill_holes(self, binary_image: np.ndarray) -> np.ndarray:
        return ndimage.binary_fill_holes(binary_image)

    def skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        return morphology.skeletonize(binary_image)

    def convex_hull(self, binary_image: np.ndarray) -> np.ndarray:
        return morphology.convex_hull_image(binary_image)

    def boundary_extraction(self, binary_image: np.ndarray) -> np.ndarray:
        # Extract object boundaries
        eroded = self.erosion(binary_image, 1)
        return binary_image ^ eroded

    def distance_transform(self, binary_image: np.ndarray) -> np.ndarray:
        return ndimage.distance_transform_edt(binary_image)

    def ultimate_erosion(self, binary_image: np.ndarray) -> np.ndarray:
        # Find ultimate eroded points (centers)
        distance = self.distance_transform(binary_image)
        local_maxima = morphology.h_maxima(distance, h=1)
        return local_maxima

    def separate_touching_objects(self, binary_image: np.ndarray) -> np.ndarray:
        # Use distance transform and watershed to separate touching objects
        distance = ndimage.distance_transform_edt(binary_image)

        # Find local maxima
        from skimage.feature import peak_local_maxima
        local_maxima = peak_local_maxima(distance, min_distance=5)

        # Create markers
        markers = np.zeros_like(binary_image, dtype=int)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1

        # Watershed
        from skimage.segmentation import watershed
        labels = watershed(-distance, markers, mask=binary_image)

        return labels > 0

    def _get_structuring_element(self, radius: int, shape: str):
        if shape == 'disk':
            return disk(radius)
        elif shape == 'square':
            return square(2 * radius + 1)
        elif shape == 'diamond':
            return diamond(radius)
        else:
            raise ValueError(f"Unknown structuring element shape: {shape}")

    def morphological_gradient(self, binary_image: np.ndarray,
                               radius: int = 1) -> np.ndarray:
        selem = disk(radius)
        dilated = morphology.dilation(binary_image, selem)
        eroded = morphology.erosion(binary_image, selem)
        return dilated - eroded

    def top_hat(self, image: np.ndarray, radius: int = 5) -> np.ndarray:
        selem = disk(radius)
        return morphology.white_tophat(image, selem)

    def bottom_hat(self, image: np.ndarray, radius: int = 5) -> np.ndarray:
        selem = disk(radius)
        return morphology.black_tophat(image, selem)