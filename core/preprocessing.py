import numpy as np
from scipy import ndimage
from skimage import exposure, filters
from skimage.morphology import disk
from typing import Optional


class ImagePreprocessor:

    def __init__(self):
        pass

    def apply_filters(self, image: np.ndarray, sigma: float = 1.0,
                      use_clahe: bool = True, clahe_clip_limit: float = 0.01) -> np.ndarray:
        processed = image.copy()

        # Gaussian smoothing
        if sigma > 0:
            processed = self.gaussian_smooth(processed, sigma)

        # CLAHE contrast enhancement
        if use_clahe:
            processed = self.clahe_enhancement(processed, clahe_clip_limit)

        return processed

    def gaussian_smooth(self, image: np.ndarray, sigma: float) -> np.ndarray:
        return ndimage.gaussian_filter(image, sigma=sigma)

    def clahe_enhancement(self, image: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
        # Convert to uint8 for CLAHE
        image_uint8 = (image * 255).astype(np.uint8)

        # Apply CLAHE
        clahe_image = exposure.equalize_adapthist(
            image_uint8, clip_limit=clip_limit
        )

        return clahe_image

    def median_filter(self, image: np.ndarray, disk_size: int = 3) -> np.ndarray:
        return filters.median(image, disk(disk_size))

    def unsharp_mask(self, image: np.ndarray, radius: float = 1.0,
                     amount: float = 1.0) -> np.ndarray:
        return filters.unsharp_mask(image, radius=radius, amount=amount)

    def normalize_intensity(self, image: np.ndarray,
                            percentile_range: tuple = (1, 99)) -> np.ndarray:
        p_low, p_high = np.percentile(image, percentile_range)
        return exposure.rescale_intensity(image, in_range=(p_low, p_high))

    def remove_background(self, image: np.ndarray,
                          rolling_ball_radius: int = 50) -> np.ndarray:
        # Simple background subtraction using morphological opening
        from skimage.morphology import opening, ball

        if len(image.shape) == 2:
            selem = disk(rolling_ball_radius)
        else:
            selem = ball(rolling_ball_radius)

        background = opening(image, selem)
        return image - background

    def enhance_edges(self, image: np.ndarray, method: str = 'sobel') -> np.ndarray:
        if method == 'sobel':
            return filters.sobel(image)
        elif method == 'scharr':
            return filters.scharr(image)
        elif method == 'prewitt':
            return filters.prewitt(image)
        else:
            raise ValueError(f"Unknown edge enhancement method: {method}")