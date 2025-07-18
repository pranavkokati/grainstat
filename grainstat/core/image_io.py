import numpy as np
from PIL import Image
import skimage.io as io
from skimage import img_as_float
from typing import Union, List
import os


class ImageLoader:
    SUPPORTED_FORMATS = {'.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp'}

    def __init__(self):
        pass

    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_ext}")

        if file_ext in {'.tiff', '.tif'}:
            return self._load_tiff(image_path)
        else:
            return self._load_standard(image_path)

    def _load_tiff(self, image_path: str) -> np.ndarray:
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    # Multi-page TIFF - use first page
                    img.seek(0)

                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')

                return np.array(img, dtype=np.float64) / 255.0
        except Exception as e:
            raise ValueError(f"Failed to load TIFF image: {e}")

    def _load_standard(self, image_path: str) -> np.ndarray:
        try:
            image = io.imread(image_path)

            # Convert to grayscale if RGB
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # RGB to grayscale
                    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
                elif image.shape[2] == 4:
                    # RGBA to grayscale (ignore alpha)
                    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

            return img_as_float(image)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    def load_image_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        images = []
        for path in image_paths:
            try:
                images.append(self.load_image(path))
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
        return images

    def get_image_info(self, image_path: str) -> dict:
        with Image.open(image_path) as img:
            return {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'n_frames': getattr(img, 'n_frames', 1)
            }