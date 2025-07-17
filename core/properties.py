import numpy as np
from skimage import measure
from typing import Dict, List, Any


class PropertyCalculator:

    def __init__(self):
        pass

    def calculate_properties(self, labeled_image: np.ndarray,
                             scale: float = 1.0) -> Dict[int, Dict[str, Any]]:

        regions = measure.regionprops(labeled_image)
        properties = {}

        for region in regions:
            grain_id = region.label

            # Basic geometric properties
            area_px = region.area
            area_um2 = area_px * (scale ** 2)

            perimeter_px = region.perimeter
            perimeter_um = perimeter_px * scale

            # Axis lengths
            major_axis_px = region.axis_major_length
            minor_axis_px = region.axis_minor_length
            major_axis_um = major_axis_px * scale
            minor_axis_um = minor_axis_px * scale

            # Centroid
            centroid_px = region.centroid
            centroid_um = (centroid_px[0] * scale, centroid_px[1] * scale)

            # Bounding box
            bbox = region.bbox
            bbox_um = tuple(coord * scale for coord in bbox)

            properties[grain_id] = {
                # Pixel measurements
                'area_px': area_px,
                'perimeter_px': perimeter_px,
                'major_axis_px': major_axis_px,
                'minor_axis_px': minor_axis_px,
                'centroid_px': centroid_px,
                'bbox_px': bbox,

                # Physical measurements (micrometers)
                'area_um2': area_um2,
                'perimeter_um': perimeter_um,
                'major_axis_um': major_axis_um,
                'minor_axis_um': minor_axis_um,
                'centroid_um': centroid_um,
                'bbox_um': bbox_um,

                # Shape descriptors
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
                'extent': region.extent,
                'orientation': region.orientation,
                'euler_number': region.euler_number,

                # Convex properties
                'convex_area_px': region.convex_area,
                'convex_area_um2': region.convex_area * (scale ** 2),
                'convex_perimeter_px': self._calculate_convex_perimeter(region),

                # Moments
                'moments': region.moments,
                'moments_central': region.moments_central,
                'moments_hu': region.moments_hu,
                'moments_normalized': region.moments_normalized,

                # Additional properties
                'filled_area_px': region.filled_area,
                'filled_area_um2': region.filled_area * (scale ** 2),
                'equivalent_diameter_px': region.equivalent_diameter,
                'equivalent_diameter_um': region.equivalent_diameter * scale,

                # Image intensity properties (if intensity image provided)
                'mean_intensity': getattr(region, 'mean_intensity', None),
                'max_intensity': getattr(region, 'max_intensity', None),
                'min_intensity': getattr(region, 'min_intensity', None),
            }

        return properties

    def calculate_properties_with_intensity(self, labeled_image: np.ndarray,
                                            intensity_image: np.ndarray,
                                            scale: float = 1.0) -> Dict[int, Dict[str, Any]]:

        regions = measure.regionprops(labeled_image, intensity_image=intensity_image)
        properties = self.calculate_properties(labeled_image, scale)

        # Add intensity-based properties
        for region in regions:
            grain_id = region.label
            if grain_id in properties:
                properties[grain_id].update({
                    'mean_intensity': region.mean_intensity,
                    'max_intensity': region.max_intensity,
                    'min_intensity': region.min_intensity,
                    'weighted_centroid_px': region.weighted_centroid,
                    'weighted_centroid_um': tuple(c * scale for c in region.weighted_centroid),
                })

        return properties

    def _calculate_convex_perimeter(self, region) -> float:
        # Calculate perimeter of convex hull
        from skimage import measure
        convex_coords = region.convex_image
        contours = measure.find_contours(convex_coords.astype(float), 0.5)

        if contours:
            # Use the longest contour
            longest_contour = max(contours, key=len)
            return self._contour_perimeter(longest_contour)

        return 0.0

    def _contour_perimeter(self, contour: np.ndarray) -> float:
        # Calculate perimeter from contour coordinates
        if len(contour) < 2:
            return 0.0

        # Add the closing segment
        contour_closed = np.vstack([contour, contour[0]])

        # Calculate distances between consecutive points
        diffs = np.diff(contour_closed, axis=0)
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        return np.sum(distances)

    def get_grain_coordinates(self, labeled_image: np.ndarray,
                              grain_id: int) -> np.ndarray:
        # Get all pixel coordinates for a specific grain
        coords = np.where(labeled_image == grain_id)
        return np.column_stack(coords)

    def get_grain_boundary(self, labeled_image: np.ndarray,
                           grain_id: int) -> np.ndarray:
        # Extract boundary coordinates for a specific grain
        from skimage import morphology

        grain_mask = labeled_image == grain_id
        boundary = grain_mask ^ morphology.erosion(grain_mask)

        coords = np.where(boundary)
        return np.column_stack(coords)

    def calculate_inertia_tensor(self, region) -> np.ndarray:
        # Calculate the inertia tensor for the region
        mu = region.moments_central

        # Inertia tensor components
        Ixx = mu[2, 0]
        Iyy = mu[0, 2]
        Ixy = mu[1, 1]

        return np.array([[Ixx, -Ixy], [-Ixy, Iyy]])

    def calculate_shape_descriptors(self, region) -> Dict[str, float]:
        # Additional shape descriptors
        area = region.area
        perimeter = region.perimeter

        # Compactness (circularity)
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else np.inf

        # Roundness
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Elongation
        elongation = region.axis_major_length / region.axis_minor_length if region.axis_minor_length > 0 else np.inf

        return {
            'compactness': compactness,
            'roundness': roundness,
            'elongation': elongation
        }