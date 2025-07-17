import numpy as np
from typing import Dict, Any


class MetricsCalculator:

    def __init__(self):
        pass

    def calculate_derived_metrics(self, properties: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        metrics = {}

        for grain_id, props in properties.items():
            grain_metrics = props.copy()

            # Equivalent Circular Diameter (ECD)
            area_um2 = props['area_um2']
            ecd_um = 2 * np.sqrt(area_um2 / np.pi) if area_um2 > 0 else 0

            # Aspect Ratio
            major_axis = props['major_axis_um']
            minor_axis = props['minor_axis_um']
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else np.inf

            # Shape Factor (Circularity)
            perimeter_um = props['perimeter_um']
            shape_factor = (4 * np.pi * area_um2) / (perimeter_um ** 2) if perimeter_um > 0 else 0

            # Grain Boundary Length (same as perimeter)
            grain_boundary_length = perimeter_um

            # Compactness
            compactness = (perimeter_um ** 2) / (4 * np.pi * area_um2) if area_um2 > 0 else np.inf

            # Elongation
            elongation = aspect_ratio

            # Roundness (inverse of compactness)
            roundness = shape_factor

            # Form Factor
            form_factor = (4 * np.pi * area_um2) / (perimeter_um ** 2) if perimeter_um > 0 else 0

            # Convexity
            convex_area = props['convex_area_um2']
            convexity = area_um2 / convex_area if convex_area > 0 else 0

            # Rectangularity
            bbox = props['bbox_um']
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            rectangularity = area_um2 / bbox_area if bbox_area > 0 else 0

            # Diameter measures
            feret_diameter_max = major_axis  # Approximation
            feret_diameter_min = minor_axis  # Approximation
            feret_diameter_mean = (feret_diameter_max + feret_diameter_min) / 2

            # Hydraulic diameter
            hydraulic_diameter = 4 * area_um2 / perimeter_um if perimeter_um > 0 else 0

            # Sphericity (3D approximation for 2D data)
            sphericity = np.sqrt(area_um2 / np.pi) / (perimeter_um / (2 * np.pi)) if perimeter_um > 0 else 0

            # Add calculated metrics
            grain_metrics.update({
                'ecd_um': ecd_um,
                'aspect_ratio': aspect_ratio,
                'shape_factor': shape_factor,
                'grain_boundary_length_um': grain_boundary_length,
                'compactness': compactness,
                'elongation': elongation,
                'roundness': roundness,
                'form_factor': form_factor,
                'convexity': convexity,
                'rectangularity': rectangularity,
                'feret_diameter_max_um': feret_diameter_max,
                'feret_diameter_min_um': feret_diameter_min,
                'feret_diameter_mean_um': feret_diameter_mean,
                'hydraulic_diameter_um': hydraulic_diameter,
                'sphericity': sphericity,
            })

            # Calculate additional shape indices
            additional_metrics = self._calculate_additional_shape_metrics(props)
            grain_metrics.update(additional_metrics)

            metrics[grain_id] = grain_metrics

        return metrics

    def _calculate_additional_shape_metrics(self, props: Dict[str, Any]) -> Dict[str, float]:
        area = props['area_um2']
        perimeter = props['perimeter_um']
        major_axis = props['major_axis_um']
        minor_axis = props['minor_axis_um']

        metrics = {}

        # Wadell's sphericity
        if perimeter > 0:
            equivalent_diameter = 2 * np.sqrt(area / np.pi)
            wadell_sphericity = equivalent_diameter / major_axis if major_axis > 0 else 0
            metrics['wadell_sphericity'] = wadell_sphericity

        # Crofton's perimeter
        if area > 0:
            crofton_perimeter = np.pi * np.sqrt(area / np.pi) * 2
            crofton_ratio = perimeter / crofton_perimeter if crofton_perimeter > 0 else 0
            metrics['crofton_ratio'] = crofton_ratio

        # Thinness ratio
        if major_axis > 0:
            thinness_ratio = (4 * np.pi * area) / (major_axis ** 2)
            metrics['thinness_ratio'] = thinness_ratio

        # Curl (using eccentricity as approximation)
        eccentricity = props.get('eccentricity', 0)
        curl = 1 - eccentricity
        metrics['curl'] = curl

        # Fiber length (approximation for elongated particles)
        if minor_axis > 0:
            fiber_length = major_axis - minor_axis
            metrics['fiber_length_um'] = fiber_length

        # Breadth (minimum width approximation)
        breadth = minor_axis
        metrics['breadth_um'] = breadth

        return metrics

    def calculate_fractal_dimension(self, boundary_coords: np.ndarray) -> float:
        # Estimate fractal dimension using box-counting method
        if len(boundary_coords) < 4:
            return 1.0

        # Simple implementation - could be improved
        perimeter = self._calculate_perimeter_from_coords(boundary_coords)
        area = self._calculate_area_from_coords(boundary_coords)

        if area > 0:
            # Simplified fractal dimension estimate
            fd = 2 * np.log(perimeter / 4) / np.log(area)
            return np.clip(fd, 1.0, 2.0)

        return 1.0

    def _calculate_perimeter_from_coords(self, coords: np.ndarray) -> float:
        if len(coords) < 2:
            return 0.0

        # Calculate perimeter from boundary coordinates
        coords_closed = np.vstack([coords, coords[0]])
        diffs = np.diff(coords_closed, axis=0)
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        return np.sum(distances)

    def _calculate_area_from_coords(self, coords: np.ndarray) -> float:
        if len(coords) < 3:
            return 0.0

        # Shoelace formula for polygon area
        x = coords[:, 0]
        y = coords[:, 1]

        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def calculate_orientation_metrics(self, props: Dict[str, Any]) -> Dict[str, float]:
        orientation = props.get('orientation', 0)

        # Convert to degrees
        orientation_degrees = np.degrees(orientation)

        # Normalize to 0-180 range
        if orientation_degrees < 0:
            orientation_degrees += 180

        return {
            'orientation_degrees': orientation_degrees,
            'orientation_radians': orientation,
            'orientation_normalized': orientation_degrees / 180.0
        }

    def calculate_size_class(self, ecd_um: float) -> str:
        # ASTM grain size classification
        if ecd_um < 1:
            return 'ultrafine'
        elif ecd_um < 10:
            return 'fine'
        elif ecd_um < 50:
            return 'medium'
        elif ecd_um < 100:
            return 'coarse'
        else:
            return 'very_coarse'