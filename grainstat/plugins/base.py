from typing import Dict, Any, Callable, List
import inspect
from functools import wraps


class PluginManager:

    def __init__(self):
        self.registered_features = {}

    def register_feature(self, name: str, func: Callable):
        self.registered_features[name] = func

    def apply_features(self, grain_properties: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        if not self.registered_features:
            return {}

        custom_features = {}

        for grain_id, properties in grain_properties.items():
            grain_features = {}

            for feature_name, feature_func in self.registered_features.items():
                try:
                    # Create a mock region object from properties
                    region = MockRegion(properties)
                    result = feature_func(region)

                    if isinstance(result, dict):
                        grain_features.update(result)
                    else:
                        grain_features[feature_name] = result

                except Exception as e:
                    print(f"Warning: Feature {feature_name} failed for grain {grain_id}: {e}")
                    continue

            if grain_features:
                custom_features[grain_id] = grain_features

        return custom_features

    def list_features(self) -> List[str]:
        return list(self.registered_features.keys())

    def remove_feature(self, name: str):
        if name in self.registered_features:
            del self.registered_features[name]


class MockRegion:
    def __init__(self, properties: Dict[str, Any]):
        for key, value in properties.items():
            setattr(self, key, value)

        # Ensure standard region properties exist
        if not hasattr(self, 'area'):
            self.area = properties.get('area_px', 0)
        if not hasattr(self, 'perimeter'):
            self.perimeter = properties.get('perimeter_px', 0)
        if not hasattr(self, 'major_axis_length'):
            self.major_axis_length = properties.get('major_axis_px', 0)
        if not hasattr(self, 'minor_axis_length'):
            self.minor_axis_length = properties.get('minor_axis_px', 0)


# Global plugin manager instance
_plugin_manager = PluginManager()


def feature(func: Callable = None, *, name: str = None):
    def decorator(f):
        feature_name = name or f.__name__

        # Validate function signature
        sig = inspect.signature(f)
        if len(sig.parameters) != 1:
            raise ValueError("Feature function must accept exactly one parameter (region)")

        # Register the feature
        _plugin_manager.register_feature(feature_name, f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


# Expose plugin manager instance
def get_plugin_manager() -> PluginManager:
    return _plugin_manager


# Built-in example features
@feature
def custom_roundness(region):
    area = getattr(region, 'area', getattr(region, 'area_px', 0))
    perimeter = getattr(region, 'perimeter', getattr(region, 'perimeter_px', 0))

    if perimeter == 0:
        return 0

    return (4 * 3.14159 * area) / (perimeter ** 2)


@feature
def grain_complexity(region):
    eccentricity = getattr(region, 'eccentricity', 0)
    solidity = getattr(region, 'solidity', 1)

    # Custom complexity metric
    complexity = eccentricity * (2 - solidity)
    return complexity


@feature(name="advanced_shape_metrics")
def calculate_advanced_shape_metrics(region):
    area = getattr(region, 'area', getattr(region, 'area_px', 0))
    perimeter = getattr(region, 'perimeter', getattr(region, 'perimeter_px', 0))
    major_axis = getattr(region, 'major_axis_length', getattr(region, 'major_axis_px', 0))
    minor_axis = getattr(region, 'minor_axis_length', getattr(region, 'minor_axis_px', 0))

    if area == 0 or perimeter == 0 or major_axis == 0 or minor_axis == 0:
        return {
            'shape_regularity': 0,
            'elongation_index': 0,
            'surface_roughness': 0
        }

    # Shape regularity (how close to circle)
    equivalent_radius = (area / 3.14159) ** 0.5
    shape_regularity = (2 * equivalent_radius * 3.14159) / perimeter

    # Elongation index
    elongation_index = 1 - (minor_axis / major_axis)

    # Surface roughness (perimeter vs smooth perimeter)
    smooth_perimeter = 2 * 3.14159 * equivalent_radius
    surface_roughness = perimeter / smooth_perimeter - 1

    return {
        'shape_regularity': shape_regularity,
        'elongation_index': elongation_index,
        'surface_roughness': surface_roughness
    }


@feature
def grain_size_category(region):
    # Categorize grain size
    ecd = getattr(region, 'ecd_um', 0)

    if ecd < 1:
        return 'ultrafine'
    elif ecd < 10:
        return 'fine'
    elif ecd < 50:
        return 'medium'
    elif ecd < 100:
        return 'coarse'
    else:
        return 'very_coarse'


@feature
def texture_strength(region):
    # Calculate texture strength based on orientation and shape
    orientation = getattr(region, 'orientation', 0)
    eccentricity = getattr(region, 'eccentricity', 0)

    # Normalize orientation to 0-1 range
    normalized_orientation = abs(orientation) / (3.14159 / 2)

    # Texture strength combines orientation preference and elongation
    texture_strength = normalized_orientation * eccentricity

    return texture_strength


@feature
def geometric_moments(region):
    # Calculate additional geometric moments
    try:
        import numpy as np

        # Get central moments if available
        moments_central = getattr(region, 'moments_central', None)

        if moments_central is None:
            return {'moment_ratio': 0}

        # Calculate moment ratio (mu20 + mu02) / mu00^2
        mu20 = moments_central[2, 0] if hasattr(moments_central, 'shape') else 0
        mu02 = moments_central[0, 2] if hasattr(moments_central, 'shape') else 0
        mu00 = moments_central[0, 0] if hasattr(moments_central, 'shape') else 1

        if mu00 == 0:
            return {'moment_ratio': 0}

        moment_ratio = (mu20 + mu02) / (mu00 ** 2)

        return {'moment_ratio': moment_ratio}

    except Exception:
        return {'moment_ratio': 0}


# Example of a custom feature that uses external data
@feature
def grain_neighborhood_density(region):
    # This would typically use additional spatial information
    # For now, return a placeholder
    centroid = getattr(region, 'centroid_um', (0, 0))

    # Placeholder calculation - in real implementation, this would
    # analyze neighboring grains within a certain radius
    x, y = centroid
    density_factor = ((x + y) % 100) / 100  # Placeholder calculation

    return density_factor


# Utility function to create custom features dynamically
def create_ratio_feature(numerator_attr: str, denominator_attr: str,
                         feature_name: str = None):
    if feature_name is None:
        feature_name = f"{numerator_attr}_to_{denominator_attr}_ratio"

    def ratio_calculator(region):
        num = getattr(region, numerator_attr, 0)
        den = getattr(region, denominator_attr, 1)

        return num / den if den != 0 else 0

    # Register the dynamically created feature
    _plugin_manager.register_feature(feature_name, ratio_calculator)

    return ratio_calculator


# Utility function to create threshold-based classification features
def create_classification_feature(attribute: str, thresholds: List[float],
                                  labels: List[str], feature_name: str = None):
    if feature_name is None:
        feature_name = f"{attribute}_classification"

    if len(labels) != len(thresholds) + 1:
        raise ValueError("Number of labels must be one more than number of thresholds")

    def classifier(region):
        value = getattr(region, attribute, 0)

        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return labels[i]

        return labels[-1]

    # Register the classification feature
    _plugin_manager.register_feature(feature_name, classifier)

    return classifier


# Example usage functions for demonstration
def register_example_features():
    # Create some example dynamic features
    create_ratio_feature('area_um2', 'perimeter_um', 'area_perimeter_ratio')
    create_ratio_feature('major_axis_um', 'minor_axis_um', 'axis_ratio')

    create_classification_feature(
        'ecd_um',
        [1, 10, 50, 100],
        ['ultrafine', 'fine', 'medium', 'coarse', 'very_coarse'],
        'size_class'
    )

    create_classification_feature(
        'aspect_ratio',
        [1.2, 1.5, 2.0],
        ['equiaxed', 'slightly_elongated', 'elongated', 'very_elongated'],
        'shape_class'
    )


# Initialize example features
register_example_features()