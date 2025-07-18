import numpy as np
from scipy import stats
from typing import Dict, List, Any


class StatisticsCalculator:

    def __init__(self):
        pass

    def calculate_statistics(self, grain_metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        if not grain_metrics:
            return {}

        # Extract key measurements
        ecds = [grain['ecd_um'] for grain in grain_metrics.values()]
        areas = [grain['area_um2'] for grain in grain_metrics.values()]
        aspect_ratios = [grain['aspect_ratio'] for grain in grain_metrics.values()]
        shape_factors = [grain['shape_factor'] for grain in grain_metrics.values()]

        statistics = {
            'grain_count': len(grain_metrics),
            'ecd_statistics': self._calculate_distribution_stats(ecds, 'ECD (μm)'),
            'area_statistics': self._calculate_distribution_stats(areas, 'Area (μm²)'),
            'aspect_ratio_statistics': self._calculate_distribution_stats(aspect_ratios, 'Aspect Ratio'),
            'shape_factor_statistics': self._calculate_distribution_stats(shape_factors, 'Shape Factor'),
        }

        # ASTM E112 grain size number
        astm_grain_size = self._calculate_astm_grain_size(ecds)
        statistics['astm_grain_size'] = astm_grain_size

        # Additional derived statistics
        statistics.update(self._calculate_population_metrics(grain_metrics))

        return statistics

    def _calculate_distribution_stats(self, values: List[float], name: str) -> Dict[str, float]:
        if not values:
            return {}

        values_array = np.array(values)

        # Remove infinite and NaN values
        finite_values = values_array[np.isfinite(values_array)]

        if len(finite_values) == 0:
            return {}

        stats_dict = {
            'count': len(finite_values),
            'mean': np.mean(finite_values),
            'median': np.median(finite_values),
            'std': np.std(finite_values, ddof=1) if len(finite_values) > 1 else 0,
            'variance': np.var(finite_values, ddof=1) if len(finite_values) > 1 else 0,
            'min': np.min(finite_values),
            'max': np.max(finite_values),
            'range': np.max(finite_values) - np.min(finite_values),
            'q25': np.percentile(finite_values, 25),
            'q75': np.percentile(finite_values, 75),
            'iqr': np.percentile(finite_values, 75) - np.percentile(finite_values, 25),
            'p05': np.percentile(finite_values, 5),
            'p95': np.percentile(finite_values, 95),
        }

        # Skewness and kurtosis
        if len(finite_values) > 2:
            stats_dict['skewness'] = stats.skew(finite_values)
            stats_dict['kurtosis'] = stats.kurtosis(finite_values)

        # Coefficient of variation
        if stats_dict['mean'] != 0:
            stats_dict['cv'] = stats_dict['std'] / stats_dict['mean']

        return stats_dict

    def _calculate_astm_grain_size(self, ecds: List[float]) -> Dict[str, float]:
        if not ecds:
            return {}

        finite_ecds = np.array([ecd for ecd in ecds if np.isfinite(ecd) and ecd > 0])

        if len(finite_ecds) == 0:
            return {}

        # Mean lineal intercept approximation: L ≈ π/2 * mean_ECD
        mean_ecd = np.mean(finite_ecds)
        mean_lineal_intercept = (np.pi / 2) * mean_ecd

        # ASTM E112 grain size number: G = -6.6438 * log2(L) - 3.293
        # where L is in mm, so convert from μm
        L_mm = mean_lineal_intercept / 1000

        if L_mm > 0:
            grain_size_number = -6.6438 * np.log2(L_mm) - 3.293
        else:
            grain_size_number = np.nan

        return {
            'mean_ecd_um': mean_ecd,
            'mean_lineal_intercept_um': mean_lineal_intercept,
            'mean_lineal_intercept_mm': L_mm,
            'grain_size_number': grain_size_number
        }

    def _calculate_population_metrics(self, grain_metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        # Calculate various population-level metrics

        ecds = [grain['ecd_um'] for grain in grain_metrics.values()]
        areas = [grain['area_um2'] for grain in grain_metrics.values()]

        finite_ecds = np.array([ecd for ecd in ecds if np.isfinite(ecd)])
        finite_areas = np.array([area for area in areas if np.isfinite(area)])

        metrics = {}

        if len(finite_areas) > 0:
            # Total area
            total_area = np.sum(finite_areas)
            metrics['total_grain_area_um2'] = total_area

            # Area-weighted mean ECD
            if len(finite_ecds) == len(finite_areas):
                area_weighted_ecd = np.sum(finite_ecds * finite_areas) / total_area
                metrics['area_weighted_mean_ecd_um'] = area_weighted_ecd

        if len(finite_ecds) > 0:
            # Number-weighted mean
            metrics['number_weighted_mean_ecd_um'] = np.mean(finite_ecds)

            # Grain size uniformity (based on CV)
            cv = np.std(finite_ecds) / np.mean(finite_ecds) if np.mean(finite_ecds) > 0 else np.inf
            metrics['grain_size_uniformity'] = 1 / (1 + cv) if cv != np.inf else 0

        # Size class distribution
        size_classes = self._classify_grain_sizes(finite_ecds)
        metrics['size_class_distribution'] = size_classes

        return metrics

    def _classify_grain_sizes(self, ecds: np.ndarray) -> Dict[str, int]:
        # Classify grains by size
        size_classes = {
            'ultrafine': 0,  # < 1 μm
            'fine': 0,  # 1-10 μm
            'medium': 0,  # 10-50 μm
            'coarse': 0,  # 50-100 μm
            'very_coarse': 0  # > 100 μm
        }

        for ecd in ecds:
            if ecd < 1:
                size_classes['ultrafine'] += 1
            elif ecd < 10:
                size_classes['fine'] += 1
            elif ecd < 50:
                size_classes['medium'] += 1
            elif ecd < 100:
                size_classes['coarse'] += 1
            else:
                size_classes['very_coarse'] += 1

        return size_classes

    def calculate_spatial_statistics(self, grain_metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        # Calculate spatial distribution statistics
        centroids = [(grain['centroid_um'][0], grain['centroid_um'][1])
                     for grain in grain_metrics.values()]

        if len(centroids) < 2:
            return {}

        centroids_array = np.array(centroids)

        # Nearest neighbor distances
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(centroids_array)
        distance_matrix = squareform(distances)

        # For each grain, find nearest neighbor distance
        np.fill_diagonal(distance_matrix, np.inf)
        nearest_distances = np.min(distance_matrix, axis=1)

        return {
            'nearest_neighbor_stats': self._calculate_distribution_stats(
                nearest_distances.tolist(), 'Nearest Neighbor Distance (μm)'
            ),
            'mean_nearest_neighbor_um': np.mean(nearest_distances),
            'spatial_uniformity': self._calculate_spatial_uniformity(centroids_array)
        }

    def _calculate_spatial_uniformity(self, centroids: np.ndarray) -> float:
        # Simple spatial uniformity metric based on coefficient of variation of distances
        if len(centroids) < 3:
            return 0.0

        from scipy.spatial.distance import pdist
        distances = pdist(centroids)

        cv = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else np.inf

        # Convert to uniformity (0 = non-uniform, 1 = perfectly uniform)
        return 1 / (1 + cv) if cv != np.inf else 0

    def calculate_distribution_fit(self, values: List[float]) -> Dict[str, Any]:
        # Fit common distributions and return best fit
        finite_values = np.array([v for v in values if np.isfinite(v) and v > 0])

        if len(finite_values) < 10:
            return {}

        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gamma': stats.gamma,
            'weibull': stats.weibull_min
        }

        fit_results = {}

        for name, distribution in distributions.items():
            try:
                params = distribution.fit(finite_values)
                ks_stat, p_value = stats.kstest(finite_values,
                                                lambda x: distribution.cdf(x, *params))

                fit_results[name] = {
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'aic': -2 * np.sum(distribution.logpdf(finite_values, *params)) + 2 * len(params)
                }
            except Exception:
                continue

        # Find best fit based on AIC
        if fit_results:
            best_fit = min(fit_results.items(), key=lambda x: x[1]['aic'])
            fit_results['best_fit'] = best_fit[0]

        return fit_results