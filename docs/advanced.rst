Advanced Features
=================

This page covers advanced GrainStat features for power users and specialized applications.

Custom Analysis Pipelines
--------------------------

Building Custom Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~

For specialized analysis needs, you can build custom workflows using GrainStat's modular components:

.. code-block:: python

   from grainstat.core.image_io import ImageLoader
   from grainstat.core.preprocessing import ImagePreprocessor
   from grainstat.core.segmentation import GrainSegmenter
   from grainstat.core.morphology import MorphologyProcessor
   from grainstat.core.properties import PropertyCalculator
   from grainstat.core.metrics import MetricsCalculator

   class CustomGrainAnalyzer:
       """Custom analyzer with specialized preprocessing."""

       def __init__(self):
           self.loader = ImageLoader()
           self.preprocessor = ImagePreprocessor()
           self.segmenter = GrainSegmenter()
           self.morphology = MorphologyProcessor()
           self.properties = PropertyCalculator()
           self.metrics = MetricsCalculator()

       def analyze_dual_phase_steel(self, image_path, scale):
           """Specialized analysis for dual-phase steel."""

           # Load image
           image = self.loader.load_image(image_path)

           # Custom preprocessing for dual-phase structures
           processed = self.preprocessor.normalize_intensity(
               image, percentile_range=(2, 98)
           )
           processed = self.preprocessor.unsharp_mask(
               processed, radius=1.5, amount=0.8
           )
           processed = self.preprocessor.clahe_enhancement(
               processed, clip_limit=0.02
           )

           # Multi-scale segmentation
           binary1 = self.segmenter.otsu_threshold(processed)
           binary2 = self.segmenter.adaptive_threshold(processed, block_size=25)

           # Combine segmentations
           combined_binary = binary1 & binary2

           # Enhanced morphological cleaning
           cleaned = self.morphology.opening(combined_binary, radius=1)
           cleaned = self.morphology.remove_small_objects(cleaned, min_area=30)
           cleaned = self.morphology.separate_touching_objects(cleaned)

           # Watershed with custom markers
           labeled = self.segmenter.watershed_segmentation(cleaned)

           # Calculate properties and metrics
           properties = self.properties.calculate_properties(labeled, scale)
           metrics = self.metrics.calculate_derived_metrics(properties)

           return {
               'binary_image': combined_binary,
               'labeled_image': labeled,
               'properties': properties,
               'metrics': metrics
           }

Advanced Segmentation Techniques
---------------------------------

Multi-Scale Analysis
~~~~~~~~~~~~~~~~~~~~

For complex microstructures with multiple grain size populations:

.. code-block:: python

   import numpy as np
   from scipy import ndimage
   from skimage import segmentation, morphology

   def multi_scale_segmentation(image, scales=[1, 2, 4]):
       """Segment grains at multiple scales and combine results."""

       segmentations = []

       for scale in scales:
           # Apply Gaussian smoothing at different scales
           smoothed = ndimage.gaussian_filter(image, sigma=scale)

           # Threshold and label
           from grainstat.core.segmentation import GrainSegmenter
           segmenter = GrainSegmenter()

           binary = segmenter.otsu_threshold(smoothed)
           labeled = segmenter.watershed_segmentation(binary)

           segmentations.append(labeled)

       # Combine segmentations using consensus approach
       consensus = combine_segmentations(segmentations)
       return consensus

   def combine_segmentations(segmentations):
       """Combine multiple segmentations using voting."""

       # Implementation of consensus-based combination
       # This is a simplified version - real implementation would be more sophisticated

       # For now, use the middle-scale segmentation
       return segmentations[len(segmentations) // 2]

Machine Learning Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate machine learning models for advanced segmentation:

.. code-block:: python

   import numpy as np
   from sklearn.cluster import KMeans
   from sklearn.mixture import GaussianMixture

   def ml_enhanced_segmentation(image, method='kmeans'):
       """Use machine learning for enhanced segmentation."""

       # Prepare feature vectors
       features = extract_texture_features(image)

       if method == 'kmeans':
           # K-means clustering
           kmeans = KMeans(n_clusters=3, random_state=42)
           labels = kmeans.fit_predict(features)

       elif method == 'gmm':
           # Gaussian Mixture Model
           gmm = GaussianMixture(n_components=3, random_state=42)
           labels = gmm.fit_predict(features)

       # Reshape labels back to image shape
       labeled_image = labels.reshape(image.shape)

       # Post-process to get grain boundaries
       grain_mask = labeled_image == 1  # Assume label 1 is grains

       return grain_mask

   def extract_texture_features(image, window_size=5):
       """Extract texture features for each pixel."""

       from skimage.feature import local_binary_pattern
       from skimage.filters import gabor

       h, w = image.shape
       features = []

       # Local Binary Pattern
       lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

       # Gabor filters
       gabor_responses = []
       for angle in [0, 45, 90, 135]:
           real, _ = gabor(image, frequency=0.1, theta=np.radians(angle))
           gabor_responses.append(real)

       # Combine features for each pixel
       for i in range(h):
           for j in range(w):
               pixel_features = [
                   image[i, j],  # Intensity
                   lbp[i, j],    # LBP
                   *[resp[i, j] for resp in gabor_responses]  # Gabor responses
               ]
               features.append(pixel_features)

       return np.array(features)

Statistical Analysis Extensions
-------------------------------

Advanced Distribution Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive statistical analysis with multiple distribution models:

.. code-block:: python

   import numpy as np
   from scipy import stats
   import matplotlib.pyplot as plt

   class AdvancedStatisticalAnalysis:
       """Advanced statistical analysis for grain size distributions."""

       def __init__(self):
           self.distributions = {
               'normal': stats.norm,
               'lognormal': stats.lognorm,
               'gamma': stats.gamma,
               'weibull': stats.weibull_min,
               'beta': stats.beta,
               'exponential': stats.expon,
               'rayleigh': stats.rayleigh
           }

       def comprehensive_analysis(self, grain_sizes):
           """Perform comprehensive statistical analysis."""

           results = {
               'basic_stats': self.calculate_basic_statistics(grain_sizes),
               'distribution_fits': self.fit_all_distributions(grain_sizes),
               'goodness_of_fit': self.test_goodness_of_fit(grain_sizes),
               'outlier_analysis': self.detect_outliers(grain_sizes),
               'shape_analysis': self.analyze_distribution_shape(grain_sizes)
           }

           return results

       def fit_all_distributions(self, data):
           """Fit all available distributions to the data."""

           fit_results = {}

           for name, distribution in self.distributions.items():
               try:
                   # Fit parameters
                   params = distribution.fit(data)

                   # Calculate log-likelihood
                   log_likelihood = np.sum(distribution.logpdf(data, *params))

                   # Calculate AIC and BIC
                   k = len(params)  # Number of parameters
                   n = len(data)    # Sample size

                   aic = 2 * k - 2 * log_likelihood
                   bic = k * np.log(n) - 2 * log_likelihood

                   # Kolmogorov-Smirnov test
                   ks_stat, ks_p = stats.kstest(data,
                                               lambda x: distribution.cdf(x, *params))

                   fit_results[name] = {
                       'parameters': params,
                       'log_likelihood': log_likelihood,
                       'aic': aic,
                       'bic': bic,
                       'ks_statistic': ks_stat,
                       'ks_p_value': ks_p
                   }

               except Exception as e:
                   print(f"Failed to fit {name}: {e}")

           return fit_results

       def detect_outliers(self, data, method='iqr'):
           """Detect statistical outliers in grain size data."""

           if method == 'iqr':
               Q1 = np.percentile(data, 25)
               Q3 = np.percentile(data, 75)
               IQR = Q3 - Q1

               lower_bound = Q1 - 1.5 * IQR
               upper_bound = Q3 + 1.5 * IQR

               outliers = data[(data < lower_bound) | (data > upper_bound)]

           elif method == 'zscore':
               z_scores = np.abs(stats.zscore(data))
               outliers = data[z_scores > 3]

           elif method == 'modified_zscore':
               median = np.median(data)
               mad = np.median(np.abs(data - median))
               modified_z_scores = 0.6745 * (data - median) / mad
               outliers = data[np.abs(modified_z_scores) > 3.5]

           return {
               'outliers': outliers,
               'outlier_indices': np.where((data < lower_bound) | (data > upper_bound))[0] if method == 'iqr' else None,
               'outlier_percentage': len(outliers) / len(data) * 100
           }

Spatial Analysis
~~~~~~~~~~~~~~~~

Analyze spatial distribution patterns of grains:

.. code-block:: python

   import numpy as np
   from scipy.spatial.distance import pdist, squareform
   from scipy.spatial import cKDTree

   class SpatialAnalysis:
       """Spatial analysis of grain distributions."""

       def analyze_spatial_patterns(self, grain_centroids, grain_sizes):
           """Comprehensive spatial pattern analysis."""

           results = {
               'nearest_neighbor': self.nearest_neighbor_analysis(grain_centroids),
               'ripley_k': self.ripley_k_function(grain_centroids),
               'size_clustering': self.analyze_size_clustering(grain_centroids, grain_sizes),
               'spatial_autocorrelation': self.spatial_autocorrelation(grain_centroids, grain_sizes)
           }

           return results

       def nearest_neighbor_analysis(self, centroids):
           """Analyze nearest neighbor distances."""

           tree = cKDTree(centroids)

           # Find nearest neighbor distances
           distances, indices = tree.query(centroids, k=2)
           nn_distances = distances[:, 1]  # Exclude self (distance=0)

           # Expected distance for random distribution
           n = len(centroids)
           area = self.estimate_area(centroids)
           density = n / area
           expected_distance = 1 / (2 * np.sqrt(density))

           # R statistic
           observed_mean = np.mean(nn_distances)
           R = observed_mean / expected_distance

           return {
               'mean_distance': observed_mean,
               'expected_distance': expected_distance,
               'R_statistic': R,
               'interpretation': self.interpret_R_statistic(R)
           }

       def ripley_k_function(self, centroids, max_distance=None):
           """Calculate Ripley's K function for spatial pattern analysis."""

           if max_distance is None:
               # Use 1/4 of the maximum inter-point distance
               distances = pdist(centroids)
               max_distance = np.max(distances) / 4

           # Distance values to evaluate
           r_values = np.linspace(0.1, max_distance, 50)

           n = len(centroids)
           area = self.estimate_area(centroids)

           k_values = []

           for r in r_values:
               # Count pairs within distance r
               distances = squareform(pdist(centroids))
               within_r = np.sum(distances <= r) - n  # Exclude diagonal

               # K function estimate
               k_r = (area / (n * (n - 1))) * within_r
               k_values.append(k_r)

           # L function (normalized)
           l_values = np.sqrt(np.array(k_values) / np.pi) - r_values

           return {
               'r_values': r_values,
               'k_values': k_values,
               'l_values': l_values
           }

       def estimate_area(self, centroids):
           """Estimate the area of the observation window."""

           min_x, max_x = np.min(centroids[:, 0]), np.max(centroids[:, 0])
           min_y, max_y = np.min(centroids[:, 1]), np.max(centroids[:, 1])

           return (max_x - min_x) * (max_y - min_y)

Performance Optimization
------------------------

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large images, use memory-efficient processing:

.. code-block:: python

   import numpy as np
   from pathlib import Path

   class MemoryEfficientAnalyzer:
       """Memory-efficient analyzer for large images."""

       def __init__(self, chunk_size=1024):
           self.chunk_size = chunk_size

       def analyze_large_image(self, image_path, scale, overlap=128):
           """Analyze large images using chunking strategy."""

           from grainstat.core.image_io import ImageLoader

           loader = ImageLoader()

           # Get image info without loading full image
           with Image.open(image_path) as img:
               width, height = img.size

           # Calculate chunks
           chunks = self.calculate_chunks(width, height, self.chunk_size, overlap)

           all_grains = []

           for chunk_info in chunks:
               # Load and process chunk
               chunk_image = self.load_image_chunk(image_path, chunk_info)
               chunk_results = self.analyze_chunk(chunk_image, scale, chunk_info)

               all_grains.extend(chunk_results)

           # Merge overlapping grains
           merged_grains = self.merge_overlapping_grains(all_grains, overlap)

           return merged_grains

       def calculate_chunks(self, width, height, chunk_size, overlap):
           """Calculate chunk boundaries."""

           chunks = []

           for y in range(0, height, chunk_size - overlap):
               for x in range(0, width, chunk_size - overlap):
                   x_end = min(x + chunk_size, width)
                   y_end = min(y + chunk_size, height)

                   chunks.append({
                       'x_start': x, 'x_end': x_end,
                       'y_start': y, 'y_end': y_end
                   })

           return chunks

GPU Acceleration
~~~~~~~~~~~~~~~~

Use GPU acceleration for intensive computations:

.. code-block:: python

   try:
       import cupy as cp
       GPU_AVAILABLE = True
   except ImportError:
       GPU_AVAILABLE = False
       cp = None

   class GPUAcceleratedAnalysis:
       """GPU-accelerated grain analysis operations."""

       def __init__(self):
           self.use_gpu = GPU_AVAILABLE

           if self.use_gpu:
               print("GPU acceleration enabled")
           else:
               print("GPU acceleration not available, using CPU")

       def gpu_morphology(self, binary_image, operation='opening', radius=2):
           """GPU-accelerated morphological operations."""

           if not self.use_gpu:
               # Fallback to CPU
               from grainstat.core.morphology import MorphologyProcessor
               processor = MorphologyProcessor()

               if operation == 'opening':
                   return processor.opening(binary_image, radius)
               elif operation == 'closing':
                   return processor.closing(binary_image, radius)

           # GPU implementation
           gpu_image = cp.asarray(binary_image)

           # Create structuring element
           from cupyx.scipy import ndimage as gpu_ndimage

           if operation == 'opening':
               result = gpu_ndimage.binary_opening(gpu_image, iterations=radius)
           elif operation == 'closing':
               result = gpu_ndimage.binary_closing(gpu_image, iterations=radius)

           return cp.asnumpy(result)

Integration with Other Tools
----------------------------

ImageJ Integration
~~~~~~~~~~~~~~~~~~

Interface with ImageJ for additional processing:

.. code-block:: python

   import subprocess
   import tempfile
   from pathlib import Path

   class ImageJIntegration:
       """Interface with ImageJ for additional processing."""

       def __init__(self, imagej_path=None):
           self.imagej_path = imagej_path or self.find_imagej()

       def find_imagej(self):
           """Attempt to find ImageJ installation."""
           common_paths = [
               '/Applications/ImageJ.app/Contents/MacOS/ImageJ',
               'C:/Program Files/ImageJ/ImageJ.exe',
               '/usr/local/bin/imagej'
           ]

           for path in common_paths:
               if Path(path).exists():
                   return path

           return None

       def enhance_with_imagej(self, image_path, operations):
           """Apply ImageJ operations to enhance image."""

           if not self.imagej_path:
               raise RuntimeError("ImageJ not found")

           # Create ImageJ macro
           macro = self.create_macro(operations)

           with tempfile.NamedTemporaryFile(mode='w', suffix='.ijm', delete=False) as f:
               f.write(macro)
               macro_path = f.name

           try:
               # Run ImageJ
               cmd = [self.imagej_path, '-batch', macro_path, image_path]
               subprocess.run(cmd, check=True)

               # Return processed image path
               output_path = image_path.replace('.tif', '_processed.tif')
               return output_path

           finally:
               Path(macro_path).unlink()

MATLAB Integration
~~~~~~~~~~~~~~~~~~

Interface with MATLAB for specialized analysis:

.. code-block:: python

   try:
       import matlab.engine
       MATLAB_AVAILABLE = True
   except ImportError:
       MATLAB_AVAILABLE = False

   class MATLABIntegration:
       """Interface with MATLAB for specialized analysis."""

       def __init__(self):
           if MATLAB_AVAILABLE:
               self.eng = matlab.engine.start_matlab()
           else:
               self.eng = None

       def mtex_analysis(self, ebsd_data_path):
           """Use MTEX for crystallographic analysis."""

           if not self.eng:
               raise RuntimeError("MATLAB engine not available")

           # Call MATLAB function
           result = self.eng.analyze_ebsd_grains(ebsd_data_path)

           return result

Custom Export Formats
----------------------

Specialized Export Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom export formats for specific applications:

.. code-block:: python

   import xml.etree.ElementTree as ET
   from pathlib import Path

   class CustomExporter:
       """Custom export formats for specialized applications."""

       def export_to_xml(self, grain_metrics, output_path):
           """Export to XML format for database integration."""

           root = ET.Element("GrainAnalysis")

           # Metadata
           metadata = ET.SubElement(root, "Metadata")
           ET.SubElement(metadata, "Software").text = "GrainStat"
           ET.SubElement(metadata, "Version").text = "1.0.0"
           ET.SubElement(metadata, "Timestamp").text = datetime.now().isoformat()

           # Grains
           grains_elem = ET.SubElement(root, "Grains")

           for grain_id, grain_data in grain_metrics.items():
               grain_elem = ET.SubElement(grains_elem, "Grain")
               grain_elem.set("id", str(grain_id))

               for key, value in grain_data.items():
                   if isinstance(value, (int, float, str)):
                       elem = ET.SubElement(grain_elem, key)
                       elem.text = str(value)

           # Write to file
           tree = ET.ElementTree(root)
           tree.write(output_path, encoding='utf-8', xml_declaration=True)

       def export_to_vtk(self, labeled_image, grain_metrics, output_path):
           """Export to VTK format for 3D visualization."""

           # This is a simplified example
           # Real VTK export would be more complex

           with open(output_path, 'w') as f:
               f.write("# vtk DataFile Version 3.0\n")
               f.write("Grain structure\n")
               f.write("ASCII\n")
               f.write("DATASET STRUCTURED_POINTS\n")

               h, w = labeled_image.shape
               f.write(f"DIMENSIONS {w} {h} 1\n")
               f.write("SPACING 1.0 1.0 1.0\n")
               f.write("ORIGIN 0.0 0.0 0.0\n")

               f.write(f"POINT_DATA {w * h}\n")
               f.write("SCALARS grain_id int\n")
               f.write("LOOKUP_TABLE default\n")

               for value in labeled_image.flatten():
                   f.write(f"{value}\n")

This comprehensive advanced features documentation provides power users with the tools they need to extend GrainStat for specialized applications and research needs.