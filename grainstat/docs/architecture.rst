Architecture
============

This document describes the technical architecture and design principles behind GrainStat.

Overview
--------

GrainStat follows a modular, pipeline-based architecture that separates concerns and enables extensibility. The system is designed for:

- **Modularity**: Each component has a single, well-defined responsibility
- **Extensibility**: Easy to add new features without modifying core code
- **Performance**: Efficient processing of large images and datasets
- **Maintainability**: Clean interfaces and comprehensive testing
- **Usability**: Simple high-level API with powerful low-level access

System Architecture
-------------------

High-Level Components
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    GrainStat Architecture                   │
   ├─────────────────────────────────────────────────────────────┤
   │                                                             │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
   │  │     CLI     │  │   Python    │  │ Interactive │         │
   │  │   Interface │  │     API     │  │   Viewer    │         │
   │  └─────────────┘  └─────────────┘  └─────────────┘         │
   │                           │                                  │
   │  ┌─────────────────────────┼─────────────────────────────┐   │
   │  │              Main Analysis Engine               │   │
   │  │  ┌─────────────────────────────────────────────┐   │   │
   │  │  │            GrainAnalyzer                     │   │   │
   │  │  └─────────────────────────────────────────────┘   │   │
   │  └─────────────────────────┼─────────────────────────────┘   │
   │                           │                                  │
   │  ┌─────────────────────────┼─────────────────────────────┐   │
   │  │                  Core Processing                    │   │
   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
   │  │  │Image I/O│  │Preproc. │  │Segment. │  │Morphol. │ │   │
   │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │
   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
   │  │  │Props.   │  │Metrics  │  │Stats.   │  │Plugins  │ │   │
   │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │
   │  └─────────────────────────────────────────────────────────┘   │
   │                                                             │
   │  ┌─────────────────────────────────────────────────────────┐   │
   │  │              Output and Visualization                   │   │
   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
   │  │  │  Plots  │  │ Export  │  │Reports  │  │ Batch   │     │   │
   │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │   │
   │  └─────────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────┘

Data Flow Pipeline
~~~~~~~~~~~~~~~~~~

The grain analysis follows a clear data pipeline:

.. code-block:: text

   Raw Image → Preprocessing → Segmentation → Morphology → Properties → Metrics → Statistics
        │            │              │             │            │          │          │
        │            │              │             │            │          │          └─→ Export
        │            │              │             │            │          └─→ Visualization
        │            │              │             │            └─→ Plugin Features
        │            │              │             └─→ Labeled Image
        │            │              └─→ Binary Image
        │            └─→ Enhanced Image
        └─→ Calibrated Image

Core Modules
------------

grainstat.core Package
~~~~~~~~~~~~~~~~~~~~~~

The core package contains the fundamental analysis components:

**image_io.py**
   - **Purpose**: Handle image loading and format conversion
   - **Key Class**: ``ImageLoader``
   - **Responsibilities**:
     - Multi-format image loading (TIFF, PNG, JPEG, BMP)
     - Multi-page TIFF support
     - Automatic grayscale conversion
     - Image validation and metadata extraction

**preprocessing.py**
   - **Purpose**: Prepare images for segmentation
   - **Key Class**: ``ImagePreprocessor``
   - **Responsibilities**:
     - Gaussian smoothing for noise reduction
     - CLAHE contrast enhancement
     - Intensity normalization
     - Edge enhancement filters

**segmentation.py**
   - **Purpose**: Identify grain boundaries and regions
   - **Key Class**: ``GrainSegmenter``
   - **Responsibilities**:
     - Otsu and adaptive thresholding
     - Watershed segmentation
     - Region growing algorithms
     - Binary image labeling

**morphology.py**
   - **Purpose**: Clean and refine segmented images
   - **Key Class**: ``MorphologyProcessor``
   - **Responsibilities**:
     - Opening and closing operations
     - Small object removal
     - Hole filling
     - Boundary extraction

**properties.py**
   - **Purpose**: Calculate basic grain properties
   - **Key Class**: ``PropertyCalculator``
   - **Responsibilities**:
     - Area, perimeter, centroid calculation
     - Shape descriptors (eccentricity, solidity)
     - Spatial scaling conversion
     - RegionProps interface

**metrics.py**
   - **Purpose**: Calculate derived grain metrics
   - **Key Class**: ``MetricsCalculator``
   - **Responsibilities**:
     - Equivalent circular diameter (ECD)
     - Shape factors and ratios
     - Geometric moments
     - Custom shape indices

**statistics.py**
   - **Purpose**: Population-level statistical analysis
   - **Key Class**: ``StatisticsCalculator``
   - **Responsibilities**:
     - Distribution statistics
     - ASTM E112 grain size calculation
     - Percentile analysis
     - Distribution fitting

Design Patterns
---------------

Pipeline Pattern
~~~~~~~~~~~~~~~~

The analysis workflow follows the Pipeline pattern:

.. code-block:: python

   class AnalysisPipeline:
       """Represents the grain analysis pipeline."""

       def __init__(self):
           self.stages = [
               ImageLoader(),
               ImagePreprocessor(),
               GrainSegmenter(),
               MorphologyProcessor(),
               PropertyCalculator(),
               MetricsCalculator(),
               StatisticsCalculator()
           ]

       def process(self, image_path, **params):
           """Process through all pipeline stages."""
           data = {'image_path': image_path, 'params': params}

           for stage in self.stages:
               data = stage.process(data)

           return data

Strategy Pattern
~~~~~~~~~~~~~~~~

Different algorithms are implemented using the Strategy pattern:

.. code-block:: python

   class ThresholdingStrategy:
       """Abstract base for thresholding algorithms."""

       def threshold(self, image):
           raise NotImplementedError

   class OtsuThresholding(ThresholdingStrategy):
       """Otsu thresholding implementation."""

       def threshold(self, image):
           return filters.threshold_otsu(image)

   class AdaptiveThresholding(ThresholdingStrategy):
       """Adaptive thresholding implementation."""

       def threshold(self, image):
           return filters.threshold_local(image)

Plugin Pattern
~~~~~~~~~~~~~~

The plugin system uses decorators and registration:

.. code-block:: python

   class PluginManager:
       """Manages feature plugins."""

       def __init__(self):
           self.plugins = {}

       def register(self, name, func):
           """Register a plugin function."""
           self.plugins[name] = func

       def apply_all(self, region):
           """Apply all registered plugins."""
           results = {}
           for name, func in self.plugins.items():
               results[name] = func(region)
           return results

Factory Pattern
~~~~~~~~~~~~~~~

Object creation uses the Factory pattern:

.. code-block:: python

   class AnalyzerFactory:
       """Factory for creating analyzers."""

       @staticmethod
       def create_analyzer(analyzer_type='standard'):
           """Create analyzer instance."""

           if analyzer_type == 'standard':
               return GrainAnalyzer()
           elif analyzer_type == 'batch':
               return BatchProcessor()
           elif analyzer_type == 'interactive':
               return InteractiveAnalyzer()
           else:
               raise ValueError(f"Unknown analyzer type: {analyzer_type}")

Memory Management
-----------------

Efficient Memory Usage
~~~~~~~~~~~~~~~~~~~~~~

GrainStat implements several strategies for efficient memory usage:

**Lazy Loading**
   Images are loaded only when needed and released after processing.

**In-Place Operations**
   Many operations modify arrays in-place to reduce memory allocation.

**Chunked Processing**
   Large images can be processed in chunks to manage memory usage.

**Object Pooling**
   Reuse of expensive objects like morphological structuring elements.

.. code-block:: python

   class MemoryEfficientProcessor:
       """Memory-efficient image processor."""

       def __init__(self, chunk_size=1024):
           self.chunk_size = chunk_size
           self.object_pool = {}

       def process_large_image(self, image_path):
           """Process large images in chunks."""

           # Process in chunks to manage memory
           for chunk in self.get_chunks(image_path):
               processed_chunk = self.process_chunk(chunk)
               yield processed_chunk

               # Explicit memory cleanup
               del chunk
               gc.collect()

Error Handling
--------------

Comprehensive Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GrainStat implements layered error handling:

.. code-block:: python

   class GrainStatError(Exception):
       """Base exception for GrainStat errors."""
       pass

   class ImageLoadError(GrainStatError):
       """Error loading image file."""
       pass

   class SegmentationError(GrainStatError):
       """Error in segmentation process."""
       pass

   class AnalysisError(GrainStatError):
       """Error in analysis process."""
       pass

**Error Recovery**
   - Graceful degradation when possible
   - Detailed error messages with context
   - Suggestions for resolution

**Validation**
   - Input parameter validation
   - Image format and content validation
   - Results validation and sanity checks

Performance Optimization
------------------------

Algorithmic Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Efficient Data Structures**
   - NumPy arrays for numerical computations
   - SciPy sparse matrices for large, sparse data
   - Efficient image representations

**Algorithm Selection**
   - Optimized algorithms based on input characteristics
   - Parallel processing where beneficial
   - Memory-efficient implementations

**Caching Strategies**
   - Cache expensive computations
   - Reuse of intermediate results
   - Intelligent cache invalidation

.. code-block:: python

   class OptimizedAnalyzer:
       """Performance-optimized analyzer."""

       def __init__(self):
           self.cache = {}
           self.parallel_threshold = 1000  # Grains

       def analyze(self, image, **params):
           """Optimized analysis with caching."""

           cache_key = self.get_cache_key(image, params)

           if cache_key in self.cache:
               return self.cache[cache_key]

           # Choose algorithm based on problem size
           if self.estimate_grain_count(image) > self.parallel_threshold:
               result = self.parallel_analyze(image, **params)
           else:
               result = self.sequential_analyze(image, **params)

           self.cache[cache_key] = result
           return result

Parallel Processing
~~~~~~~~~~~~~~~~~~~

**Multiprocessing Support**
   - CPU-bound tasks use multiprocessing
   - Automatic worker count based on CPU cores
   - Efficient data sharing between processes

**Threading for I/O**
   - I/O-bound operations use threading
   - Asynchronous file operations
   - Non-blocking user interface updates

**GPU Acceleration (Future)**
   - CUDA support for compatible operations
   - Automatic fallback to CPU
   - Memory management for GPU operations

Testing Architecture
--------------------

Test Structure
~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── unit/                    # Unit tests for individual modules
   │   ├── test_image_io.py
   │   ├── test_preprocessing.py
   │   ├── test_segmentation.py
   │   └── ...
   ├── integration/             # Integration tests for workflows
   │   ├── test_full_pipeline.py
   │   ├── test_batch_processing.py
   │   └── ...
   ├── performance/             # Performance and benchmark tests
   │   ├── test_memory_usage.py
   │   ├── test_processing_speed.py
   │   └── ...
   └── fixtures/                # Test data and utilities
       ├── synthetic_images.py
       ├── sample_data/
       └── ...

Testing Strategies
~~~~~~~~~~~~~~~~~~

**Unit Testing**
   - Individual function and class testing
   - Mock objects for external dependencies
   - Property-based testing for algorithms

**Integration Testing**
   - End-to-end workflow testing
   - Real image processing validation
   - Cross-platform compatibility testing

**Performance Testing**
   - Memory usage profiling
   - Speed benchmarking
   - Scalability testing

Extensibility
-------------

Plugin Architecture
~~~~~~~~~~~~~~~~~~~

The plugin system allows extension without core modification:

.. code-block:: python

   # Plugin registration is automatic
   @feature
   def custom_metric(region):
       """Custom grain metric."""
       return region.area / region.perimeter

   # Plugins are automatically discovered and used
   analyzer = GrainAnalyzer()
   results = analyzer.analyze("image.tif")
   # custom_metric is automatically included

Custom Analyzers
~~~~~~~~~~~~~~~~

Users can create specialized analyzers:

.. code-block:: python

   class SteelAnalyzer(GrainAnalyzer):
       """Specialized analyzer for steel microstructures."""

       def __init__(self):
           super().__init__()
           # Custom preprocessing for steel
           self.preprocessor.default_sigma = 1.5

       def analyze_dual_phase(self, image_path, **params):
           """Specialized dual-phase analysis."""
           # Custom analysis workflow
           pass

Future Architecture Plans
-------------------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~~

**Distributed Processing**
   - Cluster computing support
   - Cloud-based analysis
   - MapReduce-style processing

**Machine Learning Integration**
   - Deep learning for segmentation
   - Feature learning and extraction
   - Predictive modeling capabilities

**3D Analysis Support**
   - Volumetric image processing
   - 3D grain reconstruction
   - Tomography integration

**Real-Time Processing**
   - Live image analysis
   - Streaming data processing
   - Real-time quality control

**Web Integration**
   - REST API development
   - Web-based user interface
   - Browser-based analysis

Scalability Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Horizontal Scaling**
   - Microservices architecture
   - Container-based deployment
   - Load balancing and distribution

**Vertical Scaling**
   - Optimized memory usage
   - GPU acceleration
   - High-performance computing integration

**Data Management**
   - Database integration
   - Metadata management
   - Result versioning and tracking

See Also
--------

- :doc:`api` - Complete API reference
- :doc:`contributing` - How to contribute to the architecture
- :doc:`advanced` - Advanced usage patterns
- :doc:`plugins` - Plugin development guide