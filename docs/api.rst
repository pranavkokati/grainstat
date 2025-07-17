API Reference
=============

This page provides detailed documentation for all GrainStat classes, functions, and methods.

Main Interface
--------------

.. automodule:: grainstat.main
   :members:
   :undoc-members:
   :show-inheritance:

GrainAnalyzer
~~~~~~~~~~~~~

.. autoclass:: grainstat.main.GrainAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Core Modules
------------

Image I/O
~~~~~~~~~~

.. automodule:: grainstat.core.image_io
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
~~~~~~~~~~~~~

.. automodule:: grainstat.core.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Segmentation
~~~~~~~~~~~~

.. automodule:: grainstat.core.segmentation
   :members:
   :undoc-members:
   :show-inheritance:

Morphology
~~~~~~~~~~

.. automodule:: grainstat.core.morphology
   :members:
   :undoc-members:
   :show-inheritance:

Properties
~~~~~~~~~~

.. automodule:: grainstat.core.properties
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
~~~~~~~

.. automodule:: grainstat.core.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Statistics
~~~~~~~~~~

.. automodule:: grainstat.core.statistics
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

Plots
~~~~~

.. automodule:: grainstat.visualization.plots
   :members:
   :undoc-members:
   :show-inheritance:

Interactive
~~~~~~~~~~~

.. automodule:: grainstat.visualization.interactive
   :members:
   :undoc-members:
   :show-inheritance:

Export and Reporting
--------------------

Data Export
~~~~~~~~~~~

.. automodule:: grainstat.export.data_export
   :members:
   :undoc-members:
   :show-inheritance:

Reports
~~~~~~~

.. automodule:: grainstat.export.reports
   :members:
   :undoc-members:
   :show-inheritance:

Processing
----------

Batch Processing
~~~~~~~~~~~~~~~~

.. automodule:: grainstat.processing.batch
   :members:
   :undoc-members:
   :show-inheritance:

Plugin System
-------------

Base Classes
~~~~~~~~~~~~

.. automodule:: grainstat.plugins.base
   :members:
   :undoc-members:
   :show-inheritance:

Plugin Decorators
~~~~~~~~~~~~~~~~~~

.. autofunction:: grainstat.plugins.base.feature

.. autofunction:: grainstat.plugins.base.get_plugin_manager

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: grainstat.plugins.base.create_ratio_feature

.. autofunction:: grainstat.plugins.base.create_classification_feature

Command Line Interface
----------------------

.. automodule:: grainstat.cli
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

Analysis Results
~~~~~~~~~~~~~~~~

The main analysis functions return dictionaries with the following structure:

.. code-block:: python

   {
       'properties': {
           grain_id: {
               # Basic measurements
               'area_px': int,
               'area_um2': float,
               'perimeter_px': float,
               'perimeter_um': float,
               'centroid_px': tuple,
               'centroid_um': tuple,
               'bbox_px': tuple,
               'bbox_um': tuple,

               # Shape descriptors
               'eccentricity': float,
               'solidity': float,
               'extent': float,
               'orientation': float,
               'major_axis_px': float,
               'major_axis_um': float,
               'minor_axis_px': float,
               'minor_axis_um': float,

               # Additional properties...
           }
       },
       'metrics': {
           grain_id: {
               # All properties plus derived metrics
               'ecd_um': float,
               'aspect_ratio': float,
               'shape_factor': float,
               'compactness': float,
               'elongation': float,
               'roundness': float,

               # Custom plugin features...
           }
       },
       'statistics': {
           'grain_count': int,
           'ecd_statistics': {
               'mean': float,
               'median': float,
               'std': float,
               'min': float,
               'max': float,
               'q25': float,
               'q75': float,
               'skewness': float,
               'kurtosis': float
           },
           'astm_grain_size': {
               'grain_size_number': float,
               'mean_ecd_um': float,
               'mean_lineal_intercept_um': float
           },
           'size_class_distribution': {
               'ultrafine': int,
               'fine': int,
               'medium': int,
               'coarse': int,
               'very_coarse': int
           }
       }
   }

Grain Properties
~~~~~~~~~~~~~~~~

Each grain is characterized by the following properties:

**Geometric Properties**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``area_px``
     - int
     - Area in pixels
   * - ``area_um2``
     - float
     - Area in square micrometers
   * - ``perimeter_px``
     - float
     - Perimeter in pixels
   * - ``perimeter_um``
     - float
     - Perimeter in micrometers
   * - ``centroid_px``
     - tuple
     - Centroid coordinates in pixels (y, x)
   * - ``centroid_um``
     - tuple
     - Centroid coordinates in micrometers
   * - ``bbox_px``
     - tuple
     - Bounding box in pixels (min_row, min_col, max_row, max_col)
   * - ``bbox_um``
     - tuple
     - Bounding box in micrometers

**Shape Descriptors**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``major_axis_px``
     - float
     - Major axis length in pixels
   * - ``major_axis_um``
     - float
     - Major axis length in micrometers
   * - ``minor_axis_px``
     - float
     - Minor axis length in pixels
   * - ``minor_axis_um``
     - float
     - Minor axis length in micrometers
   * - ``eccentricity``
     - float
     - Eccentricity (0=circle, 1=line)
   * - ``solidity``
     - float
     - Ratio of grain area to convex hull area
   * - ``extent``
     - float
     - Ratio of grain area to bounding box area
   * - ``orientation``
     - float
     - Orientation angle in radians

**Derived Metrics**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Metric
     - Type
     - Description
   * - ``ecd_um``
     - float
     - Equivalent circular diameter
   * - ``aspect_ratio``
     - float
     - Ratio of major to minor axis
   * - ``shape_factor``
     - float
     - Circularity measure (4πA/P²)
   * - ``compactness``
     - float
     - Inverse of shape factor
   * - ``elongation``
     - float
     - Same as aspect ratio
   * - ``roundness``
     - float
     - Same as shape factor
   * - ``convexity``
     - float
     - Ratio of area to convex area
   * - ``rectangularity``
     - float
     - Ratio of area to bounding box area

Error Handling
--------------

GrainStat defines custom exceptions for better error handling:

.. code-block:: python

   # Example error handling
   from grainstat import GrainAnalyzer

   try:
       analyzer = GrainAnalyzer()
       results = analyzer.analyze("nonexistent.tif")
   except FileNotFoundError:
       print("Image file not found")
   except ValueError as e:
       print(f"Invalid parameter: {e}")
   except Exception as e:
       print(f"Analysis failed: {e}")

Type Hints
----------

GrainStat uses type hints throughout the codebase. Common types:

.. code-block:: python

   from typing import Dict, List, Tuple, Optional, Any, Union
   import numpy as np

   # Common type aliases used in GrainStat
   GrainID = int
   GrainProperties = Dict[str, Any]
   GrainMetrics = Dict[GrainID, GrainProperties]
   Statistics = Dict[str, Any]
   ImageArray = np.ndarray
   Coordinate = Tuple[float, float]

Constants
---------

Mathematical constants used in calculations:

.. code-block:: python

   import math

   # Used in ECD calculation
   PI = math.pi  # 3.141592653589793

   # Used in ASTM grain size calculation
   ASTM_COEFFICIENT = -6.6438
   ASTM_OFFSET = -3.293

   # Size classification thresholds (micrometers)
   SIZE_THRESHOLDS = {
       'ultrafine': 1.0,
       'fine': 10.0,
       'medium': 50.0,
       'coarse': 100.0
   }

Configuration
-------------

Default analysis parameters:

.. code-block:: python

   DEFAULT_ANALYSIS_PARAMS = {
       'scale': 1.0,
       'min_area': 50,
       'gaussian_sigma': 1.0,
       'threshold_method': 'otsu',
       'use_watershed': True,
       'morphology_radius': 2,
       'use_clahe': True,
       'clahe_clip_limit': 0.01
   }

   DEFAULT_EXPORT_PARAMS = {
       'include_metadata': True,
       'decimal_places': 3,
       'date_format': '%Y-%m-%d %H:%M:%S'
   }

Version Information
-------------------

.. code-block:: python

   import grainstat

   print(f"Version: {grainstat.__version__}")
   print(f"Author: {grainstat.__author__}")

Examples
--------

See the :doc:`examples` page for comprehensive usage examples, or check the individual method documentation for specific examples.

For more advanced usage patterns, see:

- :doc:`tutorials/index` - Step-by-step tutorials
- :doc:`advanced` - Advanced features and customization
- :doc:`plugins` - Creating custom analysis features