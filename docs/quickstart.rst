Quick Start Guide
=================

Get up and running with GrainStat in minutes! This guide covers the essential steps to analyze your first microstructure image.

Prerequisites
-------------

Before starting, ensure you have:

- Python 3.8 or higher installed
- A microstructure image (TIFF, PNG, JPEG, or BMP format)
- Knowledge of your image's spatial scale (micrometers per pixel)

If you haven't installed GrainStat yet, see the :doc:`installation` guide.

Your First Analysis
-------------------

Let's analyze a microstructure image step by step.

1. Import GrainStat
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer

2. Create an Analyzer
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   analyzer = GrainAnalyzer()

3. Analyze Your Image
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Replace with your image path and scale
   results = analyzer.analyze(
       image_path="microstructure.tif",
       scale=0.5,        # micrometers per pixel
       min_area=50       # minimum grain area in pixels
   )

4. View the Results
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Print summary statistics
   stats = results['statistics']
   print(f"Total grains: {stats['grain_count']}")
   print(f"Mean grain size: {stats['ecd_statistics']['mean']:.2f} μm")
   print(f"ASTM grain size: {stats['astm_grain_size']['grain_size_number']:.1f}")

Understanding the Scale Parameter
---------------------------------

The ``scale`` parameter is crucial for accurate measurements. It represents micrometers per pixel.

**How to determine scale:**

1. **From microscope settings**: Check the scale bar or imaging parameters
2. **From scale bar**: Measure a known scale bar in your image
3. **From magnification**: Calculate using the formula:

   .. math::

      \text{scale} = \frac{\text{pixel size (μm)}}{\text{magnification}}

**Common scales:**

- **Optical microscopy (100x)**: ~0.1 to 1.0 μm/pixel
- **SEM (1000x)**: ~0.01 to 0.1 μm/pixel
- **SEM (10000x)**: ~0.001 to 0.01 μm/pixel

.. note::
   Incorrect scale will lead to wrong size measurements! Always verify your scale parameter.

Basic Analysis Parameters
-------------------------

Key parameters for the ``analyze()`` method:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``image_path``
     - Required
     - Path to your microstructure image
   * - ``scale``
     - 1.0
     - Micrometers per pixel
   * - ``min_area``
     - 50
     - Minimum grain area in pixels
   * - ``gaussian_sigma``
     - 1.0
     - Smoothing strength (0 = no smoothing)
   * - ``threshold_method``
     - 'otsu'
     - Thresholding: 'otsu' or 'adaptive'
   * - ``use_watershed``
     - True
     - Separate touching grains
   * - ``morphology_radius``
     - 2
     - Morphological operation size

Example with custom parameters:

.. code-block:: python

   results = analyzer.analyze(
       image_path="noisy_image.tif",
       scale=0.2,
       min_area=100,           # Larger minimum size
       gaussian_sigma=2.0,     # More smoothing
       threshold_method='adaptive',  # Better for uneven lighting
       use_watershed=True,     # Separate touching grains
       morphology_radius=3     # Stronger cleaning
   )

Exporting Results
-----------------

Save your analysis results in various formats:

CSV Export
~~~~~~~~~~

.. code-block:: python

   # Export grain-by-grain data
   analyzer.export_csv("grain_data.csv")

The CSV includes all measurements for each grain:

- ``grain_id``: Unique identifier
- ``ecd_um``: Equivalent circular diameter
- ``area_um2``: Grain area
- ``aspect_ratio``: Major/minor axis ratio
- ``shape_factor``: Circularity measure
- And many more...

JSON Export
~~~~~~~~~~~

.. code-block:: python

   # Export complete analysis results
   analyzer.export_json("complete_analysis.json")

This includes all grain data, statistics, and metadata.

HTML Report
~~~~~~~~~~~

.. code-block:: python

   # Generate professional report
   analyzer.generate_report("analysis_report.html")

Creates a comprehensive report with:

- Summary statistics
- Embedded plots
- Grain data tables
- Analysis metadata

Generating Plots
----------------

Create publication-quality visualizations:

Size Distribution
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Histogram of grain sizes
   analyzer.plot_histogram(
       bins=30,
       save_path="size_distribution.png"
   )

Cumulative Distribution
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Cumulative size distribution
   analyzer.plot_cumulative_distribution(
       save_path="cumulative_plot.png"
   )

Grain Overlay
~~~~~~~~~~~~~

.. code-block:: python

   # Original image with grain boundaries
   analyzer.plot_overlay(
       save_path="grain_overlay.png"
   )

All plots are automatically saved with high resolution (300 DPI) suitable for publications.

Understanding the Results
-------------------------

Key Statistics
~~~~~~~~~~~~~~

GrainStat calculates comprehensive statistics:

**Grain Count**
   Total number of detected grains

**ECD Statistics**
   Equivalent Circular Diameter measurements:
   - Mean, median, standard deviation
   - Percentiles (5th, 25th, 75th, 95th)
   - Min, max, range

**ASTM Grain Size**
   Standardized grain size number (ASTM E112)

**Shape Analysis**
   - Aspect ratio distribution
   - Shape factor (circularity)
   - Eccentricity measurements

**Size Classification**
   Grain count by size category:
   - Ultrafine (< 1 μm)
   - Fine (1-10 μm)
   - Medium (10-50 μm)
   - Coarse (50-100 μm)
   - Very coarse (> 100 μm)

Example Output
~~~~~~~~~~~~~~

.. code-block:: text

   GRAIN ANALYSIS SUMMARY
   ======================
   Total grains detected: 1,247

   Grain Size Statistics (ECD):
     Mean:   12.3 μm
     Median: 10.8 μm
     Std:    4.7 μm
     Range:  2.1 - 45.6 μm

   ASTM Grain Size Number: 8.2

   Size Class Distribution:
     Fine: 423 grains (33.9%)
     Medium: 681 grains (54.6%)
     Coarse: 143 grains (11.5%)

Common Workflows
----------------

Quality Control
~~~~~~~~~~~~~~~

For routine quality control:

.. code-block:: python

   from grainstat import GrainAnalyzer

   def analyze_sample(image_path, sample_id):
       analyzer = GrainAnalyzer()
       results = analyzer.analyze(image_path, scale=0.3)

       # Extract key metrics
       stats = results['statistics']
       mean_size = stats['ecd_statistics']['mean']
       astm_size = stats['astm_grain_size']['grain_size_number']

       # Save results
       analyzer.export_csv(f"{sample_id}_grains.csv")

       return {
           'sample_id': sample_id,
           'grain_count': stats['grain_count'],
           'mean_ecd': mean_size,
           'astm_grain_size': astm_size
       }

   # Process multiple samples
   samples = ['sample_001.tif', 'sample_002.tif', 'sample_003.tif']
   results = [analyze_sample(img, f"S{i:03d}") for i, img in enumerate(samples, 1)]

Research Analysis
~~~~~~~~~~~~~~~~~

For detailed research analysis:

.. code-block:: python

   analyzer = GrainAnalyzer()

   # Analyze with custom parameters
   results = analyzer.analyze(
       "research_sample.tif",
       scale=0.1,              # High resolution SEM
       min_area=20,            # Detect small grains
       gaussian_sigma=0.5,     # Light smoothing
       use_watershed=True      # Separate touching grains
   )

   # Generate comprehensive outputs
   analyzer.export_csv("detailed_grains.csv")
   analyzer.export_json("complete_data.json")
   analyzer.generate_report("research_report.html")

   # Create all visualization types
   analyzer.plot_histogram(save_path="histogram.png")
   analyzer.plot_cumulative_distribution(save_path="cdf.png")
   analyzer.plot_overlay(save_path="overlay.png")

Command Line Usage
------------------

For automation and scripting, use the command line interface:

Single Image
~~~~~~~~~~~~

.. code-block:: bash

   grainstat analyze microstructure.tif \
       --scale 0.5 \
       --min-area 50 \
       --export-csv results.csv \
       --report report.html

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   grainstat batch input_folder/ output_folder/ \
       --scale 0.3 \
       --pattern "*.tif" \
       --workers 4

Interactive Viewer
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   grainstat interactive microstructure.tif --scale 0.5

This launches an interactive viewer where you can:

- Click on grains to see their properties
- Zoom and pan the image
- Export individual grain data

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**No grains detected**
   - Check your scale parameter
   - Adjust ``min_area`` (try smaller values)
   - Try different ``threshold_method``
   - Increase ``gaussian_sigma`` for noisy images

**Too many small objects**
   - Increase ``min_area``
   - Increase ``morphology_radius``
   - Use stronger smoothing (higher ``gaussian_sigma``)

**Grains not separated**
   - Ensure ``use_watershed=True``
   - Try ``threshold_method='adaptive'``
   - Adjust image contrast before analysis

**Memory errors**
   - Reduce image size
   - Increase ``min_area`` to detect fewer objects
   - Process smaller image regions

Best Practices
~~~~~~~~~~~~~~

1. **Validate your scale**: Double-check the micrometers per pixel value
2. **Start with defaults**: Use default parameters first, then optimize
3. **Check segmentation**: View the overlay plot to verify grain detection
4. **Document parameters**: Save analysis parameters with results
5. **Use consistent settings**: Keep parameters consistent across samples

Next Steps
----------

Now that you've completed your first analysis:

1. **Explore advanced features**: :doc:`tutorials/index`
2. **Learn batch processing**: :doc:`tutorials/batch_processing`
3. **Create custom features**: :doc:`plugins`
4. **See more examples**: :doc:`examples`
5. **Read the full API**: :doc:`api`

.. tip::
   The interactive viewer (``grainstat interactive``) is great for exploring your data and understanding how different parameters affect segmentation.