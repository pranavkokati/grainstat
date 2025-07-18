Basic Analysis Tutorial
=======================

**Estimated time:** 30 minutes

**Learning objectives:**
- Perform your first grain size analysis
- Understand the basic GrainStat workflow
- Generate and interpret results
- Export data and create visualizations

Prerequisites
-------------

- GrainStat installed (:doc:`../installation`)
- A microstructure image (TIFF, PNG, or JPEG)
- Basic Python knowledge
- Image scale information (micrometers per pixel)

Getting Started
---------------

Let's start with a simple analysis of a microstructure image.

Step 1: Import GrainStat
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   import matplotlib.pyplot as plt

Step 2: Create an Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize the analyzer
   analyzer = GrainAnalyzer()
   print("GrainStat analyzer created successfully!")

Step 3: Prepare Your Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this tutorial, we'll create a synthetic microstructure:

.. code-block:: python

   import numpy as np
   from PIL import Image

   # Create a synthetic microstructure image
   def create_sample_microstructure():
       """Generate a synthetic grain structure for testing."""

       image = np.zeros((300, 300))
       np.random.seed(42)  # For reproducible results

       # Add circular grains
       grain_centers = []
       for i in range(25):
           # Random center position
           x = np.random.randint(30, 270)
           y = np.random.randint(30, 270)

           # Check minimum distance to existing grains
           too_close = False
           for cx, cy in grain_centers:
               if np.sqrt((x-cx)**2 + (y-cy)**2) < 20:
                   too_close = True
                   break

           if not too_close:
               grain_centers.append((x, y))

               # Random radius
               radius = np.random.randint(8, 18)

               # Create circular grain
               yy, xx = np.ogrid[:300, :300]
               mask = (xx - x)**2 + (yy - y)**2 <= radius**2
               image[mask] = 1.0

       # Add some noise
       noise = np.random.normal(0, 0.05, image.shape)
       image = np.clip(image + noise, 0, 1)

       return image

   # Generate and save the synthetic image
   synthetic_image = create_sample_microstructure()

   # Save as TIFF file
   img_pil = Image.fromarray((synthetic_image * 255).astype(np.uint8))
   img_pil.save('sample_microstructure.tif')

   print("Created sample_microstructure.tif")

.. note::
   In real applications, you would use your own microstructure images from optical microscopy, SEM, or other imaging techniques.

Step 4: Analyze the Image
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's perform the grain analysis:

.. code-block:: python

   # Analyze the microstructure
   results = analyzer.analyze(
       image_path='sample_microstructure.tif',
       scale=0.5,          # micrometers per pixel
       min_area=50         # minimum grain area in pixels
   )

   print("Analysis completed!")

**Understanding the parameters:**

- ``image_path``: Path to your microstructure image
- ``scale``: Conversion factor from pixels to micrometers
- ``min_area``: Minimum grain size in pixels (filters out noise)

Step 5: Examine the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's look at what the analysis produced:

.. code-block:: python

   # Get the statistics
   stats = results['statistics']

   print("=== ANALYSIS SUMMARY ===")
   print(f"Total grains detected: {stats['grain_count']}")

   # ECD (Equivalent Circular Diameter) statistics
   ecd_stats = stats['ecd_statistics']
   print(f"\nGrain Size Statistics (ECD):")
   print(f"  Mean:   {ecd_stats['mean']:.2f} μm")
   print(f"  Median: {ecd_stats['median']:.2f} μm")
   print(f"  Std:    {ecd_stats['std']:.2f} μm")
   print(f"  Range:  {ecd_stats['min']:.2f} - {ecd_stats['max']:.2f} μm")

Expected output:

.. code-block:: text

   === ANALYSIS SUMMARY ===
   Total grains detected: 23

   Grain Size Statistics (ECD):
     Mean:   8.45 μm
     Median: 8.12 μm
     Std:    2.31 μm
     Range:  4.23 - 13.67 μm

Step 6: Explore Individual Grain Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also examine individual grain properties:

.. code-block:: python

   # Get grain-by-grain data
   grain_metrics = results['metrics']

   print(f"\nFirst few grains:")
   print("-" * 50)

   for i, (grain_id, grain_data) in enumerate(list(grain_metrics.items())[:5]):
       print(f"Grain {grain_id}:")
       print(f"  ECD: {grain_data['ecd_um']:.2f} μm")
       print(f"  Area: {grain_data['area_um2']:.2f} μm²")
       print(f"  Aspect Ratio: {grain_data['aspect_ratio']:.2f}")
       print(f"  Shape Factor: {grain_data['shape_factor']:.3f}")
       print()

Step 7: Generate Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create plots to visualize your results:

.. code-block:: python

   # Create size distribution histogram
   analyzer.plot_histogram(bins=15, save_path='grain_histogram.png')
   print("Histogram saved as 'grain_histogram.png'")

   # Create overlay showing detected grains
   analyzer.plot_overlay(save_path='grain_overlay.png')
   print("Overlay saved as 'grain_overlay.png'")

   # Create cumulative distribution plot
   analyzer.plot_cumulative_distribution(save_path='grain_cdf.png')
   print("Cumulative distribution saved as 'grain_cdf.png'")

Step 8: Export Results
~~~~~~~~~~~~~~~~~~~~~~

Save your analysis results in various formats:

.. code-block:: python

   # Export grain data to CSV
   analyzer.export_csv('grain_data.csv')
   print("Grain data exported to 'grain_data.csv'")

   # Export complete analysis to JSON
   analyzer.export_json('complete_analysis.json')
   print("Complete analysis exported to 'complete_analysis.json'")

   # Generate HTML report
   analyzer.generate_report('analysis_report.html', format_type='html')
   print("HTML report generated: 'analysis_report.html'")

Understanding Your Results
--------------------------

Key Metrics Explained
~~~~~~~~~~~~~~~~~~~~~

**Grain Count**
   Total number of grains detected in the image.

**Equivalent Circular Diameter (ECD)**
   The diameter of a circle with the same area as the grain.

   .. math:: \text{ECD} = 2\sqrt{\frac{A}{\pi}}

**Aspect Ratio**
   Ratio of the major axis to minor axis length. Values close to 1 indicate equiaxed grains.

**Shape Factor**
   Measure of how circular a grain is. Perfect circles have a shape factor of 1.

   .. math:: \text{Shape Factor} = \frac{4\pi A}{P^2}

**ASTM Grain Size Number**
   Standardized grain size measurement (ASTM E112).

Interpreting the Histogram
~~~~~~~~~~~~~~~~~~~~~~~~~~

The size distribution histogram shows:

- **Peak position**: Most common grain size
- **Width**: Size distribution spread
- **Shape**: Whether distribution is normal, skewed, or bimodal
- **Tail behavior**: Presence of very large or small grains

Verifying Results
~~~~~~~~~~~~~~~~~

Always verify that the analysis worked correctly:

1. **Check the overlay image**: Do the detected boundaries match actual grains?
2. **Review grain count**: Does it seem reasonable for your image?
3. **Examine size distribution**: Does it match your visual impression?
4. **Look for artifacts**: Any obvious segmentation errors?

Common Issues and Solutions
---------------------------

No Grains Detected
~~~~~~~~~~~~~~~~~~~

**Symptoms**: Grain count is 0 or very low

**Solutions**:

.. code-block:: python

   # Try reducing minimum area
   results = analyzer.analyze(
       'sample_microstructure.tif',
       scale=0.5,
       min_area=20  # Reduced from 50
   )

   # Or try adaptive thresholding
   results = analyzer.analyze(
       'sample_microstructure.tif',
       scale=0.5,
       threshold_method='adaptive'
   )

Too Many Small Objects
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Very high grain count with many tiny objects

**Solutions**:

.. code-block:: python

   # Increase minimum area
   results = analyzer.analyze(
       'sample_microstructure.tif',
       scale=0.5,
       min_area=100  # Increased from 50
   )

   # Add more smoothing
   results = analyzer.analyze(
       'sample_microstructure.tif',
       scale=0.5,
       gaussian_sigma=2.0  # Increased from default 1.0
   )

Grains Not Separated
~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Large, irregular grains that should be multiple smaller grains

**Solutions**:

.. code-block:: python

   # Ensure watershed is enabled (default)
   results = analyzer.analyze(
       'sample_microstructure.tif',
       scale=0.5,
       use_watershed=True
   )

   # Try adaptive thresholding
   results = analyzer.analyze(
       'sample_microstructure.tif',
       scale=0.5,
       threshold_method='adaptive'
   )

Next Steps
----------

Now that you've completed your first analysis:

1. **Try with your own images**: Apply this workflow to your microstructure images
2. **Experiment with parameters**: See how different settings affect results
3. **Learn about batch processing**: :doc:`batch_processing`
4. **Understand parameters better**: :doc:`understanding_parameters`

Exercise
--------

Practice with this exercise:

1. Create three synthetic microstructures with different grain sizes
2. Analyze each one with the same parameters
3. Compare the results and create a summary table
4. Generate plots showing the size distributions

Solution outline:

.. code-block:: python

   # Create microstructures with different scales
   scales = [0.3, 0.5, 0.8]  # Different μm/pixel values

   results_comparison = {}

   for i, scale in enumerate(scales):
       # Create image (modify the radius range for variety)
       # Analyze with current scale
       # Store results
       pass

   # Compare mean grain sizes
   # Create comparison plots

Summary
-------

In this tutorial, you learned:

✅ How to set up and run a basic grain analysis
✅ The key parameters for analysis
✅ How to interpret and visualize results
✅ How to export data in multiple formats
✅ Common troubleshooting approaches

You're now ready to analyze your own microstructure images and explore more advanced features of GrainStat!

**Next recommended tutorial**: :doc:`understanding_parameters`