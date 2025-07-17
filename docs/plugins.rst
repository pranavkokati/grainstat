Plugin System
=============

GrainStat's plugin system allows you to create custom grain features and extend the analysis capabilities for specialized applications.

Overview
--------

The plugin system enables you to:

- Define custom grain measurements
- Create specialized shape descriptors
- Implement domain-specific analysis features
- Extend GrainStat without modifying core code

Features are automatically integrated into the analysis workflow and included in all export formats.

Creating Custom Features
-------------------------

Basic Feature Decorator
~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``@feature`` decorator to create custom grain features:

.. code-block:: python

   from grainstat import feature

   @feature
   def custom_roundness(region):
       """Calculate a custom roundness metric."""
       area = region.area
       perimeter = region.perimeter

       if perimeter == 0:
           return 0

       return (4 * 3.14159 * area) / (perimeter ** 2)

Named Features
~~~~~~~~~~~~~~

Specify a custom name for your feature:

.. code-block:: python

   @feature(name="grain_complexity")
   def calculate_complexity(region):
       """Calculate grain complexity index."""
       eccentricity = region.eccentricity
       solidity = region.solidity

       # Custom complexity metric
       complexity = eccentricity * (2 - solidity)
       return complexity

Multiple Features from One Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a dictionary to define multiple features:

.. code-block:: python

   @feature(name="advanced_shape_metrics")
   def calculate_advanced_metrics(region):
       """Calculate multiple advanced shape metrics."""

       area = region.area
       perimeter = region.perimeter
       major_axis = region.major_axis_length
       minor_axis = region.minor_axis_length

       # Calculate multiple metrics
       shape_regularity = (2 * (area / 3.14159)**0.5 * 3.14159) / perimeter
       elongation_index = 1 - (minor_axis / major_axis) if major_axis > 0 else 0
       surface_roughness = perimeter / (2 * 3.14159 * (area / 3.14159)**0.5) - 1

       return {
           'shape_regularity': shape_regularity,
           'elongation_index': elongation_index,
           'surface_roughness': surface_roughness
       }

Available Region Properties
---------------------------

The ``region`` parameter provides access to all standard grain properties:

Geometric Properties
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @feature
   def example_feature(region):
       # Basic measurements
       area = region.area                    # Area in pixels
       perimeter = region.perimeter          # Perimeter in pixels

       # Size measurements
       major_axis = region.major_axis_length # Major axis length
       minor_axis = region.minor_axis_length # Minor axis length
       equivalent_diameter = region.equivalent_diameter

       # Position
       centroid = region.centroid           # (y, x) coordinates
       bbox = region.bbox                   # Bounding box

       # Shape descriptors
       eccentricity = region.eccentricity   # 0=circle, 1=line
       solidity = region.solidity           # Area/convex_area
       extent = region.extent               # Area/bbox_area
       orientation = region.orientation      # Orientation angle

       return area / perimeter

Physical Unit Properties
~~~~~~~~~~~~~~~~~~~~~~~~

For properties already converted to physical units:

.. code-block:: python

   @feature
   def physical_units_example(region):
       # Physical measurements (micrometers)
       area_um2 = region.area_um2
       perimeter_um = region.perimeter_um
       major_axis_um = region.major_axis_um
       minor_axis_um = region.minor_axis_um
       centroid_um = region.centroid_um     # (y, x) in micrometers

       # Derived measurements
       ecd_um = region.ecd_um
       aspect_ratio = region.aspect_ratio

       return area_um2 / (perimeter_um ** 2)

Example Custom Features
-----------------------

Materials Science Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Domain-specific features for materials characterization:

.. code-block:: python

   @feature
   def grain_size_category(region):
       """Classify grains by ASTM size categories."""
       ecd = region.ecd_um

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
       """Calculate crystallographic texture strength."""
       orientation = abs(region.orientation)
       eccentricity = region.eccentricity

       # Normalize orientation to 0-1 range
       normalized_orientation = orientation / (3.14159 / 2)

       # Combine orientation preference with elongation
       texture_strength = normalized_orientation * eccentricity
       return texture_strength

   @feature(name="anisotropy_metrics")
   def calculate_anisotropy(region):
       """Calculate grain anisotropy metrics."""

       major_axis = region.major_axis_um
       minor_axis = region.minor_axis_um
       area = region.area_um2

       # Anisotropy ratio
       anisotropy_ratio = major_axis / minor_axis if minor_axis > 0 else 0

       # Shape anisotropy
       expected_minor = (area / 3.14159) ** 0.5 * 2  # Expected for circle
       shape_anisotropy = abs(minor_axis - expected_minor) / expected_minor

       return {
           'anisotropy_ratio': anisotropy_ratio,
           'shape_anisotropy': shape_anisotropy
       }

Statistical Features
~~~~~~~~~~~~~~~~~~~~

Features based on statistical properties:

.. code-block:: python

   @feature
   def local_size_deviation(region):
       """Calculate deviation from local mean size."""
       # This would require neighborhood information
       # For now, return a placeholder
       return region.ecd_um / 10.0  # Simplified example

   @feature
   def shape_moments(region):
       """Calculate shape-based moment features."""
       try:
           moments_central = region.moments_central

           if moments_central is not None:
               # Calculate moment-based features
               mu20 = moments_central[2, 0]
               mu02 = moments_central[0, 2]
               mu11 = moments_central[1, 1]
               mu00 = moments_central[0, 0]

               if mu00 > 0:
                   # Normalized central moments
                   eta20 = mu20 / (mu00 ** 2)
                   eta02 = mu02 / (mu00 ** 2)
                   eta11 = mu11 / (mu00 ** 2)

                   # Moment-based shape descriptor
                   moment_descriptor = eta20 + eta02

                   return moment_descriptor

           return 0

       except Exception:
           return 0

Industrial QC Features
~~~~~~~~~~~~~~~~~~~~~~

Features for quality control applications:

.. code-block:: python

   @feature
   def qc_pass_fail(region):
       """Quality control pass/fail based on specifications."""

       ecd = region.ecd_um
       aspect_ratio = region.aspect_ratio
       shape_factor = region.shape_factor

       # Define specifications
       size_ok = 5.0 <= ecd <= 50.0
       shape_ok = aspect_ratio <= 2.0
       roundness_ok = shape_factor >= 0.7

       if size_ok and shape_ok and roundness_ok:
           return 'PASS'
       else:
           return 'FAIL'

   @feature(name="process_indicators")
   def calculate_process_indicators(region):
       """Calculate indicators for process control."""

       ecd = region.ecd_um
       aspect_ratio = region.aspect_ratio
       solidity = region.solidity

       # Cooling rate indicator (grain size related)
       cooling_rate_indicator = 1 / ecd if ecd > 0 else 0

       # Deformation indicator (elongation related)
       deformation_indicator = aspect_ratio - 1

       # Recrystallization indicator (regularity related)
       recryst_indicator = solidity * (2 - aspect_ratio)

       return {
           'cooling_rate_indicator': cooling_rate_indicator,
           'deformation_indicator': deformation_indicator,
           'recrystallization_indicator': recryst_indicator
       }

Dynamic Feature Creation
------------------------

Create features programmatically for automated analysis:

Ratio Features
~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat.plugins.base import create_ratio_feature

   # Create area-to-perimeter ratio feature
   area_perimeter_ratio = create_ratio_feature(
       'area_um2', 'perimeter_um', 'area_perimeter_ratio'
   )

   # Create major-to-minor axis ratio
   axis_ratio = create_ratio_feature(
       'major_axis_um', 'minor_axis_um', 'axis_ratio'
   )

Classification Features
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat.plugins.base import create_classification_feature

   # Size classification
   size_classifier = create_classification_feature(
       attribute='ecd_um',
       thresholds=[1, 10, 50, 100],
       labels=['ultrafine', 'fine', 'medium', 'coarse', 'very_coarse'],
       feature_name='size_class'
   )

   # Shape classification
   shape_classifier = create_classification_feature(
       attribute='aspect_ratio',
       thresholds=[1.2, 1.5, 2.0],
       labels=['equiaxed', 'slightly_elongated', 'elongated', 'very_elongated'],
       feature_name='shape_class'
   )

Batch Feature Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Register multiple features at once:

.. code-block:: python

   def register_steel_analysis_features():
       """Register features specific to steel analysis."""

       # Size-based features
       create_classification_feature(
           'ecd_um', [5, 15, 30],
           ['fine', 'medium', 'coarse'], 'steel_size_class'
       )

       # Shape-based features
       create_ratio_feature('major_axis_um', 'minor_axis_um', 'elongation_ratio')

       # Combined features
       @feature(name="steel_quality_index")
       def steel_quality(region):
           size_score = 1.0 if 10 <= region.ecd_um <= 25 else 0.5
           shape_score = 1.0 if region.aspect_ratio <= 1.5 else 0.5
           return size_score * shape_score

   # Call to register all steel features
   register_steel_analysis_features()

Using Custom Features
---------------------

Once defined, custom features are automatically included in analysis:

.. code-block:: python

   from grainstat import GrainAnalyzer

   # Define custom features (shown above)
   @feature
   def my_custom_metric(region):
       return region.area_um2 / region.perimeter_um

   # Run analysis - custom features are automatically included
   analyzer = GrainAnalyzer()
   results = analyzer.analyze('sample.tif', scale=0.5)

   # Access custom features in results
   grain_metrics = results['metrics']

   for grain_id, grain_data in grain_metrics.items():
       custom_value = grain_data['my_custom_metric']
       print(f"Grain {grain_id}: custom metric = {custom_value:.3f}")

Export with Custom Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom features are included in all export formats:

.. code-block:: python

   # CSV export includes custom features
   analyzer.export_csv('grains_with_custom_features.csv')

   # JSON export includes custom features
   analyzer.export_json('complete_analysis_with_custom.json')

   # HTML report includes custom features
   analyzer.generate_report('report_with_custom_features.html')

Advanced Plugin Patterns
-------------------------

Conditional Features
~~~~~~~~~~~~~~~~~~~~

Create features that depend on grain properties:

.. code-block:: python

   @feature
   def adaptive_feature(region):
       """Feature that adapts based on grain characteristics."""

       ecd = region.ecd_um

       if ecd < 5:
           # Fine grain feature
           return region.solidity * 2
       elif ecd > 50:
           # Coarse grain feature
           return region.aspect_ratio * region.eccentricity
       else:
           # Medium grain feature
           return region.shape_factor

External Data Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate external data sources:

.. code-block:: python

   # Example: Load composition data
   composition_data = {
       'sample_1': {'mg_content': 2.0, 'si_content': 0.5},
       'sample_2': {'mg_content': 4.0, 'si_content': 0.8}
   }

   @feature
   def composition_corrected_size(region):
       """Size metric corrected for composition effects."""

       # This would need sample identification mechanism
       # For demonstration, use a fixed composition
       mg_content = 2.0  # Would be dynamically determined

       # Apply composition correction
       correction_factor = 1