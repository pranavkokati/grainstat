Examples
========

This page provides practical examples of using GrainStat for various materials science applications.

Basic Analysis
--------------

Simple Grain Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   import matplotlib.pyplot as plt

   # Initialize analyzer
   analyzer = GrainAnalyzer()

   # Analyze microstructure
   results = analyzer.analyze(
       image_path="steel_microstructure.tif",
       scale=0.2,  # μm per pixel
       min_area=100
   )

   # Print summary
   stats = results['statistics']
   print(f"Total grains: {stats['grain_count']}")
   print(f"Mean grain size: {stats['ecd_statistics']['mean']:.1f} μm")

   # Export results
   analyzer.export_csv("steel_grains.csv")
   analyzer.plot_histogram(save_path="steel_histogram.png")

Optical Microscopy Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer

   def analyze_optical_microscopy(image_path, magnification=100):
       """Analyze optical microscopy image with standard parameters."""

       # Typical scale for 100x optical microscopy
       scale = 1.0  # Adjust based on your microscope

       analyzer = GrainAnalyzer()

       # Optical microscopy often needs different parameters
       results = analyzer.analyze(
           image_path=image_path,
           scale=scale,
           min_area=50,              # Smaller minimum area
           gaussian_sigma=1.5,       # More smoothing for optical images
           threshold_method='otsu',   # Good for etched samples
           use_watershed=True,       # Separate touching grains
           morphology_radius=2
       )

       return results

   # Process optical microscopy image
   results = analyze_optical_microscopy("optical_sample.jpg")

SEM Analysis
~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer

   def analyze_sem_image(image_path, magnification=1000):
       """Analyze SEM image with appropriate parameters."""

       # Calculate scale based on SEM magnification
       # Adjust pixel_size_mm based on your SEM detector
       pixel_size_mm = 0.01  # Example: 10 μm pixel size
       scale = (pixel_size_mm * 1000) / magnification  # Convert to μm/pixel

       analyzer = GrainAnalyzer()

       results = analyzer.analyze(
           image_path=image_path,
           scale=scale,
           min_area=20,              # Smaller grains in SEM
           gaussian_sigma=0.8,       # Less smoothing for clean SEM images
           threshold_method='otsu',
           use_watershed=True,
           morphology_radius=1       # Smaller morphology operations
       )

       return results

   # Process SEM image
   results = analyze_sem_image("sem_sample.tif", magnification=2000)

Comparative Analysis
--------------------

Heat Treatment Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   import pandas as pd
   import matplotlib.pyplot as plt

   def compare_heat_treatments():
       """Compare grain sizes before and after heat treatment."""

       conditions = {
           'as_received': 'as_received_sample.tif',
           'annealed_400C': 'annealed_400C_sample.tif',
           'annealed_600C': 'annealed_600C_sample.tif'
       }

       results = {}
       analyzer = GrainAnalyzer()

       for condition, image_path in conditions.items():
           print(f"Analyzing {condition}...")

           result = analyzer.analyze(
               image_path=image_path,
               scale=0.3,  # μm per pixel
               min_area=50
           )

           stats = result['statistics']
           results[condition] = {
               'grain_count': stats['grain_count'],
               'mean_ecd': stats['ecd_statistics']['mean'],
               'median_ecd': stats['ecd_statistics']['median'],
               'std_ecd': stats['ecd_statistics']['std'],
               'astm_grain_size': stats['astm_grain_size']['grain_size_number']
           }

           # Save individual results
           analyzer.export_csv(f"{condition}_grains.csv")

       # Create comparison DataFrame
       df = pd.DataFrame(results).T
       print("\nComparison Results:")
       print(df)

       # Plot comparison
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

       # Mean grain size comparison
       df['mean_ecd'].plot(kind='bar', ax=ax1)
       ax1.set_title('Mean Grain Size by Condition')
       ax1.set_ylabel('ECD (μm)')
       ax1.tick_params(axis='x', rotation=45)

       # ASTM grain size comparison
       df['astm_grain_size'].plot(kind='bar', ax=ax2)
       ax2.set_title('ASTM Grain Size Number')
       ax2.set_ylabel('ASTM G#')
       ax2.tick_params(axis='x', rotation=45)

       plt.tight_layout()
       plt.savefig('heat_treatment_comparison.png', dpi=300)
       plt.show()

       return results

   # Run comparison
   comparison_results = compare_heat_treatments()

Alloy Composition Study
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt

   def analyze_alloy_series():
       """Analyze grain size vs. alloy composition."""

       alloys = {
           'Al_0Mg': ('al_0mg.tif', 0.0),
           'Al_2Mg': ('al_2mg.tif', 2.0),
           'Al_4Mg': ('al_4mg.tif', 4.0),
           'Al_6Mg': ('al_6mg.tif', 6.0),
           'Al_8Mg': ('al_8mg.tif', 8.0)
       }

       data = []
       analyzer = GrainAnalyzer()

       for alloy_name, (image_path, mg_content) in alloys.items():
           result = analyzer.analyze(
               image_path=image_path,
               scale=0.5,
               min_area=30
           )

           # Extract ECD values for all grains
           ecd_values = [grain['ecd_um'] for grain in result['metrics'].values()]

           # Add to dataset
           for ecd in ecd_values:
               data.append({
                   'alloy': alloy_name,
                   'mg_content': mg_content,
                   'ecd_um': ecd
               })

       # Create DataFrame
       df = pd.DataFrame(data)

       # Statistical analysis
       summary = df.groupby('mg_content')['ecd_um'].agg(['mean', 'std', 'count'])
       print("Summary by Mg Content:")
       print(summary)

       # Visualization
       plt.figure(figsize=(12, 8))

       # Box plot
       plt.subplot(2, 2, 1)
       sns.boxplot(data=df, x='mg_content', y='ecd_um')
       plt.title('Grain Size Distribution by Mg Content')
       plt.xlabel('Mg Content (wt%)')
       plt.ylabel('ECD (μm)')

       # Mean grain size trend
       plt.subplot(2, 2, 2)
       plt.plot(summary.index, summary['mean'], 'o-')
       plt.errorbar(summary.index, summary['mean'], yerr=summary['std'], capsize=5)
       plt.title('Mean Grain Size vs. Mg Content')
       plt.xlabel('Mg Content (wt%)')
       plt.ylabel('Mean ECD (μm)')
       plt.grid(True)

       # Histogram overlay
       plt.subplot(2, 1, 2)
       for mg in sorted(df['mg_content'].unique()):
           subset = df[df['mg_content'] == mg]['ecd_um']
           plt.hist(subset, alpha=0.6, label=f'{mg}% Mg', bins=20)
       plt.xlabel('ECD (μm)')
       plt.ylabel('Frequency')
       plt.title('Grain Size Distributions')
       plt.legend()

       plt.tight_layout()
       plt.savefig('alloy_composition_study.png', dpi=300)
       plt.show()

       return df

   # Run alloy study
   alloy_data = analyze_alloy_series()

Batch Processing
----------------

Quality Control Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat.processing.batch import BatchProcessor
   import pandas as pd
   from pathlib import Path

   def quality_control_analysis(input_dir, output_dir, specification_limits):
       """Automated quality control for production samples."""

       processor = BatchProcessor(n_workers=4)

       # Process all TIFF images in directory
       batch_results = processor.process_directory(
           input_dir=input_dir,
           output_dir=output_dir,
           pattern="*.tif",
           scale=0.25,  # Adjust for your imaging setup
           analysis_params={
               'min_area': 100,
               'gaussian_sigma': 1.0,
               'threshold_method': 'otsu',
               'use_watershed': True
           }
       )

       # Extract quality metrics
       qc_data = []

       for result in batch_results['results']:
           if result['success']:
               stats = result['statistics']

               image_name = Path(result['image_path']).stem
               mean_grain_size = stats['ecd_statistics']['mean']
               astm_grain_size = stats['astm_grain_size']['grain_size_number']
               grain_count = stats['grain_count']

               # Check against specifications
               size_ok = (specification_limits['min_ecd'] <= mean_grain_size <=
                         specification_limits['max_ecd'])
               count_ok = grain_count >= specification_limits['min_grain_count']

               qc_data.append({
                   'sample': image_name,
                   'mean_grain_size': mean_grain_size,
                   'astm_grain_size': astm_grain_size,
                   'grain_count': grain_count,
                   'size_spec_ok': size_ok,
                   'count_spec_ok': count_ok,
                   'overall_pass': size_ok and count_ok
               })

       # Create QC report
       qc_df = pd.DataFrame(qc_data)
       qc_df.to_csv(Path(output_dir) / 'qc_report.csv', index=False)

       # Summary statistics
       pass_rate = qc_df['overall_pass'].mean() * 100

       print(f"Quality Control Summary:")
       print(f"Total samples: {len(qc_df)}")
       print(f"Pass rate: {pass_rate:.1f}%")
       print(f"Failed samples: {(~qc_df['overall_pass']).sum()}")

       # List failed samples
       failed_samples = qc_df[~qc_df['overall_pass']]['sample'].tolist()
       if failed_samples:
           print(f"Failed samples: {', '.join(failed_samples)}")

       return qc_df

   # Define specifications
   specs = {
       'min_ecd': 5.0,      # μm
       'max_ecd': 25.0,     # μm
       'min_grain_count': 100
   }

   # Run QC analysis
   qc_results = quality_control_analysis(
       input_dir="production_samples/",
       output_dir="qc_results/",
       specification_limits=specs
   )

Research Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat.processing.batch import BatchProcessor
   from grainstat import GrainAnalyzer
   import pandas as pd
   import json

   def process_research_dataset(dataset_info):
       """Process a research dataset with multiple conditions and replicates."""

       all_results = []
       processor = BatchProcessor()

       for condition, sample_info in dataset_info.items():
           print(f"Processing condition: {condition}")

           condition_results = processor.process_directory(
               input_dir=sample_info['path'],
               output_dir=f"results/{condition}/",
               scale=sample_info['scale'],
               analysis_params=sample_info.get('params', {})
           )

           # Extract data for each replicate
           for result in condition_results['results']:
               if result['success']:
                   stats = result['statistics']

                   replicate_data = {
                       'condition': condition,
                       'replicate': Path(result['image_path']).stem,
                       'temperature': sample_info.get('temperature'),
                       'time_hours': sample_info.get('time_hours'),
                       'mean_ecd': stats['ecd_statistics']['mean'],
                       'median_ecd': stats['ecd_statistics']['median'],
                       'std_ecd': stats['ecd_statistics']['std'],
                       'grain_count': stats['grain_count'],
                       'astm_grain_size': stats['astm_grain_size']['grain_size_number']
                   }

                   all_results.append(replicate_data)

       # Create comprehensive dataset
       df = pd.DataFrame(all_results)
       df.to_csv('research_dataset.csv', index=False)

       # Statistical summary by condition
       summary = df.groupby('condition').agg({
           'mean_ecd': ['mean', 'std', 'count'],
           'astm_grain_size': ['mean', 'std']
       }).round(2)

       print("\nStatistical Summary:")
       print(summary)

       return df

   # Define research dataset
   dataset = {
       'control': {
           'path': 'samples/control/',
           'scale': 0.3,
           'temperature': 20,
           'time_hours': 0
       },
       'heat_treatment_1': {
           'path': 'samples/ht1/',
           'scale': 0.3,
           'temperature': 400,
           'time_hours': 2,
           'params': {'gaussian_sigma': 1.5}  # Custom parameters
       },
       'heat_treatment_2': {
           'path': 'samples/ht2/',
           'scale': 0.3,
           'temperature': 600,
           'time_hours': 4,
           'params': {'gaussian_sigma': 1.5}
       }
   }

   # Process dataset
   research_data = process_research_dataset(dataset)

Advanced Features
-----------------

Custom Feature Development
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer, feature
   import numpy as np

   # Define custom features
   @feature
   def grain_elongation_index(region):
       """Calculate a custom elongation index."""
       aspect_ratio = region.aspect_ratio
       eccentricity = region.eccentricity

       # Custom formula combining aspect ratio and eccentricity
       elongation_index = (aspect_ratio - 1) * eccentricity
       return elongation_index

   @feature(name="texture_parameter")
   def texture_strength(region):
       """Calculate texture strength based on orientation and shape."""
       orientation = abs(region.orientation)
       eccentricity = region.eccentricity

       # Normalize orientation to 0-1 range
       normalized_orientation = orientation / (np.pi / 2)

       # Combine orientation preference with elongation
       texture_param = normalized_orientation * eccentricity
       return texture_param

   @feature
   def size_classification(region):
       """Classify grains by size."""
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

   # Use custom features in analysis
   analyzer = GrainAnalyzer()
   results = analyzer.analyze("sample.tif", scale=0.5)

   # Custom features are automatically included in results
   grain_metrics = results['metrics']

   # Analyze custom features
   elongation_values = [grain['grain_elongation_index']
                       for grain in grain_metrics.values()]

   print(f"Mean elongation index: {np.mean(elongation_values):.3f}")

Interactive Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   from grainstat.visualization.interactive import InteractiveViewer

   def interactive_grain_analysis(image_path):
       """Launch interactive analysis session."""

       # Perform initial analysis
       analyzer = GrainAnalyzer()
       results = analyzer.analyze(image_path, scale=0.4)

       print(f"Detected {results['statistics']['grain_count']} grains")
       print("Launching interactive viewer...")
       print("Instructions:")
       print("- Click on grains to see properties")
       print("- Press 'h' for help")
       print("- Press 'c' to clear highlights")
       print("- Press 's' to save grain info")

       # Launch interactive viewer
       viewer = InteractiveViewer(
           analyzer.image,
           analyzer.labeled_image,
           analyzer.grain_metrics
       )

       viewer.show_interactive()

       return results

   # Launch interactive session
   interactive_results = interactive_grain_analysis("complex_microstructure.tif")

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   from scipy import stats
   import numpy as np
   import matplotlib.pyplot as plt

   def detailed_statistical_analysis(image_path, scale):
       """Perform detailed statistical analysis of grain size distribution."""

       analyzer = GrainAnalyzer()
       results = analyzer.analyze(image_path, scale=scale)

       # Extract ECD values
       ecd_values = np.array([grain['ecd_um']
                             for grain in results['metrics'].values()])

       # Test different distributions
       distributions = {
           'normal': stats.norm,
           'lognormal': stats.lognorm,
           'gamma': stats.gamma,
           'weibull': stats.weibull_min
       }

       fit_results = {}

       for name, distribution in distributions.items():
           try:
               # Fit distribution
               params = distribution.fit(ecd_values)

               # Kolmogorov-Smirnov test
               ks_stat, p_value = stats.kstest(ecd_values,
                                              lambda x: distribution.cdf(x, *params))

               # Calculate AIC
               log_likelihood = np.sum(distribution.logpdf(ecd_values, *params))
               aic = -2 * log_likelihood + 2 * len(params)

               fit_results[name] = {
                   'parameters': params,
                   'ks_statistic': ks_stat,
                   'p_value': p_value,
                   'aic': aic
               }

           except Exception as e:
               print(f"Failed to fit {name}: {e}")

       # Find best fit
       best_fit = min(fit_results.items(), key=lambda x: x[1]['aic'])

       print("Distribution Fitting Results:")
       print("-" * 40)
       for name, result in fit_results.items():
           print(f"{name:10}: AIC={result['aic']:.1f}, p={result['p_value']:.3f}")

       print(f"\nBest fit: {best_fit[0]}")

       # Plot results
       fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

       # Histogram with fitted distributions
       ax1.hist(ecd_values, bins=30, density=True, alpha=0.7, label='Data')

       x = np.linspace(ecd_values.min(), ecd_values.max(), 100)
       for name, result in fit_results.items():
           distribution = distributions[name]
           params = result['parameters']
           pdf = distribution.pdf(x, *params)
           ax1.plot(x, pdf, label=f"{name} (AIC={result['aic']:.1f})")

       ax1.set_xlabel('ECD (μm)')
       ax1.set_ylabel('Density')
       ax1.set_title('Distribution Fitting')
       ax1.legend()

       # Q-Q plot for best fit
       distribution = distributions[best_fit[0]]
       params = best_fit[1]['parameters']
       stats.probplot(ecd_values, dist=lambda x: distribution.ppf(x, *params), plot=ax2)
       ax2.set_title(f'Q-Q Plot: {best_fit[0]}')

       # Box plot
       ax3.boxplot(ecd_values)
       ax3.set_ylabel('ECD (μm)')
       ax3.set_title('Box Plot')

       # Cumulative distribution
       sorted_values = np.sort(ecd_values)
       cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
       ax4.plot(sorted_values, cumulative, label='Empirical')

       cdf = distribution.cdf(sorted_values, *params)
       ax4.plot(sorted_values, cdf, label=f'{best_fit[0]} fit')
       ax4.set_xlabel('ECD (μm)')
       ax4.set_ylabel('Cumulative Probability')
       ax4.set_title('Cumulative Distribution')
       ax4.legend()

       plt.tight_layout()
       plt.savefig('statistical_analysis.png', dpi=300)
       plt.show()

       return fit_results

   # Run detailed statistical analysis
   stats_results = detailed_statistical_analysis("sample.tif", scale=0.3)

Specialized Applications
------------------------

Dual-Phase Steel Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   import numpy as np
   from skimage import filters, morphology

   def analyze_dual_phase_steel(image_path, scale):
       """Specialized analysis for dual-phase steel microstructures."""

       analyzer = GrainAnalyzer()

       # Use adaptive thresholding for dual-phase structures
       results = analyzer.analyze(
           image_path=image_path,
           scale=scale,
           min_area=30,
           threshold_method='adaptive',  # Better for dual-phase
           gaussian_sigma=1.2,
           use_watershed=True,
           morphology_radius=1
       )

       # Separate phases based on size
       grain_metrics = results['metrics']

       # Assume smaller grains are martensite, larger are ferrite
       ecd_values = [grain['ecd_um'] for grain in grain_metrics.values()]
       threshold_ecd = np.percentile(ecd_values, 50)  # Use median as threshold

       ferrite_grains = []
       martensite_grains = []

       for grain_id, grain in grain_metrics.items():
           if grain['ecd_um'] > threshold_ecd:
               ferrite_grains.append(grain)
           else:
               martensite_grains.append(grain)

       # Calculate phase statistics
       ferrite_stats = {
           'count': len(ferrite_grains),
           'mean_ecd': np.mean([g['ecd_um'] for g in ferrite_grains]),
           'fraction': len(ferrite_grains) / len(grain_metrics)
       }

       martensite_stats = {
           'count': len(martensite_grains),
           'mean_ecd': np.mean([g['ecd_um'] for g in martensite_grains]),
           'fraction': len(martensite_grains) / len(grain_metrics)
       }

       print("Dual-Phase Steel Analysis:")
       print(f"Ferrite: {ferrite_stats['count']} grains, "
             f"mean size {ferrite_stats['mean_ecd']:.1f} μm, "
             f"fraction {ferrite_stats['fraction']:.1%}")
       print(f"Martensite: {martensite_stats['count']} grains, "
             f"mean size {martensite_stats['mean_ecd']:.1f} μm, "
             f"fraction {martensite_stats['fraction']:.1%}")

       return {
           'overall': results,
           'ferrite': ferrite_stats,
           'martensite': martensite_stats
       }

   # Analyze dual-phase steel
   dp_results = analyze_dual_phase_steel("dual_phase_steel.tif", scale=0.15)

Recrystallization Study
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grainstat import GrainAnalyzer
   import matplotlib.pyplot as plt
   import numpy as np

   def recrystallization_kinetics_study(time_series_images, scale, temperature):
       """Study recrystallization kinetics from time series images."""

       time_points = []
       grain_sizes = []
       grain_counts = []
       recrystallized_fractions = []

       analyzer = GrainAnalyzer()

       for time_min, image_path in time_series_images.items():
           print(f"Analyzing t = {time_min} minutes...")

           results = analyzer.analyze(
               image_path=image_path,
               scale=scale,
               min_area=20,  # Detect small recrystallized grains
               gaussian_sigma=0.8
           )

           stats = results['statistics']
           grain_metrics = results['metrics']

           # Estimate recrystallized fraction based on grain size bimodality
           ecd_values = [grain['ecd_um'] for grain in grain_metrics.values()]

           # Simple threshold-based approach (improve with more sophisticated methods)
           small_grain_threshold = 5.0  # μm
           recryst_grains = [ecd for ecd in ecd_values if ecd < small_grain_threshold]
           recryst_fraction = len(recryst_grains) / len(ecd_values)

           time_points.append(time_min)
           grain_sizes.append(stats['ecd_statistics']['mean'])
           grain_counts.append(stats['grain_count'])
           recrystallized_fractions.append(recryst_fraction)

       # Fit Avrami equation: X = 1 - exp(-k*t^n)
       from scipy.optimize import curve_fit

       def avrami_equation(t, k, n):
           return 1 - np.exp(-k * (t ** n))

       try:
           popt, _ = curve_fit(avrami_equation, time_points, recrystallized_fractions)
           k, n = popt
           print(f"Avrami parameters: k = {k:.2e}, n = {n:.2f}")
       except:
           print("Could not fit Avrami equation")
           k, n = 0, 0

       # Plot results
       fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

       # Recrystallized fraction vs time
       ax1.plot(time_points, recrystallized_fractions, 'o-', label='Data')
       if k > 0:
           t_fit = np.linspace(0, max(time_points), 100)
           X_fit = avrami_equation(t_fit, k, n)
           ax1.plot(t_fit, X_fit, '--', label=f'Avrami fit (n={n:.2f})')
       ax1.set_xlabel('Time (minutes)')
       ax1.set_ylabel('Recrystallized Fraction')
       ax1.set_title(f'Recrystallization Kinetics at {temperature}°C')
       ax1.legend()
       ax1.grid(True)

       # Mean grain size vs time
       ax2.plot(time_points, grain_sizes, 's-', color='red')
       ax2.set_xlabel('Time (minutes)')
       ax2.set_ylabel('Mean Grain Size (μm)')
       ax2.set_title('Grain Size Evolution')
       ax2.grid(True)

       # Grain count vs time
       ax3.plot(time_points, grain_counts, '^-', color='green')
       ax3.set_xlabel('Time (minutes)')
       ax3.set_ylabel('Grain Count')
       ax3.set_title('Grain Count Evolution')
       ax3.grid(True)

       # Final microstructure overlay
       final_results = analyzer.analyze(list(time_series_images.values())[-1], scale=scale)
       analyzer.plot_overlay()
       ax4.remove()  # Remove empty subplot

       plt.tight_layout()
       plt.savefig(f'recrystallization_study_{temperature}C.png', dpi=300)
       plt.show()

       return {
           'time_points': time_points,
           'grain_sizes': grain_sizes,
           'recrystallized_fractions': recrystallized_fractions,
           'avrami_k': k,
           'avrami_n': n
       }

   # Define time series
   time_series = {
       0: 'recryst_t0.tif',
       5: 'recryst_t5.tif',
       10: 'recryst_t10.tif',
       20: 'recryst_t20.tif',
       30: 'recryst_t30.tif',
       60: 'recryst_t60.tif'
   }

   # Study recrystallization kinetics
   kinetics_results = recrystallization_kinetics_study(
       time_series, scale=0.2, temperature=500
   )

Integration Examples
--------------------

Jupyter Notebook Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # In Jupyter notebook
   %matplotlib inline
   import ipywidgets as widgets
   from IPython.display import display
   from grainstat import GrainAnalyzer

   def interactive_analysis_widget():
       """Create interactive widget for grain analysis."""

       # Create widgets
       image_upload = widgets.FileUpload(
           accept='.tif,.tiff,.png,.jpg',
           multiple=False,
           description='Upload Image'
       )

       scale_slider = widgets.FloatSlider(
           value=0.5,
           min=0.01,
           max=5.0,
           step=0.01,
           description='Scale (μm/px)',
           style={'description_width': 'initial'}
       )

       min_area_slider = widgets.IntSlider(
           value=50,
           min=10,
           max=500,
           description='Min Area (px)',
           style={'description_width': 'initial'}
       )

       analyze_button = widgets.Button(
           description='Analyze',
           button_style='primary'
       )

       output = widgets.Output()

       def on_analyze_click(b):
           with output:
               output.clear_output()

               if not image_upload.value:
                   print("Please upload an image first.")
                   return

               # Save uploaded image temporarily
               import tempfile
               with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                   tmp.write(image_upload.value[0]['content'])
                   tmp_path = tmp.name

               try:
                   analyzer = GrainAnalyzer()
                   results = analyzer.analyze(
                       tmp_path,
                       scale=scale_slider.value,
                       min_area=min_area_slider.value
                   )

                   # Display results
                   stats = results['statistics']
                   print(f"Analysis Results:")
                   print(f"Total grains: {stats['grain_count']}")
                   print(f"Mean ECD: {stats['ecd_statistics']['mean']:.2f} μm")

                   # Show plots
                   analyzer.plot_histogram()
                   analyzer.plot_overlay()

               except Exception as e:
                   print(f"Analysis failed: {e}")
               finally:
                   import os
                   os.unlink(tmp_path)

       analyze_button.on_click(on_analyze_click)

       # Layout
       controls = widgets.VBox([
           image_upload,
           scale_slider,
           min_area_slider,
           analyze_button
       ])

       display(widgets.HBox([controls, output]))

   # Run interactive widget
   interactive_analysis_widget()

Database Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import sqlite3
   import pandas as pd
   from grainstat import GrainAnalyzer
   from datetime import datetime

   class GrainAnalysisDatabase:
       """Database interface for storing grain analysis results."""

       def __init__(self, db_path="grain_analysis.db"):
           self.db_path = db_path
           self.init_database()

       def init_database(self):
           """Initialize database tables."""
           conn = sqlite3.connect(self.db_path)

           # Samples table
           conn.execute('''
               CREATE TABLE IF NOT EXISTS samples (
                   id INTEGER PRIMARY KEY,
                   sample_name TEXT UNIQUE,
                   material TEXT,
                   condition TEXT,
                   image_path TEXT,
                   scale_um_per_px REAL,
                   analysis_date TEXT,
                   notes TEXT
               )
           ''')

           # Analysis results table
           conn.execute('''
               CREATE TABLE IF NOT EXISTS analysis_results (
                   id INTEGER PRIMARY KEY,
                   sample_id INTEGER,
                   grain_count INTEGER,
                   mean_ecd_um REAL,
                   median_ecd_um REAL,
                   std_ecd_um REAL,
                   astm_grain_size REAL,
                   FOREIGN KEY (sample_id) REFERENCES samples (id)
               )
           ''')

           # Individual grains table
           conn.execute('''
               CREATE TABLE IF NOT EXISTS grains (
                   id INTEGER PRIMARY KEY,
                   sample_id INTEGER,
                   grain_id INTEGER,
                   ecd_um REAL,
                   area_um2 REAL,
                   aspect_ratio REAL,
                   shape_factor REAL,
                   eccentricity REAL,
                   FOREIGN KEY (sample_id) REFERENCES samples (id)
               )
           ''')

           conn.close()

       def add_analysis(self, sample_name, image_path, scale, material=None,
                       condition=None, notes=None):
           """Analyze sample and store results in database."""

           # Perform analysis
           analyzer = GrainAnalyzer()
           results = analyzer.analyze(image_path, scale=scale)

           conn = sqlite3.connect(self.db_path)

           try:
               # Insert sample record
               cursor = conn.execute('''
                   INSERT OR REPLACE INTO samples
                   (sample_name, material, condition, image_path, scale_um_per_px,
                    analysis_date, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
               ''', (sample_name, material, condition, image_path, scale,
                     datetime.now().isoformat(), notes))

               sample_id = cursor.lastrowid

               # Insert analysis results
               stats = results['statistics']
               conn.execute('''
                   INSERT INTO analysis_results
                   (sample_id, grain_count, mean_ecd_um, median_ecd_um,
                    std_ecd_um, astm_grain_size)
                   VALUES (?, ?, ?, ?, ?, ?)
               ''', (sample_id,
                     stats['grain_count'],
                     stats['ecd_statistics']['mean'],
                     stats['ecd_statistics']['median'],
                     stats['ecd_statistics']['std'],
                     stats['astm_grain_size']['grain_size_number']))

               # Insert individual grain data
               grain_data = []
               for grain_id, grain in results['metrics'].items():
                   grain_data.append((
                       sample_id, grain_id, grain['ecd_um'], grain['area_um2'],
                       grain['aspect_ratio'], grain['shape_factor'],
                       grain['eccentricity']
                   ))

               conn.executemany('''
                   INSERT INTO grains
                   (sample_id, grain_id, ecd_um, area_um2, aspect_ratio,
                    shape_factor, eccentricity)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
               ''', grain_data)

               conn.commit()
               print(f"Successfully stored analysis for {sample_name}")

           except Exception as e:
               conn.rollback()
               print(f"Error storing analysis: {e}")
           finally:
               conn.close()

       def query_samples(self, material=None, condition=None):
           """Query samples with optional filters."""

           query = "SELECT * FROM samples WHERE 1=1"
           params = []

           if material:
               query += " AND material = ?"
               params.append(material)

           if condition:
               query += " AND condition = ?"
               params.append(condition)

           conn = sqlite3.connect(self.db_path)
           df = pd.read_sql_query(query, conn, params=params)
           conn.close()

           return df

       def get_grain_size_trends(self, material=None):
           """Get grain size trends over time."""

           query = '''
               SELECT s.analysis_date, s.material, s.condition,
                      ar.mean_ecd_um, ar.astm_grain_size
               FROM samples s
               JOIN analysis_results ar ON s.id = ar.sample_id
               WHERE 1=1
           '''
           params = []

           if material:
               query += " AND s.material = ?"
               params.append(material)

           query += " ORDER BY s.analysis_date"

           conn = sqlite3.connect(self.db_path)
           df = pd.read_sql_query(query, conn, params=params)
           conn.close()

           return df

   # Usage example
   db = GrainAnalysisDatabase()

   # Add samples to database
   samples_to_analyze = [
       ("steel_as_received", "steel_ar.tif", 0.3, "Steel", "As Received"),
       ("steel_annealed", "steel_annealed.tif", 0.3, "Steel", "Annealed"),
       ("aluminum_t6", "al_t6.tif", 0.2, "Aluminum", "T6")
   ]

   for sample_name, image_path, scale, material, condition in samples_to_analyze:
       db.add_analysis(sample_name, image_path, scale, material, condition)

   # Query results
   steel_samples = db.query_samples(material="Steel")
   print("Steel samples in database:")
   print(steel_samples)

   # Get trends
   trends = db.get_grain_size_trends()
   print("\nGrain size trends:")
   print(trends)

This comprehensive examples page demonstrates the versatility and power of GrainStat for various materials science applications, from basic analysis to advanced research workflows.