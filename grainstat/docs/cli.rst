Command Line Interface
======================

GrainStat provides a comprehensive command-line interface (CLI) for automation, batch processing, and integration into analysis pipelines.

Overview
--------

The CLI is accessed through the ``grainstat`` command and provides several subcommands:

.. code-block:: bash

   grainstat <command> [options]

Available commands:

- ``analyze`` - Analyze a single image
- ``batch`` - Process multiple images in parallel
- ``interactive`` - Launch interactive grain viewer
- ``compare`` - Compare grain analysis between conditions
- ``version`` - Show version information

Global Options
--------------

All commands support these global options:

.. option:: --help, -h

   Show help message and exit

.. option:: --verbose, -v

   Enable verbose output

.. option:: --quiet, -q

   Suppress non-essential output

Single Image Analysis
---------------------

The ``analyze`` command processes a single microstructure image.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   grainstat analyze image.tif --scale 0.5

Required Arguments
~~~~~~~~~~~~~~~~~~

.. option:: image

   Path to the input image file. Supported formats: TIFF, PNG, JPEG, BMP.

Optional Arguments
~~~~~~~~~~~~~~~~~~

**Scale and Size Parameters**

.. option:: --scale SCALE

   Scale factor (micrometers per pixel). Default: 1.0

.. option:: --min-area PIXELS

   Minimum grain area in pixels. Default: 50

**Image Processing Parameters**

.. option:: --gaussian-sigma SIGMA

   Gaussian smoothing sigma value. Default: 1.0

   - 0: No smoothing
   - 0.5-1.0: Light smoothing
   - 1.0-2.0: Moderate smoothing
   - >2.0: Heavy smoothing

.. option:: --threshold-method {otsu,adaptive}

   Thresholding method. Default: otsu

   - ``otsu``: Global Otsu thresholding (good for uniform illumination)
   - ``adaptive``: Local adaptive thresholding (better for uneven lighting)

.. option:: --no-watershed

   Disable watershed segmentation (grains may not be separated)

.. option:: --morphology-radius RADIUS

   Morphological operations radius. Default: 2

**Output Options**

.. option:: --export-csv FILE

   Export grain data to CSV file

.. option:: --export-json FILE

   Export complete analysis to JSON file

.. option:: --report FILE

   Generate HTML report

.. option:: --plot-histogram FILE

   Save histogram plot to file

.. option:: --plot-overlay FILE

   Save overlay plot to file

.. option:: --plot-cumulative FILE

   Save cumulative distribution plot to file

Examples
~~~~~~~~

**Basic analysis with scale:**

.. code-block:: bash

   grainstat analyze steel_sample.tif --scale 0.3

**Analysis with custom parameters:**

.. code-block:: bash

   grainstat analyze noisy_image.tif \
       --scale 0.2 \
       --min-area 100 \
       --gaussian-sigma 2.0 \
       --threshold-method adaptive

**Complete analysis with all outputs:**

.. code-block:: bash

   grainstat analyze sample.tif \
       --scale 0.5 \
       --export-csv grain_data.csv \
       --export-json complete_results.json \
       --report analysis_report.html \
       --plot-histogram size_distribution.png \
       --plot-overlay grain_overlay.png

**Quick analysis for SEM images:**

.. code-block:: bash

   grainstat analyze sem_1000x.tif \
       --scale 0.05 \
       --min-area 20 \
       --gaussian-sigma 0.8 \
       --export-csv sem_results.csv

Batch Processing
----------------

The ``batch`` command processes multiple images in parallel.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   grainstat batch input_directory/ output_directory/ --scale 0.5

Required Arguments
~~~~~~~~~~~~~~~~~~

.. option:: input_dir

   Directory containing input images

.. option:: output_dir

   Directory for output files (created if it doesn't exist)

Optional Arguments
~~~~~~~~~~~~~~~~~~

**File Selection**

.. option:: --pattern PATTERN

   File pattern to match. Default: "*.tif*"

   Examples:

   - ``"*.tif"`` - TIFF files only
   - ``"*.png"`` - PNG files only
   - ``"sample_*.tif"`` - TIFF files starting with "sample_"

**Processing Parameters**

All single-image analysis parameters are supported:

.. option:: --scale SCALE

   Scale factor for all images

.. option:: --min-area PIXELS

   Minimum grain area

.. option:: --gaussian-sigma SIGMA

   Gaussian smoothing

.. option:: --threshold-method {otsu,adaptive}

   Thresholding method

.. option:: --no-watershed

   Disable watershed segmentation

.. option:: --morphology-radius RADIUS

   Morphological operations radius

**Parallel Processing**

.. option:: --workers N

   Number of parallel workers. Default: CPU count - 1

**Output Control**

.. option:: --no-individual-results

   Skip saving individual image results (only generate batch summary)

.. option:: --no-summary

   Skip generating batch summary

Examples
~~~~~~~~

**Basic batch processing:**

.. code-block:: bash

   grainstat batch samples/ results/ --scale 0.3

**Batch with custom parameters:**

.. code-block:: bash

   grainstat batch input/ output/ \
       --scale 0.2 \
       --pattern "*.tif" \
       --min-area 75 \
       --workers 8

**Fast processing (summary only):**

.. code-block:: bash

   grainstat batch production_samples/ qc_results/ \
       --scale 0.4 \
       --no-individual-results \
       --workers 12

**Process specific file types:**

.. code-block:: bash

   # Process only PNG files
   grainstat batch images/ results/ --pattern "*.png" --scale 0.6

   # Process files with specific naming pattern
   grainstat batch data/ output/ --pattern "experiment_*.tif" --scale 0.25

Batch Output Structure
~~~~~~~~~~~~~~~~~~~~~~

The batch command creates the following output structure:

.. code-block:: text

   output_directory/
   ├── batch_summary.csv         # Summary of all analyses
   ├── batch_summary.json        # Detailed batch statistics
   ├── batch_report.html         # HTML batch report
   ├── failed_images.csv         # List of failed analyses (if any)
   ├── sample1_grains.csv        # Individual grain data
   ├── sample1_analysis.json     # Complete analysis results
   ├── sample2_grains.csv
   ├── sample2_analysis.json
   └── ...

Interactive Viewer
------------------

Launch an interactive grain viewer for detailed analysis.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   grainstat interactive image.tif --scale 0.5

The interactive viewer provides:

- **Click to inspect**: Click on grains to see their properties
- **Zoom and pan**: Explore the microstructure in detail
- **Keyboard shortcuts**: Press 'h' for help

Keyboard Shortcuts
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Action
   * - ``h``
     - Show help dialog
   * - ``c``
     - Clear all highlights
   * - ``s``
     - Save selected grain info to file
   * - ``a``
     - Show summary of all grains

Condition Comparison
--------------------

Compare grain analysis results between different conditions.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   grainstat compare \
       --conditions control:control_samples/ treated:treated_samples/ \
       output_comparison/

Required Arguments
~~~~~~~~~~~~~~~~~~

.. option:: --conditions CONDITION:PATH [CONDITION:PATH ...]

   Condition directories in format ``name:path``

.. option:: output_dir

   Directory for comparison results

Optional Arguments
~~~~~~~~~~~~~~~~~~

.. option:: --scale SCALE

   Scale factor for all conditions. Default: 1.0

Examples
~~~~~~~~

**Heat treatment comparison:**

.. code-block:: bash

   grainstat compare \
       --conditions as_received:ar_samples/ annealed_400:400c_samples/ annealed_600:600c_samples/ \
       --scale 0.3 \
       heat_treatment_study/

**Alloy composition study:**

.. code-block:: bash

   grainstat compare \
       --conditions pure_al:pure/ al_2mg:2percent/ al_4mg:4percent/ \
       --scale 0.2 \
       composition_study/

Version Information
-------------------

Show GrainStat version and system information.

.. code-block:: bash

   grainstat version

Output includes:

- GrainStat version
- Python version
- Key dependency versions
- System information

Integration with Shell Scripts
------------------------------

Bash Script Example
~~~~~~~~~~~~~~~~~~~

Create automated analysis workflows:

.. code-block:: bash

   #!/bin/bash
   # analyze_batch.sh - Automated grain analysis script

   set -e  # Exit on error

   # Configuration
   INPUT_DIR="$1"
   OUTPUT_DIR="$2"
   SCALE="$3"

   if [ $# -ne 3 ]; then
       echo "Usage: $0 <input_dir> <output_dir> <scale>"
       exit 1
   fi

   # Create output directory
   mkdir -p "$OUTPUT_DIR"

   # Log analysis start
   echo "Starting analysis: $(date)" | tee "$OUTPUT_DIR/analysis.log"
   echo "Input: $INPUT_DIR" | tee -a "$OUTPUT_DIR/analysis.log"
   echo "Scale: $SCALE μm/pixel" | tee -a "$OUTPUT_DIR/analysis.log"

   # Count input files
   num_files=$(find "$INPUT_DIR" -name "*.tif" | wc -l)
   echo "Found $num_files TIFF files" | tee -a "$OUTPUT_DIR/analysis.log"

   # Run batch analysis
   grainstat batch "$INPUT_DIR" "$OUTPUT_DIR" \
       --scale "$SCALE" \
       --workers 4 \
       --pattern "*.tif" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"

   # Check results
   if [ -f "$OUTPUT_DIR/batch_summary.csv" ]; then
       echo "Analysis completed successfully: $(date)" | tee -a "$OUTPUT_DIR/analysis.log"

       # Show quick summary
       echo "Results summary:" | tee -a "$OUTPUT_DIR/analysis.log"
       tail -n 1 "$OUTPUT_DIR/batch_summary.csv" | tee -a "$OUTPUT_DIR/analysis.log"
   else
       echo "Analysis failed: $(date)" | tee -a "$OUTPUT_DIR/analysis.log"
       exit 1
   fi

Python Script Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Combine CLI with Python for advanced workflows:

.. code-block:: python

   #!/usr/bin/env python3
   """
   Advanced analysis workflow combining CLI and Python API
   """

   import subprocess
   import pandas as pd
   import matplotlib.pyplot as plt
   from pathlib import Path
   import sys

   def run_grainstat_analysis(input_dir, output_dir, scale):
       """Run GrainStat CLI analysis."""

       cmd = [
           'grainstat', 'batch',
           str(input_dir), str(output_dir),
           '--scale', str(scale),
           '--workers', '6'
       ]

       try:
           result = subprocess.run(cmd, check=True, capture_output=True, text=True)
           print("Analysis completed successfully")
           return True
       except subprocess.CalledProcessError as e:
           print(f"Analysis failed: {e}")
           print(f"Error output: {e.stderr}")
           return False

   def post_process_results(output_dir):
       """Post-process CLI results with Python."""

       # Load batch summary
       summary_file = Path(output_dir) / 'batch_summary.csv'

       if not summary_file.exists():
           print("No batch summary found")
           return

       df = pd.read_csv(summary_file)

       # Custom analysis
       print(f"Processed {len(df)} samples")
       print(f"Mean grain size: {df['mean_ecd_um'].mean():.2f} μm")
       print(f"Size range: {df['mean_ecd_um'].min():.2f} - {df['mean_ecd_um'].max():.2f} μm")

       # Custom visualization
       plt.figure(figsize=(10, 6))
       plt.subplot(1, 2, 1)
       plt.hist(df['mean_ecd_um'], bins=15, alpha=0.7)
       plt.xlabel('Mean ECD (μm)')
       plt.ylabel('Frequency')
       plt.title('Distribution of Mean Grain Sizes')

       plt.subplot(1, 2, 2)
       plt.scatter(df['grain_count'], df['mean_ecd_um'])
       plt.xlabel('Grain Count')
       plt.ylabel('Mean ECD (μm)')
       plt.title('Grain Count vs. Size')

       plt.tight_layout()
       plt.savefig(Path(output_dir) / 'custom_analysis.png', dpi=300)
       plt.show()

   if __name__ == "__main__":
       if len(sys.argv) != 4:
           print("Usage: python advanced_workflow.py <input_dir> <output_dir> <scale>")
           sys.exit(1)

       input_dir = Path(sys.argv[1])
       output_dir = Path(sys.argv[2])
       scale = float(sys.argv[3])

       # Run CLI analysis
       if run_grainstat_analysis(input_dir, output_dir, scale):
           # Post-process with Python
           post_process_results(output_dir)

Error Handling and Troubleshooting
-----------------------------------

Common Error Messages
~~~~~~~~~~~~~~~~~~~~~

**"Image file not found"**

.. code-block:: bash

   grainstat analyze nonexistent.tif --scale 0.5
   # Error: Image file not found: nonexistent.tif

Solution: Check the file path and ensure the file exists.

**"No image files found"**

.. code-block:: bash

   grainstat batch empty_directory/ output/ --scale 0.5
   # Error: No image files found in empty_directory/ with pattern *.tif*

Solutions:
- Check the directory path
- Verify files exist with the specified pattern
- Use ``--pattern`` to match different file types

**"Unsupported format"**

.. code-block:: bash

   grainstat analyze document.pdf --scale 0.5
   # Error: Unsupported format: .pdf

Solution: Convert to a supported format (TIFF, PNG, JPEG, BMP).

**"Memory error"**

For very large images:

.. code-block:: bash

   grainstat analyze huge_image.tif --scale 0.1 --min-area 200

Solutions:
- Increase ``--min-area`` to detect fewer objects
- Reduce image size before analysis
- Use a computer with more RAM

Verbose Output
~~~~~~~~~~~~~~

Use ``--verbose`` for detailed progress information:

.. code-block:: bash

   grainstat analyze image.tif --scale 0.5 --verbose

This shows:
- Processing steps
- Parameter values
- Timing information
- Intermediate results

Performance Tips
----------------

**Optimize for Speed**

.. code-block:: bash

   # Use more workers for batch processing
   grainstat batch input/ output/ --workers 8 --scale 0.5

   # Skip individual results for large batches
   grainstat batch input/ output/ --no-individual-results --scale 0.5

   # Use larger minimum area to reduce processing
   grainstat analyze image.tif --scale 0.5 --min-area 100

**Optimize for Quality**

.. code-block:: bash

   # Use adaptive thresholding for uneven illumination
   grainstat analyze image.tif --scale 0.5 --threshold-method adaptive

   # Increase smoothing for noisy images
   grainstat analyze noisy.tif --scale 0.5 --gaussian-sigma 2.0

   # Use smaller minimum area for fine-grained materials
   grainstat analyze fine_grains.tif --scale 0.1 --min-area 10

The CLI provides a powerful interface for integrating GrainStat into automated workflows, production systems, and research pipelines.