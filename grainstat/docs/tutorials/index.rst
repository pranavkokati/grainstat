Tutorials
=========

This section provides step-by-step tutorials for using GrainStat effectively. Each tutorial builds on previous knowledge and demonstrates specific use cases.

.. toctree::
   :maxdepth: 2
   :caption: Beginner Tutorials

   basic_analysis
   understanding_parameters
   working_with_results

.. toctree::
   :maxdepth: 2
   :caption: Intermediate Tutorials

   batch_processing
   custom_analysis_parameters
   visualization_customization

.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   custom_features
   interactive_analysis
   automation_workflows

.. toctree::
   :maxdepth: 2
   :caption: Specialized Applications

   optical_microscopy
   sem_analysis
   quality_control
   research_workflows

Tutorial Overview
-----------------

Beginner Level
~~~~~~~~~~~~~~

**Basic Analysis**
   Learn the fundamental workflow for analyzing microstructure images, from loading data to generating reports.

**Understanding Parameters**
   Understand how different analysis parameters affect results and how to optimize them for your images.

**Working with Results**
   Learn how to interpret, export, and visualize your analysis results effectively.

Intermediate Level
~~~~~~~~~~~~~~~~~~

**Batch Processing**
   Process multiple images efficiently using GrainStat's batch processing capabilities.

**Custom Analysis Parameters**
   Learn how to fine-tune analysis parameters for specific materials and imaging conditions.

**Visualization Customization**
   Create publication-quality plots and customize visualizations for your needs.

Advanced Level
~~~~~~~~~~~~~~

**Custom Features**
   Develop custom grain features using the plugin system for specialized analysis.

**Interactive Analysis**
   Use interactive tools for detailed grain inspection and data exploration.

**Automation Workflows**
   Build automated analysis pipelines for production environments.

Specialized Applications
~~~~~~~~~~~~~~~~~~~~~~~~

**Optical Microscopy**
   Specific techniques and parameters for analyzing optical microscopy images.

**SEM Analysis**
   Best practices for scanning electron microscopy image analysis.

**Quality Control**
   Implement GrainStat in quality control and production monitoring workflows.

**Research Workflows**
   Advanced research applications including statistical analysis and comparative studies.

Prerequisites
-------------

Before starting these tutorials, ensure you have:

1. **GrainStat installed** - See :doc:`../installation` for instructions
2. **Python basics** - Basic familiarity with Python programming
3. **Sample images** - Microstructure images to analyze (we provide examples)
4. **Scale information** - Know the micrometers per pixel for your images

Getting Sample Data
-------------------

To follow along with the tutorials, you can download sample microstructure images:

.. code-block:: bash

   # Create sample data using GrainStat
   python -c "
   import numpy as np
   from PIL import Image
   import os

   # Create sample directory
   os.makedirs('tutorial_data', exist_ok=True)

   # Generate synthetic microstructures
   for i in range(3):
       image = np.zeros((200, 200))
       np.random.seed(i * 42)

       # Add random circular grains
       for _ in range(20):
           x = np.random.randint(20, 180)
           y = np.random.randint(20, 180)
           r = np.random.randint(5, 15)

           yy, xx = np.ogrid[:200, :200]
           mask = (xx - x)**2 + (yy - y)**2 <= r**2
           image[mask] = 1.0

       # Add noise
       image += np.random.normal(0, 0.05, image.shape)
       image = np.clip(image, 0, 1)

       # Save
       img_pil = Image.fromarray((image * 255).astype(np.uint8))
       img_pil.save(f'tutorial_data/sample_{i+1}.tif')

   print('Created sample data in tutorial_data/')
   "

Or download real microstructure examples:

.. code-block:: bash

   # Download example dataset (hypothetical)
   wget https://github.com/materialslab/grainstat-examples/archive/main.zip
   unzip main.zip
   mv grainstat-examples-main/images tutorial_data/

Learning Path
-------------

We recommend following the tutorials in order:

**Week 1: Basics**
   1. :doc:`basic_analysis` (30 minutes)
   2. :doc:`understanding_parameters` (45 minutes)
   3. :doc:`working_with_results` (30 minutes)

**Week 2: Intermediate Skills**
   1. :doc:`batch_processing` (60 minutes)
   2. :doc:`custom_analysis_parameters` (45 minutes)
   3. :doc:`visualization_customization` (45 minutes)

**Week 3: Advanced Features**
   1. :doc:`custom_features` (90 minutes)
   2. :doc:`interactive_analysis` (30 minutes)
   3. :doc:`automation_workflows` (60 minutes)

**Week 4: Specialization**
   Choose tutorials based on your specific needs:
   - :doc:`optical_microscopy` for optical imaging
   - :doc:`sem_analysis` for electron microscopy
   - :doc:`quality_control` for industrial applications
   - :doc:`research_workflows` for academic research

Tutorial Format
---------------

Each tutorial follows a consistent format:

**Learning Objectives**
   Clear goals for what you'll learn

**Prerequisites**
   Required knowledge and setup

**Step-by-Step Instructions**
   Detailed code examples with explanations

**Expected Output**
   Screenshots and results you should see

**Troubleshooting**
   Common issues and solutions

**Further Reading**
   Links to related documentation

**Exercises**
   Practice problems to reinforce learning

Getting Help
------------

If you encounter issues while following tutorials:

1. **Check the FAQ** - Common solutions in our documentation
2. **Review Prerequisites** - Ensure you have the required setup
3. **Try the Examples** - Use provided sample data first
4. **Search Issues** - https://github.com/materialslab/grainstat/issues
5. **Ask Questions** - https://github.com/materialslab/grainstat/discussions

Contributing Tutorials
----------------------

We welcome contributions to improve these tutorials:

- **Report Issues** - Found errors or unclear instructions?
- **Suggest Improvements** - Ideas for better explanations?
- **Add New Tutorials** - Specialized use cases not covered?
- **Provide Feedback** - What worked well or could be improved?

See our :doc:`../contributing` guide for details on how to contribute.

Quick Reference
---------------

**Common Commands**

.. code-block:: bash

   # Single analysis
   grainstat analyze image.tif --scale 0.5 --export-csv results.csv

   # Batch processing
   grainstat batch input/ output/ --scale 0.5 --workers 4

   # Interactive viewer
   grainstat interactive image.tif --scale 0.5

**Common Import Pattern**

.. code-block:: python

   from grainstat import GrainAnalyzer

   analyzer = GrainAnalyzer()
   results = analyzer.analyze("image.tif", scale=0.5)

**Typical Analysis Workflow**

1. Load and inspect image
2. Determine scale (μm/pixel)
3. Run analysis with appropriate parameters
4. Review and validate results
5. Export data and generate reports
6. Iterate and optimize as needed

Ready to Start?
---------------

Begin with :doc:`basic_analysis` to learn the fundamental GrainStat workflow, or jump to a specific tutorial that matches your current needs and experience level.

.. note::
   These tutorials use synthetic and example data. When working with your own images, always validate that the analysis parameters are appropriate for your specific material and imaging conditions.