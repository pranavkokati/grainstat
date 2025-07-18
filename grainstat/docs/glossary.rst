Glossary
========

This glossary defines key terms used in grain analysis and GrainStat documentation.

.. glossary::
   :sorted:

   Aspect Ratio
      The ratio of the major axis length to the minor axis length of a grain. Values close to 1 indicate equiaxed grains, while higher values indicate elongated grains.

   ASTM E112
      American Society for Testing and Materials standard E112, which provides methods for determining the average grain size of metals. The standard defines the grain size number G.

   Binary Image
      An image containing only two pixel values (typically 0 and 1, or black and white), used to represent grain boundaries and grain regions after thresholding.

   CLAHE
      Contrast Limited Adaptive Histogram Equalization. An image processing technique that enhances local contrast while limiting noise amplification.

   Compactness
      A shape descriptor that measures how compact a grain is. Calculated as the square of the perimeter divided by the area. Lower values indicate more compact (circular) grains.

   Convexity
      The ratio of a grain's area to the area of its convex hull. Values close to 1 indicate convex grains, while lower values indicate grains with concave regions.

   Eccentricity
      A measure of how much a grain deviates from being circular. Values range from 0 (perfect circle) to 1 (line segment).

   ECD
      Equivalent Circular Diameter. The diameter of a circle that has the same area as the grain. Calculated as 2√(A/π) where A is the grain area.

   Equiaxed
      Describes grains that have approximately equal dimensions in all directions (roughly spherical in 3D or circular in 2D cross-sections).

   Euler Number
      A topological property that describes the connectivity of a shape. For grains, it indicates the number of holes or separate parts.

   Gaussian Filter
      A smoothing filter that reduces image noise by convolving the image with a Gaussian kernel. The sigma parameter controls the amount of smoothing.

   Grain
      A crystalline region in a material where the crystal lattice has a uniform orientation. Grains are separated by grain boundaries.

   Grain Boundary
      The interface between two grains with different crystallographic orientations. These appear as lines or curves in 2D microstructure images.

   Grain Size Number
      The ASTM grain size number G, calculated from the mean lineal intercept length. Higher numbers indicate smaller grains.

   Labeled Image
      An image where each grain is assigned a unique integer label (ID), allowing individual grain analysis.

   Major Axis
      The length of the longest line that can be drawn through a grain while staying within its boundaries.

   Mean Lineal Intercept
      The average length of line segments that intersect grains when a straight line is drawn across the microstructure.

   Microstructure
      The structure of a material as revealed by microscopy, typically showing features like grains, phases, and defects at the microscopic scale.

   Minor Axis
      The length of the shortest line that can be drawn through a grain while staying within its boundaries, perpendicular to the major axis.

   Morphological Operations
      Image processing operations that process images based on their shape and structure, including erosion, dilation, opening, and closing.

   Optical Microscopy
      Microscopy technique using visible light to image materials, typically after polishing and etching to reveal microstructural features.

   Orientation
      The angle of the major axis of a grain relative to a reference direction, typically measured in radians.

   Otsu Thresholding
      An automatic threshold selection method that finds the optimal threshold to separate foreground (grains) from background by minimizing intra-class variance.

   Perimeter
      The length of the boundary around a grain, measured in pixels or physical units.

   Phase
      A distinct region in a material with uniform crystal structure and composition, different from surrounding phases.

   Pixel
      The smallest unit of a digital image, representing a single point in the image with a specific intensity value.

   RegionProps
      A function in image analysis libraries that calculates various properties of labeled regions, such as area, perimeter, and shape descriptors.

   Recrystallization
      A process where new grains form and grow in a deformed material, typically resulting in smaller, equiaxed grains.

   Scale
      The conversion factor between pixels and physical units, typically expressed as micrometers per pixel (μm/px).

   SEM
      Scanning Electron Microscopy. A high-resolution imaging technique that uses an electron beam to create images of material surfaces.

   Segmentation
      The process of partitioning an image into meaningful regions, such as separating grains from grain boundaries.

   Shape Factor
      A measure of how circular a grain is, calculated as 4πA/P² where A is area and P is perimeter. Values close to 1 indicate circular grains.

   Solidity
      The ratio of a grain's area to the area of its convex hull. Values close to 1 indicate solid, convex grains.

   Thresholding
      The process of converting a grayscale image to a binary image by selecting a threshold value to separate features of interest.

   Watershed Segmentation
      A segmentation algorithm that treats the image as a topographic surface and finds watershed lines that separate different regions.

Mathematical Symbols
--------------------

.. glossary::
   :sorted:

   A
      Area of a grain (typically in μm²)

   AR
      Aspect Ratio (major axis / minor axis)

   C
      Compactness (P²/4πA)

   D
      Diameter (various types: ECD, Feret, etc.)

   ECD
      Equivalent Circular Diameter (2√(A/π))

   G
      ASTM grain size number

   L
      Mean lineal intercept length

   P
      Perimeter of a grain (typically in μm)

   SF
      Shape Factor (4πA/P²)

   a
      Major axis length

   b
      Minor axis length

   φ
      Shape factor (same as SF)

   σ
      Standard deviation or Gaussian sigma parameter

Acronyms and Abbreviations
--------------------------

.. glossary::
   :sorted:

   API
      Application Programming Interface

   ASTM
      American Society for Testing and Materials

   BMP
      Bitmap image format

   CLI
      Command Line Interface

   CSV
      Comma-Separated Values file format

   DPI
      Dots Per Inch (image resolution)

   EBSD
      Electron Backscatter Diffraction

   GUI
      Graphical User Interface

   HDF5
      Hierarchical Data Format version 5

   HTML
      HyperText Markup Language

   ISO
      International Organization for Standardization

   JPEG
      Joint Photographic Experts Group image format

   JSON
      JavaScript Object Notation

   MTEX
      MATLAB Toolbox for crystallographic texture analysis

   PDF
      Portable Document Format

   PNG
      Portable Network Graphics image format

   QC
      Quality Control

   RGB
      Red, Green, Blue color model

   ROI
      Region of Interest

   RTD
      Read the Docs

   SEM
      Scanning Electron Microscopy

   TIFF
      Tagged Image File Format

   XML
      eXtensible Markup Language

Units and Conversions
--------------------

.. glossary::
   :sorted:

   mm
      Millimeter (10⁻³ meters)

   μm
      Micrometer (10⁻⁶ meters)

   nm
      Nanometer (10⁻⁹ meters)

   px
      Pixel (picture element)

   μm/px
      Micrometers per pixel (scale factor)

   μm²
      Square micrometers (area unit)

Common Conversion Factors:
   - 1 mm = 1,000 μm
   - 1 μm = 1,000 nm
   - 1 inch = 25,400 μm
   - Scale calculation: Physical size = Pixel size × Scale factor

Quality Control Terms
--------------------

.. glossary::
   :sorted:

   Acceptance Criteria
      Predetermined standards that grain measurements must meet to be considered acceptable.

   Control Chart
      A statistical tool used to monitor grain size measurements over time to detect process variations.

   Process Capability
      A measure of how well a manufacturing process can produce grains within specified size limits.

   Specification Limits
      The acceptable range of grain sizes for a particular application or standard.

   Statistical Process Control
      The use of statistical methods to monitor and control grain size during manufacturing.

Research and Development Terms
------------------------------

.. glossary::
   :sorted:

   Annealing
      Heat treatment process that can cause grain growth and recrystallization.

   Cold Working
      Deformation of metals at room temperature, which can affect grain shape and size.

   Grain Growth
      The increase in average grain size, typically during heat treatment.

   Heat Treatment
      Controlled heating and cooling processes used to modify material properties, including grain structure.

   Phase Transformation
      Changes in crystal structure that can affect grain characteristics.

   Precipitation
      Formation of second-phase particles that can influence grain size and shape.

   Texture
      Preferred crystallographic orientation of grains in a polycrystalline material.

See Also
--------

- :doc:`api` - Complete API reference
- :doc:`examples` - Practical usage examples
- :doc:`tutorials/index` - Step-by-step tutorials
- `ASTM E112 Standard <https://www.astm.org/Standards/E112.htm>`_ - Official ASTM standard
- `ISO 643 Standard <https://www.iso.org/standard/75819.html>`_ - International grain size standard