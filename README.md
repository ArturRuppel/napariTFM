# napariTFM User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Module Overview](#module-overview)
5. [Preprocessing](#preprocessing)
6. [Displacement Analysis](#displacement-analysis)
7. [Force Calculation](#force-calculation)
8. [Stress Analysis](#stress-analysis)
9. [Batch Processing](#batch-processing)
10. [Tips and Troubleshooting](#tips-and-troubleshooting)

## Introduction

napariTFM is a comprehensive tool for Traction Force Microscopy (TFM) analysis, built as a plugin for the napari image viewer. It provides a complete analysis pipeline for investigating cell-generated forces through displacement field measurements, traction force reconstruction, and Monolayer Stress Microscopy (MSM).

The software combines established TFM algorithms with napari's visualization capabilities to enable systematic analysis of cell-substrate interactions. It supports both single-frame and time series analysis, making it suitable for studying various experimental setups from individual cells to cell monolayers.

### Key Features
- Complete TFM analysis pipeline from preprocessing to stress calculation
- Monolayer Stress Microscopy (MSM) for internal stress analysis
- Interactive visualization of results
- Support for both single images and time series data
- Integration with napari's image viewing capabilities
- Customizable analysis parameters
- Results export for further analysis

## Installation

napariTFM requires specific system dependencies. Choose the appropriate installation method for your operating system:

### Windows Installation

1. Install [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) if you haven't already.
2. Open Anaconda Prompt and navigate to the downloaded package directory:
```bash
cd path/to/napariTFM
```
Replace "path/to" with the actual path to where you downloaded the package.

3. Create and activate a new environment:
```bash
mamba create -n napariTFM python=3.9
mamba activate napariTFM
```

4. Install required dependencies:
```bash
mamba install -c conda-forge opencv=4.10.0 napari
```

5. Install napariTFM:
```bash
pip install .
```

### Linux Installation

1. Install [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) if you haven't already.
2. Open a terminal and navigate to the downloaded package directory:
```bash
cd path/to/napariTFM
```
Replace "path/to" with the actual path to where you downloaded the package.

3. Create and activate a new environment:
```bash
mamba create -n napariTFM python=3.9
mamba activate napariTFM
```

4. Install napari (without OpenCV to avoid conflicts):
```bash
mamba install -c conda-forge napari
```

5. Install napariTFM:
```bash
pip install .
```

6. **Important for Linux**: Install OpenCV with contrib modules to avoid symbol conflicts:
```bash
pip install opencv-contrib-python==4.10.0.84 --force-reinstall
```

### Starting napari

7. Start napari:
```bash
python -m napari
```
You can also try the shorter command `napari`, but if that doesn't work, use `python -m napari`.

The napariTFM plugin will be available in the Plugins menu.

## Getting Started

### Required Data
To perform TFM analysis, you need:
- Bead images (single image or time series)
- Reference image (relaxed state)
- Cell images (optional, for stress analysis)

### Data Format
- Images should be grayscale TIFF format
- Time series should be 3D stacks (time, height, width)
- Single images should be 2D (height, width)
- Both single-frame and time series analysis are supported


## Module Overview

napariTFM consists of four main analysis modules:

1. **Preprocessing**: Image enhancement and registration
2. **Displacement Analysis**: Displacement field measurement
3. **Force Calculation**: Traction force computation using FTTC
4. **Stress Analysis**: Internal stress field calculation using MSM

## Preprocessing

### Purpose
Prepare raw microscopy images for analysis by:
- Correcting background illumination
- Enhancing contrast
- Reducing noise
- Aligning image sequences

### Steps
1. Load Data:
   - Click "Load Bead Stack" to load bead images
   - Click "Load Reference Image" to load the reference image
   - Click "Load Cell Stack" (optional) for cell images

2. Adjust Parameters:
   - Rolling Ball Radius: Background correction (0-50 pixels)
   - Intensity Range: Set min/max percentiles for contrast
   - Gaussian Blur: Noise reduction (0-10 sigma)
   - Registration Mode: Translation or Rigid alignment

3. Preview Results:
   - Toggle "Show Preview" to visualize effects
   - Select data type (Beads/Reference/Cells) to preview
   - Adjust parameters until satisfied

4. Process Data:
   - Click "Run Preprocessing" to process all images
   - Save results using "Save Result Images"

### Tips
- Start with default parameters and adjust as needed
- Use preview to fine-tune settings
- Save preprocessed data for later use


## Displacement Analysis

### Purpose
Calculate displacement fields from bead movements using optical flow algorithms. The analysis determines how fluorescent beads embedded in the substrate move between a reference state (relaxed) and subsequent images (deformed), providing a quantitative measure of substrate deformation.

### Technical Background
#### Optical Flow Algorithm
napariTFM uses the TV-L1 optical flow algorithm, which is particularly well-suited for TFM analysis because it:
- Handles steep gradients in displacement fields effectively
- Handles large displacements through multi-scale analysis
- Provides sub-pixel accuracy
- Is robust to intensity variations

The algorithm works by:
1. Minimizing an energy functional that combines:
   - Brightness constancy assumption (beads maintain intensity)
   - Total variation regularization (smooth displacement fields)
   - Additional constraints for numerical stability

2. Using a multi-scale pyramid approach:
   - Images are analyzed at different resolution levels
   - Large displacements are captured at coarse scales
   - Fine details are refined at higher resolutions

### Parameters

#### Basic Parameters
- **Lambda (λ)**: Controls the balance between data fitting and smoothness
  - Lower values (0.01-0.1): More smoothing, good for noisy data and small displacements
  - Higher values (0.1-1.0): Less smoothing, better for clear bead images and larger displacements
  - Default: 0.1

#### Advanced Parameters
- **Pyramid Scales**
  - Number of resolution levels
  - More scales handle larger displacements
  - Typical range: 3-5 for standard TFM data

- **Warps**
  - Number of iterative refinements per scale
  - More warps increase accuracy for large displacements
  - Typical range: 3-5

- **Epsilon**
  - Stopping criterion for optimization
  - Smaller values give more precise results but increase computation time
  - Default: 0.01

- **Scale Step**
  - Factor between pyramid levels (0.5-0.8)
  - Smaller values create more intermediate scales
  - Example: 0.5 means each level is half the size of the previous

#### Visualization Parameters
- **Vector Stride**: Display every nth vector
- **Arrow Scale**: Adjust vector arrow size
- **Maximum Displacement**: Color scale limit in μm

### Analysis Steps

1. **Preparation**
   - Ensure preprocessed bead images are loaded
   - Verify reference image is set

2. **Parameter Adjustment**
   - Start with default parameters
   - Use "Preview Current Frame" to test settings
   - Observe displacement field visualization
   - Check displacement magnitude ranges

3. **Quality Control**
   - Verify displacement field smoothness
   - Check for outliers or artifacts
   - Ensure displacement magnitudes are physically reasonable
   - Compare with raw bead movements visually

4. **Full Analysis**
   - Run "Calculate All Frames" for time series
   - Monitor progress
   - Review results frame by frame
   - Save displacement data for force calculation

### Tips for Optimal Results

#### Image Quality
- Good signal-to-noise ratio is crucial
- Beads should be well-focused
- Adequate bead density improves accuracy
- Avoid saturated or very dim regions

#### Parameter Selection
1. Start with Default Parameters:
   - Lambda = 0.1
   - Pyramid scales = 4
   - Warps = 3

2. Adjust Based on Data:
   - Increase scales for larger displacements
   - Adjust lambda if result is too noisy or too smooth
   - Fine-tune warps for accuracy

3. Common Issues and Solutions:
   - Noisy results: Decrease lambda, increase smoothing
   - Missed displacements: Increase pyramid scales
   - Artifacts: Check preprocessing, adjust parameters
   - Slow processing: Reduce scales or warps

#### Validation
- Compare different parameter sets
- Check consistency across frames
- Verify displacement patterns match visual inspection
- Consider using control regions for baseline noise

### Data Export
- Displacement fields saved as NumPy arrays
- Units in micrometers
- Format compatible with force calculation
- Include metadata about analysis parameters

## Force Calculation

### Purpose
Convert displacement fields to traction forces using FTTC algorithm.

### Steps
1. Load displacement data:
   - Continue from displacement analysis or
   - Load saved displacement results

2. Set Parameters:
   - Material Properties:
     - Young's Modulus (kPa)
     - Poisson's Ratio
     - Gel Height
   - Regularization:
     - Manual parameter or
     - Auto-GCV selection

3. Calculate Forces:
   - Preview current frame
   - Calculate all frames
   - Save results

### Tips
- Verify substrate properties carefully
- Use Auto-GCV for optimal regularization
- Check force magnitude ranges

## Stress Analysis

### Purpose
Calculate internal stress fields in cell monolayers.

### Steps
1. Load Required Data:
   - Force calculation results
   - Cell masks or images

2. Configure Analysis:
   - Mask Parameters:
     - Threshold
     - Dilation
     - Smoothing
   - Mesh Parameters:
     - Density
     - Algorithm selection
   - Material Properties

3. Analysis Steps:
   - Create/load masks
   - Preview mesh
   - Calculate stress tensors
   - Save results

### Tips
- Verify mask quality before analysis
- Adjust mesh density for balance
- Use appropriate visualization settings

## Batch Processing

### Purpose
Process multiple experiments with consistent parameters.

### Steps
1. Setup:
   - Add experiment folders
   - Configure file names
   - Set analysis parameters

2. Configure Steps:
   - Select analysis steps to perform
   - Choose visualizations to save

3. Execution:
   - Run in napari console or
   - Launch new console
   - Monitor progress

### Tips
- Organize data consistently
- Test parameters on single dataset first
- Use descriptive folder names

## Tips and Troubleshooting

### General Tips
- Always save intermediate results
- Start with test datasets
- Monitor system resources
- Keep original data backed up

### Common Issues
1. Memory Errors:
   - Reduce image size
   - Process fewer frames
   - Close other applications

2. Poor Results:
   - Check image quality
   - Verify parameter settings
   - Ensure correct calibration

3. Performance:
   - Use appropriate downscaling
   - Adjust mesh density
   - Consider batch processing

### Best Practices
1. Data Organization:
   - Use consistent naming
   - Maintain folder structure
   - Document parameters

2. Analysis Flow:
   - Validate each step
   - Save intermediate results
   - Document modifications

3. Quality Control:
   - Check visualizations
   - Verify physical values
   - Compare with controls

### Getting Help
- Check documentation
- Review error messages
- Contact support with:
  - Error description
  - Sample data
  - Parameter settings