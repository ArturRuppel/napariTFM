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

napariTFM is a comprehensive tool for Traction Force Microscopy (TFM) analysis, built as a plugin for the napari image viewer. It enables researchers to analyze cell-generated forces by processing microscopy images of cells on deformable substrates.

### Key Features
- Complete TFM analysis pipeline from preprocessing to stress calculation
- Interactive visualization of results
- Support for both single experiments and batch processing
- Integration with napari's powerful image viewing capabilities
- Customizable analysis parameters
- Results export in standard formats

## Installation

napariTFM can be installed through pip:

```bash
pip install napari-tfm
```

## Getting Started

### Required Data
To perform TFM analysis, you need:
- Bead images (time series)
- Reference image (relaxed state)
- Cell images (optional, for stress analysis)

### Data Format
- Images should be grayscale
- Time series should be 3D stacks (time, height, width)
- Single images should be 2D (height, width)
- Supported formats: TIFF, TIF, PNG, JPG

## Module Overview

napariTFM consists of four main analysis modules:

1. **Preprocessing**: Image enhancement and registration
2. **Displacement Analysis**: Bead tracking and displacement field calculation
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
Calculate displacement fields from bead movements using optical flow algorithms.

### Steps
1. Load preprocessed data:
   - Either continue from preprocessing or
   - Load previously saved preprocessed data

2. Configure Parameters:
   - Basic Parameters:
     - Lambda: Smoothness weight (0.01-1.0)
   - Advanced Parameters:
     - Pyramid Scales: For large displacements
     - Warps: Iteration count
     - Other flow parameters

3. Analysis:
   - Use "Preview Current Frame" to test settings
   - "Calculate All Frames" for full analysis
   - Save results for force calculation

### Tips
- Start with preview to validate parameters
- Increase scales for larger displacements
- Monitor visualization quality

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