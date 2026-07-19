# napariTFM User Manual

> **📦 Download the latest stable release**
>
> The `master` branch is under active development and may be unstable or untested. For analysis work, download the latest stable release rather than cloning this branch:
>
> **➡️ [Download the latest release](https://github.com/ArturRuppel/napariTFM/releases/latest)** — grab the "Source code (zip)" asset, extract it, then follow the [Installation](#installation) steps below.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Module Overview](#module-overview)
5. [Displacement Analysis](#displacement-analysis)
6. [Force Calculation](#force-calculation)
7. [Stress Analysis](#stress-analysis)
8. [Batch Processing](#batch-processing)
9. [Tips and Troubleshooting](#tips-and-troubleshooting)

## Introduction

napariTFM is a comprehensive tool for Traction Force Microscopy (TFM) analysis, built as a plugin for the napari image viewer. It provides a complete analysis pipeline for investigating cell-generated forces through displacement field measurements, traction force reconstruction, and Bayesian Inversion Stress Microscopy (BISM).

The software combines established TFM algorithms with napari's visualization capabilities to enable systematic analysis of cell-substrate interactions. It supports both single-frame and time series analysis, making it suitable for studying various experimental setups from individual cells to cell monolayers.

A test data set, including recommended configuration parameters, can be found here: https://zenodo.org/records/18390989

### Algorithm Sources and Acknowledgments

The core algorithms implemented in napariTFM are based on established methods developed by our colleagues:

**FTTC (Fourier Transform Traction Cytometry):** The force calculation implementation is based on the DirectMethod package by Usschwarz (https://github.com/usschwarz/DirectMethod, MIT License) and incorporates methods from Blumberg & Schwarz, "Comparison of direct and inverse methods for 2.5D traction force microscopy" (2022). Gel height corrections are adapted from the pyTFM package (https://github.com/fabrylab/pyTFM, GNU GPL v3.0 License).

**BISM (Bayesian Inversion Stress Microscopy):** The stress field calculation implementation is a dependency-light port of the MATLAB reference (Nier et al., Biophys. J. 110(7):1625-1635, 2016; original BISM.m by Vincent Nier).

### Key Features
- Complete TFM analysis pipeline from displacement to stress calculation
- Bayesian Inversion Stress Microscopy (BISM) for internal stress analysis
- Interactive visualization of results
- Support for both single images and time series data
- Integration with napari's image viewing capabilities
- Customizable analysis parameters
- Results export for further analysis

## Installation

First, [download the latest release](https://github.com/ArturRuppel/napariTFM/releases/latest) ("Source code (zip)") and extract it. The extracted folder is referred to as the package directory below.

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

napariTFM consists of three main analysis modules:

1. **Displacement Analysis**: Displacement field measurement (PIV, Lucas-Kanade, or FFD on the raw bead images)
2. **Force Calculation**: Traction force computation using FTTC
3. **Stress Analysis**: Internal stress field calculation using BISM

There is no separate preprocessing stage: the displacement stage first registers
the reference and every bead frame to the first bead frame (parameter-free phase
cross-correlation, translation only), removing bulk stage drift before any method
runs, then measures the residual cell-induced deformation. Registering up front,
rather than subtracting drift afterward, keeps the motion within each method's
capture range. The optional cell channel is contrast-scaled and shifted by the
same per-frame drift so it lines up with the traction field in the overlay.

## Displacement Analysis

### Purpose
Calculate displacement fields from bead movements using one of three displacement algorithms (PIV, Lucas-Kanade, or FFD). The analysis determines how fluorescent beads embedded in the substrate move between a reference state (relaxed) and subsequent images (deformed), providing a quantitative measure of substrate deformation.

> **Which method should I use?** In the one imaging regime we benchmarked, the methods agreed on accuracy and separated on noise and capture range. [Choosing a displacement method](docs/choosing-a-displacement-method.md) reports what the test found (PIV a forgiving default, FFD-pyr strongest under large deformation), the regimes it does not cover, and how to tune each method on your own data.

### Technical Background

napariTFM offers three displacement algorithms, chosen in the **Method** dropdown. They agreed on accuracy in the one regime we benchmarked and separated on off-cell noise and capture range; [Choosing a displacement method](docs/choosing-a-displacement-method.md) reports what the benchmark found, the regimes it does not cover, and how to tune each on your own data.

- **PIV** (particle image velocimetry): FFT cross-correlation of interrogation windows, coarse-to-fine with window deformation. The forgiving default: quietest off-cell in our benchmark and graceful up to large motion.
- **Lucas-Kanade** (iterative optical flow): a dense local least-squares solve at every pixel over an image pyramid. Fast and light, and it tracked PIV closely at small motion.
- **FFD** (free-form deformation): a cubic B-spline control grid fit to the image pair over a pyramid. Strongest under large deformation. GPU-only.

Each method runs on the CPU by default (openpiv for PIV, scikit-image for Lucas-Kanade) with no extra dependency. Installing the GPU extra (`pip install napariTFM[gpu]`, which provides PyTorch) adds a CUDA-accelerated backend for all three and enables FFD. The **Device** dropdown selects `auto` (GPU when present, else CPU), `cuda` (require a GPU), or `cpu`. For Lucas-Kanade the GPU port is numerically identical to the CPU reference; for PIV it is at measured parity on dense beads, not bit-identical.

### Parameters

Every knob is in the parameter panel, the most important first in each method's group, each with a tooltip. Only the selected method's group is active. Start from the defaults and check on a frame with **Preview Current Frame** before a full run: no method is set-once, and the right values depend on your beads and motion.

#### Method Parameters
- **PIV**: **Interrogation Window** (px) is the primary peak-versus-noise knob (smaller sharpens the peak and raises noise). **Window Overlap** samples the field more finely (higher recovers sharp peaks, at more compute and GPU memory). **Passes** drive capture range and convergence.
- **Lucas-Kanade**: **Window Radius** (px) is a noise aperture (larger for noisier images, at the cost of peak sharpness). **Warp Iterations** refine convergence, not capture range.
- **FFD**: **Control Spacing** (px) is the bias-variance dial (fine recovers sharp peaks, coarse regularizes noise). **Pyramid Levels** set capture range. **Image Metric** (`lncc` or `mse`) chooses the match objective; `lncc` preserves peaks better.

**Downscale Factor** (shared) reduces the output field resolution by block-mean averaging.

#### Visualization Parameters
- **Vector Stride**: Display every nth vector
- **Arrow Scale**: Adjust vector arrow size
- **Maximum Displacement**: Color scale limit in μm

### Analysis Steps

1. **Preparation**
   - Ensure the raw bead images are loaded
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
1. Start from the defaults for your chosen method and preview one frame. Each knob's
   tooltip says which way to move it; [Choosing a displacement method](docs/choosing-a-displacement-method.md)
   gives worked starting points per beads-and-motion regime.

2. Adjust based on the preview:
   - Blunted peak: shrink the window (PIV) or control spacing (FFD), or raise PIV overlap.
   - Missed large displacements: raise PIV passes or FFD pyramid levels.
   - Noisy off-cell background: enlarge the window (PIV), radius (iLK), or control spacing (FFD).

3. Common issues and solutions:
   - Poor results everywhere: check input image quality (focus, density, saturation).
   - Out-of-memory on large frames (GPU): lower PIV overlap, or switch Device to `cpu`.
   - Slow processing: install the GPU extra and use Device `auto`, or increase Downscale Factor.

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
   - Regularization (choose one):
     - Manual parameter, or
     - Bayesian L2 (auto λ) — evidence-maximizing, noise-robust selection
       (Huang et al. 2019); infers λ per frame with no manual tuning, which is
       preferable to a manual value when comparing cells across conditions or
       over a time series. Uses a loaded mask's cell-free exterior for the noise
       estimate when present. The auto-λ button fills the manual field for the
       current frame (overridable).
   - Sparse inversion (L1):
     - L1 Sparsity for sparse, peak-preserving traction (group-L1) — thresholds
       small forces to zero rather than spreading them, so it recovers discrete
       adhesion forces and preserves peaks better than plain FTTC.

3. Calculate Forces:
   - Preview current frame
   - Calculate all frames
   - Save results

### Tips
- Verify substrate properties carefully
- For automatic regularization, use Bayesian L2 (it chooses λ per frame),
  especially at higher noise or when comparing conditions
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
   - BISM regularization (Lambda)

3. Analysis Steps:
   - Create/load masks
   - Calculate stress tensors
   - Save results

### Tips
- Verify mask quality before analysis
- BISM is mesh-free, so there's no mesh density/algorithm to tune
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
