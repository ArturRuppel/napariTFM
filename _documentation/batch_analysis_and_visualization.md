# Batch Analysis API Documentation

## BatchAnalysis Class

The `BatchAnalysis` class provides a comprehensive pipeline for processing Traction Force Microscopy (TFM) data in batch mode. It coordinates multiple analysis steps including preprocessing, displacement analysis, force calculation, mask creation, and stress analysis.

### Constructor

```python
BatchAnalysis(config: dict)
```

Initializes the batch analysis pipeline with configuration settings.

#### Parameters
- `config` (dict): Configuration dictionary containing:
  - root_folders: List of folders to process
  - input_files: Dict of input file names
    - beads: Bead image stack filename
    - reference: Reference image filename
    - cells: Cell image stack filename (optional)
  - analysis_steps: Dict of enabled analysis steps
    - preprocessing: bool
    - displacement: bool
    - force: bool
    - create_masks: bool
    - stress: bool
  - parameters: Dict of analysis parameters
  - visualizations: Dict of enabled visualizations

### Alternative Constructor

```python
@classmethod
def from_yaml(cls, yaml_path: str) -> 'BatchAnalysis'
```

Creates a BatchAnalysis instance from a YAML configuration file.

#### Parameters
- `yaml_path` (str): Path to YAML configuration file

#### Example Usage
```python
analyzer = BatchAnalysis.from_yaml("config.yaml")
analyzer.process_all_folders()
```

### Main Methods

#### process_all_folders

```python
process_all_folders() -> None
```

Processes all folders specified in the configuration sequentially.

##### Example Usage
```python
config = {
    'root_folders': ['experiment1', 'experiment2'],
    'analysis_steps': {
        'preprocessing': True,
        'displacement': True,
        'force': True
    }
}
analyzer = BatchAnalysis(config)
analyzer.process_all_folders()
```

#### process_folder

```python
process_folder(folder_path: str) -> None
```

Processes a single folder containing TFM experiment data.

##### Parameters
- `folder_path` (str): Path to folder containing raw data

### Processing Methods

#### _execute_preprocessing

```python
_execute_preprocessing(
    folder: Path,
    tfm_folder: Path
) -> Optional[dict]
```

Executes preprocessing of bead and cell images.

##### Parameters
- `folder` (Path): Input folder path
- `tfm_folder` (Path): Output folder path

##### Returns
- Optional[dict]: Preprocessed data including:
  - beads: Processed bead images
  - reference: Processed reference image
  - cells: Processed cell images (optional)
  - parameters: Processing parameters used

#### _execute_displacement_analysis

```python
_execute_displacement_analysis(
    tfm_folder: Path,
    preprocessed_data: Optional[dict]
) -> Optional[dict]
```

Executes displacement field analysis.

##### Parameters
- `tfm_folder` (Path): Output folder path
- `preprocessed_data` (Optional[dict]): Preprocessed image data

##### Returns
- Optional[dict]: Displacement analysis results

#### _execute_force_analysis

```python
_execute_force_analysis(
    tfm_folder: Path,
    displacement_data: DisplacementResult
) -> Optional[dict]
```

Executes force field calculation using FTTC.

##### Parameters
- `tfm_folder` (Path): Output folder path
- `displacement_data` (DisplacementResult): Displacement analysis results

##### Returns
- Optional[dict]: Force analysis results

#### _execute_mask_creation

```python
_execute_mask_creation(
    tfm_folder: Path,
    cell_images: np.ndarray
) -> Optional[np.ndarray]
```

Creates binary masks from cell images.

##### Parameters
- `tfm_folder` (Path): Output folder path
- `cell_images` (np.ndarray): Preprocessed cell images

##### Returns
- Optional[np.ndarray]: Binary masks

#### _execute_stress_analysis

```python
_execute_stress_analysis(
    tfm_folder: Path,
    mask_data: np.ndarray,
    force_data: FTTCResult
) -> Optional[dict]
```

Executes stress field calculation using MSM.

##### Parameters
- `tfm_folder` (Path): Output folder path
- `mask_data` (np.ndarray): Binary masks
- `force_data` (FTTCResult): Force analysis results

##### Returns
- Optional[dict]: Stress analysis results

### Output Structure

The analysis creates a 'TFM_data' subdirectory in each processed folder containing:

```
TFM_data/
├── preprocessed_beads.tif        # Processed bead images
├── preprocessed_reference.tif    # Processed reference image
├── preprocessed_cells.tif        # Processed cell images (if available)
├── displacements.npy            # Displacement field data
├── traction_forces.npy          # Force field data
├── masks.tif                    # Binary masks
├── stress_results.npy           # Stress tensor data
└── processing_log.txt           # Processing log
```

### Logging

The class includes a custom TeeLogger that:
- Captures all console output
- Writes to both console and log file
- Includes timestamps and progress information
- Generates processing summaries with timing information

### Error Handling

The implementation includes comprehensive error handling:
- Each processing step is independently validated
- Failures in one step don't prevent execution of others
- Detailed error messages are logged
- Results from failed steps are safely skipped

### Configuration Parameters

Key configuration parameters include:

#### Preprocessing
- rolling_ball_radius: Background subtraction parameter
- gaussian_sigma: Smoothing parameter
- min/max_intensity_percentile: Intensity scaling range

#### Displacement Analysis
- tau, lambda_, theta: Optical flow parameters
- nscales, warps: Multi-scale analysis parameters
- downscale_factor: Spatial downsampling factor

#### Force Analysis
- young_modulus: Substrate Young's modulus
- poisson_ratio_substrate: Substrate Poisson ratio
- regularization: FTTC regularization parameter

#### Stress Analysis
- poisson_ratio_cells: Cell monolayer Poisson ratio
- density_factor: Mesh density parameter
- mesh_algorithm: Choice of meshing algorithm


## BatchVisualizationSaver Class

The `BatchVisualizationSaver` class provides methods for generating and saving visualizations of TFM analysis results. It supports creation of GIF animations for various analysis outputs including bead tracking, displacement fields, force fields, and stress distributions. This class is used by the BatchAnalysis Class to visualize its analysis reults.

### Service Constructor

```python
BatchVisualizationSaver(base_folder: str)
```

Initializes the visualization service with output path configuration.

#### Parameters
- `base_folder` (str): Base directory where visualizations will be saved
  - Creates a "TFM_visualizations" subdirectory if it doesn't exist

### Main Methods

#### save_bead_overlay

```python
save_bead_overlay(
    bead_stack: np.ndarray,
    reference_image: np.ndarray,
    fps: int = 10
) -> None
```

Creates a color-coded overlay animation showing bead positions relative to the reference.

##### Parameters
- `bead_stack` (np.ndarray): Stack of bead images (t, y, x)
- `reference_image` (np.ndarray): Reference image (y, x)
- `fps` (int, optional): Frames per second for output GIF

##### Example Usage
```python
visualizer = BatchVisualizationSaver("experiment_folder")
visualizer.save_bead_overlay(
    bead_stack=preprocessed_beads,
    reference_image=reference_image,
    fps=15
)
```

#### save_displacement_visualization

```python
save_displacement_visualization(
    displacement_results: DisplacementResult,
    fps: int = 10
) -> None
```

Creates an animation showing displacement fields as color maps with vector overlays.

##### Parameters
- `displacement_results` (DisplacementResult): Contains:
  - displacement_field: Array of displacement vectors
  - parameters: Visualization parameters including:
    - d_max: Maximum displacement for color scaling
    - disp_arrow_scale: Vector arrow scaling factor
    - disp_vector_stride: Spacing between vectors
- `fps` (int, optional): Frames per second for output GIF

#### save_force_visualization

```python
save_force_visualization(
    force_results: FTTCResult,
    fps: int = 10
) -> None
```

Creates an animation showing traction force fields as color maps with vector overlays.

##### Parameters
- `force_results` (FTTCResult): Contains:
  - force_field: Array of force vectors
  - parameters: Visualization parameters including:
    - f_max: Maximum force for color scaling
    - force_arrow_scale: Vector arrow scaling factor
    - force_vector_stride: Spacing between vectors
- `fps` (int, optional): Frames per second for output GIF

#### save_force_cell_overlay

```python
save_force_cell_overlay(
    force_results: FTTCResult,
    cell_images: np.ndarray,
    fps: int = 10
) -> None
```

Creates an animation overlaying force vectors on phase contrast cell images.

##### Parameters
- `force_results` (FTTCResult): Force field data and parameters
- `cell_images` (np.ndarray): Stack of cell phase contrast images (t, y, x)
- `fps` (int, optional): Frames per second for output GIF

#### save_stress_visualization

```python
save_stress_visualization(
    stress_results: MSMResult,
    plot_sigma_xx: bool = True,
    plot_sigma_yy: bool = True,
    plot_normal_stress: bool = True,
    fps: int = 10
) -> None
```

Creates animations for different components of the stress tensor field.

##### Parameters
- `stress_results` (MSMResult): Contains:
  - stress_tensor: Array of stress tensors
  - parameters: Visualization parameters including:
    - max_stress: Maximum stress for color scaling
- `plot_sigma_xx` (bool): Generate XX normal stress visualization
- `plot_sigma_yy` (bool): Generate YY normal stress visualization
- `plot_normal_stress` (bool): Generate average normal stress visualization
- `fps` (int, optional): Frames per second for output GIFs

#### save_mesh_visualization

```python
save_mesh_visualization(
    stress_results: MSMResult,
    fps: int = 10
) -> None
```

Creates an animation showing the finite element mesh evolution.

##### Parameters
- `stress_results` (MSMResult): Contains:
  - nodes: List of node coordinates for each frame
  - elements: List of element connectivity for each frame
  - stress_shape: Tuple of (height, width) for output sizing
- `fps` (int, optional): Frames per second for output GIF

### Visualization Features

The visualization system provides several key features:
1. Automatic colormap selection:
   - 'viridis' for displacement fields
   - 'inferno' for force fields
   - 'seismic' for stress fields
2. Vector field visualization:
   - Adaptive scaling based on maximum values
   - Configurable stride for clarity
   - Color-coding by magnitude
3. Overlay capabilities:
   - Bead tracking with two-color overlay
   - Force vectors on cell images
4. High-quality output:
   - Anti-aliased rendering
   - Configurable frame rate
   - Looping GIF animation
5. Colorbar legends:
   - Physical units display
   - Automatic range scaling
   - Clear labeling

### Output Format

All visualizations are saved as GIF animations with the following characteristics:
- Looping enabled (loop=0)
- No color palette optimization for better quality
- Consistent naming convention:
  - bead_overlay.gif
  - displacement_map.gif
  - force_map.gif
  - force_cell_overlay.gif
  - sigma_xx.gif
  - sigma_yy.gif
  - normal_stress.gif
  - mesh_visualization.gif