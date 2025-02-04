# Displacement Analysis API Documentation

## DisplacementService Class

The `DisplacementService` class provides a high-level interface for calculating displacement fields between microscopy images using TV-L1 optical flow. It handles parameter management, data validation, and supports both single-frame and time series analysis.

### Service Constructor

```python
DisplacementService(params: DisplacementParameters)
```

Initializes the displacement service with analysis parameters.

#### Parameters
- `params` (DisplacementParameters): Configuration including:
  - TV-L1 algorithm parameters (tau, lambda_, theta, etc.)
  - Physical parameters (pixel size, frame interval)
  - Processing options (downscaling factor)
  - Visualization settings

### Main Methods

#### calculate_displacement_field

```python
calculate_displacement_field(
    reference: np.ndarray,
    target: np.ndarray
) -> Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]
```

Calculates displacement fields between reference and target images using optical flow.

##### Parameters
- `reference` (np.ndarray): Reference (fixed) image (2D array)
- `target` (np.ndarray): Target image(s)
  - 2D array for single frame
  - 3D array (t, y, x) for time series

##### Returns
Generator yielding progress updates and returning final results:
- Yields: (current_displacement_field, frame_number, total_frames)
- Returns: DisplacementResult through StopIteration.value

##### Example Usage
```python
# Initialize service
params = DisplacementParameters(
    pixel_size=0.1,  # 0.1 μm per pixel
    downscale_factor=4,
    tau=0.25,
    lambda_=0.15
)
service = DisplacementService(params)

# Get the generator
disp_generator = service.calculate_displacement_field(ref_img, target_imgs)

# Process intermediate results
try:
    while True:
        # Get next frame result
        disp_field, frame, total = next(disp_generator)
        print(f"Processed frame {frame}/{total}")
except StopIteration as e:
    # Get final result from generator's return value
    final_result = e.value

# Access results
print(f"Final displacement field shape: {final_result.displacement_field.shape}")
print(f"Max displacement: {np.max(final_result.displacement_field)} μm")
```

### Result Type

#### DisplacementResult

Dataclass containing calculation results:

```python
@dataclass
class DisplacementResult:
    displacement_field: np.ndarray  # Shape: (t, y, x, 2), units in μm
    original_shape: tuple          # Original image shape (y, x)
    displacement_field_shape: tuple # Result field shape (y, x)
    parameters: DisplacementParameters
    physical_scale: dict          # Physical scaling information
```

The `physical_scale` dictionary contains:
- pixel_size: Size of each pixel
- grid_spacing: Effective grid spacing after downsampling
- time_interval: Time between frames
- displacement_units: Displacement units (μm)
- grid_spacing_units: Spatial units (μm)
- time_interval_units: Time units (min)

## DisplacementAnalyzer Class

The `DisplacementAnalyzer` class implements the core displacement analysis using the TV-L1 optical flow algorithm. This class is typically used through the DisplacementService, but can be used directly for more control over the analysis process.

### Constructor

```python
DisplacementAnalyzer(params: Optional[DisplacementParameters] = None)
```

#### Parameters
- `params` (DisplacementParameters, optional): Algorithm parameters including:
  - tau: Time step for TV-L1
  - lambda_: Weight parameter for data term
  - theta: Weight parameter for gradient term
  - nscales: Number of scales for pyramid
  - warps: Number of warpings per scale
  - epsilon: Stopping criterion threshold
  - inner_iterations: Inner iteration count
  - outer_iterations: Outer iteration count
  - scale_step: Scale step for pyramid
  - median_filtering: Whether to apply median filtering

### Key Methods

#### calculate_flow

```python
calculate_flow(reference: np.ndarray, moving: np.ndarray) -> np.ndarray
```

Calculates optical flow between two images at full resolution.

##### Parameters
- `reference` (np.ndarray): Reference image
- `moving` (np.ndarray): Moving image
  Both should be 2D arrays of same shape

##### Returns
- np.ndarray: Flow field (H, W, 2) containing (dx, dy) displacements in pixels

#### downscale_flow

```python
downscale_flow(flow: np.ndarray, factor: int) -> np.ndarray
```

Downscales a flow field while preserving vector information.

##### Parameters
- `flow` (np.ndarray): Input flow field (H, W, 2)
- `factor` (int): Downscaling factor

##### Returns
- np.ndarray: Downscaled flow field (H/factor, W/factor, 2)

## Algorithm Details

The implementation uses the TV-L1 optical flow algorithm, which is particularly suitable for microscopy analysis because:
- It preserves discontinuities in the displacement field
- Is robust to brightness changes
- Provides sub-pixel accuracy

The algorithm minimizes an energy functional combining:
1. Data term (L1 norm of brightness constancy)
2. Total variation regularization
3. Additional constraints for numerical stability

Processing steps include:
1. Image normalization
2. Multi-scale pyramid decomposition
3. Iterative optimization at each scale
4. Optional median filtering
5. Conversion to physical units