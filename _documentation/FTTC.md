# FTTC API Documentation

## FTTCService Class

The `FTTCService` class provides a high-level interface for calculating traction forces from displacement field measurements. It handles data validation, parameter management, and both single-frame and time series calculations.

### Service Constructor

```python
FTTCService(params: FTTCParameters)
```

Initializes the FTTC service with calculation parameters.

#### Parameters
- `params` (FTTCParameters): Configuration including:
  - Material properties (Young's modulus, Poisson ratio)
  - Calculation parameters (regularization, downscaling)
  - Visualization settings

### Main Methods

#### calculate_forces

```python
calculate_forces(displacement_field: np.ndarray) -> Generator[Tuple[np.ndarray, int, int], None, FTTCResult]
```

Calculates traction forces from displacement field data, supporting both single frames and time series.

##### Parameters
- `displacement_field` (np.ndarray): Displacement measurements with shape:
  - (y, x, 2) for single frame
  - (t, y, x, 2) for time series
  where final dimension contains (dx, dy) displacements in μm

##### Returns
Generator yielding progress updates and returning final results:
- Yields: (current_force_field, frame_number, total_frames)
- Returns: FTTCResult through StopIteration.value

##### Example Usage
```python
# Initialize service
service = FTTCService(params)

# Get the generator
force_generator = service.calculate_forces(displacements)

# Process intermediate results
try:
    while True:
        # Get next frame result
        force_field, frame, total = next(force_generator)
        print(f"Processed frame {frame}/{total}")
except StopIteration as e:
    # Get final result from generator's return value
    final_result = e.value

# Access results
print(f"Final force field shape: {final_result.force_field.shape}")
print(f"Max force: {np.max(final_result.force_field)} Pa")
```

#### find_optimal_regularization

```python
find_optimal_regularization(displacement_field: np.ndarray) -> float
```

Finds optimal regularization parameter using Generalized Cross-Validation.

##### Parameters
- `displacement_field` (np.ndarray): Displacement field with shape (y, x, 2)

##### Returns
- float: Optimal regularization parameter λ

### Result Type

#### FTTCResult

Dataclass containing calculation results:

```python
@dataclass
class FTTCResult:
    force_field: np.ndarray  # Shape: (t, y, x, 2) for time series, units in Pa
    original_shape: tuple    # Original displacement field shape (y, x)
    force_shape: tuple      # Force field shape (y, x)
    parameters: FTTCParameters
    physical_scale: dict    # Physical scaling information
```

The `physical_scale` dictionary contains:
- pixel_size: Size of each pixel
- grid_spacing: Effective grid spacing after downsampling
- time_interval: Time between frames
- force_units: Force units (Pa)
- grid_spacing_units: Spatial units (μm)
- time_interval_units: Time units (min)

---

## FTTC Class


The `FTTC` (Fourier Transform Traction Cytometry) class implements force calculations for Traction Force Microscopy (TFM) using the FTTC method with Generalized Cross-Validation (GCV) for regularization parameter optimization.

## Class Constructor

### FTTC(params: FTTCParameters)

Initializes the FTTC calculator with substrate material properties and calculation parameters.

#### Parameters

- `params` (FTTCParameters): Configuration object containing:
  - `young_modulus` (float): Young's modulus of the substrate in Pascals (Pa)
  - `poisson_ratio_substrate` (float): Poisson ratio of the substrate
  - `lanczos_exp` (float): Lanczos filter exponent for noise reduction
  - `gel_height` (float, optional): Gel height in micrometers for finite thickness correction. Use None for infinite thickness.

## Main Methods

### calculate_traction

```python
def calculate_traction(
    self,
    displacements: Tuple[np.ndarray, np.ndarray],
    pixel_size: float,
    downscale_factor: int = 1,
    regularization: float = None
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
```

Calculates traction forces from displacement field measurements using FTTC.

#### Parameters

- `displacements` (Tuple[np.ndarray, np.ndarray]): Displacement field components
  - Shape: H × W × 2 array containing:
    - dx: x-direction displacements (displacements[..., 0])
    - dy: y-direction displacements (displacements[..., 1])
  - Units: micrometers (μm)
  - Note: Fields represent how far each point in the gel has moved from its original position

- `pixel_size` (float):
  - Physical size of each pixel in the displacement field
  - Units: micrometers (μm)
  - Example: 0.1 for a 100x objective with 0.1 μm/pixel

- `downscale_factor` (int, default=1):
  - Factor representing spatial downsampling already applied to the displacement field
  - Used to correctly scale pixel size for force calculations

- `regularization` (float, optional):
  - Tikhonov regularization parameter (λ) for the inverse problem
  - Units: dimensionless
  - If None, automatically determined using GCV
  - Typical range: 1e-6 to 1e-3
  - Higher values produce smoother force fields but may underestimate peak forces

#### Returns

Returns a tuple containing:

1. Coordinate grids `(x, y)`:
   - `x, y`: np.ndarray, shape (H, W)
   - Physical position corresponding to each point in the force field
   - Units: micrometers (μm)

2. Forces array:
   - Shape: 2 × H × W array containing:
     - forces[0]: x-direction forces
     - forces[1]: y-direction forces
   - Units: N/m² (Pascals)
   - Represents forces exerted by cells on the substrate at each point

#### Example Usage

```python
# Initialize FTTC calculator with substrate properties
fttc = FTTC(FTTCParameters(
    young_modulus=10000,  # 10 kPa
    poisson_ratio_substrate=0.5,
    lanczos_exp=2,
    gel_height=None  # infinite thickness
))

# Calculate forces from displacement field
(x, y), forces = fttc.calculate_traction(
    displacements=(dx, dy),  # displacement fields in μm
    pixel_size=0.1,  # 0.1 μm per pixel
    downscale_factor=4  # if data was previously downsampled
)

# Calculate force magnitude
force_magnitude = np.sqrt(forces[0]**2 + forces[1]**2)
```

## Implementation Details

The calculation involves several key steps:

1. **Preprocessing**:
   - Interpolation of displacement field to regular grid
   - Conversion of pixel coordinates to physical units

2. **Force Calculation**:
   - Fourier transform of displacement field
   - Application of Green's function (with optional gel height correction)
   - Regularization using Tikhonov method
   - Inverse Fourier transform to obtain force field

3. **Regularization**:
   - Automatic parameter selection using GCV if not specified
   - Optimization to find optimal regularization strength
   - Handles noise in displacement measurements

4. **Post-processing**:
   - Lanczos filtering for noise reduction
   - Conversion to physical units
   - Generation of coordinate grids

## References

The implementation is based on:
- DirectMethod package (MIT License)
- pyTFM package (GNU GPL v3.0)
- Butler et al. (2002)
- Sabass et al. (2008)
- Trepat et al. (2009)
- Hansen's Regularization Tools
- Golub, Heath, & Wahba (2012)