# MSM API Documentation

## Backend Function API

The MSM backend provides functions for mask creation, mesh generation, mask/force-field alignment, and stress calculation. Widgets and batch analysis call these functions directly.

#### Parameters
- `params` (MSMParameters): Configuration including:
  - Material properties (Young's modulus, Poisson ratio)
  - Mesh parameters (density, algorithm)
  - Mask processing settings
  - Physical scaling information

### Main Methods

#### create_mask_stack

```python
create_mask_stack(
    image_stack: np.ndarray,
    params: MSMParameters,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Generator[Tuple[np.ndarray, int, int], None, np.ndarray]
```

Creates analysis masks from image data, supporting both single frames and time series.

##### Parameters
- `image_stack` (np.ndarray): Input images with shape:
  - (y, x) for single frame
  - (t, y, x) for time series
- `params` (MSMParameters): Parameters for mask creation
- `target_shape` (tuple, optional): Shape to resize masks to (height, width)

##### Returns
Generator yielding progress updates and returning final masks:
- Yields: (current_mask, frame_number, total_frames)
- Returns: Complete mask stack with shape (t, y, x)

##### Example Usage
```python
# Process masks with progress tracking
mask_generator = create_mask_stack(
    images, params, target_shape=(512, 512)
)

for mask, frame, total in mask_generator:
    print(f"Processed frame {frame}/{total}")

result = mask_generator.send(None)  # Get final stack
```

#### generate_mesh_stack

```python
generate_mesh_stack(
    mask_stack: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int], None, List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]]
```

Generates finite element meshes for all frames in the mask stack.

##### Parameters
- `mask_stack` (np.ndarray): Binary masks with shape:
  - (y, x) for single frame
  - (t, y, x) for time series

##### Returns
Generator yielding progress updates and mesh data:
- Yields: (nodes, elements, quality_metrics, frame_number, total_frames)
  - nodes: Node coordinates array (n_nodes, 2)
  - elements: Element connectivity array (n_elements, 3)
  - quality_metrics: Dictionary of mesh quality metrics
- Returns: List of (nodes, elements, quality_metrics) tuples for all frames

##### Example Usage
```python
# Single frame preview
mesh_generator = generate_mesh_stack(mask, params)
nodes, elements, metrics, frame, total = next(mesh_generator)
print(f"Generated mesh with {len(nodes)} nodes")

# Process all frames
mesh_generator = generate_mesh_stack(masks, params)
mesh_data = []
try:
    while True:
        nodes, elements, metrics, frame, total = next(mesh_generator)
        mesh_data.append((nodes, elements, metrics))
        print(f"Frame {frame + 1}/{total}")
except StopIteration as e:
    final_mesh_data = e.value
```

#### calculate_stresses

```python
calculate_stresses(
    force_field: np.ndarray,
    masks: np.ndarray,
    mesh_data: Optional[List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]] = None
) -> Generator[Tuple[MSMResult, int, int], None, MSMResult]
```

Calculates stress fields from traction force measurements.

##### Parameters
- `force_field` (np.ndarray): Traction forces with shape:
  - (y, x, 2) for single frame
  - (t, y, x, 2) for time series
  containing (tx, ty) components in Pa
- `masks` (np.ndarray): Binary masks defining monolayer regions
- `mesh_data` (List[Tuple], optional): Pre-generated mesh data

##### Returns
Generator yielding progress updates and returning final results:
- Yields: (intermediate_result, frame_number, total_frames)
- Returns: MSMResult through StopIteration.value

##### Example Usage
```python
stress_generator = calculate_stresses(forces, masks, params, mesh_data=mesh_data)

try:
    while True:
        result, frame, total = next(stress_generator)
        print(f"Frame {frame}/{total}")
except StopIteration as e:
    final_result = e.value
    print(f"Max stress: {np.max(final_result.stress_tensor)} mN/m")
```

### Result Type

#### MSMResult

Dataclass containing calculation results:

```python
@dataclass
class MSMResult:
    stress_tensor: np.ndarray  # Shape: (t, y, x, 2, 2) for time series
    nodes: List[np.ndarray]    # List of node coordinate arrays
    elements: List[np.ndarray] # List of element connectivity arrays
    condition_number: float    # Matrix condition number
    residual: float           # Solution residual
    parameters: MSMParameters # Calculation parameters
    physical_scale: dict      # Physical scaling information
    original_shape: tuple     # Force field shape (y, x)
    stress_shape: tuple       # Stress field shape (y, x)
```

The `stress_tensor` array contains components:
- [..., 0, 0], σxx (normal stress in x)
- [..., 1, 1], σyy (normal stress in y)
- [..., 0, 1] and [..., 1, 0], σxy (shear stress)
Units: mN/m

The `physical_scale` dictionary contains:
- pixel_size: Size of each pixel
- grid_spacing: Effective grid spacing after downsampling
- time_interval: Time between frames
- stress_units: Stress units (mN/m)
- grid_spacing_units: Spatial units (μm)
- time_interval_units: Time units (min)

## Implementation Details

The calculation involves several key steps:

1. **Mask Processing**:
   - Thresholding and morphological operations
   - Optional resizing to match force field dimensions
   - Validation and quality checks

2. **Mesh Generation**:
   - Adaptive triangulation based on mask geometry
   - Quality metrics calculation
   - Optional mesh optimization

3. **Stress Calculation**:
   - Force preprocessing and balance correction
   - FEM system assembly with proper constraints
   - Solution using regularized LSQR
   - Stress field interpolation
   - Physical unit conversion

4. **Quality Control**:
   - Condition number monitoring
   - Residual calculation
   - Mesh quality metrics
   - Solution stability checks

## References

The implementation is based on:
- pyTFM package (GNU GPL v3.0)
- SolidsPy package (MIT License)
- Bauer et al. (2021)
- Tambe et al. (2011)
- Tambe et al. (2013)
