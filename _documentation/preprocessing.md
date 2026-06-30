# Preprocessing API Documentation

## Backend Function API

The preprocessing backend provides functions for validating images and preprocessing single frames or stacks. Widgets and batch analysis call these functions directly.

#### Parameters
- `params` (PreprocessingParameters): Configuration including:
  - Intensity scaling parameters (percentile ranges)
  - Gaussian smoothing parameters (sigma values)
  - Registration mode and parameters
  - Separate parameter sets for cell and bead images

### Main Methods

#### preprocess_stack

```python
preprocess_stack(
    image_stack: Optional[np.ndarray] = None,
    reference_image: Optional[np.ndarray] = None,
    is_cell: bool = False
) -> Generator[Tuple[PreprocessingIntermediateResult, int, int], None, List[PreprocessingIntermediateResult]]
```

Processes a stack of images with progress tracking.

##### Parameters
- `image_stack` (np.ndarray, optional): Stack of images to process
  - 2D array for single frame
  - 3D array (t, y, x) for time series
- `reference_image` (np.ndarray, optional): Reference image for registration
- `is_cell` (bool): Whether images contain cells (affects parameter selection)

##### Returns
Generator yielding progress updates and returning final results:
- Yields: (current_result, frame_number, total_frames)
- Returns: List of PreprocessingIntermediateResult through StopIteration.value

##### Example Usage
```python
params = PreprocessingParameters(
    min_intensity_percentile=1,
    max_intensity_percentile=99,
    gaussian_sigma=1.0
)

# Get the generator
prep_generator = preprocess_stack(image_stack, params)

# Process intermediate results
try:
    while True:
        # Get next frame result
        result, frame, total = next(prep_generator)
        print(f"Processed frame {frame}/{total}")
except StopIteration as e:
    # Get final results from generator's return value
    final_results = e.value

# Access results
print(f"Processed {len(final_results)} frames")
print(f"Final shape: {final_results[0].processed_image.shape}")
```

#### preprocess_frame

```python
preprocess_frame(
    image: np.ndarray,
    is_cell: bool = False,
    reference_image: Optional[np.ndarray] = None
) -> PreprocessingIntermediateResult
```

Processes a single microscopy image frame.

##### Parameters
- `image` (np.ndarray): Input image to process
- `is_cell` (bool): Whether the image contains cells
- `reference_image` (np.ndarray, optional): Reference for registration

##### Returns
- PreprocessingIntermediateResult: Complete processing results

### Result Type

#### PreprocessingIntermediateResult

Dataclass containing preprocessing results:

```python
@dataclass
class PreprocessingIntermediateResult:
    processed_image: np.ndarray
    transform_matrix: Optional[np.ndarray] = None
    info: Dict[str, Any] = None
```

The `info` dictionary contains:
- original_dtype: Original data type
- original_range: (min, max) of original data
- original_mean: Mean of original data
- original_std: Standard deviation of original data
- final_mean: Mean after processing
- final_std: Standard deviation after processing
- intensity_range: (min, max) used for scaling
- gaussian_sigma: Applied Gaussian smoothing sigma

## ImageProcessor Class

The `ImageProcessor` class implements core image processing operations for microscopy analysis. This class provides stateless methods for individual processing steps; higher-level callers usually use `preprocess_frame` or `preprocess_stack`.

### Constructor

```python
ImageProcessor()
```

No parameters required as all methods are stateless.

### Key Methods

#### apply_gaussian_filter

```python
apply_gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray
```

Applies Gaussian smoothing for noise reduction.

##### Parameters
- `image` (np.ndarray): Input image
- `sigma` (float): Standard deviation of Gaussian kernel

##### Returns
- np.ndarray: Filtered image in same dtype as input

#### apply_intensity_scaling

```python
apply_intensity_scaling(
    image: np.ndarray,
    min_percentile: float,
    max_percentile: float
) -> tuple[np.ndarray, tuple[float, float]]
```

Normalizes image intensities using percentile-based scaling.

##### Parameters
- `image` (np.ndarray): Input image
- `min_percentile` (float): Lower percentile for scaling (0-100)
- `max_percentile` (float): Upper percentile for scaling (0-100)

##### Returns
- Tuple containing:
  - Normalized image with values in [0, 1]
  - (min_val, max_val) used for scaling

#### register_to_reference

```python
register_to_reference(
    moving_image: np.ndarray,
    reference_image: np.ndarray,
    mode: str
) -> tuple[np.ndarray, np.ndarray]
```

Registers images using Enhanced Correlation Coefficient maximization.

##### Parameters
- `moving_image` (np.ndarray): Image to be registered
- `reference_image` (np.ndarray): Reference image
- `mode` (str): 'translation' or 'rigid'

##### Returns
- Tuple containing:
  - Registered image in same dtype as input
  - 2x3 transformation matrix

## Processing Pipeline

The preprocessing implementation follows a systematic pipeline:

1. Noise Reduction
   - Gaussian filtering with configurable sigma
   - Edge-preserving implementation
   - Separate parameters for cell/bead images

2. Intensity Normalization
   - Percentile-based scaling to [0, 1]
   - Handles outliers robustly
   - Separate ranges for cell/bead images

3. Registration (optional)
   - ECC-based alignment
   - Supports translation-only or rigid registration
   - Automatic intensity normalization
   - Robust error handling with identity fallback

The pipeline is optimized for microscopy data, providing:
- Dtype preservation
- Proper handling of both 8-bit and 16-bit images
- Accurate image registration
