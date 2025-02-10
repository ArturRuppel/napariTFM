from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Generator

import numpy as np
from skimage.transform import resize

from napariTFM.backend.mesh_generator import MeshParameters, MeshGenerator
from napariTFM.backend.msm import MonolayerStressMicroscopy
from napariTFM.backend.parameter_dataclasses import MSMParameters


@dataclass
class MSMResult:
    """Results from Monolayer Stress Microscopy calculations.

    Attributes:
        stress_tensor (np.ndarray): Calculated stress tensor with shape (t, y, x, 2, 2).
            t is time points (1 for single frame), final dimensions contain the stress tensor
            components [σxx, σxy; σyx, σyy] in mN/m.
        nodes (List[np.ndarray]): List of node coordinate arrays, one per frame.
            Each array has shape (n_nodes, 2) containing (x,y) coordinates.
        elements (List[np.ndarray]): List of element connectivity arrays, one per frame.
            Each array has shape (n_elements, 3) containing node indices.
        condition_number (float): Condition number of the system matrix,
            indicating numerical stability. Lower values (closer to 1) are better.
        residual (float): Residual norm of the solution, indicating accuracy.
            Lower values indicate better solution quality.
        parameters (MSMParameters): Parameters used for calculation
        physical_scale (dict): Physical scaling information including:
            - pixel_size: Size of each pixel
            - grid_spacing: Effective grid spacing after downsampling
            - time_interval: Time between frames
            - stress_units: Stress units (mN/m)
            - grid_spacing_units: Spatial units (μm)
            - time_interval_units: Time units (min)
        original_shape (tuple): Original force field shape (y, x)
        stress_shape (tuple): Stress field shape (y, x)
    """
    stress_tensor: np.ndarray  # Shape: (frames, height, width, 2, 2) for xx, yy, xy, yx components
    nodes: List[np.ndarray]  # Shape: (n_nodes, 2) for node coordinates
    elements: List[np.ndarray]  # Shape: (n_elements, 3) for element connectivity
    condition_number: float
    residual: float
    parameters: MSMParameters
    physical_scale: dict  # Dictionary containing physical scaling information
    original_shape: tuple  # Original force field shape (y, x)
    stress_shape: tuple  # Stress field shape (y, x)


class MSMService:
    """Service layer for handling Monolayer Stress Microscopy calculations.

    This class provides a high-level interface for calculating internal stresses
    in cell monolayers from traction force measurements. It handles:
    - Mask creation and processing
    - Finite element mesh generation
    - Stress field calculation
    - Parameter validation and management
    - Progress tracking for time series data

    The service implements both the mechanical equilibrium solver and necessary
    pre/post-processing steps for accurate stress field reconstruction.
    """

    def __init__(self, params: MSMParameters):
        """Initialize MSM service with calculation parameters.

        Args:
            params (MSMParameters): Configuration for MSM calculations including:
                - Material properties (Young's modulus, Poisson ratio)
                - Mesh parameters (density, algorithm)
                - Mask processing settings
                - Physical scaling information
                - Numerical parameters

        Raises:
            ValueError: If any parameters are invalid
        """
        is_valid, error_msg = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(error_msg)

        self.analyzer = MonolayerStressMicroscopy(
            params,
            mask=None,  # Will be set during calculation
        )
        self.params = params

    def create_preview_mask(
            self,
            image: np.ndarray,
            threshold_percentile: float,
            dilation: int,
            smoothing_sigma: float,
            target_shape: Optional[Tuple[int, int]] = None,
            downscale_factor: int = 1
    ) -> np.ndarray:
        """Create a preview mask from a single image frame.

        Creates a binary mask suitable for MSM analysis through:
        1. Intensity-based thresholding
        2. Morphological operations (dilation, hole filling)
        3. Gaussian smoothing
        4. Optional resizing for analysis/visualization

        Args:
            image (np.ndarray): 2D input image
            threshold_percentile (float): Threshold percentile for mask creation (0-100)
            dilation (int): Number of pixels to dilate the mask
            smoothing_sigma (float): Sigma value for Gaussian smoothing
            target_shape (tuple, optional): Shape to resize analysis mask to (height, width)
            downscale_factor (int): Factor to upscale visualization mask by

        Returns:
            np.ndarray: Binary mask ready for visualization

        Example:
            >>> phase_image = load_microscopy_image()
            >>> preview = service.create_preview_mask(
            ...     image=phase_image,
            ...     threshold_percentile=5,
            ...     dilation=10,
            ...     smoothing_sigma=5.0,
            ...     target_shape=(512, 512),
            ...     downscale_factor=2
            ... )
        """
        # Create base mask using MSM class method
        base_mask = MonolayerStressMicroscopy.create_mask_from_image(
            image,
            threshold_percentile=threshold_percentile,
            dilation=dilation,
            smoothing_sigma=smoothing_sigma
        )

        # Handle analysis mask resizing if target shape is provided
        if target_shape is not None and base_mask.shape != target_shape:
            analysis_mask = resize(
                base_mask.astype(float),
                target_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
        else:
            analysis_mask = base_mask

        # Upscale for visualization if needed
        if downscale_factor > 1:
            vis_shape = (
                analysis_mask.shape[0] * downscale_factor,
                analysis_mask.shape[1] * downscale_factor
            )
            preview_mask = resize(
                analysis_mask.astype(float),
                vis_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
        else:
            preview_mask = analysis_mask

        return preview_mask

    def create_mask_stack(
            self,
            image_stack: np.ndarray,
            params: MSMParameters,
            target_shape: Optional[Tuple[int, int]] = None,
    ) -> Generator[Tuple[np.ndarray, int, int], None, np.ndarray]:
        """Create a stack of analysis masks from an image sequence.

        Processes each frame in the image stack to create corresponding masks,
        yielding intermediate results for progress tracking.

        Args:
            image_stack (np.ndarray): Input images with shape:
                - (y, x) for single frame
                - (t, y, x) for time series
            params (MSMParameters): Parameters for mask creation
            target_shape (tuple, optional): Shape to resize masks to (height, width)

        Yields:
            Tuple[np.ndarray, int, int]:
                - analysis_mask: Current frame's processed mask
                - current_frame: Frame index (1-based)
                - total_frames: Total number of frames

        Returns:
            np.ndarray: Complete stack of analysis masks with shape (t, y, x)

        Example:
            >>> images = load_image_sequence()
            >>> # Process masks with progress tracking
            >>> mask_generator = service.create_mask_stack(
            ...     images, params, target_shape=(512, 512)
            ... )
            >>> for mask, frame, total in mask_generator:
            ...     print(f"Processed frame {frame}/{total}")
            >>> result = mask_generator.send(None)  # Get final stack
        """
        # Handle 2D input
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]

        total_frames = image_stack.shape[0]

        # Initialize mask stacks
        analysis_masks = []

        # Process each frame
        for frame in range(total_frames):
            # Create base mask using MSM class method
            base_mask = MonolayerStressMicroscopy.create_mask_from_image(
                image_stack[frame],
                threshold_percentile=params.threshold,
                dilation=params.dilation,
                smoothing_sigma=params.smoothing_sigma
            )

            # Handle analysis mask resizing
            if target_shape is not None and base_mask.shape != target_shape:
                analysis_mask = resize(
                    base_mask.astype(float),
                    target_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5
            else:
                analysis_mask = base_mask

            # Store masks
            analysis_masks.append(analysis_mask)

            # Yield intermediate results
            yield analysis_mask, frame, total_frames

        # Convert lists to arrays for final return
        analysis_stack = np.stack(analysis_masks)

        return analysis_stack
    def process_mask_data(
            self,
            mask_data: np.ndarray,
            force_field: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Process mask data into required format with validation.

        Performs necessary preprocessing on mask data including:
        1. Binary conversion
        2. Dimensionality checks
        3. Optional resizing to match force field
        4. Validation and warning generation

        Args:
            mask_data (np.ndarray): Input mask data with shape:
                - (y, x) for single frame
                - (t, y, x) for time series
            force_field (np.ndarray, optional): Force field to match dimensions to

        Returns:
            Tuple[np.ndarray, List[str]]:
                - Processed mask data
                - List of warning messages

        Raises:
            ValueError: If mask data is invalid or incompatible
        """
        warnings = []

        if mask_data is None:
            raise ValueError("No mask data provided")

        # Check for multiple values
        unique_values = np.unique(mask_data)
        unique_values = unique_values[unique_values != 0]  # Exclude zero
        if len(unique_values) > 1:
            warnings.append("Multiple non-zero values detected in mask. Converting to binary (0 and 1).")

        # Convert to binary mask
        mask_data = mask_data > 0

        # If this is a single mask, add time dimension
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, ...]

        # Ensure we have a 3D array (time, height, width)
        if mask_data.ndim != 3:
            raise ValueError(f"Mask data must be 2D or 3D, got shape {mask_data.shape}")

        # If force data exists, resize masks to match
        if force_field is not None:
            force_shape = force_field.shape[1:3]  # Get height, width
            if mask_data.shape[1:] != force_shape:
                mask_data = np.stack([
                    resize(
                        frame.astype(float),
                        force_shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False
                    ) > 0.5
                    for frame in mask_data
                ])

        return mask_data, warnings
    def generate_mesh_stack(
            self,
            mask_stack: np.ndarray,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int], None, List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]]:
        """Generate finite element meshes for all frames in the mask stack.

        Creates high-quality triangular meshes suitable for FEM analysis,
        yielding intermediate results for progress tracking.

        Args:
            mask_stack (np.ndarray): Binary masks with shape:
                - (y, x) for single frame
                - (t, y, x) for time series

        Yields:
            Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int]:
                - nodes: Node coordinates array (n_nodes, 2)
                - elements: Element connectivity array (n_elements, 3)
                - quality_metrics: Dictionary of mesh quality metrics
                - current_frame: Frame index (0-based)
                - total_frames: Total number of frames

        Returns:
            List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
                Complete mesh data for all frames

        Example:
            >>> # Single frame preview
            >>> mesh_generator = service.generate_mesh_stack(mask)
            >>> nodes, elements, metrics, frame, total = next(mesh_generator)
            >>> print(f"Generated mesh with {len(nodes)} nodes")

            >>> # Process all frames
            >>> mesh_generator = service.generate_mesh_stack(masks)
            >>> mesh_data = []
            >>> try:
            ...     while True:
            ...         nodes, elements, metrics, frame, total = next(mesh_generator)
            ...         mesh_data.append((nodes, elements, metrics))
            ...         print(f"Frame {frame + 1}/{total}")
            ... except StopIteration as e:
            ...     final_mesh_data = e.value
            """
        # Handle 2D input
        if mask_stack.ndim == 2:
            mask_stack = mask_stack[np.newaxis, ...]

        total_frames = mask_stack.shape[0]

        # Initialize mesh generator with parameters
        mesh_params = MeshParameters(
            mask=mask_stack[0],  # Use first frame for initial setup
            density_factor=self.params.density_factor,
            mesh_algorithm=self._get_algorithm_code(self.params.mesh_algorithm),
            use_optimization=self.params.use_optimization
        )
        mesh_generator = MeshGenerator(mesh_params)

        # Initialize results list
        mesh_results = []

        # Process each frame
        for frame in range(total_frames):
            # Generate mesh for current frame
            nodes, elements = mesh_generator.generate_mesh(mask_stack[frame])

            # Calculate quality metrics
            quality_metrics = mesh_generator.analyze_mesh_quality(nodes, elements)

            # Store results
            mesh_results.append((nodes, elements, quality_metrics))

            # Yield intermediate results
            yield nodes, elements, quality_metrics, frame, total_frames

        return mesh_results

    def calculate_stresses(
            self,
            force_field: np.ndarray,
            masks: np.ndarray,
            mesh_data: Optional[List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]] = None
    ) -> Generator[Tuple[MSMResult, int, int], None, MSMResult]:
        """Calculate stress fields from traction force measurements.

        Implements the core MSM algorithm to compute internal stresses within
        the cell monolayer, yielding intermediate results for progress tracking.
        The calculation includes:
        1. Force preprocessing and balancing
        2. Mesh generation (if not provided)
        3. FEM system assembly and solution
        4. Stress field interpolation
        5. Physical unit conversion

        Args:
            force_field (np.ndarray): Traction forces with shape:
                - (y, x, 2) for single frame
                - (t, y, x, 2) for time series
                containing (tx, ty) components in Pa
            masks (np.ndarray): Binary masks defining monolayer regions
            mesh_data (List[Tuple], optional): Pre-generated mesh data

        Yields:
            Tuple[MSMResult, int, int]:
                - Intermediate calculation result
                - Frame index (1-based)
                - Total number of frames

        Returns:
            MSMResult: Complete calculation results including:
                - Full stress tensor array
                - Mesh data
                - Quality metrics
                - Physical scaling information
                Accessible via StopIteration.value when generator completes

        Example:
            >>> # Calculate stresses with progress tracking
            >>> stress_generator = service.calculate_stresses(
            ...     forces, masks, mesh_data
            ... )
            >>> try:
            ...     while True:
            ...         result, frame, total = next(stress_generator)
            ...         print(f"Frame {frame}/{total}")
            ... except StopIteration as e:
            ...     final_result = e.value
            ...     print(f"Max stress: {np.max(final_result.stress_tensor)} mN/m")
        """
        # Ensure force field is 4D and masks is 3D
        if force_field.ndim == 3:
            force_field = force_field[np.newaxis, ...]
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]

        total_frames = force_field.shape[0]
        stress_shape = (*force_field.shape[1:3], 2, 2)  # (y, x, 2, 2)
        stress_stack = np.zeros((total_frames, *stress_shape), dtype=np.float32)
        nodes_stack = []
        elements_stack = []
        condition_numbers = []
        residuals = []

        physical_scale = self._create_physical_scale()

        def process_frame(frame_idx):
            # Prepare current frame data
            tx = force_field[frame_idx, ..., 0]
            ty = force_field[frame_idx, ..., 1]
            current_mask = masks[frame_idx]

            # Resize mask if needed
            if current_mask.shape != tx.shape:
                current_mask = resize(
                    current_mask.astype(float),
                    tx.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5

            if mesh_data is not None:
                # Use provided mesh data
                nodes, elements, _ = mesh_data[frame_idx]
            else:
                # Generate mesh
                mesh_params = MeshParameters(
                    mask=current_mask,
                    density_factor=self.params.density_factor,
                    mesh_algorithm=self._get_algorithm_code(self.params.mesh_algorithm),
                    use_optimization=self.params.use_optimization
                )
                mesh_generator = MeshGenerator(mesh_params)
                nodes, elements = mesh_generator.generate_mesh(current_mask)

            # Set mask and mesh in analyzer
            self.analyzer.mask = current_mask
            self.analyzer.nodes = nodes
            self.analyzer.elements = elements

            # Calculate stress field
            stress_tensor, condition_number, residual = self.analyzer.calculate_stress_field(tx, ty)

            # Scale stress tensor to mN/m
            stress_tensor *= self.params.downscale_factor * self.params.pixel_size * 1e-3

            return stress_tensor, nodes, elements, condition_number, residual

        for frame in range(total_frames):
            # Process current frame
            stress_tensor, nodes, elements, condition_number, residual = process_frame(frame)

            # Store results
            stress_stack[frame] = stress_tensor
            nodes_stack.append(nodes)
            elements_stack.append(elements)
            condition_numbers.append(condition_number)
            residuals.append(residual)

            # Create intermediate result
            result = MSMResult(
                stress_tensor=stress_stack[:frame + 1],
                nodes=nodes,  # Current frame's nodes
                elements=elements,  # Current frame's elements
                condition_number=condition_number,  # Current frame's condition number
                residual=residual,  # Current frame's residual
                parameters=self.params,
                physical_scale=physical_scale,
                original_shape=force_field.shape[1:3],
                stress_shape=stress_stack.shape[1:3]
            )

            yield result, frame + 1, total_frames

        # Return final result with mean condition number and residual
        return MSMResult(
            stress_tensor=stress_stack,
            nodes=nodes_stack,  # All frames' nodes
            elements=elements_stack,  # All frames' elements
            condition_number=np.mean(np.stack(condition_numbers)),  # Mean condition number
            residual=np.mean(residuals),  # Mean residual
            parameters=self.params,
            physical_scale=physical_scale,
            original_shape=force_field.shape[1:3],
            stress_shape=stress_stack.shape[1:3]
        )

    def _create_physical_scale(self) -> dict:
        """Create physical scale information dictionary."""
        return {
            'pixel_size': self.params.pixel_size,
            'grid_spacing': self.params.pixel_size * self.params.downscale_factor,
            'time_interval': self.params.frame_interval,
            'stress_units': 'mN/m',
            'grid_spacing_units': 'µm',
            'time_interval_units': 'min',
        }

    def _get_algorithm_code(self, algorithm_name: str) -> int:
        """Convert algorithm name to corresponding code."""
        algorithm_map = {
            "frontal-del": 6,
            "delaunay": 5,
            "meshadapt": 1,
            "bamg": 7,
            "fd quads": 8,
            "para pack": 9
        }
        normalized_name = algorithm_name.lower().replace(".", "").replace("-", " ")
        return algorithm_map.get(normalized_name, 6)  # Default to Frontal-Delaunay

    @staticmethod
    def validate_parameters(params: MSMParameters) -> Tuple[bool, str]:
        """Validate MSM calculation parameters.

        Checks all parameters for physical and numerical validity including:
        - Material properties (positive Young's modulus, valid Poisson ratio)
        - Mesh parameters (valid density factor and algorithm)
        - Mask processing settings
        - Physical scaling parameters
        - Numerical thresholds

        Args:
            params (MSMParameters): Parameters to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
                error_message is empty string if valid

        Example:
            >>> params = MSMParameters(young_modulus=1000, ...)
            >>> is_valid, msg = MSMService.validate_parameters(params)
            >>> if not is_valid:
            ...     print(f"Invalid parameters: {msg}")
        """
        if params.density_factor < 0.005:
            return False, "Density factor is too low (< 0.005). This may lead to numerical instabilities."

        if params.density_factor > 0.05:
            return False, "Density factor is too high (> 0.05). This may lead to poor resolution."

        if not 0 <= params.poisson_ratio_cells <= 0.5:
            return False, "Poisson ratio must be between 0 and 0.5"

        if params.threshold < 0 or params.threshold > 100:
            return False, "Threshold percentile must be between 0 and 100"

        if params.dilation < 0:
            return False, "Dilation must be non-negative"

        if params.smoothing_sigma < 0:
            return False, "Smoothing sigma must be non-negative"

        if params.max_stress <= 0:
            return False, "Maximum stress must be positive"

        if params.frame_interval <= 0:
            return False, "Frame interval must be positive"

        if params.pixel_size <= 0:
            return False, "Pixel size must be positive"

        if params.downscale_factor < 1:
            return False, "Downscale factor must be at least 1"

        if params.young_modulus <= 0:
            return False, "Young's modulus must be positive"

        return True, ""

    def update_parameters(self, parameters: MSMParameters):
        """Update MSM parameters"""
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)
        self.params = parameters