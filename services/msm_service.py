from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Generator

import numpy as np
from skimage.transform import resize

from backend.mesh_generator import MeshParameters, MeshGenerator
from backend.msm import MonolayerStressMicroscopy
from backend.parameter_dataclasses import MSMParameters


@dataclass
class MSMResult:
    """Results from stress field calculation"""
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
    """Service layer handling business logic for Monolayer Stress Microscopy calculations."""

    def __init__(self, params: MSMParameters):
        """
        Initialize the MSM service with calculation parameters.

        Parameters
        ----------
        params : MSMParameters
            Parameters for the MSM calculations
        """
        is_valid, error_msg = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(error_msg)

        self.analyzer = MonolayerStressMicroscopy(
            mask=None,  # Will be set during calculation
            density_factor=params.density_factor,
            mesh_algorithm=self._get_algorithm_code(params.mesh_algorithm),
            use_optimization=params.use_optimization,
            poisson_ratio=params.poisson_ratio_cells,
            young_modulus=params.young_modulus
        )
        self.params = params

    def create_mask_stack(
            self,
            image_stack: np.ndarray,
            params: MSMParameters,
            target_shape: Optional[Tuple[int, int]] = None,
    ) -> Generator[Tuple[np.ndarray, int, int], None, np.ndarray]:
        """
        Create analysis and visualization mask stacks as a generator that yields intermediate results.

        Parameters
        ----------
        image_stack : np.ndarray
            3D array of images (frames, height, width) or 2D single image
        params : MSMParameters
            Parameters containing threshold, dilation, and smoothing settings
        target_shape : tuple, optional
            Shape to resize analysis masks to (height, width)


        Yields
        ------
        Tuple[np.ndarray, np.ndarray, int, int]
            (analysis_mask, visualization_mask, current_frame, total_frames)
            Yields each processed frame's masks along with progress information

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (analysis_mask_stack, visualization_mask_stack)
            Final complete stacks after all processing
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
        """Process mask data into required format with validation and resizing."""
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
        """
        Generate meshes for all frames in the mask stack as a generator that yields intermediate results.

        Parameters
        ----------
        mask_stack : np.ndarray
            3D array of masks (frames, height, width) or 2D single mask

        Yields
        ------
        Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int]
            (nodes, elements, quality_metrics, current_frame, total_frames)
            Yields each frame's mesh data along with quality metrics and progress information

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]
            List of tuples containing the complete mesh data for all frames:
            - nodes: (n_nodes, 2) array of node coordinates
            - elements: (n_elements, 3) array of element connectivity
            - quality_metrics: Dictionary of mesh quality metrics
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
        """
        Calculate stress tensor stack from force field data.
        Always returns stress tensor with shape (t, y, x, 2, 2) where t=1 for single frames.
        Yields intermediate results during calculation.

        Parameters
        ----------
        force_field : np.ndarray
            Force field data with shape (t, y, x, 2)
        masks : np.ndarray
            Mask data with shape (t, y, x) or (y, x)
        mesh_data : Optional[List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]]
            Optional list of (nodes, elements, quality_metrics) tuples for each frame
            If not provided, meshes will be generated automatically

        Returns
        -------
        Generator[Tuple[MSMResult, int, int], None, MSMResult]
            Generator yielding (intermediate_result, frame_index, total_frames) during calculation
            and returning final MSMCalculationResult when exhausted
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
        """
        Validate MSM calculation parameters.

        Parameters
        ----------
        params : MSMParameters
            Parameters to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
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