from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator
import numpy as np
from napari.layers import Layer
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from skimage.transform import resize

from napariTFM.backend.msm import MonolayerStressMicroscopy
from napariTFM.backend.mesh_generator import MeshParameters, MeshGenerator


@dataclass
class MSMParameters:
    """Parameters for MSM calculations"""
    # Mesh parameters
    density_factor: float
    algorithm: str
    use_optimization: bool

    # Material parameters
    poisson_ratio_cells: float
    young_modulus: float

    # Mask creation parameters
    threshold: float
    dilation: int
    smoothing_sigma: float

    # Visualization parameters
    max_stress: float


@dataclass
class MSMCalculationResult:
    """Results from stress field calculation"""
    stress_tensor: np.ndarray
    condition_number: float
    residual: float
    parameters: MSMParameters


@dataclass
class MeshPreviewResult:
    """Results from mesh preview generation"""
    nodes: np.ndarray
    elements: np.ndarray
    quality_metrics: Dict[str, float]


class MSMService:
    """Service layer handling business logic for Monolayer Stress Microscopy calculations."""

    def create_mask_from_image(
            self,
            image: np.ndarray,
            params: MSMParameters
    ) -> np.ndarray:
        """Create single mask from image using specified parameters."""
        return MonolayerStressMicroscopy.create_mask_from_image(
            image,
            threshold_percentile=params.threshold,
            dilation=params.dilation,
            smoothing_sigma=params.smoothing_sigma
        )

    def create_mask_stack(
            self,
            image_stack: np.ndarray,
            params: MSMParameters,
            target_shape: Optional[Tuple[int, int]] = None,
            downscale_factor: int = 1
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
        downscale_factor : int, optional
            Factor to downscale masks by for visualization

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

    def process_mask_data(self, mask_data: np.ndarray, force_field: np.ndarray = None) -> tuple[np.ndarray, list[str]]:
        """
        Process mask data into the required format with validation and resizing.

        Args:
            mask_data: Input mask data array
            force_field: Optional force field for shape validation

        Returns:
            Tuple of (processed_mask_data, warning_messages)
        """
        warnings = []

        if mask_data is None:
            raise ValueError("No mask data provided")

        # Check for multiple values
        unique_values = np.unique(mask_data)
        unique_values = unique_values[unique_values != 0]  # Exclude zero
        if len(unique_values) > 1:
            warnings.append("Multiple non-zero values detected in the mask. Converting to binary (0 and 1).")

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

    def resize_mask_to_forces(
            self,
            mask: np.ndarray,
            force_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Resize mask to match force field shape."""
        if mask.shape != force_shape:
            return resize(
                mask.astype(float),
                force_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
        return mask


    def generate_mesh(
            self,
            mask: np.ndarray,
            params: MSMParameters
    ) -> MeshPreviewResult:
        """Generate mesh for preview/visualization."""
        mesh_params = MeshParameters(
            mask=mask,
            density_factor=params.density_factor,
            algorithm=self._get_algorithm_code(params.algorithm),
            use_optimization=params.use_optimization
        )

        mesh_generator = MeshGenerator(mesh_params)
        nodes, elements = mesh_generator.generate_mesh(mask)
        quality_metrics = mesh_generator.analyze_mesh_quality(nodes, elements)

        return MeshPreviewResult(
            nodes=nodes,
            elements=elements,
            quality_metrics=quality_metrics
        )

    def generate_mesh_stack(
            self,
            mask_stack: np.ndarray,
            params: MSMParameters
    ) -> Generator[Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int], None, List[MeshPreviewResult]]:
        """
        Generate meshes for all frames in the mask stack as a generator that yields intermediate results.

        Parameters
        ----------
        mask_stack : np.ndarray
            3D array of masks (frames, height, width) or 2D single mask
        params : MSMParameters
            Parameters containing mesh generation settings

        Yields
        ------
        Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int]
            (nodes, elements, quality_metrics, current_frame, total_frames)
            Yields each frame's mesh data along with quality metrics and progress information

        Returns
        -------
        List[MeshPreviewResult]
            List of MeshPreviewResult objects containing the complete mesh data for all frames
        """
        # Handle 2D input
        if mask_stack.ndim == 2:
            mask_stack = mask_stack[np.newaxis, ...]

        total_frames = mask_stack.shape[0]

        # Initialize mesh generator with parameters
        mesh_params = MeshParameters(
            mask=mask_stack[0],  # Use first frame for initial setup
            density_factor=params.density_factor,
            algorithm=self._get_algorithm_code(params.algorithm),
            use_optimization=params.use_optimization
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

            # Create and store result object
            result = MeshPreviewResult(
                nodes=nodes,
                elements=elements,
                quality_metrics=quality_metrics
            )
            mesh_results.append(result)

            # Yield intermediate results
            yield nodes, elements, quality_metrics, frame, total_frames

        return mesh_results

    def calculate_stress_field(
            self,
            mask: np.ndarray,
            traction_x: np.ndarray,
            traction_y: np.ndarray,
            params: MSMParameters,
            nodes: Optional[np.ndarray] = None,
            elements: Optional[np.ndarray] = None
    ) -> MSMCalculationResult:
        """Calculate stress field for single frame."""
        # Initialize MSM analyzer
        analyzer = MonolayerStressMicroscopy(
            mask=mask,
            density_factor=params.density_factor,
            algorithm=self._get_algorithm_code(params.algorithm),
            use_optimization=params.use_optimization,
            poisson_ratio=params.poisson_ratio_cells,
            young_modulus=params.young_modulus,
            nodes=nodes,
            elements=elements
        )

        # Calculate stress field
        stress_tensor, condition_number, residual = analyzer.calculate_stress_field(
            traction_x,
            traction_y
        )

        return MSMCalculationResult(
            stress_tensor=stress_tensor,
            condition_number=condition_number,
            residual=residual,
            parameters=params
        )

    def calculate_stress_stack(
            self,
            force_field: np.ndarray,
            params: MSMParameters,
            mesh_results: List[MeshPreviewResult]
    ) -> Generator[Tuple[MSMCalculationResult, int, int], None, List[MSMCalculationResult]]:
        """
        Calculate stress fields for multiple frames, yielding intermediate results.

        Parameters
        ----------
        force_field : np.ndarray
            4D array of force vectors (frames, height, width, 2)
        params : MSMParameters
            Parameters for stress calculation
        mesh_results : List[MeshPreviewResult]
            Pre-generated mesh data for each frame

        Yields
        ------
        Tuple[MSMCalculationResult, int, int]
            (stress_result, current_frame, total_frames)
            Yields each frame's stress calculation result along with progress information

        Returns
        -------
        List[MSMCalculationResult]
            Complete list of stress calculation results for all frames
        """
        results = []
        num_frames = force_field.shape[0]

        # Validate input dimensions
        if len(mesh_results) != num_frames:
            raise ValueError(
                f"Number of mesh results ({len(mesh_results)}) "
                f"does not match number of frames ({num_frames})"
            )

        # Calculate stress fields using pre-generated meshes
        for frame in range(num_frames):
            # Extract mesh data for current frame
            mesh_data = mesh_results[frame]
            nodes = mesh_data.nodes
            elements = mesh_data.elements

            # Extract force components
            tx = force_field[frame, ..., 0]
            ty = force_field[frame, ..., 1]

            # Calculate stress field for current frame
            result = self.calculate_stress_field(
                mask=None,  # Mask not needed since we're providing nodes/elements
                traction_x=tx,
                traction_y=ty,
                params=params,
                nodes=nodes,
                elements=elements
            )
            results.append(result)

            # Yield intermediate results
            yield result, frame, num_frames

        return results


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
        # Convert to lowercase and remove special characters for matching
        normalized_name = algorithm_name.lower().replace(".", "").replace("-", " ")
        return algorithm_map.get(normalized_name, 6)  # Default to Frontal-Delaunay

    def validate_parameters(self, params: MSMParameters) -> Tuple[bool, str]:
        """Validate MSM parameters."""
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

        return True, ""


    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics from last calculation."""
        # This would need to be implemented in the MonolayerStressMicroscopy class
        # and propagated through the service layer
        pass


