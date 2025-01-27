from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create analysis and visualization mask stacks."""
        analysis_masks, vis_masks = MonolayerStressMicroscopy.create_mask_stack(
            image_stack,
            threshold_percentile=params.threshold,
            dilation=params.dilation,
            smoothing_sigma=params.smoothing_sigma,
            target_shape=target_shape,
            downscale_factor=downscale_factor
        )
        return analysis_masks, vis_masks

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
            masks: np.ndarray,
            force_field: np.ndarray,
            params: MSMParameters,
            progress_callback: Optional[callable] = None
    ) -> List[MSMCalculationResult]:
        """Calculate stress fields for multiple frames."""
        results = []
        num_frames = masks.shape[0]

        # Pre-generate meshes if using same parameters for all frames
        mesh_params = MeshParameters(
            mask=masks[0],  # Use first frame for initial sizing
            density_factor=params.density_factor,
            algorithm=self._get_algorithm_code(params.algorithm),
            use_optimization=params.use_optimization
        )
        mesh_generator = MeshGenerator(mesh_params)

        mesh_data = []
        for frame in range(num_frames):
            nodes, elements = mesh_generator.generate_mesh(masks[frame])
            mesh_data.append((nodes, elements))

            if progress_callback:
                progress_callback(frame, num_frames, "Generating meshes")

        # Calculate stress fields using pre-generated meshes
        for frame in range(num_frames):
            nodes, elements = mesh_data[frame]
            tx = force_field[frame, ..., 0]
            ty = force_field[frame, ..., 1]

            result = self.calculate_stress_field(
                mask=masks[frame],
                traction_x=tx,
                traction_y=ty,
                params=params,
                nodes=nodes,
                elements=elements
            )
            results.append(result)

            if progress_callback:
                progress_callback(frame, num_frames, "Calculating stress fields")

        return results

    def generate_mesh_preview(
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

    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics from last calculation."""
        # This would need to be implemented in the MonolayerStressMicroscopy class
        # and propagated through the service layer
        pass


