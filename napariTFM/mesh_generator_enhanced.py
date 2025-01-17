from dataclasses import dataclass
from typing import Tuple, Dict

import gmsh
import numpy as np
from skimage import measure


@dataclass
class MeshParameters:
    """Parameters for mesh generation"""
    mask: np.ndarray
    t_x: np.ndarray  # X component of tensor field
    t_y: np.ndarray  # Y component of tensor field
    density_factor: float = 0.01
    algorithm: int = 2
    use_optimization: bool = True
    tensor_refinement: bool = False  # Enable tensor-based refinement
    refinement_factor: float = 2.0  # How much to refine in high-gradient areas


class MeshGenerator:
    """Generate and analyze triangular meshes using GMSH"""

    def __init__(self, params: MeshParameters):
        self.input_params = params
        self.mesh_params = self._transform_parameters()

    def _create_gradient_field(self) -> int:
        """Create a mesh size field based on the gradient of tensor magnitude"""
        # Calculate tensor magnitude
        t_mag = np.sqrt(self.input_params.t_x ** 2 + self.input_params.t_y ** 2)

        # Calculate gradient of tensor magnitude
        grad_y, grad_x = np.gradient(t_mag)
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)


        # Normalize gradient to [0, 1] range
        if gradient_mag.max() > 0:
            gradient_mag = gradient_mag / gradient_mag.max()

        # Create attractor points at locations of high gradient
        threshold = 0.2  # Adjust this to control where refinement occurs
        high_gradient_points = []
        for i in range(gradient_mag.shape[0]):
            for j in range(gradient_mag.shape[1]):
                if gradient_mag[i, j] > threshold:
                    point_tag = gmsh.model.geo.addPoint(j, i, 0)
                    high_gradient_points.append(point_tag)

        # Create attractor field
        attractor = gmsh.model.mesh.field.add("Attractor")
        gmsh.model.mesh.field.setNumbers(attractor, "NodesList", high_gradient_points)

        # Create threshold field
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "IField", attractor)

        base_size = self.mesh_params['target_size']
        min_size = base_size / self.input_params.refinement_factor

        gmsh.model.mesh.field.setNumber(threshold, "LcMin", min_size)
        gmsh.model.mesh.field.setNumber(threshold, "LcMax", base_size)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", base_size * 2)

        return threshold

    def _setup_mesh_parameters(self):
        """Configure GMSH mesh generation parameters"""
        # Basic mesh parameters
        gmsh.option.setNumber("Mesh.Algorithm", self.mesh_params['algorithm'])

        # Disable automatic mesh size computation
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Set global mesh size constraints
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_params['min_size'])
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_params['max_size'])

        if self.input_params.tensor_refinement:
            # Create gradient-based mesh size field
            gradient_field = self._create_gradient_field()

            # Create constant field for base size
            constant = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(constant, "VIn", self.mesh_params['target_size'])

            # Create minimum field to combine both
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [gradient_field, constant])

            # Set as background field
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        else:
            # Use constant field for uniform mesh
            constant = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(constant, "VIn", self.mesh_params['target_size'])
            gmsh.model.mesh.field.setAsBackgroundMesh(constant)

    def _transform_parameters(self) -> dict:
        """Transform input parameters into GMSH-specific parameters"""
        max_dim = max(self.input_params.mask.shape)
        target_size = max_dim * self.input_params.density_factor

        return {
            'target_size': target_size,
            'min_size': target_size * 0.5,
            'max_size': target_size * 2.0,
            'algorithm': self.input_params.algorithm,
            'optimize_netgen': self.input_params.use_optimization,
            'optimize_steps': 10 if self.input_params.use_optimization else 0,
            'smoothing_steps': 5 if self.input_params.use_optimization else 0
        }


    def generate_mesh(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate triangular mesh from binary mask"""
        gmsh.initialize()
        gmsh.model.add("mask_mesh")

        try:
            self._setup_mesh_parameters()
            self._create_geometry(mask)

            # Generate initial mesh
            gmsh.model.mesh.generate(2)

            if self.mesh_params['optimize_netgen']:
                gmsh.model.mesh.optimize("Netgen")

            return self._extract_mesh()
        finally:
            gmsh.finalize()

    def _create_geometry(self, mask: np.ndarray):
        """Create geometry from mask contours"""
        contours = measure.find_contours(mask, 0.5)
        contours = [self._simplify_contour(contour) for contour in contours]
        contours = [c for c in contours if len(c) >= 3]

        surface_loops = []
        for contour in contours:
            points = []
            lines = []

            # Add points
            for x, y in contour:
                point_tag = gmsh.model.geo.addPoint(y, x, 0)
                points.append(point_tag)

            # Create lines between points
            for i in range(len(points)):
                line = gmsh.model.geo.addLine(points[i], points[(i + 1) % len(points)])
                lines.append(line)

            # Create curve loop
            curve_loop = gmsh.model.geo.addCurveLoop(lines)
            surface_loops.append(curve_loop)

        # Create surface
        if surface_loops:
            gmsh.model.geo.addPlaneSurface([surface_loops[0]] + surface_loops[1:])
            gmsh.model.geo.synchronize()
        else:
            raise ValueError("No valid contours found in mask")

    def _simplify_contour(self, contour: np.ndarray, tolerance: float = 2.0) -> np.ndarray:
        """Simplify contour by removing points that are too close together"""
        if len(contour) < 3:
            return contour

        simplified = [contour[0]]
        for point in contour[1:]:
            if np.linalg.norm(point - simplified[-1]) > tolerance:
                simplified.append(point)

        if np.linalg.norm(simplified[-1] - simplified[0]) <= tolerance:
            simplified.pop()

        return np.array(simplified)

    def _extract_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mesh points and triangles from GMSH"""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = node_coords.reshape(-1, 3)[:, :2]

        elem_types, elem_tags, elem_conn = gmsh.model.mesh.getElements()
        triangle_type_index = np.where(elem_types == 2)[0]

        if len(triangle_type_index) == 0:
            raise ValueError("No triangular elements found in the mesh")

        triangles = elem_conn[triangle_type_index[0]].reshape(-1, 3) - 1
        return points, triangles

    def analyze_mesh_quality(self, points: np.ndarray, triangles: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for the generated mesh"""
        # Get triangle vertices
        v0 = points[triangles[:, 0]]
        v1 = points[triangles[:, 1]]
        v2 = points[triangles[:, 2]]

        # Compute edge vectors
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # Compute edge lengths
        lengths = np.stack([
            np.linalg.norm(e0, axis=1),
            np.linalg.norm(e1, axis=1),
            np.linalg.norm(e2, axis=1)
        ]).T

        # Compute angles
        angles = []
        for i in range(3):
            e_prev = -e2 if i == 0 else -e0 if i == 1 else -e1
            e_next = e0 if i == 0 else e1 if i == 1 else e2
            cos_angle = np.sum(e_prev * e_next, axis=1) / (
                    np.linalg.norm(e_prev, axis=1) * np.linalg.norm(e_next, axis=1)
            )
            angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        angles = np.concatenate(angles) * 180 / np.pi

        # Compute areas
        areas = 0.5 * np.abs(
            (v0[:, 0] * v1[:, 1] + v1[:, 0] * v2[:, 1] + v2[:, 0] * v0[:, 1]) -
            (v1[:, 0] * v0[:, 1] + v2[:, 0] * v1[:, 1] + v0[:, 0] * v2[:, 1])
        )

        # Quality metrics
        aspect_ratios = np.max(lengths, axis=1) / np.min(lengths, axis=1)
        s = np.sum(lengths, axis=1) / 2
        r_in = 2 * areas / (s * 2)
        r_out = np.prod(lengths, axis=1) / (4 * areas)
        quality = 2 * r_in / r_out

        return {
            "min_angle": np.min(angles),
            "mean_angle": np.mean(angles),
            "min_quality": np.min(quality),
            "mean_quality": np.mean(quality),
            "max_aspect_ratio": np.max(aspect_ratios),
            "mean_aspect_ratio": np.mean(aspect_ratios),
            "n_elements": len(triangles)
        }


if __name__ == "__main__":
    import os
    from pathlib import Path

    # Get the current file's directory and construct paths
    current_dir = Path(__file__).parent
    benchmark_dir = current_dir / "benchmarks" / "synthetic_cell_physical_units"

    # Load traction data
    t_x = np.load(benchmark_dir / "Traction_x_warped.npy")
    t_y = np.load(benchmark_dir / "Traction_y_warped.npy")

    from scipy.interpolate import RegularGridInterpolator

    # Define coordinate grids
    old_coords = np.arange(200)
    new_coords = np.linspace(0, 199, 1000)
    X_new, Y_new = np.meshgrid(new_coords, new_coords, indexing='ij')
    # Create and apply interpolators for both components
    t_x = RegularGridInterpolator((old_coords, old_coords), t_x, bounds_error=False)(np.stack([X_new, Y_new], axis=-1))
    t_y = RegularGridInterpolator((old_coords, old_coords), t_y, bounds_error=False)(np.stack([X_new, Y_new], axis=-1))


    mask = np.abs(t_x) > 0  # or load actual mask if available

    # Create mesh parameters
    params = MeshParameters(
        mask=mask,
        t_x=t_x,
        t_y=t_y,
        density_factor=0.01,  # Adjust based on your needs
        tensor_refinement=True,
        refinement_factor=1,
    )

    # Generate mesh
    generator = MeshGenerator(params)
    points, triangles = generator.generate_mesh(mask)

    # Analyze mesh quality
    quality_metrics = generator.analyze_mesh_quality(points, triangles)
    print("\nMesh Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value:.3f}")

    # Optionally visualize the mesh
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # Plot traction magnitude
        plt.subplot(121)
        t_mag = np.sqrt(t_x ** 2 + t_y ** 2)
        plt.imshow(t_mag, cmap='viridis')
        plt.colorbar(label='Traction Magnitude')
        plt.title('Traction Field')

        # Plot mesh
        plt.subplot(122)
        plt.triplot(points[:, 0], -points[:, 1], triangles, 'k-', lw=0.5)
        plt.title('Generated Mesh')
        plt.axis('equal')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")