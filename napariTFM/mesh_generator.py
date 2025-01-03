import numpy as np
from scipy.spatial import Delaunay
from skimage import measure
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy.typing as npt


@dataclass
class MeshQuality:
    min_angle: float
    max_angle: float
    aspect_ratios: npt.NDArray[np.float64]
    edge_ratios: npt.NDArray[np.float64]


class AdaptiveTriangleMeshGenerator:
    def __init__(self,
                 base_refinement: float = 1.0,
                 boundary_refinement: float = 2.0,
                 gradient_refinement: float = 1.5,
                 quality_threshold: float = 0.3):
        """
        Initialize adaptive mesh generator with refinement controls

        Args:
            base_refinement: Base mesh density (higher = finer mesh)
            boundary_refinement: Additional refinement factor at boundaries
            gradient_refinement: Additional refinement in high gradient regions
            quality_threshold: Minimum acceptable element quality (0-1)
        """
        self.base_refinement = base_refinement
        self.boundary_refinement = boundary_refinement
        self.gradient_refinement = gradient_refinement
        self.quality_threshold = quality_threshold

    def generate_mesh(self, mask: np.ndarray,
                      force_x: Optional[np.ndarray] = None,
                      force_y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adaptive triangular mesh based on domain shape and force gradients

        Args:
            mask: Binary mask defining the domain
            force_x: X component of force field (optional)
            force_y: Y component of force field (optional)

        Returns:
            nodes: Array of node coordinates
            elements: Array of element connectivity
        """
        # Calculate local refinement field
        refinement_field = self._calculate_refinement_field(mask, force_x, force_y)

        # Extract and refine boundary points
        boundary_points = self._extract_boundary_points(mask, refinement_field)

        # Generate interior points with adaptive density
        interior_points = self._generate_interior_points(mask, refinement_field)

        # Combine points
        all_points = np.vstack((boundary_points, interior_points))

        # Generate initial triangulation
        tri = Delaunay(all_points)

        # Filter and improve mesh
        elements = self._filter_and_improve_mesh(tri, all_points, mask)

        return all_points, elements

    def _calculate_refinement_field(self,
                                    mask: np.ndarray,
                                    force_x: Optional[np.ndarray],
                                    force_y: Optional[np.ndarray]) -> np.ndarray:
        """Calculate local refinement factor field"""
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        # Initialize refinement field with base value
        refinement = np.ones_like(mask, dtype=float) * self.base_refinement

        # Add boundary refinement
        distance = distance_transform_edt(mask)
        max_dist = np.max(distance)
        if max_dist > 0:
            boundary_factor = np.exp(-2 * distance / max_dist)
            refinement += boundary_factor * (self.boundary_refinement - 1)

        # Add gradient-based refinement if force fields are provided
        if force_x is not None and force_y is not None:
            # Calculate force magnitude gradients
            force_mag = np.sqrt(force_x ** 2 + force_y ** 2)
            force_mag[~mask] = 0

            # Compute gradient magnitude
            grad_y, grad_x = np.gradient(force_mag)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Normalize gradient magnitude
            if np.max(grad_mag) > 0:
                grad_mag = grad_mag / np.max(grad_mag)

                # Add gradient-based refinement
                refinement += grad_mag * (self.gradient_refinement - 1)

        # Smooth refinement field
        refinement = gaussian_filter(refinement, sigma=2.0)

        return refinement

    def _extract_boundary_points(self,
                                 mask: np.ndarray,
                                 refinement_field: np.ndarray) -> np.ndarray:
        """Extract boundary points with adaptive spacing"""
        # Find contours
        contours = measure.find_contours(mask, 0.5)
        boundary_points = []

        for contour in contours:
            # Calculate local refinement along contour
            contour_refinement = np.array([
                refinement_field[int(y), int(x)]
                for y, x in contour
            ])

            # Calculate adaptive spacing
            cumulative_length = np.cumsum(
                np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
            )
            total_length = cumulative_length[-1]

            # Target number of points based on average refinement
            avg_refinement = np.mean(contour_refinement)
            n_points = int(total_length * avg_refinement)

            if n_points > 0:
                # Generate points with adaptive spacing
                desired_spacing = total_length / n_points
                current_length = 0
                current_point = contour[0]
                new_points = [current_point]

                for i in range(1, len(contour)):
                    vec = contour[i] - current_point
                    seg_length = np.sqrt(np.sum(vec ** 2))

                    # Local refinement factor
                    local_refinement = contour_refinement[i]
                    local_spacing = desired_spacing / local_refinement

                    while current_length + seg_length > local_spacing:
                        # Add new point
                        frac = (local_spacing - current_length) / seg_length
                        new_point = current_point + frac * vec
                        new_points.append(new_point)

                        # Update for next point
                        current_point = new_point
                        vec = contour[i] - current_point
                        seg_length = np.sqrt(np.sum(vec ** 2))
                        current_length = 0

                    current_length += seg_length
                    current_point = contour[i]

                boundary_points.extend(new_points)

        return np.array(boundary_points)

    def _generate_interior_points(self,
                                  mask: np.ndarray,
                                  refinement_field: np.ndarray) -> np.ndarray:
        """Generate interior points with adaptive density"""
        # Calculate base grid size
        base_spacing = 1.0 / self.base_refinement

        # Generate denser grid and subsample based on refinement field
        y, x = np.mgrid[0:mask.shape[0]:base_spacing / 2,
               0:mask.shape[1]:base_spacing / 2]

        points = np.column_stack((y.ravel(), x.ravel()))
        valid_mask = mask[points[:, 0].astype(int),
        points[:, 1].astype(int)]
        points = points[valid_mask]

        # Get local refinement values
        local_refinement = refinement_field[points[:, 0].astype(int),
        points[:, 1].astype(int)]

        # Probabilistic point selection based on refinement
        probs = local_refinement / np.max(refinement_field)
        keep_mask = np.random.random(len(points)) < probs

        return points[keep_mask]

    def _filter_and_improve_mesh(self,
                                 tri: Delaunay,
                                 points: np.ndarray,
                                 mask: np.ndarray) -> np.ndarray:
        """Filter and improve triangulation"""
        # Filter elements to only those inside mask
        centroids = np.mean(points[tri.simplices], axis=1)
        valid_elements = mask[centroids[:, 0].astype(int),
        centroids[:, 1].astype(int)]
        elements = tri.simplices[valid_elements]

        # Calculate element quality metrics
        qualities = self._calculate_element_qualities(points, elements)

        # Filter out poor quality elements
        good_elements = qualities > self.quality_threshold
        elements = elements[good_elements]

        return elements

    def _calculate_element_qualities(self,
                                     points: np.ndarray,
                                     elements: np.ndarray) -> np.ndarray:
        """Calculate quality metrics for mesh elements"""
        qualities = np.zeros(len(elements))

        for i, element in enumerate(elements):
            # Get vertex coordinates
            vertices = points[element]

            # Calculate edge lengths
            edges = np.roll(vertices, -1, axis=0) - vertices
            lengths = np.sqrt(np.sum(edges ** 2, axis=1))

            # Calculate area using cross product
            area = np.abs(np.cross(edges[0], edges[1])) / 2

            # Calculate quality metric (ratio of area to sum of squared edge lengths)
            qualities[i] = 4 * np.sqrt(3) * area / np.sum(lengths ** 2)

        return qualities

    def plot_mesh(self,
                  nodes: np.ndarray,
                  elements: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  refinement_field: Optional[np.ndarray] = None,
                  force_x: Optional[np.ndarray] = None,
                  force_y: Optional[np.ndarray] = None,
                  figsize: Tuple[int, int] = (15, 5)):
        """
        Fast visualization of the generated mesh using vectorized operations
        """
        from matplotlib.collections import LineCollection, PolyCollection
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)

        if force_x is not None and force_y is not None and refinement_field is not None:
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)
            axes = [ax1, ax2, ax3]
        else:
            ax1 = plt.subplot(111)
            axes = [ax1]

        # Plot base mesh using vectorized operations
        if mask is not None:
            ax1.imshow(mask, alpha=0.3, cmap='gray', interpolation='nearest')

        # Create edge segments for all triangles at once
        edges = np.zeros((len(elements) * 3, 2, 2))
        for i, element in enumerate(elements):
            # Get node coordinates for triangle vertices
            triangle = nodes[element]
            # Create edge segments
            edges[i * 3] = triangle[[0, 1]][:, [1, 0]]  # x,y order for plotting
            edges[i * 3 + 1] = triangle[[1, 2]][:, [1, 0]]
            edges[i * 3 + 2] = triangle[[2, 0]][:, [1, 0]]

        # Create line collection for efficient edge plotting
        line_collection = LineCollection(edges, linewidths=0.5, alpha=0.6, color='b')
        ax1.add_collection(line_collection)

        # Plot nodes efficiently if refinement field is available
        if refinement_field is not None:
            node_refinements = refinement_field[nodes[:, 0].astype(int),
            nodes[:, 1].astype(int)]
            node_sizes = 20 * node_refinements / np.max(refinement_field)
            # Single scatter call instead of multiple plots
            ax1.scatter(nodes[:, 1], nodes[:, 0], s=node_sizes, c='r',
                        alpha=0.3, rasterized=True)
        else:
            ax1.scatter(nodes[:, 1], nodes[:, 0], s=10, c='r',
                        alpha=0.3, rasterized=True)

        ax1.set_title('Mesh Structure')
        ax1.set_aspect('equal')

        if len(axes) > 1:
            # Plot refinement field efficiently
            im2 = ax2.imshow(refinement_field, cmap='viridis',
                             interpolation='nearest')
            plt.colorbar(im2, ax=ax2, label='Refinement Factor')
            ax2.set_title('Refinement Field')

            # Plot force magnitude efficiently
            force_mag = np.sqrt(force_x ** 2 + force_y ** 2)
            im3 = ax3.imshow(force_mag, cmap='magma',
                             interpolation='nearest')
            plt.colorbar(im3, ax=ax3, label='Force Magnitude')
            ax3.set_title('Force Field')

            # Efficient quiver plot with reduced density
            step = max(1, int(force_x.shape[0] / 20))
            y, x = np.mgrid[0:force_x.shape[0]:step, 0:force_x.shape[1]:step]
            # Single quiver call with downsampled data
            ax3.quiver(x, y,
                       force_x[::step, ::step],
                       force_y[::step, ::step],
                       angles='xy', scale_units='xy', scale=2,
                       color='w', alpha=0.6, width=0.003)

        # Set limits and labels
        for ax in axes:
            ax.set_xlim(-1, mask.shape[1] if mask is not None else nodes[:, 1].max() + 1)
            ax.set_ylim(-1, mask.shape[0] if mask is not None else nodes[:, 0].max() + 1)
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')

        plt.tight_layout()
        return fig


