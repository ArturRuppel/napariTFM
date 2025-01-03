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


class TriangleMeshGenerator:
    def __init__(self, refinement_factor: float = 1.0, quality_threshold: float = 0.3):
        """
        Initialize mesh generator with quality parameters

        Args:
            refinement_factor: Controls mesh density (higher = finer mesh)
            quality_threshold: Minimum acceptable element quality (0-1)
        """
        self.refinement_factor = refinement_factor
        self.quality_threshold = quality_threshold

    def generate_mesh(self, mask: npt.NDArray[np.bool_]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate triangular mesh from binary mask

        Args:
            mask: Binary mask defining the domain

        Returns:
            nodes: Array of node coordinates
            elements: Array of element connectivity
        """
        # Extract boundary points using contour detection
        contours = measure.find_contours(mask, 0.5)
        boundary_points = []
        for contour in contours:
            # Subsample contour based on refinement factor
            step = max(1, int(1 / self.refinement_factor))
            boundary_points.extend(contour[::step])

        boundary_points = np.array(boundary_points)

        # Add interior points using structured grid
        y, x = np.mgrid[0:mask.shape[0]:int(1 / self.refinement_factor),
               0:mask.shape[1]:int(1 / self.refinement_factor)]
        interior_points = np.column_stack((y.ravel(), x.ravel()))

        # Filter interior points to only those inside mask
        valid_points = mask[interior_points[:, 0].astype(int),
        interior_points[:, 1].astype(int)]
        interior_points = interior_points[valid_points]

        # Combine boundary and interior points
        all_points = np.vstack((boundary_points, interior_points))

        # Generate initial triangulation
        tri = Delaunay(all_points)

        # Filter elements to only those inside mask
        centroids = np.mean(all_points[tri.simplices], axis=1)
        valid_elements = mask[centroids[:, 0].astype(int),
        centroids[:, 1].astype(int)]

        elements = tri.simplices[valid_elements]
        nodes = all_points

        # Improve mesh quality
        nodes, elements = self._improve_mesh_quality(nodes, elements)

        return nodes, elements

    def _improve_mesh_quality(self,
                              nodes: np.ndarray,
                              elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Improve mesh quality through Laplacian smoothing and element filtering
        """
        # Laplacian smoothing of interior nodes
        boundary_nodes = self._find_boundary_nodes(nodes, elements)

        for _ in range(3):  # Number of smoothing iterations
            new_nodes = nodes.copy()

            # For each non-boundary node
            for i in range(len(nodes)):
                if i not in boundary_nodes:
                    # Find connected nodes
                    connected = elements[np.any(elements == i, axis=1)]
                    connected = connected.flatten()
                    connected = connected[connected != i]

                    # Average position of connected nodes
                    if len(connected) > 0:
                        new_pos = np.mean(nodes[connected], axis=0)
                        new_nodes[i] = new_pos

            nodes = new_nodes

        # Filter poor quality elements
        quality = self._calculate_element_quality(nodes, elements)
        good_elements = quality.aspect_ratios < 1 / self.quality_threshold
        elements = elements[good_elements]

        return nodes, elements

    def _find_boundary_nodes(self,
                             nodes: np.ndarray,
                             elements: np.ndarray) -> set:
        """
        Identify boundary nodes in the mesh
        """
        edges = np.vstack((
            elements[:, [0, 1]],
            elements[:, [1, 2]],
            elements[:, [2, 0]]
        ))

        # Sort edges and count occurrences
        edges = np.sort(edges, axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)

        # Boundary edges appear only once
        boundary_edges = edges[counts == 1]

        return set(boundary_edges.flatten())

    def _calculate_element_quality(self,
                                   nodes: np.ndarray,
                                   elements: np.ndarray) -> MeshQuality:
        """
        Calculate quality metrics for mesh elements
        """
        angles = []
        aspect_ratios = []
        edge_ratios = []

        for element in elements:
            vertices = nodes[element]

            # Calculate edge vectors and lengths
            edges = np.roll(vertices, -1, axis=0) - vertices
            lengths = np.sqrt(np.sum(edges ** 2, axis=1))

            # Calculate angles
            for i in range(3):
                e1 = edges[i]
                e2 = -edges[(i - 1) % 3]
                cos_angle = np.dot(e1, e2) / (lengths[i] * lengths[(i - 1) % 3])
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(np.degrees(angle))

            # Calculate aspect ratio (longest/shortest edge)
            aspect_ratios.append(np.max(lengths) / np.min(lengths))

            # Calculate edge ratio
            edge_ratios.append(np.max(lengths) / np.min(lengths))

        return MeshQuality(
            min_angle=np.min(angles),
            max_angle=np.max(angles),
            aspect_ratios=np.array(aspect_ratios),
            edge_ratios=np.array(edge_ratios)
        )

    def plot_mesh(self,
                  nodes: np.ndarray,
                  elements: np.ndarray,
                  mask: Optional[np.ndarray] = None):
        """
        Visualize the generated mesh
        """
        plt.figure(figsize=(10, 10))

        if mask is not None:
            plt.imshow(mask, alpha=0.3, cmap='gray')

        for element in elements:
            vertices = nodes[element]
            vertices = np.vstack((vertices, vertices[0]))  # Close the triangle
            plt.plot(vertices[:, 1], vertices[:, 0], 'b-', linewidth=0.5)

        plt.axis('equal')
        plt.title('Triangle Mesh')
        plt.show()