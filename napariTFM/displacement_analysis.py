import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar  # Add this import
from typing import Tuple, List, Optional
import tifffile
from dataclasses import dataclass

@dataclass
class TVL1Parameters:
    """Parameters for TV-L1 optical flow analysis."""
    tau: float = 0.25
    lambda_: float = 0.15
    theta: float = 0.3
    nscales: int = 5
    warps: int = 5
    epsilon: float = 0.01
    inner_iterations: int = 30
    outer_iterations: int = 10
    scale_step: float = 0.8
    gamma: float = 0.0
    median_filtering: int = 5
    use_initial_flow: bool = False

class DisplacementAnalyzer:
    """Analyzes displacements using TV-L1 optical flow."""

    def __init__(self, params: Optional[TVL1Parameters] = None):
        """
        Initialize TV-L1 optical flow analyzer.

        Args:
            params: TVL1Parameters instance with algorithm parameters
        """
        self.params = params or TVL1Parameters()
        self.flow_algorithm = cv2.optflow.DualTVL1OpticalFlow_create(
            self.params.tau, self.params.lambda_, self.params.theta,
            self.params.nscales, self.params.warps, self.params.epsilon,
            self.params.inner_iterations, self.params.outer_iterations,
            self.params.scale_step, self.params.gamma,
            self.params.median_filtering, self.params.use_initial_flow
        )

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Calculate optical flow between reference and moving image."""
        # Ensure images are float32 and normalized
        ref_float = reference.astype(np.float32) / 255.0 if reference.max() > 1 else reference.astype(np.float32)
        mov_float = moving.astype(np.float32) / 255.0 if moving.max() > 1 else moving.astype(np.float32)
        return self.flow_algorithm.calc(ref_float, mov_float, None)

    def apply_flow(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Apply flow field to an image using interpolation."""
        h, w = image.shape
        flow = flow.copy()
        flow[..., 0] += np.arange(w)
        flow[..., 1] += np.arange(h)[:, np.newaxis]
        return cv2.remap(image, flow, None, cv2.INTER_LINEAR)

class DisplacementVisualizer:
    """Handles visualization of displacement analysis results."""

    @staticmethod
    def create_overlay(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Create colored overlay of two images that sums to white in overlapping regions."""
        # Normalize images
        img1_norm = img1.astype(float) / img1.max()
        img2_norm = img2.astype(float) / img2.max()

        # Create RGB overlay
        overlay = np.zeros((*img1.shape, 3))

        # First image: magenta-ish (red + blue)
        overlay[..., 0] += img1_norm * 0.9  # red
        overlay[..., 2] += img1_norm * 0.5  # blue

        # Second image: cyan-ish (green + blue)
        overlay[..., 1] += img2_norm * 0.9  # green
        overlay[..., 2] += img2_norm * 0.5  # blue

        return np.clip(overlay, 0, 1)

    @staticmethod
    def plot_vector_field(ax, flow: np.ndarray, background: Optional[np.ndarray] = None,
                         step: int = 20, scale: float = 50, color='k',
                         with_magnitude_color: bool = False) -> Optional[Colorbar]:  # Fixed return type
        """Plot vector field with optional background and magnitude-based coloring."""
        if background is not None:
            ax.imshow(background, cmap='gray')

        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h:step, 0:w:step]
        fx = flow[..., 0][y, x]
        fy = -flow[..., 1][y, x]  # Flip y direction for visualization
        magnitude = np.sqrt(fx ** 2 + fy ** 2)
        mask = magnitude > 0.1

        if with_magnitude_color:
            quiver = ax.quiver(x[mask], y[mask], fx[mask], fy[mask],
                             magnitude[mask], cmap='viridis', scale=scale,
                             width=0.002, headwidth=4, headlength=5, minshaft=2)
            return plt.colorbar(quiver, ax=ax, label='Displacement magnitude')
        else:
            ax.quiver(x[mask], y[mask], fx[mask], fy[mask],
                     color=color, scale=scale, width=0.002,
                     headwidth=4, headlength=5, minshaft=2)
            return None

    def visualize_analysis(self, reference: np.ndarray, moving: np.ndarray,
                         flow: np.ndarray, cells: Optional[np.ndarray] = None,
                         output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive visualization of displacement analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # 1. Overlay of original images
        overlay = self.create_overlay(reference, moving)
        axes[0, 0].imshow(overlay)
        axes[0, 0].set_title('Original Images Overlay')

        # 2. Overlay with displaced image
        analyzer = DisplacementAnalyzer()
        displaced = analyzer.apply_flow(moving, flow)
        overlay_displaced = self.create_overlay(reference, displaced)
        axes[0, 1].imshow(overlay_displaced)
        axes[0, 1].set_title('Reference vs Displaced Image')

        # 3. Vector field over cells with magnitude colormap
        if cells is not None:
            axes[1, 0].imshow(cells, cmap='gray')
        else:
            axes[1, 0].imshow(reference, cmap='gray')
        colorbar = self.plot_vector_field(axes[1, 0], flow, with_magnitude_color=True)
        axes[1, 0].set_title('Displacement Field over Cells')

        # 4. Vector field over magnitude heatmap
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        im = axes[1, 1].imshow(magnitude, cmap='viridis')
        self.plot_vector_field(axes[1, 1], flow, scale=50)
        plt.colorbar(im, ax=axes[1, 1], label='Displacement magnitude')
        axes[1, 1].set_title('Displacement Field and Magnitude')

        # Format all subplots
        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)

        return fig

def analyze_image_pair(reference_path: str, moving_path: str,
                      cells_path: Optional[str] = None,
                      output_path: Optional[str] = None,
                      params: Optional[TVL1Parameters] = None):
    """Analyze a pair of images and visualize results."""
    # Read images
    reference = tifffile.imread(reference_path)
    moving = tifffile.imread(moving_path)
    cells = tifffile.imread(cells_path) if cells_path else None

    # Analyze
    analyzer = DisplacementAnalyzer(params)
    flow = analyzer.calculate_flow(reference, moving)

    # Visualize
    visualizer = DisplacementVisualizer()
    fig = visualizer.visualize_analysis(reference, moving, flow, cells, output_path)
    plt.show()

    return flow



if __name__ == "__main__":
    # Example usage with custom parameters
    params = TVL1Parameters(
        tau=0.25,
        lambda_=0.2,  # Adjusted for better flow detection
        nscales=6  # Increased scale levels
    )

    flow = analyze_image_pair(
        "C:/Users/aruppel/Desktop/test_simple/reference.tif",
        "C:/Users/aruppel/Desktop/test_simple/beads.tif",
        "C:/Users/aruppel/Desktop/test_simple/beads.tif",
        "C:/Users/aruppel/Desktop/test_simple/output_visualization.png"
    )




