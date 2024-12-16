import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import tifffile


class DisplacementAnalyzer:
    """Analyzes displacements using TV-L1 optical flow."""

    def __init__(self, tau: float = 0.25, lambda_: float = 0.15, theta: float = 0.3,
                 nscales: int = 5, warps: int = 5, epsilon: float = 0.01,
                 inner_iterations: int = 30, outer_iterations: int = 10,
                 scale_step: float = 0.8, gamma: float = 0.0,
                 median_filtering: int = 5, use_initial_flow: bool = False):
        """
        Initialize TV-L1 optical flow analyzer.

        Args:
            tau: Time step for TV-L1 optical flow (default: 0.25)
            lambda_: Weight parameter for data term (default: 0.15)
            theta: Weight parameter for (u - v) (default: 0.3)
            nscales: Number of scales for pyramid (default: 5)
            warps: Number of warps per scale (default: 5)
            epsilon: Stopping criterion (default: 0.01)
            inner_iterations: Maximum number of inner iterations (default: 30)
            outer_iterations: Maximum number of outer iterations (default: 10)
            scale_step: Scale step for pyramid (default: 0.8)
            gamma: Gamma correction value (default: 0.0)
            median_filtering: Median filter kernel size (default: 5)
            use_initial_flow: Whether to use initial flow (default: False)
        """
        self.flow_algorithm = cv2.optflow.DualTVL1OpticalFlow_create(
            tau, lambda_, theta, nscales, warps, epsilon,
            inner_iterations, outer_iterations, scale_step,
            gamma, median_filtering, use_initial_flow
        )

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow between reference and moving image.

        Args:
            reference: Reference image
            moving: Moving image

        Returns:
            Flow field as numpy array
        """
        # Ensure images are float32 and normalized
        ref_float = reference.astype(np.float32) / 255.0 if reference.max() > 1 else reference.astype(np.float32)
        mov_float = moving.astype(np.float32) / 255.0 if moving.max() > 1 else moving.astype(np.float32)

        # Calculate flow
        flow = self.flow_algorithm.calc(ref_float, mov_float, None)

        return flow

    def analyze_stack(self, reference: np.ndarray, stack: np.ndarray,
                      progress_callback: Optional[callable] = None) -> List[np.ndarray]:
        """
        Analyze entire stack against reference image.

        Args:
            reference: Reference image
            stack: Stack of frames to analyze
            progress_callback: Optional callback for progress updates

        Returns:
            List of flow fields for each frame
        """
        results = []
        total_frames = len(stack)

        for i, frame in enumerate(stack):
            if progress_callback:
                progress_callback(i / total_frames * 100)

            flow = self.calculate_flow(reference, frame)
            results.append(flow)

        return results


def visualize_analysis(reference: np.ndarray, moving: np.ndarray,
                       flow: np.ndarray, output_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize displacement analysis results.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Process images for overlay
    # First invert the images
    ref_inv = 255 - reference if reference.max() > 1 else 1 - reference
    mov_inv = 255 - moving if moving.max() > 1 else 1 - moving

    # Normalize to 0-1 if needed
    ref_norm = ref_inv / 255 if ref_inv.max() > 1 else ref_inv
    mov_norm = mov_inv / 255 if mov_inv.max() > 1 else mov_inv

    # Create RGB overlays
    ref_img_RGB = np.zeros((*reference.shape, 3), dtype=np.float32)
    moving_img_RGB = np.zeros((*reference.shape, 3), dtype=np.float32)

    # Assign channels
    ref_img_RGB[:, :, 0] = ref_norm
    ref_img_RGB[:, :, 1] = ref_norm * 0.5
    moving_img_RGB[:, :, 2] = mov_norm
    moving_img_RGB[:, :, 1] = mov_norm * 0.5

    # Create overlay
    overlay = ref_img_RGB + moving_img_RGB
    overlay = np.clip(overlay * 4 - 3, 0, 1)  # Enhance contrast and clip to valid range

    # Show overlay
    ax.imshow(overlay)

    # Add flow field
    step = 20
    h, w = reference.shape
    y, x = np.mgrid[0:h:step, 0:w:step]

    fx = flow[..., 0][y, x]
    fy = -flow[..., 1][y, x]  # Flip y direction for visualization
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    mask = magnitude > 0.1

    quiver = ax.quiver(x[mask], y[mask], fx[mask], fy[mask],
                       magnitude[mask], cmap='jet', scale=50,
                       width=0.002, headwidth=4, headlength=5, minshaft=2)
    plt.colorbar(quiver, ax=ax, label='Flow magnitude')

    ax.set_title('Displacement Analysis')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    return fig, ax


def analyze_image_pair(reference_path: str, moving_path: str, output_path: Optional[str] = None):
    """
    Analyze a pair of images and visualize results.
    """
    # Read images
    reference = tifffile.imread(reference_path)
    moving = tifffile.imread(moving_path)

    # Analyze
    analyzer = DisplacementAnalyzer()
    flow = analyzer.calculate_flow(reference, moving)

    # Visualize
    fig, ax = visualize_analysis(reference, moving, flow, output_path)
    plt.show()

    return flow




if __name__ == "__main__":
    # Example usage
    # Single image pair analysis
    flow = analyze_image_pair(
        "C:/Users/aruppel/Desktop/test_simple/reference.tif",
        "C:/Users/aruppel/Desktop/test_simple/beads.tif",
        "C:/Users/aruppel/Desktop/test_simple/output_visualization.png"
    )

    # # Stack analysis
    # flows = analyze_image_stack(
    #     "path/to/reference.tif",
    #     "path/to/stack.tif",
    #     "output_directory"
    # )