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
    lambda_: float = 0.4
    theta: float = 0.3
    nscales: int = 3
    warps: int = 3
    epsilon: float = 0.01
    inner_iterations: int = 15
    outer_iterations: int = 5
    scale_step: float = 0.5
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






