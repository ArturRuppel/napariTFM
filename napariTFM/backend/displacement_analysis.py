from dataclasses import dataclass
from typing import Optional, Dict, Generator

import cv2
import numpy as np


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
    downscale_factor: int = 1

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

    def analyze_displacement_generator(self, reference: np.ndarray, bead_stack: np.ndarray,
                                       pixel_size: float, downscale_factor: int = 1,
                                       visualization_params: Optional[Dict] = None) -> Generator:
        """Generator version for external threading"""
        total_frames = len(bead_stack)
        flows = []

        for i in range(total_frames):
            yield {  # Progress updates
                'progress': (i + 1) / total_frames * 100,
                'message': f"Processing frame {i + 1}/{total_frames}...\nComputing optical flow..."
            }

            flow_pixels = self.calculate_flow(reference, bead_stack[i])

            if downscale_factor > 1:
                flow_pixels = self.downscale_flow(flow_pixels, downscale_factor)

            flows.append(flow_pixels * pixel_size)

        # Package final results
        return {
            'flows': flows,
            'parameters': {
                'tvl1_params': self.params.__dict__,
                'downscale_factor': downscale_factor,
                'pixel_size': pixel_size
            },
            'visualization_params': visualization_params or {
                'd_max': 10.0,
                'vector_stride': 20,
                'arrow_scale': 1.0
            },
            'original_shape': reference.shape,
            'flow_shape': flows[0].shape[:2],
            'units': 'micrometers'
        }

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Calculate optical flow between reference and moving image at full resolution."""
        # Ensure images are float32 and normalized
        ref_float = (reference.astype(np.float32) - reference.min()) / (reference.max() - reference.min())
        mov_float = (moving.astype(np.float32) - moving.min()) / (moving.max() - moving.min())

        return self.flow_algorithm.calc(ref_float, mov_float, None)

    def downscale_flow(self, flow: np.ndarray, factor: int) -> np.ndarray:
        """Downscale flow field using local averaging."""
        if factor <= 1:
            return flow

        h, w = flow.shape[:2]
        new_h, new_w = h // factor, w // factor

        # Handle each component separately to preserve vector information
        downscaled = np.zeros((new_h, new_w, 2))

        for i in range(new_h):
            for j in range(new_w):
                # Extract block
                y_start = i * factor
                y_end = min((i + 1) * factor, h)
                x_start = j * factor
                x_end = min((j + 1) * factor, w)

                block = flow[y_start:y_end, x_start:x_end]
                # Average the x and y components separately
                downscaled[i, j] = np.mean(block, axis=(0, 1))

        return downscaled

    def apply_flow(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Apply flow field to an image using interpolation."""
        h, w = image.shape
        flow = flow.copy()
        flow[..., 0] += np.arange(w)
        flow[..., 1] += np.arange(h)[:, np.newaxis]
        return cv2.remap(image, flow, None, cv2.INTER_LINEAR)






