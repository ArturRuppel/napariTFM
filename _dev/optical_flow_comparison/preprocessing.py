import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from napariTFM.backend.preprocessing import ImageProcessor

MIN_PERCENTILE = 80.0
MAX_PERCENTILE = 99.9
GAUSSIAN_SIGMA = 1.0

_processor = ImageProcessor()


def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply the project-standard TFM preprocessing pipeline.

    Steps: percentile intensity scaling [80, 99.9], then Gaussian filter
    (sigma=1). Matches `_validation/benchmark_TFM/validate_TFM.py`.

    Returns: float32 array in [0, 1], same shape as input.
    """
    scaled, _ = _processor.apply_intensity_scaling(
        image.astype(np.float32, copy=False), MIN_PERCENTILE, MAX_PERCENTILE
    )
    filtered = _processor.apply_gaussian_filter(scaled, sigma=GAUSSIAN_SIGMA)
    return filtered.astype(np.float32, copy=False)
