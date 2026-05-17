import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from napariTFM.backend.displacement_analysis import DisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters

from _dev.optical_flow_comparison.adapters.base import sample_dense_at_points


class DISAdapter:
    """OpenCV DIS optical flow, sampled at query points."""

    name = "DIS"

    def __init__(self) -> None:
        self._analyzer = DisplacementAnalyzer(DisplacementParameters())

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        flow = self._analyzer.calculate_flow(reference, deformed)
        displacements = sample_dense_at_points(flow, query_points)
        valid = np.ones(len(query_points), dtype=bool)
        return displacements, valid
