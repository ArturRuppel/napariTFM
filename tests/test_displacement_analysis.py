import numpy as np

from napariTFM.backend.displacement_analysis import DisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters


def test_displacement_analyzer_initializes_with_standard_opencv():
    analyzer = DisplacementAnalyzer(DisplacementParameters())

    assert hasattr(analyzer.flow_algorithm, "calc")


def test_displacement_analyzer_returns_dense_xy_flow():
    reference = np.zeros((24, 24), dtype=np.float32)
    moving = np.zeros((24, 24), dtype=np.float32)
    reference[8:16, 8:16] = 1.0
    moving[8:16, 9:17] = 1.0

    analyzer = DisplacementAnalyzer(DisplacementParameters())
    flow = analyzer.calculate_flow(reference, moving)

    assert flow.shape == (24, 24, 2)
    assert flow.dtype == np.float32
    assert np.isfinite(flow).all()
