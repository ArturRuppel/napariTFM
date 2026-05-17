import numpy as np

from napariTFM.backend.displacement_analysis import (
    DisplacementAnalyzer,
    DisplacementResult as BackendDisplacementResult,
    calculate_displacement_field,
    validate_displacement_image,
)
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.services.displacement_service import (
    DisplacementResult as ServiceDisplacementResult,
    DisplacementService,
)


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


def test_backend_validates_displacement_images():
    assert validate_displacement_image(None) == (False, "No image data provided")
    assert validate_displacement_image([[1, 2], [3, 4]]) == (False, "Image must be a numpy array")
    assert validate_displacement_image(np.zeros((2, 2, 2, 2))) == (
        False,
        "Image must be 2D or 3D (time series)",
    )
    assert validate_displacement_image(np.full((2, 2), np.nan)) == (
        False,
        "Image contains only NaN values",
    )
    assert validate_displacement_image(np.zeros((2, 2), dtype=np.float32)) == (True, "")


def test_backend_calculates_displacement_result_with_progress():
    reference = np.zeros((24, 24), dtype=np.float32)
    moving = np.zeros((2, 24, 24), dtype=np.float32)
    reference[8:16, 8:16] = 1.0
    moving[:, 8:16, 9:17] = 1.0
    params = DisplacementParameters(pixel_size=0.2, downscale_factor=2)

    generator = calculate_displacement_field(reference, moving, params)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        result = exc.value

    assert [(frame, total) for _, frame, total in progress] == [(1, 2), (2, 2)]
    assert result.displacement_field.shape == (2, 12, 12, 2)
    assert result.displacement_field.dtype == np.float32
    assert result.original_shape == (24, 24)
    assert result.displacement_field_shape == (12, 12)
    assert result.parameters == params
    assert result.physical_scale == {
        "pixel_size": 0.2,
        "grid_spacing": 0.4,
        "time_interval": 1,
        "displacement_units": "µm",
        "grid_spacing_units": "µm",
        "time_interval_units": "min",
    }
    assert np.isfinite(result.displacement_field).all()


def test_displacement_service_keeps_result_import_compatibility():
    assert ServiceDisplacementResult is BackendDisplacementResult


def test_displacement_service_delegates_field_calculation_to_backend_shape_contract():
    reference = np.zeros((16, 16), dtype=np.float32)
    moving = np.zeros((16, 16), dtype=np.float32)
    reference[5:11, 5:11] = 1.0
    moving[5:11, 6:12] = 1.0
    params = DisplacementParameters(pixel_size=0.3, downscale_factor=1)
    service = DisplacementService(params)

    generator = service.calculate_displacement_field(reference, moving)
    next(generator)
    try:
        next(generator)
    except StopIteration as exc:
        result = exc.value

    assert isinstance(result, BackendDisplacementResult)
    assert result.displacement_field.shape == (1, 16, 16, 2)
    assert result.parameters == params
    assert result.physical_scale["pixel_size"] == 0.3
