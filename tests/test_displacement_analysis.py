from pathlib import Path

import numpy as np

from napariTFM.backend.displacement_analysis import (
    DisplacementAnalyzer,
    calculate_displacement_field,
    validate_displacement_image,
)
from napariTFM.backend.parameter_dataclasses import DisplacementParameters


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_production_code_does_not_depend_on_displacement_service_layer():
    removed_module = ".".join(("services", "displacement_service"))
    removed_path = Path("napariTFM") / "services" / "displacement_service.py"

    assert not (REPO_ROOT / removed_path).exists()

    production_files = [
        path
        for root in ("napariTFM",)
        for path in (REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in production_files
        if removed_module in path.read_text()
    ]

    assert offenders == []
