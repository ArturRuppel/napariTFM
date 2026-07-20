from pathlib import Path

import numpy as np

from napariTFM.backend import fttc
from napariTFM.backend.parameter_dataclasses import FTTCParameters


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_validates_fttc_displacement_fields():
    assert fttc.validate_displacement_field(None) == (False, "No displacement field data provided")
    assert fttc.validate_displacement_field([[1, 2], [3, 4]]) == (
        False,
        "Displacement field must be a numpy array",
    )
    assert fttc.validate_displacement_field(np.zeros((2, 2))) == (
        False,
        "Displacement field must be 3D (y,x,2) or 4D (t,y,x,2)",
    )
    assert fttc.validate_displacement_field(np.zeros((2, 2, 3))) == (
        False,
        "Last dimension must be 2 (x,y components), got 3",
    )
    assert fttc.validate_displacement_field(np.full((2, 2, 2), np.nan)) == (
        False,
        "Displacement field contains only NaN values",
    )
    assert fttc.validate_displacement_field(np.zeros((2, 2, 2), dtype=np.float32)) == (True, "")


def test_backend_calculates_fttc_result_with_progress(monkeypatch):
    params = FTTCParameters(pixel_size=0.2, downscale_factor=3, frame_interval=2, regularization=1e-5)
    displacement_field = np.zeros((2, 4, 5, 2), dtype=np.float32)
    displacement_field[0, ..., 0] = 1
    displacement_field[1, ..., 1] = 2

    class FakeFTTC:
        def __init__(self, received_params):
            assert received_params == params

        def calculate_traction(self, displacements, pixel_size, downscale_factor, regularization):
            assert displacements.shape == (4, 5, 2)
            assert pixel_size == params.pixel_size
            assert downscale_factor == params.downscale_factor
            assert regularization == params.regularization
            frame_value = float(displacements[..., 0].max() + displacements[..., 1].max())
            return None, np.stack(
                [
                    np.full((4, 5), frame_value, dtype=np.float32),
                    np.full((4, 5), frame_value + 10, dtype=np.float32),
                ]
            )

    monkeypatch.setattr(fttc, "FTTC", FakeFTTC)

    generator = fttc.calculate_force_field(displacement_field, params)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        result = exc.value

    assert [(frame, total) for _, frame, total in progress] == [(1, 2), (2, 2)]
    assert result.force_field.shape == (2, 4, 5, 2)
    assert result.force_field.dtype == np.float32
    assert result.force_field[0, ..., 0].max() == 1
    assert result.force_field[1, ..., 1].max() == 12
    assert result.original_shape == (4, 5)
    assert result.force_shape == (4, 5)
    assert result.parameters == params
    assert result.physical_scale == {
        "pixel_size": 0.2,
        "grid_spacing": 0.6000000000000001,
        "time_interval": 2,
        "force_units": "Pa",
        "grid_spacing_units": "µm",
        "time_interval_units": "min",
    }


def test_production_code_does_not_depend_on_fttc_service_layer():
    removed_module = ".".join(("services", "fttc_service"))
    removed_path = Path("napariTFM") / "services" / "fttc_service.py"

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
