from pathlib import Path

import numpy as np

from napariTFM.backend import msm
from napariTFM.backend.parameter_dataclasses import MSMParameters


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_creates_mask_stack_with_progress(monkeypatch):
    params = MSMParameters(threshold=25, dilation=2, smoothing_sigma=1)
    images = np.array(
        [
            [[0, 1], [2, 0]],
            [[3, 0], [0, 4]],
        ],
        dtype=np.float32,
    )

    def fake_create_mask_from_image(image, threshold_percentile, dilation, smoothing_sigma):
        assert threshold_percentile == params.threshold
        assert dilation == params.dilation
        assert smoothing_sigma == params.smoothing_sigma
        return image > 0

    monkeypatch.setattr(
        msm.MonolayerStressMicroscopy,
        "create_mask_from_image",
        staticmethod(fake_create_mask_from_image),
    )

    generator = msm.create_mask_stack(images, params)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        masks = exc.value

    assert [(frame, total) for _, frame, total in progress] == [(0, 2), (1, 2)]
    assert masks.shape == (2, 2, 2)
    assert masks.dtype == bool
    assert masks[0, 0, 1]
    assert masks[1, 1, 1]


def test_backend_processes_mask_data_and_resizes_to_force_field():
    mask_data = np.array([[0, 2], [3, 0]], dtype=np.uint8)
    force_field = np.zeros((1, 4, 4, 2), dtype=np.float32)

    processed, warnings = msm.process_mask_data(mask_data, force_field)

    assert processed.shape == (1, 4, 4)
    assert processed.dtype == bool
    assert warnings == ["Multiple non-zero values detected in mask. Converting to binary (0 and 1)."]


def test_backend_generates_mesh_stack_with_progress(monkeypatch):
    params = MSMParameters(density_factor=0.02, mesh_algorithm="delaunay", use_optimization=False)
    masks = np.ones((2, 3, 3), dtype=bool)

    class FakeMeshGenerator:
        def __init__(self, mesh_params):
            self.mesh_params = mesh_params
            assert mesh_params.density_factor == params.density_factor
            assert mesh_params.mesh_algorithm == 5
            assert mesh_params.use_optimization is False

        def generate_mesh(self, mask):
            assert mask.shape == (3, 3)
            return np.array([[0, 0], [1, 0], [0, 1]], dtype=float), np.array([[0, 1, 2]])

        def analyze_mesh_quality(self, nodes, elements):
            return {"mean_quality": 0.8, "min_angle": 30.0}

    monkeypatch.setattr(msm, "MeshGenerator", FakeMeshGenerator)

    generator = msm.generate_mesh_stack(masks, params)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        mesh_data = exc.value

    assert [(frame, total) for *_, frame, total in progress] == [(0, 2), (1, 2)]
    assert len(mesh_data) == 2
    assert mesh_data[0][2] == {"mean_quality": 0.8, "min_angle": 30.0}


def test_backend_calculates_msm_result_with_progress(monkeypatch):
    params = MSMParameters(pixel_size=0.2, downscale_factor=5, frame_interval=3)
    force_field = np.zeros((2, 3, 4, 2), dtype=np.float32)
    force_field[0, ..., 0] = 1
    force_field[1, ..., 1] = 2
    masks = np.ones((2, 3, 4), dtype=bool)
    nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    elements = np.array([[0, 1, 2]])
    mesh_data = [(nodes, elements, {"mean_quality": 1.0}), (nodes + 1, elements, {"mean_quality": 0.9})]

    class FakeAnalyzer:
        def __init__(self, received_params, mask=None, nodes=None, elements=None):
            assert received_params == params
            self.mask = mask
            self.nodes = nodes
            self.elements = elements

        def calculate_stress_field(self, tx, ty):
            value = float(tx.max() + ty.max())
            stress = np.full((*tx.shape, 2, 2), value, dtype=np.float32)
            return stress, value + 10, value + 20

    monkeypatch.setattr(msm, "MonolayerStressMicroscopy", FakeAnalyzer)

    generator = msm.calculate_stresses(force_field, masks, params, mesh_data=mesh_data)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        result = exc.value

    assert [(frame, total) for _, frame, total in progress] == [(1, 2), (2, 2)]
    assert result.stress_tensor.shape == (2, 3, 4, 2, 2)
    assert result.stress_tensor.dtype == np.float32
    assert result.stress_tensor[0].max() == np.float32(0.001)
    assert result.stress_tensor[1].max() == np.float32(0.002)
    assert len(result.nodes) == 2
    assert len(result.elements) == 2
    assert result.condition_number == np.mean([11, 12])
    assert result.residual == np.mean([21, 22])
    assert result.parameters == params
    assert result.physical_scale == {
        "pixel_size": 0.2,
        "grid_spacing": 1.0,
        "time_interval": 3,
        "stress_units": "mN/m",
        "grid_spacing_units": "µm",
        "time_interval_units": "min",
    }
    assert result.original_shape == (3, 4)
    assert result.stress_shape == (3, 4)


def test_production_code_does_not_depend_on_msm_service_layer():
    removed_module = ".".join(("services", "msm_service"))
    removed_path = Path("napariTFM") / "services" / "msm_service.py"

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
