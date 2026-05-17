from pathlib import Path

import numpy as np

from napariTFM.backend import preprocessing
from napariTFM.backend.parameter_dataclasses import PreprocessingParameters


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_validates_preprocessing_images():
    assert preprocessing.validate_preprocessing_image(None) == (False, "No image data provided")
    assert preprocessing.validate_preprocessing_image([[1, 2], [3, 4]]) == (
        False,
        "Image must be a numpy array",
    )
    assert preprocessing.validate_preprocessing_image(np.zeros((1, 2, 3, 4))) == (
        False,
        "Image must be 2D or 3D (time series)",
    )
    assert preprocessing.validate_preprocessing_image(np.full((2, 2), np.nan)) == (
        False,
        "Image contains only NaN values",
    )
    assert preprocessing.validate_preprocessing_image(np.zeros((2, 2), dtype=np.float32)) == (True, "")


def test_backend_preprocesses_frame_with_selected_parameter_set(monkeypatch):
    params = PreprocessingParameters(
        rolling_ball_radius=7,
        min_intensity_percentile=10,
        max_intensity_percentile=90,
        gaussian_sigma=2,
        cell_min_intensity_percentile=20,
        cell_max_intensity_percentile=80,
        cell_gaussian_sigma=3,
    )
    calls = []

    def fake_rolling_ball(image, radius):
        calls.append(("rolling_ball", radius))
        return image + 1

    def fake_gaussian(image, sigma):
        calls.append(("gaussian", sigma))
        return image + 2

    def fake_scaling(image, min_percentile, max_percentile):
        calls.append(("scaling", min_percentile, max_percentile))
        return image / 10, (min_percentile, max_percentile)

    monkeypatch.setattr(preprocessing.ImageProcessor, "apply_rolling_ball", staticmethod(fake_rolling_ball))
    monkeypatch.setattr(preprocessing.ImageProcessor, "apply_gaussian_filter", staticmethod(fake_gaussian))
    monkeypatch.setattr(preprocessing.ImageProcessor, "apply_intensity_scaling", staticmethod(fake_scaling))

    result = preprocessing.preprocess_frame(np.array([[1, 2], [3, 4]], dtype=np.float32), params)
    cell_result = preprocessing.preprocess_frame(
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        params,
        is_cell=True,
    )

    assert calls == [
        ("rolling_ball", 7),
        ("gaussian", 2),
        ("scaling", 10, 90),
        ("gaussian", 3),
        ("scaling", 20, 80),
    ]
    assert np.allclose(result.processed_image, np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float32))
    assert result.info["rolling_ball_radius"] == 7
    assert result.info["intensity_range"] == (10, 90)
    assert cell_result.info["rolling_ball_radius"] is None
    assert cell_result.info["gaussian_sigma"] == 3


def test_backend_preprocesses_stack_with_progress_and_reference(monkeypatch):
    params = PreprocessingParameters(registration_mode="translation")
    stack = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    reference = np.full((2, 2), 10, dtype=np.float32)
    calls = []

    def fake_preprocess_frame(image, received_params, is_cell=False, reference_image=None):
        calls.append((float(image.mean()), is_cell, reference_image is not None))
        transform = np.eye(2, 3, dtype=np.float32) if reference_image is not None else None
        return preprocessing.PreprocessingIntermediateResult(
            processed_image=image + 100,
            transform_matrix=transform,
            info={"final_mean": float(image.mean())},
        )

    monkeypatch.setattr(preprocessing, "preprocess_frame", fake_preprocess_frame)

    generator = preprocessing.preprocess_stack(stack, params, reference_image=reference)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        results = exc.value

    assert [(frame, total) for _, frame, total in progress] == [(0, 2), (1, 2)]
    assert calls == [
        (10.0, False, False),
        (1.5, False, True),
        (5.5, False, True),
    ]
    assert len(results) == 2
    assert np.array_equal(results[0].transform_matrix, np.eye(2, 3, dtype=np.float32))


def test_production_code_does_not_depend_on_preprocessing_service_layer():
    assert not (REPO_ROOT / "napariTFM/services/preprocessing_service.py").exists()

    production_files = [
        path
        for root in ("napariTFM",)
        for path in (REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in production_files
        if "services.preprocessing_service" in path.read_text()
    ]

    assert offenders == []
