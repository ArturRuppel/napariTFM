from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from napariTFM.widgets._stage_dependencies import (
    computational_parameters,
    parameters_match,
)


class _Method(Enum):
    DIRECT = "direct"


@dataclass
class _Displacement:
    window: object
    d_max: float
    disp_vector_stride: int
    disp_arrow_scale: float


@dataclass
class _Force:
    young_modulus: object
    f_max: float
    force_vector_stride: int = 4
    force_arrow_scale: float = 1.0


@dataclass
class _Stress:
    method: object
    max_stress: float


def test_identical_computational_parameters_match():
    params = _Force(young_modulus=10.0, f_max=100.0)

    assert parameters_match("force", params, params)


def test_visualization_only_changes_do_not_make_results_stale():
    displacement = _Displacement(32, 5.0, 4, 1.0)
    force = _Force(10.0, 100.0)
    stress = _Stress(_Method.DIRECT, 50.0)

    assert parameters_match(
        "displacement",
        displacement,
        replace(
            displacement,
            d_max=10.0,
            disp_vector_stride=8,
            disp_arrow_scale=2.0,
        ),
    )
    assert parameters_match(
        "force",
        force,
        replace(
            force,
            f_max=200.0,
            force_vector_stride=8,
            force_arrow_scale=2.0,
        ),
    )
    assert parameters_match("stress", stress, replace(stress, max_stress=100.0))


def test_solver_parameter_change_makes_result_stale():
    stored = _Force(young_modulus=10.0, f_max=100.0)

    assert not parameters_match(
        "force", stored, replace(stored, young_modulus=20.0)
    )


def test_absent_parameter_metadata_is_stale():
    params = _Force(young_modulus=10.0, f_max=100.0)

    assert not parameters_match("force", None, params)
    assert not parameters_match("force", params, None)


def test_numpy_scalars_normalize_like_python_scalars_recursively():
    stored = {
        "solver": _Stress(_Method.DIRECT, max_stress=50.0),
        "values": [np.int64(4), (np.float32(1.5),)],
    }
    current = {
        "solver": _Stress("direct", max_stress=75.0),
        "values": [4, (1.5,)],
    }

    assert computational_parameters("stress", stored) == {
        "solver": {"method": "direct"},
        "values": [4, [1.5]],
    }
    assert parameters_match("stress", stored, current)
