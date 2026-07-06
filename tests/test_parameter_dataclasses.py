"""Guards for UnifiedParameters -> per-stage projection.

UnifiedParameters.to_*_parameters() build each per-stage dataclass by copying
the fields whose names match (see UnifiedParameters._project). That is only
correct if every per-stage field name is also a UnifiedParameters field, and if
the values it carries match the unified source. These tests pin both.
"""
from dataclasses import fields

import pytest

from napariTFM.backend.parameter_dataclasses import (
    DisplacementParameters,
    FTTCParameters,
    StressParameters,
    UnifiedParameters,
)

_SUBCLASSES = [
    DisplacementParameters,
    FTTCParameters,
    StressParameters,
]


@pytest.mark.parametrize("cls", _SUBCLASSES)
def test_every_stage_field_exists_on_unified(cls):
    unified = {f.name for f in fields(UnifiedParameters)}
    missing = {f.name for f in fields(cls)} - unified
    assert not missing, f"{cls.__name__} has fields absent from UnifiedParameters: {missing}"


@pytest.mark.parametrize(
    "method, cls",
    [
        ("to_displacement_parameters", DisplacementParameters),
        ("to_fttc_parameters", FTTCParameters),
        ("to_stress_parameters", StressParameters),
    ],
)
def test_projection_copies_unified_values(method, cls):
    # Use non-default values so a stray hard-coded default would be caught.
    u = UnifiedParameters(
        pixel_size=0.23,
        frame_interval=3.5,
        downscale_factor=2,
        young_modulus=12345.0,
        regularization=5e-5,
        bism_regularization=2e-6,
        max_stress=4.0,
        d_max=7.0,
    )
    projected = getattr(u, method)()
    for f in fields(cls):
        assert getattr(projected, f.name) == getattr(u, f.name), f.name
