"""Dependency and freshness helpers for interactive analysis stages."""

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum

import numpy as np


DISPLAY_ONLY_FIELDS = {
    "displacement": {"d_max", "disp_vector_stride", "disp_arrow_scale"},
    "force": {"f_max", "force_vector_stride", "force_arrow_scale"},
    "stress": {"max_stress"},
}


def computational_parameters(stage: str, params: object) -> object:
    """Return a recursively normalized, display-independent parameter value."""
    display_fields = DISPLAY_ONLY_FIELDS.get(stage, set())

    def normalize(value: object) -> object:
        if isinstance(value, Enum):
            return normalize(value.value)
        if isinstance(value, np.generic):
            return normalize(value.item())
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: normalize(getattr(value, field.name))
                for field in fields(value)
                if field.name not in display_fields
            }
        if isinstance(value, Mapping):
            return {
                normalize(key): normalize(item)
                for key, item in value.items()
                if key not in display_fields
            }
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [normalize(item) for item in value]
        return value

    return normalize(params)


def parameters_match(stage: str, stored: object, current: object) -> bool:
    """Return whether stored and current computational parameters are equal."""
    if stored is None or current is None:
        return False
    return computational_parameters(stage, stored) == computational_parameters(
        stage, current
    )
