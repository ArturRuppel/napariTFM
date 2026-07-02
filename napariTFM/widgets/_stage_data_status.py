from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

@dataclass(frozen=True)
class DataArtifactSpec:
    key: str
    label: str
    attr: str | None
    role: str = "input"
    required: bool = True
    on_view: Callable[[], None] | None = None
    on_action: Callable[[], None] | None = None


def artifact_value(data_manager: Any, spec: DataArtifactSpec):
    if spec.attr is None:
        return None
    return getattr(data_manager, spec.attr, None)


def artifact_available(data_manager: Any, spec: DataArtifactSpec) -> bool:
    checker = getattr(data_manager, "artifact_available", None)
    if checker is not None:
        try:
            return bool(checker(spec.key))
        except (KeyError, AttributeError):
            pass
    return artifact_value(data_manager, spec) is not None


def artifact_state(data_manager: Any, spec: DataArtifactSpec):
    get_artifact = getattr(data_manager, "get_artifact", None)
    if get_artifact is None:
        return None
    try:
        return get_artifact(spec.key)
    except (KeyError, AttributeError):
        return None


def _shape_text(value: Any) -> str:
    try:
        shape = getattr(value, "shape", None)
        if shape is not None:
            return "×".join(str(size) for size in shape)
    except Exception:
        pass

    for attr in ("displacement_field", "force_field", "stress_tensor"):
        array = getattr(value, attr, None)
        if array is not None and hasattr(array, "shape"):
            return "×".join(str(size) for size in array.shape)
    return ""


def artifact_info_text(data_manager: Any, spec: DataArtifactSpec) -> str:
    """A human description of an artifact's state, honest about cache vs disk.

    Distinguishes a cache-only (unsaved) value — which run-all will clobber with
    whatever is on disk — from a value already backed by a file on disk.
    """
    state = artifact_state(data_manager, spec)
    available = artifact_available(data_manager, spec)
    value = state.value if state is not None else artifact_value(data_manager, spec)
    if value is not None:
        base = _shape_text(value) or "Loaded"
    elif available:
        base = "Saved"
    else:
        return "Missing" if spec.required else "Optional"

    path = getattr(state, "path", None) if state is not None else None
    dirty = bool(getattr(state, "dirty", False)) if state is not None else False
    if dirty:
        return f"{base} · cache (unsaved)"
    if path is not None:
        return f"{base} · {Path(path).name}"
    return base


def compute_stage_status(data_manager: Any, artifacts: list[DataArtifactSpec]) -> str:
    """Overall stage status: 'done' (output present) / 'ready' (inputs ready) / 'not_started'."""
    required_inputs_available = True
    output_available = False
    for spec in artifacts:
        available = artifact_available(data_manager, spec)
        if spec.role == "input" and spec.required and not available:
            required_inputs_available = False
        if spec.role == "output" and available:
            output_available = True
    if output_available:
        return "done"
    if required_inputs_available:
        return "ready"
    return "not_started"
