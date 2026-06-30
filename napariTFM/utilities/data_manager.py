from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from napariTFM.backend.displacement_analysis import DisplacementResult
from napariTFM.backend.fttc import FTTCResult
from napariTFM.backend.msm import MSMResult


@dataclass
class ArtifactState:
    key: str
    label: str
    value: object = None
    path: Optional[Path] = None
    source: str = ""
    dirty: bool = False
    error: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None


class DataManager:
    """
    DataManager storing complete analysis result objects and pipeline artifact state.
    """

    ARTIFACT_LABELS = {
        "bead_stack": "Raw bead stack",
        "reference": "Raw reference image",
        "cell_stack": "Raw cell stack",
        "preprocessed_bead_stack": "Preprocessed bead stack",
        "preprocessed_reference": "Preprocessed reference image",
        "preprocessed_cell_stack": "Preprocessed cell stack",
        "displacement_results": "Displacement result",
        "force_results": "Force/traction result",
        "stress_results": "Stress result",
        "mask_stack": "Mask stack",
    }

    # Raw-input artifact key -> the discovery input-file slot that names its file
    # on disk. These are the only artifacts whose availability can be proven by a
    # file in the active experiment folder; everything else is in-memory only.
    RAW_INPUT_FILE_SLOTS = {
        "reference": "reference",
        "bead_stack": "beads",
        "cell_stack": "cells",
        "mask_stack": "masks",
    }

    def __init__(self):
        self._callbacks = []
        self._output_dir: Optional[Path] = None
        self._active_input_folder: Optional[Path] = None
        self._active_input_files: Dict[str, str] = {}
        self._artifacts: Dict[str, ArtifactState] = {
            key: ArtifactState(key=key, label=label)
            for key, label in self.ARTIFACT_LABELS.items()
        }

    def add_change_callback(self, callback: Callable[[], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_changed(self) -> None:
        for callback in list(self._callbacks):
            callback()

    @property
    def output_dir(self) -> Optional[Path]:
        return self._output_dir

    def set_output_dir(self, path) -> None:
        self._output_dir = Path(path).expanduser() if path else None
        self._notify_changed()

    def has_valid_output_dir(self) -> bool:
        return self._output_dir is not None and self._output_dir.exists() and self._output_dir.is_dir()

    def set_active_inputs(self, folder, input_files) -> None:
        """Point the raw-input disk check at the active experiment.

        *folder* is the experiment directory; *input_files* is the discovery
        config (``{"beads": "beads.tif", "reference": ..., ...}``) naming the raw
        files inside it. With these set, a raw input reads available the moment
        its file is on disk — so the preprocessing input dots turn green before
        anything is loaded into memory. Passing ``None``/``{}`` clears the check.
        """
        self._active_input_folder = Path(folder).expanduser() if folder else None
        self._active_input_files = dict(input_files or {})
        self._notify_changed()

    def _raw_input_on_disk(self, key: str) -> bool:
        """True when *key*'s discovery-named file exists in the active folder."""
        if self._active_input_folder is None:
            return False
        slot = self.RAW_INPUT_FILE_SLOTS.get(key)
        if slot is None:
            return False
        name = self._active_input_files.get(slot)
        if not name:
            return False
        return (self._active_input_folder / name).exists()

    def get_artifact(self, key: str) -> ArtifactState:
        return self._artifacts[key]

    def artifact_available(self, key: str) -> bool:
        """Availability follows the in-memory value (preview-only; ROADMAP §4).

        Interactive stage runs hold results in memory and write nothing to disk
        — batch is the only path to persisted data. So a generated artifact is
        "available" exactly when its value is present in memory. The one
        exception is a *raw input*: its discovery-named file on disk in the
        active experiment folder also counts, so the input dots read green from
        the files alone (see :meth:`set_active_inputs`).
        """
        if self.get_artifact(key).available:
            return True
        return self._raw_input_on_disk(key)

    def set_artifact(self, key: str, value, path=None, source: str = "", dirty: bool = False) -> None:
        state = self.get_artifact(key)
        state.value = value
        state.path = Path(path) if path else None
        state.source = source
        state.dirty = dirty
        state.error = ""
        self._notify_changed()

    def mark_artifact_error(self, key: str, error: str) -> None:
        state = self.get_artifact(key)
        state.error = error
        self._notify_changed()

    def set_bead_stack(self, data: np.ndarray, path=None, source: str = "") -> None:
        """Set and validate input bead stack."""
        self._validate_input_stack(data, "bead stack")
        self.set_artifact("bead_stack", data, path=path, source=source)

    def set_reference(self, data: np.ndarray, path=None, source: str = "") -> None:
        """Set and validate input reference image."""
        self._validate_reference_image(data)
        self.set_artifact("reference", data, path=path, source=source)

    def set_cell_stack(self, data: np.ndarray, path=None, source: str = "") -> None:
        """Set and validate input cell stack."""
        self._validate_input_stack(data, "cell stack")
        self.set_artifact("cell_stack", data, path=path, source=source)

    def set_preprocessed_bead_stack(self, data: np.ndarray, path=None, source: str = "", dirty: bool = False) -> None:
        """Set and validate preprocessed bead stack."""
        self._validate_input_stack(data, "bead stack")
        self.set_artifact("preprocessed_bead_stack", data, path=path, source=source, dirty=dirty)

    def set_preprocessed_cell_stack(self, data: np.ndarray, path=None, source: str = "", dirty: bool = False) -> None:
        """Set and validate preprocessed cell stack."""
        self._validate_input_stack(data, "cell stack")
        self.set_artifact("preprocessed_cell_stack", data, path=path, source=source, dirty=dirty)

    def set_preprocessed_reference(self, data: np.ndarray, path=None, source: str = "", dirty: bool = False) -> None:
        """Set and validate preprocessed reference image."""
        self._validate_reference_image(data)
        self.set_artifact("preprocessed_reference", data, path=path, source=source, dirty=dirty)

    def set_mask_stack(self, data: np.ndarray, path=None, source: str = "") -> None:
        """Set and validate mask stack."""
        self._validate_input_stack(data, "mask stack")
        self.set_artifact("mask_stack", data, path=path, source=source)

    # Pipeline dependency chain: (re)computing or clearing a stage makes every
    # stage downstream of it stale. Invalidating them keeps a stale result from
    # being shown or — now that interactive runs persist — written into a .ntfm
    # alongside a freshly recomputed upstream stage.
    _DOWNSTREAM = {
        "displacement_results": ("force_results", "stress_results"),
        "force_results": ("stress_results",),
    }

    def _invalidate_downstream(self, key: str) -> None:
        for downstream in self._DOWNSTREAM.get(key, ()):
            if self.get_artifact(downstream).value is not None:
                self.set_artifact(downstream, None)

    def clear_generated_results(self) -> None:
        """Drop every in-memory derived result (preprocessing, analyses, mask).

        Called when the active experiment changes so one experiment's results can
        never bleed into the next experiment's persisted ``.ntfm``. Raw inputs are
        left for the new experiment's load to overwrite; the output dir is kept.

        Mutates state in place and fires a single change notification at the end
        (not one per artifact), so observers reconcile once.
        """
        changed = False
        for key in (
            "preprocessed_bead_stack",
            "preprocessed_reference",
            "preprocessed_cell_stack",
            "displacement_results",
            "force_results",
            "stress_results",
            "mask_stack",
        ):
            state = self.get_artifact(key)
            if state.value is not None or state.error:
                changed = True
            state.value = None
            state.path = None
            state.source = ""
            state.dirty = False
            state.error = ""
        if changed:
            self._notify_changed()

    def set_displacement_results(self, results: DisplacementResult, path=None, source: str = "", dirty: bool = False) -> None:
        """Store displacement results and invalidate dependent analyses."""
        self.set_artifact("displacement_results", results, path=path, source=source, dirty=dirty)
        self._invalidate_downstream("displacement_results")

    def set_force_results(self, results: FTTCResult, path=None, source: str = "", dirty: bool = False) -> None:
        """Store force results and invalidate dependent analyses."""
        self.set_artifact("force_results", results, path=path, source=source, dirty=dirty)
        self._invalidate_downstream("force_results")

    def set_stress_results(self, results: MSMResult, path=None, source: str = "", dirty: bool = False) -> None:
        """Store stress results."""
        self.set_artifact("stress_results", results, path=path, source=source, dirty=dirty)

    # Input data properties
    @property
    def bead_stack(self) -> Optional[np.ndarray]:
        return self.get_artifact("bead_stack").value

    @property
    def reference(self) -> Optional[np.ndarray]:
        return self.get_artifact("reference").value

    @property
    def cell_stack(self) -> Optional[np.ndarray]:
        return self.get_artifact("cell_stack").value

    # Result properties
    @property
    def preprocessed_bead_stack(self) -> Optional[np.ndarray]:
        return self.get_artifact("preprocessed_bead_stack").value

    @property
    def preprocessed_reference(self) -> Optional[np.ndarray]:
        return self.get_artifact("preprocessed_reference").value

    @property
    def preprocessed_cell_stack(self) -> Optional[np.ndarray]:
        return self.get_artifact("preprocessed_cell_stack").value

    @property
    def mask_stack(self) -> Optional[np.ndarray]:
        return self.get_artifact("mask_stack").value

    @property
    def displacement_results(self) -> Optional[DisplacementResult]:
        return self.get_artifact("displacement_results").value

    @property
    def force_results(self) -> Optional[FTTCResult]:
        return self.get_artifact("force_results").value

    @property
    def stress_results(self) -> Optional[MSMResult]:
        return self.get_artifact("stress_results").value

    # Validation methods
    def _validate_input_stack(self, data: np.ndarray, name: str) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        if data.ndim not in [2, 3]:
            raise ValueError(f"{name} must be 2D or 3D (got {data.ndim}D)")

    def _validate_reference_image(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError("Reference image must be a numpy array")
        if data.ndim != 2:
            raise ValueError(f"Reference image must be 2D (got {data.ndim}D)")
