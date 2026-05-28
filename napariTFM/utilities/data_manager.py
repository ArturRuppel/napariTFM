from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import tifffile

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

    GENERATED_FILENAMES = {
        "preprocessed_bead_stack": "preprocessed_beads.tif",
        "preprocessed_reference": "preprocessed_reference.tif",
        "preprocessed_cell_stack": "preprocessed_cells.tif",
        "displacement_results": "displacement_results.npy",
        "force_results": "force_results.npy",
        "stress_results": "stress_results.npy",
    }

    def __init__(self):
        self._callbacks = []
        self._output_dir: Optional[Path] = None
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

    def ensure_output_dir(self) -> Path:
        if self._output_dir is None:
            raise ValueError("Output directory is not set")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir

    def get_artifact(self, key: str) -> ArtifactState:
        return self._artifacts[key]

    def artifact_disk_path(self, key: str):
        """Expected on-disk path for a generated artifact, or None if N/A."""
        filename = self.GENERATED_FILENAMES.get(key)
        if filename is None or self._output_dir is None:
            return None
        return self._output_dir / filename

    def artifact_available(self, key: str) -> bool:
        if key in self.GENERATED_FILENAMES:
            path = self.artifact_disk_path(key)
            return path is not None and path.exists()
        return self.get_artifact(key).available

    def set_artifact(self, key: str, value, path=None, source: str = "", dirty: bool = False) -> None:
        state = self.get_artifact(key)
        state.value = value
        state.path = Path(path) if path else None
        state.source = source
        state.dirty = dirty
        state.error = ""
        self._notify_changed()

    def mark_artifact_saved(self, key: str, path) -> None:
        state = self.get_artifact(key)
        state.path = Path(path)
        state.source = "generated"
        state.dirty = False
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

    def set_preprocessed_bead_stack(self, data: np.ndarray, path=None, source: str = "", dirty: bool = True) -> None:
        """Set and validate preprocessed bead stack."""
        self._validate_input_stack(data, "bead stack")
        self.set_artifact("preprocessed_bead_stack", data, path=path, source=source, dirty=dirty)

    def set_preprocessed_cell_stack(self, data: np.ndarray, path=None, source: str = "", dirty: bool = True) -> None:
        """Set and validate preprocessed cell stack."""
        self._validate_input_stack(data, "cell stack")
        self.set_artifact("preprocessed_cell_stack", data, path=path, source=source, dirty=dirty)

    def set_preprocessed_reference(self, data: np.ndarray, path=None, source: str = "", dirty: bool = True) -> None:
        """Set and validate preprocessed reference image."""
        self._validate_reference_image(data)
        self.set_artifact("preprocessed_reference", data, path=path, source=source, dirty=dirty)

    def set_mask_stack(self, data: np.ndarray, path=None, source: str = "") -> None:
        """Set and validate mask stack."""
        self._validate_input_stack(data, "mask stack")
        self.set_artifact("mask_stack", data, path=path, source=source)

    def set_displacement_results(self, results: DisplacementResult, path=None, source: str = "", dirty: bool = True) -> None:
        """Store displacement results and invalidate dependent analyses."""
        self.set_artifact("displacement_results", results, path=path, source=source, dirty=dirty)

    def set_force_results(self, results: FTTCResult, path=None, source: str = "", dirty: bool = True) -> None:
        """Store force results and invalidate dependent analyses."""
        self.set_artifact("force_results", results, path=path, source=source, dirty=dirty)

    def set_stress_results(self, results: MSMResult, path=None, source: str = "", dirty: bool = True) -> None:
        """Store stress results."""
        self.set_artifact("stress_results", results, path=path, source=source, dirty=dirty)

    def auto_save_artifact(self, key: str, pixel_size: Optional[float] = None, frame_interval: Optional[float] = None) -> Optional[Path]:
        state = self.get_artifact(key)
        if state.value is None:
            return None
        output_dir = self.ensure_output_dir()
        path = output_dir / self.GENERATED_FILENAMES[key]
        if key.startswith("preprocessed_"):
            self._save_calibrated_tiff(state.value, path, pixel_size=pixel_size, frame_interval=frame_interval)
        else:
            np.save(path, state.value)
        self.mark_artifact_saved(key, path)
        return path

    def auto_save_generated_artifacts(self, keys, pixel_size: Optional[float] = None, frame_interval: Optional[float] = None) -> Dict[str, Path]:
        saved = {}
        for key in keys:
            path = self.auto_save_artifact(key, pixel_size=pixel_size, frame_interval=frame_interval)
            if path is not None:
                saved[key] = path
        return saved

    def load_result_artifact(self, key: str, path) -> None:
        result = np.load(path, allow_pickle=True).item()
        if key == "displacement_results":
            self.set_displacement_results(result, path=path, source="file", dirty=False)
        elif key == "force_results":
            self.set_force_results(result, path=path, source="file", dirty=False)
        elif key == "stress_results":
            self.set_stress_results(result, path=path, source="file", dirty=False)
        else:
            raise ValueError(f"Unsupported result artifact: {key}")

    def _save_calibrated_tiff(
            self,
            data: np.ndarray,
            filepath: Path,
            pixel_size: Optional[float] = None,
            frame_interval: Optional[float] = None
    ) -> None:
        pixel_size = 1.0 if pixel_size is None else pixel_size
        frame_interval = 1.0 if frame_interval is None else frame_interval
        data_float = data.astype(float)
        data_range = data_float.max() - data_float.min()
        if data_range == 0:
            data_16bit = np.zeros_like(data_float, dtype=np.uint16)
        else:
            data_normalized = (data_float - data_float.min()) / data_range
            data_16bit = (data_normalized * 65535).astype(np.uint16)

        imagej_metadata = {
            'ImageJ': '1.53c',
            'spacing': pixel_size,
            'unit': 'um',
            'frame_interval': frame_interval,
            'frame_interval_unit': 'minute'
        }
        if data.ndim > 2:
            imagej_metadata.update({
                'frames': data.shape[0],
                'slices': 1,
                'channels': 1
            })
        metadata = {
            'PhysicalSizeX': pixel_size,
            'PhysicalSizeXUnit': 'um',
            'PhysicalSizeY': pixel_size,
            'PhysicalSizeYUnit': 'um',
            'TimeIncrement': frame_interval,
            'TimeIncrementUnit': 'min',
            **imagej_metadata
        }
        tifffile.imwrite(
            str(filepath),
            data_16bit,
            imagej=True,
            metadata=metadata,
            resolution=(1 / pixel_size, 1 / pixel_size),
            photometric='minisblack'
        )

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
