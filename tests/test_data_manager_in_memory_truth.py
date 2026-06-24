"""Generated-artifact availability follows the in-memory value (ROADMAP §4).

Interactive stage runs are preview-only: results live in memory and nothing is
written to disk. So ``artifact_available`` reflects whether the value is present
in memory — not whether any file exists on disk.
"""

import importlib.util as _ilu
from pathlib import Path

import numpy as np


def _load_real_data_manager():
    # Load by path: another test module stubs ``napariTFM.utilities.data_manager``
    # in ``sys.modules`` at import time, so a plain import is order-dependent.
    spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.data_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "data_manager.py",
    )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DataManager


DataManager = _load_real_data_manager()


def _stub_force_result():
    return np.zeros((2, 4, 4, 2), dtype=np.float32)


def test_generated_artifact_available_follows_memory(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)

    assert dm.artifact_available("force_results") is False
    dm.set_force_results(_stub_force_result())
    assert dm.artifact_available("force_results") is True


def test_generated_artifact_ignores_disk(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)

    # A bare file on disk is irrelevant — availability is in-memory only.
    (tmp_path / "force_results.npy").write_bytes(b"x")
    assert dm.artifact_available("force_results") is False


def test_generated_artifact_available_without_output_dir():
    dm = DataManager()
    assert dm.artifact_available("displacement_results") is False
    dm.set_displacement_results(object())
    assert dm.artifact_available("displacement_results") is True


def test_raw_input_available_follows_memory(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)

    assert dm.artifact_available("bead_stack") is False
    dm.set_bead_stack(np.zeros((2, 4, 4), dtype=np.float32))
    assert dm.artifact_available("bead_stack") is True


def test_results_are_not_marked_dirty(tmp_path):
    # Preview state, not an unsaved-to-disk state: no misleading "Unsaved" hint.
    dm = DataManager()
    dm.set_force_results(_stub_force_result())
    assert dm.get_artifact("force_results").dirty is False
