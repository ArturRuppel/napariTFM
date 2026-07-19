"""Generated-artifact availability follows the in-memory value.

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


def test_raw_input_available_from_discovery_files_on_disk(tmp_path):
    # The active experiment's raw inputs read available when the discovery-named
    # files exist on disk — green before anything is loaded into memory.
    dm = DataManager()
    dm.set_active_inputs(tmp_path, {"beads": "b.tif", "reference": "r.tif"})

    assert dm.artifact_available("reference") is False
    assert dm.artifact_available("bead_stack") is False

    (tmp_path / "r.tif").write_bytes(b"x")
    assert dm.artifact_available("reference") is True
    assert dm.artifact_available("bead_stack") is False

    (tmp_path / "b.tif").write_bytes(b"x")
    assert dm.artifact_available("bead_stack") is True


def test_raw_input_disk_check_honours_discovery_names(tmp_path):
    # Discovery may name the files anything; disk presence follows those names.
    dm = DataManager()
    dm.set_active_inputs(tmp_path, {"masks": "segmentation.tif"})
    assert dm.artifact_available("mask_stack") is False
    (tmp_path / "segmentation.tif").write_bytes(b"x")
    assert dm.artifact_available("mask_stack") is True


def test_generated_output_never_reads_discovery_disk(tmp_path):
    # Only raw inputs gain the disk path; generated outputs stay memory-only.
    dm = DataManager()
    dm.set_active_inputs(tmp_path, {"beads": "b.tif"})
    (tmp_path / "b.tif").write_bytes(b"x")
    assert dm.artifact_available("displacement_results") is False


def test_clearing_active_inputs_reverts_to_memory_only(tmp_path):
    dm = DataManager()
    dm.set_active_inputs(tmp_path, {"beads": "b.tif"})
    (tmp_path / "b.tif").write_bytes(b"x")
    assert dm.artifact_available("bead_stack") is True

    dm.set_active_inputs(None, {})
    assert dm.artifact_available("bead_stack") is False


def test_batch_changes_coalesces_notifications():
    # One selection mutates several artifacts back-to-back; batch_changes must
    # collapse the burst into a single observer callback (the whole-list status
    # walk is what that callback drives, so firing it per-mutation is the freeze).
    dm = DataManager()
    calls = []
    dm.add_change_callback(lambda: calls.append(1))
    dm.set_active_inputs(None, {})  # sanity: one mutation -> one callback
    assert len(calls) == 1

    calls.clear()
    with dm.batch_changes():
        dm.clear_generated_results()
        dm.set_active_inputs(None, {})
        dm.set_mask_stack(np.ones((1, 4, 4), dtype=np.uint8))
    assert len(calls) == 1


def test_batch_changes_fires_nothing_when_nothing_changed():
    # clear_generated_results on an empty manager mutates nothing, so an empty
    # batch must not fire a spurious callback.
    dm = DataManager()
    calls = []
    dm.add_change_callback(lambda: calls.append(1))
    with dm.batch_changes():
        dm.clear_generated_results()
    assert calls == []


def test_batch_changes_is_reentrant():
    # Nested batches collapse into the outermost: callbacks fire once, on the
    # outer exit, not when an inner batch closes.
    dm = DataManager()
    calls = []
    dm.add_change_callback(lambda: calls.append(1))
    with dm.batch_changes():
        with dm.batch_changes():
            dm.set_mask_stack(np.ones((1, 4, 4), dtype=np.uint8))
        assert calls == []  # inner exit must not fire
    assert len(calls) == 1
