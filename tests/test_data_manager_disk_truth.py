import numpy as np

from napariTFM.utilities.data_manager import DataManager


def test_generated_artifact_available_follows_disk(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)

    assert dm.artifact_available("displacement_results") is False

    (tmp_path / "displacement_results.npy").write_bytes(b"x")
    assert dm.artifact_available("displacement_results") is True


def test_generated_artifact_unavailable_without_output_dir(tmp_path):
    dm = DataManager()
    assert dm.artifact_available("force_results") is False


def test_generated_artifact_ignores_in_memory_value(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)
    dm.get_artifact("force_results").value = object()
    assert dm.artifact_available("force_results") is False


def test_raw_input_available_follows_memory(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)

    assert dm.artifact_available("bead_stack") is False
    dm.set_bead_stack(np.zeros((2, 4, 4), dtype=np.float32))
    assert dm.artifact_available("bead_stack") is True


def test_artifact_disk_path_resolves_generated_filename(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)
    assert dm.artifact_disk_path("force_results") == tmp_path / "force_results.npy"
    assert dm.artifact_disk_path("bead_stack") is None
