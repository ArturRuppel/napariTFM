from types import SimpleNamespace

from napariTFM.widgets._stage_data_status import (
    DataArtifactSpec,
    artifact_info_text,
    compute_stage_status,
)


class _DM:
    """Fake data manager: one artifact, with a fixed availability + state."""

    def __init__(self, available=False, state=None):
        self._available = available
        self._state = state

    def artifact_available(self, key):
        return self._available

    def get_artifact(self, key):
        return self._state


def _spec(role="output", required=True):
    return DataArtifactSpec("foo", "Foo", "foo", role, required=required)


def test_info_text_reports_array_shape_when_loaded():
    state = SimpleNamespace(value=SimpleNamespace(shape=(512, 512)), dirty=False, path=None, error="")
    assert artifact_info_text(_DM(available=True, state=state), _spec()) == "512×512"


def test_info_text_flags_cache_unsaved_when_dirty():
    # A cache-only (unsaved) value is what run-all clobbers — say so plainly.
    state = SimpleNamespace(value=object(), dirty=True, path=None, error="")
    assert artifact_info_text(_DM(available=True, state=state), _spec()) == "Loaded · cache (unsaved)"


def test_info_text_names_the_file_on_disk(tmp_path):
    path = tmp_path / "foo.npy"
    state = SimpleNamespace(value=object(), dirty=False, path=path, error="")
    assert artifact_info_text(_DM(available=True, state=state), _spec()) == "Loaded · foo.npy"


def test_info_text_surfaces_error():
    state = SimpleNamespace(value=object(), dirty=True, path=None, error="save failed")
    assert artifact_info_text(_DM(available=True, state=state), _spec()) == "save failed"


def test_info_text_saved_when_on_disk_without_value():
    state = SimpleNamespace(value=None, dirty=False, path=None, error="")
    assert artifact_info_text(_DM(available=True, state=state), _spec()) == "Saved"


def test_info_text_missing_vs_optional():
    assert artifact_info_text(_DM(available=False), _spec(role="input", required=True)) == "Missing"
    assert artifact_info_text(_DM(available=False), _spec(role="input", required=False)) == "Optional"


def test_compute_stage_status_tracks_inputs_and_outputs():
    inp = DataArtifactSpec("in", "In", "in", "input")
    out = DataArtifactSpec("out", "Out", "out", "output")

    class _Multi:
        def __init__(self, present):
            self._present = set(present)

        def artifact_available(self, key):
            return key in self._present

        def get_artifact(self, key):
            return None

    assert compute_stage_status(_Multi([]), [inp, out]) == "not_started"
    assert compute_stage_status(_Multi(["in"]), [inp, out]) == "ready"
    assert compute_stage_status(_Multi(["in", "out"]), [inp, out]) == "done"
