from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._stage_data_status import DataArtifactSpec, _ArtifactRow


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_row_shows_check_glyph_when_available(app):
    spec = DataArtifactSpec("foo", "Foo artifact", "foo", "output")
    row = _ArtifactRow(spec)

    row.refresh(available=True, info_text="512x512")

    assert row.glyph_label.text() == "✓"
    assert row.info_label.text() == "512x512"


def test_row_shows_cross_glyph_when_required_missing(app):
    spec = DataArtifactSpec("foo", "Foo", "foo", "input", required=True)
    row = _ArtifactRow(spec)

    row.refresh(available=False, info_text="Missing")

    assert row.glyph_label.text() == "✗"


def test_row_shows_circle_glyph_when_optional_missing(app):
    spec = DataArtifactSpec("foo", "Foo", "foo", "input", required=False)
    row = _ArtifactRow(spec)

    row.refresh(available=False, info_text="Optional")

    assert row.glyph_label.text() == "○"


def test_output_row_with_on_view_and_on_action_shows_both_buttons(app):
    views = []
    actions = []
    spec = DataArtifactSpec(
        "foo",
        "Foo",
        "foo",
        "output",
        on_view=lambda: views.append(True),
        on_action=lambda: actions.append(True),
    )
    row = _ArtifactRow(spec)
    row.refresh(available=True, info_text="ok")

    assert row.view_btn is not None
    assert row.action_btn is not None
    row.view_btn.click()
    row.action_btn.click()
    assert views == [True]
    assert actions == [True]


def test_input_row_missing_hides_view_button(app):
    spec = DataArtifactSpec("foo", "Foo", "foo", "input", on_action=lambda: None)
    row = _ArtifactRow(spec)
    row.refresh(available=False, info_text="Missing")

    assert row.view_btn is None or not row.view_btn.isVisible()
    assert row.action_btn is not None


def test_row_with_no_callables_has_no_action_buttons(app):
    spec = DataArtifactSpec("foo", "Foo", "foo", "output")
    row = _ArtifactRow(spec)
    row.refresh(available=True, info_text="ok")

    assert row.view_btn is None
    assert row.action_btn is None


def test_row_appends_unsaved_when_artifact_is_dirty(app):
    spec = DataArtifactSpec("foo", "Foo", "foo", "output")
    row = _ArtifactRow(spec)
    state = SimpleNamespace(value=object(), dirty=True, path=None, error="")

    row.refresh_state(state, info_text="Loaded")

    assert row.glyph_label.text() == "✓"
    assert row.info_label.text() == "Loaded · Unsaved"


def test_row_appends_saved_filename_when_path_exists(app, tmp_path):
    spec = DataArtifactSpec("foo", "Foo", "foo", "output")
    row = _ArtifactRow(spec)
    path = tmp_path / "foo.npy"
    state = SimpleNamespace(value=object(), dirty=False, path=path, error="")

    row.refresh_state(state, info_text="Loaded")

    assert row.info_label.text() == "Loaded · foo.npy"
    assert row.info_label.toolTip() == str(path)


def test_row_shows_error_glyph_and_message(app):
    spec = DataArtifactSpec("foo", "Foo", "foo", "output")
    row = _ArtifactRow(spec)
    state = SimpleNamespace(value=object(), dirty=True, path=None, error="save failed")

    row.refresh_state(state, info_text="Loaded")

    assert row.glyph_label.text() == "⚠"
    assert row.info_label.text() == "save failed"


def test_row_shows_available_glyph_when_on_disk_without_value(app):
    from napariTFM.widgets._ui_style import STATUS_GLYPHS

    spec = DataArtifactSpec("foo", "Foo", "foo", "output")
    row = _ArtifactRow(spec)

    class _State:
        value = None
        error = ""
        path = None
        dirty = False

    row.refresh_state(_State(), info_text="Saved", available=True)

    assert row.glyph_label.text() == STATUS_GLYPHS["available"]
    assert "Saved" in row.info_label.text()
