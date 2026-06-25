from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QToolButton

from napariTFM.widgets._stage_data_status import DataArtifactSpec
from napariTFM.widgets._stage_file_status import StageFileStatusRow
from napariTFM.widgets._ui_style import file_status_color


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeDataManager:
    """Minimal data manager: an artifact is 'available' iff its key is present."""

    def __init__(self, present=(), states=None):
        self._present = set(present)
        self._states = states or {}

    def artifact_available(self, key):
        return key in self._present

    def get_artifact(self, key):
        return self._states.get(key)


def _specs(calls):
    """Two inputs (one required, one optional) and one output, with click hooks."""
    return [
        DataArtifactSpec(
            "ref", "Reference", "ref", "input",
            on_view=lambda: calls.append("view:ref"),
            on_action=lambda: calls.append("assign:ref"),
        ),
        DataArtifactSpec(
            "mask", "Mask", "mask", "input", required=False,
            on_action=lambda: calls.append("assign:mask"),
        ),
        DataArtifactSpec(
            "out", "Result", "out", "output",
            on_view=lambda: calls.append("view:out"),
        ),
    ]


def test_row_makes_one_dot_per_artifact_keyed_by_key(app):
    row = StageFileStatusRow("force", _FakeDataManager(), _specs([]))
    assert set(row.dots) == {"ref", "mask", "out"}
    assert all(isinstance(dot, QToolButton) for dot in row.dots.values())
    assert row.objectName() == "stage_force_file_status_row"


def test_dots_colour_by_presence(app):
    # ref present, mask (optional) absent, out (required) absent.
    row = StageFileStatusRow("force", _FakeDataManager(present=["ref"]), _specs([]))
    row.refresh()
    assert file_status_color("present") in row.dots["ref"].styleSheet()
    assert file_status_color("optional") in row.dots["mask"].styleSheet()
    assert file_status_color("missing") in row.dots["out"].styleSheet()


def test_refresh_returns_pipeline_status(app):
    specs = _specs([])
    # Required input absent → not_started.
    assert StageFileStatusRow("force", _FakeDataManager(), specs).refresh() == "not_started"
    # Required input present, output absent → ready.
    assert StageFileStatusRow("force", _FakeDataManager(present=["ref"]), specs).refresh() == "ready"
    # Output present → done.
    assert StageFileStatusRow("force", _FakeDataManager(present=["ref", "out"]), specs).refresh() == "done"


def test_clicking_present_dot_views_it(app):
    calls = []
    row = StageFileStatusRow("force", _FakeDataManager(present=["ref"]), _specs(calls))
    row.refresh()
    row.dots["ref"].click()
    assert calls == ["view:ref"]


def test_clicking_missing_input_assigns_active_layer(app):
    calls = []
    row = StageFileStatusRow("force", _FakeDataManager(), _specs(calls))
    row.refresh()
    row.dots["ref"].click()
    assert calls == ["assign:ref"]


def test_missing_output_dot_is_inert(app):
    calls = []
    row = StageFileStatusRow("force", _FakeDataManager(), _specs(calls))
    row.refresh()
    row.dots["out"].click()  # no on_action, not present → nothing happens
    assert calls == []
    assert row.dots["out"].isEnabled() is False


def test_arrow_separates_inputs_from_outputs_with_a_tooltip(app):
    row = StageFileStatusRow("force", _FakeDataManager(), _specs([]))
    assert isinstance(row.arrow, QLabel)
    assert "→" in row.arrow.text()
    assert row.arrow.toolTip()  # explains inputs-produce-outputs


def test_dot_tooltip_names_the_artifact_and_state(app):
    state = SimpleNamespace(value=object(), path="/data/ref.tif", dirty=False, error="")
    row = StageFileStatusRow(
        "force", _FakeDataManager(present=["ref"], states={"ref": state}), _specs([])
    )
    row.refresh()
    tip = row.dots["ref"].toolTip()
    assert "Reference" in tip
    assert "ref.tif" in tip
