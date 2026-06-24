from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._experiments_list import (
    MiniRail,
    PIPELINE_STAGES,
    discover_experiment_folders,
)


def test_discover_finds_folders_with_all_required_inputs(tmp_path):
    for sub in ("Ctrl/pos_00", "Ctrl/pos_01"):
        d = tmp_path / sub
        d.mkdir(parents=True)
        (d / "beads.tif").write_bytes(b"x")
        (d / "reference.tif").write_bytes(b"x")
    incomplete = tmp_path / "Ctrl" / "pos_02"  # has beads but no reference
    incomplete.mkdir(parents=True)
    (incomplete / "beads.tif").write_bytes(b"x")

    found = discover_experiment_folders(tmp_path, ["beads.tif", "reference.tif"])

    assert sorted(Path(p).name for p in found) == ["pos_00", "pos_01"]


def test_discover_ignores_blank_names_and_missing_root(tmp_path):
    assert discover_experiment_folders(tmp_path / "nope", ["beads.tif"]) == []
    d = tmp_path / "pos"
    d.mkdir()
    (d / "beads.tif").write_bytes(b"x")
    # all-blank requirement set -> nothing to match on
    assert discover_experiment_folders(tmp_path, ["", None]) == []
    # blank entries are dropped; the real requirement still discovers the folder
    found = discover_experiment_folders(tmp_path, ["beads.tif", ""])
    assert sorted(Path(p).name for p in found) == ["pos"]


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_minirail_has_a_dot_per_pipeline_stage(app):
    rail = MiniRail()
    assert rail.stages == PIPELINE_STAGES
    assert len(PIPELINE_STAGES) == 4


def test_minirail_done_dot_is_filled_with_stage_accent(app):
    from napariTFM.widgets._ui_style import stage_accent

    rail = MiniRail()
    rail.set_statuses({"force": "done"})
    fill, ring = rail.appearance("force")
    assert fill == stage_accent("force")
    assert ring == stage_accent("force")


def test_minirail_ready_dot_is_hollow_ring(app):
    rail = MiniRail()
    rail.set_statuses({"displacement": "ready"})
    fill, ring = rail.appearance("displacement")
    assert fill is None
    assert ring is not None


def test_minirail_off_dot_is_recessed_and_distinct_from_not_started(app):
    rail = MiniRail()
    rail.set_statuses({"stress": "off"})
    off_fill, off_ring = rail.appearance("stress")
    none_fill, none_ring = rail.appearance("preprocessing")  # not_started default
    assert off_fill is None and none_fill is None
    assert off_ring != none_ring  # off uses the recessed grey, not the dim grey


from napariTFM.widgets._experiments_list import ExperimentRow, overall_status


def test_overall_status_done_when_all_enabled_stages_done():
    statuses = {"preprocessing": "done", "displacement": "done",
                "force": "done", "stress": "off"}
    assert overall_status(statuses) == "done"


def test_overall_status_running_when_any_stage_running():
    statuses = {"preprocessing": "done", "displacement": "running",
                "force": "not_started", "stress": "off"}
    assert overall_status(statuses) == "running"


def test_overall_status_queued_otherwise():
    statuses = {"preprocessing": "ready", "displacement": "not_started",
                "force": "not_started", "stress": "off"}
    assert overall_status(statuses) == "queued"


def test_experiment_row_exposes_path_and_name(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    assert row.path == "/data/Ctrl/pos_00"
    assert row.name == "pos_00"


def test_experiment_row_click_emits_selected_path(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    seen = []
    row.selected.connect(seen.append)
    row._emit_selected()  # proxy for mousePressEvent (offscreen-safe)
    assert seen == ["/data/Ctrl/pos_00"]


def test_experiment_row_set_selected_toggles_state(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    assert row.is_selected() is False
    row.set_selected(True)
    assert row.is_selected() is True


from napariTFM.widgets._experiments_list import ExperimentsList


def test_list_starts_empty(app):
    widget = ExperimentsList()
    assert widget.experiments() == []
    assert widget.active() is None


def test_set_experiments_populates_rows(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b"])
    assert widget.experiments() == ["/data/a", "/data/b"]


def test_add_folders_appends_without_duplicates(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    widget.add_folders(["/data/a", "/data/b"])
    assert widget.experiments() == ["/data/a", "/data/b"]


def test_selecting_a_row_sets_single_active_and_emits(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b"])
    seen = []
    widget.active_changed.connect(seen.append)

    widget.set_active("/data/b")

    assert widget.active() == "/data/b"
    assert seen == ["/data/b"]
    rows = widget._rows
    assert rows[1].is_selected() is True
    assert rows[0].is_selected() is False


def test_set_active_ignores_unknown_path(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    seen = []
    widget.active_changed.connect(seen.append)

    widget.set_active("/data/not-in-list")

    assert widget.active() is None
    assert seen == []


def test_set_experiments_clears_stale_active(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b"])
    widget.set_active("/data/b")

    widget.set_experiments(["/data/a", "/data/c"])  # /data/b is gone

    assert widget.active() is None


def test_meta_line_counts_experiments(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b", "/data/c"])
    assert "3 experiments" in widget.meta_text()


def test_meta_line_singular_for_one_experiment(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    assert "1 experiment" in widget.meta_text()
    assert "experiments" not in widget.meta_text()


def test_experiment_records_default_to_empty_metadata(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    assert widget.experiment_records() == [
        {"path": "/data/a", "input_files": {}, "columns": {}}
    ]


def test_add_folders_copies_column_config_onto_each_new_row(app):
    widget = ExperimentsList()
    widget.add_folders(
        ["/data/a", "/data/b"],
        input_files={"beads": "beads.tif", "reference": "reference.tif"},
        columns={"condition": "WT"},
    )
    records = widget.experiment_records()
    assert [r["path"] for r in records] == ["/data/a", "/data/b"]
    assert all(r["columns"] == {"condition": "WT"} for r in records)
    assert all(
        r["input_files"] == {"beads": "beads.tif", "reference": "reference.tif"}
        for r in records
    )
    # Each row owns a copy — mutating one returned record never leaks across rows.
    records[0]["columns"]["condition"] = "KO"
    assert widget.experiment_records()[1]["columns"]["condition"] == "WT"


def test_existing_rows_keep_metadata_when_a_second_batch_is_added(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a"], columns={"condition": "WT"})
    widget.add_folders(["/data/b"], columns={"condition": "KO"})
    by_path = {r["path"]: r["columns"] for r in widget.experiment_records()}
    assert by_path == {"/data/a": {"condition": "WT"}, "/data/b": {"condition": "KO"}}


def test_input_file_config_has_bead_reference_defaults(app):
    widget = ExperimentsList()
    cfg = widget.input_file_config()
    assert cfg["beads"] == "beads.tif"
    assert cfg["reference"] == "reference.tif"


def test_input_file_config_drops_blank_fields(app):
    widget = ExperimentsList()
    widget.file_name_inputs["cells"].setText("")
    assert "cells" not in widget.input_file_config()


def test_add_column_field_then_config_reads_name_value(app):
    widget = ExperimentsList()
    widget.add_column_field("condition", "WT")
    assert widget.column_config() == {"condition": "WT"}


def test_column_config_drops_rows_without_a_name(app):
    widget = ExperimentsList()
    widget.add_column_field("", "orphan")  # no column name -> ignored
    widget.add_column_field("condition", "WT")
    assert widget.column_config() == {"condition": "WT"}


def test_add_column_button_spawns_one_empty_field(app):
    widget = ExperimentsList()
    before = len(widget._column_fields)
    widget.add_column_btn.click()
    assert len(widget._column_fields) == before + 1
    assert widget.column_config() == {}  # the new field is blank


def _make_qualifying(tmp_path, *names):
    for name in names:
        d = tmp_path / name
        d.mkdir()
        (d / "beads.tif").write_bytes(b"x")
        (d / "reference.tif").write_bytes(b"x")


def test_discover_stages_folders_without_adding_them(app, tmp_path):
    _make_qualifying(tmp_path, "a", "b")
    widget = ExperimentsList()
    staged = widget.discover(tmp_path)
    assert sorted(Path(p).name for p in staged) == ["a", "b"]
    assert widget.discovered() == staged
    assert widget.experiments() == []  # discovery never adds on its own


def test_commit_adds_discovered_with_current_column_config(app, tmp_path):
    _make_qualifying(tmp_path, "a")
    widget = ExperimentsList()
    widget.file_name_inputs["cells"].setText("")
    widget.add_column_field("condition", "WT")
    widget.discover(tmp_path)
    widget.commit_discovered()
    records = widget.experiment_records()
    assert len(records) == 1
    assert records[0]["columns"] == {"condition": "WT"}
    assert records[0]["input_files"] == {
        "beads": "beads.tif",
        "reference": "reference.tif",
    }
    assert widget.discovered() == []  # staging cleared after commit


def test_commit_button_enables_only_after_discovery(app, tmp_path):
    widget = ExperimentsList()
    assert widget.commit_btn.isEnabled() is False
    _make_qualifying(tmp_path, "a")
    widget.discover(tmp_path)
    assert widget.commit_btn.isEnabled() is True
    widget.commit_discovered()
    assert widget.commit_btn.isEnabled() is False


def test_refresh_statuses_calls_status_fn_for_each_row(app):
    calls = []
    def status_fn(path):
        calls.append(path)
        return {"preprocessing": "done", "displacement": "not_started",
                "force": "not_started", "stress": "off"}
    widget = ExperimentsList(status_fn=status_fn)
    widget.set_experiments(["/data/a", "/data/b"])
    # set_experiments triggers an initial refresh; clear and call explicitly
    calls.clear()
    widget.refresh_statuses()
    assert set(calls) == {"/data/a", "/data/b"}
