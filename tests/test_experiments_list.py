from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLabel

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


def test_adding_rows_to_empty_list_preloads_first_row(app):
    widget = ExperimentsList()
    seen = []
    widget.active_changed.connect(seen.append)

    widget.add_folders(["/data/a", "/data/b"])

    assert widget.active() == "/data/a"
    assert seen == ["/data/a"]


def test_adding_more_rows_does_not_steal_existing_active(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a", "/data/b"])
    widget.set_active("/data/b")

    widget.add_folders(["/data/c"])

    assert widget.active() == "/data/b"


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

    assert widget.active() == "/data/a"
    assert seen == []


def test_follow_streaming_highlights_row_without_emitting(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b"])
    seen = []
    widget.active_changed.connect(seen.append)

    widget.follow_streaming("/data/b")

    # Row highlight + active pointer track the streamed position...
    assert widget.active() == "/data/b"
    rows = widget._rows
    assert rows[1].is_selected() is True
    assert rows[0].is_selected() is False
    # ...but no active_changed fires (the sink owns the viewer mid-run, so the
    # heavy clear-and-reload that signal drives must not happen).
    assert seen == []


def test_follow_streaming_ignores_unknown_path(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    widget.set_active("/data/a")

    widget.follow_streaming("/data/not-in-list")

    assert widget.active() == "/data/a"  # unchanged


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


def _make_qualifying(tmp_path, *names):
    for name in names:
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "beads.tif").write_bytes(b"x")
        (d / "reference.tif").write_bytes(b"x")


def test_discover_stages_folders_without_adding_them(app, tmp_path):
    _make_qualifying(tmp_path, "a", "b")
    widget = ExperimentsList()
    staged = widget.discover(tmp_path)
    assert sorted(Path(p).name for p in staged) == ["a", "b"]
    assert widget.discovered() == staged
    assert widget.experiments() == []  # discovery never adds on its own


def test_commit_adds_discovered_with_nesting_columns(app, tmp_path):
    _make_qualifying(tmp_path, "Ctrl/pos_00")
    widget = ExperimentsList()
    widget.file_name_inputs["cells"].setText("")
    widget.file_name_inputs["masks"].setText("")
    widget.discover(tmp_path)
    widget.commit_discovered()
    records = widget.experiment_records()
    assert len(records) == 1
    # Each nesting level under the discovery root becomes a column.
    assert records[0]["columns"] == {"Column 1": "Ctrl", "Column 2": "pos_00"}
    assert widget.column_names() == ["Column 1", "Column 2"]
    assert records[0]["input_files"] == {
        "beads": "beads.tif",
        "reference": "reference.tif",
    }
    assert widget.discovered() == []  # staging cleared after commit


def test_commit_columns_pad_to_max_nesting_depth(app, tmp_path):
    # Ragged depths: a shallow folder leaves deeper columns blank, not missing.
    _make_qualifying(tmp_path, "Ctrl/pos_00")
    _make_qualifying(tmp_path, "solo")
    widget = ExperimentsList()
    widget.file_name_inputs["cells"].setText("")
    widget.file_name_inputs["masks"].setText("")
    widget.discover(tmp_path)
    widget.commit_discovered()
    assert widget.column_names() == ["Column 1", "Column 2"]
    by_leaf = {Path(r["path"]).name: r["columns"] for r in widget.experiment_records()}
    assert by_leaf["pos_00"] == {"Column 1": "Ctrl", "Column 2": "pos_00"}
    assert by_leaf["solo"] == {"Column 1": "solo", "Column 2": ""}


def test_commit_button_enables_only_after_discovery(app, tmp_path):
    widget = ExperimentsList()
    assert widget.commit_btn.isEnabled() is False
    _make_qualifying(tmp_path, "a")
    widget.discover(tmp_path)
    assert widget.commit_btn.isEnabled() is True
    widget.commit_discovered()
    assert widget.commit_btn.isEnabled() is False


def test_discover_renders_preview_rows_in_table(app, tmp_path):
    _make_qualifying(tmp_path, "a", "b")
    widget = ExperimentsList()
    widget.discover(tmp_path)
    assert len(widget._preview_rows) == 2
    assert all(row.is_preview for row in widget._preview_rows)
    # Preview rows are not committed rows.
    assert widget.experiments() == []


def test_discover_again_replaces_rather_than_merges_preview(app, tmp_path):
    # Two non-overlapping roots (sibling subfolders of tmp_path) — discover()
    # scans recursively, so nesting one root inside the other would make the
    # first call legitimately find both folders, masking the replace behavior
    # this test targets.
    first_root = tmp_path / "first"
    _make_qualifying(first_root, "a")
    other_root = tmp_path / "other"
    other_root.mkdir()
    _make_qualifying(other_root, "z")
    widget = ExperimentsList()
    widget.discover(first_root)
    assert len(widget._preview_rows) == 1
    widget.discover(other_root)
    assert len(widget._preview_rows) == 1
    assert Path(widget._preview_rows[0].path).name == "z"


def test_preview_row_click_toggles_selection(app, tmp_path):
    _make_qualifying(tmp_path, "a")
    widget = ExperimentsList()
    widget.discover(tmp_path)
    row = widget._preview_rows[0]
    row.clicked.emit(row.path, 0)
    assert row.path in widget._discovered_selected
    assert widget.delete_btn.isEnabled() is True
    row.clicked.emit(row.path, 0)
    assert row.path not in widget._discovered_selected
    assert widget.delete_btn.isEnabled() is False


def test_delete_selected_removes_preview_rows_before_committing(app, tmp_path):
    _make_qualifying(tmp_path, "a", "b")
    widget = ExperimentsList()
    widget.discover(tmp_path)
    row_to_drop = widget._preview_rows[0]
    row_to_drop.clicked.emit(row_to_drop.path, 0)
    widget.delete_selected()
    assert len(widget._preview_rows) == 1
    assert len(widget.discovered()) == 1
    # Committed table untouched.
    assert widget.experiments() == []


def test_commit_discovered_clears_preview_rows_and_hardens(app, tmp_path):
    _make_qualifying(tmp_path, "Ctrl/pos_00")
    widget = ExperimentsList()
    widget.file_name_inputs["cells"].setText("")
    widget.file_name_inputs["masks"].setText("")
    widget.discover(tmp_path)
    assert len(widget._preview_rows) == 1
    row = widget._preview_rows[0]
    row.clicked.emit(row.path, 0)  # select it
    widget.commit_discovered()
    assert widget._preview_rows == []
    assert len(widget._rows) == 1
    assert widget._rows[0].is_preview is False
    # The stale preview-row selection must not leak past commit (it would
    # otherwise keep the delete button/keyboard shortcut wrongly armed for a
    # row that no longer exists as a preview).
    assert widget._discovered_selected == set()


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


# -- Run all (P4.3) ------------------------------------------------------

def test_run_all_button_exists_and_is_disabled_when_empty(app):
    widget = ExperimentsList()
    assert hasattr(widget, "run_all_btn")
    assert widget.run_all_btn.isEnabled() is False


def test_run_all_button_enables_once_experiments_exist(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    assert widget.run_all_btn.isEnabled() is True


def test_run_all_button_click_emits_run_all_requested(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    seen = []
    widget.run_all_requested.connect(lambda: seen.append(True))
    widget.run_all_btn.click()
    assert seen == [True]


def test_run_all_button_becomes_cancel_while_active(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    assert widget.run_all_btn.text() == "Run all"

    widget.set_run_all_active(True)
    assert widget.run_all_btn.text() == "Cancel"
    assert widget.run_all_btn.isEnabled() is True

    widget.set_run_all_active(False)
    assert widget.run_all_btn.text() == "Run all"


def test_active_run_all_button_click_emits_cancel_not_run(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    runs, cancels = [], []
    widget.run_all_requested.connect(lambda: runs.append(True))
    widget.cancel_run_all_requested.connect(lambda: cancels.append(True))

    widget.set_run_all_active(True)
    widget.run_all_btn.click()
    assert runs == []
    assert cancels == [True]


def test_mark_running_sets_that_rows_enabled_dots_to_running(app):
    def status_fn(path):
        return {"preprocessing": "ready", "displacement": "not_started",
                "force": "not_started", "stress": "off"}
    widget = ExperimentsList(status_fn=status_fn)
    widget.set_experiments(["/data/a", "/data/b"])

    widget.mark_running("/data/a")
    row_a = widget._rows[0]
    # Enabled stages flip to running; an off stage stays off.
    assert row_a.mini_rail._statuses["preprocessing"] == "running"
    assert row_a.mini_rail._statuses["stress"] == "off"
    # Untouched row keeps its computed statuses.
    assert widget._rows[1].mini_rail._statuses["preprocessing"] == "ready"


def test_mark_running_unknown_path_is_a_noop(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    widget.mark_running("/data/zzz")  # must not raise


def test_set_records_rebuilds_rows_with_per_row_metadata(app):
    widget = ExperimentsList()
    widget.set_records([
        {"path": "/data/a", "input_files": {"beads": "b.tif"}, "columns": {"day": "1"}},
        {"path": "/data/b", "input_files": {"beads": "b.tif"}, "columns": {"day": "2"}},
    ])
    assert widget.experiments() == ["/data/a", "/data/b"]
    records = widget.experiment_records()
    assert records[0]["columns"] == {"day": "1"}
    assert records[1]["columns"] == {"day": "2"}
    assert records[0]["input_files"] == {"beads": "b.tif"}


def test_set_records_replaces_previous_rows(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/old"])
    widget.set_records([
        {"path": "/data/new", "input_files": {}, "columns": {}},
    ])
    assert widget.experiments() == ["/data/new"]


def test_set_records_seeds_input_file_header_from_first_record(app):
    widget = ExperimentsList()
    widget.set_records([
        {"path": "/data/a", "input_files": {"beads": "raw.tif", "reference": "ref.tif"}, "columns": {}},
    ])
    assert widget.file_name_inputs["beads"].text() == "raw.tif"
    assert widget.file_name_inputs["reference"].text() == "ref.tif"


def test_set_records_empty_clears_the_list(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    widget.set_records([])
    assert widget.experiments() == []


def test_input_file_config_has_optional_masks_default(app):
    widget = ExperimentsList()
    assert "masks" in widget.file_name_inputs
    assert widget.input_file_config()["masks"] == "masks.tif"


def test_masks_field_is_optional_and_excluded_from_discovery(app, tmp_path):
    # Masks are not a discovery requirement (only beads + reference are).
    _make_qualifying(tmp_path, "a")  # writes beads.tif + reference.tif, no masks
    widget = ExperimentsList()
    assert sorted(Path(p).name for p in widget.discover(tmp_path)) == ["a"]


def test_masks_field_blank_is_dropped_from_config(app):
    widget = ExperimentsList()
    widget.file_name_inputs["masks"].setText("")
    assert "masks" not in widget.input_file_config()


# ── editable columns, multi-select & delete ─────────────────────────────

def test_add_folders_with_columns_builds_shared_header(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a", "/data/b"], columns={"condition": "WT"})
    assert widget.column_names() == ["condition"]


def test_rename_column_renames_table_wide_and_keeps_values(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a"], columns={"Level 1": "Ctrl"})
    widget.add_folders(["/data/b"], columns={"Level 1": "KO"})
    widget.rename_column(0, "condition")
    assert widget.column_names() == ["condition"]
    by_path = {r["path"]: r["columns"] for r in widget.experiment_records()}
    assert by_path == {
        "/data/a": {"condition": "Ctrl"},
        "/data/b": {"condition": "KO"},
    }


def test_blank_column_name_drops_from_records(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a"], columns={"Level 1": "Ctrl"})
    widget.rename_column(0, "")
    assert widget.experiment_records()[0]["columns"] == {}


def test_header_field_edit_renames_column(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a"], columns={"Level 1": "Ctrl"})
    field = widget._header_fields[0]
    field.setText("position")
    field.editingFinished.emit()
    assert widget.column_names() == ["position"]
    assert widget.experiment_records()[0]["columns"] == {"position": "Ctrl"}


def test_plain_click_single_selects_and_activates(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b", "/data/c"])
    seen = []
    widget.active_changed.connect(seen.append)
    widget._on_row_clicked("/data/b", 0)
    assert widget.active() == "/data/b"
    assert widget.selected_rows() == ["/data/b"]
    assert seen == ["/data/b"]


def test_ctrl_click_toggles_multi_selection(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b", "/data/c"])
    widget._on_row_clicked("/data/a", 0)  # plain: select a
    widget._on_row_clicked("/data/c", 1)  # ctrl: add c
    assert widget.selected_rows() == ["/data/a", "/data/c"]
    widget._on_row_clicked("/data/a", 1)  # ctrl: remove a
    assert widget.selected_rows() == ["/data/c"]


def test_shift_click_selects_inclusive_range(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b", "/data/c", "/data/d"])
    widget._on_row_clicked("/data/b", 0)  # anchor at b
    widget._on_row_clicked("/data/d", 2)  # shift: b..d
    assert widget.selected_rows() == ["/data/b", "/data/c", "/data/d"]


def test_delete_selected_removes_rows(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b", "/data/c"])
    widget._on_row_clicked("/data/a", 0)
    widget._on_row_clicked("/data/c", 1)
    widget.delete_selected()
    assert widget.experiments() == ["/data/b"]
    assert widget.selected_rows() == []


def test_delete_selected_clears_active_when_active_deleted(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b"])
    widget._on_row_clicked("/data/a", 0)
    assert widget.active() == "/data/a"
    widget.delete_selected()
    assert widget.active() is None


def test_delete_button_enables_only_with_a_selection(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    widget.set_active(None)
    assert widget.delete_btn.isEnabled() is False
    widget._on_row_clicked("/data/a", 0)
    assert widget.delete_btn.isEnabled() is True


def test_column_header_has_one_editable_field_per_column(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a"], columns={"Column 1": "Ctrl", "Column 2": "pos_00"})
    assert [f.text() for f in widget._header_fields] == ["Column 1", "Column 2"]


# ── styling fidelity (mockup v2 aggregation layer) ──────────────────────
from qtpy.QtGui import QFont
from napariTFM.widgets._ui_style import experiment_status_color


def test_experiment_row_name_and_chip_use_the_standard_app_font(app):
    """Rows inherit napari's font — no custom monospace override remains."""
    row = ExperimentRow("/data/Ctrl/pos_00")
    assert row._name_label.font().family() == QFont().family()
    assert row._chip.font().family() == QFont().family()


def test_experiment_row_chip_is_colored_by_overall_status(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    row.set_stage_statuses({"preprocessing": "done", "displacement": "running",
                            "force": "not_started", "stress": "off"})
    assert row._chip.text() == "run"
    assert experiment_status_color("run") in row._chip.styleSheet()

    row.set_stage_statuses({"preprocessing": "done", "displacement": "done",
                            "force": "done", "stress": "off"})
    assert row._chip.text() == "done"
    assert experiment_status_color("done") in row._chip.styleSheet()


def test_experiment_row_selected_lifts_its_background(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    row.set_selected(False)
    assert "transparent" in row.styleSheet()
    row.set_selected(True)
    # selected rows raise onto a translucent white surface
    assert "rgba(255, 255, 255" in row.styleSheet()


# ── project-level config relocated into the aggregation layer ────────────
from qtpy.QtCore import QObject, Signal as _Signal


class _StubPM(QObject):
    parameter_changed = _Signal(str, object)

    def __init__(self):
        super().__init__()
        self._values = {"pixel_size": 0.1, "frame_interval": 1.0}
        self.ui_writes = []

    def get_ui_parameter(self, name):
        return self._values[name]

    def set_ui_parameter(self, name, value):
        self.ui_writes.append((name, value))
        self._values[name] = value
        self.parameter_changed.emit(name, value)


class _StubDM:
    def __init__(self):
        self.output_dir = None
        self._cbs = []

    def add_change_callback(self, cb):
        self._cbs.append(cb)

    def set_output_dir(self, path):
        self.output_dir = Path(path) if path is not None else None
        for cb in self._cbs:
            cb()


from napariTFM.widgets._collapsible_section import CollapsibleSection


def test_setup_section_exists_and_starts_expanded(app):
    widget = ExperimentsList()
    assert isinstance(widget.setup_section, CollapsibleSection)
    assert widget.setup_section.is_expanded is True


def test_setup_section_holds_calibration_input_files_and_output_dir(app):
    widget = ExperimentsList(parameter_manager=_StubPM(), data_manager=_StubDM())
    # All three groups of fields/buttons still exist, now built by the setup
    # section rather than two separate top-level layouts.
    assert "pixel_size" in widget.calibration_controls
    assert "beads" in widget.file_name_inputs
    assert widget.choose_output_dir_btn is not None


def test_setup_section_auto_collapses_after_first_commit(app, tmp_path):
    _make_qualifying(tmp_path, "a")
    widget = ExperimentsList()
    assert widget.setup_section.is_expanded is True
    widget.discover(tmp_path)
    widget.commit_discovered()
    assert widget.setup_section.is_expanded is False


def test_setup_section_does_not_collapse_while_list_stays_empty(app, tmp_path):
    _make_qualifying(tmp_path, "a")
    widget = ExperimentsList()
    widget.discover(tmp_path)  # staged but not committed
    assert widget.setup_section.is_expanded is True


def test_setup_section_is_manually_reexpandable_after_auto_collapse(app, tmp_path):
    _make_qualifying(tmp_path, "a")
    widget = ExperimentsList()
    widget.discover(tmp_path)
    widget.commit_discovered()
    assert widget.setup_section.is_expanded is False
    widget.setup_section.set_expanded(True)
    assert widget.setup_section.is_expanded is True


def test_loading_records_collapses_setup_section(app):
    widget = ExperimentsList()
    assert widget.setup_section.is_expanded is True
    widget.set_records([{"path": "/data/a", "input_files": {}, "columns": {}}])
    assert widget.setup_section.is_expanded is False


def test_experiments_list_owns_calibration_controls(app):
    widget = ExperimentsList(parameter_manager=_StubPM())
    assert "pixel_size" in widget.calibration_controls
    assert "frame_interval" in widget.calibration_controls
    assert float(widget.calibration_controls["pixel_size"].text()) == 0.1


def test_calibration_field_writes_through_ui_parameter_api(app):
    pm = _StubPM()
    widget = ExperimentsList(parameter_manager=pm)
    field = widget.calibration_controls["pixel_size"]
    field.setText("0.108")
    field.editingFinished.emit()
    assert ("pixel_size", 0.108) in pm.ui_writes


def test_calibration_field_syncs_from_parameter_changed(app):
    pm = _StubPM()
    widget = ExperimentsList(parameter_manager=pm)
    pm.set_ui_parameter("frame_interval", 2.5)
    assert float(widget.calibration_controls["frame_interval"].text()) == 2.5


def test_output_dir_starts_as_unset_add_affordance(app):
    widget = ExperimentsList(data_manager=_StubDM())
    widget.show()
    assert widget.output_dir_label.isVisible() is False
    assert widget.choose_output_dir_btn.text() == "Add custom output directory"
    assert widget.clear_output_dir_btn.isVisible() is False


def test_output_dir_shows_path_and_clear_button_once_set(app, tmp_path):
    dm = _StubDM()
    widget = ExperimentsList(data_manager=dm)
    widget.show()  # isVisible() reflects ancestor visibility; must actually show
    dm.set_output_dir(tmp_path)
    assert widget.output_dir_label.isVisible() is True
    assert widget.output_dir_label.text() == str(tmp_path)
    assert widget.clear_output_dir_btn.isVisible() is True
    assert widget.choose_output_dir_btn.text() == "Change output directory"


def test_clear_output_dir_resets_manager_and_label(app, tmp_path):
    dm = _StubDM()
    widget = ExperimentsList(data_manager=dm)
    widget.show()
    dm.set_output_dir(tmp_path)
    widget._clear_output_dir()
    assert dm.output_dir is None
    assert widget.output_dir_label.isVisible() is False
    assert widget.choose_output_dir_btn.text() == "Add custom output directory"


def test_apply_output_dir_sets_manager_and_emits(app, tmp_path):
    dm = _StubDM()
    widget = ExperimentsList(data_manager=dm)
    seen = []
    widget.output_dir_changed.connect(lambda: seen.append(True))
    widget._apply_output_dir(str(tmp_path))
    assert dm.output_dir == Path(tmp_path)
    assert seen == [True]


def test_output_dir_button_has_expected_object_name(app):
    widget = ExperimentsList(data_manager=_StubDM())
    assert widget.choose_output_dir_btn.objectName() == "experiments_output_dir_button"


def test_experiments_panel_has_no_collapse_chrome(app):
    widget = ExperimentsList()
    assert not hasattr(widget, "collapse_btn")
    assert not hasattr(widget, "toggle_collapsed")
    assert not hasattr(widget, "set_collapsed")
    assert not hasattr(widget, "is_collapsed")


def test_experiments_label_is_present_and_plain(app):
    widget = ExperimentsList()
    label = widget.findChild(QLabel, "experiments_panel_label")
    assert label is not None
    assert label.text() == "Experiments"


def test_table_and_actions_are_always_visible_regardless_of_row_count(app):
    widget = ExperimentsList()
    widget.show()  # isVisible() reflects ancestor visibility; must actually show
    # No collapse toggle exists, so the action row is always shown — only the
    # scrollable rows region itself hides/shows based on row count (existing
    # _update_table_visibility behavior, unchanged by this task).
    assert widget.add_btn.isVisible() is True
    assert widget.commit_btn.isVisible() is True
    widget.set_experiments(["/data/a"])
    assert widget.add_btn.isVisible() is True
