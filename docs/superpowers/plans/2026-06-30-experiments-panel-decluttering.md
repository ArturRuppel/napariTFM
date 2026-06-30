# Experiments Panel Decluttering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the fused "EXPERIMENTS" panel into a collapsible Setup container (calibration, input-file names, optional output dir) and a flat, always-visible experiment table; add a Discover-preview state; fix column-header defaults; move the toolbar into the title row as icon-only buttons.

**Architecture:** All changes are localized to four files: `_collapsible_section.py` gains a generic `set_expanded()`; `_experiments_list.py` is restructured (new `Setup` container, reordered output-dir, preview-row rendering, dropped collapse chrome, renamed column defaults); `_icons.py` gains four new SVG icon bodies; `_widget.py` relocates the toolbar into the title row as icon buttons. No new files, no persistence changes, no change to the G0/G1/G2 disclosure gate.

**Tech Stack:** Python, Qt via `qtpy` (PyQt5/PySide2 compatible), pytest with `QApplication` fixtures for widget tests.

**Spec:** `docs/superpowers/specs/2026-06-30-experiments-panel-decluttering-design.md`

---

## Task 1: `CollapsibleSection.set_expanded()`

A symmetric way to programmatically collapse a section is needed for the Setup container's auto-collapse. Today only `expand()` (force-open) exists.

**Files:**
- Modify: `napariTFM/widgets/_collapsible_section.py:178-179`
- Test: `tests/test_collapsible_section.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_collapsible_section.py` (it already has a fixture/pattern for constructing a `CollapsibleSection` with a stub inner widget — follow the existing `test_*` style around line 15):

```python
def test_set_expanded_can_both_open_and_close():
    sec = CollapsibleSection("Title", QWidget(), expanded=True)
    assert sec.is_expanded is True
    sec.set_expanded(False)
    assert sec.is_expanded is False
    sec.set_expanded(True)
    assert sec.is_expanded is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collapsible_section.py::test_set_expanded_can_both_open_and_close -v`
Expected: FAIL with `AttributeError: 'CollapsibleSection' object has no attribute 'set_expanded'`

- [ ] **Step 3: Implement `set_expanded`, rebase `expand()` on it**

In `napariTFM/widgets/_collapsible_section.py`, replace:

```python
    def expand(self) -> None:
        self._toggle.setChecked(True)
```

with:

```python
    def expand(self) -> None:
        self.set_expanded(True)

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collapsible_section.py -v`
Expected: PASS (all tests, including the new one and the existing `expand()`-based ones)

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_collapsible_section.py tests/test_collapsible_section.py
git commit -m "Add CollapsibleSection.set_expanded for programmatic collapse"
```

---

## Task 2: Column-header defaults: "Level N" → "Column N"

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py:1-10` (module docstring), `:68-84` (`nesting_columns`)
- Test: `tests/test_experiments_list.py:306-338, 587-590`

- [ ] **Step 1: Update the failing assertions first (TDD against existing tests)**

In `tests/test_experiments_list.py`, change the three places that assert `"Level 1"/"Level 2"`:

`test_commit_adds_discovered_with_nesting_columns` (around line 316-317):
```python
    assert records[0]["columns"] == {"Column 1": "Ctrl", "Column 2": "pos_00"}
    assert widget.column_names() == ["Column 1", "Column 2"]
```

`test_commit_columns_pad_to_max_nesting_depth` (around line 334-337):
```python
    assert widget.column_names() == ["Column 1", "Column 2"]
    by_leaf = {Path(r["path"]).name: r["columns"] for r in widget.experiment_records()}
    assert by_leaf["pos_00"] == {"Column 1": "Ctrl", "Column 2": "pos_00"}
    assert by_leaf["solo"] == {"Column 1": "solo", "Column 2": ""}
```

`test_column_header_has_one_editable_field_per_level` (line 587-590) — rename the test itself for accuracy:
```python
def test_column_header_has_one_editable_field_per_column(app):
    widget = ExperimentsList()
    widget.add_folders(["/data/a"], columns={"Column 1": "Ctrl", "Column 2": "pos_00"})
    assert [f.text() for f in widget._header_fields] == ["Column 1", "Column 2"]
```

- [ ] **Step 2: Run tests to verify they now fail against current code**

Run: `pytest tests/test_experiments_list.py -k "nesting_columns or commit_adds or commit_columns_pad or one_editable_field_per_column" -v`
Expected: FAIL (current code still emits `"Level 1"`/`"Level 2"`)

- [ ] **Step 3: Implement the rename**

In `napariTFM/widgets/_experiments_list.py`, replace the module docstring (lines 3-9):

```python
"""Experiments list (top-of-panel substrate): an editable column table of rows.

Each discovered experiment is one row. The columns are *derived from the folder
nesting* under the chosen discovery root: every nesting level becomes a column
and the folder name at that level is the row's value for it (root ``/data`` and
folder ``/data/Ctrl/pos_00`` → ``Column 1 = Ctrl``, ``Column 2 = pos_00``). The
column *names* are an editable, table-wide header; the *values* are read-only
(they are the folder names). Rows are multi-selectable (Ctrl/Shift-click) and
deletable.
"""
```

And `nesting_columns` (lines 68-84):

```python
def nesting_columns(folder: str | Path, root: str | Path) -> dict[str, str]:
    """Derive a row's columns from *folder*'s nesting under *root*.

    Every path component of *folder* relative to *root* becomes a column named
    ``Column 1``, ``Column 2`` … (left to right) whose value is that component's
    folder name. A folder that is not actually under *root* (or equals it) falls
    back to a single ``Column 1`` column holding the leaf folder name, so a row
    always carries at least one column.
    """
    folder = Path(folder)
    try:
        parts = folder.resolve().relative_to(Path(root).resolve()).parts
    except (ValueError, OSError):
        parts = ()
    if not parts:
        parts = (folder.name,)
    return {f"Column {i + 1}": part for i, part in enumerate(parts)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_experiments_list.py -v`
Expected: PASS (full file — this also confirms no other test still depended on the old "Level" wording)

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Rename default column headers from Level N to Column N"
```

---

## Task 3: Split into a `Setup` container (calibration + input files + output dir)

Build the new always-constructed, initially-expanded `Setup` `CollapsibleSection` that wraps calibration, input-file-name fields, and (after Task 4) the output-directory row, replacing today's `_build_project_strip()` + `_build_config_header()` pair mounted directly into `body_layout`.

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py`
- Test: `tests/test_experiments_list.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_experiments_list.py` (near the calibration/output-dir tests around line 662):

```python
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
```

Note: `_StubPM` and `_StubDM` are already defined later in the same test file (around line 631-659) — since Python evaluates test bodies at call time, not definition time, these new tests can reference them even though they're defined further down in the file. `_make_qualifying` is an existing helper used by the discover/commit tests already in this file — reuse it as-is.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_experiments_list.py -k "setup_section" -v`
Expected: FAIL with `AttributeError: 'ExperimentsList' object has no attribute 'setup_section'`

- [ ] **Step 3: Implement the Setup container**

In `napariTFM/widgets/_experiments_list.py`, add the import (near the top, with the other `napariTFM.widgets` imports):

```python
from napariTFM.widgets._collapsible_section import CollapsibleSection
```

Rename `_build_project_strip` (lines 445-496) into two smaller builders. Replace:

```python
    # -- project-level calibration + output (the aggregation layer) -------
    def _build_project_strip(self) -> QVBoxLayout:
        """Pixel/frame calibration + an output-directory picker, themed."""
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)

        self.calibration_controls: dict[str, QLineEdit] = {}
        cal = QHBoxLayout()
        cal.setContentsMargins(0, 0, 0, 0)
        cal.setSpacing(COMPACT_SPACING + 4)
        for name, label, min_val, max_val in _CALIBRATION_SPECS:
            field = QLineEdit()
            validator = QDoubleValidator(min_val, max_val, _INPUT_DECIMALS, field)
            validator.setNotation(QDoubleValidator.StandardNotation)
            field.setValidator(validator)
            field.setObjectName(f"workflow_parameter_{name}")
            field.setStyleSheet(mono_input_style())
            if self._parameter_manager is not None:
                field.setText(
                    _format_value(self._parameter_manager.get_ui_parameter(name))
                )
                field.editingFinished.connect(
                    lambda n=name, c=field: self._commit_parameter(n, c)
                )
            self.calibration_controls[name] = field

            caption = QLabel(label)
            caption.setStyleSheet(f"color: {TEXT_MID};")
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(1)
            cell.addWidget(caption)
            cell.addWidget(field)
            cal.addLayout(cell, 1)
        box.addLayout(cal)

        out = QHBoxLayout()
        out.setContentsMargins(0, 0, 0, 0)
        self.choose_output_dir_btn = QToolButton()
        self.choose_output_dir_btn.setObjectName("experiments_output_dir_button")
        self.choose_output_dir_btn.setToolTip("Choose output directory")
        self.choose_output_dir_btn.setIcon(
            stage_action_icon("files", muted_accent(stage_accent("project")))
        )
        self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)
        self.output_dir_label = QLabel("No output directory")
        self.output_dir_label.setObjectName("project_output_dir_label")
        self.output_dir_label.setStyleSheet(f"color: {TEXT_DIM};")
        out.addWidget(self.choose_output_dir_btn)
        out.addWidget(self.output_dir_label, 1)
        box.addLayout(out)
        return box
```

with (output-dir row body is a placeholder here — Task 4 replaces `_build_output_dir_row`'s contents; this step only relocates calibration and introduces the seam):

```python
    # -- setup: calibration + input-file names + optional output dir ------
    def _build_setup_section(self) -> CollapsibleSection:
        """The one-time-per-batch config: calibration, input names, output dir.

        Wrapped in a CollapsibleSection that starts expanded and auto-collapses
        the first time the experiment table goes from empty to non-empty (see
        ``set_experiments``/``set_records``) — these fields rarely change
        between batches, so hiding them declutters the common case while
        staying one click away.
        """
        inner = QWidget()
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)
        box.addLayout(self._build_calibration_row())
        box.addLayout(self._build_config_header())
        box.addLayout(self._build_output_dir_row())
        inner.setLayout(box)
        return CollapsibleSection("Setup", inner, expanded=True, title_color=TEXT_MID)

    def _build_calibration_row(self) -> QHBoxLayout:
        """Pixel size + frame interval, free-text fields with a soft validator."""
        self.calibration_controls: dict[str, QLineEdit] = {}
        cal = QHBoxLayout()
        cal.setContentsMargins(0, 0, 0, 0)
        cal.setSpacing(COMPACT_SPACING + 4)
        for name, label, min_val, max_val in _CALIBRATION_SPECS:
            field = QLineEdit()
            validator = QDoubleValidator(min_val, max_val, _INPUT_DECIMALS, field)
            validator.setNotation(QDoubleValidator.StandardNotation)
            field.setValidator(validator)
            field.setObjectName(f"workflow_parameter_{name}")
            field.setStyleSheet(mono_input_style())
            if self._parameter_manager is not None:
                field.setText(
                    _format_value(self._parameter_manager.get_ui_parameter(name))
                )
                field.editingFinished.connect(
                    lambda n=name, c=field: self._commit_parameter(n, c)
                )
            self.calibration_controls[name] = field

            caption = QLabel(label)
            caption.setStyleSheet(f"color: {TEXT_MID};")
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(1)
            cell.addWidget(caption)
            cell.addWidget(field)
            cal.addLayout(cell, 1)
        return cal

    def _build_output_dir_row(self) -> QHBoxLayout:
        """Optional output-directory override — last in Setup, after inputs."""
        out = QHBoxLayout()
        out.setContentsMargins(0, 0, 0, 0)
        self.choose_output_dir_btn = QToolButton()
        self.choose_output_dir_btn.setObjectName("experiments_output_dir_button")
        self.choose_output_dir_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.choose_output_dir_btn.setIcon(
            stage_action_icon("plus", muted_accent(stage_accent("project")))
        )
        self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)
        self.output_dir_label = QLabel("")
        self.output_dir_label.setObjectName("project_output_dir_label")
        self.output_dir_label.setStyleSheet(f"color: {TEXT_DIM};")
        self.clear_output_dir_btn = QToolButton()
        self.clear_output_dir_btn.setObjectName("experiments_clear_output_dir_button")
        self.clear_output_dir_btn.setText("×")
        self.clear_output_dir_btn.setToolTip("Remove custom output directory")
        self.clear_output_dir_btn.clicked.connect(self._clear_output_dir)
        out.addWidget(self.choose_output_dir_btn)
        out.addWidget(self.output_dir_label, 1)
        out.addWidget(self.clear_output_dir_btn)
        self._sync_output_dir()
        return out
```

(`_sync_output_dir`/`_choose_output_dir`/`_clear_output_dir` are filled in by Task 4 — for this task, leave the existing `_sync_output_dir`/`_choose_output_dir` methods as they are today, i.e. `_sync_output_dir` still sets `"No output directory"` text; `_clear_output_dir` doesn't exist yet, so temporarily stub it in this step as `def _clear_output_dir(self) -> None: pass` to keep the file important — Task 4 replaces the stub.)

Now wire it into `__init__`. Replace:

```python
        # Project-level calibration + output directory (the aggregation layer
        # owns these now; the old Project section is gone).
        body_layout.addLayout(self._build_project_strip())

        # Staging for the two-step Discover→Commit flow (D2). The root is kept so
        # committed rows can derive their columns from the nesting under it.
        self._discovered: list[str] = []
        self._discover_root: Optional[str] = None

        body_layout.addLayout(self._build_config_header())
```

with:

```python
        # Setup: calibration, input-file names, optional output dir — one
        # collapsible block, auto-collapsing after the first commit.
        self.setup_section = self._build_setup_section()
        body_layout.addWidget(self.setup_section)

        # Staging for the two-step Discover→Commit flow (D2). The root is kept so
        # committed rows can derive their columns from the nesting under it.
        self._discovered: list[str] = []
        self._discover_root: Optional[str] = None
```

Finally, wire the auto-collapse. In `set_experiments` (around line 716-735), replace:

```python
        if was_empty and self._paths:
            self.set_active(self._paths[0])
```

with:

```python
        if was_empty and self._paths:
            self.set_active(self._paths[0])
            self.setup_section.set_expanded(False)
```

And in `set_records` (around line 776-808), add right after the existing `self._update_delete_btn()` / before `self.experiments_changed.emit()` — actually simplest is right after rebuilding, near the end of the method. Replace the tail of `set_records`:

```python
        self._rebuild_table()
        if self._active not in self._paths:
            self._active = None
        self.refresh_statuses()
        self._update_meta()
        self._update_delete_btn()
        self.experiments_changed.emit()
```

with:

```python
        self._rebuild_table()
        if self._active not in self._paths:
            self._active = None
        self.refresh_statuses()
        self._update_meta()
        self._update_delete_btn()
        if self._paths:
            self.setup_section.set_expanded(False)
        self.experiments_changed.emit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_experiments_list.py -v`
Expected: PASS for the new `setup_section` tests. The pre-existing output-dir tests (`test_experiments_list_tracks_output_directory` et al.) will still pass at this point since `_sync_output_dir`/`_choose_output_dir` are untouched in this task — Task 4 changes their behavior and updates those tests.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Split calibration/input-files/output-dir into a collapsible Setup section"
```

---

## Task 4: Output directory as an explicit "+ Add custom output directory" opt-in

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py` (`_sync_output_dir`, `_clear_output_dir`, `__init__`'s data-manager wiring)
- Test: `tests/test_experiments_list.py:685-705`

- [ ] **Step 1: Update the failing tests**

Replace the three existing output-dir tests in `tests/test_experiments_list.py` (lines 685-705):

```python
def test_output_dir_starts_as_unset_add_affordance(app):
    widget = ExperimentsList(data_manager=_StubDM())
    assert widget.output_dir_label.isVisible() is False
    assert widget.choose_output_dir_btn.text() == "Add custom output directory"
    assert widget.clear_output_dir_btn.isVisible() is False


def test_output_dir_shows_path_and_clear_button_once_set(app, tmp_path):
    dm = _StubDM()
    widget = ExperimentsList(data_manager=dm)
    dm.set_output_dir(tmp_path)
    assert widget.output_dir_label.isVisible() is True
    assert widget.output_dir_label.text() == str(tmp_path)
    assert widget.clear_output_dir_btn.isVisible() is True
    assert widget.choose_output_dir_btn.text() == "Change output directory"


def test_clear_output_dir_resets_manager_and_label(app, tmp_path):
    dm = _StubDM()
    widget = ExperimentsList(data_manager=dm)
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
```

`_StubDM.set_output_dir` already accepts any path and calls registered callbacks (see its definition at line ~648-659); it needs to also accept `None` to support the clear test — check its current body:

```python
    def set_output_dir(self, path):
        self.output_dir = Path(path)
        for cb in self._cbs:
            cb()
```

Update it to:

```python
    def set_output_dir(self, path):
        self.output_dir = Path(path) if path is not None else None
        for cb in self._cbs:
            cb()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_experiments_list.py -k "output_dir" -v`
Expected: FAIL — `output_dir_label.text()` is currently `"No output directory"` and always visible; `clear_output_dir_btn` doesn't exist as a working button yet (Task 3 only stubbed `_clear_output_dir` as a no-op).

- [ ] **Step 3: Implement**

In `napariTFM/widgets/_experiments_list.py`, replace the stub from Task 3 and the existing `_sync_output_dir`:

```python
    def _clear_output_dir(self) -> None:
        """Reset to the default per-experiment output location (unset override)."""
        if self._data_manager is None:
            return
        self._data_manager.set_output_dir(None)
        self.output_dir_changed.emit()

    def _sync_output_dir(self) -> None:
        path = getattr(self._data_manager, "output_dir", None)
        if path is None:
            self.output_dir_label.setText("")
            self.output_dir_label.setVisible(False)
            self.output_dir_label.setToolTip("")
            self.choose_output_dir_btn.setText("Add custom output directory")
            self.choose_output_dir_btn.setToolTip(
                "Optional — overrides the default per-experiment output location"
            )
            self.clear_output_dir_btn.setVisible(False)
            return
        text = str(path)
        self.output_dir_label.setText(text)
        self.output_dir_label.setVisible(True)
        self.output_dir_label.setToolTip(text)
        self.choose_output_dir_btn.setText("Change output directory")
        self.choose_output_dir_btn.setToolTip("Choose a different output directory")
        self.clear_output_dir_btn.setVisible(True)
```

`_build_output_dir_row` (from Task 3) already ends with `self._sync_output_dir()`, so the unset state is applied at construction time regardless of whether a `data_manager` was passed (`getattr(None-ish manager, "output_dir", None)` path — actually `self._data_manager` may be `None` itself; `getattr(None, "output_dir", None)` safely returns `None`, so this works whether `data_manager=None` or a real manager with `output_dir=None`).

Now simplify the `__init__` data-manager wiring. Replace:

```python
        if self._data_manager is not None:
            self._data_manager.add_change_callback(self._sync_output_dir)
            self._sync_output_dir()
```

with:

```python
        if self._data_manager is not None:
            self._data_manager.add_change_callback(self._sync_output_dir)
```

(The initial sync already happened once inside `_build_output_dir_row` during construction; this block now only needs to register the live-update callback.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_experiments_list.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Make output directory an explicit optional opt-in with a clear action"
```

---

## Task 5: Drop the EXPERIMENTS collapsible chrome (flat panel)

Remove the hand-rolled collapse header/body for `ExperimentsList` itself — now that Setup is its own collapsible piece, the remaining action-row + table doesn't need a second, redundant collapse affordance.

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py`
- Test: `tests/test_experiments_list.py:708-743`

- [ ] **Step 1: Replace the failing tests**

Delete these four tests from `tests/test_experiments_list.py` (lines 708-743 — `test_experiments_list_starts_expanded`, `test_toggle_collapsed_folds_body_and_shows_summary`, `test_collapsed_summary_tracks_experiment_count`, `test_collapse_button_has_expected_object_name`) and replace with:

```python
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
    # No collapse toggle exists, so the action row is always shown — only the
    # scrollable rows region itself hides/shows based on row count (existing
    # _update_table_visibility behavior, unchanged by this task).
    assert widget.add_btn.isVisible() is True
    assert widget.commit_btn.isVisible() is True
    widget.set_experiments(["/data/a"])
    assert widget.add_btn.isVisible() is True
```

Add `QLabel` to the test file's qtpy import line if not already imported (check the top of `tests/test_experiments_list.py` — `QApplication` is imported from `qtpy.QtWidgets`; add `QLabel` alongside it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_experiments_list.py -k "no_collapse_chrome or label_is_present or always_visible" -v`
Expected: FAIL — `collapse_btn` still exists today.

- [ ] **Step 3: Implement the flattening**

In `napariTFM/widgets/_experiments_list.py`, replace the header + body-wrapper construction (lines 315-348):

```python
        # Header: a collapse caret, the section label, and a compact summary
        # that only earns its place while the list is folded away.
        self._collapsed = False
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.collapse_btn = QToolButton()
        self.collapse_btn.setObjectName("experiments_collapse_button")
        self.collapse_btn.setArrowType(Qt.DownArrow)
        self.collapse_btn.setAutoRaise(True)
        self.collapse_btn.setToolTip("Collapse the experiments list")
        self.collapse_btn.clicked.connect(self.toggle_collapsed)
        header.addWidget(self.collapse_btn)
        label = QLabel("EXPERIMENTS")
        label.setStyleSheet(f"color: {TEXT_MID}; font-weight: bold;")
        header.addWidget(label)
        self._header_summary = QLabel("")
        self._header_summary.setObjectName("experiments_header_summary")
        self._header_summary.setStyleSheet(f"color: {TEXT_DIM};")
        self._header_summary.setVisible(False)
        header.addSpacing(COMPACT_SPACING)
        header.addWidget(self._header_summary)
        header.addStretch()

        layout.addLayout(header)

        # Everything below the header lives in one collapsible body, so folding
        # the list is a single setVisible on the container.
        self._body = QWidget()
        self._body.setObjectName("experiments_body")
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(COMPACT_SPACING)
        self._body.setLayout(body_layout)
        layout.addWidget(self._body)
```

with:

```python
        # A plain, non-interactive label — the table below is always visible,
        # so there's nothing to fold away (unlike the Setup section above it).
        label = QLabel("Experiments")
        label.setObjectName("experiments_panel_label")
        label.setStyleSheet(f"color: {TEXT_MID}; font-weight: bold;")
        layout.addWidget(label)

        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(COMPACT_SPACING)
        layout.addLayout(body_layout)
```

Every other reference to `body_layout.addWidget(...)` / `body_layout.addLayout(...)` later in `__init__` is unchanged — `body_layout` still exists as a local variable, it's just a plain `QVBoxLayout` added directly to `layout` instead of living inside a hideable `self._body` `QWidget`.

Remove the now-dead collapse API at the bottom of the class (the `-- collapse / expand --` block, roughly lines 1011-1037):

```python
    # -- collapse / expand ----------------------------------------------
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Fold the list down to its header row (or restore it).

        Collapsing hides the whole body — calibration, input-file config, the
        rows table, the action bar and the count — leaving only the header,
        which then shows a compact experiment-count summary so the single
        remaining row still says how much is hidden.
        """
        self._collapsed = bool(collapsed)
        self._body.setVisible(not self._collapsed)
        self._header_summary.setVisible(self._collapsed)
        self.collapse_btn.setArrowType(
            Qt.RightArrow if self._collapsed else Qt.DownArrow
        )
        self.collapse_btn.setToolTip(
            "Expand the experiments list"
            if self._collapsed
            else "Collapse the experiments list"
        )

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)
```

Delete this block entirely (no replacement — the methods no longer exist, matching the new test asserting `not hasattr`).

Also remove the now-stale reference in `_update_meta` (around line 1004-1009):

```python
    def _update_meta(self) -> None:
        n = len(self._paths)
        self._meta.setText(f"{n} experiment{'s' if n != 1 else ''}")
        self.run_all_btn.setEnabled(n > 0)
        # Keep the folded-away summary current even while collapsed.
        self._header_summary.setText(self._meta.text())
```

becomes:

```python
    def _update_meta(self) -> None:
        n = len(self._paths)
        self._meta.setText(f"{n} experiment{'s' if n != 1 else ''}")
        self.run_all_btn.setEnabled(n > 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_experiments_list.py -v`
Expected: PASS for the full file.

- [ ] **Step 5: Check for other callers of the removed API**

Run: `grep -rn "collapse_btn\|toggle_collapsed\|set_collapsed\|is_collapsed\|_header_summary\|experiments_collapse_button" napariTFM/ tests/`
Expected: no remaining references outside `_experiments_list.py`'s own (now-removed) definitions and the tests just rewritten. If `_widget.py` or another test references these, update/remove that reference too before committing.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Drop the EXPERIMENTS collapsible chrome; render as a flat panel"
```

---

## Task 6: Discover → preview → harden

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py` (`ExperimentRow`, `ExperimentsList`)
- Test: `tests/test_experiments_list.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_experiments_list.py`, near the existing discover/commit tests (after `test_commit_button_enables_only_after_discovery`, around line 348):

```python
def test_discover_renders_preview_rows_in_table(app, tmp_path):
    _make_qualifying(tmp_path, "a", "b")
    widget = ExperimentsList()
    widget.discover(tmp_path)
    assert len(widget._preview_rows) == 2
    assert all(row.is_preview for row in widget._preview_rows)
    # Preview rows are not committed rows.
    assert widget.experiments() == []


def test_discover_again_replaces_rather_than_merges_preview(app, tmp_path):
    _make_qualifying(tmp_path, "a")
    other_root = tmp_path / "other"
    other_root.mkdir()
    _make_qualifying(other_root, "z")
    widget = ExperimentsList()
    widget.discover(tmp_path)
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
    widget.commit_discovered()
    assert widget._preview_rows == []
    assert len(widget._rows) == 1
    assert widget._rows[0].is_preview is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_experiments_list.py -k "preview" -v`
Expected: FAIL — `ExperimentRow` has no `is_preview`, `ExperimentsList` has no `_preview_rows`/`_discovered_selected`.

- [ ] **Step 3: Implement preview-row support on `ExperimentRow`**

In `napariTFM/widgets/_experiments_list.py`, replace `ExperimentRow.__init__`'s signature and tail (lines 191-229):

```python
    def __init__(self, path: str, values: Optional[list[str]] = None, parent=None):
        super().__init__(parent)
        self._path = path
        self._selected = False
        # The row paints its own (styled) background — selected rows lift.
        self.setObjectName("experiment_row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
```

with:

```python
    def __init__(
        self,
        path: str,
        values: Optional[list[str]] = None,
        parent=None,
        *,
        preview: bool = False,
    ):
        super().__init__(parent)
        self._path = path
        self._selected = False
        self._preview = preview
        # The row paints its own (styled) background — selected rows lift.
        self.setObjectName("experiment_row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
```

and (still in `__init__`, after the chip is built — lines 222-229):

```python
        self._chip = QLabel("queued")
        self._chip.setFixedWidth(_CHIP_W)
        self._chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._chip.setStyleSheet(f"color: {experiment_status_color('queued')};")
        layout.addWidget(self._chip)

        # Apply the deselected resting style (row + name colors).
        self.set_selected(False)
```

becomes:

```python
        self._chip = QLabel("queued")
        self._chip.setFixedWidth(_CHIP_W)
        self._chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._chip.setStyleSheet(f"color: {experiment_status_color('queued')};")
        layout.addWidget(self._chip)

        if self._preview:
            # Nothing has run for a not-yet-committed folder — no status to show.
            self.mini_rail.setVisible(False)
            self._chip.setVisible(False)
            for value_label in self._value_labels:
                value_label.setStyleSheet(f"color: {TEXT_DIM}; font-style: italic;")

        # Apply the deselected resting style (row + name colors).
        self.set_selected(False)

    @property
    def is_preview(self) -> bool:
        return self._preview
```

And `set_selected` (lines 242-251) — preview rows keep their dim/italic text regardless of selection (only the select-bar/background should respond), so guard the text-color branch:

```python
    def set_selected(self, on: bool) -> None:
        self._selected = on
        accent = stage_accent("displacement")
        self._selbar.setStyleSheet(
            f"background: {accent};" if on else "background: transparent;"
        )
        self.setStyleSheet(experiment_row_style(on, accent))
        if not self._preview:
            color = experiment_name_color(on)
            for label in self._value_labels:
                label.setStyleSheet(f"color: {color};")
```

- [ ] **Step 4: Implement preview-row state and rendering on `ExperimentsList`**

Initialize the new state alongside the existing discover-staging state (the block added in Task 3, right after `self.setup_section = self._build_setup_section()`):

```python
        # Staging for the two-step Discover→Commit flow (D2). The root is kept so
        # committed rows can derive their columns from the nesting under it.
        self._discovered: list[str] = []
        self._discover_root: Optional[str] = None
        # Preview-row selection (separate from the committed-row selection
        # machinery — preview rows have no "active"/tuning concept).
        self._discovered_selected: set[str] = set()
```

Initialize `self._preview_rows` alongside `self._rows` (line 303):

```python
        self._rows: list[ExperimentRow] = []
        self._preview_rows: list[ExperimentRow] = []
```

Update `discover()` (lines 638-651) to reset preview-selection and re-render:

```python
    def discover(self, root: str | Path) -> list[str]:
        """Step 1: stage the folders under *root* that hold the required inputs.

        Folder-presence only — required inputs are beads + reference (cells is
        optional and excluded from the requirement). The root is remembered so
        committed rows can derive their columns from the nesting under it.
        Staging never mutates the committed list; the second Commit step does.
        The staged set renders immediately as dimmed preview rows in the table;
        a second call to ``discover`` *replaces* the current preview set rather
        than merging into it.
        """
        cfg = self.input_file_config()
        required = [cfg.get("beads"), cfg.get("reference")]
        self._discover_root = str(root)
        self._discovered = discover_experiment_folders(root, required)
        self._discovered_selected = set()
        self._update_staging()
        self._rebuild_table()
        return list(self._discovered)
```

Update `commit_discovered()` (lines 656-664) — clear staging *before* calling `_add_records` so the rebuild it triggers doesn't render the just-committed paths twice (once as a real row, once as a stale preview row):

```python
    def commit_discovered(self) -> None:
        """Step 2: add the staged folders with columns from the folder nesting."""
        if not self._discovered:
            return
        root = self._discover_root
        pairs = [(path, nesting_columns(path, root)) for path in self._discovered]
        self._discovered = []
        self._discovered_selected = set()
        self._add_records(pairs, self.input_file_config())
        self._update_staging()
```

Add a preview click handler, near `_on_row_clicked` (after line 924):

```python
    def _on_preview_row_clicked(self, path: str, _flag: int) -> None:
        """Toggle one not-yet-committed row in/out of the delete selection."""
        if path in self._discovered_selected:
            self._discovered_selected.discard(path)
        else:
            self._discovered_selected.add(path)
        for row in self._preview_rows:
            row.set_selected(row.path in self._discovered_selected)
        self._update_delete_btn()
```

Update `_update_delete_btn` (lines 930-932) to also consider preview selection:

```python
    def _update_delete_btn(self) -> None:
        if hasattr(self, "delete_btn"):
            self.delete_btn.setEnabled(
                bool(self._selected_paths) or bool(self._discovered_selected)
            )
```

Update `delete_selected` (lines 828-836) to remove preview rows first, when any are selected:

```python
    def delete_selected(self) -> None:
        """Remove selected rows: preview-staged first, else committed rows."""
        if self._discovered_selected:
            self._discovered = [
                p for p in self._discovered if p not in self._discovered_selected
            ]
            self._discovered_selected = set()
            self._update_staging()
            self._rebuild_table()
            self._update_delete_btn()
            return
        if not self._selected_paths:
            return
        remaining = [p for p in self._paths if p not in self._selected_paths]
        if self._active in self._selected_paths:
            self._active = None
        self._selected_paths = set()
        self.set_experiments(remaining)
```

Update `_rebuild_table` (lines 979-996) to also build preview rows after the committed ones:

```python
    def _rebuild_table(self) -> None:
        while self._rows_box.count():
            item = self._rows_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = []
        self._preview_rows = []
        self._rows_box.addWidget(self._build_header_widget())
        for path in self._paths:
            values = [
                self._records[path]["columns"].get(name, "")
                for name in self._column_names
            ]
            row = ExperimentRow(path, values or None)
            row.clicked.connect(self._on_row_clicked)
            row.set_selected(path in self._selected_paths)
            self._rows_box.addWidget(row)
            self._rows.append(row)
        for path in self._discovered:
            row = ExperimentRow(path, preview=True)
            row.clicked.connect(self._on_preview_row_clicked)
            row.set_selected(path in self._discovered_selected)
            self._rows_box.addWidget(row)
            self._preview_rows.append(row)
        self._update_table_visibility()
```

And `_update_table_visibility` (lines 998-1002) to also show the scroll region while a preview set exists with no committed rows yet:

```python
    def _update_table_visibility(self) -> None:
        """Collapse the empty table so the action bar sits flush under the
        input-file form. The bounded scroll region (and its "Folder" header
        placeholder) only earns its 300px once there are committed or
        preview rows to show."""
        self._rows_scroll.setVisible(bool(self._paths) or bool(self._discovered))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_experiments_list.py -v`
Expected: PASS for the full file.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Render Discover results as removable preview rows before commit"
```

---

## Task 7: New toolbar icons (`new`, `load`, `save`, `reset`)

**Files:**
- Modify: `napariTFM/widgets/_icons.py:14-48`
- Test: `tests/test_icons.py`

- [ ] **Step 1: Update the failing test**

In `tests/test_icons.py`, update `test_icon_names_cover_the_header_action_set` (around line 27-30):

```python
def test_icon_names_cover_the_header_action_set():
    assert set(ICON_NAMES) == {
        "files", "params", "preview", "run", "cancel", "power", "plus",
        "gcv", "new", "load", "save", "reset",
    }
```

Add a render-sanity check alongside the existing `test_stage_specific_action_icons_render_opaque` (around line 34-38) for the new names:

```python
def test_toolbar_icons_render_opaque(app):
    for name in ("new", "load", "save", "reset"):
        assert name in ICON_NAMES
        pixmap = stage_action_pixmap(name, "#2a788e", size=18)
        assert _opaque_count(pixmap) > 0, f"{name} rendered blank"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_icons.py -v`
Expected: FAIL — `test_icon_names_cover_the_header_action_set` fails (current set is missing the 4 new names); `test_toolbar_icons_render_opaque` fails with `KeyError` from `_ICON_BODIES[name]`.

- [ ] **Step 3: Add the icon bodies**

In `napariTFM/widgets/_icons.py`, add four entries to `_ICON_BODIES` (after the existing `"gcv"` entry, before the closing `}` at line 48):

```python
    # document with a plus corner — "start a new project"
    "new": (
        '<path d="M7 3 H14 L18 7 V21 H7 Z"/>'
        '<path d="M14 3 V7 H18"/>'
        '<line x1="9.5" y1="14" x2="15.5" y2="14"/>'
        '<line x1="12.5" y1="11" x2="12.5" y2="17"/>'
    ),
    # open folder — "load a project or a preset"
    "load": '<path d="M3 6.5 H9 L11 8.5 H21 V18.5 H3 Z"/>',
    # tray with a downward arrow — "save a project or a preset"
    "save": (
        '<path d="M4 4 H20 V14 H4 Z"/>'
        '<path d="M9 8 L12 11 L15 8"/>'
        '<line x1="12" y1="4.5" x2="12" y2="11"/>'
        '<path d="M4 14 V20 H20 V14"/>'
    ),
    # circular arrow — "reset parameters to defaults"
    "reset": (
        '<path d="M5 12 A7 7 0 1 0 7.5 6.5"/>'
        '<path d="M5 6 L5 12 L11 12"/>'
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_icons.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_icons.py tests/test_icons.py
git commit -m "Add new/load/save/reset toolbar icon bodies"
```

---

## Task 8: Move the toolbar into the title row as icon-only buttons

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Test: `tests/test_workflow_shell.py:1798-1813`

- [ ] **Step 1: Update the failing test**

Replace `test_toolbar_exposes_project_and_parameter_buttons` in `tests/test_workflow_shell.py` (lines 1798-1813):

```python
def test_toolbar_exposes_project_and_parameter_buttons(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    # Project front-door buttons live on the brand row, icon-only now.
    assert widget.new_project_btn.toolTip() == "Start a new project"
    assert widget.load_project_btn.toolTip() == "Load a project"
    assert widget.save_project_btn.toolTip() == "Save project as…"
    # Parameter preset buttons, same row, grouped after a divider.
    assert widget.load_params_btn.toolTip() == "Load parameters preset"
    assert widget.save_params_btn.toolTip() == "Save parameters preset"
    assert widget.reset_params_btn.toolTip() == "Reset parameters"
    for button in (
        widget.new_project_btn, widget.load_project_btn, widget.save_project_btn,
        widget.load_params_btn, widget.save_params_btn, widget.reset_params_btn,
    ):
        assert not button.icon().isNull()
        assert button.text() == ""
    # The experiments list no longer owns its own series Open/Save.
    assert not hasattr(widget.experiments_list, "load_series_btn")
    assert not hasattr(widget.experiments_list, "save_series_btn")
    assert not hasattr(widget, "_save_config")
    assert not hasattr(widget, "_load_config")


def test_toolbar_buttons_share_the_title_row(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    assert not hasattr(widget, "toolbar_grid")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workflow_shell.py -k "toolbar" -v`
Expected: FAIL — buttons currently have `.text()` set ("New Project" etc.) and empty tooltips don't match; `.icon()` is null (no icon was ever set on these text buttons today).

- [ ] **Step 3: Implement**

In `napariTFM/widgets/_widget.py`, replace the title-row + toolbar block (lines 517-546):

```python
        # Brand row + the Project/Parameters toolbar (the front door), laid out
        # as a 3x2 grid: New / Load / Save Project on top, Load / Save Params /
        # Reset below. Save Project is always Save-as; the lower row is presets.
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("napariTFM")
        title.setStyleSheet(title_style())
        title_row.addWidget(title)
        title_row.addStretch()
        container_layout.addLayout(title_row)

        self.new_project_btn = self._make_toolbar_button("New Project", "Start a new project")
        self.load_project_btn = self._make_toolbar_button("Load Project", "Load a project")
        self.save_project_btn = self._make_toolbar_button("Save Project", "Save project as…")
        self.load_params_btn = self._make_toolbar_button("Load Params", "Load parameters preset")
        self.save_params_btn = self._make_toolbar_button("Save Params", "Save parameters preset")
        self.reset_params_btn = self._make_toolbar_button("Reset", "Reset parameters")

        toolbar_grid = QGridLayout()
        toolbar_grid.setContentsMargins(0, 0, 0, 0)
        grid_buttons = (
            self.new_project_btn, self.load_project_btn, self.save_project_btn,
            self.load_params_btn, self.save_params_btn, self.reset_params_btn,
        )
        for _idx, _btn in enumerate(grid_buttons):
            toolbar_grid.addWidget(_btn, _idx // 3, _idx % 3)
        # Park all surplus width in a trailing empty column so the buttons stay
        # compact and left-aligned instead of spreading out when the panel grows.
        toolbar_grid.setColumnStretch(3, 1)
        container_layout.addLayout(toolbar_grid)
```

with:

```python
        # Brand row + the Project/Parameters toolbar (the front door), all in
        # one row now: icon-only buttons, right-aligned opposite the title,
        # grouped Project | Params | Reset with thin dividers between groups.
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("napariTFM")
        title.setStyleSheet(title_style())
        title_row.addWidget(title)
        title_row.addStretch()

        self.new_project_btn = self._make_toolbar_button("new", "Start a new project")
        self.load_project_btn = self._make_toolbar_button("load", "Load a project")
        self.save_project_btn = self._make_toolbar_button("save", "Save project as…")
        self.load_params_btn = self._make_toolbar_button("load", "Load parameters preset")
        self.save_params_btn = self._make_toolbar_button("save", "Save parameters preset")
        self.reset_params_btn = self._make_toolbar_button("reset", "Reset parameters")

        for button in (self.new_project_btn, self.load_project_btn, self.save_project_btn):
            title_row.addWidget(button)
        title_row.addWidget(self._toolbar_divider())
        for button in (self.load_params_btn, self.save_params_btn):
            title_row.addWidget(button)
        title_row.addWidget(self._toolbar_divider())
        title_row.addWidget(self.reset_params_btn)

        container_layout.addLayout(title_row)
```

Replace `_make_toolbar_button` (lines 770-776):

```python
    def _make_toolbar_button(self, text: str, tooltip: str) -> QToolButton:
        """A compact auto-raised text button for the title-bar config toolbar."""
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button
```

with:

```python
    def _make_toolbar_button(self, icon_name: str, tooltip: str) -> QToolButton:
        """A compact, icon-only, auto-raised button for the title-row toolbar."""
        button = QToolButton()
        button.setIcon(stage_action_icon(icon_name, muted_accent(stage_accent("project"))))
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    @staticmethod
    def _toolbar_divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        return divider
```

Update the imports at the top of `napariTFM/widgets/_widget.py`. Today (lines 7-11):

```python
from qtpy.QtCore import Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QSizePolicy, QDoubleSpinBox,
    QHBoxLayout, QGridLayout, QSpinBox, QComboBox, QFileDialog, QCheckBox,
    QMenu, QToolButton, QApplication
)
```

`QGridLayout` is only used at the `toolbar_grid = QGridLayout()` line just removed above — drop it. `QFrame` is needed for `_toolbar_divider` and isn't imported yet — add it:

```python
from qtpy.QtCore import Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QSizePolicy, QDoubleSpinBox,
    QHBoxLayout, QFrame, QSpinBox, QComboBox, QFileDialog, QCheckBox,
    QMenu, QToolButton, QApplication
)
```

`stage_accent` is already imported (line 24, from `napariTFM.widgets._ui_style`); `stage_action_icon` and `muted_accent` are not. Update line 24 from:

```python
from napariTFM.widgets._ui_style import title_style, stage_accent, theme_names, active_theme_name, set_active_theme, section_grid, add_section_header, add_section_pair_row, add_section_labeled_full_row, section_label_style, section_subheader_style, TIGHT_SPACING
```

to:

```python
from napariTFM.widgets._ui_style import title_style, stage_accent, muted_accent, theme_names, active_theme_name, set_active_theme, section_grid, add_section_header, add_section_pair_row, add_section_labeled_full_row, section_label_style, section_subheader_style, TIGHT_SPACING
```

and add a new import line for `stage_action_icon` (e.g. directly below the `_ui_style` import line):

```python
from napariTFM.widgets._icons import stage_action_icon
```

Update the stale comment above `self.setMinimumWidth(400)` (line 500-502), which references the old 3-column grid:

```python
        # Give the dock a comfortable default/minimum width so the toolbar's
        # three columns fit without truncating "Save Project" → "Save".
        self.setMinimumWidth(400)
```

becomes:

```python
        # Give the dock a comfortable default/minimum width for the panel body.
        self.setMinimumWidth(400)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workflow_shell.py -v`
Expected: PASS for the full file (this also exercises `_new_project`/`_load_project`/`_save_project`/param handlers, confirming the `.clicked.connect(...)` wiring lower in `__init__` — unchanged by this task — still works against the relocated buttons).

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Move toolbar into the title row as icon-only grouped buttons"
```

---

## Task 9: Full-suite regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: All tests pass. Pay particular attention to any other test file that might reference `_build_project_strip`, `EXPERIMENTS` label text, `"Level 1"/"Level 2"`, or `.text()` on the toolbar buttons — `grep -rn "_build_project_strip\|\"Level 1\"\|'Level 1'\|EXPERIMENTS" tests/ napariTFM/` first to catch anything this plan's tasks didn't already enumerate.

- [ ] **Step 2: Fix any fallout**

If a test outside the files already touched by Tasks 1-8 fails, it's exercising one of the renamed/removed APIs (`_build_project_strip`, `collapse_btn`, the old output-dir label text, `"Level N"` columns, or toolbar `.text()`). Update it to match the new behavior established in the relevant task above — don't change production code further unless the failure reveals a genuine bug in Tasks 1-8's implementation.

- [ ] **Step 3: Manual smoke check (optional but recommended given this is a UI-heavy change)**

Use the `/run` skill to launch the napari app with the napariTFM plugin loaded, open New Project, confirm: the toolbar row sits beside "napariTFM"; the Setup section is expanded with calibration/input-files/output-dir (output dir last, reading "Add custom output directory"); clicking Discover on a folder with qualifying subfolders shows dimmed preview rows; Add to list hardens them and collapses Setup; column headers default to "Column 1"/"Column 2".

- [ ] **Step 4: Final commit (only if Step 2 produced changes)**

```bash
git add -A
git commit -m "Fix fallout from experiments-panel decluttering across the suite"
```
