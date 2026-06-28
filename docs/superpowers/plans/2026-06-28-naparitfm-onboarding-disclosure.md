# napariTFM Onboarding Progressive-Disclosure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the all-at-once napariTFM panel with a three-state gated reveal (G0 no-project → G1 workspace → G2 tuning) driven by a Project front door that bundles everything, plus a scrollable experiment list, a dynamic Discover tooltip, and a dirty-flag guard.

**Architecture:** Disclosure is a pure function of two facts the shell already tracks — `_project_open` and `experiments_list.active()` — applied by a single `_update_disclosure()` method that toggles widget-group visibility on existing signals. A Project file is the unified source of truth: it reuses the tested `build_series_config`/`series_records` helpers plus a `parameters` block. The legacy autosave-to-output-dir path is retired in favour of explicit Save Project. The experiment list and Discover tooltip changes are local to `ExperimentsList`.

**Tech Stack:** Python, Qt (qtpy/PySide), pytest, PyYAML. All UI tests run headless via the existing stub harness in `tests/test_workflow_shell.py`.

**Spec:** `docs/superpowers/specs/2026-06-28-naparitfm-onboarding-disclosure-design.md`

**One deliberate refinement vs the spec:** the spec lists the global status line under G2. Because **Run all** (a G1 action) reports its per-folder progress into `status_label`, this plan keeps `status_label` visible whenever a project is open (G1 **and** G2); only the pipeline context label and the four stage pills are strictly G2. This preserves batch feedback in G1.

**Files touched across the plan:**
- Modify: `napariTFM/widgets/_widget.py` — shell: disclosure, toolbar, project save/load/new, dirty flag, autosave removal.
- Modify: `napariTFM/widgets/_experiments_list.py` — remove series buttons, scrollable list, Discover tooltip.
- Modify: `tests/test_workflow_shell.py` — new tests + updates to tests that assumed everything is always visible.

**How to run the suite (used in every task):**
```bash
python -m pytest tests/test_workflow_shell.py -q
```

---

### Task 1: Disclosure state machine (G0 / G1 / G2)

Introduce `_project_open` and `_update_disclosure()`, gate the workspace + pipeline + stage pills, and add a G0 empty-state hint. Drive state directly in tests (no Project handlers yet — those arrive in Task 3).

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (`napariTFMWidget.__init__`, new `_update_disclosure`, `_on_active_experiment_changed`)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing tests**

Add this module-level helper near the other helpers (after `_stub_main_widget`, ~line 433) — it drives a stub widget into G2:

```python
def _enter_tuning(widget, path="/data/exp"):
    """Drive a stub shell into G2: project open + a selected experiment."""
    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_experiments([path])
    widget.experiments_list.set_active(path)
    return widget
```

Add these tests at the end of `tests/test_workflow_shell.py`:

```python
def test_g0_hides_workspace_and_pipeline_until_project_open(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    # Default state is G0: no project open.
    assert widget._project_open is False
    assert not widget.experiments_list.isVisibleTo(widget)
    assert not widget._pipeline_context_label.isVisibleTo(widget)
    for section in widget._stage_sections:
        assert not section.isVisibleTo(widget)
    assert widget._empty_hint.isVisibleTo(widget)


def test_g1_reveals_workspace_but_not_stage_pills(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._project_open = True
    widget._update_disclosure()
    # G1: workspace + status visible; pipeline label + pills still hidden.
    assert widget.experiments_list.isVisibleTo(widget)
    assert widget.status_label.isVisibleTo(widget)
    assert not widget._empty_hint.isVisibleTo(widget)
    assert not widget._pipeline_context_label.isVisibleTo(widget)
    for section in widget._stage_sections:
        assert not section.isVisibleTo(widget)


def test_g2_reveals_stage_pills_when_experiment_selected(monkeypatch, app):
    widget = _enter_tuning(_stub_main_widget(monkeypatch))
    # G2: a row is selected — pipeline label + every stage pill revealed.
    assert widget._pipeline_context_label.isVisibleTo(widget)
    for section in widget._stage_sections:
        assert section.isVisibleTo(widget)


def test_deselecting_experiment_drops_back_to_g1(monkeypatch, app):
    widget = _enter_tuning(_stub_main_widget(monkeypatch))
    widget.experiments_list.set_active(None)
    for section in widget._stage_sections:
        assert not section.isVisibleTo(widget)
    assert not widget._pipeline_context_label.isVisibleTo(widget)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_workflow_shell.py -k "g0_hides or g1_reveals or g2_reveals or deselecting_experiment" -q`
Expected: FAIL — `AttributeError: 'napariTFMWidget' object has no attribute '_project_open'` (and `_empty_hint`, `_update_disclosure`).

- [ ] **Step 3: Add the empty-state hint and the disclosure flag in `__init__`**

In `napariTFM/widgets/_widget.py`, inside `napariTFMWidget.__init__`, find the title row block that ends with `container_layout.addLayout(title_row)` (~line 409). Immediately after it, add the empty-state hint and the disclosure flag:

```python
        # G0 empty-state hint: shown only before a project is opened.
        self._empty_hint = QLabel("New Project to begin, or Load Project.")
        self._empty_hint.setObjectName("empty_state_hint")
        self._empty_hint.setStyleSheet(section_label_style())
        container_layout.addWidget(self._empty_hint)

        # Progressive-disclosure gate (G0/G1/G2). No project is open at launch.
        self._project_open = False
```

- [ ] **Step 4: Add the `_update_disclosure` method**

In `napariTFM/widgets/_widget.py`, add this method to `napariTFMWidget` (place it right after `refresh`, ~line 709):

```python
    def _update_disclosure(self) -> None:
        """Reveal only what the current state earns (G0/G1/G2 gated reveal).

        G0 (no project): only the header toolbars + the empty-state hint show.
        G1 (project open, no row selected): the experiments workspace + the
        shared status line. G2 (a row selected): additionally the pipeline
        context label and the four stage pills.
        """
        project_open = self._project_open
        tuning = project_open and self.experiments_list.active() is not None

        self._empty_hint.setVisible(not project_open)
        self.experiments_list.setVisible(project_open)
        self.status_label.setVisible(project_open)

        self._pipeline_context_label.setVisible(tuning)
        for section in self._stage_sections:
            section.setVisible(tuning)
```

- [ ] **Step 5: Call disclosure at the end of `__init__` and on active-experiment change**

In `__init__`, find the final setup lines (~617-619):

```python
        self.connect_signals()
        self.data_manager.add_change_callback(self.refresh)
        self.refresh_stage_statuses()
```

Replace with:

```python
        self.connect_signals()
        self.data_manager.add_change_callback(self.refresh)
        self.refresh_stage_statuses()
        self._update_disclosure()
```

Then in `_on_active_experiment_changed` (~833), add a disclosure refresh. The current body ends with `self._write_config()`. Leave `_write_config()` for now (Task 3 removes it) and add the disclosure call just before it:

```python
        self._update_disclosure()
        self._write_config()
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_workflow_shell.py -k "g0_hides or g1_reveals or g2_reveals or deselecting_experiment" -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Update the three existing visibility tests to enter the right state**

These three tests asserted that gated widgets are always visible; they must now drive into G2 first. Make the edits below.

In `test_main_widget_uses_stage_sections_instead_of_tabs` (~388), after `widget.show()` and before the `assert ...isVisible()` block, insert:

```python
    _enter_tuning(widget)
    app.processEvents()
```

In `test_main_widget_groups_parameters_inline_per_stage` (~986), after `widget.show()` / `app.processEvents()` near the top, insert:

```python
    _enter_tuning(widget)
    app.processEvents()
```

In `test_main_widget_embeds_always_visible_stage_file_status_rows` (~1030), after `widget.show()` / `app.processEvents()`, insert:

```python
    _enter_tuning(widget)
    app.processEvents()
```

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS (all tests green).

- [ ] **Step 9: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "feat: progressive-disclosure state machine (G0/G1/G2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Project + Parameters toolbar restructure

Rename the parameter buttons, add the three Project buttons on the brand row, move the parameter buttons to a second row, and remove the Open/Save series buttons from the experiments list header.

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (`__init__` title block, `_make_toolbar_button` reuse)
- Modify: `napariTFM/widgets/_experiments_list.py` (`__init__` header block; drop series buttons + signals)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing test**

Replace the body of `test_config_split_into_params_preset_and_series_handlers` (~1544) with a test of the new toolbar. Rename it and rewrite:

```python
def test_toolbar_exposes_project_and_parameter_buttons(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    # Project front-door buttons live on the brand row.
    assert widget.new_project_btn.text() == "New Project"
    assert widget.load_project_btn.text() == "Load Project"
    assert widget.save_project_btn.text() == "Save Project"
    # Parameter preset buttons, renamed and reordered (Load, Save, Reset).
    assert widget.load_params_btn.text() == "Load Params"
    assert widget.save_params_btn.text() == "Save Params"
    assert widget.reset_params_btn.text() == "Reset"
    # The experiments list no longer owns its own series Open/Save.
    assert not hasattr(widget.experiments_list, "load_series_btn")
    assert not hasattr(widget.experiments_list, "save_series_btn")
    assert not hasattr(widget, "_save_config")
    assert not hasattr(widget, "_load_config")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_workflow_shell.py -k "toolbar_exposes_project" -q`
Expected: FAIL — `AttributeError: 'napariTFMWidget' object has no attribute 'new_project_btn'`.

- [ ] **Step 3: Rebuild the shell title block**

In `napariTFM/widgets/_widget.py`, replace the title-row block (~398-409, from `title_row = QHBoxLayout()` through `container_layout.addLayout(title_row)`) with two rows — Project on the brand line, Parameters beneath:

```python
        # Brand row + the Project toolbar (the front door): New / Load / Save.
        # Save Project is always Save-as. Parameters are a separate preset row.
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("napariTFM")
        title.setStyleSheet(title_style())
        title_row.addWidget(title)
        title_row.addStretch()
        self.new_project_btn = self._make_toolbar_button("New Project", "Start a new project")
        self.load_project_btn = self._make_toolbar_button("Load Project", "Load a project")
        self.save_project_btn = self._make_toolbar_button("Save Project", "Save project as…")
        for _btn in (self.new_project_btn, self.load_project_btn, self.save_project_btn):
            title_row.addWidget(_btn)
        container_layout.addLayout(title_row)

        # Parameters preset toolbar (recipe only): Load / Save / Reset.
        params_row = QHBoxLayout()
        params_row.setContentsMargins(0, 0, 0, 0)
        params_row.addStretch()
        self.load_params_btn = self._make_toolbar_button("Load Params", "Load parameters preset")
        self.save_params_btn = self._make_toolbar_button("Save Params", "Save parameters preset")
        self.reset_params_btn = self._make_toolbar_button("Reset", "Reset parameters")
        for _btn in (self.load_params_btn, self.save_params_btn, self.reset_params_btn):
            params_row.addWidget(_btn)
        container_layout.addLayout(params_row)
```

- [ ] **Step 4: Wire the new buttons (temporary handlers)**

The param buttons keep their existing handlers. The project buttons get real handlers in Task 3; for now wire them to the existing series/placeholder methods so the widget constructs. Find the wiring block (~446-451):

```python
        # Wire the title-bar parameter-preset toolbar; the experiment-series
        # Open/Save lives on the list header instead.
        self.save_params_btn.clicked.connect(self._save_params)
        self.load_params_btn.clicked.connect(self._load_params)
        self.reset_params_btn.clicked.connect(self._reset_parameters)
        self.experiments_list.save_series_requested.connect(self._save_series)
        self.experiments_list.load_series_requested.connect(self._load_series)
        self.experiments_list.output_dir_changed.connect(self._reconcile_to_output_dir)
```

Replace with (project handlers land in Task 3; reference them now so the connections are final):

```python
        # Parameter preset toolbar (recipe import/export).
        self.save_params_btn.clicked.connect(self._save_params)
        self.load_params_btn.clicked.connect(self._load_params)
        self.reset_params_btn.clicked.connect(self._reset_parameters)
        # Project toolbar (the front door): handlers defined below.
        self.new_project_btn.clicked.connect(self._new_project)
        self.load_project_btn.clicked.connect(self._load_project)
        self.save_project_btn.clicked.connect(self._save_project)
```

- [ ] **Step 5: Add stub project handlers so the widget constructs**

So Task 2 stays runnable before Task 3 fills them in, add minimal placeholder methods to `napariTFMWidget` (just after `_make_toolbar_button`, ~627). Task 3 replaces their bodies:

```python
    def _new_project(self) -> None:
        self._project_open = True
        self._update_disclosure()

    def _load_project(self) -> None:
        self._project_open = True
        self._update_disclosure()

    def _save_project(self) -> None:
        pass
```

- [ ] **Step 6: Remove the series buttons from `ExperimentsList`**

In `napariTFM/widgets/_experiments_list.py`, delete the two series signals from the class signal block (~227-230):

```python
    save_series_requested = Signal()
    load_series_requested = Signal()
```

Then in `__init__`, delete the series-button block (~262-277, from the `# Series I/O lives with the series...` comment through `header.addWidget(self.save_series_btn)`). The header keeps only the `EXPERIMENTS` label and the stretch:

```python
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel("EXPERIMENTS")
        label.setStyleSheet(f"color: {TEXT_MID}; font-weight: bold;")
        header.addWidget(label)
        header.addStretch()

        layout.addLayout(header)
```

- [ ] **Step 7: Run the new test to verify it passes**

Run: `python -m pytest tests/test_workflow_shell.py -k "toolbar_exposes_project" -q`
Expected: PASS.

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: FAIL — the series tests (`test_save_series_writes_dataset_without_knobs`, `test_load_series_rebuilds_table_without_touching_params`) and reconcile tests still reference removed wiring; they are replaced in Task 3. All other tests pass. Note the failures and proceed.

- [ ] **Step 9: Commit**

```bash
git add napariTFM/widgets/_widget.py napariTFM/widgets/_experiments_list.py tests/test_workflow_shell.py
git commit -m "feat: Project/Parameters two-row toolbar; drop list series buttons

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Project bundle (save / load / new) + retire autosave

Make the Project file the single source of truth — folders + columns + optional output + calibration + run options + parameters — by reusing `build_series_config`/`series_records` plus a `parameters` block. Remove the autosave-to-output-dir path.

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (real `_new_project`/`_load_project`/`_save_project`; delete `_save_series`/`_load_series`/`_write_config`/`_read_config`/`_reconcile_to_output_dir`/`_config_path`; trim `_on_active_experiment_changed`/`_on_experiments_changed`)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing tests**

Remove the now-obsolete tests `test_save_series_writes_dataset_without_knobs`, `test_load_series_rebuilds_table_without_touching_params`, `test_reconcile_loads_existing_config_from_output_dir`, and `test_reconcile_writes_config_when_absent` (autosave + standalone-series behaviour is gone). Add these Project tests at the end of the file:

```python
def test_save_project_bundles_dataset_and_parameters(monkeypatch, app, tmp_path):
    import yaml

    widget = _stub_main_widget(monkeypatch)
    widget._project_open = True
    widget.experiments_list.add_folders(
        ["/data/a", "/data/b"],
        input_files={"beads": "beads.tif", "reference": "reference.tif"},
        columns={"day": "1"},
    )
    widget.parameter_manager.set_parameter("young_modulus", 9.0)
    out = tmp_path / "project.yaml"
    monkeypatch.setattr(
        _widget.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._save_project()

    config = yaml.safe_load(out.read_text())
    # Dataset side (reuses the series shape) ...
    assert config["root_folders"] == ["/data/a", "/data/b"]
    assert config["experiment_metadata"]["/data/a"] == {"day": "1"}
    assert config["input_files"]["beads"] == "beads.tif"
    assert "run_options" in config
    # ... plus the analysis recipe, all in one file.
    assert config["parameters"]["young_modulus"] == 9.0


def test_load_project_restores_dataset_and_parameters(monkeypatch, app, tmp_path):
    import yaml

    config = {
        "format_version": 2,
        "root_folders": ["/data/x", "/data/y"],
        "input_files": {"beads": "beads.tif", "reference": "reference.tif"},
        "experiment_metadata": {"/data/x": {"day": "1"}, "/data/y": {"day": "2"}},
        "run_options": {"disabled_stages": ["stress"], "processed_root": None},
        "parameters": {"young_modulus": 12.0},
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(config))

    widget = _stub_main_widget(monkeypatch)
    monkeypatch.setattr(
        _widget.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._load_project()

    assert widget._project_open is True
    assert widget.experiments_list.experiments() == ["/data/x", "/data/y"]
    assert widget.experiments_list.experiment_records()[1]["columns"] == {"day": "2"}
    assert widget.parameter_manager.get_parameter("young_modulus") == 12.0
    assert widget._stage_sections_by_key["stress"].is_enabled is False


def test_new_project_clears_to_empty_open_workspace(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/a"])
    widget.parameter_manager.set_parameter("young_modulus", 9.0)

    widget._new_project()

    assert widget._project_open is True
    assert widget.experiments_list.experiments() == []
    # Stress returns to its default-off state on a clean slate.
    assert widget._stage_sections_by_key["stress"].is_enabled is False


def test_autosave_path_is_gone(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    for attr in ("_write_config", "_read_config", "_reconcile_to_output_dir",
                 "_config_path", "_save_series", "_load_series"):
        assert not hasattr(widget, attr)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_workflow_shell.py -k "save_project_bundles or load_project_restores or new_project_clears or autosave_path_is_gone" -q`
Expected: FAIL — placeholder `_save_project` writes nothing; `_write_config` etc. still exist.

- [ ] **Step 3: Add the Project save/load/new handlers**

In `napariTFM/widgets/_widget.py`, ensure the series-builder imports are present at the top (they already are: `build_run_config, build_series_config, series_records`). Replace the three placeholder methods from Task 2 with real implementations:

```python
    def _new_project(self) -> None:
        """Clear to an empty, open workspace (G1): no rows, default knobs."""
        self._applying_state = True
        try:
            self.parameter_manager.reset_all_parameters()
            self.experiments_list.set_records([])
            for key, section in self._stage_sections_by_key.items():
                if section.enable_btn is not None:
                    section.set_enabled(False)  # stress is off by default (D1)
            self.data_manager.set_output_dir(None)
        finally:
            self._applying_state = False
        self._project_open = True
        self._dirty = False
        self.refresh()
        self._update_disclosure()

    def _save_project(self) -> None:
        """Save the whole project — dataset + recipe — to one YAML (Save-as)."""
        import yaml

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save project", "project.yaml", "YAML Files (*.yaml *.yml)"
            )
            if not file_path:
                return
            if not file_path.lower().endswith((".yml", ".yaml")):
                file_path += ".yaml"
            config = build_series_config(
                self.experiments_list.experiment_records(),
                disabled_stages=self._disabled_stages(),
                processed_root=self.data_manager.output_dir,
            )
            config["format_version"] = PROJECT_FORMAT_VERSION
            config["parameters"] = self.parameter_manager.get_all_parameters()
            with open(file_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            self._dirty = False
            QMessageBox.information(self, "Success", "Project saved!")
        except Exception as e:
            logger.error(f"Error saving project: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def _load_project(self) -> None:
        """Load a project bundle: dataset + run options + analysis parameters."""
        import yaml

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load project", "", "YAML Files (*.yaml *.yml)"
            )
            if not file_path:
                return
            with open(file_path) as f:
                config = yaml.safe_load(f) or {}
            self._applying_state = True
            try:
                self._apply_parameters(config.get("parameters", {}) or {})
                run_options = config.get("run_options", {}) or {}
                disabled = set(run_options.get("disabled_stages") or [])
                for key, section in self._stage_sections_by_key.items():
                    if section.enable_btn is not None:
                        section.set_enabled(key not in disabled)
                self.experiments_list.set_records(series_records(config))
            finally:
                self._applying_state = False
            self._project_open = True
            self._dirty = False
            self.refresh()
            self._update_disclosure()
            QMessageBox.information(self, "Success", "Project loaded!")
        except Exception as e:
            logger.error(f"Error loading project: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
```

- [ ] **Step 4: Add the `PROJECT_FORMAT_VERSION` constant and a `_dirty` flag**

Near the top of `napariTFM/widgets/_widget.py`, beside `STATE_VERSION = 1` (~38), add:

```python
PROJECT_FORMAT_VERSION = 2
```

In `__init__`, beside `self._project_open = False` (added in Task 1), add:

```python
        self._dirty = False
```

- [ ] **Step 5: Delete the autosave + series methods**

Delete these methods from `napariTFMWidget` entirely: `_save_series`, `_load_series`, `_config_path`, `_read_config`, `_write_config`, `_reconcile_to_output_dir`. (They span ~879-1071; remove each definition.)

Then trim their callers. In `_on_active_experiment_changed` (~833), the tail currently reads:

```python
        self._update_disclosure()
        self._write_config()
```

Replace with:

```python
        self._update_disclosure()
```

In `_on_experiments_changed` (~845), the body currently calls `self._write_config()`. Replace the whole method with:

```python
    def _on_experiments_changed(self) -> None:
        if not self._applying_state:
            self._dirty = True
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_workflow_shell.py -k "save_project_bundles or load_project_restores or new_project_clears or autosave_path_is_gone" -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS. (`test_get_set_state_round_trips_parameters`, `test_state_round_trips_experiments_and_active`, and the stress-state tests still pass — `get_state`/`set_state` are retained and untouched.)

- [ ] **Step 8: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "feat: Project bundle save/load/new; retire output-dir autosave

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Scrollable, fixed-height experiment list

Bound the rows region so hundreds of discovered positions scroll internally instead of growing the panel.

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py` (`__init__` rows region)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing test**

Add at the end of `tests/test_workflow_shell.py`:

```python
def test_experiment_rows_live_in_a_bounded_scroll_area(monkeypatch, app):
    from qtpy.QtWidgets import QScrollArea

    widget = _stub_main_widget(monkeypatch)
    scroll = widget.experiments_list._rows_scroll
    assert isinstance(scroll, QScrollArea)
    assert scroll.widgetResizable() is True
    # Capped so a long list scrolls instead of pushing the panel down.
    assert 0 < scroll.maximumHeight() <= 260
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_workflow_shell.py -k "rows_live_in_a_bounded" -q`
Expected: FAIL — `AttributeError: 'ExperimentsList' object has no attribute '_rows_scroll'`.

- [ ] **Step 3: Wrap the rows box in a bounded scroll area**

In `napariTFM/widgets/_experiments_list.py`, add `QScrollArea` to the qtpy imports (the `from qtpy.QtWidgets import (...)` block, ~9-20):

```python
    QScrollArea,
```

Then in `__init__`, replace the rows-box block (~294-297):

```python
        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(2)
        layout.addLayout(self._rows_box)
```

with a bounded scroll area wrapping the rows container:

```python
        # Rows live in a bounded scroll region: a long discovered list scrolls
        # internally instead of pushing the rest of the panel down.
        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(2)
        rows_container = QWidget()
        rows_container.setLayout(self._rows_box)
        self._rows_scroll = QScrollArea()
        self._rows_scroll.setObjectName("experiments_rows_scroll")
        self._rows_scroll.setWidgetResizable(True)
        self._rows_scroll.setMaximumHeight(220)
        self._rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._rows_scroll.setWidget(rows_container)
        layout.addWidget(self._rows_scroll)
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `python -m pytest tests/test_workflow_shell.py -k "rows_live_in_a_bounded" -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS. (`_rebuild_rows` operates on `_rows_box`, which is unchanged, so row add/remove still works.)

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_workflow_shell.py
git commit -m "feat: bound the experiment list in a fixed-height scroll area

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Dynamic Discover tooltip

Build the Discover button tooltip from the filled input-file names, rebuilt whenever a name changes; show optionals only when present.

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py` (new `_discover_tooltip_text`/`_update_discover_tooltip`, wiring in `_build_config_header` + after `add_btn` creation)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing tests**

Add at the end of `tests/test_workflow_shell.py`:

```python
def test_discover_tooltip_lists_only_filled_inputs(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    el = widget.experiments_list
    el.file_name_inputs["beads"].setText("beads.tif")
    el.file_name_inputs["reference"].setText("reference.tif")
    el.file_name_inputs["cells"].setText("")
    el.file_name_inputs["masks"].setText("")

    tip = el.add_btn.toolTip()
    assert "beads.tif" in tip and "reference.tif" in tip
    assert "and reference.tif" in tip  # two-item grammar
    assert "cells" not in tip and "masks" not in tip


def test_discover_tooltip_includes_present_optionals(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    el = widget.experiments_list
    el.file_name_inputs["beads"].setText("b.tif")
    el.file_name_inputs["reference"].setText("r.tif")
    el.file_name_inputs["cells"].setText("c.tif")
    el.file_name_inputs["masks"].setText("m.tif")

    tip = el.add_btn.toolTip()
    # Oxford-free list: "b.tif, r.tif, c.tif and m.tif"
    assert "b.tif, r.tif, c.tif and m.tif" in tip
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_workflow_shell.py -k "discover_tooltip" -q`
Expected: FAIL — the tooltip is the static "Discover…" default, so the assertions miss.

- [ ] **Step 3: Add the tooltip builder + updater**

In `napariTFM/widgets/_experiments_list.py`, add these two methods to `ExperimentsList` (place them right after `input_file_config`, ~516):

```python
    def _discover_tooltip_text(self) -> str:
        """Plain-language scan description from the filled input-file names."""
        names = [
            field.text().strip()
            for key in ("beads", "reference", "cells", "masks")
            for field in (self.file_name_inputs[key],)
            if field.text().strip()
        ]
        if not names:
            return "Choose a folder to scan for experiment subfolders."
        if len(names) == 1:
            joined = names[0]
        else:
            joined = f"{', '.join(names[:-1])} and {names[-1]}"
        return (
            f"napariTFM will scan the chosen folder for subfolders containing "
            f"{joined}, and initialize each for analysis."
        )

    def _update_discover_tooltip(self) -> None:
        self.add_btn.setToolTip(self._discover_tooltip_text())
```

- [ ] **Step 4: Wire the updater to the file-name fields and set the initial tooltip**

In `_build_config_header`, find the loop that builds `self.file_name_inputs` (~457-469). Inside the loop, after `self.file_name_inputs[key] = field`, connect changes:

```python
            self.file_name_inputs[key] = field
            field.textChanged.connect(lambda _t: self._update_discover_tooltip())
```

Then in `__init__`, right after the `self.add_btn` is fully configured (after `self.add_btn.clicked.connect(self._on_add_clicked)`, ~310), set the initial tooltip:

```python
        self._update_discover_tooltip()
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_workflow_shell.py -k "discover_tooltip" -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_workflow_shell.py
git commit -m "feat: dynamic Discover tooltip naming the filled input files

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Dirty-flag guard on New / Load Project

Track unsaved edits and confirm before discarding them — the safety net now that autosave is gone.

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (`_on_parameter_changed` dirty mark; `_confirm_discard` guard in `_new_project`/`_load_project`)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing tests**

Add at the end of `tests/test_workflow_shell.py`:

```python
def test_parameter_edit_marks_project_dirty(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._new_project()            # opens a clean (not dirty) project
    assert widget._dirty is False
    widget.parameter_manager.set_parameter("young_modulus", 7.0)
    assert widget._dirty is True


def test_new_project_on_dirty_workspace_asks_before_discarding(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._new_project()
    widget.experiments_list.add_folders(["/data/a"])  # marks dirty
    assert widget._dirty is True

    asked = {"n": 0}

    def _decline(*a, **k):
        asked["n"] += 1
        return _widget.QMessageBox.No

    monkeypatch.setattr(_widget.QMessageBox, "question", staticmethod(_decline))

    widget._new_project()
    # The user declined: the workspace is left intact.
    assert asked["n"] == 1
    assert widget.experiments_list.experiments() == ["/data/a"]


def test_new_project_proceeds_when_discard_confirmed(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._new_project()
    widget.experiments_list.add_folders(["/data/a"])

    monkeypatch.setattr(
        _widget.QMessageBox, "question",
        staticmethod(lambda *a, **k: _widget.QMessageBox.Yes),
    )

    widget._new_project()
    assert widget.experiments_list.experiments() == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_workflow_shell.py -k "marks_project_dirty or on_dirty_workspace_asks or proceeds_when_discard" -q`
Expected: FAIL — `_on_parameter_changed` doesn't set `_dirty`, and `_new_project` doesn't guard.

- [ ] **Step 3: Mark dirty on parameter edits**

In `napariTFM/widgets/_widget.py`, `_on_parameter_changed` (~1083) currently starts by routing updates. Add a dirty mark at the top of the method body:

```python
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Propagate parameter edits to the widgets that display them."""
        if not self._applying_state:
            self._dirty = True
        if param_name in ("pixel_size", "frame_interval"):
```

(Leave the rest of the method unchanged.)

- [ ] **Step 4: Add the discard-confirm guard and call it from the project handlers**

Add a helper method to `napariTFMWidget` (just before `_new_project`):

```python
    def _confirm_discard(self) -> bool:
        """True if it's safe to clear the workspace (clean, or user said yes)."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Discard changes?",
            "This project has unsaved changes. Discard them?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes
```

Then guard `_new_project` — add as its first line (before `self._applying_state = True`):

```python
        if not self._confirm_discard():
            return
```

And guard `_load_project` — add as its first line (before the `import yaml`):

```python
        if not self._confirm_discard():
            return
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_workflow_shell.py -k "marks_project_dirty or on_dirty_workspace_asks or proceeds_when_discard" -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "feat: dirty-flag guard before New/Load Project discards work

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest tests/ -q`
Expected: PASS — confirm no collateral breakage outside `test_workflow_shell.py`.

- [ ] **Step 2: Smoke-check the import path**

Run: `python -c "import napariTFM.widgets._widget, napariTFM.widgets._experiments_list; print('ok')"`
Expected: prints `ok` with no import error.

- [ ] **Step 3: Commit any remaining changes (if the suite required a fix)**

```bash
git add -A
git commit -m "test: full-suite verification for onboarding disclosure

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage:**
- G0/G1/G2 gated reveal → Task 1 (`_update_disclosure`, `_empty_hint`).
- Project front door, two-row toolbar (Project on top, Params below) → Task 2.
- Project bundles everything; Params buttons act on recipe only; autosave retired → Task 3.
- Output dir demoted to optional / no longer a gate → Task 3 (autosave removal means nothing gates on `output_dir`; `_new_project` sets it to `None`).
- Multi-batch Discover preserved → untouched (`add_folders`/columns/commit flow intact; verified by retained `test_run_all_*` tests).
- Scrollable fixed-height list → Task 4.
- Dynamic Discover tooltip → Task 5.
- Dirty-flag guard → Task 6.
- Headless tests homed in `test_workflow_shell.py` → every task.

**Status-line refinement** (status visible in G1+G2, not G2-only) is documented at the top and in Task 1's `_update_disclosure`, so batch progress stays visible during Run all.

**Type/name consistency:** `_project_open`, `_dirty`, `_update_disclosure`, `_empty_hint`, `_confirm_discard`, `_new_project`/`_load_project`/`_save_project`, `new_project_btn`/`load_project_btn`/`save_project_btn`, `load_params_btn`/`save_params_btn`/`reset_params_btn`, `_rows_scroll`, `_update_discover_tooltip`, and `PROJECT_FORMAT_VERSION` are introduced once and referenced consistently across tasks and tests.

**Retained API:** `get_state`/`set_state` are kept (still exercised by `test_get_set_state_round_trips_parameters`, `test_state_round_trips_experiments_and_active`, and the stress-state tests); only the autosave plumbing around them is removed.
