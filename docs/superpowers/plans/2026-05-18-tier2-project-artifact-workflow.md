# Tier 2 Project Artifact Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Tier 1 workflow shell operational with visible output-directory state, richer artifact rows, per-artifact actions, and targeted cleanup of duplicated stage-local UI.

**Architecture:** Extend the existing `DataManager` / `ArtifactState` model rather than adding a second project registry. `ProjectSection` writes output-directory state through `DataManager`; `StageDataStatusPanel` renders `ArtifactState`; `_widget.py` wires row actions to existing controller, visualization, and save methods. Cleanup only hides duplicated visible controls after equivalent shell/header/row behavior is tested.

**Tech Stack:** Python, qtpy/PyQt widgets, pytest, napari plugin widget shell, existing `DataManager` and `ParameterManager`.

**Spec:** `docs/superpowers/specs/2026-05-18-tier2-project-artifact-workflow-design.md`

---

## File Map

- Modify `napariTFM/widgets/_project_section.py`
  - Add output-directory row to `_GeneralBody`.
  - Accept `data_manager=None` for direct `ProjectSection` tests and pass a real `DataManager` from the shell.
  - Sync label from `DataManager.output_dir`.
  - Expose `choose_output_dir_btn` and `output_dir_label`.

- Modify `napariTFM/widgets/_widget.py`
  - Construct `ProjectSection(self.parameter_manager, self.data_manager)`.
  - Add artifact spec builders for displacement, force, stress, and generated outputs.
  - Add small shell helper methods for saving artifacts and calling existing stage load/view behavior.
  - Hide duplicated stage-local panels only after row actions exist.

- Modify `napariTFM/widgets/_stage_data_status.py`
  - Render `ArtifactState` metadata: dirty, saved path, error.
  - Keep `artifact_labels` compatibility for existing tests.

- Modify `napariTFM/widgets/displacement_analysis_widget.py`
  - Hide or expose `DisplacementDataPanel` behavior through callable helpers while keeping controller behavior alive.

- Modify `napariTFM/widgets/fttc_widget.py`
  - Hide visible load-displacement panel once shell row action calls an existing load path or focused helper.

- Modify `napariTFM/widgets/msm_widget.py`
  - Hide visible load-force/load-mask panel once shell row actions call existing handlers or focused helpers.

- Modify `napariTFM/widgets/batch_analysis_widget.py`
  - Keep existing batch-specific controls visible.
  - Ensure duplicate analysis parameter groups stay hidden and config generation uses `ParameterManager`.

- Modify tests:
  - `tests/test_project_section.py`
  - `tests/test_artifact_row.py`
  - `tests/test_workflow_shell.py`
  - `tests/test_batch_parameters.py`

---

## Task 1: Project Output Directory UI

**Files:**
- Modify: `napariTFM/widgets/_project_section.py`
- Modify: `napariTFM/widgets/_widget.py`
- Modify: `tests/test_project_section.py`
- Modify: `tests/test_workflow_shell.py`

Add an output-directory row to the Project section. The row displays `No output directory` when unset and the selected path when set. The widget writes through `DataManager.set_output_dir()` and refreshes when `DataManager` callbacks fire.

- [ ] **Step 1.1: Write failing ProjectSection tests**

Add these tests to `tests/test_project_section.py`:

```python
from pathlib import Path


class _StubDataManager:
    def __init__(self):
        self.output_dir = None
        self.set_calls = []
        self._callbacks = []

    def set_output_dir(self, path):
        self.output_dir = Path(path).expanduser() if path else None
        self.set_calls.append(self.output_dir)
        for callback in list(self._callbacks):
            callback()

    def add_change_callback(self, callback):
        self._callbacks.append(callback)


def test_project_section_shows_unset_output_directory(app):
    section = ProjectSection(_StubParameterManager(), _StubDataManager())

    assert section.output_dir_label.text() == "No output directory"
    assert section.output_dir_label.toolTip() == ""


def test_project_section_syncs_output_directory_from_data_manager(app, tmp_path):
    data_manager = _StubDataManager()
    section = ProjectSection(_StubParameterManager(), data_manager)

    data_manager.set_output_dir(tmp_path)

    assert section.output_dir_label.text() == str(tmp_path)
    assert section.output_dir_label.toolTip() == str(tmp_path)


def test_project_section_exposes_output_directory_button(app):
    section = ProjectSection(_StubParameterManager(), _StubDataManager())

    assert isinstance(section.choose_output_dir_btn, QPushButton)
    assert section.choose_output_dir_btn.objectName() == "project_choose_output_dir_button"
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
pytest tests/test_project_section.py -v
```

Expected: failures because `ProjectSection` does not accept `data_manager` and does not expose `output_dir_label` / `choose_output_dir_btn`.

- [ ] **Step 1.3: Implement output-directory row**

In `napariTFM/widgets/_project_section.py`, add imports:

```python
from pathlib import Path

from qtpy.QtWidgets import QFileDialog
```

Change `_GeneralBody.__init__` signature:

```python
def __init__(self, parameter_manager, data_manager=None):
    super().__init__()
    self.parameter_manager = parameter_manager
    self.data_manager = data_manager
```

After the parameter rows and before the parameter buttons, add:

```python
self.output_dir_label = QLabel("No output directory")
self.output_dir_label.setObjectName("project_output_dir_label")
self.choose_output_dir_btn = QPushButton("Output Directory")
self.choose_output_dir_btn.setObjectName("project_choose_output_dir_button")
self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)

output_row = QHBoxLayout()
output_row.addWidget(self.choose_output_dir_btn)
output_row.addWidget(self.output_dir_label, stretch=1)
layout.addLayout(output_row)

if self.data_manager is not None:
    self.data_manager.add_change_callback(self._sync_output_dir)
self._sync_output_dir()
```

Add methods to `_GeneralBody`:

```python
def _choose_output_dir(self):
    if self.data_manager is None:
        return
    current = self.data_manager.output_dir or Path.home()
    path = QFileDialog.getExistingDirectory(self, "Select Pipeline Output Directory", str(current))
    if path:
        self.data_manager.set_output_dir(path)

def _sync_output_dir(self):
    path = getattr(self.data_manager, "output_dir", None)
    if path is None:
        self.output_dir_label.setText("No output directory")
        self.output_dir_label.setToolTip("")
        return
    text = str(path)
    self.output_dir_label.setText(text)
    self.output_dir_label.setToolTip(text)
```

Change `ProjectSection.__init__`:

```python
def __init__(self, parameter_manager, data_manager=None):
    body = _GeneralBody(parameter_manager, data_manager)
    super().__init__("Project", body, expanded=True, accent=None)
    self.body = body
    self.run_cancel_btn.setVisible(False)
    self.preview_button.setVisible(False)
```

Add properties:

```python
@property
def output_dir_label(self):
    return self.body.output_dir_label

@property
def choose_output_dir_btn(self):
    return self.body.choose_output_dir_btn
```

- [ ] **Step 1.4: Wire shell construction**

In `napariTFM/widgets/_widget.py`, change:

```python
self.project_section = ProjectSection(self.parameter_manager)
```

to:

```python
self.project_section = ProjectSection(self.parameter_manager, self.data_manager)
```

- [ ] **Step 1.5: Add shell integration test**

Add to `tests/test_workflow_shell.py`:

```python
def test_main_widget_project_section_tracks_output_directory(monkeypatch, app, tmp_path):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)

    assert widget.project_section.output_dir_label.text() == str(tmp_path)
```

Update `_StubDataManager` in `tests/test_workflow_shell.py`:

```python
from pathlib import Path

def __init__(self):
    self._callbacks = []
    self.output_dir = None
    ...

def set_output_dir(self, path):
    self.output_dir = Path(path).expanduser() if path else None
    self.notify_changed()
```

- [ ] **Step 1.6: Run focused tests**

```bash
pytest tests/test_project_section.py tests/test_workflow_shell.py::test_main_widget_project_section_tracks_output_directory -v
```

Expected: pass.

- [ ] **Step 1.7: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 1.8: Commit**

```bash
git add napariTFM/widgets/_project_section.py napariTFM/widgets/_widget.py tests/test_project_section.py tests/test_workflow_shell.py
git commit -m "Show pipeline output directory in Project section"
```

---

## Task 2: Rich Artifact-State Row Rendering

**Files:**
- Modify: `napariTFM/widgets/_stage_data_status.py`
- Modify: `tests/test_artifact_row.py`

Teach rows to render state metadata from `ArtifactState`: dirty/unsaved, saved path, and error. Keep the existing `row.refresh(available, info_text)` API for compatibility, and add a richer `row.refresh_state(state, fallback_info)` path.

- [ ] **Step 2.1: Write failing row tests**

Add to `tests/test_artifact_row.py`:

```python
from pathlib import Path
from types import SimpleNamespace


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
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
pytest tests/test_artifact_row.py -v
```

Expected: `AttributeError` on `refresh_state`.

- [ ] **Step 2.3: Implement `refresh_state`**

In `_ArtifactRow`, add:

```python
def refresh_state(self, state, info_text: str) -> None:
    available = getattr(state, "value", None) is not None
    error = getattr(state, "error", "")
    path = getattr(state, "path", None)
    dirty = bool(getattr(state, "dirty", False))

    if error:
        self.glyph_label.setText(STATUS_GLYPHS["error"])
        self.info_label.setText(str(error))
        self.info_label.setToolTip(str(error))
        if self.view_btn is not None:
            self.view_btn.setVisible(available)
        if self.action_btn is not None and self.spec.role == "output":
            self.action_btn.setEnabled(available)
        return

    self.refresh(available=available, info_text=info_text)
    hints = []
    if available and dirty:
        hints.append("Unsaved")
    if available and path is not None and not dirty:
        hints.append(Path(path).name)
        self.info_label.setToolTip(str(path))
    else:
        self.info_label.setToolTip("")
    if hints:
        self.info_label.setText(f"{self.info_label.text()} · {' · '.join(hints)}")
```

Add import:

```python
from pathlib import Path
```

- [ ] **Step 2.4: Teach panel to use `ArtifactState`**

In `StageDataStatusPanel.refresh()`, replace:

```python
value = self._artifact_value(artifact)
available = value is not None
...
info_text = self._info_text(artifact, value, available)
self.artifact_rows[artifact.key].refresh(available=available, info_text=info_text)
```

with:

```python
state = self._artifact_state(artifact)
value = state.value if state is not None else self._artifact_value(artifact)
available = value is not None
...
info_text = self._info_text(artifact, value, available)
if state is not None:
    self.artifact_rows[artifact.key].refresh_state(state, info_text=info_text)
else:
    self.artifact_rows[artifact.key].refresh(available=available, info_text=info_text)
```

Add:

```python
def _artifact_state(self, artifact: DataArtifactSpec):
    get_artifact = getattr(self.data_manager, "get_artifact", None)
    if get_artifact is None:
        return None
    try:
        return get_artifact(artifact.key)
    except (KeyError, AttributeError):
        return None
```

- [ ] **Step 2.5: Run focused tests**

```bash
pytest tests/test_artifact_row.py tests/test_workflow_shell.py::test_stage_data_status_refreshes_from_data_manager -v
```

Expected: pass.

- [ ] **Step 2.6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 2.7: Commit**

```bash
git add napariTFM/widgets/_stage_data_status.py tests/test_artifact_row.py
git commit -m "Render artifact dirty saved and error state in rows"
```

---

## Task 3: Generated Output Save Actions

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Modify: `tests/test_workflow_shell.py`

Wire generated output rows to `DataManager.auto_save_artifact()`. Preprocessing outputs pass `pixel_size` and `frame_interval`; result object outputs use existing `.npy` saving.

- [ ] **Step 3.1: Write failing tests**

Add to `tests/test_workflow_shell.py`:

```python
def test_generated_output_row_save_calls_data_manager_with_calibration(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.preprocessed_bead_stack = object()
    widget.refresh_stage_statuses()

    row = widget._stage_status_panels_by_key["preprocessing"].artifact_rows["preprocessed_bead_stack"]
    row.action_btn.click()

    assert widget.data_manager.auto_save_calls[-1] == {
        "key": "preprocessed_bead_stack",
        "pixel_size": 1.0,
        "frame_interval": 1.0,
    }


def test_failed_generated_output_save_marks_artifact_error(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.raise_on_save = RuntimeError("disk full")
    widget.data_manager.preprocessed_reference = object()
    widget.refresh_stage_statuses()

    row = widget._stage_status_panels_by_key["preprocessing"].artifact_rows["preprocessed_reference"]
    row.action_btn.click()

    assert widget.data_manager.artifact_errors[-1] == ("preprocessed_reference", "disk full")
```

Update `_StubDataManager`:

```python
def __init__(self):
    ...
    self.auto_save_calls = []
    self.artifact_errors = []
    self.raise_on_save = None

def auto_save_artifact(self, key, pixel_size=None, frame_interval=None):
    if self.raise_on_save is not None:
        raise self.raise_on_save
    self.auto_save_calls.append(
        {"key": key, "pixel_size": pixel_size, "frame_interval": frame_interval}
    )
    return None

def mark_artifact_error(self, key, error):
    self.artifact_errors.append((key, error))
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
pytest tests/test_workflow_shell.py::test_generated_output_row_save_calls_data_manager_with_calibration tests/test_workflow_shell.py::test_failed_generated_output_save_marks_artifact_error -v
```

Expected: `row.action_btn is None` or no save call.

- [ ] **Step 3.3: Add save helper in shell**

In `napariTFM/widgets/_widget.py`, add:

```python
def _save_generated_artifact(self, key: str):
    try:
        kwargs = {}
        if key.startswith("preprocessed_"):
            kwargs = {
                "pixel_size": self.parameter_manager.get_ui_parameter("pixel_size"),
                "frame_interval": self.parameter_manager.get_ui_parameter("frame_interval"),
            }
        self.data_manager.auto_save_artifact(key, **kwargs)
    except Exception as exc:
        self.data_manager.mark_artifact_error(key, str(exc))
        QMessageBox.warning(self, "Save Failed", str(exc))
    finally:
        self.refresh_stage_statuses()
```

- [ ] **Step 3.4: Wire output specs**

Pass a `save_artifact` callable into `_build_preprocessing_specs`:

```python
def _build_preprocessing_specs(preprocessing_widget, visualization_manager, save_artifact):
    ...
    DataArtifactSpec(..., on_view=view("preprocessed_reference"), on_action=lambda: save_artifact("preprocessed_reference"))
```

In `napariTFMWidget.__init__`, call:

```python
stage_data_artifacts["preprocessing"] = _build_preprocessing_specs(
    self.preprocessing_widget,
    self.visualization_manager,
    self._save_generated_artifact,
)
```

Add similar builders or inline replacement specs for:

```python
"displacement_results" -> self._save_generated_artifact("displacement_results")
"force_results" -> self._save_generated_artifact("force_results")
"stress_results" -> self._save_generated_artifact("stress_results")
```

- [ ] **Step 3.5: Run focused tests**

```bash
pytest tests/test_workflow_shell.py::test_generated_output_row_save_calls_data_manager_with_calibration tests/test_workflow_shell.py::test_failed_generated_output_save_marks_artifact_error tests/test_artifact_row.py -v
```

Expected: pass.

- [ ] **Step 3.6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 3.7: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Wire generated artifact save actions into status rows"
```

---

## Task 4: Input Load/View Actions Beyond Preprocessing

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Modify: `tests/test_workflow_shell.py`
- Modify: `napariTFM/widgets/fttc_widget.py`
- Modify: `napariTFM/widgets/msm_widget.py`

Wire displacement, force, and stress input rows to existing input-load behavior. Keep helpers small and focused around existing UI methods.

- [ ] **Step 4.1: Write failing tests**

Add to `tests/test_workflow_shell.py`:

```python
def test_displacement_data_rows_route_assignment_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    rows = widget._stage_status_panels_by_key["displacement"].artifact_rows

    rows["preprocessed_reference"].action_btn.click()
    rows["preprocessed_bead_stack"].action_btn.click()

    assert widget.displacement_widget.loaded_active_layers == ["reference", "beads"]
```

Extend `_StubStageWidget`:

```python
self.loaded_files = []

def load_result_artifact(self, key):
    self.loaded_files.append(key)
```

Add:

```python
def test_force_and_stress_input_rows_route_load_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    widget._stage_status_panels_by_key["force"].artifact_rows["displacement_results"].action_btn.click()
    widget._stage_status_panels_by_key["stress"].artifact_rows["force_results"].action_btn.click()
    widget._stage_status_panels_by_key["stress"].artifact_rows["mask_stack"].action_btn.click()

    assert widget.force_widget.loaded_files == ["displacement_results"]
    assert widget.msm_widget.loaded_files == ["force_results", "mask_stack"]
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
pytest tests/test_workflow_shell.py::test_displacement_data_rows_route_assignment_actions tests/test_workflow_shell.py::test_force_and_stress_input_rows_route_load_actions -v
```

Expected: row action buttons are missing.

- [ ] **Step 4.3: Add shell wiring helpers**

In `_widget.py`, add:

```python
def _call_if_present(self, owner, method_name: str, *args):
    method = getattr(owner, method_name, None)
    if method is None:
        return
    method(*args)
```

Add builder functions near `_build_preprocessing_specs`:

```python
def _build_displacement_specs(displacement_widget, visualization_manager, save_artifact):
    def assign(role: str):
        return lambda: displacement_widget.load_active_layer(role)
    return [
        DataArtifactSpec("preprocessed_reference", "Preprocessed reference", "preprocessed_reference", "input", on_action=assign("reference")),
        DataArtifactSpec("preprocessed_bead_stack", "Preprocessed beads", "preprocessed_bead_stack", "input", on_action=assign("beads")),
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "output", on_action=lambda: save_artifact("displacement_results")),
    ]

def _build_force_specs(force_widget, save_artifact):
    return [
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "input", on_action=lambda: force_widget.load_result_artifact("displacement_results") if hasattr(force_widget, "load_result_artifact") else None),
        DataArtifactSpec("force_results", "Traction map", "force_results", "output", on_action=lambda: save_artifact("force_results")),
    ]

def _build_stress_specs(stress_widget, save_artifact):
    return [
        DataArtifactSpec("force_results", "Traction map", "force_results", "input", on_action=lambda: stress_widget.load_result_artifact("force_results") if hasattr(stress_widget, "load_result_artifact") else None),
        DataArtifactSpec("mask_stack", "Mask stack", "mask_stack", "input", required=False, on_action=lambda: stress_widget.load_result_artifact("mask_stack") if hasattr(stress_widget, "load_result_artifact") else None),
        DataArtifactSpec("stress_results", "Stress map", "stress_results", "output", on_action=lambda: save_artifact("stress_results")),
    ]
```

Add focused wrapper methods to `FTTCWidget` and `MSMWidget` so the shell can call one stable method. The first implementation uses existing `DataManager.load_result_artifact()` for result objects and the existing mask-load behavior for masks:

```python
def load_result_artifact(self, key: str):
    self.data_manager.load_result_artifact(key, self._choose_result_path(key))
    self._update_ui_state()
```

Add `_choose_result_path(key)` as a small file-dialog helper in the relevant widget:

```python
def _choose_result_path(self, key: str):
    path, _ = QFileDialog.getOpenFileName(
        self,
        f"Load {key.replace('_', ' ')}",
        "",
        "NumPy Files (*.npy)",
    )
    return path
```

For `mask_stack`, the `MSMWidget.load_result_artifact("mask_stack")` wrapper should call the same selected-layer mask-loading path used by the existing `load_mask_btn`.

- [ ] **Step 4.4: Wire builders into shell**

In `napariTFMWidget.__init__`, after preprocessing builder:

```python
stage_data_artifacts["displacement"] = _build_displacement_specs(
    self.displacement_widget,
    self.visualization_manager,
    self._save_generated_artifact,
)
stage_data_artifacts["force"] = _build_force_specs(
    self.force_widget,
    self._save_generated_artifact,
)
stage_data_artifacts["stress"] = _build_stress_specs(
    self.msm_widget,
    self._save_generated_artifact,
)
```

- [ ] **Step 4.5: Run focused tests**

```bash
pytest tests/test_workflow_shell.py::test_preprocessing_data_rows_route_assignment_actions tests/test_workflow_shell.py::test_displacement_data_rows_route_assignment_actions tests/test_workflow_shell.py::test_force_and_stress_input_rows_route_load_actions -v
```

Expected: pass.

- [ ] **Step 4.6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4.7: Commit**

```bash
git add napariTFM/widgets/_widget.py napariTFM/widgets/fttc_widget.py napariTFM/widgets/msm_widget.py tests/test_workflow_shell.py
git commit -m "Wire input load actions into artifact rows"
```

---

## Task 5: Stage-Local Duplicate UI Cleanup

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Modify: `tests/test_workflow_shell.py`

Hide duplicated visible data/action controls from stage bodies once the shell rows and headers own those surfaces.

- [ ] **Step 5.1: Write failing cleanup tests**

Add to `tests/test_workflow_shell.py`:

```python
def test_main_widget_hides_stage_local_data_and_action_panels_after_shell_wiring(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    for stage_widget in [
        widget.displacement_widget,
        widget.force_widget,
        widget.msm_widget,
    ]:
        assert getattr(stage_widget, "data_panel", None) is None or not stage_widget.data_panel.isVisible()
        assert getattr(stage_widget, "action_panel", None) is None or not stage_widget.action_panel.isVisible()
```

Extend `_StubStageWidget.__init__`:

```python
self.data_panel = QWidget()
self.action_panel = QWidget()
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
pytest tests/test_workflow_shell.py::test_main_widget_hides_stage_local_data_and_action_panels_after_shell_wiring -v
```

Expected: panels are visible or not explicitly hidden.

- [ ] **Step 5.3: Implement cleanup helper**

In `_widget.py`, add:

```python
def _hide_redundant_stage_shell_controls(self):
    for widget in [self.displacement_widget, self.force_widget, self.msm_widget]:
        for attr in ("data_panel", "action_panel"):
            panel = getattr(widget, attr, None)
            if panel is not None:
                panel.setVisible(False)
```

Call it after `_hide_embedded_parameter_panels()`:

```python
self._hide_embedded_parameter_panels()
self._hide_redundant_stage_shell_controls()
```

Keep controller-owned attributes intact. Do not delete `data_panel` or `action_panel`; only hide them from the visible workflow body.

- [ ] **Step 5.4: Confirm header actions still proxy**

Run:

```bash
pytest tests/test_workflow_shell.py::test_main_widget_stage_headers_wire_existing_stage_actions tests/test_workflow_shell.py::test_stage_section_header_actions_proxy_child_buttons -v
```

Expected: pass.

- [ ] **Step 5.5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5.6: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Hide redundant stage-local data and action panels"
```

---

## Task 6: Batch Slimdown Guardrails

**Files:**
- Modify: `tests/test_batch_parameters.py`
- Modify: `napariTFM/widgets/batch_analysis_widget.py`

Batch already has some guardrails. Strengthen them around Tier 2 expectations: batch-specific controls remain, analysis parameter controls stay absent, and config generation reads from `ParameterManager`.

- [ ] **Step 6.1: Add failing or strengthening tests**

Add to `tests/test_batch_parameters.py`:

```python
def test_batch_keeps_batch_specific_controls_visible_after_parameter_slimdown():
    app = _app()
    widget = BatchAnalysisWidget(None, object(), ParameterManager(), object())
    widget.show()
    app.processEvents()

    assert widget.save_config_btn.isVisibleTo(widget)
    assert widget.load_config_btn.isVisibleTo(widget)
    assert widget.run_analysis_btn.isVisibleTo(widget)
    assert widget.folder_list_widget.isVisibleTo(widget)


def test_batch_config_generation_does_not_read_duplicate_parameter_widgets():
    fake = SimpleNamespace(
        folder_list_widget=_List(),
        file_inputs={"beads": _Text("beads.tif"), "reference": _Text("ref.tif"), "cells": _Text("")},
        analysis_checkboxes={"preprocess": _Check(True)},
        visualization_checkboxes={
            "bead_overlay": _Check(False),
            "displacement_map": _Check(False),
            "force_map": _Check(False),
            "force_cell_overlay": _Check(False),
            "sigma_xx": _Check(False),
            "sigma_yy": _Check(False),
            "normal_stress": _Check(False),
            "mesh": _Check(False),
        },
        parameter_manager=_Manager(),
        parameter_spins={"young_modulus": object()},
        parameter_combos={"mesh_algorithm": object()},
        parameter_checks={"auto_gcv": object()},
    )

    config = BatchAnalysisWidget._generate_config(fake)

    assert config["parameters"]["young_modulus"] == 9000
    assert config["parameters"]["mesh_algorithm"] == "Frontal-Del."
```

- [ ] **Step 6.2: Run tests**

```bash
pytest tests/test_batch_parameters.py -v
```

Expected: pass if current implementation already satisfies this; fail if batch still depends on duplicate widgets.

- [ ] **Step 6.3: Keep config generation independent of duplicate widgets**

In `BatchAnalysisWidget._generate_config()`, ensure the parameter block is assigned from `ParameterManager`:

```python
config["parameters"] = self.parameter_manager.get_all_parameters()
```

Keep `parameter_spins`, `parameter_combos`, and `parameter_checks` initialized as empty dictionaries for compatibility, but do not read them when generating config. Keep batch-specific controls visible.

- [ ] **Step 6.4: Run focused and full tests**

```bash
pytest tests/test_batch_parameters.py -v
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6.5: Commit**

```bash
git add tests/test_batch_parameters.py napariTFM/widgets/batch_analysis_widget.py
git commit -m "Guard batch against duplicate analysis parameter controls"
```

---

## Task 7: Final Regression And User-Facing Notes

**Files:**
- Modify: `docs/superpowers/plans/2026-05-18-tier2-project-artifact-workflow.md` only to mark completed checkboxes if executing in-place is desired.

This task is a verification and handoff checkpoint.

- [ ] **Step 7.1: Scan for stale legacy references**

```bash
rg -n "PipelineDataWidget|pipeline_data_widget|config_button|save_button|setFixedWidth\\(500\\)|parameter_panel is widget.project_section|Load Bead Stack|Load Reference Image|Load Displacement Data|Load Forces|Load Masks" napariTFM tests docs
```

Expected:

- No `PipelineDataWidget`, `pipeline_data_widget`, `config_button`, `save_button`, or `setFixedWidth(500)` in production workflow code.
- Stage-local load button text may still exist in stage widget classes if hidden and controller-owned; any visible-workflow test should assert the shell replacement.

- [ ] **Step 7.2: Run full tests**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 7.3: Attempt bounded napari smoke launch**

```bash
timeout 10s python -m napari -w napariTFM
```

Expected in the current environment may be a napari/pydantic manifest error. If that happens, record it in the final report as an environment compatibility blocker and do not claim manual GUI verification.

- [ ] **Step 7.4: Commit docs or final cleanup if changed**

If no files changed in this task, do not create an empty commit. If docs changed, commit them:

```bash
git add <docs-file>
git commit -m "Document Tier 2 artifact workflow behavior"
```

- [ ] **Step 7.5: Final report**

Report:

- commit hashes created for Tier 2
- exact `pytest tests/ -v` result
- whether bounded napari smoke launch succeeded or failed
- any remaining known gaps, especially full dependency staleness and napari manifest compatibility

---

## Self-Review

Spec coverage:

- Project/output directory: Task 1.
- Rich artifact-row state: Task 2.
- Generated output save actions and save errors: Task 3.
- Input assign/load actions beyond preprocessing: Task 4.
- Stage-local duplicate UI cleanup: Task 5.
- Batch slimdown: Task 6.
- Regression and napari smoke caveat: Task 7.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified implementation placeholders.
- Conditional commits are explicitly described and do not require empty commits.

Type and name consistency:

- `ProjectSection(..., data_manager=None)` matches Task 1 tests and shell wiring.
- `_ArtifactRow.refresh_state(state, info_text)` is introduced before `StageDataStatusPanel` uses it.
- `_save_generated_artifact(key)` is introduced before builder functions receive it.
- `artifact_rows`, `action_btn`, `output_dir_label`, and `choose_output_dir_btn` names match existing naming style.

Execution note:

- The current working tree must be checked before implementation. Do not stage unrelated `_dev/`, `.gitignore`, or `pyproject.toml` changes if they reappear.
