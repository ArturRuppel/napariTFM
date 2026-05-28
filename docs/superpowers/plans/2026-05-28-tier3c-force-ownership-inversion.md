# Force Stage Ownership Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ParameterManager` + the shell's `WorkflowParameterPanel` the sole owners of force (FTTC) parameters and make the widget the sole owner of its action buttons, by deleting `FTTCParameterPanel`, `FTTCDataPanel`, and `FTTCActionPanel` and severing the controller/widget from them.

**Architecture:** This mirrors the completed preprocessing (Tier 3a) and displacement (Tier 3b) inversions — see `2026-05-28-tier3a-preprocessing-ownership-inversion.md` and `2026-05-28-tier3b-displacement-ownership-inversion.md`. The FTTC controller currently couples to three panels for `freeze_ui`/`unfreeze_ui`/`_update_ui_state`; after inversion it only emits its existing `ui_frozen` signal and the widget owns enable/disable of its own buttons. The action buttons (`preview_btn`, `process_btn` [= "Calculate Forces"], `cancel_btn`, plus a re-homed `gcv_btn`) move from `FTTCActionPanel` (and the parameter panel's GCV control) to direct attributes on `FTTCWidget`.

**Tech Stack:** Python, qtpy/PyQt, napari, pytest. Run tests with `QT_QPA_PLATFORM=offscreen pytest`.

**Key differences from displacement (Tier 3b):**
- **The widget does NOT currently connect `ui_frozen`.** Displacement already had a `_handle_ui_freeze` slot wired; FTTC's freeze is driven entirely by the controller poking panels. This plan **adds** a `_handle_ui_freeze` slot to `FTTCWidget` and connects `controller.ui_frozen` to it (Task 3). Without this, buttons would never re-enable after a run.
- **The GCV "Auto-select" button must be re-homed.** `FTTCParameterPanel` owns `gcv_button` → `controller.calculate_optimal_regularization()` and an `auto_gcv` checkbox. The `auto_gcv` checkbox already exists in the shell's `WorkflowParameterPanel` ("Force" section). The manual GCV button is a real feature (compute optimal regularization for the current frame) and is re-homed onto the widget action row as `gcv_btn`. The old cross-control coupling (auto_gcv disables the regularization spin + GCV button) is **not** restored — that coupling was already lost when parameters moved to the shell; `gcv_btn` is simply enabled whenever displacement data is present.
- **The controller has its own `_update_ui_state`** (drives `data_panel`/`action_panel`) called inside `_handle_analysis_results`. It is deleted and its call site removed; the widget's own `_update_ui_state` (driven by signals) takes over.
- **No latent `load_active_layer` bug.** The force shell rows call `force_widget.load_result_artifact("displacement_results")`, which already exists on the widget. No delegate needed.
- **No force parameter-UI test file to delete.** A grep confirms no test references `FTTCParameterPanel`/`FTTCActionPanel`/`FTTCDataPanel`. Farneback-equivalent force coverage already lives in `test_workflow_shell.py` (`{"young_modulus", "auto_gcv"}.issubset(force_panel.parameter_controls)`).

**Pre-existing surviving contract the shell depends on (must keep working):**
- `FTTCWidget.process_btn`, `.preview_btn`, `.cancel_btn` (clickable QPushButtons) — proxied by the `_StageSection` header via `_find_stage_action_targets`.
- `FTTCWidget.load_result_artifact(key)` — called by `_build_force_specs` for the "displacement_results" input row.
- `FTTCWidget._update_ui_state()` — called by the shell on data changes (lines ~753, ~798, ~808, ~818, ~828 of `_widget.py`).
- `FTTCController.ui_frozen` signal — now drives the widget's freeze handling (newly connected in Task 3).
- `FTTCWidget.force_calculated` signal — connected by the shell.

**Known env flake (do NOT chase):** `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` intermittently SIGSEGVs in its spawned subprocess (napari/pydantic plugin-manifest race). It is not caused by this work.

---

## File Structure

- `napariTFM/widgets/fttc_widget.py` (MODIFY) — delete 3 panel classes; sever controller; move buttons onto widget; re-home GCV button; add `_handle_ui_freeze` + `ui_frozen` connection.
- `napariTFM/widgets/_widget.py` (MODIFY) — simplify force `action_targets`; drop force from `_hide_redundant_stage_shell_controls`.
- `tests/test_force_ownership.py` (CREATE) — contract tests pinning the inverted structure.
- `tests/test_workflow_shell.py` (MODIFY) — drop force from the hide-list assertion loop.

---

### Task 1: Pin the inverted contract with failing tests

**Files:**
- Create: `tests/test_force_ownership.py`

- [ ] **Step 1: Write the failing contract tests**

Model on `tests/test_displacement_ownership.py`. The FTTC controller constructor (after Task 2) will be `FTTCController(viewer, data_manager, parameter_manager, visualization_manager)` — note `data_panel` is dropped.

```python
import pytest
from qtpy.QtWidgets import QApplication

import napariTFM.widgets.fttc_widget as fw


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeDims:
    class _Events:
        class _Step:
            def connect(self, *_):
                return None
        current_step = _Step()
    events = _Events()
    current_step = (0,)


class _FakeLayersSelection:
    active = None

    class _Events:
        class _Active:
            def connect(self, *_):
                return None
        active = _Active()
    events = _Events()


class _FakeLayers:
    selection = _FakeLayersSelection()

    def __iter__(self):
        return iter(())


class _FakeViewer:
    dims = _FakeDims()
    layers = _FakeLayers()


class _FakeDataManager:
    displacement_results = None
    force_results = None


class _FakeParameterManager:
    def __init__(self):
        from qtpy.QtCore import QObject, Signal

        class _PM(QObject):
            parameter_changed = Signal(str, object)
            parameters_reset = Signal(object)

        self._pm = _PM()
        self.parameter_changed = self._pm.parameter_changed
        self.parameters_reset = self._pm.parameters_reset

    def get_fttc_parameters(self):
        return object()


def test_parameter_panel_class_is_removed():
    assert not hasattr(fw, "FTTCParameterPanel")


def test_data_panel_class_is_removed():
    assert not hasattr(fw, "FTTCDataPanel")


def test_action_panel_class_is_removed():
    assert not hasattr(fw, "FTTCActionPanel")


def test_controller_has_no_panel_attributes(app):
    controller = fw.FTTCController(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )
    assert not hasattr(controller, "parameter_panel")
    assert not hasattr(controller, "data_panel")
    assert not hasattr(controller, "action_panel")
    assert not hasattr(controller, "set_panels")


def test_controller_freeze_emits_signal_without_panels(app):
    controller = fw.FTTCController(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )
    seen = []
    controller.ui_frozen.connect(seen.append)
    controller.freeze_ui()
    controller.unfreeze_ui()
    assert seen == [True, False]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_force_ownership.py -v`
Expected: FAIL — `test_*_class_is_removed` fail (classes still present); `test_controller_has_no_panel_attributes` fails (`set_panels`/`data_panel`/`action_panel` present); `test_controller_freeze_emits_signal_without_panels` fails (constructor still requires the 5th `data_panel` arg → `TypeError`).

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_force_ownership.py
git commit -m "Add failing force ownership contract tests"
```

---

### Task 2: Sever the controller from its panels

**Files:**
- Modify: `napariTFM/widgets/fttc_widget.py` — `FTTCController.__init__`, `set_panels`, `_update_ui_state`, `cancel_operation`, `_handle_analysis_results`, `freeze_ui`, `unfreeze_ui`; and `FTTCWidget.__init__` (controller/panel construction).

- [ ] **Step 1: Drop the `data_panel` constructor param and panel attributes**

In `FTTCController.__init__` (currently around lines 439–455), replace:

```python
    def __init__(self, viewer, data_manager, parameter_manager,
                 visualization_manager, data_panel):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.data_panel = data_panel
        self.active_workers = []

        # Initialize panel attributes
        self.parameter_panel = None
        self.action_panel = None

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)
```

with:

```python
    def __init__(self, viewer, data_manager, parameter_manager,
                 visualization_manager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)
```

> The `_on_parameter_changed`/`_on_parameters_reset` controller stubs are `pass` no-ops; they (and these connections) are removed in the Task 7 cleanup. Leaving them through inversion keeps this task focused.

- [ ] **Step 2: Delete `set_panels`**

Remove the entire method (currently around lines 457–462):

```python
    def set_panels(self, parameter_panel, action_panel):
        """Set the parameter and action panels."""
        self.parameter_panel = parameter_panel
        self.action_panel = action_panel
        self.parameter_panel.set_controller(self)
        self._update_ui_state()
```

- [ ] **Step 3: Delete the controller's `_update_ui_state`**

Remove the entire method (currently around lines 578–593):

```python
    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update data panel
        if self.data_panel:
            self.data_panel.update_data_status()

        # Get current data state
        has_displacement = self.data_manager.displacement_results is not None
        has_results = self.data_manager.force_results is not None

        # Update action panel button states
        if self.action_panel:
            self.action_panel.update_button_states(
                has_displacement=has_displacement,
                has_results=has_results
            )
```

- [ ] **Step 4: Remove the `_update_ui_state()` call inside `_handle_analysis_results`**

In `_handle_analysis_results` (currently around lines 752–754), remove the `self._update_ui_state()` call. Replace:

```python
            self.progress_updated.emit(100, "Analysis completed successfully")
            self.analysis_completed.emit(result)
            self._update_ui_state()
```

with:

```python
            self.progress_updated.emit(100, "Analysis completed successfully")
            self.analysis_completed.emit(result)
```

- [ ] **Step 5: Drop the action-panel block from `cancel_operation`**

In `cancel_operation` (currently around lines 595–609), remove the trailing action-panel block. Replace:

```python
        self.active_workers.clear()
        self.progress_updated.emit(0, "Operation cancelled")
        self.unfreeze_ui()
        # Ensure cancel button stays enabled
        if self.action_panel:
            self.action_panel.cancel_btn.setEnabled(True)
```

with:

```python
        self.active_workers.clear()
        self.progress_updated.emit(0, "Operation cancelled")
        self.unfreeze_ui()
```

- [ ] **Step 6: Rewrite `freeze_ui`/`unfreeze_ui` to emit the signal only**

Replace the State Management block (currently around lines 804–829):

```python
    def freeze_ui(self):
        """Disable all interactive UI elements."""
        if self.data_panel:
            self.data_panel.freeze_ui(True)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(True)
        if self.action_panel:
            self.action_panel.freeze_ui(True)
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Re-enable UI elements and refresh state."""
        if self.data_panel:
            self.data_panel.freeze_ui(False)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(False)
        if self.action_panel:
            # Get current state
            has_displacement = self.data_manager.displacement_results is not None
            has_results = self.data_manager.force_results is not None
            # Update button states instead of just unfreezing
            self.action_panel.update_button_states(
                has_displacement=has_displacement,
                has_results=has_results
            )
        self.ui_frozen.emit(False)
```

with:

```python
    def freeze_ui(self):
        """Signal the owning widget to disable interactive controls."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal the owning widget to re-enable controls."""
        self.ui_frozen.emit(False)
```

- [ ] **Step 7: Update the controller/panel construction in the widget**

In `FTTCWidget.__init__` (currently around lines 846–866), the widget builds the parameter panel, constructs the controller with `data_panel=None`, builds the action panel, and calls `set_panels(...)`. Replace:

```python
        # Store managers
        self.parameter_manager = parameter_manager

        # Initialize panels
        self.data_panel = None
        self.parameter_panel = FTTCParameterPanel(parameter_manager)

        # Initialize controller
        self.controller = FTTCController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
            data_panel=None
        )

        # Initialize action panel with controller
        self.action_panel = FTTCActionPanel(self.controller)

        # Connect controller
        self.controller.set_panels(self.parameter_panel, self.action_panel)
```

with (action buttons are built in `_create_action_row` in Task 3; do not construct any panels here):

```python
        # Store managers
        self.parameter_manager = parameter_manager

        # Initialize controller
        self.controller = FTTCController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
        )
```

> NOTE: This leaves `_create_content_container` referencing `self.parameter_panel`/`self.action_panel` (lines ~901, ~903), `_connect_signals` referencing `self.parameter_panel` (line ~935), and `_update_ui_state` referencing `self.action_panel` (line ~950) temporarily broken. Task 3 rewrites those; Task 4 deletes the classes. The suite is expected to be RED between Task 2 and Task 4 — that is acceptable within this plan; commit each task after its local steps and rely on the full-suite run in Task 6 for green.

- [ ] **Step 8: Run the ownership contract tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_force_ownership.py::test_controller_has_no_panel_attributes tests/test_force_ownership.py::test_controller_freeze_emits_signal_without_panels -v`
Expected: PASS (the two controller tests). Class-removal tests still FAIL until Task 4.

- [ ] **Step 9: Commit**

```bash
git add napariTFM/widgets/fttc_widget.py
git commit -m "Sever FTTC controller from parameter/data/action panels"
```

---

### Task 3: Move action buttons onto the widget, re-home GCV, and wire the freeze signal

**Files:**
- Modify: `napariTFM/widgets/fttc_widget.py` — `FTTCWidget._create_content_container`, `_connect_signals`, `_update_ui_state`; add `_create_action_row`, `_handle_ui_freeze`; remove `_on_parameters_reset`; touch `_on_analysis_completed`/`_on_analysis_failed`.

- [ ] **Step 1: Build the action buttons inside the widget's content container**

In `_create_content_container` (currently around lines 887–909), the layout adds `self.parameter_panel` and `self.action_panel`. Replace the panel-adding block:

```python
        # Add panels
        layout.addWidget(self.parameter_panel)
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self.action_panel)
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_status_frame())
```

with a call to build the action row directly (no parameter panel — the shell owns parameters):

```python
        layout.addWidget(self._create_action_row())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_status_frame())
```

- [ ] **Step 2: Add the `_create_action_row` builder**

Add this method to `FTTCWidget` (place it next to `_create_status_frame`). It reproduces `FTTCActionPanel`'s buttons as widget-owned attributes, using `process_btn` as the stable name for "Calculate Forces" (matching the shell's `run=["process_btn"]` and the preprocessing/displacement convention), and re-homes the GCV button as `gcv_btn`:

```python
    def _create_action_row(self) -> QWidget:
        """Build widget-owned action buttons (run/preview/cancel proxied by the stage header)."""
        container = QWidget()
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.preview_btn.setToolTip(
            "Calculate and visualize forces for the current frame only"
        )
        self.process_btn = QPushButton("Calculate Forces")
        self.process_btn.setToolTip(
            "Calculate forces for all frames in the dataset"
        )
        row.addWidget(self.preview_btn)
        row.addWidget(self.process_btn)
        layout.addLayout(row)

        self.gcv_btn = QPushButton("Auto-select Regularization (GCV)")
        self.gcv_btn.setToolTip(
            "Calculate the optimal regularization parameter for the current frame\n"
            "using Generalized Cross-Validation"
        )
        layout.addWidget(self.gcv_btn)

        self.cancel_btn = QPushButton("Cancel Operation")
        self.cancel_btn.setToolTip("Cancel the current operation")
        layout.addWidget(self.cancel_btn)

        container.setLayout(layout)
        return container
```

- [ ] **Step 3: Wire the buttons, connect the freeze signal, and drop parameter-panel wiring**

In `_connect_signals` (currently around lines 926–938), replace:

```python
    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)

        # Connect parameter panel signals
        self.parameter_panel.parameters_reset.connect(self._on_parameters_reset)

        # Connect to layer selection changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)
```

with:

```python
    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Wire widget-owned action buttons to controller operations
        self.preview_btn.clicked.connect(self.controller.preview_force)
        self.process_btn.clicked.connect(self.controller.calculate_forces)
        self.gcv_btn.clicked.connect(self.controller.calculate_optimal_regularization)
        self.cancel_btn.clicked.connect(self.controller.cancel_operation)

        # Connect to layer selection changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)
```

- [ ] **Step 4: Rewrite `_update_ui_state` to drive widget-owned buttons**

Replace (currently around lines 945–953):

```python
    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update action panel
        has_displacement = self.data_manager.displacement_results is not None
        has_results = self.data_manager.force_results is not None
        self.action_panel.update_button_states(
            has_displacement=has_displacement,
            has_results=has_results
        )
```

with:

```python
    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_displacement = self.data_manager.displacement_results is not None

        self.preview_btn.setEnabled(has_displacement)
        self.process_btn.setEnabled(has_displacement)
        self.gcv_btn.setEnabled(has_displacement)
        self.cancel_btn.setEnabled(True)
```

- [ ] **Step 5: Add `_handle_ui_freeze`**

Add this method to `FTTCWidget` (place it next to `_update_ui_state`). This is new — FTTC previously had no freeze slot:

```python
    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        self.preview_btn.setEnabled(not frozen)
        self.process_btn.setEnabled(not frozen)
        self.gcv_btn.setEnabled(not frozen)
        # Cancel button always enabled
        self.cancel_btn.setEnabled(True)
```

- [ ] **Step 6: Remove the dead `_on_parameters_reset` widget handler**

It was only connected to the deleted parameter panel. Remove the widget method (currently around lines 955–957):

```python
    def _on_parameters_reset(self):
        """Handle parameter reset."""
        self._update_status(0, "Force parameters reset to default values")
```

- [ ] **Step 7: Refresh button state after analysis completes/fails**

`_on_analysis_completed` (currently around lines 959–969) and `_on_analysis_failed` (currently around lines 971–973) should refresh the widget buttons (the controller no longer drives UI). Add `self._update_ui_state()` to each.

In `_on_analysis_completed`, replace:

```python
        # Emit results
        self.force_calculated.emit(results)
```

with:

```python
        # Emit results
        self.force_calculated.emit(results)
        self._update_ui_state()
```

In `_on_analysis_failed`, replace:

```python
    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_status(0, f"Error: {error_msg}")
```

with:

```python
    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_status(0, f"Error: {error_msg}")
        self._update_ui_state()
```

- [ ] **Step 8: Run the ownership tests + a construction smoke check**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_force_ownership.py -v`
Expected: controller tests PASS; class-removal tests still FAIL (classes deleted in Task 4).

Then a construction smoke check:

```bash
QT_QPA_PLATFORM=offscreen python -c "
import napari
from napariTFM.widgets.fttc_widget import FTTCWidget
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager
v = napari.Viewer(show=False)
dm = DataManager(); pm = ParameterManager(); vm = VisualizationManager(v, dm)
w = FTTCWidget(v, dm, pm, vm)
assert hasattr(w, 'process_btn') and hasattr(w, 'preview_btn') and hasattr(w, 'cancel_btn')
assert hasattr(w, 'gcv_btn')
assert hasattr(w, 'load_result_artifact')
print('SMOKE OK')
"
```
Expected: `SMOKE OK`.

- [ ] **Step 9: Commit**

```bash
git add napariTFM/widgets/fttc_widget.py
git commit -m "Move FTTC action buttons onto widget and wire freeze signal"
```

---

### Task 4: Delete the three panel classes

**Files:**
- Modify: `napariTFM/widgets/fttc_widget.py` — delete `FTTCDataPanel`, `FTTCParameterPanel`, `FTTCActionPanel`; clean unused imports.

- [ ] **Step 1: Delete the class definitions**

Delete the entire bodies of:
- `class FTTCDataPanel(QWidget):` (currently lines ~21–81)
- `class FTTCParameterPanel(QWidget):` (currently lines ~84–365)
- `class FTTCActionPanel(QWidget):` (currently lines ~368–426)

Leave `class FTTCController(QObject):` and `class FTTCWidget(BaseAnalysisWidget):` intact.

- [ ] **Step 2: Remove imports left unused after the deletions**

After deleting the panels, several imports become unused. Verify which symbols are still referenced before removing — only remove a symbol whose count drops to the single occurrence on the import line itself:

```bash
for sym in ParameterCategory Qt QGroupBox QSpinBox QDoubleSpinBox QCheckBox Path QScrollArea QLabel QSpacerItem; do
  echo "$sym: $(grep -c "\b$sym\b" napariTFM/widgets/fttc_widget.py)"; done
```

Expected outcome: `ParameterCategory`, `Qt`, `QGroupBox`, `QSpinBox`, `QDoubleSpinBox`, `QCheckBox` are used only by the deleted panels → remove them. `QScrollArea`, `QLabel`, `QSpacerItem` are still used by `FTTCWidget._create_content_container`/`_create_status_frame` → keep. Verify `Path` (likely unused — remove if count is 1).

Concretely, the `ParameterCategory` import becomes `from napariTFM.utilities.parameter_manager import ParameterManager` (drop `ParameterCategory`). Trim the two `qtpy.QtWidgets` import lines to only the symbols still referenced; the `from qtpy.QtCore import Signal, Qt` line becomes `from qtpy.QtCore import Signal` (drop `Qt`) — verify `Signal` is still used by `FTTCController`/`FTTCWidget` (it is: `progress_updated`, `force_calculated`, etc.).

- [ ] **Step 3: Run the ownership contract tests (now fully green)**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_force_ownership.py -v`
Expected: ALL PASS — the three `*_class_is_removed` tests now pass.

- [ ] **Step 4: Commit**

```bash
git add napariTFM/widgets/fttc_widget.py
git commit -m "Delete FTTC data, parameter, and action panel classes"
```

---

### Task 5: Simplify the shell wiring

**Files:**
- Modify: `napariTFM/widgets/_widget.py` — force `action_targets` (lines ~492–497); `_hide_redundant_stage_shell_controls` (line ~585).
- Modify: `tests/test_workflow_shell.py` — hide-list assertion loop (lines ~635–640).

- [ ] **Step 1: Collapse the force action-target fallbacks**

In `_create_stage_sections`, the force `_StageSection` currently uses dual paths that prefer the deleted `action_panel.*`. Replace (lines ~492–497):

```python
                action_targets=self._find_stage_action_targets(
                    self.force_widget,
                    run=["action_panel.calculate_btn", "calculate_btn"],
                    preview=["action_panel.preview_btn", "preview_btn"],
                    cancel=["action_panel.cancel_btn", "cancel_btn"],
                ),
```

with:

```python
                action_targets=self._find_stage_action_targets(
                    self.force_widget,
                    run=["process_btn"],
                    preview=["preview_btn"],
                    cancel=["cancel_btn"],
                ),
```

- [ ] **Step 2: Drop force from the redundant-shell-controls hide list**

`_hide_redundant_stage_shell_controls` (line ~585) hides `data_panel`/`action_panel` for stages that still have them. Force no longer does. Replace:

```python
        for widget in [self.force_widget, self.msm_widget]:
```

with:

```python
        for widget in [self.msm_widget]:
```

> Leave `_hide_embedded_parameter_panels` unchanged: it iterates with `getattr(widget, "parameter_panel", None)` and tolerates the now-missing attribute (preprocessing and displacement are still listed there for the same reason).

- [ ] **Step 3: Update the hide-list assertion test**

In `tests/test_workflow_shell.py::test_main_widget_hides_stage_local_data_and_action_panels_after_shell_wiring` (lines ~635–640), the loop iterates `[force, msm]`. Since force is no longer hidden by the shell (it has no such panels), drop it. Replace:

```python
    for stage_widget in [
        widget.force_widget,
        widget.msm_widget,
    ]:
```

with:

```python
    for stage_widget in [
        widget.msm_widget,
    ]:
```

- [ ] **Step 4: Run the shell tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py -v`
Expected: ALL PASS. In particular `test_main_widget_stage_headers_wire_existing_stage_actions` (uses the stub's `process_btn`), the force input-row test (`widget.force_widget.loaded_files == ["displacement_results"]`), and the edited hide-list test pass.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Simplify force shell wiring after panel inversion"
```

---

### Task 6: Run the full suite

**Files:** none (verification task).

- [ ] **Step 1: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen pytest -q`
Expected: green except the known `test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` env flake. Re-run that file in isolation to confirm it passes alone: `QT_QPA_PLATFORM=offscreen pytest tests/test_napari_compatibility.py -v`.

- [ ] **Step 2: Commit (only if anything changed)**

If Step 1 surfaced a fix, commit it. Otherwise there is nothing to commit — proceed to Task 7.

---

### Task 7: Remove confirmed-dead controller code

**Files:**
- Modify: `napariTFM/widgets/fttc_widget.py` — `FTTCController`: remove `_sync_parameters_with_results`, the `_on_parameter_changed`/`_on_parameters_reset` stubs and their `__init__` connections; clean newly-unused imports.

- [ ] **Step 1: Confirm `_sync_parameters_with_results` has no callers**

```bash
grep -n "_sync_parameters_with_results" napariTFM/
```
Expected: a single match — the definition in `fttc_widget.py`. No call sites → safe to delete.

- [ ] **Step 2: Delete `_sync_parameters_with_results`**

Remove the entire method (currently around lines 556–576):

```python
    def _sync_parameters_with_results(self, result):
        """Sync parameters from loaded results."""
        if not hasattr(result, 'parameters'):
            return

        params = result.parameters
        for param_name, value in vars(params).items():
            if param_name != '_sa_instance_state':  # Skip SQLAlchemy state
                if param_name == 'young_modulus':
                    # Store in Pa, UI will convert to kPa
                    self.parameter_manager.set_parameter(param_name, value)
                elif param_name == 'regularization':
                    # Store actual value, UI will convert to log10
                    self.parameter_manager.set_parameter(param_name, value)
                elif param_name == 'gel_height':
                    # Handle infinity case
                    if value == 0:
                        value = float('inf')
                    self.parameter_manager.set_parameter(param_name, value)
                else:
                    self.parameter_manager.set_parameter(param_name, value)
```

- [ ] **Step 3: Delete the `_on_parameter_changed`/`_on_parameters_reset` stubs and their connections**

These are `pass` no-ops connected to the parameter manager. Remove the two methods (currently around lines 796–802):

```python
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes from parameter manager."""
        pass

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        pass
```

And remove their connections in `FTTCController.__init__` (the two lines added/kept in Task 2):

```python
        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)
```

(Delete these three lines — the comment and both `.connect(...)` lines.)

- [ ] **Step 4: Remove the now-unused `Any` import if applicable**

`Any` was used by the deleted `_on_parameter_changed` signature and `FTTCDataPanel`/`FTTCParameterPanel`. Verify:

```bash
grep -n "\bAny\b" napariTFM/widgets/fttc_widget.py
```
If the only remaining match is the `from typing import Any` line, remove that import.

- [ ] **Step 5: Run the ownership tests + full suite**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_force_ownership.py -q && QT_QPA_PLATFORM=offscreen pytest -q`
Expected: green except the known env flake.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/fttc_widget.py
git commit -m "Remove dead FTTC controller parameter-sync scaffolding"
```

---

### Task 8: Manual smoke test (human-in-the-loop)

**Files:** none.

- [ ] **Step 1: Launch napari and exercise the force stage**

```bash
napari
```
Open Plugins → napariTFM → napariTFM. Then verify:
- The **Force Analysis** section's "Parameters" sub-section shows the force parameters (single editor — no duplicate panel), including the `auto_gcv` checkbox.
- With displacement results present (run the displacement stage or load a displacement field via the force input row), **Preview Current Frame**, **Calculate Forces**, and **Auto-select Regularization (GCV)** become enabled.
- **Auto-select Regularization (GCV)** computes and populates the regularization parameter for the current frame.
- **Preview Current Frame** computes and visualizes single-frame forces (Force Vectors + Force Magnitude layers).
- **Calculate Forces** runs to completion; controls disable during the run and re-enable after (this exercises the newly-wired `ui_frozen` → `_handle_ui_freeze`); **Cancel Operation** aborts mid-run.
- The stage header's run/preview/cancel proxy buttons mirror the in-body buttons' enabled state.

---

## Self-Review

**Spec coverage:**
- Delete `FTTCParameterPanel` → Task 4. Sever parameter ownership to shell → Tasks 2–4 (widget no longer builds/wires it; shell `WorkflowParameterPanel("Force")` already mounted with `young_modulus`/`auto_gcv`/etc.).
- Delete `FTTCDataPanel` (dead — never instantiated, confirmed by grep) → Task 4.
- Delete `FTTCActionPanel`, re-home buttons → Tasks 3–4.
- Re-home the GCV button (feature preserved) → Task 3 Step 2 + wiring in Step 3.
- Add `ui_frozen` → `_handle_ui_freeze` wiring (new for FTTC) → Task 3 Steps 3 & 5.
- Controller `freeze_ui`/`unfreeze_ui` emit-only; delete controller `_update_ui_state` + call site → Task 2.
- Shell `action_targets` + hide-list + test updates → Task 5.
- Dead-code cleanup (`_sync_parameters_with_results`, param-manager stubs) → Task 7.
- Test contract → Task 1; full suite → Task 6.

**Type/name consistency:** Run "Calculate Forces" button is named `process_btn` (not `calculate_btn`) to match the shell's `run=["process_btn"]` and the preprocessing/displacement convention; the controller method it calls remains `calculate_forces`. `preview_btn`/`cancel_btn` keep their names; the re-homed GCV button is `gcv_btn` → `controller.calculate_optimal_regularization` (an existing controller method, unchanged). The controller constructor loses its 5th positional `data_panel` arg in Task 2 and Task 1's tests assume the 4-arg form. The widget's `_update_ui_state` and `_handle_ui_freeze` both gate `preview_btn`/`process_btn`/`gcv_btn`; `cancel_btn` is always enabled.

**Intentional RED window:** The suite is knowingly red between Task 2 and Task 4 (widget references panels deleted across staggered tasks). This is called out in Task 2 Step 7; correctness is gated by the construction smoke test in Task 3 Step 8 and the full-suite run in Task 6. If executing with subagents, the spec reviewer should treat per-task local test targets (named in each task) as the green bar, not the full suite, until Task 6.

**Known follow-ons (separate plans):** stress (MSM) inversion is the last per-stage inversion — it deletes its `*ActionPanel`/`*ParameterPanel` and re-homes the same coupling (note: MSM has multiple preview buttons — `preview_frame_btn`/`preview_mesh_btn` — and an `analyze_btn`, so its action surface is richer). After stress, drop the `_find_stage_action_targets` reflection entirely and give each stage a flat action surface. Then disk-derived status, theming, config persistence.
```