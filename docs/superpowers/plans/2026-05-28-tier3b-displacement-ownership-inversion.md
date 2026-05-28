# Displacement Stage Ownership Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ParameterManager` + the shell's `WorkflowParameterPanel` the sole owners of displacement parameters and make the widget the sole owner of its action buttons, by deleting `DisplacementParameterPanel`, `DisplacementDataPanel`, and `DisplacementActionPanel` and severing the controller/widget from them.

**Architecture:** This mirrors the completed preprocessing inversion (Tier 3a, see `2026-05-28-tier3a-preprocessing-ownership-inversion.md`). The displacement controller currently couples to three panels for `freeze_ui`/`unfreeze_ui`; after inversion it only emits its existing `ui_frozen` signal and the widget owns enable/disable of its own buttons. The three action buttons (`preview_btn`, `process_btn` [= "Calculate All Frames"], `cancel_btn`) move from `DisplacementActionPanel` to direct attributes on `DisplacementAnalysisWidget`, exactly like preprocessing exposes `process_btn`/`cancel_btn`. A latent bug is fixed: the shell calls `displacement_widget.load_active_layer(role)` but that method only exists on the controller — a thin widget delegate is added.

**Tech Stack:** Python, qtpy/PyQt, napari, pytest. Run tests with `QT_QPA_PLATFORM=offscreen pytest`.

**Key differences from preprocessing (Tier 3a):**
- Preprocessing had no `ActionPanel`; displacement does — its `preview_btn`/`calculate_btn`/`cancel_btn` are proxied by the stage header via `_find_stage_action_targets`, so they must survive the panel deletion as widget-owned buttons.
- Displacement preview is a one-shot **button** (`controller.preview_displacement`), not a live checkbox like preprocessing's `preview_check`. No preview-on-toggle wiring is needed.
- `load_active_layer` lives on the controller and the shell calls it on the *widget* — add a widget delegate (fixes a latent `AttributeError`).

**Pre-existing surviving contract the shell depends on (must keep working):**
- `DisplacementAnalysisWidget.process_btn`, `.preview_btn`, `.cancel_btn` (clickable QPushButtons) — proxied by `_StageSection` header.
- `DisplacementAnalysisWidget.load_active_layer(role)` — called by `_build_displacement_specs` for the "reference"/"beads" input rows.
- `DisplacementAnalysisWidget._update_ui_state()` — called by the shell on data changes.
- `DisplacementController.ui_frozen` signal — drives the widget's freeze handling.
- `DisplacementAnalysisWidget.displacement_calculated` signal — connected by the shell.

**Known env flake (do NOT chase):** `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` intermittently SIGSEGVs in its spawned subprocess (napari/pydantic plugin-manifest race). It is not caused by this work.

---

## File Structure

- `napariTFM/widgets/displacement_analysis_widget.py` (MODIFY) — delete 3 panel classes; sever controller; move buttons onto widget; add `load_active_layer` delegate.
- `napariTFM/widgets/_widget.py` (MODIFY) — simplify displacement `action_targets`; drop displacement from `_hide_redundant_stage_shell_controls`.
- `tests/test_displacement_ownership.py` (CREATE) — contract tests pinning the inverted structure.
- `tests/test_displacement_parameter_ui.py` (DELETE) — tests the deleted `DisplacementParameterPanel`; farneback coverage already lives in `test_workflow_shell.py`.
- `tests/test_workflow_shell.py` (MODIFY) — drop displacement from the hide-list assertion loop.

---

### Task 1: Pin the inverted contract with failing tests

**Files:**
- Create: `tests/test_displacement_ownership.py`

- [ ] **Step 1: Write the failing contract tests**

Model on `tests/test_preprocessing_ownership.py`. The displacement controller constructor (after Task 2) will be `DisplacementController(viewer, data_manager, parameter_manager, visualization_manager)` — note `data_panel` is dropped.

```python
import pytest
from qtpy.QtWidgets import QApplication, QPushButton

import napariTFM.widgets.displacement_analysis_widget as dw


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
    preprocessed_bead_stack = None
    preprocessed_reference = None
    displacement_results = None


class _FakeParameterManager:
    def __init__(self):
        from qtpy.QtCore import QObject, Signal

        class _PM(QObject):
            parameter_changed = Signal(str, object)
            parameters_reset = Signal(object)

        self._pm = _PM()
        self.parameter_changed = self._pm.parameter_changed
        self.parameters_reset = self._pm.parameters_reset

    def get_displacement_parameters(self):
        return object()


def test_parameter_panel_class_is_removed():
    assert not hasattr(dw, "DisplacementParameterPanel")


def test_data_panel_class_is_removed():
    assert not hasattr(dw, "DisplacementDataPanel")


def test_action_panel_class_is_removed():
    assert not hasattr(dw, "DisplacementActionPanel")


def test_controller_has_no_panel_attributes(app):
    controller = dw.DisplacementController(
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
    controller = dw.DisplacementController(
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

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_displacement_ownership.py -v`
Expected: FAIL — `test_*_class_is_removed` fail (classes still present); `test_controller_*` fail (constructor still requires `data_panel`, and `set_panels`/panel attrs still present).

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_displacement_ownership.py
git commit -m "Add failing displacement ownership contract tests"
```

---

### Task 2: Sever the controller from its panels

**Files:**
- Modify: `napariTFM/widgets/displacement_analysis_widget.py` — `DisplacementController.__init__`, `set_panels`, `freeze_ui`, `unfreeze_ui`; and `DisplacementAnalysisWidget.__init__` (controller construction).

- [ ] **Step 1: Drop the `data_panel` constructor param and panel attributes**

In `DisplacementController.__init__` (currently around lines 436–453), replace:

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
        self.preview_enabled = False

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
        self.preview_enabled = False

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)
```

- [ ] **Step 2: Delete `set_panels`**

Remove the entire method (currently around lines 455–458):

```python
    def set_panels(self, parameter_panel, action_panel):
        """Set the parameter and action panels."""
        self.parameter_panel = parameter_panel
        self.action_panel = action_panel
```

- [ ] **Step 3: Rewrite `freeze_ui`/`unfreeze_ui` to emit the signal only**

Replace the State Management block (currently around lines 770–788):

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
            self.action_panel.freeze_ui(False)
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

- [ ] **Step 4: Update the controller construction in the widget**

In `DisplacementAnalysisWidget.__init__` (currently around lines 814–827), the controller is built with `data_panel=None` and then `set_panels(...)` is called. Replace:

```python
        # Initialize controller
        self.controller = DisplacementController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
            data_panel=None
        )

        # Initialize action panel with controller
        self.action_panel = DisplacementActionPanel(self.controller)

        # Set controller in panels
        self.controller.set_panels(self.parameter_panel, self.action_panel)
```

with (action buttons are built in `_setup_ui` in Task 3; do not construct any panels here):

```python
        # Initialize controller
        self.controller = DisplacementController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
        )
```

Also remove the now-orphaned panel construction lines just above it (currently around lines 810–812):

```python
        # Initialize panels
        self.parameter_panel = DisplacementParameterPanel(parameter_manager)
        self.data_panel = None
```

> NOTE: This leaves `_setup_ui` referencing `self.parameter_panel`/`self.action_panel` (lines ~871, ~873) and `_connect_signals` referencing `self.parameter_panel.*` (lines ~910–911) temporarily broken. Task 3 rewrites `_setup_ui`/`_handle_ui_freeze`/`_update_ui_state`/results handlers, and Task 4 removes the parameter-panel wiring. The suite is expected to be RED between Task 2 and Task 4 — that is acceptable within this plan; only commit each task after the steps below, and rely on the final full-suite run in Task 6 for green.

- [ ] **Step 5: Run the ownership contract tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_displacement_ownership.py::test_controller_has_no_panel_attributes tests/test_displacement_ownership.py::test_controller_freeze_emits_signal_without_panels -v`
Expected: PASS (the two controller tests). Class-removal tests still FAIL until Task 4.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/displacement_analysis_widget.py
git commit -m "Sever displacement controller from parameter/data/action panels"
```

---

### Task 3: Move action buttons onto the widget and add the load delegate

**Files:**
- Modify: `napariTFM/widgets/displacement_analysis_widget.py` — `DisplacementAnalysisWidget._setup_ui`, `_create_content_container`, `_connect_signals`, `_handle_ui_freeze`, `_update_ui_state`, `_on_analysis_completed`, `_on_analysis_failed`; add `load_active_layer`.

- [ ] **Step 1: Build the action buttons inside the widget's content container**

In `_create_content_container` (currently around lines 856–879), the layout adds `self.parameter_panel` and `self.action_panel`. Replace the panel-adding block:

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

Add this method to `DisplacementAnalysisWidget` (place it next to `_create_status_frame`, in the UI Creation region). It reproduces `DisplacementActionPanel`'s buttons as widget-owned attributes, using `process_btn` as the stable name for "Calculate All Frames" (matching the shell's `run=["process_btn"]` and the preprocessing convention):

```python
    def _create_action_row(self) -> QWidget:
        """Build widget-owned action buttons (proxied by the stage header)."""
        container = QWidget()
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.preview_btn.setToolTip(
            "Calculate and visualize displacement for the current frame only"
        )
        self.process_btn = QPushButton("Calculate All Frames")
        self.process_btn.setToolTip(
            "Calculate displacements for all frames in the dataset"
        )
        self.preview_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.process_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        row.addWidget(self.preview_btn)
        row.addWidget(self.process_btn)
        layout.addLayout(row)

        self.cancel_btn = QPushButton("Cancel Operation")
        self.cancel_btn.setToolTip("Cancel the current operation")
        layout.addWidget(self.cancel_btn)

        container.setLayout(layout)
        return container
```

- [ ] **Step 3: Add the `load_active_layer` delegate (fixes the latent shell bug)**

Add to `DisplacementAnalysisWidget` (in the Data/Results region):

```python
    def load_active_layer(self, data_type: str):
        """Delegate input-layer loading to the controller (called by the shell)."""
        self.controller.load_active_layer(data_type)
```

- [ ] **Step 4: Wire the buttons and drop parameter-panel signal wiring**

In `_connect_signals` (currently around lines 900–914), replace:

```python
        # Connect parameter panel signals
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_panel.parameters_reset.connect(self._on_parameters_reset)

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)
```

with:

```python
        # Wire widget-owned action buttons to controller operations
        self.preview_btn.clicked.connect(self.controller.preview_displacement)
        self.process_btn.clicked.connect(self.controller.calculate_all_frames)
        self.cancel_btn.clicked.connect(self.controller.cancel_operation)

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)
```

- [ ] **Step 5: Rewrite `_update_ui_state` to drive widget-owned buttons**

Replace (currently around lines 932–947):

```python
    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Get current data state
        has_reference = self.data_manager.preprocessed_reference is not None
        has_beads = self.data_manager.preprocessed_bead_stack is not None
        has_results = self.data_manager.displacement_results is not None  # Full results, not preview

        # Update action panel button states based on data availability
        if hasattr(self, 'action_panel'):
            # Analysis buttons require both reference and beads
            can_analyze = has_reference and has_beads
            self.action_panel.preview_btn.setEnabled(can_analyze)
            self.action_panel.calculate_btn.setEnabled(can_analyze)

            # Cancel is always enabled
            self.action_panel.cancel_btn.setEnabled(True)
```

with:

```python
    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_reference = self.data_manager.preprocessed_reference is not None
        has_beads = self.data_manager.preprocessed_bead_stack is not None

        can_analyze = has_reference and has_beads
        self.preview_btn.setEnabled(can_analyze)
        self.process_btn.setEnabled(can_analyze)
        self.cancel_btn.setEnabled(True)
```

- [ ] **Step 6: Rewrite `_handle_ui_freeze` to drive widget-owned buttons**

Replace (currently around lines 949–959):

```python
    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        if hasattr(self, 'parameter_panel'):
            self.parameter_panel.freeze_ui(frozen)

        if hasattr(self, 'action_panel'):
            # During processing, disable all buttons except cancel
            self.action_panel.preview_btn.setEnabled(not frozen)
            self.action_panel.calculate_btn.setEnabled(not frozen)
            # Cancel button always enabled
            self.action_panel.cancel_btn.setEnabled(True)
```

with:

```python
    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        self.preview_btn.setEnabled(not frozen)
        self.process_btn.setEnabled(not frozen)
        # Cancel button always enabled
        self.cancel_btn.setEnabled(True)
```

- [ ] **Step 7: Inline the button-state refresh in results handlers**

`_on_analysis_completed` and `_on_analysis_failed` (currently around lines 972–999) call `self.action_panel.update_button_states(...)`. Replace those calls with `self._update_ui_state()`.

In `_on_analysis_completed`, replace:

```python
        # Update action panel button states
        if self.action_panel:
            self.action_panel.update_button_states(
                has_reference=self.data_manager.preprocessed_reference is not None,
                has_beads=self.data_manager.preprocessed_bead_stack is not None,
                has_results=True
            )
```

with:

```python
        self._update_ui_state()
```

In `_on_analysis_failed`, replace:

```python
        if self.action_panel:
            self.action_panel.update_button_states(
                has_reference=self.data_manager.preprocessed_reference is not None,
                has_beads=self.data_manager.preprocessed_bead_stack is not None,
                has_results=False
            )
        QMessageBox.critical(self, "Error", error_msg)
```

with:

```python
        self._update_ui_state()
        QMessageBox.critical(self, "Error", error_msg)
```

- [ ] **Step 8: Remove the now-dead `_on_parameters_reset`/`_on_parameter_changed` widget handlers**

These were only connected to the deleted parameter panel. Remove the two widget methods (currently around lines 916–922):

```python
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        pass

    def _on_parameters_reset(self):
        """Handle parameter reset and update status."""
        self._update_status(0, "Displacement parameters reset to default values.")
```

> Leave the *controller's* `_on_parameter_changed`/`_on_parameters_reset` (the parameter-manager-driven preview hooks) untouched — they are independent of panel ownership.

- [ ] **Step 9: Run widget-construction smoke + ownership tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_displacement_ownership.py -v`
Expected: controller tests PASS; class-removal tests still FAIL (classes deleted in Task 4).

Then a construction smoke check:

```bash
QT_QPA_PLATFORM=offscreen python -c "
import napari
from napariTFM.widgets.displacement_analysis_widget import DisplacementAnalysisWidget
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager
v = napari.Viewer(show=False)
dm = DataManager(); pm = ParameterManager(); vm = VisualizationManager(v, dm)
w = DisplacementAnalysisWidget(v, dm, pm, vm)
assert hasattr(w, 'process_btn') and hasattr(w, 'preview_btn') and hasattr(w, 'cancel_btn')
assert hasattr(w, 'load_active_layer')
print('SMOKE OK')
"
```
Expected: `SMOKE OK`.

- [ ] **Step 10: Commit**

```bash
git add napariTFM/widgets/displacement_analysis_widget.py
git commit -m "Move displacement action buttons onto widget and add load delegate"
```

---

### Task 4: Delete the three panel classes

**Files:**
- Modify: `napariTFM/widgets/displacement_analysis_widget.py` — delete `DisplacementDataPanel`, `DisplacementParameterPanel`, `DisplacementActionPanel`; clean unused imports.

- [ ] **Step 1: Delete the class definitions**

Delete the entire bodies of:
- `class DisplacementDataPanel(QWidget):` (currently lines ~24–114)
- `class DisplacementParameterPanel(QWidget):` (currently lines ~117–350)
- `class DisplacementActionPanel(QWidget):` (currently lines ~353–422)

Leave `class DisplacementController(QObject):` and `class DisplacementAnalysisWidget(BaseAnalysisWidget):` intact.

- [ ] **Step 2: Remove imports left unused after the deletions**

After deleting the panels, several imports may become unused. Check and remove any of these that no longer have a reference in the file: `ParameterCategory` (used only by the deleted `DisplacementParameterPanel._on_parameters_reset`), `Qt` (used only by the deleted panel's `_safe_set_combo_text`), `QGroupBox`, `QSpinBox`, `QDoubleSpinBox`, `QScrollArea` (verify — `_create_content_container` still uses `QScrollArea`), `QLabel` (verify — `_create_status_frame` uses it), `Image` (used by deleted `DisplacementDataPanel.update_button_states`), `Path` (verify no remaining use).

Verify which imports are still referenced before removing:

```bash
for sym in ParameterCategory Qt QGroupBox QSpinBox QDoubleSpinBox Image Path QScrollArea QLabel; do
  echo "$sym: $(grep -c "\b$sym\b" napariTFM/widgets/displacement_analysis_widget.py)"; done
```
Remove only symbols whose count drops to the single occurrence on the import line itself.

- [ ] **Step 3: Run the ownership contract tests (now fully green)**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_displacement_ownership.py -v`
Expected: ALL PASS — the three `*_class_is_removed` tests now pass.

- [ ] **Step 4: Commit**

```bash
git add napariTFM/widgets/displacement_analysis_widget.py
git commit -m "Delete displacement data, parameter, and action panel classes"
```

---

### Task 5: Simplify the shell wiring

**Files:**
- Modify: `napariTFM/widgets/_widget.py` — displacement `action_targets` (lines ~481–486); `_hide_redundant_stage_shell_controls` (lines ~583–589).
- Modify: `tests/test_workflow_shell.py` — hide-list assertion loop (lines ~635–638).

- [ ] **Step 1: Collapse the displacement action-target fallbacks**

In `_create_stage_sections`, the displacement `_StageSection` currently uses dual paths that fall back to the deleted `action_panel.*`. Replace (lines ~481–486):

```python
                action_targets=self._find_stage_action_targets(
                    self.displacement_widget,
                    run=["process_btn", "action_panel.calculate_btn"],
                    preview=["preview_btn", "action_panel.preview_btn"],
                    cancel=["cancel_btn", "action_panel.cancel_btn"],
                ),
```

with:

```python
                action_targets=self._find_stage_action_targets(
                    self.displacement_widget,
                    run=["process_btn"],
                    preview=["preview_btn"],
                    cancel=["cancel_btn"],
                ),
```

- [ ] **Step 2: Drop displacement from the redundant-shell-controls hide list**

`_hide_redundant_stage_shell_controls` (lines ~583–589) hides `data_panel`/`action_panel` for stages that still have them. Displacement no longer does. Mirror how preprocessing was handled (it is absent from this list). Replace:

```python
        for widget in [self.displacement_widget, self.force_widget, self.msm_widget]:
```

with:

```python
        for widget in [self.force_widget, self.msm_widget]:
```

> Leave `_hide_embedded_parameter_panels` unchanged: it iterates with `getattr(widget, "parameter_panel", None)` and tolerates the now-missing attribute (preprocessing is still listed there for the same reason).

- [ ] **Step 3: Update the hide-list assertion test**

In `tests/test_workflow_shell.py::test_main_widget_hides_stage_local_data_and_action_panels_after_shell_wiring` (lines ~635–638), the loop iterates `[displacement, force, msm]`. Since displacement is no longer hidden by the shell (it has no such panels), drop it. Replace:

```python
    for stage_widget in [
        widget.displacement_widget,
        widget.force_widget,
        widget.msm_widget,
    ]:
```

with:

```python
    for stage_widget in [
        widget.force_widget,
        widget.msm_widget,
    ]:
```

- [ ] **Step 4: Run the shell tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py -v`
Expected: ALL PASS. In particular `test_main_widget_stage_headers_wire_existing_stage_actions` (uses the stub's `process_btn`), `test_displacement_data_rows_route_assignment_actions` (uses the stub's `load_active_layer`), and the edited hide-list test pass.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Simplify displacement shell wiring after panel inversion"
```

---

### Task 6: Migrate displacement parameter tests and run the full suite

**Files:**
- Delete: `tests/test_displacement_parameter_ui.py`
- Verify: `tests/test_workflow_shell.py` already covers farneback parameter exposure.

- [ ] **Step 1: Confirm farneback coverage exists in the shell tests**

The deleted `DisplacementParameterPanel` was tested by `tests/test_displacement_parameter_ui.py`. Equivalent coverage already lives in `tests/test_workflow_shell.py`:
- `test_main_widget_groups_parameters_inline_per_stage` asserts `{"nscales", "inner_iterations"}` are in the displacement `WorkflowParameterPanel` and `"young_modulus"` is not.
- `test_workflow_parameter_panel_labels_farneback_controls` asserts the farneback labels.

Verify:

```bash
grep -n "nscales\|inner_iterations\|labels_farneback" tests/test_workflow_shell.py
```
Expected: matches confirming the coverage is present.

- [ ] **Step 2: Delete the obsolete panel test**

```bash
git rm tests/test_displacement_parameter_ui.py
```

- [ ] **Step 3: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen pytest -q`
Expected: green except the known `test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` env flake (re-run that file in isolation to confirm it passes alone: `QT_QPA_PLATFORM=offscreen pytest tests/test_napari_compatibility.py -v`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Remove obsolete displacement parameter panel test"
```

---

### Task 7: Manual smoke test (human-in-the-loop)

**Files:** none.

- [ ] **Step 1: Launch napari and exercise the displacement stage**

```bash
napari
```
Open Plugins → napariTFM → napariTFM. Then verify:
- The **Displacement** section's "Parameters" sub-section shows the farneback parameters (single editor — no duplicate panel).
- Loading a reference image and a bead stack via the displacement input rows enables **Preview Current Frame** and **Calculate All Frames** (this exercises the new `load_active_layer` delegate — previously a latent `AttributeError`).
- **Preview Current Frame** computes and visualizes a single-frame displacement.
- **Calculate All Frames** runs to completion; controls disable during the run and re-enable after; **Cancel Operation** aborts mid-run.
- The stage header's run/preview/cancel proxy buttons mirror the in-body buttons' enabled state.

---

## Self-Review

**Spec coverage:**
- Delete `DisplacementParameterPanel` → Task 4. Sever parameter ownership to shell → Tasks 2–4 (widget no longer builds/wires it; shell `WorkflowParameterPanel("Displacement")` already mounted).
- Delete `DisplacementDataPanel` (dead) → Task 4.
- Delete `DisplacementActionPanel`, re-home buttons → Tasks 3–4.
- Controller `freeze_ui`/`unfreeze_ui` emit-only → Task 2.
- Fix `load_active_layer` latent bug → Task 3 Step 3.
- Shell `action_targets` + hide-list + test updates → Task 5.
- Test migration → Tasks 1 & 6.

**Type/name consistency:** Run "Calculate All Frames" button is named `process_btn` (not `calculate_btn`) to match the shell's `run=["process_btn"]` and the preprocessing convention; the controller method it calls remains `calculate_all_frames`. `preview_btn`/`cancel_btn` keep their names. The controller constructor loses its 5th positional `data_panel` arg in Task 2 and Task 1's tests assume the 4-arg form.

**Intentional RED window:** The suite is knowingly red between Task 2 and Task 4 (widget references panels deleted across staggered tasks). This is called out in Task 2 Step 4; correctness is gated by the construction smoke test in Task 3 Step 9 and the full-suite run in Task 6 Step 3. If executing with subagents, the spec reviewer should treat per-task local test targets (named in each task) as the green bar, not the full suite, until Task 6.

**Known follow-ons (separate plans):** force inversion, then stress (MSM) inversion — each deletes its `*ActionPanel`/`*ParameterPanel` and re-homes the same coupling; after stress, drop the `_find_stage_action_targets` reflection entirely and give each stage a flat action surface. Then disk-derived status, theming, config persistence.
