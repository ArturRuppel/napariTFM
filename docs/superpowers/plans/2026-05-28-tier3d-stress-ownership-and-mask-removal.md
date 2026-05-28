# Tier 3d: Remove Mask-Making (App-Wide) + Invert Stress (MSM) Widget Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the in-app mask-making feature everywhere (interactive stress widget *and* the headless batch pipeline) so masks are always supplied externally, then complete the stress (MSM) stage ownership inversion so `ParameterManager` + the shell's `WorkflowParameterPanel` own parameters and the widget owns its own action buttons.

**Architecture:** Two phases. **Phase A** strips mask creation from the backend (`msm.py`), the parameter model (`MSMParameters`/`UnifiedParameters`/validation/`ParameterManager`), the batch pipeline + batch widget, and the shell parameter editor — leaving masks to be loaded from `masks.tif` (batch) or the existing "Load Masks" artifact row (interactive). **Phase B** then performs the same ownership inversion already applied to preprocessing/displacement/force: delete `MSMParameterPanel`/`MSMDataPanel`/`MSMActionPanel`, sever the controller from panels (it emits `ui_frozen` only), and re-home the four surviving action buttons (`preview_mesh_btn`, `preview_frame_btn`, `analyze_btn`, `cancel_btn`) onto the widget. The live mask-preview (`Show Preview` checkbox + `_update_mask_preview`/`_handle_preview_toggle`) is deleted along with mask creation.

**Tech Stack:** Python, qtpy/PyQt, napari `@thread_worker`, pytest (`QT_QPA_PLATFORM=offscreen`).

**Key facts established during research:**
- `MSMDataPanel` is **never instantiated** (`self.data_panel = None` in the widget; controller receives `data_panel=None`) — pure dead code, same as `FTTCDataPanel` was.
- Batch stress (`_handle_stress_execution`) and metrics (`_handle_metrics_execution`) **already** fall back to `tifffile.imread(tfm_folder / "masks.tif")` when `mask_data is None`, so dropping the `create_masks` step leaves a working "load external masks" path.
- The batch widget's `mask_source_combo` / `force_threshold_spin` / `mask_dilation_spin` (referenced in `_update_metrics_controls`/`_update_mask_controls`) are **never created** — pre-existing dead methods unrelated to MSM mask creation. **Out of scope** here; do not touch.
- `threshold`/`dilation`/`smoothing_sigma` are used ONLY for mask creation/preview. After Phase A they are dead everywhere.
- The whole `from scipy.ndimage import ...` line in `msm.py` is used only by `create_mask_from_image`; `skimage` `resize` is still used by `process_mask_data` and must be kept.
- Tests touching this surface: `tests/test_msm_analysis.py`, `tests/test_workflow_shell.py`, `tests/test_parameter_manager.py` (the last only uses `MSMParameters(density_factor=...)`, so it is unaffected).

**Standing constraints (project-specific, still in force):** Build on `master`; do NOT change the version (stays `"1.0"`); `origin/master` stays at clean `v1.0`; WIP pushes go to `origin/ui-port`. Standard git safety (no force-push, no `--amend` unless asked, never `--no-verify`). Stage specific files, not `git add -A`. Run tests with `QT_QPA_PLATFORM=offscreen pytest`.

---

## File Structure

**Phase A (mask removal):**
- `napariTFM/backend/msm.py` — delete `create_preview_mask`, module `create_mask_stack`, class `create_mask_from_image` + deprecated class `create_mask_stack`; remove now-unused `scipy.ndimage` import.
- `napariTFM/backend/parameter_dataclasses.py` — remove `threshold`/`dilation`/`smoothing_sigma` from `MSMParameters` (76-101) and `UnifiedParameters` (141-149), and from `to_msm_parameters` (196-211).
- `napariTFM/backend/parameter_validation.py` — remove the three mask checks in `validate_msm_parameters` (111-118).
- `napariTFM/utilities/parameter_manager.py` — remove the three names from the `STRESS` category list (187-189).
- `napariTFM/backend/batch_analysis.py` — delete `_handle_mask_creation`, `_execute_mask_creation`, `_log_mask_progress`; drop the `create_masks` call in `_process_single_folder` (284); drop the 3 mask kwargs in `_create_msm_parameters` (1059-1061); drop the `create_mask_stack` import (27).
- `napariTFM/widgets/batch_analysis_widget.py` — remove `("create_masks", "Create Masks")` from the steps list (156).
- `napariTFM/widgets/_widget.py` — remove the 3 mask controls from `WorkflowParameterPanel.PARAMETER_SECTIONS` "Stress" (233-235).
- `tests/test_msm_analysis.py` — delete `test_backend_creates_mask_stack_with_progress`; add a guard test that the mask functions are gone.
- `tests/test_workflow_shell.py` — change the stress-panel assertion (674) off `threshold`.

**Phase B (stress ownership inversion):**
- `tests/test_stress_ownership.py` — **create** (contract tests).
- `napariTFM/widgets/msm_widget.py` — sever controller from panels; delete the 3 panel classes; re-home action buttons; delete mask-creation/preview code.
- `napariTFM/widgets/_widget.py` — collapse the stress `action_targets` to single paths; drop `msm_widget` from `_hide_redundant_stage_shell_controls` and `_hide_embedded_parameter_panels`.
- `tests/test_workflow_shell.py` — update the panel-hide test loop.

---

# PHASE A — Remove mask-making app-wide

### Task A1: Remove mask creation from the MSM backend

**Files:**
- Modify: `napariTFM/backend/msm.py`
- Test: `tests/test_msm_analysis.py`

- [ ] **Step 1: Update the backend test to pin the removal (RED)**

In `tests/test_msm_analysis.py`, DELETE `test_backend_creates_mask_stack_with_progress` (lines 12-46) entirely, and ADD this guard test at the top of the test functions (after the `REPO_ROOT` definition):

```python
def test_mask_creation_helpers_are_removed():
    assert not hasattr(msm, "create_mask_stack")
    assert not hasattr(msm, "create_preview_mask")
    assert not hasattr(msm.MonolayerStressMicroscopy, "create_mask_from_image")
    assert not hasattr(msm.MonolayerStressMicroscopy, "create_mask_stack")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_msm_analysis.py::test_mask_creation_helpers_are_removed -v`
Expected: FAIL (the helpers still exist).

- [ ] **Step 3: Delete the mask-creation code**

In `napariTFM/backend/msm.py`:
1. Delete the module-level function `create_preview_mask` (def at line ~69 through its `return analysis_mask`).
2. Delete the module-level function `create_mask_stack` (def at line ~112 through its `return np.stack(analysis_masks)`).
3. Inside class `MonolayerStressMicroscopy`, delete the `@staticmethod create_mask_from_image` (def ~398) and the deprecated `@classmethod create_mask_stack` (def ~482) — delete through the end of the deprecated classmethod, stopping before the next surviving method.
4. KEEP `process_mask_data`, `generate_mesh_stack`, `calculate_stresses`, and the `MonolayerStressMicroscopy` mesh/stress methods.
5. Remove the now-unused import line `from scipy.ndimage import binary_fill_holes, generate_binary_structure, binary_dilation, label, gaussian_filter, sum as ndimage_sum`. **Before deleting, grep** to confirm none of those six names appear elsewhere in the file: `grep -nE 'binary_fill_holes|generate_binary_structure|binary_dilation|\blabel\(|gaussian_filter|ndimage_sum' napariTFM/backend/msm.py` — expect no matches after the function deletions. Keep the `skimage` `resize` import (still used by `process_mask_data`).

- [ ] **Step 4: Run the MSM backend tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_msm_analysis.py -v`
Expected: PASS (guard test passes; `process_mask_data`, `generate_mesh_stack`, `calculate_stresses` tests still pass).

- [ ] **Step 5: Commit**

```bash
git add napariTFM/backend/msm.py tests/test_msm_analysis.py
git commit -m "Remove mask-creation helpers from MSM backend"
```

---

### Task A2: Drop mask parameters from the parameter model

**Files:**
- Modify: `napariTFM/backend/parameter_dataclasses.py`
- Modify: `napariTFM/backend/parameter_validation.py`
- Modify: `napariTFM/utilities/parameter_manager.py`
- Test: `tests/test_parameter_manager.py`

- [ ] **Step 1: Add a failing test (RED)**

In `tests/test_parameter_manager.py`, add (the imports `MSMParameters` and `validate_msm_parameters` already exist at lines 6-ish and 24):

```python
def test_msm_parameters_have_no_mask_fields():
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(MSMParameters)}
    assert "threshold" not in field_names
    assert "dilation" not in field_names
    assert "smoothing_sigma" not in field_names


def test_validate_msm_ignores_mask_params():
    # density_factor is the first remaining gate; a high threshold-like value
    # must no longer be rejected because the field no longer exists.
    ok, _ = validate_msm_parameters(MSMParameters(density_factor=0.01))
    assert ok is True
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_parameter_manager.py::test_msm_parameters_have_no_mask_fields -v`
Expected: FAIL (`threshold` still a field).

- [ ] **Step 3: Remove the fields and their uses**

In `napariTFM/backend/parameter_dataclasses.py`:
- In `MSMParameters`, delete the three lines under `# Mask creation parameters` (79-81: `threshold`, `dilation`, `smoothing_sigma`) and the `# Mask creation parameters` comment.
- In `UnifiedParameters`, delete the three `# Stress parameters` mask lines (142-144: `threshold`, `dilation`, `smoothing_sigma`). Keep `density_factor`/`mesh_algorithm`/`use_optimization`/`poisson_ratio_cells`/`max_stress`.
- In `to_msm_parameters` (196-211), delete the three kwargs `threshold=self.threshold,`, `dilation=self.dilation,`, `smoothing_sigma=self.smoothing_sigma,`.

In `napariTFM/backend/parameter_validation.py`, in `validate_msm_parameters`, delete these three blocks (111-118):

```python
    if params.threshold < 0 or params.threshold > 100:
        return False, "Threshold percentile must be between 0 and 100"

    if params.dilation < 0:
        return False, "Dilation must be non-negative"

    if params.smoothing_sigma < 0:
        return False, "Smoothing sigma must be non-negative"
```

In `napariTFM/utilities/parameter_manager.py`, change the `ParameterCategory.STRESS` list (187-189) from:

```python
            ParameterCategory.STRESS: [
                'threshold', 'dilation', 'smoothing_sigma', 'density_factor',
                'mesh_algorithm', 'use_optimization', 'poisson_ratio_cells',
                'max_stress'
            ],
```

to:

```python
            ParameterCategory.STRESS: [
                'density_factor', 'mesh_algorithm', 'use_optimization',
                'poisson_ratio_cells', 'max_stress'
            ],
```

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_parameter_manager.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/backend/parameter_dataclasses.py napariTFM/backend/parameter_validation.py napariTFM/utilities/parameter_manager.py tests/test_parameter_manager.py
git commit -m "Drop mask-creation parameters from the parameter model"
```

---

### Task A3: Remove mask creation from the batch pipeline

**Files:**
- Modify: `napariTFM/backend/batch_analysis.py`

**Context:** `_create_msm_parameters` currently passes the three removed kwargs (1059-1061) — after Task A2 that would raise `TypeError`, so this task is required to keep the suite importable/green. Stress + metrics already load `masks.tif` when `mask_data is None`.

- [ ] **Step 1: Fix the `MSMParameters` construction**

In `_create_msm_parameters` (line ~1051), delete the three kwargs `threshold=...`, `dilation=...`, `smoothing_sigma=...` (1059-1061).

- [ ] **Step 2: Remove the create_masks step from the pipeline**

In `_process_single_folder`, replace the block at line ~283-287:

```python
            # Handle mask creation
            mask_data = self._handle_mask_creation(tfm_folder)

            # Handle stress analysis
            stress_data = self._handle_stress_execution(tfm_folder, force_data, mask_data)
```

with:

```python
            # Masks are supplied externally (loaded from masks.tif by downstream steps)
            mask_data = None

            # Handle stress analysis
            stress_data = self._handle_stress_execution(tfm_folder, force_data, mask_data)
```

- [ ] **Step 3: Delete the now-dead mask methods**

Delete `_handle_mask_creation` (def ~353-369), `_execute_mask_creation` (def ~796-861), and `_log_mask_progress` (def ~1080-1093). Then remove `create_mask_stack` from the import on line ~27 (`from napariTFM.backend.msm import MSMResult, calculate_stresses, create_mask_stack, generate_mesh_stack` → drop `create_mask_stack,`).

- [ ] **Step 4: Grep-verify no stale references**

Run: `grep -nE '_handle_mask_creation|_execute_mask_creation|_log_mask_progress|create_mask_stack' napariTFM/backend/batch_analysis.py`
Expected: no matches.

- [ ] **Step 5: Run the batch + full backend tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/ -q`
Expected: PASS (collection succeeds — note batch may have no dedicated test; the key is the suite still imports `batch_analysis` cleanly and all green).

- [ ] **Step 6: Commit**

```bash
git add napariTFM/backend/batch_analysis.py
git commit -m "Remove mask-creation step from batch pipeline"
```

---

### Task A4: Remove the "Create Masks" step from the batch widget

**Files:**
- Modify: `napariTFM/widgets/batch_analysis_widget.py`

- [ ] **Step 1: Drop the checkbox entry**

In `_create_analysis_steps_group` (line ~152-159), remove the tuple `("create_masks", "Create Masks"),` from the `steps` list. The resulting list:

```python
        steps = [
            ("preprocessing", "Preprocessing"),
            ("displacement", "Displacement"),
            ("force", "Force"),
            ("stress", "Stress"),
            ("calculate_metrics", "Calculate Metrics (Strain Energy & Polarization)")
        ]
```

(The `analysis_steps` dict in `_build_config` is derived from `self.analysis_checkboxes`, so removing the step here also removes it from the emitted config. `_load_config` guards with `if key in config.get('analysis_steps', {})`, so older configs containing `create_masks` load harmlessly.)

- [ ] **Step 2: Verify the batch widget still constructs**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py -q`
Expected: PASS (the shell builds `BatchAnalysisWidget`; constructs without the step).

- [ ] **Step 3: Commit**

```bash
git add napariTFM/widgets/batch_analysis_widget.py
git commit -m "Remove Create Masks step from batch analysis widget"
```

---

### Task A5: Remove mask params from the shell parameter editor

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Update the shell param test (RED)**

In `tests/test_workflow_shell.py`, change line 674 from:

```python
    assert {"threshold", "mesh_algorithm"}.issubset(stress_panel.parameter_controls)
```

to:

```python
    assert {"density_factor", "mesh_algorithm"}.issubset(stress_panel.parameter_controls)
    assert "threshold" not in stress_panel.parameter_controls
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py::test_main_widget_groups_parameters_inline_per_stage -v`
Expected: FAIL (`threshold` still present in the Stress section).

- [ ] **Step 3: Remove the three controls from the "Stress" section**

In `WorkflowParameterPanel.PARAMETER_SECTIONS`, delete these three rows from the `"Stress"` section (233-235):

```python
            ("threshold", "Threshold Percentile (%)", "float", 0.0, 100.0, 0.1, 1, None),
            ("dilation", "Mask Dilation (px)", "int", 0, 50, 1, 0, None),
            ("smoothing_sigma", "Boundary Smoothing", "float", 0.0, 40.0, 0.1, 1, None),
```

The `"Stress"` section now begins with `density_factor`.

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Remove mask parameters from shell stress editor"
```

---

# PHASE B — Invert stress (MSM) widget ownership

> Mirrors the completed force inversion. Tasks B1-B4 contain an intentional RED window (the contract test from B1 stays red until B4 deletes the panel classes). Run the full suite only at B6.

### Task B1: Pin the inverted contract with failing tests

**Files:**
- Create: `tests/test_stress_ownership.py`

- [ ] **Step 1: Write the contract tests (RED)**

Create `tests/test_stress_ownership.py` (modeled on `tests/test_force_ownership.py`; the fake managers mirror it, with `mask_stack`/`stress_results` added to the fake data manager):

```python
import pytest
from qtpy.QtWidgets import QApplication

import napariTFM.widgets.msm_widget as mw


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
    force_results = None
    mask_stack = None
    stress_results = None


class _FakeParameterManager:
    def __init__(self):
        from qtpy.QtCore import QObject, Signal

        class _PM(QObject):
            parameter_changed = Signal(str, object)
            parameters_reset = Signal(object)

        self._pm = _PM()
        self.parameter_changed = self._pm.parameter_changed
        self.parameters_reset = self._pm.parameters_reset

    def get_msm_parameters(self):
        return object()


def test_parameter_panel_class_is_removed():
    assert not hasattr(mw, "MSMParameterPanel")


def test_data_panel_class_is_removed():
    assert not hasattr(mw, "MSMDataPanel")


def test_action_panel_class_is_removed():
    assert not hasattr(mw, "MSMActionPanel")


def test_controller_has_no_panel_attributes(app):
    controller = mw.MSMController(
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
    controller = mw.MSMController(
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

Note the controller is constructed with **4 kwargs** (no `data_panel`).

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_stress_ownership.py -v`
Expected: FAIL — panel classes still exist; `MSMController.__init__` still requires `data_panel`; `freeze_ui`/`unfreeze_ui` still poke panels.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_stress_ownership.py
git commit -m "Add failing stress ownership contract tests"
```

---

### Task B2: Sever the controller from its panels and delete mask/preview code

**Files:**
- Modify: `napariTFM/widgets/msm_widget.py`

**Context:** This is the largest edit. After it the suite has a known RED window (B1 + import of deleted panels) until B4; that's expected.

- [ ] **Step 1: Rewrite `MSMController.__init__` and freeze/unfreeze**

Change the constructor signature to drop `data_panel` and the panel attributes:

```python
    def __init__(self, viewer: Viewer,
                 data_manager: DataManager, parameter_manager: ParameterManager,
                 visualization_manager: VisualizationManager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []
```

Delete `set_panels` (def ~1166-1169). Replace `freeze_ui`/`unfreeze_ui` (def ~1171-1189) with emit-only:

```python
    def freeze_ui(self):
        """Signal the owning widget to disable interactive controls."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal the owning widget to re-enable controls."""
        self.ui_frozen.emit(False)
```

- [ ] **Step 2: Delete mask-creation + live-preview controller code**

Delete these `MSMController` members entirely:
- `_update_mask_preview` (def ~756-820)
- `_handle_preview_toggle` (def ~822-835)
- `create_masks_from_images` (def ~1195-1255)
- The three mask-creation signal class attributes (740-742): `mask_creation_progress`, `mask_creation_completed`, `mask_creation_failed`.

Also delete the dead helpers that referenced mask/old worker plumbing (verify each has no remaining caller with grep before deleting): `_start_stress_calculation` (def ~1110-1164, references a non-existent `_handle_worker_error`), `_generate_mesh_stack` (def ~1257-1282), `_handle_progress` (def ~1284-1287). Grep: `grep -nE '_start_stress_calculation|_generate_mesh_stack|_handle_progress|mask_creation_' napariTFM/widgets/msm_widget.py` after deletion — expect no matches.

KEEP on the controller: `start_analysis`, `preview_current_frame`, `preview_mesh`, `cancel_all_operations`, `_validate_prerequisites`, `_get_current_parameters`, `_update_progress`, and the `data_updated`/`progress_updated`/`analysis_*`/`ui_frozen` signals.

- [ ] **Step 3: Drop the now-unused backend imports**

At the top of `msm_widget.py`, change:

```python
from napariTFM.backend.msm import (
    MSMResult,
    calculate_stresses,
    create_mask_stack,
    create_preview_mask,
    generate_mesh_stack,
    process_mask_data,
)
```

to:

```python
from napariTFM.backend.msm import (
    MSMResult,
    calculate_stresses,
    generate_mesh_stack,
    process_mask_data,
)
```

- [ ] **Step 4: Simplify the widget `__init__` (construct controller with 4 kwargs, no panels)**

In `MSMWidget.__init__`, replace the panel-construction block (1390-1419) with:

```python
        # Initialize controller (owns no panels; emits ui_frozen)
        self.controller = MSMController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
        )

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

        # Keep service parameters synced with the shared parameter manager
        parameter_manager.parameters_reset.connect(self._update_service_parameters)
        parameter_manager.parameter_changed.connect(self._handle_parameter_change)

        self.controller.unfreeze_ui()
```

Delete the lines that created `self.parameter_panel = MSMParameterPanel(...)`, `self.data_panel = None`, `self.action_panel = MSMActionPanel(...)`, `self.controller.set_panels(...)`, and the `self.msm_params = parameter_manager.get_msm_parameters()` line may stay (still used by `_update_service_parameters`/`_handle_parameter_change`) — keep `self.parameter_manager = parameter_manager` and `self.msm_params = parameter_manager.get_msm_parameters()`.

> NOTE: The widget body still references panels in `_setup_ui`/`_connect_signals`/handlers — those are fixed in Task B3. The module still defines the panel classes — deleted in Task B4. The suite stays red until then.

- [ ] **Step 5: Commit (intentional RED)**

```bash
git add napariTFM/widgets/msm_widget.py
git commit -m "Sever MSM controller from panels and delete mask creation"
```

---

### Task B3: Move action buttons onto the widget + freeze wiring

**Files:**
- Modify: `napariTFM/widgets/msm_widget.py`

**Context:** The stress header proxies `run`/`preview`/`cancel`. `create_mask_btn` is gone (mask creation removed). The four surviving buttons: `preview_mesh_btn`, `preview_frame_btn`, `analyze_btn`, `cancel_btn`.

- [ ] **Step 1: Rewrite `_create_content_container` to use a widget-owned action row**

Replace the body of `_create_content_container` (1432-1454) so it no longer adds `self.parameter_panel`/`self.action_panel`:

```python
    def _create_content_container(self) -> QWidget:
        """Create the main content container."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_action_row())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll

    def _create_action_row(self) -> QWidget:
        """Build widget-owned action buttons (run/preview/cancel proxied by the stage header)."""
        container = QWidget()
        layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.preview_mesh_btn = QPushButton("Preview Mesh")
        self.preview_mesh_btn.setToolTip("Generate and display a mesh preview for the current frame")
        self.preview_frame_btn = QPushButton("Preview Current Frame")
        self.preview_frame_btn.setToolTip("Calculate and visualize stress for the current frame only")
        row1.addWidget(self.preview_mesh_btn)
        row1.addWidget(self.preview_frame_btn)
        layout.addLayout(row1)

        self.analyze_btn = QPushButton("Calculate Stress Tensors")
        self.analyze_btn.setToolTip("Calculate stress for all frames in the dataset")
        layout.addWidget(self.analyze_btn)

        self.cancel_btn = QPushButton("Cancel All Operations")
        self.cancel_btn.setToolTip("Cancel the current operation")
        layout.addWidget(self.cancel_btn)

        container.setLayout(layout)
        return container
```

- [ ] **Step 2: Rewrite `_connect_signals`**

Replace `_connect_signals` (1471-1508) with a version that wires controller signals + widget-owned buttons and drops all panel/preview-checkbox/mask wiring:

```python
    def _connect_signals(self):
        """Connect all widget signals."""
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_started.connect(self._on_analysis_started)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Wire widget-owned action buttons to controller operations
        self.preview_mesh_btn.clicked.connect(self.controller.preview_mesh)
        self.preview_frame_btn.clicked.connect(self.controller.preview_current_frame)
        self.analyze_btn.clicked.connect(self.controller.start_analysis)
        self.cancel_btn.clicked.connect(self.controller.cancel_all_operations)

        # Update enablement when the active layer changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)
```

> NOTE: previously the "Calculate Stress" path went through `MSMActionPanel._handle_analyze_click` (which disabled buttons then called `controller.start_analysis`). `start_analysis` already emits `analysis_started` and the controller freezes via worker flow; button enablement is now driven by `_handle_ui_freeze`/`_update_ui_state`.

- [ ] **Step 3: Rewrite `_update_ui_state` and `_handle_ui_freeze`**

Replace `_update_ui_state` (1576-1589) and `_handle_ui_freeze` (1597-1600):

```python
    def _update_ui_state(self, event=None):
        """Update action button enablement based on available data."""
        has_force = self.data_manager.force_results is not None
        has_mask = self.data_manager.mask_stack is not None

        self.preview_mesh_btn.setEnabled(has_mask)
        self.preview_frame_btn.setEnabled(has_force and has_mask)
        self.analyze_btn.setEnabled(has_force and has_mask)
        self.cancel_btn.setEnabled(True)

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        if frozen:
            self.preview_mesh_btn.setEnabled(False)
            self.preview_frame_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
        else:
            self._update_ui_state()
```

- [ ] **Step 4: Remove mask-creation/preview handlers and panel references in the rest of the widget**

- Delete `_on_parameter_changed` (def ~1606-1609, the dead `preview_active` stub) and `_on_mask_creation_completed` (1633-1648) and `_on_mask_creation_failed` (1650-1655).
- In `_on_analysis_completed`, `_on_analysis_failed`, `_on_analysis_started` — keep, but ensure they call `self._update_ui_state()` (they already do) and don't touch panels.
- `load_result_artifact` (1522-1533) and `_load_mask_stack_from_active_layer` (1544-1559) and `_choose_result_path` (1535-1542): KEEP — these are the external-mask load path used by the shell artifact rows. `load_result_artifact` already ends with `self._update_ui_state()`.
- KEEP `_update_service_parameters`, `_handle_parameter_change`, `_on_frame_changed`, `cleanup`, `_update_status`, `_create_status_frame`, `_setup_ui`.
- Grep to confirm no widget code still references `self.parameter_panel`, `self.action_panel`, `self.data_panel`, `preview_checkbox`, or `mask_creation`: `grep -nE 'parameter_panel|action_panel|data_panel|preview_checkbox|mask_creation' napariTFM/widgets/msm_widget.py` — only the class definitions (to be deleted in B4) should remain, no `self.`-qualified references in `MSMWidget`.

- [ ] **Step 5: Commit (still RED until B4)**

```bash
git add napariTFM/widgets/msm_widget.py
git commit -m "Move MSM action buttons onto widget and wire freeze signal"
```

---

### Task B4: Delete the three panel classes

**Files:**
- Modify: `napariTFM/widgets/msm_widget.py`

- [ ] **Step 1: Delete the classes**

Delete `MSMParameterPanel` (def ~44-316), `MSMDataPanel` (def ~319-599), and `MSMActionPanel` (def ~602-728) in full. Only `_is_valid_image_layer` (module helper — see Step 2), `MSMController`, and `MSMWidget` should remain as top-level definitions.

- [ ] **Step 2: Clean up now-unused imports and helpers**

- `_is_valid_image_layer` (30-41): grep for remaining uses — `grep -n '_is_valid_image_layer' napariTFM/widgets/msm_widget.py`. After the panel deletions it has no callers (it was used only by the data/action panels). Delete it.
- Remove now-unused qtpy imports. After deletion the widget uses: `QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy, QSpacerItem, QFrame, QProgressBar, QLabel, QPushButton, QMessageBox, QFileDialog`. Verify with grep which of `QGroupBox, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QApplication, Qt` are still referenced; remove any with zero remaining uses. (`QApplication` is used inside `start_analysis` via `QApplication.processEvents()` — keep it. `Qt` — grep; likely removable.) Remove the `from napariTFM.utilities.parameter_manager import ParameterManager, ParameterCategory` extras only if `ParameterCategory` is unused — it's still used by `_update_service_parameters`/`_handle_parameter_change`, so KEEP `ParameterCategory`. Remove `from typing import Any` only if `Any` no longer appears (it's used in `_handle_parameter_change` signature — KEEP).
- Run `python -c "import napariTFM.widgets.msm_widget"` (with `QT_QPA_PLATFORM=offscreen`) to confirm no `NameError`/`ImportError`.

- [ ] **Step 3: Run the stress ownership contract tests (GREEN)**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_stress_ownership.py -v`
Expected: PASS (all 5).

- [ ] **Step 4: Commit**

```bash
git add napariTFM/widgets/msm_widget.py
git commit -m "Delete MSM parameter, data, and action panel classes"
```

---

### Task B5: Simplify the shell wiring

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Collapse the stress `action_targets` to single paths**

In `_build... _stage_sections_by_key["stress"]` (503-508), change:

```python
                action_targets=self._find_stage_action_targets(
                    self.msm_widget,
                    run=["action_panel.analyze_btn", "analyze_btn"],
                    preview=["action_panel.preview_frame_btn", "action_panel.preview_mesh_btn"],
                    cancel=["action_panel.cancel_btn", "cancel_btn"],
                ),
```

to:

```python
                action_targets=self._find_stage_action_targets(
                    self.msm_widget,
                    run=["analyze_btn"],
                    preview=["preview_frame_btn", "preview_mesh_btn"],
                    cancel=["cancel_btn"],
                ),
```

- [ ] **Step 2: Drop `msm_widget` from the redundant-controls and embedded-panel hiders**

`_hide_redundant_stage_shell_controls` (583-589) currently loops over `[self.msm_widget]` hiding `data_panel`/`action_panel` — the MSM widget no longer has those. Since no widget remains in the loop, simplify the method to a no-op-with-comment or remove its body and its call. Recommended: delete the method entirely and its call site (search `_hide_redundant_stage_shell_controls`), since after force + stress inversions no stage widget exposes `data_panel`/`action_panel`.

In `_hide_embedded_parameter_panels` (563-571), remove `self.msm_widget,` from the list (the MSM widget no longer has a `parameter_panel`; the `getattr(..., "parameter_panel", None)` guard tolerates its absence, but drop it for clarity). The `findChildren(QGroupBox)` block that hides batch's "Stress Parameters" group stays.

- [ ] **Step 2b: Verify the call site**

Run: `grep -n '_hide_redundant_stage_shell_controls' napariTFM/widgets/_widget.py` — if you deleted the method, ensure the call (in `__init__`/setup) is also removed.

- [ ] **Step 3: Update the shell panel-hide test**

In `tests/test_workflow_shell.py`, `test_main_widget_hides_stage_local_data_and_action_panels_after_shell_wiring` (623-639): the loop `for stage_widget in [widget.msm_widget,]` now asserts that `msm_widget` has no `data_panel`/`action_panel`. Since the stub `_StubStageWidget` still creates `data_panel`/`action_panel` (lines 138-141) and is substituted for `MSMWidget`, the `getattr(...) is None or ...isHidden()` assertion would require them hidden. After removing `_hide_redundant_stage_shell_controls`, the stub's panels would NOT be hidden → test fails.

Resolve by deleting this test entirely (it asserted behavior of a now-removed mechanism — the real `MSMWidget` no longer has those panels, so there is nothing to hide). Confirm no other test references `_hide_redundant_stage_shell_controls`.

> If the spec/code reviewer prefers keeping a guard: instead replace the test body with an assertion that the real (non-stubbed) `MSMWidget` class has no `MSMDataPanel`/`MSMActionPanel` — but that duplicates `test_stress_ownership.py`. Deleting is cleaner.

- [ ] **Step 4: Run the shell tests**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Simplify stress shell wiring after panel inversion"
```

---

### Task B6: Run the full suite

**Files:** none (verification).

- [ ] **Step 1: Run everything**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/ -q`
Expected: PASS. (Known env flake: `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` may intermittently SIGSEGV in its spawned subprocess — a napari/pydantic manifest race, not caused by this work. Re-run in isolation to confirm green: `QT_QPA_PLATFORM=offscreen pytest tests/test_napari_compatibility.py -v`.)

- [ ] **Step 2: Final cross-cutting grep**

Run:
```bash
grep -rnE 'create_mask_stack|create_preview_mask|create_mask_from_image|create_masks|MSMParameterPanel|MSMDataPanel|MSMActionPanel|_update_mask_preview|_handle_preview_toggle' napariTFM/ tests/
```
Expected: no matches (or only the new `test_stress_ownership.py`/`test_msm_analysis.py` *negative* assertions and the `_load_config` backwards-compat guard if any). Investigate anything else.

- [ ] **Step 3: Commit (only if grep surfaced a fix; otherwise skip)**

---

### Task B7: Manual smoke test (human-in-the-loop)

**Not automatable — requires the user to launch napari.**

- [ ] Launch napari, open the napariTFM widget, expand **Stress Analysis**.
- [ ] Confirm there is a single Stress parameter editor (in the shell), with **no** Threshold/Dilation/Boundary-Smoothing controls and **no** "Show Preview" checkbox.
- [ ] Confirm there is **no** "Create Masks from Image" button anywhere.
- [ ] Load force results and an external mask (via the "Load Forces"/"Load Masks" artifact rows). With both present, **Preview Mesh**, **Preview Current Frame**, and **Calculate Stress Tensors** enable; with only a mask, just **Preview Mesh** enables.
- [ ] Run **Preview Mesh** and **Preview Current Frame** — they visualize and the controls freeze/unfreeze correctly.
- [ ] Run **Calculate Stress Tensors** — progress updates, freeze/unfreeze works, result auto-saves.
- [ ] **Cancel All Operations** aborts a running job and unfreezes.
- [ ] The stage header proxy buttons (run/preview/cancel) mirror the in-body buttons.
- [ ] In **Batch Analysis**, confirm the Analysis Steps list no longer shows "Create Masks"; a batch run uses an existing `masks.tif`.

---

## Self-Review

**Spec coverage:**
- Remove mask making (interactive): Phase B deletes `create_masks_from_images`, the preview, the `create_mask_btn`, and the mask params from the editor. ✓
- Remove mask making (batch): Task A3/A4 remove the step, methods, and checkbox. ✓
- Backend cleanup: Task A1 deletes the four mask functions + dead import. ✓
- Parameter model cleanup: Task A2. ✓
- Stress ownership inversion (mirror force): Tasks B1-B5. ✓

**Placeholder scan:** No TODO/TBD/"handle edge cases"; every code step shows the code. ✓

**Type/name consistency:** Buttons named `preview_mesh_btn`/`preview_frame_btn`/`analyze_btn`/`cancel_btn` consistently across B3 (creation), B3 (wiring), B5 (action_targets), B7 (smoke). Controller methods `preview_mesh`/`preview_current_frame`/`start_analysis`/`cancel_all_operations` match existing controller definitions retained in B2. ✓

**Risk notes:**
- The batch widget's `_update_metrics_controls`/`_update_mask_controls` are pre-existing dead methods referencing never-created widgets; left untouched (out of scope).
- `_load_config` tolerates legacy configs that still contain `create_masks` via its `if key in config.get('analysis_steps', {})` guard — no migration needed.
- Intentional RED window across B1-B4; full suite is only asserted green at B6.
