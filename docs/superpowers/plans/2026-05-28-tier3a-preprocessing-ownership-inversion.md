# Tier 3a: Preprocessing Ownership Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PreprocessingWidget` own its workflow surface directly and delete its duplicated `PreprocessingParameterPanel` and `PreprocessingDataPanel`, so the shell's `WorkflowParameterPanel` is the *only* preprocessing parameter editor and the controller no longer depends on any embedded panel.

**Architecture:** This is the first of four per-stage "ownership inversion" plans (preprocessing → displacement → force → stress). Preprocessing is first because it has no `ActionPanel` — its `process_btn`/`cancel_btn` already live directly on the widget, and its controller already emits `ui_frozen` and drives button state from it. So the only duplicated panels to remove are the parameter and data panels. We sever the controller/widget from `PreprocessingParameterPanel` (parameter sync already flows through `ParameterManager`; the panel channel is redundant), keep `process_btn`/`preview_check`/`cancel_btn` on the widget as the stable action contract the header proxies, then delete both panel classes. `ParameterManager` stays the only parameter owner; no backend or numerical changes.

**Tech Stack:** Python, qtpy/PyQt, pytest, napari plugin.

**Spec:** north-star `docs/cellflow-ui-concept-for-naparitfm.md`; prior phases `docs/superpowers/plans/2026-05-18-tier1-cellflow-alignment.md` and `2026-05-18-tier2-project-artifact-workflow.md`.

---

## Background facts (verified against current code)

- `PreprocessingWidget.__init__` (`preprocessing_widget.py:1079`) creates `self.parameter_panel = PreprocessingParameterPanel(parameter_manager)` (line 1092), `self.data_panel = None` (1093), and calls `self.controller.set_panels(self.parameter_panel, None)` (1103).
- The controller's `freeze_ui`/`unfreeze_ui` (`:1054`, `:1062`) call `parameter_panel.freeze_ui(...)` **and** emit `ui_frozen` (`:1060`, `:1068`). The widget's `_handle_ui_freeze` (`:1280`) already disables `preview_check`/`process_btn` from that signal — so the panel `freeze_ui` calls are redundant.
- Param-change-driven preview is handled twice: by the controller via `parameter_manager.parameter_changed → _on_parameter_changed → _update_preview` (`:759`, `:862`), and again by the widget via `parameter_panel.parameter_changed → _on_parameter_changed` (`:1216`, `:1241`). The controller path through `ParameterManager` is the canonical one and survives panel deletion.
- `_sync_parameter` (`:1223`) pushes `parameter_manager` changes back into `parameter_panel.update_parameter` — pure panel plumbing, removable.
- `preview_check` lives on the widget (`_create_preview_frame`, `:1157`), not the panel. `controller.preview_check` is set to it (`:1107`).
- `PreprocessingDataPanel` (`:26`) is never instantiated by `PreprocessingWidget` (`data_panel` is always `None`). It is dead in the live widget.
- Shell coupling: `_widget.py:_hide_embedded_parameter_panels` (`_widget.py:561`) does `panel = getattr(widget, "parameter_panel", None); if panel is not None: panel.setVisible(False)`. After this plan `preprocessing_widget.parameter_panel` will not exist; the `getattr(..., None)` guard already tolerates that, so no shell change is required, but we add a regression test to lock it.
- Test contract: `tests/test_workflow_shell.py` drives the shell with `_StubStageWidget`, which exposes `process_btn`, `preview_btn`, `cancel_btn`, `parameter_panel` (a bare `QWidget`), `load_active_layer`. None of those stub attributes change in this plan, so shell tests are unaffected. Real-widget tests live in `tests/test_preprocessing_analysis.py` and `tests/test_preprocessing_ui_redesign.py`.

---

## File Map

- Modify `napariTFM/widgets/preprocessing_widget.py`
  - Delete `PreprocessingDataPanel` (dead) and `PreprocessingParameterPanel` (duplicated).
  - Remove the controller's `parameter_panel`/`data_panel` attributes, `set_panels`, and panel `freeze_ui` calls.
  - Remove the widget's `parameter_panel` construction, layout placement, and signal wiring; keep `preview_check`/`process_btn`/`cancel_btn` and the `ui_frozen`-driven state.
- Modify `tests/test_preprocessing_ui_redesign.py`
  - Drop/za­dapt assertions that reference `PreprocessingParameterPanel` / `widget.parameter_panel`.
- Create `tests/test_preprocessing_ownership.py`
  - Pin the post-inversion contract: no `parameter_panel`/`data_panel`; classes gone; preview still tracks `ParameterManager`; freeze still disables run.
- Modify `tests/test_workflow_shell.py`
  - Add one regression test that the real-shaped widget (via stub is unaffected) — actually add a guard test that `_hide_embedded_parameter_panels` tolerates a missing `parameter_panel`. (Uses a local stub; see Task 5.)

---

## Task 1: Pin the target contract with failing tests

**Files:**
- Create: `tests/test_preprocessing_ownership.py`

This task writes the spec for the finished state as executable tests. They will fail now and pass after Tasks 2–4.

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_preprocessing_ownership.py`:

```python
import pytest
from qtpy.QtWidgets import QApplication

import napariTFM.widgets.preprocessing_widget as pw


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_parameter_panel_class_is_removed():
    assert not hasattr(pw, "PreprocessingParameterPanel")


def test_data_panel_class_is_removed():
    assert not hasattr(pw, "PreprocessingDataPanel")


def test_controller_has_no_panel_attributes(app):
    controller = pw.PreprocessingController(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )
    assert not hasattr(controller, "parameter_panel")
    assert not hasattr(controller, "data_panel")
    assert not hasattr(controller, "set_panels")


def test_controller_freeze_emits_signal_without_panels(app):
    controller = pw.PreprocessingController(
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

Add these minimal fakes at the top of the file (below imports, above the tests):

```python
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
    bead_stack = None
    reference = None
    cell_stack = None
    preprocessed_bead_stack = None
    preprocessed_reference = None
    preprocessed_cell_stack = None


class _FakeParameterManager:
    def __init__(self):
        from qtpy.QtCore import QObject, Signal

        class _PM(QObject):
            parameter_changed = Signal(str, object)
            parameters_reset = Signal(object)

        self._pm = _PM()
        self.parameter_changed = self._pm.parameter_changed
        self.parameters_reset = self._pm.parameters_reset

    def get_preprocessing_parameters(self):
        return object()
```

- [ ] **Step 1.2: Run the tests to verify they fail**

```bash
cd /home/aruppel/Projects/napariTFM
pytest tests/test_preprocessing_ownership.py -v
```

Expected: `test_parameter_panel_class_is_removed` and `test_data_panel_class_is_removed` FAIL (classes still exist); `test_controller_has_no_panel_attributes` FAILS (`set_panels` still exists); the freeze test may pass already (the signal is emitted today) — that is fine, it guards against regression.

- [ ] **Step 1.3: Commit the failing tests**

```bash
git add tests/test_preprocessing_ownership.py
git commit -m "Add failing contract tests for preprocessing ownership inversion"
```

---

## Task 2: Sever the controller from its panels

**Files:**
- Modify: `napariTFM/widgets/preprocessing_widget.py` (`PreprocessingController`)

Remove `parameter_panel`/`data_panel` state, `set_panels`, and the panel `freeze_ui` calls. Keep the `ui_frozen` signal and the `ParameterManager` connection that already drives preview. `preview_check` handling stays (it is a widget attribute assigned onto the controller, not a panel).

- [ ] **Step 2.1: Remove panel attributes from `__init__`**

In `PreprocessingController.__init__` (`preprocessing_widget.py:746`), delete these three lines (`:754`–`:756` minus `preview_enabled`):

```python
        self.parameter_panel = None
        self.data_panel = None
        self.preview_enabled = False
```

Replace with:

```python
        self.preview_enabled = False
```

- [ ] **Step 2.2: Delete `set_panels`**

Delete the whole `set_panels` method (`:765`–`:768`):

```python
    def set_panels(self, parameter_panel, data_panel):
        """Set the parameter and data panels."""
        self.parameter_panel = parameter_panel
        self.data_panel = data_panel
```

- [ ] **Step 2.3: Simplify `freeze_ui`/`unfreeze_ui`**

Replace the `freeze_ui`/`unfreeze_ui` methods (`:1054`–`:1068`) with:

```python
    def freeze_ui(self):
        """Signal that interactive UI elements should be disabled."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal that interactive UI elements should be re-enabled."""
        self.ui_frozen.emit(False)
```

- [ ] **Step 2.4: Drop the panel fallback in `toggle_preview`**

In `toggle_preview` (`:917`), the `except` block (`:929`–`:931`) reads `preview_check` off the panel as a fallback. Replace:

```python
            preview_check = getattr(self, "preview_check", None)
            if preview_check is None and self.parameter_panel is not None:
                preview_check = getattr(self.parameter_panel, "preview_check", None)
            if preview_check is not None:
                preview_check.setChecked(False)
```

with:

```python
            preview_check = getattr(self, "preview_check", None)
            if preview_check is not None:
                preview_check.setChecked(False)
```

- [ ] **Step 2.5: Run controller contract tests**

```bash
pytest tests/test_preprocessing_ownership.py::test_controller_has_no_panel_attributes tests/test_preprocessing_ownership.py::test_controller_freeze_emits_signal_without_panels -v
```

Expected: both PASS.

- [ ] **Step 2.6: Run the preprocessing analysis tests (no regressions)**

```bash
pytest tests/test_preprocessing_analysis.py -v
```

Expected: all PASS. If any test calls `controller.set_panels(...)` or asserts `controller.parameter_panel`, update it to remove that call/assertion (the controller no longer owns panels). Re-run until green.

- [ ] **Step 2.7: Commit**

```bash
git add napariTFM/widgets/preprocessing_widget.py tests/test_preprocessing_analysis.py
git commit -m "Sever preprocessing controller from embedded panels

Controller no longer holds parameter_panel/data_panel or set_panels;
freeze_ui/unfreeze_ui emit ui_frozen only. Parameter-driven preview
still flows through ParameterManager.parameter_changed."
```

---

## Task 3: Sever the widget from `PreprocessingParameterPanel`

**Files:**
- Modify: `napariTFM/widgets/preprocessing_widget.py` (`PreprocessingWidget`)

Stop constructing/placing/wiring the parameter panel. Preview-on-parameter-change is already handled by the controller via `ParameterManager`, so the widget's panel-driven duplicates are removed. The widget keeps `preview_check`, `process_btn`, `cancel_btn`, status frame, and the `ui_frozen` handler.

- [ ] **Step 3.1: Remove panel construction and `set_panels` call**

In `PreprocessingWidget.__init__` (`:1079`), replace this block (`:1091`–`:1103`):

```python
        # Initialize panels
        self.parameter_panel = PreprocessingParameterPanel(parameter_manager)
        self.data_panel = None

        # Initialize controller
        self.controller = PreprocessingController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager
        )

        self.controller.set_panels(self.parameter_panel, None)
```

with:

```python
        # Initialize controller (parameters flow through ParameterManager;
        # there is no embedded parameter/data panel — the shell's
        # WorkflowParameterPanel is the sole preprocessing parameter editor).
        self.controller = PreprocessingController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager
        )
```

- [ ] **Step 3.2: Remove the panel from the layout**

In `_create_content_container` (`:1124`), delete the parameter-panel placement (`:1139`):

```python
        # Add components
        layout.addWidget(self.parameter_panel)
```

Leave the rest of the method (preview frame, action frame, status frame) intact.

- [ ] **Step 3.3: Remove panel signal wiring in `_connect_signals`**

In `_connect_signals` (`:1200`), delete the three panel-coupled lines (`:1216`, `:1217`-pair, `:1218`). Specifically remove:

```python
        # Connect parameter panel changes
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)
        self.parameter_panel.parameters_reset.connect(self._on_parameters_reset)
```

Replace with:

```python
        # Parameter-change-driven preview is handled by the controller via
        # ParameterManager.parameter_changed; no panel wiring needed here.
```

> Note: we intentionally drop `parameter_manager.parameter_changed → _sync_parameter` because `_sync_parameter` only pushed values into the now-deleted panel.

- [ ] **Step 3.4: Delete the now-orphaned widget methods**

Delete `_sync_parameter` (`:1223`–`:1226`), `_on_parameter_changed` (`:1241`–`:1244`), and `_on_parameters_reset` (`:1246`–`:1248`). They only existed to bridge the panel; the controller covers preview refresh, and reset status messaging is no longer panel-driven.

Verify nothing else references them:

```bash
grep -n "_sync_parameter\|_on_parameter_changed\|_on_parameters_reset" napariTFM/widgets/preprocessing_widget.py
```

Expected: no matches after deletion.

- [ ] **Step 3.5: Run the ownership + analysis tests**

```bash
pytest tests/test_preprocessing_ownership.py tests/test_preprocessing_analysis.py -v
```

Expected: `test_controller_*` PASS; `test_*_panel_class_is_removed` still FAIL (classes not deleted yet — Task 4); analysis tests PASS. If an analysis test references `widget.parameter_panel`, update it to remove that reference.

- [ ] **Step 3.6: Commit**

```bash
git add napariTFM/widgets/preprocessing_widget.py tests/test_preprocessing_analysis.py
git commit -m "Stop PreprocessingWidget from creating/wiring its parameter panel

The shell's WorkflowParameterPanel is now the only preprocessing
parameter editor. Preview refresh on parameter change is handled by
the controller via ParameterManager."
```

---

## Task 4: Delete the panel classes

**Files:**
- Modify: `napariTFM/widgets/preprocessing_widget.py`

Remove the two dead/duplicated classes and prune unused imports.

- [ ] **Step 4.1: Delete `PreprocessingDataPanel`**

Delete the entire `class PreprocessingDataPanel(QWidget):` block (`:26`–`:137`, ending just before `class PreprocessingParameterPanel`).

- [ ] **Step 4.2: Delete `PreprocessingParameterPanel`**

Delete the entire `class PreprocessingParameterPanel(QWidget):` block (`:139`–`:733`, ending just before `class PreprocessingController`).

- [ ] **Step 4.3: Prune unused imports**

Run a quick check for imports that were only used by the deleted classes:

```bash
grep -nE "QRangeSlider|QSlider|QLineEdit|QComboBox|QGridLayout" napariTFM/widgets/preprocessing_widget.py
```

For each symbol with **no** remaining usage outside the import line, remove it from the `from qtpy...` / `from qtrangeslider...` imports at the top (`:11`–`:19`). Do not remove symbols still used by `PreprocessingController`/`PreprocessingWidget` (e.g. `QCheckBox`, `QPushButton`, `QProgressBar`, `QLabel`, `QFrame`, `QScrollArea`, `QVBoxLayout`, `QHBoxLayout`, `QWidget`, `QSizePolicy`, `Qt`). If `qtrangeslider` import (`:19`) is now unused, remove that line.

- [ ] **Step 4.4: Verify the module imports cleanly**

```bash
python -c "import napariTFM.widgets.preprocessing_widget as m; print(hasattr(m,'PreprocessingParameterPanel'), hasattr(m,'PreprocessingDataPanel'))"
```

Expected: `False False`.

- [ ] **Step 4.5: Run the full ownership test file**

```bash
pytest tests/test_preprocessing_ownership.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 4.6: Commit**

```bash
git add napariTFM/widgets/preprocessing_widget.py
git commit -m "Delete PreprocessingParameterPanel and PreprocessingDataPanel

Both were duplicated/dead: data panel was never instantiated, and the
parameter panel duplicated the shell's WorkflowParameterPanel. Pruned
imports left unused after their removal."
```

---

## Task 5: Update redesign tests and lock shell tolerance

**Files:**
- Modify: `tests/test_preprocessing_ui_redesign.py`
- Modify: `tests/test_workflow_shell.py`

- [ ] **Step 5.1: Inspect redesign-test references to the deleted panels**

```bash
grep -nE "PreprocessingParameterPanel|PreprocessingDataPanel|\.parameter_panel|\.data_panel" tests/test_preprocessing_ui_redesign.py
```

For each hit:
- If the assertion verifies the panel exists / is visible / counts its controls → delete that assertion (the panel is gone; the shell's `WorkflowParameterPanel` covers parameters and is tested in `tests/test_workflow_shell.py`).
- If it constructs `PreprocessingParameterPanel(...)` directly → delete the test; its concern (parameter controls wired to `ParameterManager`) is already covered by `test_workflow_parameter_panel_*` in `tests/test_workflow_shell.py`.
- If it references `widget.parameter_panel` for a non-parameter reason → replace with the equivalent shell panel or remove if obsolete.

Make the edits, keeping every other assertion intact.

- [ ] **Step 5.2: Run the redesign tests**

```bash
pytest tests/test_preprocessing_ui_redesign.py -v
```

Expected: all PASS (after the edits above).

- [ ] **Step 5.3: Add a shell-tolerance regression test**

In `tests/test_workflow_shell.py`, add a test that `_hide_embedded_parameter_panels` does not assume a `parameter_panel` attribute exists on a stage widget (so a fully-inverted widget is safe). Append:

```python
def test_hide_embedded_parameter_panels_tolerates_missing_attribute(monkeypatch, app):
    class _NoPanelStage(_StubStageWidget):
        def __init__(self, *args):
            super().__init__(*args)
            del self.parameter_panel  # simulate a fully-inverted stage widget

    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _NoPanelStage)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())  # must not raise

    assert not hasattr(widget.preprocessing_widget, "parameter_panel")
```

- [ ] **Step 5.4: Run the shell tests**

```bash
pytest tests/test_workflow_shell.py -v
```

Expected: all PASS, including the new tolerance test. If `_hide_embedded_parameter_panels` raises because the loop body assumes the attribute, confirm it uses `getattr(widget, "parameter_panel", None)` (it does today at `_widget.py:565`); if a future edit broke that, restore the `getattr` guard.

- [ ] **Step 5.5: Run the full suite**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 5.6: Commit**

```bash
git add tests/test_preprocessing_ui_redesign.py tests/test_workflow_shell.py
git commit -m "Update preprocessing tests for panel deletion; lock shell tolerance"
```

---

## Task 6: Manual smoke test

**Files:** none (verification only).

- [ ] **Step 6.1: Launch napari and exercise preprocessing**

```bash
python -m napari --plugin napariTFM
```

Verify:
1. The shell loads without error; the Preprocessing section is present and expandable.
2. Clicking the params (`⚙`) icon expands the inner "Parameters" section with preprocessing parameter controls (these come from the shell's `WorkflowParameterPanel`, not the deleted widget panel).
3. Load a bead stack + reference via the data-status row load (`↑`) buttons; the run icon becomes enabled.
4. Toggling "Show Preview" and editing a preprocessing parameter updates the preview (confirms the `ParameterManager`-driven preview path survived).
5. Running preprocessing: run icon swaps to cancel while running, then back; progress/status update; outputs appear and their data-status rows flip to ✓.

If the GUI cannot be launched in this environment, record exactly which checks were skipped — do not claim success on unverified items.

- [ ] **Step 6.2: No commit unless code changed.** If smoke testing surfaced a fix, make it as a new commit with its own message.

---

## Self-Review

**Spec coverage (against the chosen scope: full re-architecture, delete panels, preprocessing first):**
- Delete duplicated panels → Task 4 (both classes). ✓
- Controller no longer depends on panels → Task 2. ✓
- Widget no longer builds/wires panels; shell `WorkflowParameterPanel` is sole editor → Task 3. ✓
- Parameter ownership stays with `ParameterManager` → preview path verified to flow through it (Tasks 2/3); no parameter logic moved into a new owner. ✓
- Reflection proxy (`_find_stage_action_targets`): **out of scope for this plan.** Preprocessing keeps `process_btn`/`preview_btn`(n/a)/`cancel_btn` on the widget as the stable action contract; removing the string-path reflection is deferred to the displacement/force/stress plans where `ActionPanel` deletion forces the issue, then a final "drop reflection" cleanup. Noted here so it isn't assumed done.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Each code step shows exact code or an exact grep+rule. Task 5.1 uses a rule-per-match form rather than literal code because the target test file content was not read line-by-line in planning; the rule is concrete (delete/replace per category) and bounded by a grep.

**Type/name consistency:** `ui_frozen` signal name unchanged; `preview_check`, `process_btn`, `cancel_btn`, `_handle_ui_freeze`, `_has_required_data` all retained as referenced. Deleted symbols (`set_panels`, `_sync_parameter`, `_on_parameter_changed`, `_on_parameters_reset`, `parameter_panel`, `data_panel`, `PreprocessingParameterPanel`, `PreprocessingDataPanel`) are each removed at their definition and checked for stragglers via grep (Steps 3.4, 4.4).

**Known follow-ons (separate plans):** displacement/force/stress inversions (each deletes its `*ActionPanel` + `*ParameterPanel` and re-homes `freeze_ui`/`update_button_states`/`preview_checkbox`/`parameter_widgets` coupling), then drop `_find_stage_action_targets` reflection in favor of explicit action attributes; then the remaining Tier 3 workstreams (disk-derived status, theming, config persistence).
```
