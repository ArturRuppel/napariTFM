# Tier 4: State Architecture — Param Consolidation, Single Refresh, Config Round-Trip

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining gap between napariTFM's CellFlow-shaped *front* and its in-memory *back*. Three concerns, in order:

- **A — One parameter system.** Remove the now-dead bridging scaffold so there is unambiguously a single visible parameter editor.
- **B — One refresh path.** Replace the five near-identical completion/data-change fan-out handlers (each looping all five widgets) with a single `refresh()` method.
- **C — Serializable config + reconcile-from-disk.** Give the shell `get_state`/`set_state` and a sticky `napariTFM_config.json` in `output_dir`, loaded when a dir is chosen — CellFlow's anchor, adapted to single-dataset.

**Scope decisions (owner, 2026-05-29):**
- **Single-dataset model**, not CellFlow's multi-position (`pos00`/`pos01`). Keep the one-`output_dir` model; config lives *inside* that dir. No position selector.
- **One combined plan** (this document), phased A → B → C.
- **`stale` status remains out of scope** (deferred, as in Tier 3e).

**Architecture:**
- Concern A is mostly *already done* by Tier 3 (see Findings). What remains is deleting `napariTFMWidget._hide_embedded_parameter_panels`, which is dead under real widgets.
- Concern B introduces `napariTFMWidget._stage_widgets()` (the canonical list) and `napariTFMWidget.refresh()` (`_update_ui_state` on each widget + `refresh_stage_statuses()`). The DataManager change callback and the four `*_completed` signals all route through it.
- Concern C adds `get_state()`/`set_state()` on the shell (parameters via `ParameterManager.get_all_parameters()`/`set_parameter`, plus `output_dir` for reference). A new `output_dir_changed` signal on the Project section drives `_reconcile_to_output_dir()`: if a config file exists in the dir, `set_state` from it; otherwise write the current state to claim the dir. `refresh()` (from B) is reused after applying state.

**Tech Stack:** Python, qtpy/PyQt, pytest (`QT_QPA_PLATFORM=offscreen`). Config serialization uses stdlib `json`.

**Line endings:** All target files are pure LF today. Edit with normal tools but verify each commit with `git diff -w` (must match `git diff`) to confirm no CRLF churn — `batch_analysis.py`/widget files have a history of mixed-line-ending churn.

---

## Findings (verified against current code — do NOT redo)

Confirmed by reading the tree at this commit:

1. **The stage widgets no longer own parameter panels.** `grep "self.parameter_panel"` across `preprocessing_widget.py`, `displacement_analysis_widget.py`, `fttc_widget.py`, `msm_widget.py` returns nothing — Tier 3a–3d deleted them. The ownership tests (`tests/test_*_ownership.py::test_parameter_panel_class_is_removed`) lock this.
2. **The batch widget has no parameter QGroupBoxes.** Its groups are `Metadata`, `File Paths`, `Analysis Steps`, `Visualizations`, `Folder Management` — none of the titles `_hide_embedded_parameter_panels` tries to hide (`"General Parameters"`, `"Preprocessing Parameters"`, `"Farneback Displacement Parameters"`, `"Force Parameters"`, `"Stress Parameters"`). The batch widget already reads parameters from the shared `ParameterManager`.
3. **Therefore `_hide_embedded_parameter_panels` is dead under real widgets** — its first loop finds no `parameter_panel`; its second loop matches no group. (Under the `_StubStageWidget` harness the first loop *would* hide the stub's throwaway `parameter_panel`, but that panel is never mounted in the shell tree and no test asserts its visibility — removal is safe.)
4. **The single visible parameter editor is `WorkflowParameterPanel`**, instantiated per stage in `_create_stage_parameter_panels()` and mounted as nested "Parameters" sub-sections via `StageSection.add_inner_section` (see `napariTFMWidget.__init__`, the `_stage_inner_param_sections_by_key` loop).
5. **No `get_state`/`set_state` exists anywhere** — Concern C is greenfield.
6. **The batch widget already has its own YAML config** (`save_config_btn`/`load_config_btn`, `_apply_config_parameters`, with `parameters`/`visualizations`/`root_folders`). This is a *batch-job spec* (which folders/steps to run), semantically distinct from the interactive single-dataset state. **Do not modify it** — Concern C's JSON config is a separate, additive store. The two coexisting is acceptable; unifying them is explicitly out of scope.
7. **No test pins the shell's `_on_*_completed` or `_on_pipeline_data_changed` methods by name** (`grep` confirms). Tests call `refresh_stage_statuses()` directly. So B is free to collapse those handlers.

---

## File Structure

- `napariTFM/widgets/_widget.py` — **primary target.** Remove `_hide_embedded_parameter_panels` (A). Add `_stage_widgets()` + `refresh()`; reroute completion/data-change handlers through it (B). Add `get_state`/`set_state`, `_config_path`/`_read_config`/`_write_config`/`_reconcile_to_output_dir`, wire `output_dir_changed` and hook `_write_config` into `_save_generated_artifact` (C).
- `napariTFM/widgets/_project_section.py` — add `output_dir_changed = Signal()` to `_GeneralBody`, emit it after `set_output_dir` in `_choose_output_dir` (C).
- `tests/test_workflow_shell.py` — add tests for B (single `refresh()`) and C (round-trip, reconcile-load, reconcile-write). Reuses the existing `app` fixture and `_Stub*` harness; C tests monkeypatch in the **real** `DataManager` and `ParameterManager`.

No new test files needed; all additions land in `test_workflow_shell.py` next to the existing shell tests.

---

# Phase A — One parameter system

### Task A1: Remove the dead `_hide_embedded_parameter_panels`

**Files:**
- Modify: `napariTFM/widgets/_widget.py` — delete the method and its call site.
- Test: `tests/test_workflow_shell.py` — add one lock test.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_workflow_shell.py` (near the other `test_main_widget_*` tests). This locks that each pipeline stage has exactly one parameter editor home (the nested inner section) and that the dead method is gone:

```python
def test_each_stage_has_single_inline_parameter_editor(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    # The four parameterized stages each expose exactly one inline editor.
    assert set(widget._stage_inner_param_sections_by_key) == {
        "preprocessing", "displacement", "force", "stress",
    }
    # The dead bridging scaffold is gone.
    assert not hasattr(widget, "_hide_embedded_parameter_panels")
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_each_stage_has_single_inline_parameter_editor" -v`
Expected: FAIL on the `not hasattr(...)` assertion (the method still exists).

- [ ] **Step 3: Remove the method and its call**

In `napariTFM/widgets/_widget.py`:
- Delete the call `self._hide_embedded_parameter_panels()` inside `__init__` (currently right after `self.batch_widget = BatchAnalysisWidget(...)`).
- Delete the entire `_hide_embedded_parameter_panels` method definition.
- If `QGroupBox` is now unused in the file, remove it from the `qtpy.QtWidgets` import. (`WorkflowParameterPanel._setup_ui` still uses `QGroupBox`, so it likely stays — verify with `grep "QGroupBox" napariTFM/widgets/_widget.py` and only remove the import if zero hits remain.)

- [ ] **Step 4: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_each_stage_has_single_inline_parameter_editor" -v`
Expected: PASS.

- [ ] **Step 5: Run the full shell test file (no regression)**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_workflow_shell.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Remove dead parameter-panel bridging scaffold"
```

---

# Phase B — One refresh path

### Task B1: Introduce `refresh()` and collapse the fan-out handlers

**Files:**
- Modify: `napariTFM/widgets/_widget.py` — add `_stage_widgets()` + `refresh()`; reroute `connect_signals`, the data-change callback, `_clear_all_data`, `_reset_parameters`.
- Test: `tests/test_workflow_shell.py` — add a call-counting test.

Current shape to replace: `_on_pipeline_data_changed`, `_on_preprocessing_completed`, `_on_displacement_completed`, `_on_force_completed`, `_on_stress_completed` each contain the same five-widget `_update_ui_state()` loop + `refresh_stage_statuses()`. `_reset_parameters` and `_clear_all_data` repeat the five-widget literal list too.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_workflow_shell.py`. The `_StubStageWidget` already increments `update_count` in its `_update_ui_state`, so we can assert one refresh touches every widget once and a completion signal triggers exactly one refresh:

```python
def test_refresh_updates_every_stage_widget_once(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    stage_widgets = widget._stage_widgets()
    assert len(stage_widgets) == 5

    before = [w.update_count for w in stage_widgets]
    widget.refresh()
    after = [w.update_count for w in stage_widgets]
    assert all(a == b + 1 for a, b in zip(after, before))


def test_completion_signal_triggers_single_refresh(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    calls = {"n": 0}
    original = widget.refresh
    widget.refresh = lambda: (calls.__setitem__("n", calls["n"] + 1), original())[1]

    widget.force_widget.force_calculated.emit(object())
    assert calls["n"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_refresh_updates_every_stage_widget_once" "tests/test_workflow_shell.py::test_completion_signal_triggers_single_refresh" -v`
Expected: FAIL — `_stage_widgets`/`refresh` don't exist yet (AttributeError).

- [ ] **Step 3: Implement `_stage_widgets()` + `refresh()`**

In `napariTFMWidget`, add:

```python
    def _stage_widgets(self):
        return [
            self.preprocessing_widget,
            self.displacement_widget,
            self.force_widget,
            self.msm_widget,
            self.batch_widget,
        ]

    def refresh(self):
        """Single reconcile pass: update every stage widget, then statuses."""
        for widget in self._stage_widgets():
            update = getattr(widget, "_update_ui_state", None)
            if callable(update):
                update()
        self.refresh_stage_statuses()
```

- [ ] **Step 4: Reroute every fan-out through `refresh()`**

- Delete `_on_preprocessing_completed`, `_on_displacement_completed`, `_on_force_completed`, `_on_stress_completed`, and `_on_pipeline_data_changed`.
- In `connect_signals`, connect all four completion signals directly to `self.refresh`:

```python
    def connect_signals(self):
        self.preprocessing_widget.preprocessing_completed.connect(lambda *_: self.refresh())
        self.displacement_widget.displacement_calculated.connect(lambda *_: self.refresh())
        self.force_widget.force_calculated.connect(lambda *_: self.refresh())
        self.msm_widget.stress_calculated.connect(lambda *_: self.refresh())
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
```

  (The `lambda *_:` absorbs the emitted `results` payload — see the PyQt `checked`-arg gotcha noted in the project's reflection-shim history.)
- In `__init__`, change the data-change registration from `self.data_manager.add_change_callback(self._on_pipeline_data_changed)` to `self.data_manager.add_change_callback(self.refresh)`.
- In `_clear_all_data`, after `self.data_manager.__init__()`, change the re-registration to `self.data_manager.add_change_callback(self.refresh)` and replace the five-widget `_update_ui_state` block + `refresh_stage_statuses()` with a single `self.refresh()`.
- In `_reset_parameters`, replace the five-widget `_update_ui_state` block with `self.refresh()`.
- Leave `_on_parameter_changed` in place for now (it handles calibration propagation, a param→widget concern distinct from completion refresh). You *may* simplify its body to iterate `self._stage_widgets()` instead of the literal list, but that is optional cleanup — keep its existing behavior.

- [ ] **Step 5: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_refresh_updates_every_stage_widget_once" "tests/test_workflow_shell.py::test_completion_signal_triggers_single_refresh" -v`
Expected: PASS.

- [ ] **Step 6: Run the full shell test file (no regression)**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_workflow_shell.py -v`
Expected: all PASS — in particular `test_data_manager_change_callback_refreshes_stage_widgets` (it drove the old `_on_pipeline_data_changed`; the `self.refresh` callback satisfies it). If that test asserts the *method* `_on_pipeline_data_changed` exists, update it to assert on `refresh` behavior instead.

- [ ] **Step 7: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Collapse fan-out completion handlers into single refresh()"
```

---

# Phase C — Serializable config + reconcile-from-disk

### Task C1: Shell `get_state` / `set_state`

**Files:**
- Modify: `napariTFM/widgets/_widget.py` — add module constants + `get_state`/`set_state`.
- Test: `tests/test_workflow_shell.py` — round-trip test with the real `ParameterManager`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_workflow_shell.py`. This monkeypatches in the **real** `ParameterManager` and `DataManager` (stub stage widgets are fine):

```python
def test_get_set_state_round_trips_parameters(monkeypatch, app, tmp_path):
    from napariTFM.utilities.data_manager import DataManager
    from napariTFM.utilities.parameter_manager import ParameterManager

    monkeypatch.setattr(_widget, "DataManager", DataManager)
    monkeypatch.setattr(_widget, "ParameterManager", ParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)

    widget.parameter_manager.set_parameter("rolling_ball_radius", 7)
    state = widget.get_state()
    assert state["parameters"]["rolling_ball_radius"] == 7
    assert state["output_dir"] == str(tmp_path)

    widget.parameter_manager.set_parameter("rolling_ball_radius", 0)
    widget.set_state(state)
    assert widget.parameter_manager.get_parameter("rolling_ball_radius") == 7
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_get_set_state_round_trips_parameters" -v`
Expected: FAIL — `get_state` does not exist.

- [ ] **Step 3: Implement state serialization**

At module top of `napariTFM/widgets/_widget.py` (after imports), add:

```python
CONFIG_FILENAME = "napariTFM_config.json"
STATE_VERSION = 1
```

Add `import json` to the imports.

In `napariTFMWidget`, add:

```python
    def get_state(self) -> dict:
        output_dir = self.data_manager.output_dir
        return {
            "version": STATE_VERSION,
            "parameters": self.parameter_manager.get_all_parameters(),
            "output_dir": str(output_dir) if output_dir else None,
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self._applying_state = True
        try:
            params = state.get("parameters", {})
            if isinstance(params, dict):
                valid = set(self.parameter_manager.get_all_parameters())
                for name, value in params.items():
                    if name not in valid:
                        continue
                    try:
                        if name == "registration_mode" and isinstance(value, str):
                            value = value.lower()
                        self.parameter_manager.set_parameter(name, value)
                    except Exception as exc:
                        logger.warning("Skipped parameter %s: %s", name, exc)
            # output_dir is intentionally NOT re-applied: the config lives
            # inside output_dir, so the dir is already known when we load it.
        finally:
            self._applying_state = False
        self.refresh()
```

Initialize the guard in `__init__` (before any code that could write config — put it at the very top of `__init__`, right after `super().__init__()`):

```python
        self._applying_state = False
```

Note: `get_all_parameters()` returns values in *internal* units (Young's modulus in Pa, `regularization` as the raw value, `gel_height` as `0` for infinity), and `set_parameter` consumes the same internal units — so the round-trip is unit-consistent without UI conversions. The unknown-key skip mirrors the batch widget's `_apply_config_parameters`, keeping older config files loadable.

- [ ] **Step 4: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_get_set_state_round_trips_parameters" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Add shell get_state/set_state for UI parameter round-trip"
```

---

### Task C2: Config file + reconcile when the output dir changes

**Files:**
- Modify: `napariTFM/widgets/_project_section.py` — `output_dir_changed` signal on `_GeneralBody`.
- Modify: `napariTFM/widgets/_widget.py` — config I/O helpers, `_reconcile_to_output_dir`, wire the signal, hook `_write_config` into `_save_generated_artifact`.
- Test: `tests/test_workflow_shell.py` — reconcile-load and reconcile-write tests.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_workflow_shell.py`:

```python
def test_reconcile_loads_existing_config_from_output_dir(monkeypatch, app, tmp_path):
    import json
    from napariTFM.utilities.data_manager import DataManager
    from napariTFM.utilities.parameter_manager import ParameterManager

    monkeypatch.setattr(_widget, "DataManager", DataManager)
    monkeypatch.setattr(_widget, "ParameterManager", ParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    (tmp_path / "napariTFM_config.json").write_text(
        json.dumps({"version": 1, "parameters": {"rolling_ball_radius": 9},
                    "output_dir": str(tmp_path)})
    )

    widget.data_manager.set_output_dir(tmp_path)
    widget.project_section.body.output_dir_changed.emit()  # simulate dir chosen

    assert widget.parameter_manager.get_parameter("rolling_ball_radius") == 9


def test_reconcile_writes_config_when_absent(monkeypatch, app, tmp_path):
    from napariTFM.utilities.data_manager import DataManager
    from napariTFM.utilities.parameter_manager import ParameterManager

    monkeypatch.setattr(_widget, "DataManager", DataManager)
    monkeypatch.setattr(_widget, "ParameterManager", ParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)
    widget.project_section.body.output_dir_changed.emit()

    assert (tmp_path / "napariTFM_config.json").exists()
```

- [ ] **Step 2: Run to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_reconcile_loads_existing_config_from_output_dir" "tests/test_workflow_shell.py::test_reconcile_writes_config_when_absent" -v`
Expected: FAIL — `body.output_dir_changed` and the reconcile wiring don't exist (AttributeError).

- [ ] **Step 3: Add the `output_dir_changed` signal**

In `napariTFM/widgets/_project_section.py`:
- Add `Signal` to the `qtpy.QtCore` import: `from qtpy.QtCore import Signal`.
- On `_GeneralBody`, declare the signal at class scope: `output_dir_changed = Signal()`.
- In `_choose_output_dir`, after `self.data_manager.set_output_dir(path)`, add `self.output_dir_changed.emit()`.

(`ProjectSection` already exposes `body`, so the shell reaches the signal via `self.project_section.body.output_dir_changed`. No new property needed.)

- [ ] **Step 4: Add config I/O + reconcile, and wire it**

In `napariTFMWidget`, add:

```python
    def _config_path(self):
        output_dir = self.data_manager.output_dir
        return (output_dir / CONFIG_FILENAME) if output_dir else None

    def _read_config(self):
        path = self._config_path()
        if path is None or not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read config %s: %s", path, exc)
            return None

    def _write_config(self):
        if self._applying_state:
            return
        path = self._config_path()
        if path is None:
            return
        try:
            self.data_manager.ensure_output_dir()
            with open(path, "w") as f:
                json.dump(self.get_state(), f, indent=2)
        except Exception as exc:
            logger.warning("Failed to write config %s: %s", path, exc)

    def _reconcile_to_output_dir(self):
        """On a new output dir: load its config if present, else claim it."""
        state = self._read_config()
        if state is not None:
            self.set_state(state)   # set_state() calls refresh()
        else:
            self._write_config()
            self.refresh()
```

Wire the signal in `__init__` (after `self.project_section` is constructed and after the managers exist — place it next to the other Project-section button connections):

```python
        self.project_section.body.output_dir_changed.connect(self._reconcile_to_output_dir)
```

Hook config persistence into result saves so the dir's sticky state tracks its outputs. In `_save_generated_artifact`, inside the `finally:` block, after `self.refresh_stage_statuses()`, add:

```python
            self._write_config()
```

- [ ] **Step 5: Run to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_reconcile_loads_existing_config_from_output_dir" "tests/test_workflow_shell.py::test_reconcile_writes_config_when_absent" -v`
Expected: PASS.

- [ ] **Step 6: Run the full shell + project-section test files (no regression)**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_workflow_shell.py tests/test_project_section.py -v`
Expected: all PASS. (The added `output_dir_changed` signal is inert for the existing `_StubDataManager`-driven project tests, which never call `_choose_output_dir`.)

- [ ] **Step 7: Commit**

```bash
git add napariTFM/widgets/_widget.py napariTFM/widgets/_project_section.py tests/test_workflow_shell.py
git commit -m "Add config round-trip and reconcile-on-output-dir-change"
```

---

# Phase D — Verification

### Task D1: Full-suite + clean diffs + manual smoke

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest`
Expected: all PASS (145 prior + the 6 new tests). Known env flake: `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` may intermittently SIGSEGV in its spawned subprocess (napari/pydantic manifest race) — re-run in isolation to confirm it is the flake and not a regression.

- [ ] **Step 2: Confirm no line-ending churn**

Run `git diff -w --stat` and compare to `git diff --stat` for the branch range (must match). Confirm each changed file has zero CR: `grep -rc $'\r' napariTFM/widgets/_widget.py napariTFM/widgets/_project_section.py` returns 0.

- [ ] **Step 3: Manual smoke (needs napari — owner runs)**

Launch napari, add the napariTFM widget, then:
1. Choose an empty output dir → a `napariTFM_config.json` appears; stages read from disk.
2. Change a few parameters across stages; run preprocessing on test data → output file saved, config rewritten, stage flips to `done`.
3. Close and relaunch; choose the **same** dir → parameters restore from config and completed stages show `done` without loading anything into memory.
4. Confirm there is exactly one parameter editor per stage (the nested "Parameters" sub-section) — no duplicate/hidden editors.

---

## Self-Review

**Spec coverage:**
- A — one param system: Task A1 removes the dead `_hide_embedded_parameter_panels`; the lock test asserts exactly four single inline editors and the method's absence. The deeper consolidation was already delivered by Tier 3 (see Findings). ✔
- B — one refresh: Task B1 adds `_stage_widgets()` + `refresh()`, deletes five fan-out handlers, routes completion signals + the data-change callback + clear/reset through `refresh()`. Tests assert one refresh touches all five widgets once and a completion signal triggers exactly one refresh. ✔
- C — config round-trip + reconcile: C1 adds `get_state`/`set_state` (round-trip test); C2 adds the JSON config, the `output_dir_changed` signal, `_reconcile_to_output_dir` (load-else-claim), and the save-time `_write_config` hook (load + write tests). ✔
- Single-dataset (no position): no position concept introduced; config lives in the one `output_dir`. ✔
- `stale` excluded: no task computes `stale`. ✔

**Re-entrancy:** `set_state` sets `self._applying_state` while applying, and `_write_config` early-returns under that guard — so loading a config never rewrites it mid-apply. `set_state` deliberately does **not** re-apply `output_dir`, so it cannot recurse into the dir-change path. The `set_output_dir` → data-change → `refresh` path and the `output_dir_changed` → `_reconcile_to_output_dir` → `refresh` path can both fire on a single dir choice (two refreshes); `refresh()` is idempotent, so this is harmless.

**Deliberate boundaries (flag for reviewer/owner):**
- The new JSON config stores **parameters + output_dir only** — not transient batch-run selections (which folders/steps/visualizations to run). Those remain in the batch widget's separate YAML config, which is a batch-*job* spec and is intentionally untouched. If the owner wants the JSON config to also capture batch-run selections, that is a follow-up (would need `get_state`/`set_state` on `BatchAnalysisWidget`).
- The existing "Save/Load Parameters" YAML buttons (via `ParameterManager`) are left intact as a portable param-export mechanism, distinct from the sticky per-dir JSON config. Three persistence surfaces now exist (param YAML, batch-job YAML, dir JSON); unifying them is explicitly out of Tier 4 scope.

**Type consistency:** `get_state` → `dict`; `set_state(state: dict) -> None`; `_config_path` → `Path | None`; `_read_config` → `dict | None`; `_write_config`/`_reconcile_to_output_dir`/`refresh` → `None`; `_stage_widgets` → `list[QWidget]`. `output_dir_changed` is a no-arg `Signal()`.

**Placeholder scan:** none.

**Not addressed (acknowledged, not in scope):** the two section primitives (`StageSection` vs CellFlow's `CollapsibleSection`) remain separate; `_ui_style.py` remains a single-palette subset of CellFlow's theme system; `stale` detection deferred. These were gap items #4/#5 from the UI assessment, outside the three agreed Tier 4 concerns.
