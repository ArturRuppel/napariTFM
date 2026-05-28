# DataManager File-As-Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a *generated* artifact's availability (and therefore each stage's `done`/`ready`/`not_started` status) derive from whether its file exists in `output_dir`, not from an in-memory value. Raw inputs keep in-memory availability. `stale` detection is explicitly out of scope.

**Architecture:** Files on disk become the source of truth for generated artifacts. `DataManager.artifact_available(key)` returns disk existence for keys in `GENERATED_FILENAMES`; the in-memory `ArtifactState.value` becomes a lazy cache used only for display (shape text). `StageDataStatusPanel` asks `DataManager.artifact_available` instead of testing `value is not None`, so a project folder that already contains output files reads as `done` on launch even before anything is loaded into memory. Raw inputs (`bead_stack`/`reference`/`cell_stack`/`mask_stack`) are not in `GENERATED_FILENAMES`, so they keep their current value-based availability.

**Why this reverses a documented decision:** `docs/cellflow-ui-concept-for-naparitfm.md` cautioned against making stages file-backed "if DataManager is the live source of truth." The project owner has decided the file *is* now the ground truth, so that caution no longer applies for generated artifacts.

**Tech Stack:** Python, qtpy/PyQt, pytest (`QT_QPA_PLATFORM=offscreen`).

**Line endings:** All target files are pure LF today. Edit with normal tools but verify each commit with `git diff -w` to confirm no CRLF churn (see the project's mixed-line-endings history).

---

## File Structure

- `napariTFM/utilities/data_manager.py` — gains `artifact_disk_path(key)`; `artifact_available(key)` becomes disk-aware for generated keys. Single responsibility: own pipeline artifact state + now disk truth.
- `napariTFM/widgets/_stage_data_status.py` — `StageDataStatusPanel.refresh()` and `_ArtifactRow.refresh_state()` consume a disk-aware `available`; `_info_text` distinguishes "loaded in memory" from "on disk only".
- `tests/test_data_manager_disk_truth.py` (new) — unit tests for the DataManager disk logic.
- `tests/test_workflow_shell.py` — one new integration test driving a real `DataManager` + tmp `output_dir` with real files.
- `tests/test_artifact_row.py` — one new row test for the disk-only ("Saved") display.

Existing stub-driven test `test_stage_data_status_refreshes_from_data_manager` keeps passing unchanged: `_StubDataManager` has no `artifact_available`, so the panel falls back to value-based availability for it.

---

### Task 1: DataManager disk-aware availability

**Files:**
- Modify: `napariTFM/utilities/data_manager.py:95-96` (the current `artifact_available`)
- Test: `tests/test_data_manager_disk_truth.py` (create)

Current code to replace:

```python
    def artifact_available(self, key: str) -> bool:
        return self.get_artifact(key).available
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_manager_disk_truth.py`:

```python
import numpy as np

from napariTFM.utilities.data_manager import DataManager


def test_generated_artifact_available_follows_disk(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)

    assert dm.artifact_available("displacement_results") is False

    (tmp_path / "displacement_results.npy").write_bytes(b"x")
    assert dm.artifact_available("displacement_results") is True


def test_generated_artifact_unavailable_without_output_dir(tmp_path):
    dm = DataManager()
    # No output dir set: nothing can be on disk yet.
    assert dm.artifact_available("force_results") is False


def test_generated_artifact_ignores_in_memory_value(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)
    # An in-memory value must NOT make a generated artifact "available"
    # when its file is absent; disk is the ground truth.
    dm.get_artifact("force_results").value = object()
    assert dm.artifact_available("force_results") is False


def test_raw_input_available_follows_memory(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)

    assert dm.artifact_available("bead_stack") is False
    dm.set_bead_stack(np.zeros((2, 4, 4), dtype=np.float32))
    assert dm.artifact_available("bead_stack") is True


def test_artifact_disk_path_resolves_generated_filename(tmp_path):
    dm = DataManager()
    dm.set_output_dir(tmp_path)
    assert dm.artifact_disk_path("force_results") == tmp_path / "force_results.npy"
    assert dm.artifact_disk_path("bead_stack") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_data_manager_disk_truth.py -v`
Expected: FAIL — `artifact_disk_path` does not exist (AttributeError) and the disk-truth assertions fail because the current `artifact_available` is value-based.

- [ ] **Step 3: Write minimal implementation**

In `napariTFM/utilities/data_manager.py`, replace the existing `artifact_available` (lines 95-96) with:

```python
    def artifact_disk_path(self, key: str):
        """Expected on-disk path for a generated artifact, or None if N/A."""
        filename = self.GENERATED_FILENAMES.get(key)
        if filename is None or self._output_dir is None:
            return None
        return self._output_dir / filename

    def artifact_available(self, key: str) -> bool:
        if key in self.GENERATED_FILENAMES:
            path = self.artifact_disk_path(key)
            return path is not None and path.exists()
        return self.get_artifact(key).available
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_data_manager_disk_truth.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add napariTFM/utilities/data_manager.py tests/test_data_manager_disk_truth.py
git commit -m "Make generated-artifact availability derive from disk"
```

---

### Task 2: Status panel reads disk-aware availability

**Files:**
- Modify: `napariTFM/widgets/_stage_data_status.py` — add `_artifact_available`; `refresh()` uses it; thread `available` into `refresh_state`; `_info_text` distinguishes memory vs disk.
- Test: `tests/test_workflow_shell.py` (add one integration test)

Current `refresh()` (lines 167-190) and `refresh_state` (93-119) and `_info_text` (192-196) are the edit targets.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_workflow_shell.py` (uses the REAL DataManager, not the stub, so disk logic is exercised). Place it near `test_stage_data_status_refreshes_from_data_manager`:

```python
def test_stage_status_is_done_when_output_files_exist_on_disk(monkeypatch, app, tmp_path):
    from napariTFM.utilities.data_manager import DataManager

    monkeypatch.setattr(_widget, "DataManager", DataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)
    section = widget._stage_sections_by_key["preprocessing"]

    # Output files absent -> not done even though output_dir is set.
    widget.refresh_stage_statuses()
    assert section.status_indicator.toolTip() != "Preprocessing status: done"

    # Write the generated output files; no in-memory value is set.
    (tmp_path / "preprocessed_beads.tif").write_bytes(b"x")
    (tmp_path / "preprocessed_reference.tif").write_bytes(b"x")
    widget.refresh_stage_statuses()

    assert section.status_indicator.toolTip() == "Preprocessing status: done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_stage_status_is_done_when_output_files_exist_on_disk" -v`
Expected: FAIL — status stays `ready`/`not_started` because the panel still reads `state.value is not None`, and no in-memory value was set.

- [ ] **Step 3: Write the implementation**

In `napariTFM/widgets/_stage_data_status.py`, add this helper method to `StageDataStatusPanel` (next to `_artifact_value`):

```python
    def _artifact_available(self, artifact: DataArtifactSpec) -> bool:
        checker = getattr(self.data_manager, "artifact_available", None)
        if checker is not None:
            try:
                return bool(checker(artifact.key))
            except (KeyError, AttributeError):
                pass
        return self._artifact_value(artifact) is not None
```

Replace `refresh()` (currently lines 167-190) with:

```python
    def refresh(self) -> str:
        required_inputs_available = True
        output_available = False

        for artifact in self.artifacts:
            available = self._artifact_available(artifact)
            state = self._artifact_state(artifact)
            value = state.value if state is not None else self._artifact_value(artifact)
            if artifact.role == "input" and artifact.required and not available:
                required_inputs_available = False
            if artifact.role == "output" and available:
                output_available = True

            info_text = self._info_text(artifact, value, available)
            if state is not None:
                self.artifact_rows[artifact.key].refresh_state(
                    state, info_text=info_text, available=available
                )
            else:
                self.artifact_rows[artifact.key].refresh(available=available, info_text=info_text)

        if output_available:
            return "done"
        if required_inputs_available:
            return "ready"
        return "not_started"
```

Replace `_info_text` (currently lines 192-196) with:

```python
    def _info_text(self, artifact: DataArtifactSpec, value: Any, available: bool) -> str:
        if value is not None:
            return self._shape_text(value) or "Loaded"
        if available:
            return "Saved"
        return "Missing" if artifact.required else "Optional"
```

Update `_ArtifactRow.refresh_state` (currently line 93) signature and its first computed line. Change:

```python
    def refresh_state(self, state, info_text: str) -> None:
        available = getattr(state, "value", None) is not None
```

to:

```python
    def refresh_state(self, state, info_text: str, available: bool | None = None) -> None:
        if available is None:
            available = getattr(state, "value", None) is not None
```

(The `available is None` default preserves existing callers in `tests/test_artifact_row.py` that call `refresh_state(state, info_text=...)` without the new argument.)

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_workflow_shell.py::test_stage_status_is_done_when_output_files_exist_on_disk" -v`
Expected: PASS

- [ ] **Step 5: Run the existing status + row tests to confirm no regression**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_workflow_shell.py tests/test_artifact_row.py -v`
Expected: all PASS (the stub-driven `test_stage_data_status_refreshes_from_data_manager` still passes via the value-based fallback).

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_stage_data_status.py tests/test_workflow_shell.py
git commit -m "Drive stage data status from disk-aware artifact availability"
```

---

### Task 3: Row shows "Saved" for disk-only artifacts

**Files:**
- Test: `tests/test_artifact_row.py` (add one test)

This locks the display contract that an artifact present on disk but not loaded into memory shows the available glyph and a "Saved" caption, rather than appearing missing.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_artifact_row.py`:

```python
def test_row_shows_available_glyph_when_on_disk_without_value(app):
    from napariTFM.widgets._ui_style import STATUS_GLYPHS

    spec = DataArtifactSpec("foo", "Foo", "foo", "output")
    row = _ArtifactRow(spec)

    class _State:
        value = None
        error = ""
        path = None
        dirty = False

    row.refresh_state(_State(), info_text="Saved", available=True)

    assert row.glyph_label.text() == STATUS_GLYPHS["available"]
    assert "Saved" in row.info_label.text()
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_artifact_row.py::test_row_shows_available_glyph_when_on_disk_without_value" -v`
Expected: PASS if Task 2 is already implemented (this test characterizes Task 2's behavior). If you run it before Task 2's `refresh_state` change lands, it FAILs with a `TypeError` (unexpected `available` kwarg) — confirming the signature change is required.

- [ ] **Step 3: No new implementation**

Behavior is provided by Task 2. If the test fails, the Task 2 `refresh_state` signature change was not applied correctly — fix it there.

- [ ] **Step 4: Commit**

```bash
git add tests/test_artifact_row.py
git commit -m "Lock disk-only artifact row display contract"
```

---

### Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest`
Expected: all PASS (previously 138 + the new tests). Note: `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` may intermittently SIGSEGV — that is a known napari/pydantic env flake, not a regression. Re-run in isolation to confirm if it fires.

- [ ] **Step 2: Confirm clean diffs**

Run: `git diff -w --stat` and compare to `git diff --stat` (must match — no whitespace/CRLF churn). Confirm `grep -c $'\r$'` is 0 for each changed file.

---

## Self-Review

**Spec coverage:**
- Generated availability from disk → Task 1. ✔
- Stage status from disk → Task 2 (integration test with real DataManager + tmp files). ✔
- Inputs stay in-memory → Task 1 (`test_raw_input_available_follows_memory`). ✔
- Disk-only display ("Saved") → Tasks 2 + 3. ✔
- Skip `stale` → no task computes `stale`; `refresh()` still returns only `done`/`ready`/`not_started`. ✔
- Existing stub test stays green via fallback → Task 2 Step 5. ✔

**Placeholder scan:** none.

**Type consistency:** `artifact_disk_path` returns `Optional[Path]` (`Path | None`); `artifact_available` returns `bool`; `_artifact_available` returns `bool`; `refresh_state(..., available: bool | None = None)`. `refresh()` returns `str`. Consistent across tasks.

**Open judgment call (flag for reviewer/owner):** A freshly *computed* generated result is auto-saved to `output_dir` immediately (each stage widget calls `auto_save_artifact`/`auto_save_generated_artifacts` right after compute, gated by `ensure_output_dir_for_generated_artifacts`). So compute → file exists → `done`. If a future change lets a result exist in memory *without* being saved, it will read as not-`done` under this plan — which is the intended meaning of "file is ground truth."
