# On-Demand Stage Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep stage Preview and Run actions available and resolve missing or stale upstream results on demand with current UI parameters.

**Architecture:** Add a focused `InteractiveStageCoordinator` between stage header actions and the existing controllers. Controllers gain completion callbacks and optional transient preview inputs, while a separate freshness helper compares only computation-affecting parameters. The coordinator owns dependency order, stale-result prompts, asynchronous continuation, cancellation, and failure termination.

**Tech Stack:** Python, Qt/qtpy signals and dialogs, napari thread workers, dataclasses, NumPy, pytest/pytest-qt.

---

## File Structure

- Create `napariTFM/widgets/_stage_dependencies.py`: freshness normalization, dependency metadata, prompt decision enum, and `InteractiveStageCoordinator`.
- Modify `napariTFM/widgets/_base_widget.py`: expose one-shot completion hooks without changing the sealed run/cancel lifecycle.
- Modify `napariTFM/widgets/displacement_analysis_widget.py`: coordinator-friendly preview entry point and source-input action enablement.
- Modify `napariTFM/widgets/fttc_widget.py`: accept a transient displacement result for preview and report preview completion.
- Modify `napariTFM/widgets/stress_widget.py`: accept a transient force result for preview and report preview completion.
- Modify `napariTFM/widgets/_widget.py`: construct the coordinator, route stage actions through it, and expose experiment-selected availability.
- Create `tests/test_stage_dependencies.py`: deterministic unit tests for freshness and dependency-chain behavior.
- Modify `tests/test_preview_offthread.py`: completion-hook and transient-input controller tests.
- Modify `tests/test_workflow_shell.py`: button enablement, routing, prompt, and source-input integration tests.

### Task 1: Parameter freshness rules

**Files:**
- Create: `napariTFM/widgets/_stage_dependencies.py`
- Create: `tests/test_stage_dependencies.py`

- [ ] **Step 1: Write failing freshness tests**

Add tests using small dataclasses that assert `parameters_match(stage, stored, current)` returns true for identical computational values and for visualization-only changes, false for solver changes, and false when stored metadata is absent. Cover NumPy scalar normalization.

```python
def test_displacement_display_parameters_do_not_make_result_stale():
    stored = _Disp(window=32, d_max=5.0, disp_vector_stride=4, disp_arrow_scale=1.0)
    current = replace(stored, d_max=10.0, disp_vector_stride=8)
    assert parameters_match("displacement", stored, current)

def test_solver_parameter_change_makes_result_stale():
    stored = _Force(young_modulus=10.0, f_max=100.0)
    assert not parameters_match("force", stored, replace(stored, young_modulus=20.0))

def test_missing_parameter_metadata_is_stale():
    assert not parameters_match("force", None, _Force(10.0, 100.0))
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_stage_dependencies.py -q`

Expected: collection fails because `_stage_dependencies` does not exist.

- [ ] **Step 3: Implement normalization and comparison**

Create constants for excluded fields and recursive normalization for dataclasses, mappings, sequences, enums, and NumPy scalars. Implement:

```python
DISPLAY_ONLY_FIELDS = {
    "displacement": {"d_max", "disp_vector_stride", "disp_arrow_scale"},
    "force": {"f_max", "force_vector_stride", "force_arrow_scale"},
    "stress": {"max_stress"},
}

def computational_parameters(stage: str, params: object) -> object:
    """Return a normalized, display-independent parameter value."""

def parameters_match(stage: str, stored: object, current: object) -> bool:
    if stored is None or current is None:
        return False
    return computational_parameters(stage, stored) == computational_parameters(stage, current)
```

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_stage_dependencies.py -q`

Expected: all freshness tests pass.

- [ ] **Step 5: Commit**

Stage only the two task files and commit with `feat: compare stage computation parameters`.

### Task 2: Controller completion and transient preview inputs

**Files:**
- Modify: `napariTFM/widgets/_base_widget.py`
- Modify: `napariTFM/widgets/displacement_analysis_widget.py`
- Modify: `napariTFM/widgets/fttc_widget.py`
- Modify: `napariTFM/widgets/stress_widget.py`
- Modify: `tests/test_preview_offthread.py`

- [ ] **Step 1: Write failing controller tests**

Add tests proving `_start_preview_worker(..., completion=callback)` invokes completion only after the paint callback succeeds, and that Force and Stress preview entry points use explicitly supplied upstream results without reading or replacing the data manager's full-stack artifact.

```python
def test_preview_completion_follows_paint(app):
    events = []
    ctrl._start_preview_worker(worker, lambda result: events.append(("paint", result)),
                               completion=lambda result: events.append(("done", result)))
    worker.returned.emit("R")
    assert events == [("paint", "R"), ("done", "R")]
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_preview_offthread.py -q`

Expected: failure because `_start_preview_worker` has no `completion` argument and downstream previews have no transient-input argument.

- [ ] **Step 3: Add completion plumbing**

Extend `_start_preview_worker` with `completion=None`. In its returned slot, call the paint callback first and the completion callback second. If painting raises, report the error and do not continue the chain. Do not change worker registration, cancellation, or unfreeze behavior.

- [ ] **Step 4: Add coordinator-facing preview signatures**

Use these public signatures:

```python
DisplacementController.preview_displacement(*, completion=None)
FTTCController.preview_force(*, displacement_result=None, completion=None)
StressController.preview_current_frame(*, force_result=None, completion=None)
```

When an override is supplied, derive the current-frame field and related scale/parameters from it. Otherwise retain existing validation and data-manager behavior. Pass the computed result to `completion` through `_start_preview_worker`. Do not store preview-only prerequisite results in `DataManager`.

- [ ] **Step 5: Verify GREEN and regressions**

Run: `pytest tests/test_preview_offthread.py tests/test_stage_lifecycle.py tests/test_force_ownership.py tests/test_stress_ownership.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

Stage only the controller files and focused tests, then commit with `feat: expose stage preview completion hooks`.

### Task 3: Interactive dependency coordinator

**Files:**
- Modify: `napariTFM/widgets/_stage_dependencies.py`
- Modify: `tests/test_stage_dependencies.py`

- [ ] **Step 1: Write failing chain tests with fake stages**

Test these exact cases: missing Force Preview invokes displacement preview then force preview with the returned transient result; missing Stress Run invokes displacement run, force run, stress run; matching artifacts skip prompts; stale artifacts handle recalculate, reuse, and cancel; cancel and failure clear pending continuations; source validation failure starts nothing.

Use synchronous fake stages exposing `preview(completion=..., **inputs)`, `run()`, `cancel()`, `completed`, and `failed`. Record ordered calls and make emitted results explicit.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_stage_dependencies.py -q`

Expected: failures because `InteractiveStageCoordinator` is absent.

- [ ] **Step 3: Implement the coordinator**

Define:

```python
class StaleChoice(Enum):
    RECALCULATE = "recalculate"
    REUSE = "reuse"
    CANCEL = "cancel"

class InteractiveStageCoordinator(QObject):
    def request(self, stage: str, mode: str) -> None: ...
    def cancel(self, stage: str) -> None: ...
```

Inject stage adapters, artifact getters, current-parameter getters, a `prompt(stage) -> StaleChoice`, a source validator, and a progress callback. Build only the required prefix of `("displacement", "force", "stress")`. Preview continuations pass transient result objects forward; run continuations wait for controller completion and then read the stored artifact. Guard every continuation with a monotonically increasing request token so cancelled or superseded requests cannot continue.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_stage_dependencies.py -q`

Expected: all coordinator and freshness tests pass.

- [ ] **Step 5: Commit**

Stage the coordinator and its test file, then commit with `feat: coordinate on-demand stage prerequisites`.

### Task 4: Workflow shell and always-available actions

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Modify: `napariTFM/widgets/displacement_analysis_widget.py`
- Modify: `napariTFM/widgets/fttc_widget.py`
- Modify: `napariTFM/widgets/stress_widget.py`
- Modify: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write failing shell tests**

Add tests asserting that, with an active experiment, all enabled stages report Preview and Run true even when derived results are absent; no active experiment reports them false; disabled Stress remains disabled through `StageSection`; stage header clicks call coordinator `request(stage, mode)`; cancel calls coordinator `cancel(stage)`; and the stale dialog maps its three buttons to `StaleChoice` values.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_workflow_shell.py -q`

Expected: new tests fail because actions still depend on resident artifacts and headers call controllers directly.

- [ ] **Step 3: Wire the coordinator**

Construct `InteractiveStageCoordinator` after the three stage widgets. Provide adapters around their controllers, parameter getters from `ParameterManager`, artifact getters/loaders from `DataManager`, and a source validator for raw images plus the Stress mask. Route each `StageSection` action through `_request_stage_action(stage, mode)` and cancellation through `_cancel_stage_action(stage)`.

Implement `_prompt_stale_upstream(stage)` with `QMessageBox` buttons labeled exactly **Recalculate with current parameters**, **Use existing data**, and **Cancel**, defaulting to recalculation.

- [ ] **Step 4: Make enablement experiment-based**

Give stage widgets an injected `action_context_available` predicate owned by the shell. `_update_ui_state` sets Preview and Run from that predicate (or loaded source inputs for standalone widget use), not from derived upstream artifacts. Frozen state still leaves only Cancel enabled, and `StageSection.set_enabled(False)` still disables every action.

- [ ] **Step 5: Protect unrelated `_widget.py` edits**

Before staging, inspect `git diff -- napariTFM/widgets/_widget.py` and distinguish pre-existing hunks from feature hunks. Stage only feature hunks with an explicit patch or path-safe index operation; never commit the user's unrelated widget changes.

- [ ] **Step 6: Verify GREEN and integration regressions**

Run: `pytest tests/test_workflow_shell.py tests/test_reload_on_selection.py tests/test_stage_section_action_sync.py tests/test_stage_section_header.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

Stage only feature hunks and related tests, then commit with `feat: resolve stage inputs on demand`.

### Task 5: Full verification and documentation alignment

**Files:**
- Modify if needed: `README.md`
- Modify if needed: focused tests from Tasks 1 through 4

- [ ] **Step 1: Run focused feature verification**

Run: `pytest tests/test_stage_dependencies.py tests/test_preview_offthread.py tests/test_stage_lifecycle.py tests/test_workflow_shell.py tests/test_reload_on_selection.py tests/test_force_ownership.py tests/test_stress_ownership.py -q`

Expected: all collected tests pass with no failures.

- [ ] **Step 2: Run the full test suite**

Run: `pytest -q`

Expected: all tests pass. If optional dependencies skip tests, report the exact skipped count and rerun relevant skipped feature tests in the project environment when one is available.

- [ ] **Step 3: Check syntax and diff hygiene**

Run: `python -m py_compile napariTFM/widgets/_stage_dependencies.py napariTFM/widgets/_base_widget.py napariTFM/widgets/displacement_analysis_widget.py napariTFM/widgets/fttc_widget.py napariTFM/widgets/stress_widget.py napariTFM/widgets/_widget.py`

Run: `git diff --check && git status --short`

Expected: compilation succeeds, diff check is clean, and unrelated pre-existing modifications remain unstaged.

- [ ] **Step 4: Update user documentation only if existing text contradicts behavior**

If README instructions say downstream buttons require manual upstream calculation, replace only those sentences with a concise description of automatic prerequisite computation and the stale-parameter choice. Otherwise make no documentation change.

- [ ] **Step 5: Commit verification-only changes if any**

Stage explicit files only and commit with `docs: explain on-demand stage calculations`. Skip this commit when no files changed.
