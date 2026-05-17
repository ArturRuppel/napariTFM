# Remaining Refactor Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for independent implementation slices, or superpowers:executing-plans for tightly coupled UI/backend changes. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the napariTFM workflow redesign after the committed stage-shell, parameter-manager, and DIS optical-flow groundwork.

**Architecture:** Continue in small, verified slices. Keep the existing numerical pipeline stable while moving the UI toward a stage-row workflow, centralizing parameter ownership, and simplifying the backend/frontend boundary only after tests cover current behavior.

**Tech Stack:** Python, qtpy/Qt widgets, napari plugin widgets, OpenCV DIS optical flow, pytest, YAML batch config.

---

## Current Baseline

- Commit `7c6890b`: workflow shell refactor, backend validation extraction, batch parameter centralization, DIS backend replacement.
- Commit `41fc7e4`: active displacement parameter surface aligned with OpenCV DIS.
- Commits `7571616`, `d2f7989`, `e5cdaf3`: displacement, FTTC, and MSM service layers removed.
- Current slice: preprocessing service layer removed; preprocessing orchestration now lives in `napariTFM.backend.preprocessing`.
- Verification baseline: `conda run -n napariTFMv2 python -m pytest tests -q` reports `44 passed`.
- Worktree expectation before the next implementation slice: unrelated algorithm experiment files under `_dev/` and `_validation/` may remain dirty.

## Roadmap

### Phase 1: Stage Header Actions

**Goal:** Make the stage rows useful, not just collapsible containers.

**Files likely touched:**
- `napariTFM/widgets/_widget.py`
- `tests/test_workflow_shell.py`

- [ ] Add tests for stable stage header action buttons: run, preview, cancel, config.
- [ ] Add icon-style header controls with tooltips and stable object names.
- [ ] Wire config to the existing expand/collapse behavior.
- [ ] Wire run/preview/cancel only where an existing child widget exposes an unambiguous button or controller method.
- [ ] Leave actions disabled or hidden when a stage does not support that action yet.
- [ ] Verify offscreen Qt construction and workflow shell tests.

**Checkpoint:** Commit `Add stage header workflow actions`.

### Phase 2: Main Parameter Surface

**Goal:** Move toward one visible parameter owner in the main widget instead of duplicated parameter panels per stage and batch.

**Files likely touched:**
- `napariTFM/widgets/_widget.py`
- `napariTFM/widgets/displacement_analysis_widget.py`
- `napariTFM/widgets/fttc_widget.py`
- `napariTFM/widgets/msm_widget.py`
- `napariTFM/widgets/preprocessing_widget.py`
- `napariTFM/utilities/parameter_manager.py`
- New or expanded tests under `tests/`

- [ ] Inventory current parameter panels and categorize each parameter as preprocessing, displacement, force, stress, or visualization.
- [ ] Add tests proving each parameter is instantiated once in the new shared surface.
- [ ] Create a shared parameter widget backed directly by `ParameterManager`.
- [ ] Keep existing stage parameter panels available internally until workflows are proven, but avoid duplicated visible controls.
- [ ] Route all parameter writes through `set_ui_parameter()` or `set_parameter()` consistently.
- [ ] Verify reset behavior per category and full reset behavior.

**Checkpoint:** Commit `Centralize workflow parameter controls`.

### Phase 3: Batch Widget Slimdown

**Goal:** Make batch analysis an execution mode that consumes the main parameters instead of presenting another parameter editor.

**Files likely touched:**
- `napariTFM/widgets/batch_analysis_widget.py`
- `napariTFM/backend/batch_analysis.py`
- `tests/test_batch_parameters.py`

- [ ] Add tests that batch UI no longer creates analysis parameter spinboxes.
- [ ] Keep batch-specific inputs: folders, file names, selected analysis steps, visualization outputs, metrics options, run/cancel/status.
- [ ] Keep YAML config shape backward-compatible where practical.
- [ ] Ensure `_generate_config()` still writes a complete `parameters` block from `ParameterManager.get_all_parameters()`.
- [ ] Verify batch parameter sync no longer mutates hidden duplicate controls.

**Checkpoint:** Commit `Use main parameters for batch execution`.

### Phase 4: Backend/Frontend Boundary Cleanup

**Goal:** Reduce service-layer complexity without breaking progress reporting or result DTOs.

**Files likely touched:**
- `napariTFM/services/*.py`
- `napariTFM/backend/*.py`
- `napariTFM/widgets/*_widget.py`
- Focused tests for each analysis stage.

- [x] Classify each service method as orchestration, validation, DTO packaging, or pure computation.
- [x] Move pure computation and validation into backend modules.
- [x] Remove pass-through service code one stage at a time for displacement, FTTC, MSM, and preprocessing.
- [x] After each stage, verify focused backend tests and the full test suite.
- [ ] Preserve or document compatibility for old pickled result objects whose dataclass module paths changed.
- [ ] Do a final scan for stale `services.*_service` references outside production tests and roadmap notes.

**Checkpoint:** One commit per stage, for example `Simplify displacement orchestration boundary`.

### Phase 5: Optical Flow Validation

**Goal:** Confirm DIS is a conservative operational replacement before considering sparse bead tracking.

**Files likely touched:**
- `napariTFM/backend/displacement_analysis.py`
- `tests/test_displacement_analysis.py`
- Optional sample-data scripts under a non-production path.

- [ ] Add synthetic bead-like image tests with known translation.
- [ ] Check displacement sign, units, shape, finite values, and downscale behavior.
- [ ] Add one regression test for constant or near-constant images.
- [ ] Compare DIS presets only if test data shows a real accuracy/performance tradeoff.
- [ ] Defer sparse tracking until dense DIS behavior is characterized.

**Checkpoint:** Commit `Validate DIS displacement behavior`.

### Phase 6: Latest Napari Compatibility

**Goal:** Keep the plugin working in `napariTFMv2` with current napari behavior.

**Files likely touched:**
- `pyproject.toml`
- plugin metadata files if present
- tests requiring Qt/napari smoke construction

- [ ] Confirm installed napari version in `napariTFMv2`.
- [ ] Check plugin import and widget construction with `QT_QPA_PLATFORM=offscreen`.
- [ ] Tighten dependency ranges only where failures prove they are needed.
- [ ] Add a lightweight smoke test for plugin widget construction if it can run reliably in CI/local pytest.

**Checkpoint:** Commit `Verify napari compatibility`.

### Phase 7: End-to-End Manual QA

**Goal:** Prove the redesigned workflow works with real data, not only unit tests.

**Manual checklist:**
- [ ] Launch napari from `napariTFMv2`.
- [ ] Load reference, bead stack, and optional cell image.
- [ ] Run preprocessing.
- [ ] Run displacement with DIS.
- [ ] Run FTTC.
- [ ] Run MSM where input data permits.
- [ ] Run batch on a small folder set.
- [ ] Save and reload parameters or config.
- [ ] Record any UI overlap, disabled-action, progress, cancellation, or layer-ordering issues.

**Checkpoint:** Commit fixes from manual QA in small groups, not as one mixed cleanup.

### Phase 8: Documentation And Release Cleanup

**Goal:** Make the new workflow understandable and remove stale TV-L1/batch-parameter language.

**Files likely touched:**
- `README.md`
- docs or plugin help text if present
- docstrings in `napariTFM/`

- [ ] Search docs and docstrings for stale tab, TV-L1, duplicate batch parameter, and old dependency language.
- [ ] Document the DIS optical-flow choice and note that sparse bead tracking is deferred.
- [ ] Document `napariTFMv2` setup commands if this environment is intended to be reproducible.
- [ ] Run full tests and final widget smoke check.

**Checkpoint:** Commit `Update workflow documentation`.

## Recommended Execution Order

1. Stage header actions.
2. Main parameter surface.
3. Batch widget slimdown.
4. DIS validation tests.
5. Backend/frontend boundary cleanup.
6. Latest napari compatibility sweep.
7. Manual QA.
8. Documentation cleanup.

The order keeps user-visible workflow improvements moving while preserving numerical behavior. The service-layer reduction should come after the UI and parameter ownership are covered by tests, because otherwise it will be hard to distinguish real backend simplification from accidental workflow breakage.
