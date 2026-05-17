# Workflow Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tabbed workflow with stage rows, centralize parameter ownership, preserve batch execution, and remove the worst service dependency inversion.

**Architecture:** Keep existing analysis widgets and service orchestration intact while changing the top-level shell and parameter flow. Add focused unit tests for each seam before production edits.

**Tech Stack:** Python, qtpy/Qt widgets, napari plugin widgets, pytest, YAML config.

---

## File Map

- Modify `napariTFM/widgets/_widget.py`: replace `QTabWidget` with stage-row sections while preserving existing child widget attributes and signal wiring.
- Create `tests/test_workflow_shell.py`: shell tests for stage row helper and main widget structure where dependency import allows.
- Create `napariTFM/backend/parameter_validation.py`: neutral validation functions copied from service validators.
- Modify `napariTFM/utilities/parameter_manager.py`: remove service imports, add UI conversion helpers, fix stale category field names.
- Modify service files in `napariTFM/services/`: delegate `validate_parameters()` to backend validation functions.
- Create `tests/test_parameter_manager.py`: conversion/category/validation tests.
- Modify `napariTFM/widgets/batch_analysis_widget.py`: make batch config use `ParameterManager` parameter values and reduce duplicated sync hazards.
- Modify `napariTFM/backend/batch_analysis.py`: honor `auto_gcv` from config.
- Create `tests/test_batch_parameters.py`: batch config and `auto_gcv` tests.

## Task 1: Workflow Shell Stage Rows

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Create: `tests/test_workflow_shell.py`

- [ ] Write failing tests that assert a stage section toggles its content and that `napariTFMWidget` no longer contains a `QTabWidget`.
- [ ] Run: `PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen python -m pytest tests/test_workflow_shell.py -q`
- [ ] Implement a small `_StageSection` helper with a config tool button and content container.
- [ ] Replace the tab construction in `napariTFMWidget.__init__()` with five `_StageSection` instances.
- [ ] Keep preprocessing expanded by default and other rows collapsed.
- [ ] Run the focused workflow shell test again.

## Task 2: Parameter Validation And UI Conversions

**Files:**
- Create: `napariTFM/backend/parameter_validation.py`
- Modify: `napariTFM/utilities/parameter_manager.py`
- Modify: `napariTFM/services/preprocessing_service.py`
- Modify: `napariTFM/services/displacement_service.py`
- Modify: `napariTFM/services/fttc_service.py`
- Modify: `napariTFM/services/msm_service.py`
- Create: `tests/test_parameter_manager.py`

- [ ] Write failing tests for `ParameterManager.get_ui_parameter()` and `set_ui_parameter()` for `young_modulus`, `regularization`, and `gel_height`.
- [ ] Write a failing test that preprocessing category names include `min_intensity_percentile` and not stale `min_intensity`.
- [ ] Write a failing test that `ParameterManager.validate_all_parameters()` works without importing service modules.
- [ ] Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_parameter_manager.py -q`
- [ ] Add backend validation functions and delegate service validators to them.
- [ ] Update `ParameterManager` imports, validation calls, UI conversion helpers, and category mappings.
- [ ] Run the focused parameter tests again.

## Task 3: Batch Parameters Use Single Owner

**Files:**
- Modify: `napariTFM/widgets/batch_analysis_widget.py`
- Modify: `napariTFM/backend/batch_analysis.py`
- Create: `tests/test_batch_parameters.py`

- [ ] Write a failing test that `BatchAnalysisWidget._generate_config()` uses values from `ParameterManager.get_all_parameters()`.
- [ ] Write a failing test that loading/syncing combo values preserves `mesh_algorithm` case.
- [ ] Write a failing test that `BatchAnalysis._create_fttc_parameters()` preserves `auto_gcv=True`.
- [ ] Run: `PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen python -m pytest tests/test_batch_parameters.py -q`
- [ ] Update batch config generation to use `parameter_manager.get_all_parameters()` for the flat `parameters` block.
- [ ] Route batch UI sync through `ParameterManager.get_ui_parameter()` and `set_ui_parameter()` where controls remain.
- [ ] Fix combo lowercasing to apply only to parameters that intentionally store lowercase values.
- [ ] Honor `auto_gcv` in batch FTTC parameter creation.
- [ ] Run the focused batch tests again.

## Task 4: Integration Review And Verification

**Files:**
- Review all files changed by Tasks 1-3.

- [ ] Run: `git diff -- napariTFM docs tests`
- [ ] Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q`
- [ ] Run: `PYTHONDONTWRITEBYTECODE=1 python -c "from napariTFM.utilities.parameter_manager import ParameterManager; pm=ParameterManager(); print(pm.validate_all_parameters())"`
- [ ] Run: `PYTHONDONTWRITEBYTECODE=1 python -c "from napariTFM.backend.preprocessing import preprocess_stack; from napariTFM.backend.displacement_analysis import calculate_displacement_field; from napariTFM.backend.fttc import calculate_force_field; from napariTFM.backend.msm import calculate_stresses; print('backend orchestration imports ok')"`
- [ ] Run: `git status --short`
- [ ] Report exact verification results and any skipped dependency-heavy checks.
