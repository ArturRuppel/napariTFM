# Workflow Redesign Design

## Goal

Replace the top-level tabbed workflow with a CellFlow-style stage-row workflow, make `ParameterManager`/`UnifiedParameters` the single parameter owner, preserve batch execution without duplicated parameter state, and start untangling service dependencies without changing numerical algorithms.

## Scope

This implementation is intentionally split into safe slices. The first UI pass changes the shell, not each stage's internal controller or algorithm code. Existing stage widgets remain the owners of their data panels, parameter panels, action buttons, progress bars, and cancel behavior until the shell is proven stable.

## UI Design

The top-level `QTabWidget` is replaced by a vertical workflow stack:

```text
Preprocess              config
Displacement            config
Traction / FTTC         config
Stress / MSM            config
Batch folders           config
```

Each row has a bold stage label and compact icon-style controls. In the first pass, the config control expands or collapses the existing full stage widget. Run, preview, cancel, and save/load controls remain inside the existing stage widgets to avoid rewiring worker lifecycles prematurely. After this pass, the stage rows provide the visual structure needed to migrate those actions into row headers in a later, smaller change.

Only the preprocessing row opens by default. Other rows are collapsed so the nested scroll areas and current fixed-width colorbar layouts do not overwhelm the widget.

## Parameter Ownership

`UnifiedParameters` remains the canonical parameter model, owned by `ParameterManager`. Stage widgets and batch configuration must read and write through `ParameterManager`, not keep independent batch copies.

Batch execution stays supported, but the batch widget stops manually rebuilding analysis parameters from duplicated controls. It uses `ParameterManager.get_all_parameters()` for the flat `parameters` config block and keeps only batch-specific UI: folders, input filenames, analysis-step toggles, visualization toggles, metrics options, run mode, status, and progress.

UI-only conversions for `young_modulus`, `regularization`, and `gel_height` are centralized in `ParameterManager` helper methods so internal values remain in backend units.

## Service Boundary Design

Service classes are not deleted in this pass. They still provide meaningful orchestration for stack iteration, progress reporting, result DTOs, and workflow-specific validation. The first architecture cleanup only fixes dependency direction:

- Move parameter validation into `napariTFM/backend/parameter_validation.py`.
- Keep existing `Service.validate_parameters()` APIs as delegating compatibility shims.
- Make `ParameterManager.validate_all_parameters()` call backend validation functions instead of importing service classes.

Moving result DTOs and batch orchestration out of service/backend mixed locations is explicitly deferred.

## Testing

Add a focused `tests/` tree because this repo currently has validation scripts but no unit tests. Tests cover:

- workflow shell construction and collapse behavior without `QTabWidget`;
- `ParameterManager` UI/internal conversion helpers and category mappings;
- batch config generation using `ParameterManager` values;
- backend validation import direction and service compatibility shims;
- batch `auto_gcv` propagation.

Qt tests should run offscreen and avoid real napari computation. When full widget construction is too dependency-heavy, tests should isolate the new shell helper classes and parameter/config methods.

## Migration Constraints

- Preserve existing public widget attributes such as `preprocessing_widget`, `displacement_widget`, `force_widget`, `msm_widget`, and `batch_widget`.
- Preserve existing YAML batch config shape.
- Preserve service public APIs and all numerical algorithm behavior.
- Do not touch unrelated `_dev/` files.
