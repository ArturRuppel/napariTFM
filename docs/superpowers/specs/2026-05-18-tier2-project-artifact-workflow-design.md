# Tier 2 Project Artifact Workflow — Design

## Goal

Make the Tier 1 workflow shell operational by adding project/output-directory
awareness, richer artifact-row state, per-artifact load/save/view actions, and
targeted workflow cleanup that removes duplicated visible controls only after
their replacement exists in the shell.

Tier 2 deliberately combines:

- **Project/artifact operations**: output directory, artifact state, save/load,
  view, dirty/error display.
- **Workflow cleanup**: remove or hide redundant stage-local input/action
  controls and slim batch around the shared parameter owner.

Napari manifest compatibility, full manual QA, sparse tracking, and numerical
backend changes are out of scope unless they block automated verification.

## Current Baseline

Tier 1 is committed through:

- `dc2e4e2` — CellFlow-style artifact rows and `PipelineDataWidget` deletion.
- `cd437eb` — preprocessing row assignment actions, fixed-width removal, and
  deprecated header alias cleanup.

Relevant existing primitives:

- `DataManager` already owns `ArtifactState`, `output_dir`,
  `GENERATED_FILENAMES`, `auto_save_artifact()`, `load_result_artifact()`,
  `dirty`, `path`, and `error`.
- `ProjectSection` contains general parameters plus save/load/reset/clear-data
  buttons.
- `StageDataStatusPanel` renders per-stage file rows from `DataArtifactSpec`.
- `StageSection` owns header params/run-cancel/preview controls and nested
  parameter sections.
- Existing stage widgets remain behavior owners for controller actions.

## Architecture

Tier 2 extends the existing `DataManager` artifact model instead of adding a
parallel project registry.

`DataManager` remains the source of truth for:

- artifact availability (`ArtifactState.value`)
- saved location (`ArtifactState.path`)
- dirty/unsaved state (`ArtifactState.dirty`)
- artifact errors (`ArtifactState.error`)
- output directory (`DataManager.output_dir`)

`ProjectSection` becomes the visible owner of the output directory selection.
It shows the current output directory and exposes a compact choose/change
control. It does not own file saving itself; it writes through
`DataManager.set_output_dir()`.

`StageDataStatusPanel` becomes a richer renderer of `ArtifactState`. It should
prefer `DataManager.get_artifact(spec.key)` when available and fall back to the
current attribute lookup for compatibility with tests or future synthetic
artifacts.

`DataArtifactSpec` remains a lightweight row contract but gains clearer action
metadata. The shell wires those actions to existing behavior:

- preprocessing input rows call `PreprocessingWidget.load_active_layer()`
- displacement input rows call `DisplacementController.load_active_layer()` or
  the existing widget/controller equivalent
- force/stress input rows call existing file/load handlers where available
- generated output rows call `DataManager.auto_save_artifact()`
- view rows call existing visualization methods where available

No new numerical pipeline APIs are introduced. New helper methods are allowed
only when they express existing UI behavior as a small callable for row wiring.

## Feature Behavior

### Project Output Directory

The Project section shows an output-directory row below the general parameter
controls.

The row contains:

- a read-only compact path label
- a choose/change tool button
- a clear button only if clearing is useful and testable without ambiguity

When no directory is set, the label reads `No output directory`. When a path is
set, the label shows the shortest useful display path while preserving the full
path in the tooltip.

Existing stage auto-save behavior may still prompt for a directory through
`ensure_output_dir_for_generated_artifacts()`. If the user chooses a path from
that prompt, the Project section must refresh via `DataManager` callbacks and
show the selected directory.

### Rich Artifact Rows

Artifact rows display:

- available/missing/optional/error glyph
- artifact label
- shape string or `Loaded`
- `Unsaved` when `ArtifactState.dirty` is true
- saved filename/path hint when `ArtifactState.path` exists
- error text when `ArtifactState.error` exists

Display priority:

1. Error text if `ArtifactState.error` is non-empty.
2. Available value with shape or `Loaded`.
3. Missing or Optional for unavailable artifacts.
4. Saved/dirty status is appended as a secondary hint, not used as the primary
   availability signal.

Examples:

- `512×512×20 · Unsaved`
- `Loaded · preprocessed_beads.tif`
- `Missing`
- `Auto-save failed: permission denied`

### Row Actions

Input rows:

- show assign/load (`↑`) when `on_action` exists
- hide view (`👁`) while missing
- show view (`👁`) when available and `on_view` exists

Output rows:

- hide save/view while missing
- show view when available and `on_view` exists
- show save when available and `on_action` exists
- disable save while unavailable

Save actions for generated outputs call `DataManager.auto_save_artifact()` and
pass calibration metadata where needed:

- preprocessing TIFF outputs use pixel size and frame interval
- result objects use the existing `.npy` serialization path

If saving raises, the row action records the error through
`DataManager.mark_artifact_error(key, message)` and leaves the row visible with
an error glyph/message.

### Stage Cleanup

Cleanup is allowed only after the row/header replacement for the same behavior
exists and has tests.

Target cleanup:

- preprocessing: old local data panel remains unmounted; no additional visible
  input rows return
- displacement: visible load-reference/load-beads controls are hidden once
  stage rows can assign preprocessed inputs
- force: visible load-displacement control is hidden once the row action can
  load or accept displacement input
- stress: visible load-force/load-mask controls are hidden once row actions can
  load force results and masks
- action panels: large run/preview/cancel buttons can be hidden from stage
  bodies where the `StageSection` header already proxies the same actions

Controller widgets and internal panels may remain alive for state, signals, and
backward-compatible behavior. The cleanup target is duplicated visible UI, not
behavior ownership.

### Batch Slimdown

Batch analysis remains a workflow stage.

Batch keeps visible controls for:

- input/output folders
- file naming fields
- analysis-step selection
- visualization/metric output choices
- run/cancel/status
- save/load batch config

Batch should not present duplicate analysis parameter editors for
preprocessing, displacement, force, or stress. Config generation continues to
read analysis parameters from `ParameterManager`, preserving backward-compatible
YAML shape where practical.

### Dirty And Stale Semantics

In Tier 2:

- `dirty=True` means generated/modified in memory but not saved to disk.
- `dirty=False` with a path means saved or loaded from disk.
- `error` means the last artifact operation failed.

Full dependency staleness from upstream data or parameter changes is deferred
unless it remains a small, isolated extension of the dirty/error display.

## Data Flow

1. A stage creates or loads an artifact through existing controller behavior.
2. Controller stores it in `DataManager` using existing setter methods.
3. `DataManager` updates `ArtifactState` and fires change callbacks.
4. `napariTFMWidget._on_pipeline_data_changed()` refreshes stage UI and status
   panels.
5. `StageDataStatusPanel.refresh()` reads artifact states and updates glyphs,
   info text, and action-button visibility.
6. Row action clicks call shell-wired callables, which delegate to existing
   controller, visualization, or `DataManager` methods.

## Error Handling

Artifact-row actions should catch expected file and serialization errors at the
shell/action boundary.

Expected behavior:

- show a `QMessageBox.warning()` for user-facing save/load failures when the
  action was user-triggered
- call `DataManager.mark_artifact_error(key, str(error))`
- leave existing artifact value untouched unless the load explicitly succeeds
- refresh the row so the error is visible

Unexpected exceptions may still propagate in tests if the behavior is not
user-triggered or not recoverable.

## Testing Strategy

Use TDD for each behavior slice.

Required test coverage:

- `ProjectSection` displays and updates output directory state through
  `DataManager`.
- `StageDataStatusPanel` renders shape, loaded fallback, missing, optional,
  dirty/unsaved, saved path, and error states.
- Output row save actions call `DataManager.auto_save_artifact()` with
  calibration values where required.
- Failed save actions mark artifact errors.
- Preprocessing and displacement input row actions call existing
  `load_active_layer()` paths.
- Force and stress input row actions call existing load handlers or focused
  helper methods over existing handlers.
- Stage cleanup tests assert redundant visible controls are hidden while header
  actions still proxy run/preview/cancel.
- Batch tests assert duplicate analysis parameter controls are absent and config
  generation still reads from `ParameterManager`.
- Full suite remains green with `pytest tests/ -v`.

Manual napari launch is desirable but non-blocking for this Tier 2 tranche
because the current environment has a separate napari/pydantic manifest import
issue. That compatibility issue belongs in a follow-up reliability spec unless
it blocks automated construction tests.

## Rollout Plan

Tier 2 should be implemented as independently green commits:

1. Project output directory UI.
2. Rich artifact-state row rendering.
3. Generated output save actions.
4. Input assign/load actions beyond preprocessing.
5. Stage-local duplicate UI cleanup.
6. Batch widget slimdown.
7. Final regression and documentation note if user-facing behavior changed.

Each commit should include focused tests and avoid staging unrelated local
changes. The current working tree may include unrelated `_dev/`, `.gitignore`,
or `pyproject.toml` changes; Tier 2 work must not revert or include them unless
explicitly requested.

## Non-Goals

- New numerical algorithms.
- Sparse bead tracking.
- Forward-compatible project/session file format.
- Full dependency graph staleness.
- Napari manifest/pydantic compatibility repair.
- Large visual redesign beyond removing duplicated controls.
- Rewriting stage controllers.
