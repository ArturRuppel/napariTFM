# Preprocessing UI Redesign Design

## Goal

Make the preprocessing stage compact and usable by removing obsolete large action buttons, simplifying input loading, and replacing target-specific preview controls with one composed live preview of all loaded preprocessing inputs.

## Scope

This design covers the preprocessing stage UI and its preview/post-run visualization behavior. It does not change preprocessing algorithms, parameter semantics, displacement analysis, force analysis, stress analysis, batch execution, or file formats.

The existing `PreprocessingWidget` and `PreprocessingController` remain the behavior owners for loading, previewing, running, cancelling, and saving. The workflow shell and `StageSection` expose these behaviors through compact icon actions.

## Current Problem

The preprocessing widget currently duplicates workflow controls in the stage body:

- three large load buttons for bead stack, reference image, and cell stack;
- a separate preview data-type radio group;
- a preview checkbox;
- large run, save, and cancel buttons;
- a status/progress block below all of that.

The shell already exposes run, preview, cancel, and config icons in the stage header, so the body-level run/cancel/save controls are redundant. Preview also asks the user to choose one input at a time, even though the useful inspection view is the combined preprocessing scene.

## Input Loading Design

The preprocessing body becomes a compact input assignment panel with three fixed rows:

```text
Reference   <source/status>   assign-active-layer
Beads       <source/status>   assign-active-layer
Cells       <source/status>   assign-active-layer
```

Reference and beads are required. Cells are optional. Each row has a small icon-only action that assigns the currently selected napari image layer to that role. Disabled state follows the existing rule: assignment requires an active image layer.

Each status cell shows the current state in a compact form:

- `Missing` for required inputs that are absent;
- `Optional` for absent cells;
- shape information when data is loaded;
- source layer name plus shape if source-layer tracking is available during implementation.

The row actions call the same controller loading paths that currently back `load_active_layer('reference')`, `load_active_layer('beads')`, and `load_active_layer('cells')`.

## Stage Header Actions

The preprocessing stage header owns the visible stage actions:

```text
status dot | Preprocessing | run | preview | cancel | save | config
```

All actions are icon-only with tooltips. Run, preview, and cancel continue to proxy existing controller behavior. Save is added as a header action and is enabled only when at least one preprocessed output exists.

The large body buttons are removed from the visible layout:

- `Run Preprocessing`
- `Cancel Operation`
- `Save Result Images`

Compatibility attributes may remain on `PreprocessingWidget` while migration is in progress, but they should not be visible in the compact workflow body.

## Preview Design

Preview is one stage-level action. There is no preview data-type selector.

When preview is enabled or refreshed, the controller processes the current frame from every loaded raw input using the current preprocessing parameters:

- bead stack: current frame, processed with bead/reference parameters;
- reference image: processed as a 2D reference;
- cell stack: current frame, processed with cell parameters when available.

The viewer renders separate preview layers:

- `Preview Cells`, gray;
- `Preview Reference`, magenta;
- `Preview Beads`, green.

Layers use additive blending so the preview reads as one composed scene. Missing inputs are skipped. Preview can show partial data, but the run action remains disabled until both required inputs, reference and beads, are loaded.

While preview is active, it refreshes when preprocessing parameters change and when the napari frame changes. Turning preview off removes or hides the preview layers.

## Post-Run Visualization

Post-run preprocessing visualization uses the same layer language as preview:

- `Preprocessed Cells`, gray;
- `Preprocessed Reference`, magenta;
- `Preprocessed Beads`, green.

The dedicated RGB `Bead Overlay` layer is removed. The separate napari image layers, with additive blending, become the only overlay mechanism.

## Status And Progress

The compact input rows report loaded/missing state for raw inputs. The existing stage data-status panel continues to summarize preprocessing artifacts and stage readiness.

The progress bar and status label may remain in the stage body if they are compact and only visible when useful. They should not compete with input assignment as the main body content.

## Error Handling

Existing validation behavior remains:

- reference must be 2D;
- bead and cell inputs may be 2D or 3D, with 2D promoted to a single-frame stack;
- assignment requires an active image layer;
- run requires both required inputs.

Preview failures should clear or disable preview layers and report a compact status message without changing loaded input state.

## Testing

Add or update focused Qt tests for:

- preprocessing body exposes three fixed input rows with stable object names;
- large run/cancel/save body buttons are not visible in the workflow layout;
- preprocessing stage header exposes a save icon and proxies the existing save behavior;
- preview no longer exposes bead/reference/cell radio buttons;
- preview action renders all available loaded inputs into separate named layers;
- post-run visualization no longer creates `Bead Overlay`;
- save icon enablement follows preprocessed output availability.

Backend preprocessing tests should remain unchanged unless implementation touches validation or frame processing behavior.

## Migration Constraints

- Preserve existing public attributes used by the workflow shell until tests are migrated.
- Preserve `PreprocessingController` worker lifecycle and cancel behavior.
- Preserve calibrated TIFF saving behavior and output filenames.
- Keep edits scoped to preprocessing UI, workflow stage action support, visualization manager preprocessing layers, and focused tests.
- Do not touch unrelated `_dev/` files or existing dirty worktree changes.
