# CellFlow UI Concept for napariTFM

## Purpose

CellFlow is useful as a UI reference because it treats a napari plugin as a
workflow cockpit rather than a collection of independent panels. The important
idea is not the exact biology pipeline, but the way the UI compresses a long,
stateful analysis into a vertical set of stages with consistent controls,
visible artifacts, and delegated stage behavior.

This note distills the CellFlow pattern and maps it onto napariTFM.

## What CellFlow Is Doing

### 1. One scrollable workflow shell

`src/cellflow/napari/main_widget.py` builds a single top-level
`CellFlowMainWidget` with project metadata at the top and a `QScrollArea`
containing workflow sections:

- project status
- Cellpose
- nucleus segmentation and tracking
- cell segmentation
- contact analysis

Each top-level section is a `CollapsibleSection` with a stage accent and status
dot. The shell owns the project directory and refresh cycle, then delegates
stage-specific behavior to child widgets.

The transferable concept for napariTFM: the top-level widget should be the place
where users understand the whole analysis state. It should not force users to
discover the workflow by opening separate tabs or duplicated parameter panels.

### 2. A shared section primitive

`src/cellflow/napari/widgets.py` defines `CollapsibleSection`, which centralizes:

- expand/collapse behavior
- header styling
- optional stage status dot
- accent inheritance for nested sections
- layout refresh after nested visibility changes
- hiding the built-in header when a compact external header drives the section

This matters because every stage behaves the same way. The workflow reads as one
tool, not as many unrelated Qt forms.

The transferable concept for napariTFM: `_StageSection` should become a real
shared primitive, not just a local shell helper. It should carry the stable
header behavior, stage actions, status display, and styling rules used across
all analysis stages.

### 3. Compact stage rows with delegated behavior

CellFlow stage widgets often keep controller widgets alive but reparent or expose
their visible controls in a flatter workflow layout.

Examples:

- `nucleus_workflow_widget.py` creates segmentation, tracking, database browser,
  and correction controller widgets, then aliases their controls onto the
  workflow widget for compatibility.
- `cell_workflow_widget.py` hides `CellParamsWidget` and
  `CellCorrectionWidget` as owners, then places their sections and actions into
  stage rows like flow filtering, foreground masks, contours, and segmentation.
- The row buttons are compact tool buttons: parameter toggle, preview when
  relevant, and run/cancel behavior.

The key pattern is incremental migration. CellFlow does not require every child
widget to be rewritten before the shell improves. It keeps old behavior alive,
then exposes it through a better workflow surface.

The transferable concept for napariTFM: keep `PreprocessingWidget`,
`DisplacementAnalysisWidget`, `FTTCWidget`, `MSMWidget`, and
`BatchAnalysisWidget` as behavior owners while progressively moving their
controls into consistent stage headers and inline parameter sections.

### 4. Artifact and status visibility

CellFlow makes intermediate files visible through `PipelineFilesWidget` and
`make_pipeline_files_header`. Users can see which inputs, intermediates, and
outputs exist for each stage, and can load available artifacts into napari.

The transferable concept for napariTFM: TFM has the same need, but the artifacts
are analysis products rather than CellFlow pipeline files:

- reference and deformed images
- preprocessed images
- displacement fields
- traction maps
- stress maps
- masks, meshes, and metrics
- batch output folders and summary tables

napariTFM should expose these as stage-local data status panels backed by
`DataManager` and known output paths. Users should be able to answer "what do I
already have?" without running or expanding the whole stage.

### 5. Centralized style and widget factories

CellFlow keeps small UI decisions in `ui_style.py` and `_widget_helpers.py`:

- compact margins and spacing
- fixed icon button sizing
- stage accent colors
- consistent labels and status labels
- compact spin boxes, sliders, section grids, and button grids

This avoids scattered one-off Qt styling and makes dense scientific controls
feel intentional.

The transferable concept for napariTFM: create a small local style/helper module
for workflow UI primitives. The goal is not decorative styling; it is stable,
compact, readable controls that do not resize or drift as stages change.

### 6. Compatibility-first testing

CellFlow tests assert UI contracts directly: section visibility, stable button
attributes, hidden controller widgets not intercepting clicks, compact style
helpers, and pipeline status behavior.

napariTFM already started this pattern in `tests/test_workflow_shell.py`:

- no top-level `QTabWidget`
- stable stage action button names
- header actions proxy existing child buttons
- only one visible workflow parameter panel
- embedded stage parameter panels are hidden

The transferable concept for napariTFM: continue treating the shell as a tested
contract. Tests should protect workflow behavior while internals are migrated in
small slices.

## Current napariTFM Fit

napariTFM already has the first layer of the CellFlow idea:

- `napariTFM/widgets/_widget.py` defines `napariTFMWidget` as the shell.
- `_StageSection` wraps stage widgets and exposes run, preview, cancel, and
  config header actions.
- `WorkflowParameterPanel` centralizes common parameters through
  `ParameterManager`.
- `_hide_embedded_parameter_panels()` removes duplicate visible parameter
  editors while preserving child widgets.
- Tests in `tests/test_workflow_shell.py` lock down this shell behavior.

The remaining gap is that napariTFM currently has a shell, but not yet a full
workflow language. CellFlow's stronger pieces are:

- visible per-stage data/artifact status
- shared style primitives
- explicit stage accent/status model
- compact inline parameter sections
- controller widgets treated as behavior owners rather than layout owners
- more deliberate prevention of hidden widgets intercepting clicks

## Recommended napariTFM Direction

### Shell owns workflow topology

Keep the top-level order:

1. Inputs and calibration
2. Preprocessing
3. Displacement
4. Traction / FTTC
5. Stress / MSM
6. Batch analysis

The shell should own shared managers, global parameter state, stage ordering,
stage visibility, and cross-stage status refresh. It should delegate numerical
work and napari layer manipulation to the existing stage widgets and services.

### Stage sections expose the same surface

Each stage header should eventually offer the same predictable controls:

- run
- preview, when meaningful
- cancel, when a worker can be running
- config expand/collapse
- status indicator
- optional artifact/data-status toggle

The header buttons should proxy existing child controls until the stage widget
internals are simplified. This keeps migration low-risk.

### Parameters stay globally owned

Continue making `ParameterManager` the only parameter owner. The workflow
parameter panel should write through `set_ui_parameter()` and sync from
`parameter_changed`.

Avoid rebuilding hidden, duplicate parameter controls in batch or stage widgets.
If a stage still needs old controls for compatibility, keep them as controller
implementation details and hide them from the workflow surface.

### Data status should become first-class

Add a compact status block per stage that reads from `DataManager` and/or known
output paths. Suggested stage status vocabulary:

- `not_started`: required inputs or prior stage outputs missing
- `ready`: required inputs available
- `running`: worker active
- `done`: primary output available
- `stale`: parameters or upstream data changed after output generation
- `error`: last run failed

This is the napariTFM analogue of CellFlow's `PipelineFilesWidget` plus section
status dots.

### Styling should be centralized

Create a small module such as `napariTFM/widgets/_ui_style.py` for:

- compact layout constants
- icon/tool button helpers
- stage accents
- status colors
- section labels
- parameter grids
- button grids

Then move `_StageSection` and any repeated control styling toward those helpers.
Avoid hard-coded styling spread across stage widgets.

## Phased Application Plan

### Phase 1: Harden the shell

- Extract `_StageSection` into a reusable workflow section module.
- Add status dot support and stage accent support.
- Keep current action proxying.
- Add tests for status, accent, enabled/disabled action states, and collapse
  behavior.

### Phase 2: Add stage data status

- Define expected inputs and outputs for each napariTFM stage.
- Add a compact status panel or popover per stage.
- Back it with `DataManager` first; use filesystem paths only where batch output
  files are the source of truth.
- Add tests with fake managers so this stays independent of numerical work.

### Phase 3: Normalize stage rows

- Move the most important run/preview/cancel controls into headers.
- Reparent or hide old action controls as needed.
- Preserve existing public widget attributes during the migration.
- Test that header actions trigger the same existing child behavior.

### Phase 4: Inline stage-specific parameters

- Keep global/common parameters in `WorkflowParameterPanel`.
- For stage-only parameters, use collapsible inline sections below each stage
  row.
- Remove duplicate visible parameter groups from stage widgets only after shell
  tests prove the replacement.

### Phase 5: Simplify child widgets

- Once the shell owns layout, reduce child widgets toward controllers plus
  stage-specific panels.
- Keep backend services and numerical algorithms untouched unless a later task
  explicitly targets them.

## Things Not to Copy Blindly

- Do not copy CellFlow's file paths or biological stage names. Copy the workflow
  structure, not the domain.
- Do not make every stage file-backed if `DataManager` is the live source of
  truth.
- Do not duplicate parameters to make the UI easier to wire. That recreates the
  state drift the redesign is trying to remove.
- Be careful with hidden owner widgets. If a hidden controller owns visible
  pieces, tests should confirm it does not intercept header clicks.
- Revisit `setFixedWidth(500)` in `napariTFMWidget`; a CellFlow-style dense
  workflow benefits from compact controls, but the shell should still tolerate
  dock resizing.

## Bottom Line

Apply CellFlow to napariTFM as a workflow shell pattern:

- one vertical cockpit
- shared stage section primitive
- compact, consistent header actions
- global parameter ownership
- visible stage data status
- controller widgets migrated incrementally
- UI behavior protected by focused Qt tests

That gives napariTFM the usability gain of CellFlow without forcing a large
rewrite of analysis widgets or numerical backends.
