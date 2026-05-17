# Tier 1 CellFlow Alignment — Design

## Goal

Close the largest UX gaps between napariTFM and CellFlow in a single PR by
relocating stage-specific parameters into each stage section, rebuilding the
data-status panel as CellFlow-style file rows, making `StageSection` a real
nestable primitive with accent inheritance, consolidating the stage header
button cluster, and removing the fixed 500 px shell width.

This continues the direction set by
`docs/cellflow-ui-concept-for-naparitfm.md` (Phases 1–4) and supersedes the
input-assignment portion of
`docs/superpowers/specs/2026-05-18-preprocessing-ui-redesign-design.md` only:
the preprocessing input rows are absorbed into the unified data-status panel
rather than living in a separate body panel.

## Scope

In scope:

- Top-level shell layout (`napariTFM/widgets/_widget.py`).
- `StageSection` (`napariTFM/widgets/_stage_section.py`) gains nesting and
  accent inheritance.
- `StageDataStatusPanel` (`napariTFM/widgets/_stage_data_status.py`) rebuilt as
  CellFlow-style file rows with per-artifact actions.
- New top-of-shell "Project" section (`napariTFM/widgets/_project_section.py`).
- Per-stage parameter panels move inside their stage sections.
- `_ui_style.py` gains palette tokens, muted-accent algorithm, glyph
  constants, and an artifact-row factory.
- `PipelineDataWidget` and `WorkflowParameterPanel` are deleted.
- `PreprocessingDataPanel` is removed from the visible body.

Out of scope:

- Filesystem-driven status (DataManager remains source of truth).
- Top-level project directory and per-position metadata.
- Custom slider/spinbox/range-slider factories.
- Layout grid helpers (`section_grid`, `block_grid`, `sweep_parameter_grid`).
- Forward-compatible state I/O.
- Stress/MSM body redesign beyond parameter relocation.
- Batch widget body redesign.
- Numerical algorithms and backend services.

## Top-Level Layout

```
┌─ napariTFM ─────────────────────────────────┐
│ ▼ Project                                   │
│   Pixel size (µm)   [   0.108 ]             │
│   Frame interval    [    1.000 ] min        │
│   [Save params] [Load params] [Reset]       │
│   [Clear all data]                          │
│                                             │
│ ▶ Preprocessing                  ●          │
│ ▶ Displacement                   ●          │
│ ▶ Force / FTTC                   ●          │
│ ▶ Stress / MSM                   ●          │
│ ▶ Batch analysis                 ●          │
└─────────────────────────────────────────────┘
```

Changes from today:

- Remove `self.setFixedWidth(500)` at `_widget.py:227`; the dock determines
  width.
- Delete `PipelineDataWidget` and its instantiation; per-stage status panels
  cover the artifact overview role.
- Delete the standalone `WorkflowParameterPanel` class. Its General section
  becomes the body of a top "Project" `StageSection`. The remaining
  Preprocessing/Displacement/Force/Stress parameter groups move into each
  stage's nested "Parameters" inner section.
- Section status dots render right-aligned in the header. The "Project"
  section has no status dot (it is not a workflow stage).
- All stage sections collapsed by default. The "Project" section starts
  expanded.

## StageSection Primitive

`StageSection` becomes a nestable `CollapsibleSection`-style primitive,
modeled on `CellFlow/src/cellflow/napari/widgets.py` lines 33–237.

### Header

```
[●] Preprocessing                          [⚙] [▶] [👁]
```

- 10×10 status dot, left of title, color from
  `_ui_style.STATUS_COLORS`.
- Title, bold, accent color, accent left-border stripe (preserves the
  current `STAGE_ACCENT_COLORS` system in `_ui_style.py`).
- Right-aligned tool buttons, 24×24, auto-raise, with tooltips:
  - `⚙` `params_btn`: checkable; toggles the inner "Parameters" section.
  - `▶` / `■` `run_cancel_btn`: single button whose icon and tooltip swap
    based on stage status (`running` → cancel icon).
  - `👁` `preview_btn`: checkable; present only on stages whose controller
    exposes a preview action (today: preprocessing and displacement).
- The previous separate `save_btn` is removed from the header. Save actions
  move to per-artifact rows in the data-status panel.
- `config_btn` is kept as a deprecated alias pointing at `params_btn` for one
  commit during migration, then removed.

### Body

When the outer section is expanded, the body shows in order:

1. A `StageDataStatusPanel` (always visible inside the body when the section
   is expanded).
2. A nested "Parameters" `StageSection`, initially collapsed, toggled by the
   header `⚙` button. Its body is a `QFormLayout` with the stage's
   parameters.

### Accent Inheritance

`StageSection.__init__` accepts an optional `accent_key`. If omitted, the
section walks up its parent chain to find an ancestor `StageSection` with an
accent and uses the muted variant of that accent (see Styling). The inner
"Parameters" section is created without an explicit accent so it inherits and
mutes the stage accent automatically.

### Action Proxy

The existing `_ActionStateSync` event filter pattern in
`_stage_section.py:14-30` is preserved. `run_cancel_btn` and `preview_btn`
proxy clicks and enabled state from underlying child-widget buttons.
`params_btn` is owned directly by `StageSection` and toggles the inner
section visibility.

### Run/Cancel Toggle

`set_status(status)` controls the icon and tooltip on `run_cancel_btn`:

- `not_started`, `ready`, `done`, `stale`, `error` → run icon, tooltip
  "Run".
- `running` → cancel icon, tooltip "Cancel".

The single button click is routed to the proxied run handler when in run
state and to the proxied cancel handler when in cancel state. The underlying
controller buttons stay separate; only the visible surface is unified.

## Data-Status Panel

`_stage_data_status.py` is rebuilt around an artifact-row primitive modeled
on `CellFlow/src/cellflow/napari/widgets.py:282-452`
(`_PipelineFileRow` / `PipelineFilesWidget`).

### Row Layout

```
[✓]  Preprocessed beads          512×512×100   [👁] [💾]
[✗]  Cell stack                  Missing
[○]  Cells (optional)            Optional      [↑]
```

Four columns plus a right-side action cluster:

1. Status glyph: `✓` (available), `✗` (required missing), `○` (optional
   missing), `⟳` (running), `⚠` (stale or error). Color from
   `STATUS_COLORS`.
2. Artifact label from `DataArtifactSpec.label`.
3. Info cell: shape string when available, `Missing` or `Optional`
   otherwise. Elided if narrow.
4. Actions, context-dependent:
   - Available output: `👁` view-in-viewer, `💾` save-to-file.
   - Available input: `👁` view-in-viewer, `↑` re-assign or replace.
   - Missing required input: `↑` load or assign (whichever the stage
     registers); no `👁`.
   - Missing optional input: `↑` load or assign; no `👁`.
   - Missing output: no actions.

### Grouping

Rows are split into two labeled subgroups inside the panel: "Inputs" and
"Outputs". Order within each group follows `STAGE_DATA_ARTIFACTS` in
`_widget.py:28-53`, which remains the source of truth for artifact
membership.

### Per-Artifact Action Registration

`DataArtifactSpec` is extended with two optional callable fields,
`on_view` and `on_action`. The Python representation (dataclass,
NamedTuple, etc.) follows the current definition in
`_stage_data_status.py`; only the new fields are added.

The shell sets these callables when constructing each stage. `on_view`
invokes the existing `VisualizationManager` path that today loads artifacts
as napari layers. `on_action`:

- For outputs, calls the stage controller's existing save method.
- For inputs, calls either the stage's `load_active_layer(name)` path
  (preprocessing) or the stage's file-dialog load path (displacement,
  others), whichever the stage registers.

No new controller methods are added; existing methods are referenced by
callable.

### Refresh

`DataManager.add_change_callback(self._on_pipeline_data_changed)` continues
to drive `refresh_stage_statuses()`. Each row's glyph, info cell, and
action-button enabled state refresh on the same callback. No new state
plumbing.

### Preprocessing Consolidation

`PreprocessingDataPanel` (`preprocessing_widget.py:26-132`) is no longer
instantiated in the visible body. Its three input rows (Reference, Beads,
Cells) are now ordinary rows in preprocessing's `StageDataStatusPanel` with
`on_action` wired to the existing `load_active_layer('reference' | 'beads'
| 'cells')` methods on the preprocessing controller. The class file may
remain, but the widget is not added to the layout.

This supersedes the input-rows portion of the 2026-05-18 preprocessing UI
redesign. The rest of that spec (single composed preview action, removal
of large body run/cancel/save buttons, preview layer behavior, post-run
visualization) remains in effect.

### Stale Glyph

The `⚠` glyph and its mapping to `stale` are added. `DataManager` does not
emit `stale` today, so the glyph never appears in normal use. A unit test
pins the mapping so future staleness detection lights it up without UI
changes.

## Styling

Changes are confined to `napariTFM/widgets/_ui_style.py`.

- **Palette tokens.** Replace hardcoded hex literals with named palette
  dicts and `stage_accent(key)` / `muted_stage_accent(key)` accessors. The
  default palette uses the current colors; no theme is added in this PR.
- **Muting algorithm.** Port the saturation-reduction algorithm from
  CellFlow `ui_style.py:73-85`: convert accent to HSL, set saturation to
  ~35 %, adjust lightness toward midtone.
- **Glyph constants.** Named constants for status glyphs (`✓✗○⟳⚠`) and
  action glyphs (`👁💾↑⚙▶■`).
- **Row factory.** `make_artifact_row(spec, on_view, on_action) -> QWidget`
  returns a row laid out as described above. Used by
  `StageDataStatusPanel`.

CellFlow's `block_grid`, `section_grid`, `sweep_parameter_grid`, slider /
range-slider factories, label-autosize patch, and typography helpers are
not ported in this PR.

## File Layout

```
napariTFM/widgets/
├── _widget.py                 # Slimmer; WorkflowParameterPanel deleted
├── _stage_section.py          # Nestable; accent inheritance
├── _stage_data_status.py      # File-row rebuild
├── _ui_style.py               # Palette tokens, muted accents, glyphs, row factory
├── _project_section.py        # New: general params + save/load/reset/clear
├── _stage_parameter_panel.py  # New: per-stage QFormLayout factory
├── preprocessing_widget.py    # PreprocessingDataPanel no longer in visible body
├── displacement_analysis_widget.py
├── fttc_widget.py
├── msm_widget.py
└── batch_analysis_widget.py
```

`_pipeline_data_widget.py` is deleted.

## Migration Order

Six commits, each independently green:

1. **Palette indirection.** Add `stage_accent()` and `muted_stage_accent()`
   to `_ui_style.py`; route existing call sites through them. No visible
   change.
2. **Nestable StageSection.** Add accent inheritance, parent-walk lookup,
   and an inner-section API. Existing single-level usages unchanged. Tests
   for parent-walk and accent muting.
3. **Header consolidation.** Replace `run_btn / preview_btn / cancel_btn /
   save_btn / config_btn` with `params_btn / run_cancel_btn / preview_btn`.
   Implement run/cancel toggle in `set_status()`. In this commit
   `params_btn` is wired to whatever `config_btn` toggled before (the
   existing overlay parameter panel), so the visible behavior is
   unchanged. Step 4 replaces that wiring once parameters move into the
   inner section. Keep `config_btn` as a deprecated alias of `params_btn`
   during this commit. Update `tests/test_workflow_shell.py` button-name
   assertions.
4. **Parameters move into stage sections.** Delete `WorkflowParameterPanel`.
   Create `_project_section.py` with general params (`pixel_size`,
   `frame_interval`) and Save / Load / Reset Parameters and Clear All Data
   buttons. Per-stage parameters render inside each stage's nested
   "Parameters" inner section, toggled by `params_btn`. `ParameterManager`
   ownership unchanged.
5. **Data-status file rows.** Rewrite `_stage_data_status.py` with row
   layout, per-spec `on_view` / `on_action` callables, status glyphs, and
   grouped inputs/outputs. Wire view to `VisualizationManager`; wire save
   and load to existing controller methods. Delete `PipelineDataWidget`.
   Add row-level tests.
6. **Preprocessing consolidation and width drop.** Stop adding
   `PreprocessingDataPanel` to the visible body; route preprocessing inputs
   through the unified data-status panel. Remove `setFixedWidth(500)` at
   `_widget.py:227`. Drop the deprecated `config_btn` alias from step 3.
   Update `tests/test_preprocessing_ui_redesign.py`.

Each step compiles, the workflow remains operable, and tests pass between
commits.

## Test Strategy

Existing tests:

- `tests/test_workflow_shell.py`: button-name assertions update in step 3.
- `tests/test_preprocessing_ui_redesign.py`: assertions about the separate
  body input panel update in step 6 to assert the consolidated data-status
  rows instead.
- `tests/test_pipeline_data_io.py`: review for references to
  `PipelineDataWidget`; redirect to per-stage panels or remove obsolete
  assertions in step 5.

New tests under `tests/`:

- `test_stage_section_nesting.py`: parent-walk accent lookup; muted variant
  rendering on nested sections.
- `test_artifact_row.py`: row layout, glyph per status, `on_view` and
  `on_action` invoked, button enabled state matches artifact availability.
- `test_project_section.py`: general params present, save/load/reset/clear
  buttons exist and call the right `ParameterManager` / `DataManager`
  paths.
- `test_stage_section_header.py`: `params_btn` toggles the inner section;
  `run_cancel_btn` icon and tooltip swap on status transitions.

Backend and numerical tests untouched.

## Risks and Mitigations

- **PipelineDataWidget deletion** may break external references. Grep for
  usages in step 5 before deletion; remove or redirect.
- **Header rename** breaks tests asserting `run_btn`, `cancel_btn`,
  `save_btn`, `config_btn`. Mitigated by the `config_btn → params_btn`
  alias and an explicit test update in step 3.
- **Stale glyph without staleness logic**: glyph is defined but never
  rendered today. A unit test pins the status→glyph mapping so the
  integration is ready when staleness detection lands.
- **Preprocessing input loss of affordance**: the existing assign-from-
  active-layer flow is preserved exactly; it just moves from the dedicated
  body panel into the unified data-status panel rows for preprocessing
  inputs.

## Compatibility Constraints

- `ParameterManager` remains the only parameter owner. UI changes write
  through `set_ui_parameter()` and sync from `parameter_changed`.
- `DataManager` remains the source of truth for artifact availability.
- Stage controller worker lifecycles, cancel behavior, save formats, and
  calibrated TIFF output are unchanged.
- No backend service signatures change.
- The `napariTFMWidget` public attribute surface used by tests is preserved
  except for explicit renames listed in step 3.
