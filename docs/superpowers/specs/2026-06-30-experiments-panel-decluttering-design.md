# Experiments Panel Decluttering — Design

**Date:** 2026-06-30
**Status:** Approved (design); ready for implementation planning

## Problem

The current `ExperimentsList` panel (built by the viridis/cividis UI redesign and
the 2026-06-28 onboarding-disclosure pass) bundles too much into one visual
block: calibration, output directory, input-file-name fields, the Discover/Add
to list/Run all actions, and the table all sit under one "EXPERIMENTS"
collapsible header. Reviewing a screenshot of the running app, the owner judged
it "very busy" and identified specific causes:

- Project-wide setup (calibration, output dir, input-file names) and the
  experiment table itself are visually fused into one container, with no
  separation between "things you configure once" and "the list you work with."
- The output directory field reads as an unclear/required-looking control
  rather than the optional override it actually is (backend already falls back
  to a per-experiment `TFM_data/` bucket when unset).
- Discovered-but-uncommitted folders are invisible as a list — only a text
  counter ("N folders discovered") hints at what `Add to list` will do.
- Auto-generated column headers read as engineering jargon ("Level 1", "Level
  2") instead of plain defaults.
- The six toolbar buttons (New/Load/Save Project, Load/Save Params, Reset) sit
  in their own row below the title, adding vertical bulk the title row has
  spare horizontal room for.

This is a follow-on declutter pass on top of the already-implemented
onboarding-disclosure G0/G1/G2 state machine (see
`2026-06-28-naparitfm-onboarding-disclosure-design.md`) — it does not change
that state machine, only the internal layout of what G1 renders.

## Approach

Split the current single "EXPERIMENTS" container into two pieces with
different lifecycles, fix the output-dir affordance and column-header
defaults, give Discover a live preview, and move the toolbar into the title
row.

### 1. New "Setup" container

A new collapsible section (built with the existing reusable
`CollapsibleSection` from `_collapsible_section.py` — already used for stage
parameter panels — rather than another hand-rolled collapse widget) holds, top
to bottom:

1. Calibration row — pixel size, frame interval (unchanged fields, just
   relocated).
2. Input-file-name fields — beads/reference/cells(optional)/masks(optional),
   unchanged from today's `_build_config_header`.
3. Output directory row, **last** — see §2 below.

Starts expanded. **Auto-collapses** the first time the experiment table
transitions from empty to non-empty (i.e., right after the first `Add to
list`). The user can re-expand it manually at any time to edit setup values
for a later batch — file-name patterns and calibration are typically constant
across batches, so this is an infrequent action, not a blocked one. This does
**not** touch the Discover/Add-to-list/column-rename controls described in
§4/§5, which remain permanently visible per the onboarding doc's multi-batch
requirement — they live outside this container.

### 2. Output directory as explicit opt-in

Replace the current icon-button + "No output directory" label with a
`+ Add custom output directory` text+icon affordance (visually matching the
existing plus/add action style), shown when unset. Once a directory is
chosen, it displays the chosen path with a way to clear it back to the unset
state. This makes the actual default/override relationship visible: leaving
it unset is the normal case (falls back to the per-experiment `TFM_data/`
bucket), not a missing required field.

### 3. `ExperimentsList` loses its own collapsible chrome

The remaining piece — action row (Discover, Add to list, Delete selected, Run
all) directly above the table — renders as a flat panel: a plain,
non-interactive "Experiments" label for orientation, no header arrow / no
show-hide body. This avoids stacking two near-identical-looking collapsible
headers (Setup above it does collapse; this one never needs to, since hiding
the active experiment list would hide the thing the user is usually looking
at).

### 4. Discover → preview → harden

`discover(root)` already stages found folders into `self._discovered` without
touching the table. That staged set now renders immediately as dimmed
("preview") rows in the table, using the existing `ExperimentRow` with a
preview style (reduced-opacity / `TEXT_DIM`-toned text, no live mini-rail
status since nothing has run yet). Preview rows are:

- **Selectable and removable** via the existing `Delete selected` path before
  commit, so a bad discovery hit can be pruned ahead of committing.
- **Replaced**, not merged, by a subsequent `Discover` call on a new root —
  clicking Discover again clears any current uncommitted preview rows first.

`Add to list` hardens all current preview rows into normal committed rows
(today's `commit_discovered` behavior, unchanged), clears the preview state,
and — the first time the table goes from empty to non-empty — triggers the
Setup container's auto-collapse from §1.

### 5. Column header defaults

`nesting_columns()`'s generated header text changes from `f"Level {i+1}"` to
`f"Column {i+1}"`. Purely the auto-generated default string used before a row
is renamed via the existing table-header `rename_column`; no behavioral
change to renaming itself.

### 6. Toolbar moves into the title row

The six toolbar buttons move out of their own `QGridLayout` row into the
existing `title_row`, right-aligned opposite the "napariTFM" label via a
stretch, as icon-only `QToolButton`s (tooltips carry today's button text:
"New Project", "Load a project", "Save project as…", etc.). Grouped with thin
visual dividers, preserving today's grouping logic (Project actions are
Save-as/no-autosave; Params actions are recipe-only import/export):

```
napariTFM                          [New|Load|Save] | [Load|Save] | [Reset]
                                     Project           Params
```

Needs 4 new SVG icon bodies in `_icons.py`: `new`, `load` (folder-open),
`save`, `reset`. `load`/`save` are reused for both the Project and Params
groups — grouping position + tooltip text disambiguate which action fires,
matching how other icon actions in this codebase already rely on tooltip text
for specificity (e.g. `gcv`, `mesh`).

## Data flow / state

No new persisted state. `Setup` container's collapsed/expanded flag is
transient UI state (not part of `get_state`/`set_state`/saved Project bundle)
— it always starts expanded on a fresh G1, the same way today's hand-rolled
collapse defaults to expanded. Preview rows (`self._discovered`) are already
transient, in-memory-only state today; this change only adds a rendering path
for them, not new persistence.

## Testing

- `Setup` container: starts expanded in a fresh G1; auto-collapses on first
  `Add to list`; stays manually re-expandable afterward; collapse state is
  not part of saved/loaded Project state.
- Output directory: renders `+ Add custom output directory` when unset, shows
  the chosen path with a clear affordance once set, round-trips through
  `set_output_dir`/`output_dir` unchanged.
- Discover preview: `discover(root)` renders preview rows immediately;
  preview rows are selectable/removable via `Delete selected`; a second
  `discover()` call replaces rather than merges the preview set; `Add to
  list` hardens preview rows into committed rows and clears preview state.
- Column defaults: `nesting_columns()` default header is `"Column 1"`
  (not `"Level 1"`) when nesting yields a single level.
- Toolbar: all 6 actions present as icon buttons in the title row with
  correct tooltips; existing click-handler wiring (`new_project_btn.clicked`
  etc.) unchanged, only the container/icon presentation changes.

Existing suites to extend: `tests/test_experiments_list.py`,
`tests/test_workflow_shell.py`.

## Out of scope

- Any change to the G0/G1/G2 disclosure state machine itself (still governed
  by `2026-06-28-naparitfm-onboarding-disclosure-design.md`).
- Changing what fields exist (no new calibration/input fields added or
  removed) — this is purely a layout/affordance pass.
- Persisting the Setup container's collapsed state across save/load.
- Animated expand/collapse transitions (plain show/hide, matching the
  existing `CollapsibleSection` behavior).
