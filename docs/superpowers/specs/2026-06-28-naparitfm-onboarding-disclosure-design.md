# napariTFM Onboarding — Progressive Disclosure Design

**Date:** 2026-06-28
**Status:** Approved (design); ready for implementation planning

## Problem

When a user first opens the napariTFM dock, there is no clear entry point. The
panel renders its full machinery at once — the batch tooling (discover folders,
Run all) and the interactive single-experiment tooling (four stage pills) sit
side by side, all live, with nothing signalling "start here." The most
confusing element for a newcomer is four inert stage pills under an empty
experiment list: controls for tuning data that doesn't exist yet.

A help icon would annotate this confusion rather than remove it.

## Approach: gated reveal

The panel renders only as much as the current state earns. Complexity appears as
the user progresses, so the structure itself teaches the workflow. The happy
path the UI steers toward is: **open a project → build a list of experiments →
tune or run them.**

State is **derived, never stored** — a pure function of two facts:

- `_project_open` — has the user started or loaded a project?
- `experiments_list.active()` — is a single experiment row selected?

### The three states

| State | Condition | Visible |
|---|---|---|
| **G0 — No project** | `not _project_open` | Brand + the two toolbars only. Nothing below the header exists. |
| **G1 — Project open** | open, `active is None` | The workspace: file-name fields, optional Output field, calibration, the repeatable *columns → Discover → Add to list* unit, the scrollable experiment list, Run all. |
| **G2 — Tuning** | open, a row selected | Additionally: the four stage pills + pipeline context label + status line. |

Transitions ride signals that already fire:

- **New Project / Load Project** set `_project_open = True` → G0 → G1.
- `experiments_list.active_changed` → G1 ↔ G2 (selecting a row reveals the
  pills; deselecting hides them again).

A single `_update_disclosure()` method on `napariTFMWidget` reads the two facts
and toggles the visibility of widget groups. It is called from `__init__` and on
each of the driving signals. The reveal is monotonic in practice but degrades
gracefully if a user steps backward (re-points a folder, deselects a row).

## G0 toolbar

```
napariTFM    New Project   Load Project   Save Project
             Load Params    Save Params    Reset
```

Two rows in the narrow napari dock:

- **Project row** (on the brand line — front-door priority):
  - **New Project** — clears to an empty G1 (empty list, default file names,
    default parameters, no output dir). Writes nothing.
  - **Load Project** — restores a saved project bundle.
  - **Save Project** — always **Save-as** (file dialog every time).
- **Parameters row:**
  - **Load Params** / **Save Params** — import/export a parameters preset.
  - **Reset** — reset parameters to defaults.

## Project = single source of truth

A saved **Project file bundles everything**:

- experiment folders + per-row free-form columns
- optional output directory
- calibration (pixel size, frame length)
- run options (disabled stages)
- **analysis parameters**

The **Parameters** buttons act only on the recipe portion: **Save Params**
exports the analysis parameters as the existing portable `tfm_params` preset;
**Load Params** imports such a preset into the open project, overwriting the
current parameters. This unifies what is today split across a `tfm_params`
preset and a `tfm_experiment_series` file into one Project bundle, with the
preset format retained purely as an import/export interchange for the recipe.

**Persistence model change:** explicit **Save Project** replaces today's
autosave-to-output-dir config. Because there is no longer an automatic save,
the shell tracks a **dirty flag** (set on any experiment-list or parameter
change) and **New Project / Load Project** on a dirty project prompt
"discard unsaved changes?" before proceeding.

## Knock-on changes

1. **ExperimentsList** loses its own Open/Save buttons in the EXPERIMENTS
   header — load/save is now Project-level in the shell toolbar.
2. **Output directory** demotes from a precedence-setting picker that gated the
   UI to an ordinary **optional field** in the G1 config area, alongside the
   optional cells/masks inputs.
3. **Scrollable, fixed-height experiment list.** The experiment rows live in a
   dedicated scroll region with a fixed maximum height. Hundreds of discovered
   positions scroll internally rather than pushing the rest of the panel down.
   (The whole dock already lives in an outer `QScrollArea`; this is an inner,
   bounded region for the rows specifically.)

## Multi-batch workflow (preserved)

Discovery is a **repeatable** action, not a one-shot front door. Within G1 the
user can: set columns for a batch (e.g. `condition = A`) → Discover a root →
Add to list; then repeat with `condition = B` from a different root. Rows
accumulate, each carrying its own columns. The disclosure layer must keep the
columns → Discover → Add-to-list unit permanently available in G1 — it must not
treat "discovered once" as a finished step. The existing code already supports
this (columns copied per-commit, `add_folders` accumulates); the reveal logic
only needs to avoid locking it down.

## Discover button tooltip

The Discover action carries a **dynamic tooltip**, rebuilt whenever a file-name
field changes, naming only the filled input files (required always present;
optionals only when non-empty). Examples:

- beads + reference → *"napariTFM will scan the chosen folder for subfolders
  containing **beads.tif** and **reference.tif**, and initialize each for
  analysis."*
- + cells → *"…containing **beads.tif**, **reference.tif** and **cells.tif**…"*
- + cells + masks → *"…containing **beads.tif**, **reference.tif**,
  **cells.tif** and **masks.tif**…"*

## Testing

Disclosure is a pure, signal-driven function, so it is unit-testable headless:

- Assert which widget groups are visible in each of G0 / G1 / G2.
- Assert New Project clears to an empty G1 and Load Project restores G1/G2.
- Assert Save Project → Load Project round-trips the full bundle (folders +
  columns + output + calibration + parameters).
- Assert Load Params imports only the recipe portion into an open project.
- Assert the dirty flag is set on experiment/parameter changes and that
  New/Load Project guard on it.
- Assert the experiment list has a bounded maximum height.

`tests/test_workflow_shell.py` is the home for these.

## Out of scope

- First-run coach overlay / spotlight tour (rejected: annotates rather than
  removes confusion).
- Animated transitions between states (plain show/hide is sufficient).
- Any change to the analysis pipeline itself.
