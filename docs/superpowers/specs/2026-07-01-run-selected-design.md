# Run Selected — Design

**Date:** 2026-07-01
**Status:** Approved (design); ready for implementation planning

## Problem

The experiments-list panel's batch action is currently "Run all": it always
processes every committed row in `self._paths`, with no way to run a subset.
The panel already has a general row-selection mechanism (`_selected_paths`,
populated by plain click / Ctrl-click toggle / Shift-click range), reused
today by "Delete selected" — but "Run all" ignores it entirely and is enabled
whenever the table is non-empty (`n > 0`), regardless of what's selected.

The owner wants batch runs scoped to whatever is currently selected, using the
same selection mechanism the table already has, rather than always running
the whole list.

## Approach

This is entirely a widget-layer change. `BatchAnalysis` (the backend) is
untouched — `process_all_folders`/`process_folder` already just process
whatever `root_folders` list they're handed in `config`; they have no concept
of "all" vs. "selected". Everything downstream of config construction
(`build_run_config`, sequential/parallel dispatch, progress reporting) is
unaffected.

### 1. Selection → config

`ExperimentsList.selected_rows()` already returns the currently-selected
paths in row order. `_widget.py`'s run method will filter
`experiment_records()`'s per-row dicts down to that selection before calling
`build_run_config`, instead of passing every record. If the selection is
empty the button is disabled (see §2), so no folders can be submitted with
nothing selected.

### 2. Button enablement

Today: `self.run_all_btn.setEnabled(n > 0)` — true whenever the table isn't
empty, independent of selection. New behavior: enabled iff
`bool(self._selected_paths)`, recomputed on every selection change (row
click, Ctrl+A, row deletion pruning `_selected_paths`) — mirroring the
existing `_update_delete_btn` pattern exactly. No fallback to "run everything"
when nothing is selected (decided explicitly over the alternative of
falling back to all rows).

Button text changes from "Run all" to "Run selected"; tooltip becomes "Run
the selected rows (Ctrl/Shift-click to select several, Ctrl+A for all)".

### 3. Ctrl+A select-all

`ExperimentsList.keyPressEvent` already special-cases Delete/Backspace to
call `delete_selected()`. Add a `Ctrl+A` case: sets
`self._selected_paths = set(self._paths)` (committed rows only — discovered/
preview rows are excluded, since they aren't run targets), restyles rows via
the existing `_apply_selection_styles()`, and re-evaluates both
Run-selected and Delete-selected enablement.

### 4. Renaming (mechanical)

Because this is a genuine behavior change (not a compatibility shim), the
`run_all_*` identifiers are renamed to `run_selected_*` so the code matches
what it does — no `_run_all_experiments` that actually only runs a subset.

In `napariTFM/widgets/_experiments_list.py`:
- `run_all_requested` → `run_selected_requested`
- `cancel_run_all_requested` → `cancel_run_selected_requested`
- `run_all_btn` → `run_selected_btn` (and its Qt `objectName`,
  `experiments_run_all_button` → `experiments_run_selected_button`)
- `_on_run_all_clicked` → `_on_run_selected_clicked`
- `set_run_all_active` → `set_run_selected_active`
- `_run_all_active` → `_run_selected_active`

In `napariTFM/widgets/_widget.py`:
- `_run_all_experiments` / `_run_all_experiments_sequential` /
  `_run_all_experiments_parallel` → `_run_selected_experiments` /
  `_run_selected_experiments_sequential` / `_run_selected_experiments_parallel`
- `_cancel_run_all` → `_cancel_run_selected`
- `_on_run_all_stage_progress` → `_on_run_selected_stage_progress`

User-facing strings and docstrings/comments that say "Run all"/"Run-all" are
updated to "Run selected" for accuracy (e.g. the `QMessageBox.critical`
title, the worker-count spinbox tooltip, `viewer_sink.py`'s docstring mention
of "Run all").

## Data flow / state

No new persisted state. `_selected_paths` is already transient, in-memory-only
UI state today (not part of `get_state`/`set_state`/saved Project bundle);
this change only adds a second consumer (run) of state that already exists
for a first consumer (delete). Row-deletion's existing pruning
(`self._selected_paths &= set(self._paths)`) already keeps the selection
consistent when rows disappear, so it covers the new consumer for free.

## Error handling / edge cases

- Empty selection: button is disabled, so the run method can't be invoked
  with nothing selected. The existing `if not records: return` guard stays
  as a defensive no-op.
- Deleting selected rows, clearing the list, or any row falling out of
  `_paths`: already pruned from `_selected_paths` today; enablement is
  recomputed from the same path used for Delete-selected.
- Cancel-while-running: unchanged — still a live toggle on the same button
  (text flips to "Cancel" while a run is active).
- A single selected row still goes through the same `num_workers` dispatch
  logic (sequential vs. parallel pool) unchanged; submitting one folder to an
  N-worker pool is already a valid, unremarkable case for
  `ProcessPoolExecutor`.

## Testing

Existing suites to extend: `tests/test_experiments_list.py` (button
enablement now selection-driven; Ctrl+A selects all committed rows and not
preview rows; renamed signals fire correctly), `tests/test_workflow_shell.py`
and `tests/test_batch_parallel.py` (renamed signal/method wiring). Add one
new case building a run config from a partial selection (e.g. 2 of 5 rows
selected) and asserting `root_folders` contains only the selected paths, in
row order.

## Out of scope

- Any change to `BatchAnalysis`/`build_run_config`'s handling of
  `root_folders` — they already accept an arbitrary folder list.
- A visible "select all" button/checkbox in the table header — Ctrl+A only,
  per the owner's choice.
- Persisting the selection across save/load or between sessions.
- Any change to how a single plain click sets both "active" (live preview)
  and selection — that dual role is unchanged; Run-selected simply reads
  whatever `_selected_paths` already holds.
