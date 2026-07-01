# Status dots (eager) vs. output display (lazy) — corrected model (2026-07-01)

An earlier phase built a "lazy per-stage status reveal" system: status dots
started as `"unchecked"` (dashed circles) and only read the `.ntfm` when their
circle was clicked. That inverted the actual requirement. This note records the
corrected design that replaced it, and why.

## The requirement (from Artur)

1. Discover folders, add them to the list. Some already have `TFMresults.ome.tif`
   output, some don't. **The status dots must show this right away** — no click.
2. On selecting an experiment, load **inputs only** (`beads.tif`, `reference.tif`,
   `cells.tif`, `masks.tif`). No output series is decoded.
3. Only when the user **clicks a circle** does the program decode
   `TFMresults.ome.tif`, and only the requested series — clicking the displacement
   circle loads displacement, etc.
4. This loading is **purely for display**. Any calculation reads its inputs
   directly from disk, so nothing needs to be resident in memory to run.

## The split that makes it work

- **Status = eager.** Which measures a `.ntfm` carries is a header-only walk of
  the OME-TIFF (no pixel decode), cached by `ntfm.populated_measures` on
  `(path, mtime, size)`. `_experiment_stage_status` always reads it;
  `ExperimentsList.refresh_statuses` paints every row from it synchronously. Dots
  show the truth the moment folders land in the list. There is no `"unchecked"`
  state and no dashed circle.
- **Data = lazy, per-series, display-only.** Clicking a circle calls
  `_load_stage_results(path, [stage])`, which reads the table once
  (`_read_stage_arrays`) and applies only the requested stage (`_apply_*_result`)
  into `DataManager` + the viewer. No prerequisite stages are pulled in: because
  calculations re-read from disk, force doesn't need displacement resident, and
  stress gets its force-grid downscale factor from the parsed file config rather
  than `data_manager.force_results`.

Two click surfaces both reach the same load: the pipeline-panel spine circles
(`StageSpine.clicked` → `_on_stage_node_clicked`, loads for the active
experiment) and the list row dots (`MiniRail` → `ExperimentRow.stage_clicked` →
`ExperimentsList.stage_load_requested` → `_on_row_stage_clicked`, which selects
the row then loads).

Selection (`_on_active_experiment_changed`) loads inputs only and clears
`_loaded_stage_data`; nothing output is auto-restored, so a switch away and back
requires clicking the circle again to redisplay.

## What was torn out

- The lazy-*status* layer: `read_outputs=False`, the `"unchecked"` status, dashed
  circle rendering in `MiniRail`/`StageSpine`, `ExperimentsList._revealed` +
  `revealed_stages_for` + `row_stage_revealed`, the `full_status_fn` vs
  `status_fn` split, and `_widget._revealed_stages` / `_sync_loaded_stage_data` /
  `_stage_cascade`.
- An async `thread_worker` an even earlier cut had put on `refresh_statuses`
  (`_status_worker` / `_status_token` / `_compute_statuses` / …). It was
  parallelising a header read that's cheap and cached, and it introduced a flaky
  `GeneratorWorkerSignals has been deleted` SIGABRT (worker outliving teardown).
  `refresh_statuses` is synchronous again.

## Verification

Full suite: 603 passed. Per-file runs of the touched tests are green; the only
cross-file crash seen is the pre-existing flaky offscreen-Qt teardown
segfault (non-deterministic, unrelated to this work).
