# napariTFM — Open Worklist

> Accomplished items are pruned (see git history for completed UI-redesign
> slices P0–P8, the UI-Coherence roadmap, the BISM port, the 2026-06-29
> batch/cancel/colorbar/sink work, the 2026-06-30 unified-logging (§1) +
> per-position Export-to-CSV (§2) work, and the §3 streaming-follows-active-
> position work). What remains below is **open work only**, ranked
> easy-wins-first.

---

## Open after the round-2 review merge (2026-07-04)

Leftovers from PR #3 (`CODE_REVIEW_FINDINGS_2026-07-03.md`, now in-repo) that
were *not* landed with the merge. The Tier-1 bugs (B-1 odd-grid FFT, B-2 stress
error-swallow, B-4/B-5/B-6), the Tier-2 dead-code deletions, and the A-1/A-3/A-4
refactors are all done and test-green (625 passed). What remains, ranked
easy-wins-first:

### Vectorize `downscale_flow`  ·  DONE (2026-07-04)
`displacement_analysis.py::downscale_flow` was an O(H·W) Python double loop
computing a pure block-mean, on every displacement calc. Replaced with
`flow[:nh*f,:nw*f].reshape(nh,f,nw,f,2).mean(axis=(1,3))`. Measured **~11×**
faster on a 1000×1000 flow at factor 4 (the finding's "~100×" was optimistic),
and the output is **bit-identical** to the loop (max|diff| = 0). Locked with an
equivalence test against a reference reimplementation of the old loop, plus
factor=1 passthrough and an exact-block-average test. 633 passed.

### Split compute-critical from viz-only validation in `validate_fttc_parameters`  ·  DONE (2026-07-04)
`validate_fttc_parameters` is the pre-compute gate for `calculate_force_field` /
`find_optimal_regularization`, but it was failing the whole force computation on
visualization-only params (`force_arrow_scale`, `f_max`, `force_vector_stride` —
none of which enter the traction solve) and enforcing `regularization > 0` even
under `auto_gcv=True`, where the manual value is ignored. **Fix:** dropped the
three viz-only checks from the compute gate (they belong at the rendering layer,
not here — not shuffled into a new uncalled validator, which would just recreate
dead code) and gated the regularization check on `not auto_gcv`. Test-locked:
viz-only params no longer block, reg≤0 is fine under auto-GCV but still rejected
with it off, and the existing compute-critical checks are unchanged.

### BUG: interactive upstream re-run resurrects a stale downstream stage on disk  ·  DONE (2026-07-04, B-3)
CONFIRMED bug, interactive-path only: re-running **displacement** after
displacement→force were on disk left `disp_v2` paired with the old `force_v1`
(computed from `disp_v1`), because `merge_arrays` restored the force that was
absent-from-the-write (invalidated in memory) but present-in-old.
**Fix (the `_DOWNSTREAM`-aware merge option, not `merge_existing=False`):**
`merge_arrays` (`utilities/ntfm.py`) now treats a stage *present* in the write as
proof it was recomputed, so its downstream stages (`_DOWNSTREAM_ARRAY_KEYS`) are
**not** resurrected from disk even when absent from the write. This distinguishes
an upstream re-run (displacement present → drop stale force) from a legitimate
force-only resume (displacement absent → preserve it) purely from the arrays
being merged — no call-site knowledge, so it fixes the batch path too. Chose this
over a blunt `merge_existing=False` because the latter would erase displacement
during a real force-only resume. Test-locked in `test_ntfm_merge.py`: the
buggy-direction unit test was flipped to assert non-resurrection, plus symmetric
stress-under-fresh-force, combined preserve-upstream-drop-downstream, and an
end-to-end `results_to_ntfm` disk regression. 628 passed.

### Collapse the vector-stage Widget/Controller triplication + one blessed lifecycle  ·  L  ·  needs manual in-app Qt verification (A-2 remainder)
The A-2 preview-layer helper landed; **this half did not.** Stages 2/3/4
(displacement/force/stress) are still ~75% copy-paste, and the drift is real
bugs: three different `cancel()` semantics (one `terminate()`, one a write-only
flag that cancels nothing — D-12 was the worst of these and *is* fixed), and
preview running synchronously on the GUI thread in displacement/stress but on a
`thread_worker` in force. Plan: (1) fix the remaining drift in place (small,
individually revertable), then (2) hoist run→freeze→progress→complete/fail→
unfreeze + **one** blessed cancel into the base as a *non-overridable* template
method so the invariants can't drift again, then (3) merge the disp/force
controllers into one `VectorStageController(StageSpec)` (mirroring
`_VECTOR_FIELD_CONFIG`); stress stays a thin subclass for its mask input.
Preprocessing stays bespoke. Add one parameterized drift-regression test
(freeze-on-run / unfreeze-on-terminal / cancel-actually-cancels). **This touches
Qt worker teardown across four controllers and the suite leans on fakes there —
it wants its own reviewed change with manual in-app verification, not a blind
headless refactor.**

### Minor cleanups (low priority, do opportunistically)
- **Mask resized twice per stress folder** and the resize logic is duplicated
  verbatim between `batch_analysis.py:1261-1273` and `stress.py:73-85` — resize
  once, share one helper.
- **Three near-identical `physical_scale` dicts** (`displacement_analysis.py:213`,
  `fttc.py:100`, `bism.py:365`) differ only in a unit-name string — one helper.

**Considered and deliberately dropped** (don't re-open without a reason): the
dead params `tfm_folder`/`folder`/`preprocessed_data` (leftover signatures, low
value); the fttc GCV micro-cleanups (`_interp_vec2grid` NaN branch, `np.copy`,
`np.max([minGi,0])` — fttc is numerically sensitive, not worth the risk for
cosmetics). The `metrics_calculator.py` polarization fix (`eigvals`→`eigvalsh`,
centroid-not-origin moment) is real but stays parked under the existing "#9:
wire up metrics later" decision — fix it *when* it's wired up.

---

## Ranked open work (2026-06-29)

### Remove the green/red input/output-file status icons  ·  DONE (2026-07-02)
Redundant with the colormap-spine rail, which already shows per-stage status —
drop the separate green/red icons that indicate input/output file presence.
**Done.** Removed the `StageFileStatusRow` widget (the per-*artifact* red→green
dot row under each stage header) and its logic: deleted
`widgets/_stage_file_status.py`, the `FILE_STATUS_COLORS`/`file_status_color`/
`file_status_state` helpers in `_ui_style.py`, the `_build_*_specs` builders +
`_stage_status_panels_by_key` construction + `status_panel` plumbing in
`_widget.py`/`_stage_section.py`. The colormap-spine rail (`StageSpine`) and the
experiments-list rail (`MiniRail`) — the per-*stage* status nodes — are
untouched. **One coupling preserved:** the spine node's in-memory status when no
experiment is selected used to come from `panel.refresh()`; `refresh_stage_statuses`
now calls `compute_stage_status(data_manager, STAGE_DATA_ARTIFACTS[key])` directly
(kept `STAGE_DATA_ARTIFACTS` + `_stage_data_status.py` for exactly this).
**One capability removed with the dots:** clicking a red input dot was the *only*
UI trigger for "assign the active napari layer as this input"
(`load_active_layer`/`load_result_artifact`) — no button or shortcut for it
survives. In the experiments-list-driven workflow inputs load from disk on row
selection, so this was a legacy manual override; the widget/controller methods
still exist (just unreachable from the UI) if we want to re-expose them via a
dedicated control later. Tests: deleted `test_stage_file_status.py` + the
dot-routing/embedding tests in `test_workflow_shell.py` and the file-status-color
tests in `test_ui_style.py`; adapted the status-transition test to assert the
spine node (not the dots). 623 passed (full suite).

### Replace "Run all" with "Run selected"  ·  DONE (2026-07-02)
Scope batch runs to the experiments-list's existing row-selection mechanism
(the same one "Delete selected" already uses) instead of always running
every committed row. Design spec:
[`docs/superpowers/specs/2026-07-01-run-selected-design.md`](docs/superpowers/specs/2026-07-01-run-selected-design.md).
**Done as specced** — pure widget-layer change, `BatchAnalysis` untouched.
`_run_selected_experiments` (`_widget.py`) filters `experiment_records()` down
to `ExperimentsList.selected_rows()` (row order) before `build_run_config`, so
`root_folders` carries only the selected paths. The button is now
selection-driven (enabled iff `_selected_paths`, recomputed via the new
`_update_run_btn` folded into `_update_delete_btn`; no more `n > 0`), text is
"Run selected", and a new `ExperimentsList.select_all()` (Ctrl+A, committed
rows only — preview rows excluded) covers "run everything". All `run_all_*`
identifiers/signals/strings renamed to `run_selected_*` across `_widget.py`,
`_experiments_list.py`, `viewer_sink.py`, `queue_progress_sink.py`,
`batch_analysis.py`. Tests updated + added (partial-selection config, Ctrl+A
select-all, preview-exclusion, no-selection no-op) in `test_experiments_list.py`
and `test_workflow_shell.py`; 243 passed in the affected suites, 630 in the full
run (the only 5 failures are a pre-existing imageio/tifffile `fps`-kwarg env
drift in `test_batch_visualizations.py`, unrelated).

### BUG: Run All stops before finishing all queued tasks  ·  LIKELY FALSE ALARM (2026-07-02)
Per the owner (2026-07-02): probably **not** a queue/loop bug — the run most
likely crashed on **bad input images** and aborted, rather than terminating
early through a faulty loop condition. Leave un-fixed pending a real repro; if
it resurfaces, look at the failing position's images first, not the queue logic.

### BUG: Clicking the preprocessing rail circle wouldn't load its output  ·  DONE (2026-07-01)
Clicking the first (preprocessing) icon on either rail claimed "no output" even
when `preprocessed_beads.tif`/`preprocessed_reference.tif` existed on disk, and
on the mini rail the viewer was left showing raw input data instead.
**Root cause:** `_load_stage_results` (`widgets/_widget.py`) filtered every
requested stage through `_NTFM_STAGES = ("displacement", "force", "stress")` —
a stale hardcode from before preprocessing persisted its own output — so a
"preprocessing" click was silently dropped before any disk read happened. The
status dot (`_experiment_stage_status`) correctly checked for the TIFFs and
said "done"; only the click-to-load path disagreed. The mini rail's "loads
input instead" symptom was downstream of the same no-op: selecting the row
loads raw inputs, and the failed preprocessing load never overwrote them.
**Fix:** added `_apply_preprocessing_result`, a load path for preprocessing
that reads the persisted TIFFs from `experiment_output_dir` directly (no
`.ntfm`/tidy-table involved) and binds them via
`visualization_manager.begin_preprocessing_stream()` — mirrors the live
interactive-run path. `_load_stage_results` now handles preprocessing
separately from the `_NTFM_STAGES` filter instead of dropping it. Test-locked
in `test_reload_on_selection.py` (load, no-op-when-missing, and status-line
regression tests). 624 passed.

### 3b. Per-stage layer isolation during streaming  ·  DONE (2026-06-30)
The `120d0a0` machinery only ever ran on the **run-all** path (`ViewerSink`
calls `isolate_layers` per stage) — and it was provably correct there (drove the
real `VisualizationManager` through the sink; isolation held at every
transition). The gap was the **interactive per-stage Run buttons**: they stream
via `begin_*_stream` + `stream_*_frame` and **never isolated at all**, so a
direct stage run left every other layer visible ("everything bleeds in"). Only
the previews isolated, which is why preview looked right and the run didn't.
**Fix:** moved the takeover into the three streaming entry points themselves —
`begin_vector_field_stream`, `begin_stress_stream`, `begin_preprocessing_stream`
now hide non-stage layers, so both the interactive path and the sink isolate for
free. Used a new `hide_other_layers()` (hides unrelated layers but, unlike
`isolate_layers`, does **not** force-show the stage's own layers) so the
deliberate "preserve per-layer visibility across a re-run" behavior survives — a
magnitude layer the user hid stays hidden. Test-locked in
`test_analysis_streaming.py`, `test_preprocessing_streaming.py`, and
`test_preprocessing_ui_redesign.py`. **Distinct from preview** — preview still
takes over via its own end-of-render `isolate_layers`, untouched.

### 3c. Persist the preprocessed images  ·  DONE (2026-06-30)
The actual gap was **interactive-path signal wiring**, not the batch write (the
TODO's old "optional `save_cache`" diagnosis was stale — `batch_analysis.py`
already writes the preprocessed TIFFs unconditionally since `e12731c`). The
interactive persist machinery (`_widget.py::_persist_preprocessed_tiffs` + the
`preprocessing` branch of `_persist_active_experiment`) was fully written **and
test-locked** (`test_interactive_preprocessing_persists_tiffs`), but
`connect_signals()` wired `preprocessing_completed` only to `refresh()` with a
stale "nothing of its own to persist" comment — so a GUI preprocessing run wrote
nothing to disk. The unit test hid it by calling `_on_stage_persisted` directly,
never exercising the signal. **Fix:** connect `preprocessing_completed` →
`_on_stage_persisted("preprocessing")`; added
`test_preprocessing_completed_signal_persists_tiffs` driving the real signal as a
regression guard. Verified end-to-end: by the time the signal fires the streamed
arrays are filled in place in the data manager, so the persist reads real data.
Closely related to the backlog "Load processed `.ntfm` back into memory on
selection."

### 5. BISM as a selectable stress engine  ·  DONE (2026-06-30), superseded (2026-06-30)
~~Replace MSM with BISM~~ → originally **added BISM alongside MSM** behind a
**Stress Method** dropdown. **Superseded same day**: MSM was later ripped out
entirely (see below) — BISM is now the only stress engine, no dropdown.

### 6. Rip out MSM and BISM's MAP auto-λ  ·  DONE (2026-06-30)
**Both MSM and BISM's MAP machinery are fully removed** (user's call: MSM was
never coming back as a real option, and MAP was "too much fragile cleverness for
a knob the user can set by hand").

- **MSM gone**: `backend/msm.py`, `msm_numba_functions.py`, `mesh_generator.py`,
  and `_validation/benchmark_MSM/` deleted outright; `gmsh`/`solidspy` dropped
  from `pyproject.toml`. `widgets/msm_widget.py` → `widgets/stress_widget.py`
  (`MSMWidget`/`MSMController` → `StressWidget`/`StressController`), stripped of
  every MSM branch (`preview_mesh`, the mesh header glyph, the MSM half of every
  `use_bism` conditional — those calls are now unconditional BISM). The mesh-only
  `StressResult` fields (`nodes`/`elements`/`condition_number`/`residual`) are
  gone; `method` defaults to `"BISM"`. `MSMParameters` → `StressParameters`
  (mesh/material fields dropped); `MSMResult` alias gone, everything imports
  `StressResult` from `backend/stress.py` directly (which also now hosts
  `process_mask_data`, relocated out of `msm.py`). The Stress parameter-panel
  section collapsed to a flat list (`bism_regularization` + `max_stress`) — no
  more `WHEN`/`AND`-gated engine choice, since there's only one engine.
- **MAP gone**: `_estimate_lambda_map`, `noise_value_map`, `lam_method`/`use_map`
  threading, the `bism_lambda_method` field, and the **λ Method** dropdown +
  AND-gate sentinel are all deleted from `bism.py`/`parameter_dataclasses.py`/
  `_widget.py`. BISM always uses the fixed `bism_regularization` slider. The
  L-curve idea (`meth_Lambda==2`) is genuinely moot now, not just "moot per a
  note" — no auto-λ plumbing survives to extend.
- Test suite updated to match (`test_msm_analysis.py` deleted, `process_mask_data`
  test ported to `test_bism_stress.py`; MAP-specific tests in `test_bism_stress.py`
  deleted; `test_stress_ownership.py`/`test_workflow_shell.py`/
  `test_reload_on_selection.py` fixtures and dropdown tests updated/removed).
  548 passed.

### 7. visualization engine — napari live viewer + matplotlib export  ·  DONE (2026-06-30)
**Final decision (2026-06-30): the export renders with matplotlib; napari stays
the live interactive viewer only.** The original §7 goal was one renderer (napari
for both), and that was built and worked — but it forced a GL canvas, which on a
desktop pops a window, and going windowless needs a virtual display (xvfb /
subprocess), which is **Linux-only and not pip-installable**. For a tool shipping
to PLOS/JOSS users (mostly Mac/Windows) that's the wrong foundation. A
side-by-side also showed napari's "punch" was largely **additive blending
oversaturating the magnitude map** (arrow brightness summed into the colormap it's
meant to encode) — matplotlib, done well, is the cleaner *and* more faithful
publication artifact, plus vector-grade and fully portable. So the napari-export
detour (offscreen viewer → vendored movie-maker → xvfb subprocess) was torn out.
- **`backend/batch_visualizations.py`** (`BatchVisualizationSaver`) renders each
  product with matplotlib's **Agg** backend — windowless, no display, no GL, no
  xvfb — straight to **`.mp4`** (libx264/yuv420p via `imageio-ffmpeg`, which
  bundles ffmpeg on every OS; canvas padded to even dims). Same per-stage `save_*`
  surface, renders inline (no subprocess/flush). Products: displacement_map
  (viridis + white arrows), force_map (inferno + white arrows), force_cell_overlay
  (gray inverted cells + magnitude-coloured arrows), sigma_xx/yy/normal_stress
  (seismic). Sleek inline vertical colorbar mirrors the viewer's look.
- **Consistency via shared geometry, not a shared renderer.** Arrows come from the
  same `utilities/vector_field.py: build_frame_vectors`/`upscale_field` the live
  `VisualizationManager` uses (verified: same directions/scale as napari on the
  same field), with the same colormaps + contrast + arrow-scale convention. Only
  the final raster differs (Agg vs GL).
- **Mesh dropped** (user's call): FE-mesh GIF was an MSM-only diagnostic. Removed
  from `_run_config.py` + `_handle_visualization`. The **interactive** "Preview
  mesh" button (`msm_widget`/`_icons`/`_widget`) is untouched.
- **New dep:** `imageio-ffmpeg` (added to pyproject; portable, bundles ffmpeg).
  **No system deps** — xvfb is *not* required (the whole napari-export path that
  needed it was removed). `matplotlib` was already a dep.
- Test-locked in `test_batch_visualizations.py` (Agg → no display needed): shared
  vector math + every product writes an mp4 with right frame count / distinct
  frames / stress-component gating / single-frame inputs. Live-path streaming /
  run-config / batch suites green.
- **Arrow-colour default is white** on the magnitude maps (cleanest/most faithful
  per the comparison), magnitude-coloured on the gray cell overlay. The
  additive-glow look is reproducible in matplotlib (additive RGB compositing) if
  ever wanted — not the default.

### 8. Parallel batch workers  ·  L  ·  DONE (2026-07-01)
Batch config gains a **number-of-workers** spinbox (experiments-list toolbar,
1..`os.cpu_count()`, default 1 = today's unchanged behavior); positions process
in parallel, **top positions first** (FIFO submission order into the pool).
- **Workers compute, viewer follows one**, as decided: `num_workers > 1` runs
  each position headlessly (`sink=None`) on a real `ProcessPoolExecutor`
  (`spawn` context — forking a GUI process with live Qt/BLAS/OpenMP threads is
  a deadlock hazard). `start_parallel`/`poll_parallel_progress`
  (`backend/batch_analysis.py`) are the non-blocking pair a `QTimer` drives
  from `_widget.py`, so the GUI thread never freezes for the run's duration.
  The viewer follows the selected row, else the topmost folder, by reusing the
  existing "load `.ntfm` on selection" path once that position's worker
  reports done — no live cross-process frame streaming. Cancellation cancels
  only not-yet-started futures; in-flight workers finish naturally (no
  force-kill, no torn `.ntfm` writes).
- Manual row clicks during a parallel run go through the existing, unmodified
  selection path — clicking any row, finished or not, shows current disk
  truth ("scrub through existing data"), confirmed as the intended UX.
- `num_workers <= 1` (default) is the exact pre-existing synchronous/live-
  streaming path, verified byte-identical — zero regression risk for ordinary
  batch runs.
- Known, deliberately-scoped trade-offs: the per-stage progress bar doesn't
  fill frame-by-frame during a parallel run (only reconciles when the
  followed position completes or the run ends); no executor/timer teardown if
  the widget is closed mid-run (no existing teardown hook to mirror);
  `num_workers` is in-memory GUI state only, not persisted to the experiment-
  series file (unlike `disabled_stages`/`processed_root`).
- Tests: mocked-executor unit tests (`test_batch_parallel.py`) plus one real,
  unmocked `ProcessPoolExecutor` integration test
  (`test_batch_parallel_real_pool.py`) proving the actual multiprocessing
  round-trip (spawn, pickling, subprocess execution, result hand-off) works,
  not just the mocked simulation of it. 575 passed.

---

## Backlog (ranked, quick wins first)

### Adding rows to an empty list should preload the first row  ·  S  ·  DONE (2026-06-30)
When the `ExperimentsList` is empty and the user adds rows, the first added row
should be **preloaded/selected automatically** (rather than leaving the list with
no active selection). Saves a click and gives an active position for downstream
actions to target.

### Remove the export icon from the experiments list (now just a copy-file button)  ·  S  ·  DONE (2026-06-30)
The per-row Export button no longer does anything beyond copying the position's
OME-TIFF elsewhere, so it's redundant — remove the control **and its logic**:
- `widgets/_experiments_list.py`: the `export_btn` (lines ~233-243), the
  `_EXPORT_W` spacer column (`export` placeholder at ~992-994), the
  `export_requested` signals (row + list, ~191/303) and their re-emit
  (~1011), and the enable-on-done line (~277).
- `widgets/_widget.py`: the `export_requested.connect(...)` wiring (~565) and the
  `_on_export_experiment_data` handler (~1295).
- Drop the `"export"` entry from `stage_action_icon` if nothing else uses it, and
  any test that asserts `experiment_row_export_button`.

### Polish the colorbar legend  ·  S  ·  DONE (2026-06-30)
Spacing, label alignment, endpoint-number placement — the `viewer_colorbar.py`
knobs `COLORBAR_HEIGHT_FRACTION` / `LABEL_INSET_FRACTION` are the levers. Pure
visual tuning, no plumbing.

### Output results to `TFM_data/` next to the input, not `processed/`  ·  S/M  ·  DONE (2026-06-30)
TFM results should **not** land in a folder called `processed`. They should go
in a folder called `TFM_data` sitting **right next to the input data** — i.e. as
a sibling of the input's containing folder. When an output-folder variable is
set, the input folder structure is cloned into that output directory, and the
`TFM_data` folder lives where the input data *would* be inside that cloned tree.
Also **rename the artifact**: the single multi-series OME-TIFF holding all the
results should be called `TFM_results.ome.tif` — not `<experiment_name>.ome.tif`
(currently `batch_output.py::experiment_output_path`, line ~104).

### Apply-mask-on-save option in the batch config  ·  M  ·  DONE (2026-06-30)
Opt-in `apply_mask_on_save` flag: when a mask layer is present, **zero every map
pixel where the mask is background (label 0)** before writing the `.ntfm` —
`u_x, u_y, F_x, F_y` (and stress, if present) set to `0.0` wherever `mask == 0`.
The `mask` column records which pixels were zeroed (self-documenting).
- **Why.** Off-cell substrate signal is noise; zeroing it cleans the field and
  compresses ~8× (measured `Ctrl/pos_00`, 8.3% on-cell: 177 MB → 20.5 MB). Long
  runs of exact zeros crush under snappy/zstd.
- **Scope.** Opt-in, default **off** (deliberately lossy — background values are
  discarded irreversibly; only the `mask` column survives). No-op without a mask.
- **Where.** Batch write step (`backend/batch_analysis.py` →
  `ntfm.results_to_ntfm`), a pre-write array op. Interactive/preview unaffected.

### Preprocessing param-panel layout cleanup  ·  M  ·  DONE (2026-06-30)
Tidy the preprocessing widget (`napariTFM/widgets/preprocessing_widget.py`):
- **Shorten the double sliders** a bit (they're wider than they need to be).
- **Remove rolling-ball radius** entirely — both the front-end control and the
  back-end parameter/usage.
- **Regroup the params** into rows: 1 row Intensity, 1 row Cell Intensity, 1 row
  sigma + cell sigma side by side, 1 row registration method.
- **Reorder the input-file rows** to: bead stack (top), reference stack, cell
  stack, Masks. **Rename the layers** to `Beads`, `Reference`, `Cells`, `Masks`.

### Dedup preprocessed-TIFF persistence (batch vs. interactive)  ·  M  ·  DONE (2026-06-30)
The preprocessed-image save lives as **two independent implementations** that
only share the low-level `save_calibrated_tiff` helper:
`backend/batch_analysis.py::_execute_preprocessing` (lines ~724-742) and
`widgets/_widget.py::_persist_preprocessed_tiffs`. Parity is currently held by
hand — which is exactly how the §3c bug happened (one path was wired, the other
sat dead). Collapse the interactive path so it calls into the batch's
preprocessing-save orchestration rather than reimplementing it, so there's one
place that knows how a position's preprocessed TIFFs get written. Pure tidy-up,
no behaviour change.

### Load processed `.ntfm` back into memory on selection  ·  M  ·  DONE (2026-06-30)
Follow-up from the "stage runners weren't saving" fix: selecting an
already-processed experiment reads "done" from disk, but the viewer layers stay
empty until a stage re-runs. On selection, **load its `.ntfm` back into memory**
so the viewer shows the stored result.

### Make preview toggle-vs-one-shot legible in the icons  ·  M  ·  DONE (2026-06-30)
Preview is inconsistent across stages: some stages **toggle** preview (on/off,
persistent state) while others fire it as a **one-shot** action. The icons don't
distinguish the two, so the control's behavior isn't predictable from looking at
it. Make toggle-style previews **render as toggles** in the icon set (a
pressed/active state that reflects the on/off), distinct from the one-shot
(momentary action) icons — so the UI tells the user which kind of preview each
stage offers before they click.

### Progressive per-stage loading bar  ·  L (scope first)  ·  DONE (2026-06-30)
For both **live** mode and **batch** mode: each status circle/node should **fill
up progressively** as its stage runs (not just flip empty→done). One
implementation that serves both modes. **Investigate complexity first** —
driving a smooth per-stage fill needs intra-stage progress signals from the
pipeline (the sink currently emits stage-level start/finish, not fractional
progress), so scope what granularity is actually available before committing to
a design. Biggest unknown in the backlog — do not start without scoping.