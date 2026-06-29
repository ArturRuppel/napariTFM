# napariTFM — Open Worklist

> Accomplished items are pruned (see git history for completed UI-redesign
> slices P0–P8, the UI-Coherence roadmap, the BISM port, and the 2026-06-29
> batch/cancel/colorbar/sink work). What remains below is **open work only**,
> ranked easy-wins-first.

---

## Ranked open work (2026-06-29)

### 1. Fix stage-resume caching: displacement present, yet force can't compute  ·  S–M  (bug)
Repro: a position shows **displacement done** (data available), but **Force
Analysis fails to compute**. Root cause is the stage-resume / cache path — force
reconstructs the displacement field via
`_resume_field_from_ntfm(tfm_folder, folder, "displacement_field")`
(`backend/batch_analysis.py`), which returns `None` when the experiment's
`.ntfm` is missing, lacks the displacement columns, or is **all-NaN** (the writer
emits every measure column as NaN when a stage didn't run). So even though
displacement is "available" in memory/UI, the on-disk container force reads from
doesn't actually carry it. **Audit when/where the `.ntfm` is written vs. read** so
a computed displacement field is reliably persisted *before* force runs, and the
path/name force-resume expects matches what the displacement write produced.
Also **persist the preprocessed images for consistency.** Today the preprocessed
bead/reference TIFFs (`preprocessed_beads.tif` / `preprocessed_reference.tif`)
are only an *optional* `save_cache`, while displacement/force/stress always
persist to `.ntfm`. Give preprocessing the **same persistence guarantee** so
every stage's output is reliably on disk for the next stage to resume from.
Closely related to the backlog "Load processed `.ntfm` back into memory on
selection."

### 2. Unify logging (live = batch)  ·  XS
Batch mode prints its log to the console; live/interactive mode does not. Route
the interactive path through the **same logger** so live mode prints the **same
messages batch does** (decision: match batch exactly). Keep writing full detail
to the run log file as today.

### 3. Per-position "Export to CSV" button  ·  S–M
Each position **row** in the `ExperimentsList` (next to the status dots) gets an
**Export to CSV** button that writes that position's processed `.ntfm` out as a
**full per-pixel field dump** — every pixel's `u_x, u_y, F_x, F_y` (and `stress`,
`mask` when present) per frame. The `.ntfm` is a parquet container, so this is a
read-and-flatten-to-CSV op; mind the large file sizes (stream/chunk the write,
warn if no `.ntfm` exists yet). No-op / disabled when the position isn't
processed.

### 4. Streaming follows the active position  ·  S
While the batch/live sink streams, the viewer + experiments-list selection should
**track the position currently being processed**. Add an `experiment_started`
hook on the sink (`utilities/viewer_sink.py` / `backend/pipeline_sink.py`) that
drives the `ExperimentsList` selection. Pairs with #5.

### 5. Per-stage layer isolation during streaming  ·  M
During batch streaming **and** live "Run all", the sink should **take over layer
visibility** and show only the layers relevant to the stage in flight; restore
prior visibility when the run ends. **Distinct from preview** — preview must
*never* take control of layer state.
- preprocessing → **beads + ref only**, additive blending in the two colors;
  then **cell only**.
- displacement → displacement layers only.
- force → force layers only.
- stress → stress layers only.
Reuse the existing `VisualizationManager.isolate_layers` infrastructure; define a
per-stage active-layer set and apply it on `stage_started`. Build with #4 (both
are "the streaming sink takes over the UI to show what it's doing").

### 6. Replace MSM with BISM  ·  M–L
Swap Monolayer Stress Microscopy for the validated **BISM** port
(`napariTFM/_validation/benchmark_MSM/bism.py`; no material params, gives a
stress field + uncertainty). BISM needs **no FE mesh**, which **dissolves the
meshing/mesh-rendering path** — the part of #7 least likely to map onto a napari
layer. Wire BISM into the production stress stage; confirm downstream consumers
of the stress products still get what they expect. Retires the FE-mesh overlay
rendering. **Do this before #7.**

### 7. napari-native visualization engine  ·  L  (after #6)
Swap the bespoke renderer for a **napari-native** path built on
[`napari-movie-maker`](/home/aruppel/Projects/napari-movie-maker), so viewer and
exported figures/movies share one rendering path.
- Headless renderer already DONE in napari-movie-maker (`export_movie_headless` /
  `offscreen_viewer` / `ensure_offscreen_qt`). **Deployment:** napari renders via
  OpenGL — run under a GL-capable display (`xvfb-run -a python …`; bare
  `QT_QPA_PLATFORM=offscreen` aborts). Add `xvfb` to runtime/CI.
- napariTFM side: map each `save_*` product to an `export_movie_headless` call —
  a `configure(viewer)` that adds image + vectors/quiver layers with matching
  colormaps, then sweeps the time axis. After #6 the FE-mesh case is gone, so
  every overlay maps cleanly to a napari layer.
- Retires `backend/batch_analysis_visualizations.py` (`BatchVisualizationSaver`,
  matplotlib + `imageio.mimsave` per stage).

### 8. Parallel batch workers  ·  L
Batch config gains a **number-of-workers** parameter; positions are processed in
parallel, **top positions first** (process in list order).
- Decision: **workers compute, viewer follows one.** Workers process positions in
  parallel headless (no per-worker sink); the viewer streams/shows only the
  top/selected position as results complete. Keeps §5's "viewer is an optional
  sink" model intact — parallelism lives in the headless compute layer, the
  single viewer never tries to show N positions at once.
- Note the tension with the current synchronous-on-GUI-thread batch; this is the
  most architectural item — design the worker pool + result hand-off before
  coding.

---

## Backlog

### Polish colorbar + progressive per-stage loading bar
Two threads of viewer-legend / progress polish:
- **Polish the colorbar** legend (spacing, label alignment, endpoint-number
  placement — the `viewer_colorbar.py` knobs `COLORBAR_HEIGHT_FRACTION` /
  `LABEL_INSET_FRACTION` are the levers).
- **Progressive loading bar** for both **live** mode and **batch** mode: each
  status circle/node should **fill up progressively** as its stage runs (not just
  flip empty→done). One implementation that serves both modes.
  **Investigate complexity first** — driving a smooth per-stage fill needs
  intra-stage progress signals from the pipeline (the sink currently emits
  stage-level start/finish, not fractional progress), so scope what granularity is
  actually available before committing to a design.

### Adding rows to an empty list should preload the first row
When the `ExperimentsList` is empty and the user adds rows, the first added row
should be **preloaded/selected automatically** (rather than leaving the list with
no active selection). Saves a click and gives an active position for downstream
actions to target.

### Load processed `.ntfm` back into memory on selection
Follow-up from the "stage runners weren't saving" fix: selecting an
already-processed experiment reads "done" from disk, but the viewer layers stay
empty until a stage re-runs. On selection, **load its `.ntfm` back into memory**
so the viewer shows the stored result.

### Apply-mask-on-save option in the batch config
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
