# napariTFM — Open Worklist

> Accomplished items are pruned (see git history for completed UI-redesign
> slices P0–P8, the UI-Coherence roadmap, the BISM port, the 2026-06-29
> batch/cancel/colorbar/sink work, and the 2026-06-30 unified-logging (§1) +
> per-position Export-to-CSV (§2) work). What remains below is **open work
> only**, ranked easy-wins-first.

---

## Ranked open work (2026-06-29)

### 3. Streaming follows the active position  ·  S
While the batch/live sink streams, the viewer + experiments-list selection should
**track the position currently being processed**. Add an `experiment_started`
hook on the sink (`utilities/viewer_sink.py` / `backend/pipeline_sink.py`) that
drives the `ExperimentsList` selection. Pairs with #4.

### 4. Per-stage layer isolation during streaming  ·  M
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
per-stage active-layer set and apply it on `stage_started`. Build with #3 (both
are "the streaming sink takes over the UI to show what it's doing").

### 5. Replace MSM with BISM  ·  M–L
Swap Monolayer Stress Microscopy for the validated **BISM** port
(`napariTFM/_validation/benchmark_MSM/bism.py`; no material params, gives a
stress field + uncertainty). BISM needs **no FE mesh**, which **dissolves the
meshing/mesh-rendering path** — the part of #7 least likely to map onto a napari
layer. Wire BISM into the production stress stage; confirm downstream consumers
of the stress products still get what they expect. Retires the FE-mesh overlay
rendering. **Do this before #6.**

### 6. napari-native visualization engine  ·  L  (after #5)
Swap the bespoke renderer for a **napari-native** path built on
[`napari-movie-maker`](/home/aruppel/Projects/napari-movie-maker), so viewer and
exported figures/movies share one rendering path.
- Headless renderer already DONE in napari-movie-maker (`export_movie_headless` /
  `offscreen_viewer` / `ensure_offscreen_qt`). **Deployment:** napari renders via
  OpenGL — run under a GL-capable display (`xvfb-run -a python …`; bare
  `QT_QPA_PLATFORM=offscreen` aborts). Add `xvfb` to runtime/CI.
- napariTFM side: map each `save_*` product to an `export_movie_headless` call —
  a `configure(viewer)` that adds image + vectors/quiver layers with matching
  colormaps, then sweeps the time axis. After #5 the FE-mesh case is gone, so
  every overlay maps cleanly to a napari layer.
- Retires `backend/batch_analysis_visualizations.py` (`BatchVisualizationSaver`,
  matplotlib + `imageio.mimsave` per stage).

### 7. Parallel batch workers  ·  L
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
