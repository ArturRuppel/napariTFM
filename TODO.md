# napariTFM — Open Worklist

> Accomplished items are pruned (see git history for completed UI-redesign
> slices P0–P8, the UI-Coherence roadmap, the BISM port, the 2026-06-29
> batch/cancel/colorbar/sink work, the 2026-06-30 unified-logging (§1) +
> per-position Export-to-CSV (§2) work, the §3 streaming-follows-active-
> position work, and the §4 per-stage layer isolation during streaming). What
> remains below is **open work only**, ranked easy-wins-first.

---

## Ranked open work (2026-06-29)

### 5. BISM as a selectable stress engine  ·  DONE (2026-06-30)
~~Replace MSM with BISM~~ → **added BISM alongside MSM** (user's call: keep MSM
intact, switch via a **Stress Method** dropdown). The validated BISM core moved
to `napariTFM/backend/bism.py`; a unified `backend/stress.py::StressResult` both
engines return (`MSMResult` is now an alias). `params.stress_method` ("MSM"/"BISM")
dispatches in the batch (`_run_bism_stress`) and interactive (`MSMController`)
runners; BISM skips the mesh phase. **FE mesh kept** (it's MSM's, still works).
Deferred follow-ups: persist BISM's per-pixel **uncertainty** into the `.ntfm`
columns + a viewer layer; optionally retire the now-redundant FE-material params
from the UI. Note: BISM still leaves the meshing path in place, so #7's
"mesh doesn't map onto a napari layer" tension is **not** dissolved — revisit if
BISM becomes the default.

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
