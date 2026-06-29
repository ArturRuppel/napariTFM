# napariTFM — Open Worklist

> Accomplished items have been pruned (see git history / prior commits for the
> completed UI-redesign slices P0–P8, the UI-Coherence roadmap, and the BISM
> port). What remains below is **open work only**.

---

## New requests (2026-06-29)

### 1. Cancel button on all stages ✅ DONE
Every stage (preprocessing, displacement, FTTC, MSM — and any long-running
"Run all") should expose a **Cancel** control that aborts the in-flight
computation cleanly. Cancel scaffolding exists in some widgets already; make it
present, wired, and consistent across **all** stages.

### 2. Colorbar in the preview ✅ DONE
The viewer colorbar should also render in **preview** mode, not just the
committed/result view — so previews show the same scale legend the final
visualization does. (`napariTFM/utilities/viewer_colorbar.py`.)
- Root cause: the preview path hid the legend. Each stage preview adds the
  colorbar, then `VisualizationManager.isolate_layers` (and each widget's own
  visibility loop) set every layer except the two stage layers to
  `visible=False`, hiding the colorbar — while the committed/result path never
  isolates, so its colorbar survived.
- Fix: `ViewerColorbarManager` now exposes `layer_names` / `is_colorbar_layer`;
  `isolate_layers` keeps the active legend layers visible, and the
  displacement/force/stress preview visibility loops skip colorbar layers.

### 3. Min/max scale labels on the colorbar ✅ DONE
Add the scale's **min and max numbers** to the colorbar: min at the **bottom**,
max at the **top** of the scale bar, each **centered horizontally** on the bar.
Just the two endpoints — no intermediate ticks.
- `ViewerColorbarManager.show_for_layer` now adds two text-only points layers,
  `"… Colorbar Max"` / `"… Colorbar Min"`, each sitting just right of the bar
  and flush with one end: max top-anchored (`upper_left`) to the bar's top, min
  bottom-anchored (`lower_left`) to the bar's bottom. Two layers are needed
  because napari's text anchor is per-layer. The range is read from the
  reference layer's `contrast_limits` (data min/max fallback), so no call site
  changed. `format_scale_value` keeps the numbers compact across µm / Pa /
  signed mN·m⁻¹ ranges. Both layers are tracked in `layer_names`, so
  `isolate_layers` keeps them visible in preview.

### 4. Replace the visualization engine with a napari-native one
Swap the bespoke renderer for a **napari-native** path built on
[`napari-movie-maker`](/home/aruppel/Projects/napari-movie-maker), so the viewer
and any exported figures/movies share one rendering path.
- **Headless renderer — already DONE in napari-movie-maker**
  (`export_movie_headless` / `offscreen_viewer` / `ensure_offscreen_qt`).
  **Deployment note:** napari renders via OpenGL, so the process must run under a
  GL-capable display — `xvfb-run -a python …` (bare `QT_QPA_PLATFORM=offscreen`
  has no GL context and aborts). Add `xvfb` to the runtime/CI.
- **Remaining (napariTFM side).** Map each current `save_*` product to an
  `export_movie_headless` call: a `configure(viewer)` that adds the right layers
  (image + vectors/quiver + labels/mesh) with matching colormaps, then sweep the
  time axis. Confirm the **quiver** and **FE-mesh** overlays have napari-layer
  equivalents (the parts least likely to map 1:1 from the matplotlib renderer).
- Retires `backend/batch_analysis_visualizations.py` (`BatchVisualizationSaver`,
  matplotlib + `imageio.mimsave` per stage).

### 5. Batch mode = sequential in-napari run, not a console mode
Batch should **not** be a separate mode that runs in the console. It should
simply **walk every step inside napari** — as if each stage button were pressed
sequentially. Make "run all" drive the live napari pipeline end to end rather
than spawning/processing outside the viewer.
- *Note:* earlier slices already removed the run-in-console radios and added a
  "Run all" that walks the rail; verify whether any console/out-of-viewer path
  still survives and finish converting it to a pure in-napari sequential run.

### 6. Collapsible file list ✅ DONE
The experiments/file list should be **collapsible to a single row**, so it can be
folded away to reclaim vertical space when not actively browsing experiments.
- `ExperimentsList` header gained a caret toggle (`experiments_collapse_button`).
  Everything below the header now lives in one `self._body` container, so
  `set_collapsed`/`toggle_collapsed` fold the whole list with a single
  `setVisible`. While collapsed the header shows a compact count summary
  (`experiments_header_summary`, kept current by `_update_meta`) so the single
  remaining row still says how many experiments are hidden.

---

## Bugs

### Status circles in the file list are not working properly
The per-row status dots in the experiments/file list don't reflect the true
stage state. Audit the status pipeline (`populated_measures` /
`_experiment_stage_status` → the row dots) and fix so each row's dots correctly
show done/ready/off per stage.

### Stage runners are not saving output files
Running a stage does not write its output to disk. The runners must persist
their results (the `.ntfm` measures / cache) so downstream stages and the
status dots can see them. Likely couples with the status-circles bug above
(no output → dots can never read "done").

---

## Backlog — data / output

### Apply-mask-on-save option in the batch config
Add an opt-in flag (e.g. `apply_mask_on_save`) that, when a mask layer is present
for an experiment, **zeroes every map pixel where the mask is background
(label 0)** before writing the `.ntfm` — `u_x, u_y, F_x, F_y` (and stress, if
present) set to `0.0` wherever `mask == 0`. The `mask` column records which
pixels were zeroed, so the result is self-documenting.
- **Why.** Off-cell substrate displacement/traction is noise; zeroing it cleans
  the field and compresses enormously. Measured on `Ctrl/pos_00` (8.3% on-cell):
  unmasked `.ntfm` **177 MB** → masked **20.5 MB** (~8×). Long runs of exact
  zeros crush under snappy/zstd — no special codec needed.
- **Scope.** Opt-in, default **off** (deliberately *lossy*: background values are
  discarded irreversibly; only the `mask` column survives to say so). No-op when
  no mask is present.
- **Where.** Batch write step (`backend/batch_analysis.py` → `ntfm.results_to_ntfm`),
  a pre-write array op on the result fields. Interactive/preview path unaffected.
