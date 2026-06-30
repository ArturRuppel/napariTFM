# napariTFM — Open Worklist

> Accomplished items are pruned (see git history for completed UI-redesign
> slices P0–P8, the UI-Coherence roadmap, the BISM port, the 2026-06-29
> batch/cancel/colorbar/sink work, the 2026-06-30 unified-logging (§1) +
> per-position Export-to-CSV (§2) work, and the §3 streaming-follows-active-
> position work). What remains below is **open work only**, ranked
> easy-wins-first.

---

## Ranked open work (2026-06-29)

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

### 5. BISM as a selectable stress engine  ·  DONE (2026-06-30)
~~Replace MSM with BISM~~ → **added BISM alongside MSM** (user's call: keep MSM
intact, switch via a **Stress Method** dropdown). The validated BISM core moved
to `napariTFM/backend/bism.py`; a unified `backend/stress.py::StressResult` both
engines return (`MSMResult` is now an alias). `params.stress_method` ("MSM"/"BISM")
dispatches in the batch (`_run_bism_stress`) and interactive (`MSMController`)
runners; BISM skips the mesh phase. **FE mesh kept** (it's MSM's, still works).
**Param panel is now engine-aware** (2026-06-30): a `WHEN(param, value)` sentinel
in the Stress spec (`_widget.py`) swaps the whole knob set off the Method
dropdown — MSM shows its mesh/material params, BISM shows its one real knob, the
Bayesian regularization λ (a 10^x slider, `bism_regularization`, previously
hardcoded at 1e-6 and now threaded through). `free_bc` stays fixed (validated True
for masked monolayers). Built so retiring MSM later = deleting its WHEN block.
Deferred follow-up: persist BISM's per-pixel **uncertainty** into the `.ntfm`
columns + a viewer layer. Note: BISM still leaves the meshing path in place, so #7's
"mesh doesn't map onto a napari layer" tension is **not** dissolved — revisit if
BISM becomes the default.

### 6. BISM automatic λ selection (L-curve / MAP)  ·  M
The original `BISM.m` offers **three** ways to set the regularization λ
(`meth_Lambda`): MAP auto-estimation, L-curve auto-estimation, or a fixed value.
The port (§5) shipped only the **fixed-value** path and hands the user a manual
λ slider (`bism_regularization`) — so the user now *has* to pick λ by hand, when
the original could pick it for them. Port the **L-curve** selector (sweep λ over
a log range, choose the point of maximal curvature in residual-norm ↔ prior-norm
space) and/or the **MAP** fixed-point iteration, in `backend/bism.py`, wrapping
whichever solver runs (full *and* masked paths). Surface as a **λ method**
dropdown (Fixed / L-curve / MAP) via the existing `WHEN` sentinel — hide the λ
slider when not Fixed. Threads a method enum through
UnifiedParameters/MSMParameters/parameter_manager, same plumbing the λ slider
took. Worth doing if λ turns out finicky across datasets. (Sibling deferred BISM
item: persist per-pixel **uncertainty** — see §5 — which is what the original's
`noise_value` knob feeds.)

### 7. napari-native visualization engine  ·  L  (after #5)
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

### Dedup preprocessed-TIFF persistence (batch vs. interactive)
The preprocessed-image save lives as **two independent implementations** that
only share the low-level `save_calibrated_tiff` helper:
`backend/batch_analysis.py::_execute_preprocessing` (lines ~724-742) and
`widgets/_widget.py::_persist_preprocessed_tiffs`. Parity is currently held by
hand — which is exactly how the §3c bug happened (one path was wired, the other
sat dead). Collapse the interactive path so it calls into the batch's
preprocessing-save orchestration rather than reimplementing it, so there's one
place that knows how a position's preprocessed TIFFs get written. Pure tidy-up,
no behaviour change.

### Remove the export icon from the experiments list (now just a copy-file button)
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

### Preprocessing param-panel layout cleanup
Tidy the preprocessing widget (`napariTFM/widgets/preprocessing_widget.py`):
- **Shorten the double sliders** a bit (they're wider than they need to be).
- **Remove rolling-ball radius** entirely — both the front-end control and the
  back-end parameter/usage.
- **Regroup the params** into rows: 1 row Intensity, 1 row Cell Intensity, 1 row
  sigma + cell sigma side by side, 1 row registration method.
- **Reorder the input-file rows** to: bead stack (top), reference stack, cell
  stack, Masks. **Rename the layers** to `Beads`, `Reference`, `Cells`, `Masks`.

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

### Make preview toggle-vs-one-shot legible in the icons
    Preview is inconsistent across stages: some stages **toggle** preview (on/off,
    persistent state) while others fire it as a **one-shot** action. The icons don't
    distinguish the two, so the control's behavior isn't predictable from looking at
    it. Make toggle-style previews **render as toggles** in the icon set (a
    pressed/active state that reflects the on/off), distinct from the one-shot
    (momentary action) icons — so the UI tells the user which kind of preview each
    stage offers before they click.