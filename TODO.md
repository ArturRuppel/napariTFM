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

### 6. BISM automatic λ selection — MAP DONE (2026-06-30); L-curve still open  ·  S
**DECISION (2026-06-30): rip out the whole MAP machinery — it sucks.** The
non-monotonic fixed point, the unstable separatrix, the scale-dependent λ₀, the
log-grid root-find band-aid: too much fragile cleverness for a knob the user can
set by hand. Tear out `_estimate_lambda_map`, the `noise_value_map` plumbing, the
`bism_lambda_method` enum + **λ Method** dropdown + AND-gate sentinel, and the
associated tests (`test_map_does_not_collapse_to_zero_field` et al.). Go back to
**fixed λ only** for BISM; if we ever want auto-λ again, do L-curve fresh, not MAP.
This makes the L-curve note below moot unless we revisit auto-selection from scratch.

The original `BISM.m` offers **three** ways to set the regularization λ
(`meth_Lambda`): MAP auto-estimation, L-curve auto-estimation, or a fixed value.
The §5 port shipped only **fixed-value**, forcing the user to pick λ by hand.
**MAP is now ported** (`backend/bism.py::_estimate_lambda_map`) — the MAP/Jeffreys
condition from `meth_Lambda==1` (the same fixed point `λ = l²·s_noise²/s_prior²`,
the reference's full parameter/observation denominators + Jeffreys `+2`, no
effective-parameters trace term), adapted to the masked counts (`2·ncell`, `ninf`).
It also yields the MAP **noise estimate** (`meta["noise_value_map"]`) — the dormant
`noise_value` slot that feeds §5's uncertainty maps. A `bism_lambda_method` enum
("Fixed"/"MAP") threads through UnifiedParameters/MSMParameters/parameter_manager/
msm_widget; the UI got a **λ Method** dropdown via a new `AND` WHEN-conjunction
sentinel that hides the λ slider unless Fixed.

**Robustness fix (2026-06-30):** the first cut copied BISM.m's *bare fixed-point
iteration* from a hardcoded λ₀=1e-3 verbatim, and it returned **zero stress
everywhere** on real-scale data. Root cause: `g(λ)=l²s²/s0²` is non-monotonic with
an *unstable* separatrix; λ₀=1e-3 is calibrated to the paper's reference (l≈2, tiny
tractions, natural λ≈1e-5) and only converges there. On napariTFM-scale data
(smaller l, smooth fields, natural λ≈1e-6–1e-7) 1e-3 lands on the runaway side and
λ explodes (→1e25), over-regularizing σ→0. Replaced the iteration with a
**stable-fixed-point root-find**: scan g(λ) on a log grid, bracket the largest
`+→−` crossing (attracting), refine with Brent. Scale-independent — recovers the
correct field on the benchmark (R² 0.02→1.00) while still reproducing the
reference's λ=8.88e-6. On genuinely noise-free data (no MAP optimum) it returns
`None` and the caller falls back to the fixed λ with a warning. Test-locked in
`test_bism_stress.py` incl. a `test_map_does_not_collapse_to_zero_field` regression
(smooth fittable field where the old iteration zeroed out); UI gating in
`test_workflow_shell.py`.
**Still open — L-curve:** port `meth_Lambda==2` (sweep λ over a log range, pick
the point of maximal curvature in residual-norm ↔ prior-norm space) as a third
dropdown option ("L-curve"). The plumbing (enum, dropdown, AND-gate, masked-count
adaptation) is all in place now — it's just the selector function + one dropdown
entry. Worth it only if MAP turns out finicky on some datasets; MAP is the more
coherent default since it does double duty with the §5 uncertainty follow-up.

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

### Output results to `TFM_data/` next to the input, not `processed/`
TFM results should **not** land in a folder called `processed`. They should go
in a folder called `TFM_data` sitting **right next to the input data** — i.e. as
a sibling of the input's containing folder. When an output-folder variable is
set, the input folder structure is cloned into that output directory, and the
`TFM_data` folder lives where the input data *would* be inside that cloned tree.
Also **rename the artifact**: the single multi-series OME-TIFF holding all the
results should be called `TFM_results.ome.tif` — not `<experiment_name>.ome.tif`
(currently `batch_output.py::experiment_output_path`, line ~104).

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