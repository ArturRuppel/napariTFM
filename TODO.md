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