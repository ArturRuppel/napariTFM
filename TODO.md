# napariTFM UI Coherence — Roadmap  ✅ COMPLETE

> **Status (2026-06-24): all four steps below are implemented, integrated, and
> test-locked.** The inner scroll areas + 360px width are gone (Step 1);
> `StageSection` is rebuilt on `CollapsibleSection` with a glyph-pill header and
> the `_ActionStateSync` proxy machinery retired (Steps 2+3); and the
> `section_grid` vocabulary backs `WorkflowParameterPanel` + the Project px/dt
> controls (Step 4). See ROADMAP §3 for the commit list and the locking tests.
> The step-by-step text below is kept for historical context only.

The CellFlow → napariTFM UI port is **macro-complete** (workflow shell, theming,
status dots, labeled sliders) but **meso/micro-incoherent**: the new shell is a
CellFlow-shaped frame wrapped around the old per-tab stage widgets, which still
carry their own scroll areas, fixed widths, and duplicated action controls. This
roadmap closes that visual/structural gap.

**Orthogonal (not tracked here):** state coherence — one parameter system, one
refresh path, config round-trip — is handled by the existing
`docs/superpowers/plans/2026-05-29-tier4-state-architecture.md`. It touches
*state*, not *layout*, and can land independently.

Execution order below reflects real dependencies, not the order the gaps were
found. Steps 2 and 3 are **coupled** — the section-primitive decision changes
what the control-exposure refactor touches — so they are planned and executed as
one keystone slice.

---

## Step 1 — Remove inner scroll areas + hardcoded width  *(first slice)*

Each stage widget (`preprocessing`, `displacement`, `fttc`, `msm`) builds its
**own** `QScrollArea` with `setFixedWidth(360)` inside the shell's single
resizable scroll area — yielding scroll-in-scroll and a stage body locked to
360px while the rest of the panel reflows to the dock. Drop the inner scroll and
fixed width; let the shell's scroll own layout (CellFlow's model).

- Standalone, low-risk, highest visible payoff.
- Independent of the section-primitive decision — safe to do first.
- Touches: the 4 stage widgets' `_create_content_container`; update
  `tests/test_preprocessing_ui_redesign.py::test_preprocessing_widget_keeps_parameter_content_in_scroll_area`
  (now vestigial — params no longer live in the stage widget).

## Step 2 — Unify on one section primitive  *(keystone, with Step 3)*

`StageSection` (flat, always-visible body, only the param panel collapses) and
CellFlow's `CollapsibleSection` (whole-body collapse + accent inheritance for
nested sections) are different interaction models. Pick one: either port
`CollapsibleSection`'s accent-inheritance + whole-body collapse into
`StageSection`, or adopt `CollapsibleSection` outright. This is the decision the
rest hang off — resolve it before Step 3/4.

## Step 3 — Expose inner-widget controls; retire the proxy machinery

Stage widgets build their action buttons (`process_btn`, `preview_btn`,
`cancel_btn`) then `setVisible(False)`; the header buttons proxy-click them via
`_ActionStateSync`. Each control exists twice, kept in sync by an event-filter
shim. Refactor inner widgets to expose controls to the host (CellFlow aliases
*handlers* upward, builds each control once), retiring `_ActionStateSync` /
`action_targets`. Coupled to Step 2 — the section model determines where the
single control instance lives.

## Step 4 — Port the grid layout vocabulary; rebuild the param panel

CellFlow's `ui_style` has a dense grid family (`section_grid`, `block_grid`,
`add_section_pair_row`, `add_block_button_row`, sweep grids); napariTFM's
`_ui_style` is a ~5-helper subset, so each stage falls back to ad-hoc
QHBox/QVBox/QGroupBox and they don't look uniform. Port the grid helpers and
rebuild `WorkflowParameterPanel` (and the Project px/dt controls) on them so one
control idiom rules. Sits on top of Step 2 (where params live depends on the
section model) — do last.

---

# Backlog — Batch / data

## Apply-mask-on-save option in the batch config

Add an opt-in flag in the batch config (e.g. `apply_mask_on_save`) that, when a
mask layer is provided for an experiment, **zeroes every map pixel where the
mask is background (label 0)** before writing the `.ntfm` — i.e. `u_x, u_y,
F_x, F_y` (and stress, if present) are set to `0.0` wherever `mask == 0`. The
`mask` column already records which pixels were zeroed, so the result is
self-documenting.

- **Why.** Off-cell substrate displacement/traction is noise; zeroing it both
  cleans the field and compresses enormously. Measured on
  `Ctrl/pos_00` (8.3% on-cell): unmasked `.ntfm` **177 MB** → masked `.ntfm`
  **20.5 MB** (~8×, 12% of the legacy `.npy` pair). The win comes from the long
  run of exact zeros, which snappy/zstd crush — no special codec needed.
- **Scope/behavior.** Opt-in, default **off** (it is deliberately *lossy*:
  background values are discarded irreversibly — only the `mask` column survives
  to say they were zeroed). Conditional on a mask being present (masks are an
  external input per ROADMAP §2); no-op when absent.
- **Where.** Batch-only path (`backend/batch_analysis.py` write step → the
  `ntfm.results_to_ntfm` call); the masking is a pre-write array op on the
  result fields. The interactive/preview path is unaffected.
- **Note.** The compression win is from *masking*, not the container — masked
  `.npy` would shrink the same; `.ntfm` just has a native `mask` column that
  makes it tidy.

## Replace the batch output renderer with napari-movie-maker

Retire the bespoke batch movie renderer (`backend/batch_analysis_visualizations
.py` — `BatchVisualizationSaver`, currently matplotlib + `imageio.mimsave` per
stage: bead/displacement/force/force-cell-overlay/stress/mesh) and produce the
batch figures/movies through
[`napari-movie-maker`](/home/aruppel/Projects/napari-movie-maker) instead, so
the batch outputs match what the viewer shows and we maintain one rendering
path.

- **Headless renderer — DONE (in napari-movie-maker).** Added
  `export_movie_headless` / `offscreen_viewer` / `ensure_offscreen_qt`
  (`src/napari_movie_maker/_headless.py`, tested): stands up a background napari
  viewer, lets a `configure(viewer)` callback build the layers, sweeps an axis,
  and writes the movie — no GUI code. **Deployment requirement:** napari renders
  via OpenGL, so the batch process must run under a GL-capable display — i.e.
  `xvfb-run -a python run_batch.py` (xvfb sets `DISPLAY`; the viewer uses
  `show=True` to realize an invisible canvas). Bare `QT_QPA_PLATFORM=offscreen`
  has no GL context and aborts. Add `xvfb` to the batch runtime/CI.
- **Remaining (napariTFM side).** Map each current `save_*` product to an
  `export_movie_headless` call: a `configure` that adds the right layers
  (image + vectors/quiver + labels/mesh) with matching colormaps, then sweep the
  time axis. Confirm the quiver and FE-mesh overlays have napari-layer
  equivalents (the parts least likely to map 1:1 from the matplotlib renderer).
- **Why.** Single source of truth for visuals (viewer == batch), drops a large
  hand-rolled matplotlib renderer, and inherits movie-maker's dimension-sweep
  export.
- **Scope.** Batch-only; map each current `save_*` product to a movie-maker
  configuration (layers + colormaps + the swept axis). Confirm the
  vector/quiver and FE-mesh overlays have a napari-layer equivalent.
