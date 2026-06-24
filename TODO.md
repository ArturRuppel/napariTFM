# napariTFM UI Redesign (viridis) — Active Worklist

> **Status (2026-06-24):** branch `ui-redesign` merged to `master`; Slices 1–5
> done & test-locked (290 tests). The owner's remarks below **supersede the
> original Slice 6/7 outlines** in
> `docs/superpowers/plans/2026-06-24-experiments-list-at-top.md` and redefine the
> remaining work. Build sliced & test-locked (TDD → commit), polish last. The
> existing **UI-Coherence roadmap** (now complete) and the **batch/data backlog**
> are preserved further down for historical context.

The shift: the top experiments list (Slice 5) grows into the **single config =
metadata table**; the old per-stage *enable* toggles and the whole
`BatchAnalysisWidget` scaffolding (folder mgmt, analysis/viz checkboxes, run-mode
radios, per-stage progress bars) are retired. Stages become **mandatory**
(MSM excepted) and **auto-skip when their output already exists**, reporting via a
single **global status label**.

## Resolved decisions

- **D1 — MSM optionality → keep its existing on/off glyph.** MSM is already the
  only stage carrying an enable/power glyph (Slice 4), and that glyph *is* the
  disable path. **No walk-back of Slice 4** — the MSM toggle stays; only the
  *other* batch checkboxes below go (P1). All non-MSM stages are mandatory.
- **D2 — "Discover" = folders, not columns.** We do **not** autodiscover columns
  from filenames. Discovery is **folder-presence** only (CellFlow-style:
  `src/cellflow/aggregate_quantification/catalog.py:212` `discover_catalog_entries`
  — find folders containing the named input files). Column metadata is **not**
  parsed. The flow is **two add steps**: (1) discover folders; (2) a second
  *commit* step adds those folders to the list **together with the column config**
  set up above them — the column name/value pairs are **copied to every row** of
  the batch being added.

## Priority phases (execution order = dependency + risk)

### P0 — Config = the metadata table  ✅ DONE *(commits 65b88e1, c1f2212, b9a0179, f23de7c, 25d13d6)*
Grew the Slice-5 `ExperimentsList` into the one top config table, tied to folders.
Landed test-locked across five sub-slices (suite 311 green):
- ✅ **P0.1** `discover_experiment_folders()` — folder-presence discovery (D2;
  mirrors CellFlow, recursive, beads+reference required, cells optional).
- ✅ **P0.2** per-row records carry `{input_files, columns}`; `add_folders()`
  copies the column config onto each new row; `experiment_records()` query.
- ✅ **P0.3** column-config header: input file-name fields + **"+ Add column"**
  name/value pairs; `input_file_config()` / `column_config()`.
- ✅ **P0.4** two-step **Discover → Commit** ("Add to list" button + staging
  label); commit copies the column config to **every** added row.
- ✅ **P0.5** config bridge: the shell feeds `experiment_records()` into the
  batch widget (`set_experiment_records`), which drives folders/input files and
  carries each row's columns as `experiment_metadata` in the saved config.
- **Deferred (cosmetic):** rendering the extra columns *in the rows* (polish);
  removing the now-vestigial folder-management UI rides with **P4**.

### P0b — One save, one config  ✅ DONE *(commit 1f5334a)*
Merged **Save Parameters** (params-only) and **Save Config** (full config) into a
single config save at the Project section (**Save/Load Config**), delegating to
`batch_analysis_widget.save_config_to_yaml`/`load_config_from_yaml` — it carries
params **plus** the P0 table (`experiment_metadata`). The params-only handlers and
the batch widget's duplicate config buttons/dialogs are removed.

### P1 — Strip superseded scaffolding  ✅ DONE *(commits b80a71d, 706841a, bea9847)*
Independent of P0 and safe to start first; **keep the batch run trigger until P4.**
All removals here are behavior-preserving (each control already defaults to its
"on"/"napari-console" state, hardcoded into `_generate_config`). Landed as P1a
(radios), P1b (analysis/viz checkboxes + bead overlay, incl. backend), P1c
(progress bars); full suite 295 green.
- **Folder management** (list/add/clear): `batch_analysis_widget.py:201-245`,
  `_add_folder:498`, `_clear_folders:573` — **deferred to P0/P4**: the kept batch
  run reads `folder_list` as its only source, and P0's table is its replacement.
  Cannot remove without breaking the run we keep until P4.
- ✅ **Run-in-napari vs run-in-console** radios — removed (P1a); batch runs
  in-process. `_run_in_new_console`/`_launch_console` gone too.
- ✅ **Analysis-step checkboxes** — removed (P1b); `_generate_config` emits
  constant all-mandatory `analysis_steps`.
- ✅ **Visualization checkboxes incl. bead overlay** — removed (P1b);
  `_generate_config` emits constant viz; bead overlay deleted from UI **and**
  backend (`batch_analysis.py` viz_map/branch + `save_bead_overlay`).
- ✅ **Per-stage progress bars** — removed (P1c) from all 4 stage widgets + batch;
  textual `status_label` kept as interim until P2's global label.

### P2 — Global status label  *(replaces the progress bars)*
One panel-level **text** status (no bar); stages report run/skip/done into it
(e.g. `displacement — skipped (output present)`). Pairs with the P1 bar removal.

### P3 — Auto-skip-when-output-present  *(the "don't recompute" path for mandatory stages)*
Each mandatory stage checks for its existing output and skips, reporting via P2.
(MSM is exempt — it's gated by its own on/off glyph per D1, not auto-skip.)
Refine `_widget.py:_experiment_stage_status` from the Slice-5 *coarse* `.ntfm`
check to **per-output truth**. (This is the data half of the old Slice 6.)

### P4 — Run-all walks the rail  *(was Slice 6; live)*
"Run all" iterates the P0 table through the stages via
`BatchAnalysis.process_all_folders()`, with live mini-rail updates; **drives runs
from the config table** and retires the old batch run button
(`batch_analysis_widget.py:240` `run_analysis_btn`). Couples with P1 (radios gone)
+ P3 (skip logic). Likely collapses `BatchAnalysisWidget` entirely.

### P5 — Per-stage viz-toggle glyph  *(replaces the removed viz checkboxes)*
Each stage header gains a single **viz-toggle glyph** that toggles that stage's
preview/overlay layer (one per stage, "everyone gets a viz toggle").

### P6 — Free-text px-size & frame-length inputs
Convert `_project_section.py:18-55` `pixel_size` / `frame_interval`
`QDoubleSpinBox`es to free input fields.

### P7 — Rationalize + glyph-ify remaining buttons  *(polish)*
Audit remaining action buttons, dedupe redundant ones, replace text buttons with
glyphs from the `_icons.py` SVG set. Targets: `gcv_btn` (`fttc_widget.py:401`),
`preview_mesh_btn` (`msm_widget.py:488`), and the stage-header
run/preview/files/params controls (`_stage_section.py:81-112`).

### P8 — Align status dots with the pills  *(polish; deferred from Slice 5)*
Align the mini-rail / spine status dots with the header pills (the known
dot-spacing/alignment nit).

### P9 — Aggregate → `.iris`  *(was Slice 7, minus labels)*
Labels now come from the P0 metadata table, so this is just the backend +
footer entry: a short `.iris` schema spec, then `aggregate_to_iris()` over the
`.ntfm` series. ROADMAP §5.

---

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
