# napariTFM Code Review Findings — Round 2

_Date: 2026-07-03_

Second-pass review, deliberately hunting for what the 2026-07-02 pass
(`CODE_REVIEW_FINDINGS.md`) missed: new bugs, new dead code, and the
**architecture-level** simplifications the first pass didn't reach for. Every
headline item was verified independently (F1 reproduced numerically; every
"dead" symbol grep-confirmed to have no production caller; every bug traced
end-to-end). Baseline: **623 tests pass**.

Confidence tags: **CONFIRMED** = mechanism verified/traced. **PLAUSIBLE** =
logic certain, production reachability depends on input.

---

## Executive summary

Three tiers of work, most valuable first:

1. **New correctness bugs** — a silent wrong-forces numerical bug on odd grids
   (F1), and two failure paths that report success (stress-stage swallow, stale
   downstream resurrection on interactive re-run).
2. **~400+ lines of new dead code** — an entire unused BISM formulation (~110
   lines), a whole dead `ParameterManager` method, write-only state plumbing,
   test-only UI helpers, and a config knob nothing reads.
3. **Three architecture levers** — (a) the tidy-table format is a mandatory
   round-trip intermediary nothing consumes; (b) the four stage Widget/Controller
   pairs are ~75% copy-paste that has drifted into bugs; (c) the four batch
   stage handlers want to be data. These are the real maintainability wins.

---

## Tier 1 — New correctness bugs

### B-1. FFT frequencies are wrong for odd-sized grids → silently wrong forces. CONFIRMED. HIGH.
`backend/fttc.py:599-602` (`_calculate_fourier_modes`) hand-rolls the FFT
frequency ladder:
```python
kx_vec = 2.*np.pi/i_max/forcemap_pixel_size * np.append(
    np.arange(0, (i_max // 2)), np.arange(-i_max // 2, 0))
```
This is **exact for even `i_max`/`j_max` but wrong for odd** (reproduced):

| n | code | `fftfreq(n)·n` |
|---|------|----------------|
| 5 | `[0, 1, -3, -2, -1]` | `[0, 1, 2, -2, -1]` |
| 7 | `[0, 1, 2, -4, -3, -2, -1]` | `[0, 1, 2, 3, -3, -2, -1]` |

The array length stays `n`, so there's **no crash** — the bin at index `n//2`
just gets the wrong wavenumber. The Green's function (`kabs`) and the Lanczos
filter both consume these values, so the deconvolution applies the wrong
transfer function at the Nyquist-adjacent bin → subtly wrong tractions, and it
corrupts `_svd_block` → wrong GCV λ. This **contradicts the prior review's "FFT
frequency ordering … checks out"**, which only held for even grids.

Reachability: `calculate_traction` always passes explicit `input_width/height`
(`:279-280`) = the displacement grid shape after downscale, i.e.
`image_dim // downscale_factor`. Any odd result triggers it (e.g. `998//4=249`,
or **any** odd image dimension at `downscale_factor=1`). Nothing upstream forces
even dimensions in the production path.

_Fix:_ `2*np.pi/forcemap_pixel_size * np.fft.fftfreq(i_max)` (and same for
`j_max`) — algebraically identical to the current expression on even grids,
correct on odd. One-liner, exact.

### B-2. Stress-stage compute failures still report "done" — the finding-#5 fix doesn't cover stress. CONFIRMED. HIGH.
`backend/batch_analysis.py:1310-1312` — `_execute_stress_analysis` wraps its
whole body in `try/except Exception: return None`, which swallows even its own
explicit `raise RuntimeError("Stress calculation failed")` (`:1302`). So
`_handle_stress_execution` (`:809`) receives an ordinary `None`, never calls
`_record_stage_failure`, and the folder is reported green.

`_execute_force_analysis` (`:1188-1206`) and `_execute_displacement_analysis`
have **no** such catch-all — they let the `RuntimeError` propagate, which is why
finding #5's fix works for them. Stress is the one stage that reintroduces the
exact bug #5 claimed to close. (The prior review saw this code — it noted
"stress has double-nested exception swallowing" under *simplification* — but
never connected that the inner swallow defeats its own #5 fix.) The regression
test `test_stage_exception_reports_error_but_keeps_partial` injects into
`_execute_force_analysis`, a stage without the swallow, so it never caught this.

_Fix:_ delete the inner `except Exception: return None` (let stress propagate
like force/displacement), or narrow it so `RuntimeError` re-raises.

### B-3. Interactive upstream re-run resurrects a stale downstream stage on disk. CONFIRMED. MEDIUM. (interactive-only; needs a design decision)
`utilities/data_manager.py:168-171` promises invalidation prevents a stale
downstream result being "written into a .ntfm alongside a freshly recomputed
upstream stage." The on-disk merge defeats that promise:

1. Run displacement → persist (disk: `disp_v1`).
2. Run force → persist (disk: `disp_v1 + force_v1`).
3. Change a param, re-run displacement (v2). `set_displacement_results`
   invalidates `force_results` → `None` in memory. Persist writes
   `disp_v2, force=None` via `results_to_ntfm(merge_existing=True)`
   (`ntfm_writer.py:153`, default). `merge_tidy_preserving`
   (`ntfm.py:669-678`) sees force absent-in-new / present-in-old and
   **restores `force_v1`** — which was computed from `disp_v1`.

Disk now holds `disp_v2` + physically inconsistent `force_v1`, and on the next
row reselect `_load_stage_results` reloads that stale force next to the new
displacement. `merge_tidy_preserving` has no notion of the `_DOWNSTREAM`
dependency chain, so it can't know the preserved stage was invalidated. The
batch path is unaffected (it recomputes downstream in order and writes once).

_Tension:_ the merge's preserve-behavior is *wanted* for a force-only resume
(preserve displacement). The fix must distinguish "upstream changed → clear
downstream" from "downstream-only write → preserve upstream" — e.g. the
interactive persist of an upstream stage writes with `merge_existing=False`, or
the merge mirrors `_DOWNSTREAM`. **Decide semantics before changing.**

### B-4. Preprocessing progress bar jumps backward mid-stage. CONFIRMED. LOW (cosmetic).
`_execute_preprocessing` announces `num_frames = bead_stack.shape[0]`
(`batch_analysis.py:974`) but then emits three overlapping 0-based frame streams
(beads `0..N-1`, reference at index `0`, cells `0..M-1`). Both sinks compute
`fraction = (frame+1)/num_frames`, so the bar climbs to 1.0 on the last bead,
snaps back to `1/N` for the reference, then refills for cells (and can exceed
1.0 if cells outnumber beads). Prior review noted only the cosmetic "0-based vs
1-based" detail, not the backward jump. _Fix:_ a single monotonic counter over
the total beads+reference+cells work.

### B-5. Partial-NaN frames silently corrupt optical flow. PLAUSIBLE. LOW.
`displacement_analysis.py:94-99` — `image.max()-image.min()` is `NaN` if *any*
pixel is NaN, so the `<= 1e-8` guard is skipped and `(…)/NaN*255 → NaN →
.astype(uint8)` = garbage. `validate_displacement_image` only rejects
**all**-NaN (`:157`), so a few-NaN frame passes and feeds junk to Farneback.
_Fix:_ `np.nan_to_num` before normalize, or an any-NaN guard.

### B-6. Stress run never freezes the UI. CONFIRMED. LOW (UX).
`widgets/stress_widget.py` — `StressController.start_analysis` never calls
`freeze_ui()`, yet its callbacks call `unfreeze_ui()`. Every other stage freezes
on run (preprocessing `:100`, displacement `:128`, fttc `:57`). During a live
BISM run the Run/Preview actions stay enabled, so the user can re-trigger or
change params mid-stream. A symptom of the widget duplication (Tier 3b).

---

## Tier 2 — New dead code (all grep-verified: no production caller)

| # | Symbol | Location | Note |
|---|--------|----------|------|
| D-1 | **Unmasked BISM path** — `compute_bism_stress` `mask=None` branch (`:177-236`) + `_build_A`/`_build_B`/`_interp_to_grid` + the whole `return_uncertainty`/`noise_value` block + `free_bc`/`alpha_xy`/`alpha_bc` params | `backend/bism.py` | ~110 lines. Production (`:415`) and the only test both always pass a non-None mask → `mask=None` never runs. The masked solver hardcodes its own `alpha_xy=alpha_bc=1e3`. |
| D-2 | `ParameterManager.get_category_parameters` (whole method + `category_mappings`) | `utilities/parameter_manager.py:124-179` | Test-only. `category_mappings` is a 4th drifting copy of the stage→param mapping. |
| D-3 | `ParameterManager.load_from_file` | `utilities/parameter_manager.py:188-201` | Test-only (real loads go through `_widget._apply_parameters`). |
| D-4 | `ArtifactState.source` + every `source=` arg threaded through ~10 setters | `utilities/data_manager.py:16,129,207` | Written, never read anywhere. |
| D-5 | `save_cache` config knob | `widgets/_run_config.py:40,79` | Serialized into every run config; only referenced in a `batch_analysis.py:650` docstring. TIFFs write unconditionally. |
| D-6 | `NullSink` | `backend/pipeline_sink.py:68-75` | Test-only; `_emit` already short-circuits on `sink is None`. |
| D-7 | `update_preprocessing_visualization` | `utilities/visualization_manager.py:1004-1054` | ~50 lines, test-only. Stale pre-streaming path. |
| D-8 | `STATUS_COLORS`, `STATUS_GLYPHS` | `widgets/_ui_style.py:45,53` | Test-only (live coloring is `EXPERIMENT_STATUS_COLORS` + `_stage_spine`). |
| D-9 | `muted_stage_accent`, `danger_text_style` | `widgets/_ui_style.py:120,302` | Test-only (production uses `muted_accent(stage_accent(...))` inline). |
| D-10 | `DataArtifactSpec.on_view` / `on_action` fields | `widgets/_stage_data_status.py:12-13` | No spec sets them; an adjacent comment even says so. |
| D-11 | `"files"` (magnifier) icon body | `widgets/_icons.py:16` | Defined + tested, never rendered. |
| D-12 | `worker.running` flag | `widgets/stress_widget.py:80,248` | Written True/False, never read by the worker loop — the "cancel flag" does nothing. |

Micro-dead-code (LOW): `fttc.py:585-593` NaN-fallback branch in
`_interp_vec2grid` is unreachable (values pre-filtered); `fttc.py:408` `np.copy`
never mutated; `fttc.py:418` `np.max([minGi, 0])` is a no-op (`argmin ≥ 0`);
`_record_stage_failure`'s `hasattr` guard (`batch_analysis.py:332`) is
dead-defensive. Dead **parameters** (leftovers from the old `.npy` design):
`tfm_folder` in all three `_execute_*_analysis`; `folder` in
`_resume_field_from_ntfm`; `preprocessed_data` in `_handle_displacement_execution`
(reassigned from disk before use).

---

## Tier 3 — Architecture levers (the real maintainability wins)

### A-1. The tidy-table format is a mandatory round-trip nothing consumes. (biggest lever)
**Finding.** The canonical on-disk form is already a dense multi-series
OME-TIFF of `(T,C,Y,X)` arrays. Yet the code makes the tidy long-format
DataFrame a **mandatory intermediary on every read and write**:
- **Write:** arrays → `arrays_to_tidy()` (a full `T·Y·X`-row DataFrame) →
  `write_ntfm` immediately **scatters it back** to `(T,C,Y,X)` arrays for
  `tifffile`. Array → DataFrame → array → disk.
- **Read:** disk → arrays → `arrays_to_tidy()` → and **every** caller
  (`_widget.py:1394-1395`, `batch_analysis.py:830-831`,
  `_resume_field_from_ntfm`) immediately calls `tidy_to_arrays()` to get arrays
  back. Disk → array → DataFrame → array.
- The merge (`merge_tidy_preserving`) is an `O(T·Y·X)` relational left-join to
  express an `O(#stages)` decision.
- This is the source of **three near-identical scatter/gather implementations**
  that must stay in lockstep (`tidy_to_arrays._scatter`,
  `ntfm_tidy_to_series_channels`, and the reshape path in `read_ntfm`).

I traced every production caller: **not one consumes the DataFrame *as* a
DataFrame.** The CSV exporter that originally justified it has been deleted; the
§5 aggregator that would consume it isn't built. Right now the tidy layer is
pure overhead on the hot path, and B-3's bug hides inside the merge it forces.

**Recommendation** (a second architectural opinion concurred strongly): make the
container read/write **and** the merge **array-native** — operate on a
`dict[stage → ndarray]` plus a presence set; the merge becomes
`{**old, **new}` with a shape check. Keep `arrays_to_tidy` / `tidy_to_arrays` as
a thin **export adapter** in one module, called lazily only where a DataFrame is
actually wanted (future CSV/aggregator), guarded by one round-trip property
test. The contract to preserve for the future is the **named-stage array schema**
(names, axis order, dtypes, coordinate conventions) — *not* the DataFrame.
Delete `ntfm_tidy_to_series_channels` (the third scatter). Watch-outs when going
array-native: keep the NaN-vs-absent convention explicit; carry physical
coordinates (pixel size / dt) in container metadata, not reconstructed ad hoc;
and note the array path *removes* a latent pandas int→float NaN-promotion class.

Heuristic (worth adopting): _pay early for **contracts** (schemas, on-disk
formats, invariants), never for **conversions** (cheap to write when needed).
The tidy table is a conversion masquerading as a contract — and the tell that it
was premature is already here: the one consumer that justified it was deleted,
and the representation outlived it._

### A-2. The four stage Widget/Controller pairs are ~75% copy-paste — and the drift is now bugs.
**Finding.** Stages 2/3/4 (displacement/force/stress) are 8 classes where the
Widgets are ~1:1 delegating shells (empty identical `_setup_ui` ×3, same signal
wiring, `_update_ui_state` computing a couple of `has_X` flags). The Controllers
duplicate the same begin-stream / per-frame / worker-creation scaffolding,
differing only in literal stage names and which param fields they read — the
rendering layer already solved this with a single `_VECTOR_FIELD_CONFIG`, but
the controllers never followed. The copy-paste has **drifted into bugs**: B-6
(stress forgot `freeze_ui`), three different `cancel()` semantics (one uses
`terminate()`, one a write-only flag = no-op cancel — D-12), preview on a
background thread in one stage but synchronously on the GUI thread in two.

**Recommendation.** Fix the drift bugs *in place first* (small, individually
revertable), then hoist the run→freeze→progress→complete/fail→unfreeze +
one-blessed-cancel lifecycle into the base as a **non-overridable template
method** so the invariants become structurally unforgeable. Then collapse the
shell Widgets and merge the 2/3 controllers into one
`VectorStageController(StageSpec)` (mirroring `_VECTOR_FIELD_CONFIG`); stress may
stay a thin subclass for its mask input + extra action rather than an
`if kind == 'stress'` swamp. Preprocessing stays bespoke (real body UI — the
split earns its keep there). Add one parameterized drift-regression test
(freeze-on-run / unfreeze-on-terminal / cancel-actually-cancels) as the guard the
copy-paste era never had. Also fold the three copy-pasted preview layer-reorder
blocks (`displacement:79-107`, `fttc:214-243`, `stress:180-215`) into one
`VisualizationManager.bring_layers_to_front(names)` helper.

### A-3. The four batch stage handlers want to be data.
`_handle_{preprocessing,displacement,force,stress}_execution` +
`_execute_*_analysis` + `_log_*_progress` are three repeated layers per stage,
differing only in labels/keys/units. A stage is really a tuple of
`(enabled_key, param_builder, compute_fn, stage_name, unit_label)` — a
4-row table + one `_run_stage` driver would collapse ~200 lines and make B-2
(the stress swallow) structurally impossible, since there'd be one drain path.
`_handle_visualization` already uses a `viz_map` dict, so the data-driven form is
idiomatic here. (The sequential-vs-parallel *run* paths are genuinely different
concerns — leave those split.)

### A-4. Parameter plumbing restates the same fields ~5 times.
`UnifiedParameters` re-declares every field of the four per-stage dataclasses,
and the four `to_*_parameters` methods hand-copy each field — so adding a
parameter touches ~5 places, and defaults already drift (`frame_interval` is `1`
in sub-dataclasses, `1.0` in `Unified`). Collapse the four constructors to
field-name projection: `SubCls(**{f.name: getattr(self, f.name) for f in
fields(SubCls)})`. Also: the `young_modulus`/`regularization`/`gel_height`
special-case branches in `get_all_parameters`/`get_ui_parameter` are no-ops
identical to the `else` (only `gel_height` None→0 in `get_parameter` is real),
and their comments describe conversions that happen elsewhere — actively
misleading.

---

## Smaller simplifications

- `displacement_analysis.py:131-143` `downscale_flow` is an O(H·W) Python double
  loop; it's a pure block-mean →
  `flow[:nh*f,:nw*f].reshape(nh,f,nw,f,2).mean(axis=(1,3))`, ~100× faster, same
  result. (MEDIUM perf — runs on every displacement calc, default downscale 4.)
- `parameter_validation.py:41,44-50` — `validate_fttc_parameters` gates the
  whole force computation on visualization-only params (`force_arrow_scale`,
  `f_max`) and enforces `regularization > 0` even when `auto_gcv=True` (where
  `regularization` is unused). Split compute-critical from viz checks; skip the
  reg check under auto-GCV.
- Three near-identical `physical_scale` dicts
  (`displacement_analysis.py:213`, `fttc.py:100`, `bism.py:365`) differ only in
  a unit-name string — one shared helper.
- Mask resize logic is duplicated verbatim between
  `batch_analysis.py:1261-1273` and `stress.py:73-85`, and the mask is resized
  twice per stress folder (once for BISM, once for the stored container).
- `metrics_calculator.py` (still dead, deferred): when wired up, fix
  `calculate_polarization` — `np.linalg.eigvals` on the non-symmetric moment
  tensor can return complex, and `np.max`/`np.min` then compare lexicographically
  and silently return nonsense (use `eigvalsh` on the symmetric part); and the
  moment tensor is taken about the origin, not the force centroid.

---

## Implementation status (2026-07-03)

Landed on `claude/codebase-analysis-refactor-m8do2m`, each commit test-green:

- **Tier 1 bugs** — B-1 (odd-grid FFT), B-2 (stress error-swallow, + regression
  test), B-4 (progress monotonicity), B-5 (NaN optical-flow guard), B-6 (stress
  freeze-UI). ✅
- **Tier 2 dead code** — D-1 (unmasked BISM path), D-2/D-3 (dead ParameterManager
  methods), D-4 (`source` plumbing), D-5 (`save_cache`), D-6 (`NullSink`), D-7,
  D-8/D-9, D-10, D-11 (`files` icon), D-12 (`worker.running`). ✅ Deferred: the
  dead params (`tfm_folder`/`folder`/`preprocessed_data`) and the fttc GCV
  micro-cleanups — low value, fttc numerically sensitive.
- **A-1** — container read/write **and** merge made array-native;
  `write_series_ntfm`/`read_series_ntfm`/`merge_arrays`; tidy demoted to a lazy
  adapter; the third scatter impl and the tidy merge removed. ✅
- **A-3** — one `_guard_stage` + one `_log_stage_progress` collapse the four
  batch handler triplets. ✅
- **A-4** — `UnifiedParameters` projects per-stage subsets by field name; no-op
  ParameterManager branches removed. ✅
- **A-2** — the triplicated preview layer-management extracted to
  `VisualizationManager.bring_layers_to_front` (tested). ✅ **Remaining (sequenced
  follow-up):** collapse the three result Widget/Controller pairs into one
  parameterized `VectorStageController` and hoist the
  run→freeze→progress→complete/fail→**one blessed cancel** lifecycle into the
  base as a non-overridable template method — killing the residual drift
  (divergent cancel semantics; displacement/stress preview running synchronously
  on the GUI thread while force uses a `thread_worker`). This touches Qt worker
  teardown across four controllers, and the suite leans on fakes there, so it
  wants its own reviewed change with manual in-app verification rather than a
  blind headless refactor. B-6 (the highest-impact drift bug) and D-12 are
  already fixed.

- **B-3** (stale downstream resurrection) — **left for a semantics decision**: the
  merge's preserve-behaviour is wanted for a force-only resume but wrong for an
  interactive upstream re-run. Fix once the intended rule is chosen (interactive
  upstream persist writes `merge_existing=False`, or the merge mirrors
  `_DOWNSTREAM`).

## Suggested priority

1. **B-1** (odd-grid FFT), **B-2** (stress swallow) — small, exact, high-value
   bug fixes.
2. **Tier 2 dead code** — low-risk deletions (~400+ lines), especially D-1
   (unmasked BISM), D-2/D-3 (dead ParameterManager methods), D-4 (`source`),
   D-5 (`save_cache`).
3. **B-3** (stale downstream) and **B-4** (progress) — after deciding B-3's
   semantics.
4. **A-1** (tidy → array-native) — the biggest structural simplification; retires
   two of the three scatter impls and makes B-3 tractable.
5. **A-2 / A-3 / A-4** — de-duplicate the stage widgets, batch dispatch, and
   parameter plumbing (drift-killing).
