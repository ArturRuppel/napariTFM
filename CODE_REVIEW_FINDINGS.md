# napariTFM Code Review Findings

_Date: 2026-07-02_

Thorough review focused on unwarranted complexity, refactor/simplification and dead
code, plus potential bugs — especially numerical/scientific mistakes.

## Summary

The scientific core (FTTC, BISM, metrics math) is largely correct and carefully done —
the FFT frequency ordering, Green's-function DC handling, Lanczos-filter ordering, and
BISM staggered-grid assembly all check out. The problems cluster in three areas:
**unguarded divisions that break on degenerate frames**, **error-swallowing that reports
failures as success**, and a **substantial amount of dead code** — most notably an
entire metrics module that the pipeline never runs.

---

## Numerical / scientific bugs

**1. Resume path silently changes displacement results (reproducibility bug). — DONE
(finding overstated; real residual fixed).**
`backend/batch_analysis.py:57-60` (`save_calibrated_tiff`) rescales each array to its
*own* min/max before casting to uint16, so the saved reference and the saved bead stack
no longer share an intensity scale. When preprocessing is skipped and displacement
resumes from `preprocessed_beads.tif`/`preprocessed_reference.tif` (`:711-716`),
Farneback optical flow runs on inconsistently-scaled images, whereas a fresh in-session
run uses the un-renormalized float arrays. Same inputs → different displacement field
depending on whether you resumed. Most important scientific issue: it silently
undermines reproducibility.

_Resolution:_ The stated mechanism does **not** hold. `DisplacementAnalyzer.calculate_flow`
(`backend/displacement_analysis.py:66-67`) runs both the reference and every bead frame
through `_normalize_for_optical_flow`, which renormalizes each image *independently* to
uint8 by its own min/max immediately before the Farneback call — so the reference/beads
"shared scale" is irrelevant, and Farneback is scale-invariant here. The genuine residual
was smaller: the fresh path fed displacement `float→uint8`, while the resume path fed
`float→uint16 (disk)→uint8`, so the extra uint16 round-trip could shift ≤1 uint8 level at
rare quantization boundaries — fresh and resume were not byte-identical. Fixed by making
`_handle_displacement_execution` **always** read the persisted preprocessed tiffs (fresh
run and stage-resume alike), collapsing the two into a single input path. Consistent with
the project-wide "calcs always read from disk" invariant.

**2. Unguarded divisions produce NaN/inf or crashes on constant/null frames. — DONE
(preprocessing guards added; viz portion is a false alarm).** None of
these guard the denominator:
- `backend/preprocessing.py:218` — `(processed - min_val)/(max_val - min_val)` when
  percentiles are equal (flat region).
- `backend/preprocessing.py:248-251` — `register_to_reference` normalizes by `(max-min)`.
- `backend/batch_analysis.py:58-59` — `save_calibrated_tiff` on a constant frame →
  undefined `uint16` cast.
- `utilities/visualization_manager.py:281, 408, 863` — arrow scaling `/ d_max`,
  `/ f_max`, `/ vmax`; a null field gives `d_max==0` → inf arrows *and*
  `contrast_limits=(0,0)` (`:292/:399/:899`), which napari rejects, throwing the whole
  visualization call.

_Resolution:_ The three preprocessing/writer divisions are real (a globally flat frame —
dead detector, all-black, saturated — gives `max == min` → silent NaN/inf). Guarded each:
`apply_intensity_scaling`, `register_to_reference`, and `save_calibrated_tiff` now map a
zero-span array to zeros instead of dividing. The **visualization portion is a false
alarm**: `d_max`/`f_max`/`vmax`/`max_stress` are user display parameters (defaults 1.0 /
500.0; UI min 0.1), not derived from the field — every assignment in
`visualization_manager.py` reads them straight from the params (`vmax = vis['v_max']`,
`max_stress = cfg['max_stress']`), so a null field does not zero them and the division is
unreachable in normal use. Left unchanged.

**3. GCV can return a negative λ. — DONE (real; fixed at source).**
`backend/fttc.py:417` runs `optimize.fmin`
unconstrained on `_gcvfun`, which depends on λ only via `λ**2` (even function). The
solver can land on a negative value; `_perform_tfm` squares it so the force is fine, but
the value is stored as `regularization` and `utilities/parameter_manager.py:49-52`
(`get_ui_parameter`) then calls `math.log10(negative)` → error/NaN in the UI. Wrap the
result in `abs()`.

_Resolution:_ Confirmed end to end — `_find_regularization` → `find_optimal_regularization`
→ `_handle_gcv_results` stores the raw value via `set_parameter('regularization', ...)`
(`fttc_widget.py:285`), which then hits `math.log10(value)` in `get_ui_parameter` and
raises `ValueError: math domain error` on a negative λ. Because `_gcvfun` is even in λ,
the negative root and its magnitude are the identical regularizer, so `abs()` is exact
and lossless. Applied at the source (`_find_regularization` return) so both the UI-display
path and the auto-GCV batch path (which uses `regularization ** 2`) see a non-negative λ.

**4. Displacement and stress parameters are never validated in production. — DONE
(removed the dead validators).**
`validate_displacement_parameters` and `validate_stress_parameters`
(`backend/parameter_validation.py:30,107`) are called only from tests — unlike the
FTTC/preprocessing validators which run in the backend. So an out-of-range `pyr_scale` or
a negative `bism_regularization` reaches the solver unchecked. `validate_stress_parameters`
also never checks `bism_regularization` at all.

_Resolution:_ The UI already constrains these parameters (spinbox min/max on every
displacement and stress control), so the validators duplicated a guarantee the widgets
already enforce — dead code with no production caller. Rather than wire them in, deleted
both functions and their test-only assertions. Kept `validate_preprocessing_parameters`
and `validate_fttc_parameters`, which *are* invoked in the backend.

---

## Correctness / data-integrity bugs

**5. Failed folders are reported as "done" (silent data loss). — DONE (real; fixed).**
Every batch stage
handler (`backend/batch_analysis.py:696-700, 707-724, 731-745, 752-773`) catches all
exceptions, prints, and returns `None`. A real failure cascades to all-`None` results;
`_write_experiment_ntfm` then hits the `written is None` branch (`:860-862`), prints, and
returns *without raising*. The deliberate re-raise at `:853-859` only covers a write-time
error, not upstream failures — so `process_folder` completes and the row shows green with
nothing on disk. Verified end to end.

_Resolution:_ Real bug. The headline mechanism is the swallowed exception: an enabled
stage that raises was caught, printed, and turned into `None`, indistinguishable from a
disabled stage or a legitimate skip. Each stage handler now records the caught exception
in a per-folder `self._stage_failures` list (via `_record_stage_failure`); the exception
is still caught so any stages that *did* succeed are written to the `.ntfm` first, then
`process_folder` raises a `RuntimeError` summarizing the failed stage(s). That propagates
through `process_folder_task` / `_run_position_headless` to the existing `"error"` status,
so the row goes red instead of green. Left the *graceful* `return None` skips untouched
(disabled stage; stress with no external mask; resume-from-`.ntfm` when the upstream field
isn't present) — those are deliberate, test-blessed behavior, not failures. New regression
test `test_stage_exception_reports_error_but_keeps_partial` asserts both halves: the folder
raises, and the successful upstream stage is still persisted.

**6. A cleared mask silently resurrects on re-write. — DONE (mask now always clobbered).**
`utilities/ntfm.py:681-684` treats
an all-zero `mask` in the new frame as "absent" and restores the old mask from disk
(`merge_existing=True` is the default write path). There is no way to clear a mask by
re-running a stage. The same "all-NaN means absent" logic applies to stages (`:669-678`,
at least documented there). Note: this is partly a behavioral choice (merge-preserving),
so decide the intended semantics before changing.

_Note:_ the review's `.ntfm` references are stale — the container migrated to a
multi-series OME-TIFF (`TFM_results.ome.tif`), but `merge_tidy_preserving` still runs on
the in-memory tidy df on every default write, so the behavior was live.

_Resolution (semantics decided):_ an intentionally-empty mask is not a real use case, so
an all-zero mask should always win. Removed the mask-preservation branch entirely — unlike
the measure stages (all-NaN unambiguously = "not computed this run"), an all-zero mask only
ever means "no mask supplied". The real mask is re-read from the input folder on every run
(`_load_mask`), so the branch only ever fired when the mask file had been removed before a
re-write — exactly the case where the stored mask should clear, not resurrect. New
regression test `test_merge_cleared_mask_does_not_resurrect`. (Measure-stage preservation
is untouched — that behavior is correct and desired.)

**7. `.ntfm` metadata drifts on round-trip. — DONE (false alarm; verified empirically).**
`utilities/ntfm.py:403` writes config via
`json.dumps(..., default=str)`, so numpy scalars / `Path` / tuples degrade to strings and
never recover their type on read. And the mask is written `uint16` (`:414`) but read back
`int64` (`:517`) — labels >65535 or signed sentinels wrap silently, which contradicts the
"lossless round-trip" docstring.

_Resolution:_ Neither sub-claim bites in practice; proved by a write→read round-trip test.
(a) `config` is `asdict(UnifiedParameters)`, and that dataclass is entirely native types
(float/int/bool/str/`Optional[float]`) — no numpy scalars, `Path`, or tuples — so every
field serializes natively and `default=str` never fires for the config. Measured drift is
NONE (`gel_height=None` → `None` included). `default=str` is only a safety net for
provenance fields (`git_provenance`/`inputs`/`labels`) that are never re-parsed by type,
and the sole resume consumer (`batch_analysis.py`) discards the metadata anyway.
(b) The mask read cast is a *widening* uint16→int64, which is lossless — multi-label masks
(0, 2, 3, 1000, 65535) round-trip exactly, and int64 matches the tidy-df's in-memory mask
convention (so it's consistent, not asymmetric-by-accident). The only loss is a label
>65535 wrapping *on write* (70000→4464), but TFM masks are external cell/monolayer
footprints — binary or a handful of non-negative labels — so that ceiling is physically
unreachable and there are no signed sentinels. No code change.

**8. Clicking the already-active experiment row drops in-memory overlays. — DONE
(real; fixed with a same-path guard).**
`widgets/_experiments_list.py:998-1014` (`set_active`) has no same-path guard, so a repeat
click re-emits `active_changed`, which calls `data_manager.clear_generated_results()` and
reloads from disk — streamed force/stress overlays vanish until re-triggered.

_Resolution:_ Real bug, and already tacitly known — `_on_row_stage_clicked` in `_widget.py`
guards its own `set_active` call with `if path != self._active_experiment` precisely to
dodge this cost. Centralized the guard in `set_active`: it now computes
`active_changed = path != self._active` and only re-emits `active_changed` on a genuine
change. Selection styling / delete-button state still update unconditionally, so multi-select
gestures (Ctrl/Shift) that leave the active row unchanged refresh correctly *without*
triggering the heavy clear-and-reload — a bonus fix, since building a delete-selection over
the active experiment previously wiped its overlays too. Two regression tests added
(`test_reselecting_active_row_does_not_reemit`, `test_selection_change_without_active_change_does_not_reemit`).

---

## Dead code

**9. The entire metrics feature is dead.** `backend/metrics_calculator.py` (strain
energy, moment tensor, polarization, eigenvalues) is imported nowhere in the app;
`backend/batch_analysis.py` contains no reference to `calculate_metrics` or
`metrics_parameters` even though `widgets/_run_config.py:28-32,60-66` sets
`"calculate_metrics": True` and a `metrics_parameters` block in every run config. The
config knobs and the whole module can go (or the feature needs wiring up). Note
`calculate_polarization` also uses `np.linalg.eigvals` on the non-symmetric moment
tensor, which can return complex eigenvalues — a latent bug, but moot while dead.

**10. Error-handler registration is inert.** `utilities/error_handling.py:42,60`
(`ErrorHandlingMixin._error_handlers`) is never populated — no register method exists — so
`handle_error` only logs; every `create_error` in `visualization_manager.py` is
effectively a swallow. Relatedly, `utilities/data_manager.py:139` (`mark_artifact_error`)
has no production caller, so `ArtifactState.error` is never set and the error display at
`widgets/_stage_data_status.py:66` is unreachable.

**11. Other confirmed-unused symbols:**
- `widgets/_widget.py`: the whole `WHEN`/`AND` conditional-visibility machinery
  (~50 lines, `_conditional_rows`/`_register_conditional`/`_apply_conditional_visibility`)
  is never triggered by any `PARAMETER_SECTIONS` entry; `_loaded_stage_data` is
  write-only; the `STAGE_DATA_ARTIFACTS` constant (`:45-67`) is copied then fully
  overwritten (`:657-669`); redundant local `from qtpy.QtCore import QTimer` (`:498`);
  `_load_stage_results` return value is ignored by all callers.
- `widgets/_experiments_list.py`: `ExperimentRow.selected` signal and
  `output_dir_changed` are emitted but connected only in tests — dead wiring.
- `utilities/data_manager.py`: `remove_change_callback` unused; `ArtifactState.dirty`
  write-only.
- `utilities/visualization_manager.py`: `get_displacement_statistics` duplicates the
  widget's own copy and has no caller; `PreviewConfig` fields are write-only;
  `_clear_layers` key check (`:79-81`) can never match (display names vs snake_case keys).
- `backend/fttc.py`: `from fttc_numba_functions import *` (`:32`) is redundant after the
  explicit import on `:31`; `_gcv_blockdiag` returns `G`/`reg_param` and takes a `plot`
  arg that are all unused; `i_bound_size`/`j_bound_size` are always 0; the `(x, y)`
  coordinate grid built in `_perform_tfm:331-338` is never consumed downstream
  (`calculate_force_field` uses only `result[1]`).

---

## Complexity / simplification

- **`utilities/parameter_manager.py:110-120, 167-177`** — the
  `young_modulus`/`regularization` special-case branches in
  `get_all_parameters`/`get_category_parameters` are no-ops identical to the `else`; only
  the `gel_height` None→0 case matters. The comments describe conversions that don't
  happen.
- **`backend/preprocessing.py:69-78`** — the `is_cell`/`else` branches both set
  `processed = image`; only the parameter selection differs.
- **`backend/batch_analysis.py`** — the reference image is preprocessed twice (once inside
  `preprocess_stack`, again at `:946`); stress has double-nested exception swallowing
  (`_execute_stress_analysis` and `_handle_stress_execution`); preprocessing progress is
  0-based (`:942,955`) while every other stage logs 1-based frames.
- **`utilities/ntfm.py`** — three parallel tidy↔array scatter implementations (`_scatter`
  `:240`, `ntfm_tidy_to_series_channels` `:374`, plus `read_ntfm`'s inverse) that must be
  kept in lockstep; four overlapping stage/column mapping dicts; `git_provenance` spawns
  two `git` subprocesses on *every* `write_ntfm` (redundant within a batch).

---

## Suggested priority

1. #1 (resume reproducibility), #5 (failures reported as success), #9 (delete dead
   metrics module + config).
2. Division guards (#2), GCV `abs()` (#3), production validation (#4).
3. Dead-code removals (#10, #11) — low risk.
4. `.ntfm` fidelity (#6, #7) — decide intended semantics first (#6 especially).
5. Simplifications as cleanup.
