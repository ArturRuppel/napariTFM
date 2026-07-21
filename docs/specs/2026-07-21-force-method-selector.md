# Force stage: explicit method selector

**Status:** steps 1–4 complete (2026-07-21); pending review + real-app check

## Progress

- **Step 1 — GCV restored.** `FTTC._svd_block/_gcvfun/_gcv_blockdiag/_find_regularization`,
  `auto_gcv` field, `regularization=None → GCV` in `calculate_traction`, module-level
  `find_gcv_regularization`. Regression test locks it.
- **Step 2 — `force_method` + dispatcher.** Field on both dataclasses, `infer_force_method`,
  `calculate_force_field` switches on the resolved method; `"auto"` reproduces legacy routing,
  explicit method overrides the numeric sentinels. Routing regression test.
- **Step 3 — UI.** Force `Method` dropdown + three `METHOD` blocks (Elastic net / FTTC + GCV /
  Bayesian L2), each surfacing only its own controls. Generalized `_refresh_method_visibility`
  and `_refresh_advanced_visibility` to `METHOD_DROPDOWNS`. New `"button"` control kind +
  `action_requested` signal; header crosshair removed, GCV-pick and Bayesian-freeze buttons live
  in-block. `auto_gcv` greys the manual slider + GCV button. `"auto"` resolves to the inferred
  method for display without mutating storage. Canonical method strings match the dropdown labels.
- **Step 4 — tests + compat.** Dataclass lockstep passes (generic); legacy-dict-without-force_method
  → `"auto"` test; stubs updated; full suite 709 passing.

Canonical decisions made during build: three methods (FTTC+Bayesian are *different solvers* — GCV
picks the Fourier λ and fills the slider, Bayesian evidence picks a real-space λ and cannot);
Bayesian block gets a `bayesian_per_frame` bool (checkbox) + freeze button; confinement folded into
the Elastic-net block as soft support; standalone confined solver kept backend-only (legacy `"auto"`).


**Scope:** replace the Force stage's implicit priority-ladder routing (inferred from
`l1_sparsity`/`fwd_mask_strength`/`bayesian_l2` sentinels) with an explicit method dropdown that
mirrors the Displacement stage, surfacing only the parameters each method uses.

## Motivation

The traction inversion is chosen today by a first-match ladder in `fttc.calculate_force_field`
(`l1_sparsity > 0` → group-L1, else `fwd_mask_strength > 0` + mask → confined, else `bayesian_l2`
→ BL2, else plain FTTC). Since the heuristic-sweep default flipped `l1_sparsity` to 0.05, the top
rung is always taken: the manual `regularization` field, the Bayesian checkbox, and the auto-λ
crosshair are all inert unless the operator first zeroes L1 — with no visual signal. `L1 Sparsity`
crossing zero silently swaps the whole solver. Three unrelated knobs are all labelled "L2". The
one L2 that composes with the default engine (`l2_ridge`, elastic-net ridge) was not surfaced at
all until now.

The fix: an explicit `force_method` dropdown, three methods, each showing only its own controls.
Same `METHOD`-block machinery the Displacement stage already uses.

## The three methods

| Method | Solver | Controls | Auto-λ |
|---|---|---|---|
| **FTTC + GCV** | Fourier Tikhonov (`FTTC`) | `regularization` slider | **Button**: GCV picks λ on the current frame and fills the slider (editable after). **Checkbox** `auto_gcv`: GCV per frame, slider greys. |
| **Bayesian L2** | real-space standardized (`bayesian_l2`) | none manual | **Checkbox** per-frame (default, ABL2). **Button** freeze: estimate λ once, reuse across the movie (`bayesian_lambda`). |
| **Elastic net (L1+L2) + mask confiner** | group-L1 (`forward_l1`) | `l1_sparsity`, `l2_ridge`, `fwd_mask_strength`, `fwd_mask_reach` | none |

There are **two distinct auto-λ methods**, not one: GCV selects λ for the Fourier FTTC operator
(valid, fills the manual slider), Bayesian evidence selects λ for the real-space BL2 operator (a
different λ in different units — the Fourier-evidence shortcut is degenerate, see `bayesian_l2.py`
header). GCV was removed in `8a011a9` (2026-07-19) and is restored here.

Shared, always shown: Young's modulus, Poisson ratio, gel height, and the Visualization block
(`force_vector_stride`, `force_arrow_scale`, `f_max`).

## Backend routing

Add `force_method` to `FTTCParameters` and `UnifiedParameters`.
`calculate_force_field` switches on it:

- `"auto"` (default) → today's inference, factored into `infer_force_method(params)`. Preserves
  every existing caller: the 480-scene sweep constructs `FTTCParameters(...)` directly, and every
  `.ntfm` written before the `l1_sparsity` default flip stored no `force_method`; both route
  exactly as before.
- concrete values → explicit dispatch. `"FTTC + GCV"` runs plain FTTC (with `auto_gcv` picking λ
  per frame when set); `"Bayesian L2"` runs BL2; `"Elastic net"` runs group-L1.

The standalone confined-forward solver (`forward_tfm`) stays in the backend and is reachable only
via the `"auto"` legacy path — it is no longer a UI-selectable method (confinement lives inside
the elastic-net method as soft support).

The UI always writes a concrete `force_method`; a resolver rewrites `"auto"` → the inferred
concrete method on load/build so the dropdown shows the true engine even for a legacy file.

## Build order

1. **Restore GCV** into the `FTTC` class (`_svd_block`, `_gcvfun`, `_gcv_blockdiag`,
   `_find_regularization`), plus the `auto_gcv` field and the `regularization=None → GCV`
   round-trip and a `find_gcv_regularization` one-shot wrapper for the button. Lifted from
   `8a011a9~1`, adapting the `_calculate_fourier_modes` unpack (now 2 return values).
2. **`force_method` field + dispatcher.** Add the field to both dataclasses; add
   `infer_force_method`; rewrite `calculate_force_field`'s branch selection to switch on the
   resolved method.
3. **UI.** `force_method` `choice` control + three `METHOD` blocks; generalize
   `_refresh_method_visibility`/`_build_method_block` (today keyed to `disp_method`) to a second
   owning dropdown; add a `"button"` control kind for the in-block GCV/freeze actions; remove the
   header `extra_actions` crosshair; wire the per-frame → slider-grey enablement; add the
   `_resolve_force_method("auto" → concrete)` handler.
4. **Tests + compat.** Extend the dataclass lockstep test for `force_method`/`auto_gcv`; add a GCV
   regression (recovers a known λ / ground truth); verify a legacy param dict (no `force_method`)
   loads and infers the right engine.

## Notes

- `l2_ridge` was surfaced in a prior step and now lives inside the elastic-net method block.
- Method naming in the dropdown: `"FTTC + GCV"`, `"Bayesian L2"`, `"Elastic net"` (labels TBD in
  UI; the stored `force_method` string is the contract — keep it stable once shipped).
