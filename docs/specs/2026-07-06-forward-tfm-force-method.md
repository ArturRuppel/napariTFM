# Forward-model traction inversion as an FTTC alternative

**Status:** proposed (2026-07-06)
**Scope:** add a second force-calculation method to the Force stage, selectable alongside FTTC,
exposing a tunable-smoothness / soft-support-prior inversion. All parameters exposed from the start.

---

## One line

FTTC is a regularized Fourier inversion of `u = G·t` with a *fixed* Lanczos low-pass. The forward
method is the *same* inversion with the fixed low-pass replaced by a **tunable smoothness prior γ**
(plus an optional **soft support prior β**). One method, one knob, spanning the clean→noisy regimes
where FTTC is alternately too sharp or just right.

## Motivation (from the benchmarkTFM study, 2026-07-06)

Recovering traction from a displacement field is ill-posed: the Green's operator `G` strongly
attenuates high spatial frequencies, so inverting it amplifies high-k noise. FTTC controls this two
ways at once — a Tikhonov term λ and a *fixed* Lanczos low-pass filter. We measured what that fixed
filter costs and buys (traction relL2 vs a synthetic ground truth, peak 1 µm = 10 px):

| displacement input | FTTC | forward (γ=0, i.e. no Lanczos) |
|---|---|---|
| exact (σ_u = 0) | 0.199 | **0.037** |
| + noise σ_u = 0.01 µm | **0.247** | 0.258 |
| + noise σ_u = 0.05 µm | **0.384** | 0.534 |
| + noise σ_u = 0.10 µm | **0.481** | 0.569 |

Reading: on **clean/dense** displacement the Lanczos filter throws away real signal — an un-filtered
inversion is ~5× better. The instant the displacement carries realistic noise, that same filter
*helps* (it suppresses the amplified high-k), and FTTC wins. The crossover is at very low noise.

FTTC's fixed filter is therefore a single hard-wired point on a tradeoff curve. Exposing the smoothing
as a **continuous knob γ** lets the user sit anywhere on that curve per dataset: γ→0 for clean, dense,
high-SNR beads; γ up for noisy or sparse data. That is the entire reason to add this method.

Two secondary findings shaped the parameter choices:
- **GCV auto-λ is fragile.** On structured (non-Gaussian) displacement error it badly
  under-regularizes and the traction blows up. Manual λ is the honest default; keep GCV as an opt-in.
- **A soft support prior (β) beats a hard mask** but is spatial, not spectral — it suppresses
  off-support *leak*, not the in-support high-k blowup that dominates under noise. Useful on real
  cells with a real segmentation mask; off by default.

## The method

Unknown: surface traction `t(x,y) = (t_x, t_y)` in Pa. Minimize

```
    J(t) = ‖ G·t − u ‖²  +  λ‖t‖²  +  γ‖∇t‖²  +  β‖ t·(1−mask) ‖²
```

- `G` is the existing Boussinesq/finite-thickness Green's operator (reuse
  `fttc._calculate_greens_function`; same E, ν, gel_height, pixel_size).
- `u` is the measured displacement field (the Force stage already receives it).
- `λ` Tikhonov, `γ` smoothness (penalizes high-k = tunable low-pass), `β` soft support.

**Two solve paths, chosen automatically by β:**

- **β = 0 → closed form, no iteration.** λ and γ are both diagonal in Fourier (‖∇t‖² = |k|²|t̂|²), so
  per mode `k` it is a 2×2 solve:
  `t̂(k) = (Gᴴ(k)G(k) + (λ + γ|k|²) I)⁻¹ Gᴴ(k) û(k)`.
  This is *as fast as FTTC* (one FFT pair + per-mode 2×2), and reduces to Tikhonov-FTTC-without-Lanczos
  at γ=0. γ acts as a k-dependent Tikhonov: a smooth, tunable replacement for Lanczos.
- **β > 0 → iterative.** The support term couples Fourier modes, so solve with L-BFGS on `t`
  (torch, autograd through the same FFT operator). Non-dimensionalize by E (solve for `s = t/E`,
  O(1)) for conditioning, multiply back at the end.

Relationship to FTTC: set γ=0, β=0 and add the Lanczos filter → you recover FTTC exactly. This method
is a strict superset; FTTC remains the default force method and is untouched.

Not to be confused with the *photometric* one-step prototype
(`napariTFM2.5D/_dev/onestep_tfm_prototype`), which fits traction to the raw image pair and is hostage
to the image-formation model (PSF, out-of-plane motion, bleaching) — that variant underperformed on
real data. This one takes the **displacement field** as input, so bead texture and warp modelling drop
out entirely and PIV/FFD stays the validated front-end.

## Parameters to expose

Grouped as they should appear under a "Forward" method selection in the Force widget. Shared-physics
params reuse the existing FTTC widgets unchanged; only the method-specific block is new.

| UI label | field | type / widget | default | notes |
|---|---|---|---|---|
| **Method** | `force_method` | combo: `fttc` \| `forward` | `fttc` | new selector on the Force stage |
| *— shared physics (reuse FTTC widgets) —* | | | | |
| Young's modulus | `young_modulus` | float, Pa | 5000 | same as FTTC |
| Poisson ratio | `poisson_ratio_substrate` | float | 0.5 | same |
| Gel height | `gel_height` | float or None | None | finite-thickness kernel; same |
| Pixel size | `pixel_size` | float, µm | 0.1 | same |
| Downscale | `downscale_factor` | int | 4 | same |
| Frame interval | `frame_interval` | float, min | 1 | same |
| *— method-specific (new) —* | | | | |
| Tikhonov λ | `fwd_regularization` | float via log₁₀ slider | 1e-4 | primary stability knob; mirror FTTC's exponent UI |
| Smoothness γ | `fwd_smoothness` | float via log₁₀ slider | 0.0 | **headline knob**; 0 = sharpest, up = smoother |
| Auto-λ (GCV) | `fwd_auto_gcv` | bool | False | opt-in; fragile on structured noise |
| Support prior β | `fwd_support_weight` | float via log₁₀ slider | 0.0 | soft off-mask penalty; needs a mask (below) |
| Support mask source | `fwd_support_source` | combo: none \| cell mask | none | where the (1−mask) comes from |
| Max iterations | `fwd_max_iter` | int | 200 | only used when β>0 (iterative path) |
| Device | `fwd_device` | combo: auto \| cuda \| cpu | auto | torch; mirror `ffd_device` |
| Precision | `fwd_dtype` | combo: float64 \| float32 | float64 | mirror `ffd_dtype` |
| *— visualization (reuse) —* | | | | |
| Vector stride / arrow scale / f_max | as FTTC | | | shared display params |

Design note: follow the **displacement-stage pattern** (`DisplacementParameters` with a `method`
field and method-prefixed params `piv_*` / `ffd_*`) rather than a separate dataclass — add
`force_method` and the `fwd_*` fields to `FTTCParameters`. Keeps one params object per stage, one
validation gate, and matches the codebase's existing idiom. (Optionally rename the dataclass to
`ForceParameters` for honesty; low priority, more churn.)

## Backend integration

- **`napariTFM/backend/parameter_dataclasses.py`** — add `force_method` + the `fwd_*` fields to
  `FTTCParameters` (class at line 32). Pure additive; existing configs deserialize with defaults.
- **`napariTFM/backend/forward_tfm.py`** *(new)* — the solver:
  - `forward_traction_closed_form(u, G, λ, γ)` — the per-mode 2×2 Fourier solve (β=0 path).
  - `forward_traction_iterative(u, G, λ, γ, β, mask, max_iter, device, dtype)` — the L-BFGS path.
  - both return traction `(2,H,W)` in Pa, matching FTTC's output contract.
  - reuse `fttc._calculate_greens_function` for `G` (do **not** duplicate the kernel).
- **`napariTFM/backend/fttc.py`** — in `calculate_force_field` (line 66), branch on
  `params.force_method`: `fttc` → current path; `forward` → dispatch to `forward_tfm`. Keep the same
  `Generator[..., FTTCResult]` streaming contract so the widget/batch progress plumbing is unchanged.
- **`napariTFM/backend/parameter_validation.py`** — extend `validate_fttc_parameters`: γ,β ≥ 0;
  `fwd_max_iter` > 0; if β>0 then `fwd_support_source` must resolve to a mask; device/dtype in range.
- **`napariTFM/backend/batch_analysis.py`** (calls `calculate_force_field` at line 1011) — no change
  if dispatch lives inside `calculate_force_field`. Verify the support mask is available in batch.

## UI integration

- **`napariTFM/widgets/fttc_widget.py`** — add the **Method** combo at the top of the force panel;
  show/hide the `fwd_*` controls when `forward` is selected (mirror how the displacement widget
  toggles `piv_*` vs `ffd_*` vs Farneback controls). Reuse the existing log₁₀-exponent spinbox
  pattern already used for `regularization` for λ, γ, β. The GCV button (line 122,
  `calculate_optimal_regularization`) applies to `fwd_regularization` when in forward mode.
- **Support-mask plumbing (the one non-trivial wire).** The Force stage currently receives only the
  displacement field. β needs a cell mask. Options, in order of preference:
  1. reuse the **cell mask already produced in Preprocessing/segmentation** (the same mask BISM/stress
     uses) — thread it into `calculate_force_field` as an optional arg. Cleanest.
  2. a napari **Labels layer** picker in the force widget (user points at any mask layer).
  Decide before implementing β; it is the only param that adds cross-stage data flow.

## Reuse / dependencies / risks

- **Reuse:** Green's kernel, downscale, GCV, the streaming/preview/batch plumbing, the FTTC widget
  scaffold. The math is ~80 lines; most of the work is UI wiring and the mask flow.
- **torch** is already an optional extra (the FFD displacement backend uses it). β/iterative path and
  the autograd solver ride on that; the closed-form β=0 path is pure numpy/FFT and needs no torch, so
  the *default* forward method works without the torch extra. State this in validation.
- **Risks:** (a) support-mask cross-stage flow (above); (b) param-set growth in one dataclass — keep
  `fwd_*` prefix discipline; (c) naming (`forward` vs `direct` vs `forward-model` — avoid "direct",
  which in the Blumberg–Schwarz sense means the strain-differentiation method, which this is *not*).

## Validation

- **Unit:** closed-form path at γ=0,β=0,no-Lanczos reproduces a Tikhonov reference; kernel self-check
  `forward(t_gt/E) == u_gt` (already passes in the benchmark, r=1.0).
- **Behavioral:** reproduce the benchmark table above — γ=0 beats FTTC on clean displacement, a
  γ-sweep crosses over to match FTTC under injected displacement noise. This is the acceptance test:
  one γ slider must span both regimes.
- **Real data:** the actual goal — run FTTC vs forward on a real experiment in the UI, sweep γ, and
  see which produces a physically sensible traction field. (The synthetic noise model does not match
  real bead noise-fragility, which is *why* this belongs in the interactive tool.)

## Open questions for Artur

1. Support-mask source — reuse the segmentation/cell mask (preferred) or a Labels-layer picker?
2. Rename `FTTCParameters` → `ForceParameters`, or keep the name and just add `fwd_*` fields?
3. Method name in the UI: "Forward", "Forward-model", something else?
