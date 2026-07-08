# napariTFM — Open Worklist

> Accomplished items are pruned (see git history). What remains below is the
> **open work** on the mask-confined forward traction epic, ranked
> easy-wins-first.

---

## EPIC: mask-confined forward traction — benchmark + solver hardening (updated 2026-07-07)

> **Context.** The photometric **one-shot** solver that originally headlined this
> epic is **retired** — it was hostage to the image-formation model (PSF, bleaching,
> out-of-plane motion) and underperformed on real data. It **never entered this
> repo** (untracked scratch in the `napariTFM2.5D` working copy); do **not** port
> `oneshot.py`. What survived is the idea, not the code: its one validated
> ingredient — the **log-soft mask (support) confinement** — was ported onto the
> **displacement field** as input, landing as the *forward (displacement-input,
> mask-confined) traction method* in `napariTFM/backend/forward_tfm.py` (commit
> `41322e3`), with the Force UI consolidated around the mask-confinement dial as
> the method switch (commit `78a1412`). Taking `u` as input drops bead texture and
> warp modelling entirely — the fragility that sank the photometric variant. The
> forward method is one more traction backend beside TV-L1+FTTC.

What remains open: a fair benchmark of the forward method (TASK 2) and hardening
its confined solve (TASK 4).

### One reusable artifact from the retired 2.5D scratch
The photometric prototype (`oneshot.py`, its widget, its tests, `compare_oneshot.py`,
`oneshot_bench2d/`) is dead scratch — leave it in `napariTFM2.5D`, don't port it. The
**one** piece worth reusing is `_dev/bench_generator/` — the scenario-sweep generator
(SPEC.md, fields.py, beads.py, psf.py, render.py, noise.py, writer.py, cli.py) with
vendored analytic physics in `_dev/vendor/DirectMethod/`. It's the seed for TASK 2.

### Settled findings — bake these in, don't relitigate
- **Mask confinement (survived → `forward_tfm.py`):** one log-scaled soft-β dial
  (0 = none → strong confinement). The literal hard gate was **retired** — it clips
  genuine near-edge forces (|t|r 0.95 vs 0.99). This is the ingredient that carried over
  from the photometric prototype.
- **Smoothness is now an explicit prior:** the photometric one-shot got its smoothness
  for free from a coarse basis (Gaussian-per-bead / B-spline, a few hundred DOF). The
  displacement-input port solves on a free per-pixel grid, so smoothness comes back as an
  explicit `γ‖∇t‖²` term (`fwd_smoothness`) — *the* primary regularizer of the confined
  solve. Without it, confining forces to the mask removes the solver's off-mask escape
  valve and the in-mask field overfits the delocalized displacement into high-frequency
  garbage — confinement then *hurts* (recovered error worse than zero); with it,
  confinement beats unconfined FTTC.
- **Retired with the prototype (do not resurrect):** the photometric ZNCC data term, the
  image pyramid + L-BFGS coarse-to-fine, the rigid-translation drift DOF, the
  beads-as-nodes basis, and preprocessing-inside-the-solve were all properties of the
  image→traction solver that no longer exists. Drift is now folded into the PIV
  displacement front-end; the preprocessing stage was removed outright (commit `a18d466`).
- **Dead ends (already refuted, don't retry):** zero-net-force projection (acts only on
  the null-space DC mode — useless), point-matching / Chamfer data terms (ill-posed,
  per-bead hotspots), TV vs L2 reg (negligible unless noise makes the regularizer
  binding).

### TASK 2 — a fair forward (mask-confined) vs TV-L1+FTTC benchmark
- **The cardinal rule (learned the hard way):** tune **each method's parameters per
  scenario** and report each at its *own* best. A benchmark with hand-picked FTTC reg is
  worthless — the committed `validate_TFM.py` uses `regularization=1e-6`, which
  under-regularizes (over-estimates, DTMS ~0.97) and does **not** reproduce the paper's
  Fig 3; ~1e-4 is the balanced operating point. FTTC gets GCV + a reg sweep; the forward
  method gets a grid over its `fwd_smoothness` (γ) / support-β / Tikhonov λ. (Sidebar: the
  committed validation script not reproducing the paper is itself worth a fix — see the
  standalone note if we split it out.)
- **Scenario axes:** noise level, bead density, displacement magnitude, force geometry
  (dipole / realistic cell / keratocyte), substrate stiffness, and **mask available vs not**
  (the forward method's real edge — the dipole benchmark can't show it). Build on
  `_dev/bench_generator` sweeps.
- **Metrics:** the Sabass four (correlation, DTM, DTMS, DTA — from `validate_TFM.py`) +
  strain energy + wall-clock.
- **Deliverable:** sweep curves + per-scenario best-config tables + a summary figure, plus
  the resulting production **default parameters** (the forward method's shipped defaults).
- **Regularization strategy sweep (organizing plan):** *which* regularizer + λ-selector to
  try, in what order, and why — from first principles — so the results are interpretable
  rather than a hyperparameter fishing expedition. Separates the penalty axis from the
  λ-selection axis via an oracle-λ sweep (only possible because the benchmark has ground
  truth), pre-registers a (condition → expected winner) table, and phases the work
  (support prior first → smoothness order → L1/elastic net → TV → selectors → graded prior).
  Full plan: **[`docs/specs/regularization-benchmark-plan.md`](docs/specs/regularization-benchmark-plan.md)**.

### TASK 4 (optimization) — swap the confined forward solver to preconditioned CG
**Status (2026-07-08): CPU/numpy path DONE, GPU/cupy path deferred.** `_solve_iterative`
is now PCG over the normal equations (`_build_normal_equations` + `_cg_call`), xp-neutral
(numpy | cupy), torch removed from this path; L-BFGS deleted. Matches the retired L-BFGS
output on the golden frame (corr=1.00000, rel_rms≈0, 54 CG iters, info=0). Tests green in
`tests/test_forward_pcg.py` (one-step exactness w/ discrete symbol, adjoint symmetry, ∇J,
DC-zeroing, convergence, golden regression, β=0 torch-free). Build plan:
[`forward-solver-pcg-plan.md`](docs/specs/forward-solver-pcg-plan.md).
**GPU path validated (2026-07-08):** `cupy-cuda13x` 14.1.1 installed in `.venv` (Blackwell /
CUDA-13, matching torch `cu130`); GPU==CPU to corr>0.9999 (`test_gpu_matches_cpu`). One gotcha
found and fixed: `scipy` and `cupyx` `cg` apply an `M` preconditioner inconsistently (cupyx 14.x
stalls), so the CG loop is **hand-rolled** (`_pcg`) — one algorithm on both backends. DC handled
by `P0·A·P0` (self-adjoint, zero-mean subspace).
**Remaining:** (a) add a `[gpu]` cupy extra to `pyproject`; (b) Phase-4 CPU perf
(`scipy.fft(workers=-1)` / plan reuse, warm-start across the sweep); (c) port PIV off
`torch.nn.functional` so torch can leave the package entirely. Original rationale below.

The β>0 confined path in `forward_tfm.py::_solve_iterative` previously used L-BFGS +
autograd (torch). The loss is a **convex quadratic**, so its minimizer solves the normal
equations `A t = b` with SPD `A` — the textbook case for **preconditioned Conjugate
Gradient**, using the existing Fourier closed-form (`_solve_closed_form`) as the
preconditioner. Expected ~5–10× wall-clock and — because CG needs no autograd — it
**drops the hard torch dependency** on this path (torch/cupy become an optional GPU
accelerator, not a requirement). One caveat: hand-rolled CG needs the exact adjoint (add
a dot-product test). Also the reusable inner-solve engine if the localization prior is
later upgraded L2 → L1. Full rationale, math, and acceptance criteria:
**[`docs/specs/forward-solver-pcg.md`](docs/specs/forward-solver-pcg.md)**.
