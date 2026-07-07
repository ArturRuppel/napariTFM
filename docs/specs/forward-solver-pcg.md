# Spec: preconditioned-CG solver for the confined forward traction solve

Status: proposed · Scope: `napariTFM/backend/forward_tfm.py` (`_solve_iterative`, the β>0 path)

## Summary

Replace the L-BFGS + autograd solver on the confined (β>0) forward path with a
**preconditioned Conjugate Gradient (PCG) solve of the normal equations**, using
the existing Fourier closed-form as the preconditioner. Expected ~5–10× wall-clock
speedup and removal of the hard torch dependency on this path.

## Context — how the confined solve maps onto regularization theory

The forward solver minimizes, per frame, a spatially-weighted regularized objective
(see the `forward_tfm.py` module docstring):

    J(t) = ‖W·(G·t − u)‖²  +  λ‖t‖²  +  γ‖∇t‖²  +  β‖t·(1−mask)‖²

- `G` — the Boussinesq / finite-thickness Green's operator, reused verbatim from
  FTTC (folds in E, ν, gel_height, pixel_size); diagonal in Fourier (a per-mode 2×2).
- `λ‖t‖²` — zeroth-order Tikhonov (conditioning ridge).
- `γ‖∇t‖²` — first-order Tikhonov (gradient smoothness); the *primary* regularizer.
- `β‖t·(1−mask)‖²` — the **soft support / localization prior**. Combined with λ this is
  a **spatially-varying ridge**: the effective per-pixel penalty is
  `w_eff(r) = λ + β·(1−mask(r))` — weak (λ) inside the mask, strong (λ+β) outside.
  This is the binary-mask limit of a graded probability prior `w_eff = λ + β·(1−p(r))`;
  swapping the binary `off` map for a continuous `1−p` is a one-line generalization.
- `W` — a *separate* spatial weight on the **data term** (heteroscedastic noise model):
  trust displacement only inside mask+margin. Distinct from β, which weights the prior.

`J` is a **convex quadratic** in `t`. The spatially-varying `W` and β terms are diagonal
in real space, not Fourier, so they couple Fourier modes — which is exactly why the
closed-form per-mode inversion (the β=0 path, `_solve_closed_form`) no longer applies and
the current code falls back to an iterative solve.

## Why PCG instead of L-BFGS

Because `J` is a convex quadratic, its gradient is linear and the minimizer solves the
**normal equations** `A t = b` with

    A = Gᴴ W G + λI + γ LᵀL + β D        D = diag((1−mask)²),  L = ∇
    b = Gᴴ W u

`A` is symmetric positive-definite (λI makes it strictly PD). CG is the textbook-optimal
method for SPD systems; L-BFGS is a *general* nonlinear optimizer that does not exploit
this structure. Four compounding reasons PCG wins here:

1. **Exact vs approximate Hessian.** The Hessian is exactly `A` and is *constant*. CG
   wields it through matrix–vector products `A·p`. L-BFGS reconstructs a rank-≤25
   (`history_size`) approximation of `A⁻¹` from gradient history — wasted effort for a
   quadratic whose Hessian is already known implicitly through the operator.
2. **Closed-form step, no line search.** CG's step length is analytic
   (`α = rᵀr / pᵀAp`): one `A`-apply per iteration. The current L-BFGS uses a
   strong-Wolfe line search, i.e. *several* FFT-pair evaluations per outer step.
3. **√κ convergence + clustered spectrum.** CG contracts at ~√κ (vs κ) and superlinearly
   as it exhausts clustered eigenvalues. `A`'s spectrum is tightly clustered (`GᴴG` is a
   smooth Fourier multiplier; λI piles mass at one value), which CG exploits and L-BFGS
   cannot.
4. **The Fourier preconditioner we already have.** Split
   `A = M + (real-space coupling)` where `M = GᴴG + λI + γ|k|²` is per-mode 2×2 and
   Fourier-diagonal — i.e. `M⁻¹` is exactly the `_solve_closed_form` / `calculate_traction_2d`
   machinery. Precondition CG with `M⁻¹`:
   - `W=I, β=0` → `M=A` → PCG converges in **one step** (degenerates to the closed form).
   - Turning up β / making `W` selective is a bounded perturbation → a **handful** of PCG
     iterations instead of the 200-cap L-BFGS steps.

Net: fewer iterations × cheaper iterations. Realistically ~10–30 PCG iterations vs
~100–200 L-BFGS steps (each doing multiple FFT evals) → order 5–10× wall-clock, more when
`W ≈ I`. Warm-start PCG from `_solve_closed_form` for an even lower iteration count.

## CPU / GPU

Runs on **both**, exactly like the current path — the primitives are FFTs, pointwise
multiplies, and vector dot-products, all device-agnostic.

- **CPU:** pure numpy/scipy. `scipy.sparse.linalg.cg` with a `LinearOperator` wrapping a
  hand-written `A`-apply (fft → ×G → ifft → ×W → fft → ×Gᴴ → ifft, plus `λt + γLᵀLt + βDt`
  in real space).
- **GPU:** the same operator on device arrays — cupy (`cupyx.scipy.sparse.linalg.cg` +
  cuFFT) or a hand-rolled CG loop over `torch.fft`.

**Bonus — drops the torch dependency on this path.** CG needs **no autograd**: for a
quadratic the gradient *is* the linear map `A·t − b`, hand-written from the same
forward+adjoint FFT operator. The current β>0 path exists only because it needs
`loss.backward()` through the FFT; CG removes that, so the whole forward solver can be
torch-free numpy/FFT on CPU (matching the existing β=0 path), with torch/cupy an *optional*
GPU accelerator rather than a requirement. Also lower memory (no autograd tape) and faster
per FFT (no reverse-mode overhead).

## The one correctness caveat

CG requires the **exact adjoint** `Gᴴ` (true conjugate transpose) and symmetric `LᵀL`.
Autograd supplies the adjoint for free; hand-rolling it, you must get the conjugate right.
For FFT-diagonal operators the adjoint is just the conjugate of the multiplier, so it's
trivial — but a wrong adjoint makes CG stall/diverge (not merely slow down). Add a
one-line dot-product adjoint test: `⟨A x, y⟩ ≈ ⟨x, A y⟩`.

## Forward compatibility with an L1 upgrade

If the localization prior is ever upgraded from L2 to L1/sparsity (focal-adhesion peaks),
`J` stops being a pure quadratic and CG no longer applies directly — but FISTA/ADMM, the
standard L1 solvers, have a **quadratic inner subproblem every iteration**, solved by
exactly this PCG-with-FFT core. So building the CG engine now is also the reusable
foundation for the sparsity extension.

## Acceptance

- `_solve_iterative` returns traction matching the current L-BFGS output within solver
  tolerance on the confined benchmark tiers (regression against saved outputs).
- CPU path runs with torch uninstalled.
- Wall-clock improvement demonstrated on a representative frame.
- Adjoint test passes.
