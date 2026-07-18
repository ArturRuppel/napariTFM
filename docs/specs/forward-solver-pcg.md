# Spec: preconditioned-CG solver for the confined forward traction solve

Status: accepted (2026-07-08) · Scope: `napariTFM/backend/forward_tfm.py` (`_solve_iterative`, the β>0 path)

> Revised after an external applied-maths review. Corrections from that review are
> called out inline as **[review]**; a few speculative refinements are marked
> **[optional]** and should not be built until the simple version shows it's needed.

## Summary

Replace the L-BFGS + autograd solver on the confined (β>0) forward path with a
**preconditioned Conjugate Gradient (PCG) solve of the normal equations**, using
the existing Fourier closed-form as the preconditioner. Expected speedup on
*mild-confinement* frames and removal of the hard torch dependency on this path.
The iteration-count win is regime-dependent — see the bound below; it is not
unconditional.

## Context — how the confined solve maps onto regularization theory

The forward solver minimizes, per frame, a spatially-weighted regularized objective
(see the `forward_tfm.py` module docstring):

    J(t) = ‖W·(G·t − u)‖²  +  λ²‖t‖²  +  γ‖∇t‖²  +  β‖t·(1−mask)‖²

> **Amendment (2026-07-08).** `λ` (`regularization`) is the *shared* dial with
> FTTC/`_solve_closed_form`, so its Tikhonov term is FTTC's **physical `λ²‖t‖²`** on
> this path too. The as-built β>0 solver instead used a linear `λ/N` coefficient
> (matching the retired L-BFGS closure), which made the *same* dial regularize
> differently on the β=0 (λ²) vs β>0 (λ) branches — up to ~125× at λ=1e-2. Fixed by
> setting the identity-term coefficient to `λ²·(E·T0)²/denom`, so the `denom` cancels
> against the data term and β=0,γ=0 reproduces `_solve_closed_form` at the same λ. The
> `(λ/N)` in the equations below is superseded by this; `β/γ` keep their `1/N` scaling
> (no FTTC counterpart). Guard: `test_lambda_matches_closed_form_across_branches`.

- `G` — the Boussinesq / finite-thickness Green's operator, reused verbatim from
  FTTC (folds in E, ν, gel_height, pixel_size); diagonal in Fourier (a per-mode 2×2
  Hermitian PSD tensor). It is low-pass (singular values ~1/|k|) and `Ĝ(0)=0`, so it
  has a nullspace: the spatial mean of `t` is unobservable.
- `λ‖t‖²` — zeroth-order Tikhonov (conditioning ridge).
- `γ‖∇t‖²` — first-order Tikhonov (gradient smoothness); the *primary* regularizer.
- `β‖t·(1−mask)‖²` — the **soft support / localization prior**. Combined with λ this is
  a **spatially-varying ridge**: effective per-pixel penalty `w_eff(r) = λ + β·(1−mask(r))`.
  Binary-mask limit of a graded prior `w_eff = λ + β·(1−p(r))`; swapping the binary
  `off` map for a continuous `1−p` is a one-line generalization.
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
this structure. The reasons PCG wins:

1. **Exact vs approximate Hessian.** The Hessian is exactly `A` and is *constant*. CG
   wields it through matrix–vector products `A·p`. L-BFGS reconstructs a rank-≤25
   (`history_size`) approximation of `A⁻¹` from gradient history — wasted for a quadratic.
2. **Closed-form step, no line search.** CG's step length is analytic (`α = rᵀr / pᵀAp`):
   one `A`-apply per iteration. The current L-BFGS uses a strong-Wolfe line search, i.e.
   *several* FFT-pair evaluations per outer step.
3. **√κ convergence + clustered spectrum**, further improved by the preconditioner below.

## The preconditioner — and its exact-consistency requirement

Split `A = M + (real-space coupling)` where

    M = GᴴG + λI + γ LᵀL      (Fourier-diagonal, per-mode 2×2 ⇒ M⁻¹ via FFTs)

so `M⁻¹` is exactly the `_solve_closed_form` / `calculate_traction_2d` machinery.

**[review] Use the same *discrete* `LᵀL` symbol in `M` as in `A`.** `A`'s smoothness term is
built from finite differences (`torch.roll`-based forward difference), whose symbol per axis
is `|e^{ik}−1|² = 4·sin²(k/2)`, summed over axes — **not** the continuum `|k|²`. If `M` uses
`|k|²` while `A` uses `4·sin²(k/2)`, then `M ≠ A` even at `W=I, β=0`, and the one-step-exactness
property (and the unit test built on it) silently fails. Harmless to preconditioner *quality*,
fatal to the *test*. Use `Σ_axes 4·sin²(k_axis/2)` in both.

## Convergence — the honest bound (corrects the earlier "few iterations" claim)

**[review]** With `w_min ≤ W ≤ w_max` (entrywise) and `0 ≤ D ≤ I`, and noting `λ_min(M)=λ`
(attained at `k=0`, where `Ĝ(0)=0` and the Laplacian symbol vanishes):

    λ_max(M⁻¹A) ≤ max(w_max, 1) + β/λ
    λ_min(M⁻¹A) ≥ min(w_min, 1)
    ⇒  κ(M⁻¹A) ≲ (w_max + β/λ) / min(w_min, 1)

CG iterations scale as `√κ ~ √(β/λ)`. **The support prior's whole point is `β ≫ λ`, so the
naive "few iterations" claim fails exactly when confinement is doing work.** Likewise a large
contiguous *hard*-masked region (`W=0` on a region of diameter `d`) degrades `λ_min` polynomially
in `d` (the bad subspace is low-`k` fields whose image concentrates in the masked region).

The rescue the earlier draft omitted: **CG converges in ≤ r+1 iterations for a rank-`r`
perturbation of a perfectly-preconditioned system**, and `rank(Gᴴ(W−I)G + βD) ≤ 2·(#grid
points where W≠1 or mask≠1)`. So the true iteration-count predictor is

    #iters  ≈  min( √(β/λ) · const ,  2·#perturbed points )

Fast when the perturbed region is small **or** the contrasts (β/λ, 1−w_min) are mild; slow
only when both are large. **Instrument it: log CG iteration counts vs β and vs masked
fraction ρ**, so we can see which regime a given dataset is in.

**[optional] Deflated / two-level PCG for the hard regime.** Deflate the DC block plus a small
coarse space on the masked region (a few dozen coarse indicator/bump functions restricted to
the masked region) to remove exactly the slow subspace above at negligible cost; or use one
linearized-ADMM sweep as the preconditioner inside *flexible* CG. Only if the logged iteration
counts show the hard regime actually bites.

## The DC / nullspace mode (corrects the earlier diagnosis)

**[review]** The earlier draft worried CG would struggle at the nullspace for small λ. It won't:
`Ĝ(0)=0` and the Laplacian symbol vanishes at `k=0`, so `A·const = λ·const` *and* `M·const =
λ·const`, i.e. the **preconditioned eigenvalue at DC is exactly 1** regardless of λ. No CG
problem. The real issue is numerical: `b = GᴴWu` has exact-zero DC in principle, but FFT
rounding leaves `O(ε·‖u‖)` there, which `M⁻¹` amplifies by `1/λ` and injects a spurious mean
for small λ. **Fix: explicitly zero `t̂(0)` (or the DC block of the residual) each iteration**,
and treat `mean(t)=0` as a reporting *convention*, not an inference — the data is uninformative
about it. (With β>0 the mean becomes weakly identified through the support prior; fine, but note
it, since it changes what a magnitude metric sees downstream.)

**As implemented (2026-07-08):** `M⁻¹` annihilates the DC (0,0) block, and `apply_A` projects DC
out on **both** sides (`P0·A·P0`) so it stays self-adjoint and the whole Krylov iteration lives in
the zero-mean subspace. This is required for the hand-rolled CG to be well-posed on both backends —
projecting only `apply_A`'s output makes `A` non-symmetric and CG stalls; projecting neither lets
GPU round-off inject a DC component the preconditioner cannot correct, and CG stalls there too.

## Backend, build order, and the xp-neutrality rule

**Decision (2026-07-08): CuPy + numpy via array-module dispatch; torch is removed from this
path.** The solver primitives are FFTs, pointwise multiplies, and vector dot-products — all of
which numpy and cupy expose under one API. So the operator is written **once**, against a
dispatched array module `xp = cupy if gpu else numpy`, and runs on GPU (cupy + `cupyx.scipy.fft`)
or CPU (numpy + `scipy.fft`) with no second implementation. **The CG loop itself is hand-rolled**
over `xp` primitives (`_pcg`), *not* `scipy`/`cupyx` `cg`: those two libraries apply an `M`
(LinearOperator) preconditioner **inconsistently** — verified 2026-07-08, the same Fourier
preconditioner that converges in ~54 iters under scipy `cg` stalls (2000+ iters, no convergence)
under cupyx 14.x, while *unpreconditioned* cupyx `cg` converges fine. Depending on both `cg`
implementations would silently break the single-source guarantee; one loop over `xp.sum`/axpy plus
the FFT operators is the identical algorithm on both backends. The earlier "hand-rolled CG loop over `torch.fft`" GPU option is
**dropped** — we standardize the whole package (this solver and the PIV displacement backend) on
cupy for GPU, which lets torch leave the package entirely once PIV is ported off
`torch.nn.functional` (a separate task; its cost is re-earning PIV's numpy-equivalence property).

**Build order: GPU (cupy) first, then the CPU (numpy) path; device-independence is deferred.**
The GPU path serves the research and the TASK 2 sweep now; the pure-numpy path is a
distribution/reproducibility concern that must land before publication, not before first use.

**The xp-neutrality rule — non-negotiable; this is what makes GPU-first cheap.** Write the
operator array-module-agnostic *from the first line*, even while only the cupy branch is
exercised. No `cupyx`-only calls or `cp.`-specific idioms in the core, no `asnumpy` threaded
through it; keep `.get()` / `asnumpy` strictly at the I/O boundary. Done right, the CPU path is a
config flag (`xp = numpy`); done wrong, "add CPU later" silently becomes a rewrite and we lose
the single-source property that motivated cupy in the first place.

**No autograd, real arithmetic.** CG needs no autograd: for a quadratic the gradient *is*
`A·t − b`, hand-written from the same forward+adjoint FFT operator (lower memory, no reverse-mode
overhead). Run the CG recurrence in **real arithmetic** — apply symbols on the complex spectrum
but never let complex residuals leak into the Krylov iteration.

## Correctness caveats on the hand-coded operator

**[review] The adjoint test needs teeth, because `Gᴴ = G`.** Since `Ĝ(k)` is Hermitian (and
`Ĝ(−k)=conj(Ĝ(k))` for a real operator), the conjugate transpose is a **no-op** — so a wrongly
constructed *non-Hermitian* symbol would pass any transposition check trivially. Instead:
- enforce Hermiticity at construction: `Ĝ ← (Ĝ + Ĝᴴ)/2`;
- validate with a dot-product test `⟨A x, y⟩ ≈ ⟨x, A y⟩` on random *real* fields;
- add a finite-difference check of `∇J` against `A t − b`.
These two tests replace everything autograd was buying.

**[review] FFT normalization for the adjoint.** With full complex FFTs, `Gᴴ = IFFT ∘ Ĝᴴ ∘ FFT`
holds for any normalization (the factors cancel). With **`rfft2`** it does **not**: the
Hermitian-redundant columns (`k_x=0` and Nyquist) need factor-of-2 weights in the adjoint.
Prefer full complex FFTs (take the real part at the end), or write the weighted adjoint
explicitly. Note also that a bare first-derivative symbol `ik` is sign-ambiguous at the Nyquist
mode — `LᵀL = 4·sin²(k/2)` is safe, a raw spectral `∇` is not.

## Stopping criterion

**[review]** A small *preconditioned residual* does not imply small *error* in the weakly
determined directions (low-`k` inside a masked region — the very directions the convergence
bound flags), and those are what downstream metrics measure. Use a **tight relative-residual
tolerance (≈1e-8, cheap here)** rather than one "tuned to metric precision", or monitor
`‖t_{j+1}−t_j‖` directly. Otherwise the benchmark (see `regularization-benchmark-plan.md`) will
confound solver truncation error with regularizer bias.

## Alternatives considered

- **Woodbury / Sherman–Morrison.** `A = M + UCUᵀ` with rank `2·(#perturbed points)`; an exact
  direct solve, but the capacitance matrix is dense `r×r` — viable only when the masked +
  off-support region is ≲ a few thousand points. **[optional]** Worth implementing as the *exact
  reference* for small masks; not the workhorse.
- **ADMM / operator splitting.** Introduce `v=Gt`, `z=t`: the `v`/`z` updates are pointwise 2×2
  (absorbing `W` and `βD`), the `t`-update is a pure Fourier 2×2 solve. Typically slower than
  well-preconditioned CG for a *quadratic* — **but the benchmark's L1/elastic-net/TV methods need
  exactly this splitting anyway** (ADMM/FISTA with the Fourier-diagonal solve as the prox of the
  smooth part). So build the splitting infrastructure regardless and let the quadratic path use
  PCG. See the shared-primitive note below.
- **Multigrid** — wrong tool: `GᴴWG` is nonlocal in real space, so standard smoothing analysis
  doesn't apply, and FFT preconditioning already handles the elliptic part.
- **Direct sparse factorization** — `A` is dense in both bases (convolution is dense in space;
  `W` scatters in frequency). Not applicable.

## Shared Fourier-diagonal primitive (ties this spec to the benchmark spec)

The per-mode 2×2 Fourier-diagonal solve `M⁻¹` should be built as **one reusable primitive**,
because it is simultaneously:
1. the **PCG preconditioner** (this spec);
2. the **ADMM/FISTA prox** for the smooth part of the L1 / elastic-net / TV solvers
   (`regularization-benchmark-plan.md`, Phase 1.3);
3. the fast inner solve for **stochastic trace / log-det probes** used by weighted-GCV and
   Bayesian evidence (`regularization-benchmark-plan.md`, Phase 2).
Building it once is the single decision that unifies both documents.

## Practical accelerators

**[review] Warm-start across the λ/γ/β sweep** in the benchmark (initialize the solve at each
hyperparameter from the previous one's solution). This typically cuts iteration counts 5–10×
across a sweep — more impactful for total benchmark cost than any preconditioner micro-tuning.

**Precompute the shape-invariant symbols once.** `Ĝ`, `Ĝᴴ`, the per-mode `M⁻¹` inverse, and the
`4·sin²(k/2)` Laplacian symbol depend only on grid size and E/ν/gel_height/pixel_size — not on
the frame. Build them once per shape and reuse across every frame and every point in the sweep.

**CPU FFT: library + plan reuse, not custom kernels.** The dominant cost per PCG iteration is
~6 FFTs, and CPU FFTs are already optimal compiled binaries (pocketfft; FFTW/MKL as drop-ins).
Get CPU speed from `scipy.fft(workers=-1)` or pyFFTW with cached plans (array shape is constant
across iterations and frames) — **not** from a bespoke Numba/Cython kernel. There is **no
separate compiled CPU code block**: the one hot op is the FFT and it is already compiled;
everything else is memory-bound vectorized array math a hand kernel would improve only marginally
(Amdahl); and a parallel CPU implementation would re-introduce the two-backends-kept-equivalent
maintenance tax that single-source xp-dispatch exists to eliminate. If — and only if — profiling
a real frame shows a *non-FFT* op dominating, wrap that single op behind an optional `@njit`
variant; never a parallel code path.

## Acceptance

- `_solve_iterative` returns traction matching the current L-BFGS output within solver
  tolerance on the confined benchmark tiers (regression against saved outputs).
- **GPU (cupy) path is the first deliverable**; its correctness is established by the identity
  tests below (which need no CPU reference), not by diffing against a numpy twin.
- **CPU (numpy) path runs with neither torch nor cupy installed** — the pure-numpy fallback,
  landed by flipping `xp` to numpy (proof the xp-neutrality rule held). Deferred, but required
  before publication.
- Wall-clock improvement demonstrated on a representative *mild-confinement* frame; CG iteration
  count logged vs β and masked fraction ρ to characterize the hard regime.
- Unit tests pass (on cupy first, and on numpy once that path lands): (i) `W=I, β=0` one-step
  exactness with the **discrete** Laplacian symbol; (ii) adjoint dot-product test on random real
  fields; (iii) `∇J` finite-difference check; (iv) DC-mode zeroing verified (no spurious mean as
  λ→0).
