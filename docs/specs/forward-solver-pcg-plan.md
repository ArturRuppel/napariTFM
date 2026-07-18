# Build plan: preconditioned-CG confined forward solver

Implements [`forward-solver-pcg.md`](forward-solver-pcg.md) (accepted 2026-07-08).
Scope: replace the L-BFGS+autograd `_solve_iterative` (β>0 path) in
`napariTFM/backend/forward_tfm.py` with a preconditioned-CG solve of the normal
equations, on a single `xp`-dispatched (cupy | numpy) operator. GPU (cupy) first.

## Invariants we must not break

- **Same objective `J`.** PCG must minimize the *exact* quadratic the L-BFGS closure
  minimizes today (data term normalized by `denom`, regularizers by `.mean()` i.e.
  `1/N`, `N=2·H·W`), so `λ` (`regularization`), `β` (`confinement_to_beta`), and `γ`
  (`fwd_smoothness`) keep their current meaning and the UI dial calibration
  (`MASK_BETA_MIN/MAX`) and defaults (`fwd_smoothness=0.05`) carry over untouched.
  This is what makes the "matches L-BFGS output" regression meaningful.
- **Output contract.** `(2, H, W)` float32 Pa, `[0]=t_x`; `forward_traction_frame`
  dispatch (β≤0 → `_solve_closed_form`, β>0 → iterative) unchanged.
- **xp-neutrality rule** (spec): the operator is array-module-agnostic from line one;
  `.get()`/`asnumpy` only at the I/O boundary.

## The math we implement (in the non-dim variable `w`, `t = E·T0·w`)

Linear forward map `P(w) = T0·IFFT( GE ·ₘ FFT(w) )` (real part), `GE = E·G`, per-mode
2×2 Hermitian. Its adjoint is `Pᵀ(r) = T0·IFFT( GE ·ₘ FFT(r) )` (GE Hermitian ⇒ same
symbol). Normal equations `A w = b`:

    A w = (1/denom)·Pᵀ(wf·P w) + (λ/N)·w + (β/N)·(off·w) + (γ/N)·(LᵀL w)
    b   = (1/denom)·Pᵀ(wf·u)

- `wf` = `_fit_weight` (data-term weight `W`), `off` = `(1-mask)`, `denom` =
  `Σ wf·u²` (relative-residual normalizer, as today).
- `LᵀL` via `roll` forward-difference (`2w − roll(w,-1) − roll(w,+1)` per axis) — the
  **discrete** stencil. The factor-2 on A and b cancels, so it is dropped.

Preconditioner `M` (Fourier-diagonal, the data term at `W=I`, no `β`):

    M̂(k) = (T0²/denom)·GEᴴGE  +  (λ/N)·I  +  (γ/N)·(4sin²(kx/2)+4sin²(ky/2))·I

`M⁻¹` = FFT → per-mode 2×2 solve → IFFT. **The Laplacian symbol in `M̂` is
`4·sin²(k/2)`, identical to the `roll` stencil in `A`** — required for one-step
exactness (spec `[review]`). At `W=I, β=0`, `A = M` exactly ⇒ PCG converges in 1 step.

DC: zero the DC (`k=0`) of `b`; in `M̂(0)` keep only the `(λ/N)` term (no `1/λ`
blow-up). Report `mean(t)` as convention, not inference.

## Phases

### Phase 0 — scaffolding & backend dispatch
- `_resolve_backend(fwd_device) -> (xp, fft_mod, cg, LinearOperator, gpu: bool)`:
  `"cpu"`→numpy/scipy; `"cuda"`→cupy (raise if unavailable); `"auto"`→cupy if
  importable and a device is present, else numpy. Replaces `_resolve_torch_device`.
- Add `fwd_cg_tol: float = 1e-8` to both `FTTCParameters` dataclasses; reuse
  `fwd_max_iter` as CG `maxiter` (update its comment). Keep `fwd_device`/`fwd_dtype`.

### Phase 1 — the operator (GPU/cupy, xp-neutral)
- `_forward_symbols(shape, params, xp, fft_mod)` → cached `(GE, MinvTensor, lap, denom_scale…)`
  built once per `(shape, E, ν, gel_height, pixelsize, λ, γ)` key (module-level dict;
  the precompute/reuse accelerator from the spec).
- `_apply_A(w, ...)`, `_apply_Minv(r, ...)` as pure `xp` functions (FFT + pointwise +
  roll). Real arithmetic in the Krylov vector; complex only inside the symbol apply.
- `_solve_iterative` rewritten: build `wf/off/denom`, assemble `b` (DC-zeroed), wrap
  `A`/`M` as `LinearOperator`s over the flattened real `2HW` vector, call `cg(A, b,
  M=Minv, rtol=fwd_cg_tol, maxiter=fwd_max_iter, x0=warm)`, reshape, `t = E·T0·w`,
  return float32 via `asnumpy`. **Log CG iteration count vs β and masked fraction ρ.**

### Phase 2 — correctness tests (`tests/test_forward_pcg.py`), run on the available backend
Identity tests need no CPU reference (spec):
- (i) **one-step exactness**: `wf≡1, β=0` ⇒ CG converges in ≤1 iteration; result ==
  `_solve_closed_form` within tol.
- (ii) **adjoint/symmetry**: `⟨A x, y⟩ ≈ ⟨x, A y⟩` on random real fields.
- (iii) **`∇J` finite-difference** vs `A w − b`.
- (iv) **DC zeroing**: no spurious mean as `λ→0`.
- (v) **regression vs L-BFGS**: matches the current output within solver tol on a
  synthetic confined frame (guards the "same J" invariant).

### Phase 3 — wire in & dependency
- `pyproject.toml`: add a `[gpu]`/`[cupy]` extra (`cupy-cuda12x`); torch stays only
  until PIV is ported (separate task, noted in TODO). No hard cupy dep.
- Confirm `forward_traction_frame` β>0 path drives the new solver; run the full
  backend test suite green.

### Phase 4 — deferred (before publication, not now)
- Flip `xp=numpy` CPU path on, run the same tests torch/cupy-free (proves xp-neutrality).
- `scipy.fft(workers=-1)`/pyFFTW plan reuse; warm-start across the TASK 2 sweep.

## Acceptance (from the spec)
GPU path first, validated by identity tests (i)–(iv); regression (v) vs L-BFGS;
CG-iters logged vs β and ρ; CPU-numpy path deferred but reachable by an `xp` flip.

## Build outcome (2026-07-08)
Done and green (`tests/test_forward_pcg.py`, 8 tests; full suite 600 passed, 1
pre-existing movie-writer env failure). Two deviations from the plan as written:
- **CG is hand-rolled (`_pcg`), not `scipy`/`cupyx` `cg`.** Those libraries apply an
  `M` preconditioner inconsistently — cupyx 14.x stalls where scipy converges — so a
  single `xp` loop is the only way to keep one algorithm on both backends.
- **DC handled by `P0·A·P0`** (project both sides of `apply_A`, zero `M⁻¹`'s DC block);
  output-only projection breaks symmetry, neither breaks GPU convergence.
GPU validated: `cupy-cuda13x` 14.1.1 in `.venv`, GPU==CPU corr>0.9999. CPU matches the
retired L-BFGS to corr=1.00000 in 54 CG iters. Remaining: `[gpu]` extra, Phase-4 perf,
PIV torch removal (TODO TASK 4).

### Follow-up fix (2026-07-08): λ cross-branch consistency
Matching the L-BFGS closure (data term `/denom`, Tikhonov `λ/N`) reproduced L-BFGS
faithfully — but L-BFGS itself never matched FTTC. `λ` is the *shared* `regularization`
dial, yet the β=0 branch (`_solve_closed_form`, = FTTC) uses `λ²‖t‖²` while the β>0
branch used linear `λ/N`, so the same dial regularized differently across the
confinement switch (~125× at λ=1e-2; the λ=1e-4 default sat in the accidental
crossover band, which is why it read as fine). Fix: identity-term coefficient →
`λ²·(E·T0)²/denom`, making β=0,γ=0 reproduce `_solve_closed_form` at the same λ on
every frame; β/γ keep `1/N`. Golden regenerated at the corrected operating point
(λ 1e-4→1e-6, same corr=0.9998 recovery of `t_true`). New guard test; 9 tests green.
