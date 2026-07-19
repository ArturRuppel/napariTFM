# Bayesian L2 and Elastic Net regularization

_Added 2026-07-19. Motivated by a comparison of napariTFM's regularization scheme with
Huang et al., "Traction force microscopy with optimized regularization and automated
Bayesian parameter selection for comparing cells", Sci. Rep. 9:539 (2019)._

## Why

Huang et al. benchmark the TFM inverse problem `u = Mf + s` across regularizers and
parameter-selection strategies. Two of their headline findings pointed at gaps in our
scheme:

1. **Elastic Net (L1 + L2) is the most accurate regularizer.** Pure L1 gives a clean
   background but *overshoots* the peak traction; pure L2 scales the peak correctly but
   leaves a noisy background. The elastic net combines both — clean background *and*
   reined-in peak. We had pure group-L1 (`forward_l1`) and pure L2 (FTTC / `forward_tfm`),
   but no combined solve.
2. **Classical automatic λ selection (L-curve, GCV) is unreliable at high noise**, which
   biases cross-condition and time-series comparisons. The paper's fix is to pick λ by
   maximizing the Bayesian *evidence* (BL2 / ABL2). Our only automatic selector was GCV —
   exactly the family the paper flags.

This note records the two additions that close those gaps. Both are opt-in and leave the
existing defaults untouched.

## 1. Elastic Net — `l2_ridge` on the group-L1 solver

`forward_l1.py` already minimizes `½‖W·(G·t − u)‖²/denom + λ₁·Σ‖t[:,pixel]‖₂` by FISTA.
The elastic net just adds a global L2 ridge:

    + ½ λ₂ ‖t‖²

whose gradient is `λ₂·t`. Implementation notes:

- **Dial (`l2_ridge`, 0..1):** a scene-independent *fraction* of the **median** per-mode
  data curvature — deliberately *not* the max (`l_data`, which the Lipschitz constant
  uses). The Boussinesq spectrum decays steeply (the max is ~100× the median on our test
  fields), and the traction peak lives in the low-gain high-k modes, so scaling the ridge
  to the max would crush the peak instead of gently reining in the L1 overshoot. Median
  scaling keeps the useful band at ~0.1..1 with a gentle, monotone effect.
- **λ₁_max is unchanged.** The ridge vanishes at `t = 0`, so the per-pixel gradient at
  zero — which sets λ₁_max, the scale of the sparsity dial — is untouched. The two dials
  stay orthogonal; `l1_sparsity = 1` still empties the field regardless of `l2_ridge`.
- **Stability:** `λ₂` is a constant added curvature, so it folds into the FISTA Lipschitz
  constant / step (`L = l_data + pen.max() + λ₂`) exactly like the soft-support penalty.
- `l2_ridge = 0` reproduces the pure-L1 solve bit-for-bit.

Locked by `tests/test_forward_l1_elastic_net.py` (zero-ridge identity, monotone
peak/energy shrinkage, λ₁_max invariance).

## 2. Bayesian L2 — `bayesian_l2` on the plain-FTTC path

`bayesian_l2.py` picks the Tikhonov λ by maximizing the marginal likelihood (evidence) of
the displacement, over the *same* Fourier-SVD blocks GCV already consumes
(`FTTC._svd_block`) — so it is a drop-in swap for the λ search, no new linear algebra.

With a Gaussian traction prior (variance `1/α`) and Gaussian noise (variance `1/β`), the
MAP traction is the L2 solution with `λ = α/β`. Rather than the coupled α/β MacKay
fixed-point (which is **unstable** here — the operator's many near-null noise modes run λ
off to ∞, the boundary extremum the paper itself warns of for ABL2), we maximize the
log-evidence **directly in 1-D over λ**. In the resolved subspace (`s > 0`, m modes):

    BL2  (β pinned):   ℓ(λ) = −½ β λ Σ dᵢ²/(sᵢ²+λ) + (m/2)log λ − ½ Σ log(λ+sᵢ²)
    ABL2 (β profiled): ℓ(λ) = −(m/2) log Σ dᵢ²/(sᵢ²+λ) − ½ Σ log(λ+sᵢ²)

(the ABL2 form drops out by maximizing over β in closed form, `β* = m/(λ Σ dᵢ²/(sᵢ²+λ))`).
Maximized on a log-λ grid bracketed by the singular-value spectrum, then refined by a
bounded scalar optimizer. Null/DC modes are excluded from the sums — they carry no
traction information, and dropping the DC mode is exactly the mean-subtraction
("standardization") the paper applies before inference.

**BL2 vs ABL2.** The paper recommends BL2 (noise measured, one-parameter search) as the
more robust variant, and our experiments agree: ABL2's profiled objective is dominated by
the log-determinant of the many tiny singular values and drives λ→0 (no regularization).
So we run **BL2 whenever a noise estimate is available, which is always**:

- **Noise estimate** (`estimate_noise_variance`): a robust **MAD high-pass** (Laplacian
  convolution → MAD → σ). Restricted to the cell exterior when a mask is loaded (the
  paper's "far from any cell"), else over the whole field. Using a high-pass rather than a
  raw far-field variance is what lets it use the *near* exterior: the substrate-deformation
  halo just outside a tight mask is real signal, but it is *smooth*, so the Laplacian
  annihilates it and only noise survives. Recovers injected noise to within ~10% on
  synthetic blobs, masked or maskless.
- The real-space per-component variance `σ_r²` is converted to the per-Fourier-coefficient
  variance `N·σ_r²` (N = grid points; unnormalized FFT Parseval factor, preserved by the
  unitary per-mode SVD) to pin `β = 1/(N·σ_r²)`.
- ABL2 (`noise_var = None`) remains as the last-ditch fallback only if noise estimation
  fails outright.

**Precedence:** `bayesian_l2` > `auto_gcv` > manual `regularization`. Under either
automatic selector the manual λ is ignored (validation exempts it, as it already did for
GCV). The returned value is `√(α/β)` because the force path applies `regularization**2`.

Locked by `tests/test_bayesian_l2.py` (λ grows with noise and sits in the GCV ballpark,
noise recovery masked/maskless, FTTC override, validation exemption, degenerate-input
guard).

## What was *not* done

- **ABL2 as a user-facing mode.** Given its instability on this operator, it is an internal
  fallback only, not a dial.
- **Proximal-gradient (wavelet) methods** from the paper — out of scope; the elastic net
  was the paper's accuracy winner.
- **Per-frame GCV→Bayesian for the time-series default.** `bayesian_l2` already runs
  per-frame (like `auto_gcv`); making it the shipped default is a separate call.
