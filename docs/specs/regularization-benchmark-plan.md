# Spec: regularization strategy benchmark plan

Status: proposed · Scope: `_validation/benchmark_TFM/` + the traction backends
(`napariTFM/backend/fttc.py`, `forward_tfm.py`) · Pairs with TASK 2 in `TODO.md`

> Revised after an external applied-maths review. Corrections from that review are
> called out inline as **[review]**; speculative refinements are marked **[optional]**.

## Purpose

We have several candidate regularization strategies (L2 Tikhonov at various orders,
L1/sparsity, elastic net, TV, the support/localization prior) and several λ-selection
methods (fixed, GCV, L-curve, discrepancy, Bayesian evidence). We have a synthetic-cell
benchmark with **ground-truth tractions** across conditions. This spec organizes *what to
try, in what order, and why*, from first principles, so the results are interpretable
rather than a hyperparameter fishing expedition.

The deliverable is **not** a single "best regularizer" — it is a **(condition → best
method) decision table** plus the resulting production defaults.

## Principle 1 — separate the two orthogonal axes; use ground truth to decouple them

Every strategy is a point in two independent spaces, and confounding them is the classic
mistake (a comparison where each method uses a *different* λ measures the selectors, not
the penalties):

- **Axis A — penalty functional `R(t)`:** L2 (0th/1st/2nd order), L1, elastic net, TV,
  support/localization prior.
- **Axis B — strength selector λ:** fixed, GCV, L-curve, discrepancy, Bayesian evidence.

Because the benchmark has ground truth, compute the **oracle λ** for every method (the λ
minimizing error vs ground truth). Then Phase 1 compares penalties at their oracle λ
(best-case fidelity, selector-independent) and Phase 2 measures how close each practical
selector gets. Phase 1 **must** precede Phase 2 — you cannot rank selectors before you know
the target they aim for. Total error factorizes as **(penalty's attainable floor) +
(selector's suboptimality gap)**; the natural Phase-2 statistic is oracle-normalized
efficiency `err_selector / err_oracle`.

### [review] Locate the oracle by *joint* search, not one-at-a-time ablation
The earlier draft's "ablate one hyperparameter at a time" is **wrong for finding the oracle**.
The error surface over `(log λ, log γ, log β)` has strong off-diagonal structure — λ and γ
both control effective shrinkage and trade off along a valley; β trades against W. Coordinate
descent lands on a ridge, not the joint optimum, which **systematically understates the
multi-parameter penalties** and biases the comparison toward single-parameter ones. Find the
oracle with a **joint coarse log-grid or Latin-hypercube over `(λ, γ, β)` + golden-section
refinement**. One-at-a-time ablation plots remain useful, but *only for interpretation* — never
for locating the oracle.

### [review] Report sensitivity, not just the oracle minimum
The oracle is unachievable, so a penalty with a sharp minimum can lose in practice to one with
a flat plateau slightly above. Per penalty and condition, report: (i) the full error-vs-`log λ`
curve; (ii) the **tolerance width** — decades of λ for which `err(λ) ≤ 1.1·err(λ*)`; (iii) the
*selector-achieved* error as the **decision** statistic, with oracle error only as the reference.
Also: the oracle is metric-dependent (four metrics → four oracle λ's). **Designate one primary
scalar metric** for defining the oracle — weighted relative L2 on the zero-mean-projected field
is the natural choice — and report the others at that λ.

## Principle 2 — diagnose bias explicitly, don't just infer it from a metric

Each penalty trades variance for bias with a characteristic signature. The Sabass metrics in
`validate_TFM.py` are the headline instruments:

| Penalty          | Bias signature                                          | Metric that exposes it |
|------------------|--------------------------------------------------------|------------------------|
| L2, 0th order    | shrinks all forces → underestimates peaks, spreads     | DTM, DTMS              |
| L2, gradient     | blurs edges                                            | DTA, correlation       |
| L1 / sparsity    | thresholds small forces to zero → kills weak real ones | DTMS, DTM on weak FAs  |
| Support prior    | ~zero bias inside mask, removes off-mask variance      | DTMS                   |

**[review] Add an explicit Monte-Carlo bias–variance decomposition as the diagnostic layer.**
"Bias → exposing metric" is heuristic and cannot separate a *biased* method from a
*high-variance* one on the same metric. At fixed `f*` and condition, run `n` noise realizations,
form the mean recovery `f̄`, and decompose `MSE = ‖f̄−f*‖² + E‖f−f̄‖²`, **resolved radially in
Fourier bands and spatially in/out of support**. This directly displays each signature (ridge:
bias at all k; gradient: bias growing with k; support: bias off-support; L1: threshold bias at
small `|f*|`) and diagnoses *why* a headline metric moved. Cost `n×` solves — cheap for the
quadratic models with warm-started PCG.

**[review] Two metric defects to fix:**
- Normalized correlation is **scale-invariant** and therefore blind to ridge shrinkage — only
  admissible if the magnitude metric is *always* reported alongside it.
- The angular-deviation metric is undefined where `|f*| ≈ 0` — weight it by `|f*|` or restrict
  it to the support.
- **[review] Nullspace convention:** since `mean(f)` is unobservable (`Ĝ(0)=0`), compute *all*
  metrics on **zero-mean-projected** `f` and `f*` (or report the DC error separately); otherwise
  methods are scored on an arbitrary convention. **[optional]** Also consider splitting error into
  well-observed vs weakly-observed spectral subspaces (by `σ(k)` relative to noise) — error in the
  latter measures the *prior*, not the method.

## Principle 3 — pre-register the predictions (makes the benchmark falsifiable)

Predict the winners from first principles *before* running, so a failed prediction is a finding:

| Condition                  | Expected winner              | Why (first principles) |
|----------------------------|------------------------------|------------------------|
| High noise                 | Support prior + higher-order L2 | noise lives at high-k; support kills off-cell haze, smoothing kills in-cell noise |
| Sparse discrete adhesions  | group-L1 / elastic net       | matches true sparse support; peaks preserved |
| Extended/smooth tractions  | L2 (L1 *hurts*)              | L1 over-sparsifies a genuinely delocalized field |
| Mask available             | Support/localization prior   | encodes highest-confidence physics — zero traction off-cell |
| Low bead density           | strongest *correct* prior    | weak data term → answer is prior-dominated |

## Phased priority list

Ordering rule: **measure before optimizing; cheapest-and-highest-confidence physics before
expensive-and-conditional methods; automate only survivors; research frontier last.**

### [review] Phase 0 — screen, then instrument
Two cheap things before the graded work:
1. **Design-of-experiments screen.** Quadratic solves are seconds, so run *all* quadratic methods
   on a coarse full-factorial (or fractional-factorial) grid over the five condition axes at coarse
   λ grids. This yields a **condition-sensitivity map** for near-zero cost — which cells actually
   *discriminate* methods. Spend the expensive L1/TV budget only on discriminative cells. (The
   cost-ordered phasing below conflates cheapness with informativeness; this screen fixes that.)
2. **Instrument the harness.** Lock the metric set (Sabass four + strain energy + a dedicated
   localization/background metric + **wall-clock**, a first-class axis). Build the joint oracle-λ
   search (Principle 1). Place the two *existing* methods as baselines everything must beat: plain
   FTTC (L2 + GCV, `fttc.py`) and the confined L2 solver (`forward_tfm.py`).

### Phase 1 — penalty sweep at oracle λ (ordered by confidence-of-physics × cheapness)
1. **Support prior (L2, already built) vs plain FTTC.** *First:* highest-confidence prior (a cell
   exerts zero traction where there is no cell), encodes side info you often possess (the
   footprint), already coded, targets the largest error source in unconstrained FTTC (background
   haze). Highest ROI, lowest risk, zero new code.
2. **Order of the smoothness penalty (0th / 1st / 2nd order Tikhonov).** *Cheap* sweep,
   near-closed-form; measures the natural smoothness scale of the ground truth, needed before
   tuning anything with an L2 component. Informs step 3.
3. **L1 / sparsity + elastic net.** *Third:* iterative/expensive (needs the PCG-inside-FISTA/ADMM
   engine from `forward-solver-pcg.md`), and value is *conditional on force geometry* — only
   interpretable once step 2 fixes the smoothness scale.
   - **[review] Convexity:** L1 + a convex quadratic is **convex — there are no local minima**. The
     earlier draft's "non-convexity/local-minima" concern was wrong. The *real* issues are: (i)
     **non-uniqueness** (G's nullspace + no ridge → the L1 minimizer can be non-unique; elastic net
     restores strict convexity, which is why it's included); (ii) **solver-tolerance confounding** —
     a loosely-converged FISTA run looks like a bad *penalty*; fix per-solver optimality-gap
     tolerances and report them.
   - **[review] Use the *group / vectorial* variants.** For a 2-vector field, componentwise L1
     `Σ|f_x|+|f_y|` biases recovered vectors toward the coordinate axes, and componentwise TV does
     the same. The intended prior is **per-point group sparsity `Σ_r ‖f(r)‖₂`** (group lasso) and
     **vectorial/isotropic TV**. Test these as first-class methods, or "L1 loses on vector fields"
     will be an artifact of the wrong L1. (The angular-deviation metric will expose the
     axis-alignment artifact — a useful cross-check.)
4. **TV (vectorial).** *Last in Phase 1, and only if the benchmark includes patch-like /
   tissue-scale conditions.* For single cells with peaked adhesions, piecewise-constant is the
   wrong prior.

### Phase 2 — selector comparison, per penalty (only for Phase-1 survivors)
**[review] Most classical selectors are one-parameter constructs.** The L-curve and the
discrepancy principle select a single scalar and do **not** generalize to joint `(λ, γ, β)`
selection; only **Bayesian evidence maximization** (Gaussian marginal likelihood, jointly
optimizable over all three) and **GCV/SURE-type risk estimates** (jointly minimizable) scale to
the real model. Run Phase 2 **per penalty** (the best selector genuinely differs by penalty
class), not once for a single "winning" penalty. Concrete requirements:

- **Weighted GCV — whiten first.** With hat matrix `Ã_λ = W^{1/2}G(GᴴWG+R)⁻¹GᴴW^{1/2}`, minimize
  `V(λ) = n_eff⁻¹‖(I−Ã_λ)W^{1/2}u‖² / (n_eff⁻¹ tr(I−Ã_λ))²`, with `n_eff` = #points with `w>0`
  (masked points excluded from both numerator and trace). `W` breaks Fourier diagonality so the
  trace has no closed form: use **Hutchinson / Hutch++ stochastic trace estimation** (10–50
  Rademacher probes, each one PCG solve — warm-startable). The same stochastic-Lanczos-quadrature
  machinery gives `log det A` for the evidence. *(This reuses the solver spec's shared Fourier
  primitive verbatim.)*
- **Discrepancy with W:** the target residual is `E‖W^{1/2}η‖² = 2·Σᵢ wᵢσᵢ²`, **not** `nσ²`. Also
  test discrepancy with an *estimated* noise level (e.g. MAD of high-frequency residuals) — the
  known-σ version is unrealistically favorable on synthetic data.
- **GCV for L1/TV:** the estimator is no longer a fixed linear smoother; `df` must be the **Stein
  degrees of freedom** (# active groups for group lasso; # connected flat regions for vectorial TV,
  per Tibshirani–Taylor). The quadratic trace formula is simply wrong here.

### Phase 3 — graded localization prior (frontier)
The probability-image generalization of the binary support prior (`w_eff = λ + β·(1−p(r))`).
*Last:* strictly more general than the binary prior (step 1), worth it only if step 1 proves the
support prior's value, and the real test is whether graded beats binary under *realistic,
imperfect* probability maps (see the mask-misspecification arm below).

## [review] Failure-mode arms the earlier draft omitted (ranked by importance)

1. **Inverse crime — mandatory.** Generating `u` with the *same* discrete `G` used for inversion
   overstates every method, and the "operator conditioning" axis does **not** fix it (both sides
   still share the operator). Add a mismatch arm: synthesize on a 2–4× finer grid with the
   *continuum* kernel, downsample, and/or perturb the inversion symbol by a few percent. Regularizer
   rankings frequently reorder under mismatch.
2. **Mask misspecification — arguably the single most decision-relevant experiment, and absent.**
   The support prior's entire advantage is conditional on the mask. Sweep dilation / erosion /
   translation and false-positive/false-negative regions of `m`; report the **breakeven mask-error
   rate** at which `β>0` stops helping and starts hurting. Same for `W`'s trust region.
3. **Nullspace convention** — covered under Principle 2 (zero-mean-project all metrics).
4. **Noise-model mismatch** — correlated noise (GCV is known to undersmooth badly under
   correlation), heavy tails (breaks discrepancy), and `W` misspecified relative to true variances.
5. **Compute cost** as a reported dimension (iterations / wall time per method per condition) —
   L1/TV vs quadratic is partly a cost–accuracy trade.

## [review] Statistical protocol (fix before running anything)

- Multiple `f*` draws × multiple noise seeds per condition cell; **common random numbers** (same
  seeds across all methods and λ values) so comparisons are *paired*.
- Per-instance paired differences of **log-error**; summarize by median paired log-ratio with
  bootstrap CIs; paired Wilcoxon (or t on log-errors) per cell; **Benjamini–Hochberg FDR** across
  the condition grid.
- **[optional] Better:** a mixed model `log-error ~ method × condition + (1|instance)` — borrows
  strength across the grid and directly estimates the method×condition interactions the
  pre-registered predictions are about. **Effect sizes, not p-values, drive adoption.**

## Experimental hygiene

The confined solver has three coupled knobs (λ, γ, β) plus the data weight W. Locate oracles by
joint search (Principle 1); use one-at-a-time ablation *only* to interpret, never to optimize.

## Recommended starting point

Do **not** start by coding L1. Start with **Phase 0 (screen + instrument) + Phase 1 step 1**:
build the joint oracle-λ harness and settle rigorously *how much the support prior alone buys over
plain FTTC across conditions*. It is already implemented, the highest-confidence physics, and very
likely the single biggest win. It also gates the expensive Phase-1.3: if support + smoothness
already saturates the metrics, sparsity is a marginal refinement; if a large localization error
remains *inside* the mask, that is group-L1's signal to shine.

## Cross-references

- `docs/specs/forward-solver-pcg.md` — the PCG engine and, crucially, the **shared Fourier-diagonal
  primitive** that is simultaneously the preconditioner, the FISTA/ADMM prox for Phase-1.3 (L1/TV),
  and the stochastic trace/log-det probe for Phase-2 (weighted-GCV / evidence). Build it once.
- `TODO.md` TASK 2 — the fair TVL1+FTTC vs one-shot benchmark this plan extends; "tune each method
  at its own best" is the same principle as oracle-λ here.
