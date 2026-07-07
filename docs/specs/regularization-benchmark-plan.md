# Spec: regularization strategy benchmark plan

Status: proposed · Scope: `_validation/benchmark_TFM/` + the traction backends
(`napariTFM/backend/fttc.py`, `forward_tfm.py`) · Pairs with TASK 2 in `TODO.md`

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
minimizing error vs ground truth, via a sweep). Then:

- **Phase 1 compares penalties at their oracle λ** — pure question: which prior matches the
  physics?
- **Phase 2 compares selectors against that oracle** — pure question: how close does a
  data-driven selector get without peeking at ground truth?

Phase 1 **must** precede Phase 2: you cannot rank selectors before you know the target they
aim for. This is a hard dependency.

## Principle 2 — each penalty has a bias signature; pick metrics that expose it

Every regularizer trades variance (noise amplification) for bias, with a characteristic
signature. The Sabass metrics in `validate_TFM.py` are the discriminating instruments:

| Penalty          | Bias signature                                          | Metric that exposes it |
|------------------|--------------------------------------------------------|------------------------|
| L2, 0th order    | shrinks all forces → underestimates peaks, spreads     | DTM, DTMS              |
| L2, gradient     | blurs edges                                            | DTA, correlation       |
| L1 / sparsity    | thresholds small forces to zero → kills weak real ones | DTMS, DTM on weak FAs  |
| Support prior    | ~zero bias inside mask, removes off-mask variance      | DTMS                   |

Metric set: the Sabass four (correlation, DTM, DTMS, DTA) + strain energy + a dedicated
localization/background metric + **wall-clock** (a first-class axis, not an afterthought —
it is the deployment tie-breaker).

## Principle 3 — pre-register the predictions (makes the benchmark falsifiable)

Predict the winners from first principles *before* running, so a failed prediction is a
finding, not a shrug:

| Condition                  | Expected winner              | Why (first principles) |
|----------------------------|------------------------------|------------------------|
| High noise                 | Support prior + higher-order L2 | noise lives at high-k; support kills off-cell haze, smoothing kills in-cell noise |
| Sparse discrete adhesions  | L1 / elastic net             | matches true sparse support; peaks preserved |
| Extended/smooth tractions  | L2 (L1 *hurts*)              | L1 over-sparsifies a genuinely delocalized field |
| Mask available             | Support/localization prior   | encodes highest-confidence physics — zero traction off-cell |
| Low bead density           | strongest *correct* prior    | weak data term → answer is prior-dominated |

## Phased priority list

Ordering rule: **measure before optimizing; cheapest-and-highest-confidence physics before
expensive-and-conditional methods; automate only survivors; research frontier last.**

### Phase 0 — instrument the harness (do first; everything depends on it)
- Lock the metric set above.
- Build the **oracle-λ sweep** as reusable infrastructure.
- Place the two *existing* methods on the map as baselines everything must beat: plain FTTC
  (L2 + GCV, `fttc.py`) and the confined L2 solver (`forward_tfm.py`).
- Rationale: no ruler and no upper-bound reference → no interpretable comparison.

### Phase 1 — penalty sweep at oracle λ (ordered by confidence-of-physics × cheapness)
1. **Support prior (L2, already built) vs plain FTTC.** *First:* highest-confidence prior
   (a cell exerts zero traction where there is no cell), encodes side info you often possess
   (the footprint), already coded, targets the largest error source in unconstrained FTTC
   (background haze). Highest ROI, lowest risk, zero new code.
2. **Order of the smoothness penalty (0th / 1st / 2nd order Tikhonov).** *Cheap* sweep,
   near-closed-form; measures the natural smoothness scale of the ground truth, which you
   need before tuning anything with an L2 component. Informs step 3.
3. **L1 / sparsity + elastic net.** The big methodological fork, but *third* because it is
   iterative/expensive (needs the PCG-inside-FISTA/ADMM engine from
   `forward-solver-pcg.md`) and its value is *conditional on force geometry* — only
   interpretable once step 2 fixes the smoothness scale. Run L1 and elastic net together;
   elastic net should dominate or tie both and reveals how much pure sparsity is too much.
4. **TV.** *Last in Phase 1, and only if the benchmark includes patch-like / tissue-scale
   conditions.* For single cells with peaked adhesions, piecewise-constant is the wrong
   prior. Prioritize strictly by whether those conditions exist.

### Phase 2 — selector comparison, only for Phase-1 survivors
Rank selectors by how close they get to the oracle across conditions. Invest in **GCV**
(already implemented) and **Bayesian evidence maximization** (principled; gives λ +
uncertainty; extends to non-Gaussian priors). Run the **discrepancy principle** as a
*reference* (the synthetic benchmark hands you the true noise level). Do not deep-invest in
the L-curve. Rationale: automation only matters for deployable methods, and its whole job
is to approach the oracle already computed.

### Phase 3 — graded localization prior (frontier)
The probability-image generalization of the binary support prior
(`w_eff = λ + β·(1−p(r))`). *Last:* strictly more general than the binary prior (step 1),
worth it only if step 1 proves the support prior's value, and the real test is whether
graded beats binary under *realistic, imperfect* probability maps — which depends on
everything before it.

## Experimental hygiene — ablate one knob at a time

The confined solver has three coupled knobs (λ, γ smoothness, β support) plus the data
weight W. A joint sweep is combinatorially explosive and uninterpretable. **Fix the others
at defaults and sweep one**, following the priority order. You can only attribute error to a
cause you varied in isolation.

## Recommended starting point

Do **not** start by coding L1. Start with **Phase 0 + Phase 1 step 1**: build the oracle-λ
harness and settle rigorously *how much the support prior alone buys over plain FTTC across
conditions*. It is already implemented, the highest-confidence physics, and very likely the
single biggest win. That result also gates the expensive Phase-1.3: if support + smoothness
already saturates the metrics on your cell geometries, sparsity is a marginal refinement; if
a large localization error remains *inside* the mask, that is L1's signal to shine.

## Cross-references

- `docs/specs/forward-solver-pcg.md` — the PCG engine that makes Phase-1.3 (L1 via
  FISTA/ADMM) and the iterative sweeps cheap enough to run.
- `TODO.md` TASK 2 — the fair TVL1+FTTC vs one-shot benchmark this plan extends; the
  "tune each method at its own best" cardinal rule is the same principle as oracle-λ here.
