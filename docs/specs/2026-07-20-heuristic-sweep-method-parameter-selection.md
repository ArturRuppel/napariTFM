# Choosing displacement method and regularization by regime

**Status:** analysis complete (2026-07-20)
**Scope:** how to pick the displacement method (PIV, iLK, FFD) and the force-inversion
regularization (L1 sparsity, L2 ridge) for a given measurement, grounded in a 480-scene
synthetic sweep with analytic ground truth. Extends the displacement-method decision in
[displacement-method-selection.md](./displacement-method-selection.md) onto the two axes that
benchmark left unswept: source geometry and image SNR.

---

## One line

There is no best method. The winner is a function of three things the operator can read off a
sample before analysis: how broad the force source is, how large the displacement is, and how
good the images are. **PIV** is the robust default, **FFD** takes broad or noisy data, **iLK**
holds a thin niche on large soft sources at sub-pixel motion. Regularization is almost pure L1,
and its strength tracks how hard the recovery is.

## The governing idea

Every result below refers back to one map: a grid of source **footprint** against peak
**displacement**. A traction measurement lands in one cell of that grid, and the cell decides
everything: whether the force is recoverable at all, which method recovers it best, and what
regularization that method wants. Image quality does not move a measurement to a better cell; it
decides how far toward the hard cells the recoverable region reaches.

## The sweep

The sweep replaces the three opaque noise points of the earlier benchmark with a generative
ladder that has clean analytic ground truth on a common grid. Each scene is a balanced,
contractile traction dipole of known footprint, magnitude, and axis, forward-projected through
the Boussinesq Green's operator to a ground-truth displacement, then used to warp a real
synthetic bead frame. Reference and deformed frames are a known registration jitter apart, so
native camera noise rides along while the ground truth stays exact.

The grid is large:

- **30 regimes** per imaging condition: 5 footprints (0.1 to 5.0 µm) crossed with 6 peak
  displacements (0.5 to 50 px).
- **16 imaging conditions**: 8 synthetic bead stacks spanning bead density, numerical aperture,
  and exposure, each at 2 registration-jitter levels.
- **3 displacement methods** per scene, each swept over its own resolution knob, with a
  convergence ladder on top.
- **72 regularization points** per displacement field: 8 L1 sparsity values crossed with 9 L2
  ridge values.

Scoring is the Sabass composite `J = |DTM| + DTMS + DTA/45`: signed magnitude error on the
adhesion, spurious background force, and angular error, the three modes the L1 and L2 knobs
trade off. Lower `J` is better. Every figure reduces to each method's best-`J` operating point
per scene, then aggregates across the 16 imaging conditions.

## Which method wins where

![Three heatmaps of median best-J over footprint (rows, 0.1 to 5 µm) against peak displacement (columns, 0.5 to 50 px), one per method, plus a fourth panel coloring each cell by the winning method with its margin to the runner-up and win-fraction. PIV holds the small-footprint column and the compact-source band; FFD holds the broad-source, low-to-mid-displacement cells where the absolute best J values live (0.04 to 0.08); iLK holds four cells at large footprint and low displacement. Every method fails together in the small-footprint, low-displacement corner where J exceeds 1.](../images/heuristic-sweep-competence.png)

Read the fourth panel as three jobs, not a ranking. Across all 480 scenes:

| method | outright wins | mean rank | within 10% of best | regime cells won |
|---|---|---|---|---|
| **PIV** (cross-correlation) | 238 / 480 | 1.70 | 67% | 16 / 30 |
| **FFD** (B-spline) | 153 / 480 | 1.97 | 53% | 10 / 30 |
| **iLK** (optical flow) | 89 / 480 | 2.33 | 37% | 4 / 30 |

- **Compact sources (0.7 µm) go to PIV.** This is the clean-recovery band: `J` bottoms out near
  0.20 and PIV wins 11 of 16 conditions per cell. The correlation window integrates enough signal
  to localize a tight source without smearing it.
- **Broad sources (1.9 µm) at low-to-mid displacement go to FFD.** The lowest `J` in the whole
  sweep lives here, 0.04 to 0.08, roughly 1.5 to 2 times better than PIV. A B-spline's built-in
  smoothness is the right prior for a broad smooth blob.
- **Very broad soft sources (5 µm) at sub-pixel displacement go to iLK, thinly.** iLK preserves
  peak magnitude on a soft gradient where the others over-smooth, but its margins are 0.02 to 0.08
  and it holds only 44 to 81% of conditions. Real, and second-order.
- **Small sources (below ~0.3 µm, ~2 px) fail for everyone.** `J` exceeds 0.9 across that whole
  band. This is the resolution and noise floor: below it, no method produces a trustworthy traction
  map, and the choice between them is a choice between bad answers.

The margins matter as much as the winners. PIV is within 10% of the best method two-thirds of the
time: it is rarely the winner by a wide gap, and rarely far behind. If one method must serve a
whole pipeline, it is PIV.

## Regularization: L1 does the work

![Three panels over the same footprint-by-displacement grid. The first colors each cell by the winning method and prints its recommended resolution and L1. The second is a heatmap of the winner's L1 sparsity: light values (0.02 to 0.07) fill the resolvable regime, saturating at the grid ceiling of 0.40 in the small-footprint, low-displacement corner. The third is the winner's L2 ridge, near zero across almost the entire grid.](../images/heuristic-sweep-parameters.png)

The inversion wants **L1 sparsity with the ridge switched off**. Two facts set the strength:

- **In the resolvable regime, L1 is light: 0.02 to 0.07, L2 is 0.** Pure sparsity, matching the
  earlier regularization benchmark's `l1` pinning near 0.11 and confirming that the ridge term of
  elastic-net earns little here.
- **In the hard corner, L1 saturates at the grid ceiling.** The sweep wanted more sparsity than the
  grid offered (0.40) exactly where recovery is hardest. The grid is truncated where it matters
  most: a follow-up sweep should extend the L1 axis upward before trusting any recommendation in the
  small-footprint cells.

Resolution tracks the source. PIV wants large correlation windows (24 to 32 px) in the resolvable
regime, dropping to 8 px only to localize the smallest sources. FFD wants mid-to-large B-spline
spacing (18 to 24). iLK wants a small radius (5 to 7).

### The cost of a wrong L1

L1 is a free parameter the operator sets. Two questions decide whether that is a burden: how much a
wrong value costs, and whether the parameter-free Bayesian-L2 alternative is worth escaping the
choice. To answer both, hold the displacement input fixed (PIV window 24) so only the regularization
varies, then compare four strategies on identical fields: the per-scene *oracle* L1, a single
*fixed* L1 applied everywhere, the oracle grid L2 ridge, and parameter-free Bayesian-L2.

![Two panels. Left: J divided by each scene's own best, against L1 sparsity on a log axis, one faint line per scene plus a bold median. The median is flat near 1.2 from L1 0.02 to 0.11, then climbs steeply to 3.6 at L1 0.4; a horizontal line marks Bayesian-L2 at 3.2 times the oracle, level with the worst L1. Right: boxplots of Sabass J for four strategies, medians 0.15 (oracle L1), 0.21 (fixed L1), 0.21 (oracle grid L2), and 0.45 (Bayesian-L2, far higher).](../images/heuristic-sweep-regularization-sensitivity.png)

**The L1 basin is asymmetric, with a wide safe plateau.** Any L1 from 0.02 to 0.11 lands within 25%
of the per-scene optimum (median penalty 1.14 to 1.25 times). Above 0.11 the penalty climbs fast:
1.75 times at 0.17, 2.55 times at 0.26, 3.56 times at 0.40. The mechanism is one-sided: too little
sparsity leaves a little background noise, which the composite `J` barely charges for, while too
much sparsity thresholds away real force on the adhesion, which it charges for heavily. The rule
follows: **err low.** Under-regularizing is nearly free, over-regularizing is the expensive mistake.

**Parameter-free Bayesian-L2 costs far more than any sensible L1.** Median `J` by strategy on the
resolvable regime:

| strategy | median J | vs oracle L1 |
|---|---|---|
| oracle per-scene L1 | 0.15 | 1.0× |
| a wrong-but-sensible fixed L1 | 0.21 | 1.4× |
| oracle grid L2 ridge | 0.21 | 1.4× |
| Bayesian-L2 (no tuning) | 0.45 | 3.2× |

Bayesian-L2 lands at 3.2 times the tuned L1 and 2.4 times even a fixed L1, and it never once beat a
fixed L1 across 144 scenes. Read against the basin: BL2 sits level with the *worst* L1 on the grid.
The two mistakes are not the same size. Choosing L1 wrong but sensibly costs about 20%; dropping to
the parameter-free path costs 140 to 220%. **The mistake of using Bayesian-L2 instead of L1 is
roughly ten times larger than the mistake of a suboptimal L1.** L1's tuning is cheap insurance.

This verdict is specific to the target. The benchmark source is a compact dipole, and L1's sparsity
prior matches a localized source by construction while Bayesian-L2's smoothness prior does not: its
coarse traction mesh cannot represent sharp poles, so it under-recovers peak magnitude and `J`
punishes that. The result transfers to real focal adhesions, which are compact. On broad, diffuse
cell fields the gap narrows to about 1.3 times but does not reverse: sparsity still wins, only by
less (see [Diffuse fields](#diffuse-fields-realistic-cells)). The comparison also bundles the
regularizer with the mesh resolution. The defensible
claim is narrow and useful: for compact sources scored on magnitude fidelity, sparsity beats
smoothness by enough that L1 is worth tuning, and L1 forgives any choice made on the low side.

## What winning looks like

![Four scenes, one per row, each showing the ground-truth traction beside the best-J recovery of PIV, iLK, and FFD, with the winner boxed. Row 1 (broad source, mid displacement): FFD gives two clean discs at J=0.04 while PIV and iLK ring with speckle. Row 2 (large soft source, sub-pixel displacement): iLK wins at J=0.11 with a blocky, noisy field while PIV and FFD give smoother but higher-J blobs. Row 3 (compact source): PIV nails both poles at J=0.08 while the others weaken or scatter. Row 4 (small footprint): all three are noise, PIV least-bad at J=0.75.](../images/heuristic-sweep-examples.png)

The examples make one honest point the table cannot: winning on `J` is not the same as looking
right. In row 2 iLK wins the metric by preserving peak magnitude, yet its field is blockier and
uglier than the smooth PIV and FFD blobs it beats. The composite `J` rewards magnitude fidelity on
the adhesion over visual smoothness off it. Trust the number, but know what it rewards.

## How imaging parameters set quality

![Three panels. Left: median best-J per imaging condition as horizontal bars colored by exposure, with the three exposure-4.0 conditions cleanest and the three exposure-0.25 conditions worst. Middle: a scatter of median best-J against bead-image NCC, error collapsing as NCC rises toward 1, exposure groups falling into separate clusters. Right: a bar chart of driver strength, image NCC at -0.63 and exposure at -0.59 dominating, density, NA, and jitter all below 0.15.](../images/heuristic-sweep-imaging-drivers.png)

Exposure is the one imaging lever that matters. Ranked by Spearman correlation with achievable `J`:

| parameter | ρ | verdict |
|---|---|---|
| image NCC (SNR proxy) | −0.63 | dominant |
| **exposure** | −0.59 | dominant, and the cause of the NCC |
| density | −0.15 | second-order |
| NA | −0.14 | second-order |
| jitter | +0.14 | second-order |

The three weak drivers are each informative:

- **Density barely moves quality.** For a localized dipole, once bead count clears a minimum, more
  beads add no information: the signal is spatially concentrated. This flips for confluent or
  extended force fields, so read it as a property of the point-source test, not of TFM.
- **NA barely moves quality on its own.** A higher NA sharpens the point-spread function but starves
  photons per pixel, and the two effects cancel. NA helps only when the photons exist to use it.
- **Jitter is nearly free.** Raising registration error from 0.067 to 0.2 px costs almost nothing in
  the resolvable regime.

## Where imaging quality actually pays off

![Two footprint-by-displacement heatmaps of median best-J, one for high exposure (4.0) and one for low exposure (0.25), plus a difference panel. The high and low maps look similar in the core resolvable cells and the dead corner. The difference panel lights up a diagonal band at the envelope edge (footprint 0.7 to 1.9 µm, displacement 1 to 8 px) where low exposure costs up to +0.60 in J, and is near zero everywhere else.](../images/heuristic-sweep-imaging-envelope.png)

The counterintuitive result: on the best images, the best-achievable `J` in the core resolvable
regime barely improves. Across the whole quality ladder it moves from 0.097 to 0.156, a factor of
1.6. Tuning parameters per image compensates for most of the exposure deficit where the source is
already recoverable.

Imaging quality pays off at the **envelope edge** instead. The difference panel is near zero in the
already-good cells and near zero in the dead corner, and lights up a diagonal band at the margin
where low exposure costs up to +0.60 in `J`. Good images do not lower the floor of measurements you
could already make: they extend the recoverable region into smaller and weaker sources.

State it as a rule: **exposure does not make good measurements better, it makes marginal
measurements possible.** For broad sources with healthy displacements, do not spend photons or
phototoxicity chasing SNR. For small or weak adhesions, exposure is the most important knob there
is.

Image quality also shifts the method choice. Winner share by exposure, across all regimes:

| exposure | PIV | iLK | FFD |
|---|---|---|---|
| high (4.0) | 60% | 15% | 23% |
| mid (1.0) | 41% | 18% | 40% |
| low (0.25) | 43% | 21% | 34% |

On pristine images PIV dominates outright: sharp localization wins when there is no noise to fight.
As images degrade, FFD's share rises toward 40%, because a smoothness prior is a denoiser and earns
its keep when SNR is poor. This sharpens the earlier decision: iLK was carried as the low-SNR hedge
on the strength of a literature report, but on this sweep the low-SNR beneficiary is FFD. iLK's
measured niche is elsewhere, on large soft sources at sub-pixel motion. The caveat is that this
sweep varies SNR through exposure on a balanced dipole, not through the camera-noise axis of the
warped-real-bead benchmark, so treat it as a second, agreeing line of evidence rather than a
retraction.

## Diffuse fields: realistic cells

The dipole grid isolates one localized source. A real cell is the opposite: fifty to a hundred
contractile stress fibres overlapping into a diffuse, centripetal field. The regularization section
earned a caveat there, that sparsity's edge might narrow when smoothness becomes the right prior. This
run settles the part that governs the defaults.

The scenes are the benchmarkTFM synth cells: four real cell outlines carrying 16 to 82 fitted stress
fibres, their traction fitted to real bead measurements. That traction is the ground truth. It is
forward-projected through this pipeline's own Green's operator (which reproduces the fitted
displacement to a cosine of 0.9996), scaled to a ladder of strengths, and used to rewarp the single
best-imaging bead pair, scenario 6. Everything else matches the dipole run: substrate, pixel size,
imaging. The one variable is the field, one localized source becoming a whole cell. The Sabass `J` is
undefined here, because a centripetal field's mean traction vector cancels, so recovery is ranked on
whole-field nRMSE.

![Two-by-two panel. A: a grid of the four cells against six strengths, each cell coloured by the
winning method, PIV filling the middle strengths, FFD the rightmost column, iLK two cells at the
lowest strength. B: best nRMSE against strength for the three methods, each a U-shaped curve bottoming
near 0.5 at 3 to 8 px, PIV lowest through the middle. C: nRMSE relative to the best against L1, a flat
basin under 1.02 across L1 0.07 to 0.17, the dipole plateau shaded just left of it. D: median optimal
L1 and L2 against strength, both high at the 0.5 px noise floor and relaxing through the useful
range.](../images/heuristic-sweep-cells-competence.png)

Three findings, all continuous with the dipole run.

**Method competence keeps its shape.** PIV wins the useful window (16 of 24 scenes), FFD takes the
breakdown end where displacements pass ~20 px and cross-correlation decorrelates (6 scenes), and iLK
holds only the noise floor (2 scenes, thin). The regime structure survives intact: PIV the generalist,
FFD the hedge under stress, iLK a narrow niche.

**Recovery is a U in strength.** Best nRMSE bottoms near 0.5 at a peak displacement of 3 to 8 px,
degrades into the registration-jitter floor below 1 px, and into decorrelation above 20 px. The floor
is high because a diffuse field carries fine focal-adhesion structure below the band-limited operator's
reach: the honest ceiling on a real cell is not a clean recovery, it is a smoothed one.

**The L1 heuristic transfers.** The safe plateau, where any L1 choice costs under 2% of the best
nRMSE, runs 0.07 to 0.17, sitting almost on top of the dipole plateau of 0.07 to 0.11 and nudged
slightly higher. The optimum still tracks SNR: at the noise floor both L1 and L2 ramp hard (L1 to
0.40, L2 to 16 and beyond), and through the useful range they relax to L1 near 0.11 with L2 below 1.
Sparsity does not cede to smoothness even here: the optimum keeps L1 in the plateau with only modest
ridge, so the worry that a diffuse field would want a smoothness prior does not show up in where the
knobs land.

![Three cell scenes, one per row, each showing the ground-truth fibre traction beside the best-nRMSE
recovery of PIV, iLK, and FFD, winner boxed. Rows 1 and 2 (mid strength): PIV wins, recovering the
ring of adhesions closest to truth at nRMSE 0.77 and 0.51. Row 3 (high strength): FFD wins at 0.79
where PIV and iLK trail. Every recovery is visibly dimmer than the ground truth, and every winning
panel uses L1 between 0.07 and 0.17.](../images/heuristic-sweep-cells-examples.png)

The examples put a face on it. At the sweet spot PIV recovers the ring of adhesions closest to the
fitted truth; at high strength FFD's smoothness holds the field together where PIV starts to break.
The recoveries are dimmer than the truth, the magnitude under-estimate the metric reports as a
negative bias: regularized TFM buys a clean background by shrinking the peaks.

### Does smoothness win on diffuse fields?

The dipole run found sparsity beats parameter-free Bayesian-L2 by two and a half to three times, and
flagged the diffuse field as the case where that might reverse: a whole cell is closer to smooth than
a pair of poles. Re-running the comparison on the cells, displacement fixed at PIV window 24 so only
the regularizer varies, settles it.

![Two panels. Left: nRMSE relative to each scene's best against L1 sparsity, a flat basin near 1.0
across the low-to-middle grid with a red Bayesian-L2 reference line drawn at 1.27x, far above the
basin. Right: a boxplot ladder of whole-field nRMSE by strategy across 12 cell scenes, oracle
per-scene L1 and best fixed L1 both at 0.75, oracle grid L2 at 0.83, parameter-free Bayesian-L2 worst
at 0.94.](../images/heuristic-sweep-cells-regularization.png)

The gap narrows but does not close. With no tuning, Bayesian-L2 lands at 1.27 times the tuned-L1
error, down from the 2.5 to 3 times it cost on dipoles: smoothness is genuinely more competitive when
the field is diffuse. It is still not competitive enough. It sits above the best fixed L1 across the
useful window, beats that fixed L1 in zero of twelve scenes, and even the elastic net's own tuned
ridge (0.83) beats parameter-free smoothness (0.94). The ordering holds on cells as on dipoles: tuned
sparsity, then tuned smoothness, then no tuning at all. The L1 basin stays flat and forgiving across
the low-to-middle of the grid, so the practical verdict is unchanged, only the margin shrinks: tune
L1 and it beats the parameter-free path on every field tested, dipole or cell.

The tuned elastic-net heuristic, plateau and SNR-tracking both, carries from one dipole to a whole
cell, and sparsity keeps its edge over smoothness the whole way.

### Does mask confinement earn its keep?

Everything above ran with no cell mask, on purpose: the sweep set the defaults, and the defaults have
to ship safe for the user who has not drawn a segmentation. That leaves one shipped feature untested.
The forward solver can confine traction to a cell mask, a soft off-mask penalty gated by a strength
dial, and the sweep never asked whether that prior is worth turning on. The diffuse cells are the only
fair place to ask, since a dipole's mask is just its two blobs and confining to it merely restates the
ground truth. Here the honest prior is the cell *outline*, the thing a user actually segments from
brightfield, which is genuinely looser than the traction: the fibres pull hardest at the periphery,
well inside the edge. Confining to the true traction support would be cheating, so that appears only
as an oracle ceiling. Displacement stays fixed at PIV window 24 and regularization at the shipped
default (L1 0.05, no ridge); the mask is the only thing that moves, its dial run from off to full.

![Two panels. Left: nRMSE relative to the no-mask baseline against the confinement dial, one curve per
strength band. The noise-floor band plunges to 0.55 by full strength; the useful and strong bands hug
1.0, dipping a couple of percent at most, each with a dotted GT-support oracle ceiling just below.
Right: mean fraction of recovered energy lying outside the cell falls from 0.24 to 0.07 as the dial
rises, while in-cell nRMSE stays flat near 0.84.](../images/heuristic-sweep-cells-confinement.png)

**It earns its keep, and the size of the win tracks SNR inversely.** At the noise floor (peak
displacement at or below 1.2 px) confinement cuts whole-field nRMSE by about 45%, from 4.0 to 2.5, and
helps in every scene. That headline number flatters it, though: both fields are above 1, meaning the
recovery is worse than predicting zero, so confinement is rescuing an already-hopeless field into a
merely-bad one. The operationally honest number is the useful window, where recovery is real: there
the gain is a steady ~5% (0.82 to 0.78), positive in every scene, and the loose cell outline captures
almost all of what the GT-support oracle could (0.78 against a 0.75 ceiling), so a tight mask buys
nothing extra. At high strength there is little exterior leak left to remove and the gain fades to ~2%.

**The mechanism is exactly background removal, and it never touches the cell.** The win comes entirely
from killing spurious traction in the bare substrate: mean off-cell energy falls from 0.24 to 0.07 as
the dial climbs. This is where the noise-floor payoff comes from, because a buried-signal field
inverts into rampant background garbage that the metric, whole-field, pays for in full. Meanwhile
in-cell error stays flat to within a couple percent, and across 120 scene-and-dial combinations the
error inside the outline rose by more than 1% in exactly none of them. The apron around the mask
protects real rim traction, so confinement is a one-sided lever: it removes exterior leak and leaves
the cell alone. That makes it safe to turn on whenever a mask exists.

The default stays off, because the no-mask user must not be handed a prior they did not ask for and the
useful-window gain is small. But the recommendation for the user who has a segmentation is clean:
switch confinement on. It is a noise-regime tool, worth the most exactly when the images are worst, and
it cannot hurt.

## The recipe

- **Default to PIV**, window ~24 px, L1 ~0.05, L2 = 0. It wins or ties about two-thirds of the time
  and is rarely far behind. On clean images this is not only safe, it is usually best.
- **Broad or diffuse sources at moderate displacement: switch to FFD**, spacing ~18 px, L1 ~0.05.
  Worth a real 1.5 to 2 times improvement in `J`.
- **Noisy images (low exposure, dim beads): reach for FFD sooner.** Its smoothing is denoising.
- **Large soft sources at sub-pixel displacement: iLK**, radius ~7, can edge the others, but the
  margin is thin and the field is blocky. Only worth it when squeezing the last few percent.
- **Have a cell mask? Turn on confinement.** It only removes traction from the bare substrate and
  never touches the cell, so it cannot hurt: ~5% in the useful window, far more at the noise floor.
  Off by default because the no-mask user must not inherit a prior they did not draw.
- **Sources below ~2 px footprint: expect failure from every method.** Use PIV with strong L1 and
  treat the magnitudes skeptically.
- **To resolve small or weak sources, raise exposure**, not density or NA. It is the only imaging
  parameter that extends the recoverable envelope.
- **On whole cells, the same defaults hold.** PIV through the useful window, FFD once displacements
  pass ~20 px, L1 in the 0.07 to 0.17 plateau: the heuristic carries from one dipole to a diffuse
  50-to-100-fibre field, so the per-regime starting points do not need a separate cell mode.

## Provenance

The run generated **1.4 GB** across 480 scenes: 960 synthetic bead frames (481 MB), 6,240 cached
displacement fields (725 MB), and 480 result shards holding **449,280 scored inversions** (89 MB).
Compute ran as a two-stage cross-partition pipeline on the Maestro cluster: GPU displacement caching
feeding a CPU force sweep through an element-wise `aftercorr` dependency, about 2.5 hours wall-clock
end to end.

The diffuse-cell run adds 24 scenes (4 benchmarkTFM synth cells x 6 strengths) under the isolated
condition `cell_s6j1`, rewarping the single best-imaging bead pair; its result shards carry the
whole-field metrics alongside the dipole schema. Same two-stage pipeline, minutes of wall-clock at 24
scenes.

Every figure here is regenerated from the result shards by a script in
`_validation/heuristic_sweep/`: `aggregate.py` (method competence, winners, parameter heuristics),
`imaging_quality.py` (imaging-parameter drivers, envelope), `compare_reg.py` (L1 sensitivity and the
Bayesian-L2 comparison, which re-runs the parameter-free solver on the cached fields),
`compare_methods.py` (the illustrative dipole recoveries), `cell_aggregate.py` (the diffuse-cell
competence and heuristic transfer), `cell_examples.py` (the cell recoveries), and `cell_compare_reg.py`
(the Bayesian-L2 comparison on cells, which re-runs the parameter-free solver on the cached fields).
The cells are staged by `make_cells.py`. The exact commands, the script-to-figure map, and the data-generation pipeline that
produces the shards are in the
[sweep README](../../_validation/heuristic_sweep/README.md#reproducing-the-report). The data itself
lives in the private stage directory, not this repo; the scripts take its path from `$STAGE`.

## Open

The L1 grid is truncated at 0.40 exactly where the hardest cells want more sparsity. Before any
small-footprint recommendation is trusted, extend the L1 axis upward and re-sweep the saturated
corner. The diffuse-cell run closes both the transfer question and the smoothness-versus-sparsity
question (sparsity still wins, by about 1.3 times), and leaves one thread: the cells were run only at
best imaging, so the SNR-tracking seen there is inferred from the strength axis, not a second imaging
ladder. With the L1 grid extended, the defaults above can ship as the UI's per-regime starting points,
the same role the tuning defaults play in
[displacement-method-selection.md](./displacement-method-selection.md).
