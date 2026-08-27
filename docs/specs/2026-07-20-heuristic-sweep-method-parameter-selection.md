# Choosing displacement method and regularization by regime

**Status:** analysis complete (rewritten 2026-07-24 against the rebuilt pipeline; the
regularization recommendation was revised the same day after re-scoring on criteria the ranked
objective does not contain — see [What the objective was hiding](#what-the-objective-was-hiding))
**Scope:** how to pick the displacement method (PIV, iLK, FFD) and the force-inversion
regularization for a given measurement, grounded in a 504-scene synthetic sweep with
analytic ground truth. Extends the displacement-method decision in
[displacement-method-selection.md](./displacement-method-selection.md) onto the two axes that
benchmark left unswept: source geometry and image SNR.

> **This supersedes the first (2026-07-20) version of this report.** That analysis ran on a
> 128-px working grid and searched a mixed elastic-net `l1 × l2` grid. Both changed: the
> pipeline now works natively at 512 (the downsample band-limited the traction so sharp
> adhesions were unrecoverable, and the rescale back to the scoring grid injected a
> progressive spatial offset), and the force stage now caches the *ceiling* of two pure
> paths instead of sampling a mixed grid. Numbers below replace the earlier ones wholesale;
> where a conclusion survived the rebuild it is called out, and where one did not, likewise.

---

## One line

There is no best method. The winner is a function of three things the operator can read off a
sample before analysis: how broad the force source is, how large the displacement is, and how
good the images are. **PIV** is the robust default, **iLK** takes broad sources at small
displacement, **FFD** takes broad sources at large displacement. For the force inversion, prefer
**smoothness (`FTTC + L2`) as the general-purpose default**: sparsity wins the sweep's ranked
objective, but that objective explicitly rewards a clean background, and once that term is
removed sparsity's accuracy edge is marginal while its traction *direction* is 2.5 to 3.5× worse.

## The governing idea

Every result below refers back to one map: a grid of source **footprint** against peak
**displacement**. A traction measurement lands in one cell of that grid, and the cell decides
everything: whether the force is recoverable at all, which method recovers it best, and what
regularization that method wants. Image quality does not move a measurement to a better cell; it
decides how far toward the hard cells the recoverable region reaches.

## The sweep

Each scene is a balanced, contractile traction dipole of known footprint, magnitude and axis,
forward-projected through the Boussinesq Green's operator to a ground-truth displacement, then
used to warp a real synthetic bead frame. Reference and deformed frames are a known registration
jitter apart, so native camera noise rides along while the ground truth stays exact. The pair is
always *balanced* — the operator zeroes the DC mode, so a net-force monopole would manufacture a
"heuristic" out of an artifact.

- **30 regimes** per imaging condition: 5 footprints (0.1 to 5.0 µm) crossed with 6 peak
  displacements (0.5 to 50 px).
- **16 imaging conditions**: 8 synthetic bead stacks spanning bead density, numerical aperture
  and exposure, each at 2 registration-jitter levels.
- **3 displacement methods** per scene, each over its own resolution ladder with a convergence
  ladder on top — 18 cached displacement fields per scene, 9,072 in total.
- **2 regularizer paths** per displacement field, each swept over its own grid: `FISTA + L1`
  over 8 sparsity values and `FTTC + L2` over 33 λ values. **371,952 scored inversions.**

**Everything reported here is a ceiling.** For each displacement field the regularization is
chosen by the *scored* objective against ground truth, so each number is the best that path
could do on that input — never the result of a lucky guess. That makes method comparisons fair
and it makes the tuning question separable: the basins in the regularization section are read
off the stored objective curves, not off a second search.

Scoring is the Sabass composite `J = |DTM| + DTMS + DTA/45`: signed magnitude error on the
adhesion, spurious background force, and angular error — the three modes the regularizers trade
off. Lower is better. Displacement error is never scored on its own; it does not propagate
uniformly through the DC-stripped, band-limited operator, so minimizing it can rank displacement
settings wrongly.

## Which method wins where

![Four panels over a grid of footprint (rows, 0.1 to 5 µm) against peak displacement (columns, 0.5 to 50 px). Three heatmaps of median best J, one per method, share a colour scale: all three are dark (J above 1) along the bottom 0.1 µm row and brighten toward the top-middle, PIV reaching 0.03 at footprint 1.88 µm and 7.9 px. A fourth panel colours each regime cell by its modal winning method with the share of imaging conditions that agree: iLK holds the top-left (large footprint, sub-pixel to few-pixel displacement), FFD holds two large-footprint cells at high displacement, and PIV holds everything else, usually at 88 to 100 percent agreement.](../images/heuristic-sweep-competence.png)

Read the fourth panel as three jobs, not a ranking. Across all 480 dipole scenes:

| method | outright wins | mean rank | within 10% of best | median J |
|---|---|---|---|---|
| **PIV** (cross-correlation) | 299 / 480 | 1.54 | 72% | 0.64 |
| **iLK** (optical flow) | 91 / 480 | 2.40 | 29% | 0.91 |
| **FFD** (B-spline) | 90 / 480 | 2.07 | 34% | 0.82 |

iLK and FFD split the remainder almost exactly evenly on wins, but they are not
interchangeable: FFD is the better *runner-up* (mean rank 2.07 against 2.40, within 10% of
best a third of the time against a quarter), so its wins come from a broad competence and
iLK's from a narrow speciality.

Those medians are dragged upward by regime cells where nothing works. Restricted to the
**core regime** — footprint ≥ 0.266 µm and peak displacement between 1.2 and 20 px, 256 scenes —
the three sit much closer together: PIV wins 161 with median `J` 0.220, FFD wins 54 at 0.229,
iLK wins 41 at 0.272.

- **PIV is the generalist.** It wins 62% of all scenes and is within 10% of the best method
  nearly three-quarters of the time. In the middle of the grid it wins 88 to 100% of imaging
  conditions per cell. If one method must serve a whole pipeline, it is PIV.
- **The best recovery in the sweep is PIV on a broad source at moderate displacement**:
  `J` = 0.03 at footprint 1.88 µm and 7.9 px, an essentially clean two-pole reconstruction.
- **Broad sources at small displacement go to iLK.** It takes the top-left block — 5 µm at
  0.5 to 3.2 px and 1.88 µm at 0.5 px — holding 50 to 88% of conditions. This is the niche the
  first report identified, and it survived the rebuild intact.
- **Broad sources at large displacement go to FFD**, which takes 5 µm at 7.9 and 19.9 px (94%
  of conditions at 19.9 px) plus a cell at 1.88 µm / 1.3 px. A B-spline's built-in smoothness
  holds a field together where cross-correlation windows start to decorrelate. At the extreme
  50 px column PIV takes the lead back, so this is a band rather than a half-plane.
- **Small sources fail for everyone.** The entire 0.1 µm row sits at `J` 1.16 to 2.44, and
  0.266 µm is only marginally better at 0.77 to 1.88. Below roughly 2 px of footprint no method
  produces a trustworthy traction map, and the choice between them is a choice between bad
  answers.

## Sparsity or smoothness?

![Four heatmaps over the same footprint-by-displacement grid. The first shows the winning configuration's resolution knob, mixing PIV windows, iLK radii and FFD spacings. The second, on a blue-to-red diverging scale, shows the share of conditions won by FISTA+L1: deep red at 88 to 100 percent across the whole resolvable region, fading to blue at 19 to 25 percent only in the smallest-footprint, smallest-displacement corner. The third shows the winner's oracle L1 sparsity, light at 0.02 to 0.11 through the resolvable band and saturating black at the 0.40 grid ceiling in the hard corner. The fourth shows the oracle FTTC λ, clustered between 3e-4 and 6e-3.](../images/heuristic-sweep-parameters.png)

**On the ranked objective, sparsity wins overwhelmingly — and that turns out to be mostly an
artifact of the objective.** `FISTA + L1` supplies the winning configuration in 87% of all dipole
scenes and 96% of the core regime. But `J` contains `DTMS`, spurious force in the ring around the
adhesion: a *direct* reward for a zero background, which group-L1 supplies by construction because
it thresholds whole regions to exactly zero. Re-rank the same configurations on criteria that
carry no background term and the verdict changes character entirely — see
[What the objective was hiding](#what-the-objective-was-hiding) below. Read the panels in this
section as "what the sweep's objective selected", not as "what you should use".

Where L2 does win on `J`, it is in the corner where nothing is recoverable — the 0.1 µm row at
sub-pixel displacement, where L1's share falls to 19 to 25%.

Two facts set the strength:

- **In the resolvable regime, L1 is light: 0.02 to 0.11.** This reproduces the first report and
  the earlier bridge benchmark, which pinned `l1` near 0.11.
- **In the hard corner, L1 saturates at the grid ceiling of 0.40.** The sweep wanted more
  sparsity than the grid offered, exactly where recovery is hardest. That truncation was flagged
  as the open question last time and it is *still open* — the grid was not extended.

Resolution tracks the source. PIV wants large correlation windows (24 to 32 px) on broad sources,
dropping to 8 px to localize the smallest ones; iLK wants a small radius (5 to 7); FFD wants
mid-to-large B-spline spacing (12 to 24).

### The cost of a wrong regularization

![Four panels. The first two plot the objective divided by each field's own best against the regularization, as a bold median line with an interquartile ribbon. For FISTA+L1 the median is flat near 1.25 from L1 0.031 to 0.17 and climbs to 1.44 at 0.40, with a shaded plateau band across that range. For FTTC+L2 the median is flat at 1.38 for every λ below 1e-4 — the regularizer is inert there — dips to a sharp minimum near 1.1 around λ 1e-3, and climbs steeply above 1e-2. The third is a boxplot ladder of Sabass J in the core regime for four strategies, medians rising left to right from FISTA+L1 oracle to FTTC+L2 best-fixed. The fourth is a stacked bar chart over three scoring criteria showing which regularizer path wins: on Sabass J FISTA+L1 wins 89 percent, on whole-field nRMSE it wins 56 percent, and on angular error it wins only 19 percent — FTTC+L2 takes 81 percent.](../images/heuristic-sweep-regularization.png)

Both basins are wide, and the L1 basin is wider:

| L1 sparsity | 0.020 | 0.031 | 0.047 | 0.072 | 0.111 | 0.170 | 0.261 | 0.400 |
|---|---|---|---|---|---|---|---|---|
| median penalty vs that field's own best | 1.32 | 1.28 | 1.28 | 1.25 | **1.24** | 1.25 | 1.32 | 1.44 |

Anything from **0.031 to 0.17** lands within 5% of the best single choice; the optimum sits at
0.11. The penalty is asymmetric — going too sparse costs more than not going sparse enough,
because excess sparsity thresholds away real force on the adhesion while a shortfall only leaves
a little background noise the composite barely charges for. The rule survives from the first
report: **err low.**

For the L2 path, λ below about 1e-4 is simply **inert** — the median curve is flat at 1.38 across
five decades, meaning the regularizer is doing nothing and the recovery is whatever the raw
inversion gives. The useful band is narrow, 0.001 to 0.003, and the climb above 1e-2 is steep.
A smoothness parameter is both less forgiving and easier to set uselessly than a sparsity one.

What tuning actually buys, on the 256 core-regime scenes:

| strategy | median J | penalty vs its own oracle |
|---|---|---|
| `FISTA + L1`, per-field oracle | 0.375 | 1.00× |
| `FISTA + L1`, best single fixed value (0.031) | 0.471 | 1.26× |
| `FTTC + L2`, per-field oracle | 0.558 | 1.00× |
| `FTTC + L2`, best single fixed value (0.0018) | 0.896 | 1.61× |

Read the table by columns, not rows. **Choosing the path matters more than tuning it**: a
*fixed* L1 (0.471) beats a per-scene *oracle* L2 (0.558). Tuning L1 buys a further 26%, and
tuning is worth more to the smoothness path (61%) than to the sparsity path — smoothness is both
worse and fussier. This restates the first report's headline in a cleaner form: that version
compared tuned L1 against parameter-free Bayesian-L2 and found a 3.2× gap, bundling the
regularizer with the mesh and with the λ-selection rule. Here the same conclusion falls out of a
comparison where *both* paths get their best possible λ, so nothing is hidden in the tuning.

The caveat is unchanged and it matters: the benchmark source is a compact dipole, and L1's prior
matches a localized source by construction. That transfers to real focal adhesions, which are
compact. On diffuse cell fields the margin narrows — see below — but it does not reverse.

### What the objective was hiding

`J` was chosen because it decomposes error into the three modes the regularizers trade off. That
is a virtue for diagnosis and a trap for ranking: one of those three modes, `DTMS`, is a direct
reward for a clean background, and one of the two candidates produces an exactly-zero background
by construction. Re-scoring the *same* configurations — same displacement fields, same
per-path-optimal regularization — on criteria with no background term:

| criterion (core regime, 4,608 configurations) | L1 better in | median L2 | median L1 |
|---|---|---|---|
| Sabass `J` (contains `DTMS`) | **89.4%** | 0.558 | 0.375 |
| whole-field nRMSE | 56.2% | 1.021 | 0.957 |
| background leak | 94.5% | 20.28 | 1.06 |
| **angular error** | **18.6%** | **12.6°** | **30.3°** |

On accuracy the two paths are a coin flip. L1's entire advantage is background suppression — 20×
less leak — which is precisely what `DTMS` pays for. It buys that by getting the traction
*direction* wrong: 30° median angular error against L2's 13°, with L2 directionally better in
81% of configurations.

The diffuse cells make the point without any ambiguity, because there the ranked objective **is**
whole-field nRMSE, so both paths were already tuned on a metric with no background term:

| criterion (24 cells × 18 inputs) | L1 better in | median L2 | median L1 |
|---|---|---|---|
| whole-field nRMSE | 65.0% | 0.916 | 0.876 |
| **angular error** | **3.0%** | **13.0°** | **44.6°** |

A 4.5% edge in nRMSE, against a traction field pointing 45° away from the truth in 97% of cases.

**Post-hoc mask clipping does not rescue L2's accuracy, and does not need to.** Applying the
shipped "Clip Outside Mask" to both paths on the cells (`analyze.py --clip-test`): clipped L2
still loses to clipped L1 on nRMSE (32% of configurations), and clipping cannot move the angular
comparison at all, because it only deletes exterior pixels while the angular error is measured on
the source region. Clipped L2 beats clipped L1 on angle in 97% of configurations. What clipping
*does* fix is L1's behaviour at the noise floor, where raw L1 reaches nRMSE 1.419 — worse than
predicting zero everywhere — against L2's 0.983.

| cells, whole-field nRMSE | L2 raw | L2 clipped | L1 raw | L1 clipped |
|---|---|---|---|---|
| noise floor (≤ 1.2 px) | 0.983 | 0.978 | 1.419 | 1.023 |
| useful window (1.2 to 8 px) | 0.908 | 0.885 | 0.830 | 0.820 |
| strong (> 8 px) | 0.897 | 0.887 | 0.887 | 0.887 |

#### Where the artifacts actually are

Both estimators invent structure; they invent different structure, and the ground truth
distinguishes them. The GT pole is a Gaussian — smooth, monotone, no boundary anywhere.

![Four panels. Three zoomed crops of one traction pole show, left to right: the ground truth as a smooth Gaussian blob peaking at 258 Pa; the FTTC+L2 recovery as a smooth but broader and dimmer blob at 168 Pa with a faint halo around it; and the FISTA+L1 recovery at 221 Pa as a blob with a visibly hard circular edge and a checkerboard texture across its interior. The fourth panel plots magnitude as a percentage of each field's own peak against distance from the pole centre, on a log axis. The ground truth decays smoothly over four decades. FISTA+L1 tracks it down to about 10 percent and then falls vertically off a cliff at 27 px, annotated as the point where the L1 support ends while the ground truth still carries 8 percent of peak. FTTC+L2 tracks the ground truth further out but then flattens into a persistent oscillating halo between 0.3 and 2 percent of peak that never decays.](../images/heuristic-sweep-edge-artifact.png)

- **`FISTA + L1` manufactures an edge.** Group soft-thresholding makes a per-pixel
  keep-or-zero decision, so the blob terminates at a level set of the field. On the sweep's
  *best* L1 recovery the magnitude falls from 22% of peak to the storage floor across about
  three pixels, at a radius where the ground truth still carries 8% and is still decaying
  smoothly. Nothing in the ground truth or in the displacement field has a boundary there.
  The same mechanism produces the checkerboard visible across the blob interior: neighbouring
  pixels cross the threshold independently.
- **`FTTC + L2` manufactures a halo.** It tracks the Gaussian tail honestly much further out,
  then flattens into a persistent oscillating floor at 0.3 to 2% of peak that never decays —
  Fourier ringing. That is the mechanism behind its 20× background leak.
- **Peak magnitude goes the other way.** L1 recovers 221 Pa of the true 258 (−14%); L2
  recovers 168 (−35%). Sparsity preserves the peak, smoothness shrinks it.

This is why the three `J` terms disagree: `DTM` (magnitude on the adhesion) favours L1,
`DTMS` (spurious background) favours L1, and `DTA` (direction) favours L2. Blending them into
one number hid a genuine three-way trade behind a single winner.

The practical asymmetry is that the two artifacts are not equally correctable or equally
honest. A low-amplitude ringing halo is visible, bounded, removable by a mask, and does not
claim the force *stopped* anywhere. A hard support boundary is none of those: it is
indistinguishable from a real adhesion edge, so it invites exactly the wrong biological
reading — "the traction ends here" — from a field where it does not.

**The verdict, stated as a trade rather than a winner.** Choosing L1 as a general-purpose default
means trading roughly 3 to 5% of whole-field accuracy for 2.5 to 3.5× the angular error, a
nonlinear estimator, and divergence risk at low SNR. Three further properties matter for this
tool specifically and none of them appear anywhere in `J`:

- **Linearity.** L2 is a linear estimator, so error propagates predictably from displacement to
  traction. L1 is not: its support — which pixels are nonzero — is a discontinuous function of
  the input, so there is no error propagation to speak of and no defensible uncertainty on a
  recovered magnitude.
- **Downstream stress inference.** Traction here feeds monolayer stress. Stress integrates the
  traction field, so directional error and invented sharp structure propagate straight into the
  stress map, and BISM's uncertainty estimates assume a linear forward chain.
- **Temporal stability.** In a time-lapse, an L1 support that flips between frames produces
  discontinuities in the traction time series that are not present in the displacement data.

So the honest recommendation inverts the one `J` alone would give: **`FTTC + L2` is the right
general-purpose default**, and `FISTA + L1` is the right choice when the specific goal is a clean,
sparse adhesion map and directional fidelity is not the deliverable. The sweep's `J`-ranked
results above remain valid as what they are — a map of what each path can reach under a
background-weighted objective — and the parameter heuristics for L1 stand for anyone who chooses
that path deliberately.

## What the ceiling looks like

![Three rows, each showing a ground-truth traction magnitude beside the FTTC+L2 and FISTA+L1 oracle recoveries, with the winning panel boxed. Row one, a broad source at 7.9 px: FISTA+L1 recovers both poles almost exactly at J=0.009 while FTTC+L2 smears them at 0.255. Row two, a median-difficulty scene: both are poor, L1 at 0.555 against L2 at 0.884, and the L2 panel shows a haze of background speckle the L1 panel does not have. Row three, a diffuse cell at mid strength: both recover the ring of peripheral adhesions, L1 at nRMSE 0.488 against 0.657, and both are visibly dimmer than the truth.](../images/heuristic-sweep-examples.png)

Three things the tables cannot show. The best case is genuinely excellent — `J` = 0.009, two
clean poles at the right magnitude. The difference between the paths is mostly *background*: the
L2 panels carry a haze of spurious traction across the whole field that L1 simply does not
produce, which is exactly what the `DTMS` term charges for. And every recovery is dimmer than its
ground truth: regularized TFM buys a clean background by shrinking the peaks, and that magnitude
under-estimate is a property of the method, not a tuning failure.

## How imaging parameters set quality

![Three panels. The first ranks the 16 imaging conditions by median best J as horizontal bars coloured by exposure, spanning 0.56 to 1.12; the exposure groups interleave rather than separating cleanly, with one exposure-0.25 condition among the best. The second shows Spearman rank correlations of each imaging parameter against achievable J on the 16 per-condition medians: NA -0.53, density -0.46, exposure -0.36, jitter +0.27. The third is a difference heatmap of J at exposure 0.25 minus J at exposure 4, near zero across most of the grid and lighting up to +0.86 in a band at mid footprint and low displacement.](../images/heuristic-sweep-imaging.png)

**Better images help, but this design cannot say which knob deserves the credit.** The 8
scenarios sample density × NA × exposure sparsely and non-orthogonally, so the rank correlations
(NA −0.53, density −0.46, exposure −0.36, jitter +0.27, on 16 points) describe the ladder rather
than attribute the effect. The marginal medians tell the same story more honestly:

| parameter | levels (median best J) |
|---|---|
| exposure | 0.25 → 0.95 · 1.0 → 0.73 · 4.0 → 0.72 |
| NA | 0.6 → 1.05 · 0.8 → 0.76 · 1.0 → 0.73 · 1.2 → 0.77 · 1.4 → 0.63 |
| density | 0.023 → 0.89 · 0.046 → 0.71 · 0.092 → 0.86 · 0.183 → 0.66 |
| jitter | 0.067 px → 0.72 · 0.2 px → 0.77 |

Every quality axis points the same way and none of them dominates. Two readings are safe:

- **Exposure saturates.** Going from 0.25 to 1.0 buys a lot (0.95 → 0.73); going from 1.0 to 4.0
  buys essentially nothing (0.73 → 0.72). There is a sufficiency threshold, not a linear return.
  *This corrects the first report, which named exposure the single dominant lever; on the rebuilt
  pipeline it is one of three comparable ones, and its returns stop early.*
- **Registration jitter is nearly free.** Tripling it, 0.067 to 0.2 px, costs 7%. That conclusion
  survived the rebuild — and it is why the severe-jitter arm was dropped from the refined grid.

The best condition (`s3`: 12k beads, NA 1.4, exposure 4.0) reaches median `J` 0.56 and the worst
(`s4`: 24k beads, NA 0.6, exposure 0.25) reaches 1.12 — a factor of two across the whole quality
ladder. Notably, `s7` (48k beads, NA 1.2, exposure 0.25) is among the best *despite* the lowest
exposure: density and NA bought back what exposure lost.

**Where the quality pays off is the more useful result.** The difference panel is near zero in
the already-good cells and near zero in the dead corner, and lights up in a band at mid footprint
and low displacement, peaking at +0.86 `J` at 1.88 µm and 0.5 px. Good images do not lower the
floor of measurements you could already make; they extend the recoverable region into weaker
sources. **Imaging quality does not make good measurements better, it makes marginal
measurements possible.**

## Diffuse fields: realistic cells

The dipole grid isolates one localized source. A real cell is the opposite: fifty to a hundred
contractile stress fibres overlapping into a diffuse, centripetal field. The scenes are the
benchmarkTFM synth cells — four real cell outlines carrying 16 to 82 fitted stress fibres, their
traction fitted to real bead measurements. That traction is the ground truth, forward-projected
through this pipeline's own Green's operator, scaled to a ladder of strengths, and used to rewarp
the single best-imaging bead pair. Everything else matches the dipole run; the one variable is
the field. `J` is undefined here — a centripetal field's mean traction vector cancels — so
recovery is ranked on whole-field nRMSE.

![Four panels. A: a grid of four cells against six strengths coloured by winning method, PIV filling all but one cell. B: best whole-field nRMSE against strength for the three methods, each a U bottoming at 3 to 8 px, PIV lowest throughout at 0.49. C: nRMSE relative to each field's best against L1 sparsity, comparing dipoles and cells: the dipole curve dips to 1.24 in a plateau across 0.031 to 0.17, while the cell curve is nearly flat from 1.16 at L1 0.02 down to 1.01 and stays there across 0.047 to the 0.40 grid ceiling. D: the oracle L1 sparsity and oracle FTTC λ against strength, both falling steeply from the noise floor through the useful range.](../images/heuristic-sweep-cells.png)

**PIV owns the diffuse field.** It wins 23 of 24 cell scenes; iLK takes one at the noise floor and
FFD takes none. *This differs from the first report*, which gave FFD six wins at the
high-displacement end; on the rebuilt native-resolution pipeline that advantage disappears.

**Recovery is a U in strength.** Best nRMSE bottoms at 0.49 for peak displacements of 3 to 8 px,
degrades into the registration-jitter floor below 1 px (0.85 at 0.5 px), and into decorrelation
above 20 px (0.76 at 50 px). The floor is high because a diffuse field carries fine focal-adhesion
structure below the band-limited operator's reach: the honest ceiling on a real cell is not a
clean recovery, it is a smoothed one.

**Sparsity still wins on nRMSE, but only just, and it loses direction badly.** `FISTA + L1`
supplies the winner in 96% of cell scenes and edges the median nRMSE (0.876 against 0.916) — but
its median angular error is 44.6° against L2's 13.0°, and L2 is directionally better in 97% of
configurations. On a diffuse field the accuracy question is close to a tie and the directional
question is not close at all; see
[What the objective was hiding](#what-the-objective-was-hiding). The L1 basin is *much* flatter
than the dipole one:
the median penalty runs 1.16 at L1 0.02 down to 1.01, and everything from 0.047 to the 0.40
ceiling is within 5% of the best. On a diffuse field the L1 value is close to a free parameter.
Its optimum sits *at* the grid ceiling, which is the same truncation the dipole hard corner
shows — the grid does not bracket the cell optimum, so "0.4 is best" here means "0.4 or more".

*This revises the first report*, which put the cell plateau at a narrow 0.07 to 0.17. On the
rebuilt pipeline the plateau is far wider and pushed against the ceiling. The practical advice is
unchanged in direction and stronger in confidence: a default in the low tenths is safe on both
field types, and on cells almost anything is.

Panel D shows the optimum still tracking SNR: both knobs ramp hard at the 0.5 px noise floor and
relax by an order of magnitude through the useful range.

### Does mask confinement earn its keep?

> **The feature this section tested no longer exists.** The first report measured an *in-solver*
> soft exterior penalty and recommended it specifically over post-hoc masking. Commit `468cb38`
> (2026-07-23, "refactor force mask clipping") removed that penalty from `forward_l1.py`;
> `fwd_mask_strength` is now an on/off gate for **post-hoc clipping** applied in `fttc.py` after
> an ordinary solve, and the widget exposes it as a checkbox, "Clip Outside Mask". The section
> below is rewritten against the shipped behaviour. `cell_confinement.py` has not caught up — it
> still sweeps the old 0→100 dial, which is why its "soft" arm now returns results bit-identical
> to no-mask; only its post-hoc arm is meaningful, and that is what is quoted here.

Everything above ran with no cell mask, on purpose: the sweep sets the defaults, and the
defaults have to ship safe for the user who has not drawn a segmentation. The diffuse cells are
the only fair place to ask whether a mask is worth turning on — a dipole's mask is just its two
blobs, so confining to it merely restates the ground truth. The honest prior is the cell
*outline*, which a user segments from brightfield and which is genuinely looser than the
traction support, since the fibres pull hardest at the periphery, well inside the edge.
Displacement is held at PIV window 24 and regularization at the shipped default (`l1 = 0.05`);
the mask is the only thing that moves.

**Post-hoc clipping helps exactly where there is background to delete, and nowhere else.**
Whole-field nRMSE over the 24 cell scenes:

| strength band | no mask | clipped outside the outline | change |
|---|---|---|---|
| noise floor (\|u\| ≤ 1.2 px, n=4) | 2.051 | 1.321 | −36% |
| useful window (1.2 to 8 px, n=12) | 0.801 | 0.773 | −3.5% |
| strong (\|u\| > 8 px, n=8) | 0.854 | 0.854 | 0% |

The gradient is the whole story. At the noise floor the unconstrained solve scatters spurious
traction across the exterior and deleting it is a large win — though both numbers are above 1,
meaning the recovery is worse than predicting zero, so this is rescuing a hopeless field into a
merely bad one. In the useful window the gain is a real but modest 3.5%. At high strength there
is essentially no background left to delete and clipping does nothing at all.

**By construction it cannot do more than that.** Post-hoc clipping leaves every in-cell pixel
bit-identical to the no-mask solve — the measured in-cell nRMSE is unchanged in all 24 scenes —
so it captures background removal and only background removal. That is the limitation the
removed in-solver penalty existed to address: the Green's operator is non-local, so an
unconstrained solve can explain in-cell displacement with force parked *outside* the cell, where
it sits in the fit's near-nullspace. Deleting that force afterwards leaves the interior
under-fit for displacement it had been explaining, whereas a penalty in the objective forces the
same displacement to be re-explained by interior force. Whether that interior refit is worth
restoring is now an open design question, not a settled result — see Open.

**Practical advice, matching what ships today:** if you have a segmentation, turning on "Clip
Outside Mask" is free and helps most when the data is marginal. It is off by default because the
no-mask user must not inherit a prior they did not draw. Do not expect it to improve traction
*inside* the cell; it cannot.

## The recipe

- **Default to PIV**, window ~24 px, `FISTA + L1` with `l1_sparsity` ~0.05 to 0.11, no ridge.
  PIV wins 62% of scenes outright and sits within 10% of the best method 72% of the time; on
  diffuse cells it wins 23 of 24.
- **Broad source at sub-pixel to few-pixel displacement: try iLK**, radius 5 to 7. It holds the
  top-left of the grid at 50 to 88% of conditions.
- **Broad source at large displacement (≳ 8 px): switch to FFD**, spacing 12 to 24. Its
  smoothness holds the field together where correlation windows decorrelate.
- **Default to `FTTC + L2` for the inversion**, λ in 0.001 to 0.003. It is linear, its traction
  directions are right (13° median against L1's 30°), and it does not diverge at low SNR. Its
  weakness is a dirty background, which "Clip Outside Mask" partly removes when you have a
  segmentation. **Know that λ below 1e-4 does nothing at all** — the useful band is narrow.
- **Reach for `FISTA + L1` deliberately, not by default.** It is the right tool when you want a
  clean sparse adhesion map and directional fidelity is not the deliverable. It buys ~3 to 5% of
  whole-field accuracy and 20× less background, and costs 2.5 to 3.5× the angular error, a
  nonlinear estimator with no usable error propagation, and divergence at the noise floor.
- **If you do use L1, err low.** Anything in 0.031 to 0.17 is within 5% of optimal;
  over-sparsifying costs more than under-sparsifying. On diffuse cells the basin is flatter still.
- **Do not read the sweep's `J` rankings as a recommendation between the two paths.** `J`
  contains a background term that one of the two candidates zeroes by construction.
- **Have a segmentation? Turn on "Clip Outside Mask".** It is free, helps most when the data is
  marginal (−36% whole-field nRMSE at the noise floor, −3.5% in the useful window, nothing at
  high strength), and by construction cannot change traction *inside* the cell. Off by default,
  because the no-mask user must not inherit a prior they did not draw.
- **Sources below ~0.3 µm footprint: expect failure from every method** (`J` > 1.1 across the
  whole 0.1 µm row). Treat the magnitudes skeptically or don't report them.
- **To resolve marginal sources, improve the images — but expect a threshold, not a slope.**
  Exposure saturates by 1.0, NA and density matter comparably, and registration jitter up to
  0.2 px is nearly free.

## Provenance

The run covers **504 scenes** (480 dipole + 24 diffuse cell) across 17 conditions, holding
**9,072 cached displacement fields** and **371,952 scored inversions** in the GT-tuned oracle
force cache. Compute ran as a two-stage pipeline on the Maestro cluster: GPU displacement
caching feeding the force cache through an element-wise `aftercorr` dependency.

Every figure is regenerated from that cache by `_validation/heuristic_sweep/analyze.py`, which
reads `renders/index.csv` and the stored objective curves — no re-solving. The confinement
figure comes from `cell_confinement.py`, the one analysis that re-inverts the cache. Exact
commands and the data-generation pipeline are in the
[sweep README](../../_validation/heuristic_sweep/README.md#reproducing-the-report). The data
lives in the private stage directory, not this repo; the scripts take its path from `$STAGE`.

## Open

1. **The L1 grid is still truncated at 0.40**, and it now bites twice: in the small-footprint
   dipole corner and at the *optimum* of every diffuse cell scene. Extend the axis upward before
   trusting any recommendation that lands on the ceiling.
2. **The whole sweep was ranked on an objective that prefers one of its candidates.** `DTMS`
   rewards a zero background and group-L1 produces one by construction, so every `J`-ranked
   result here is tilted. It is probably harmless for the *displacement*-method comparison —
   all three methods are scored through the same two regularizer paths — but it invalidated the
   regularizer verdict, and a future sweep should rank on a criterion set (accuracy **and**
   direction) rather than a single blended composite. The deeper version of the problem: every
   ground truth in this sweep is sparse — Gaussian poles or discrete fitted fibre adhesions on a
   zero background — so a sparsity prior is *correct by construction* everywhere in it. A truly
   diffuse ground truth would be needed to test the priors against each other fairly.
3. **The imaging design cannot attribute credit** between density, NA and exposure — the 8
   scenarios sample them non-orthogonally. A factorial arm would be needed to separate them.
4. **The scene grid still spends half its cells on the unrecoverable.** The refined grid staged
   in `sweep_config.py` (footprint 0.3 to 3.0 µm, displacement 1.5 to 25 px, mild jitter only)
   targets the detectability frontier instead of the dead corners. It has not been run; when it
   is, `analyze.py` reproduces this entire report against it unchanged.
5. **The cells were run only at best imaging**, so their SNR-tracking is inferred from the
   strength axis rather than a second imaging ladder.
6. **Mask confinement lost its interior refit and nobody has decided whether to restore it.**
   `468cb38` replaced the in-solver exterior penalty with post-hoc clipping, which is simpler
   and cheaper but structurally cannot improve in-cell traction. The earlier measurement
   claimed the interior refit was the larger half of confinement's benefit; that claim is now
   untested against the current pipeline and untestable with the current probe.
   `cell_confinement.py` needs rewriting against the new design — it still sweeps a 0→100 dial
   that is now an on/off gate — before the question can be reopened.
