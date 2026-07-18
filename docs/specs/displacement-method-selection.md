# Displacement methods for napariTFM v2: ship PIV, iLK, FFD-pyr

**Status:** decided (2026-07-18)
**Scope:** which displacement algorithms ship in the next napariTFM version, and why. Justification
for keeping three of the five benchmarked methods and dropping the other two. Tuning defaults for the
three we keep are recorded here so the UI ships sane starting points.

---

## One line

Five methods were benchmarked on a warped-real-bead field with known ground truth. Accuracy did not
separate them; off-cell noise and capture range did. **PIV** is the quiet default, **FFD-pyr** wins
under large deformation, and **iLK** is a fast optical-flow hedge that ties PIV on clean images and is
reported to overtake it at low SNR. FFD (elastix) is dominated by FFD-pyr and TV-L1 is
implementation-crippled, so both are dropped.

## The decision

We ship three methods and expose all of their first-order knobs:

- **PIV** — the default. Quietest method at every scale, degrades gracefully out to large motion.
- **FFD-pyr** — the large-deformation option. Leads once motion passes ~15 px.
- **iLK** (Lucas–Kanade) — the light optical-flow option. Ties PIV at small motion and is our hedge
  for the low-SNR regime the benchmark did not test.

We drop two:

- **FFD (elastix)** — strong at small motion but CPU-only, slow, and weaker in the mid regime.
  FFD-pyr covers the same small-motion strength, adds capture range, and is faster; elastix buys us
  nothing FFD-pyr does not already give.
- **TV-L1** — trailed at every scale, but for an implementation reason rather than a property of the
  method: the reference build has no image pyramid and runs out of capture range under large motion.
  Shipping a knowingly crippled build would misrepresent optical flow. iLK is the optical-flow method
  we carry instead.

The three we keep span the two axes that mattered: PIV and iLK hold the low-noise corner at small
motion, FFD-pyr holds capture range at large motion, and iLK additionally hedges the noise axis the
benchmark left unswept.

## The benchmark

The benchmark warps real bead frames by a known ground-truth field. The answer is therefore known
exactly, while the images still carry the native noise of the camera. Each method is scored on two
numbers.

- **Peak recovery**: recovered peak displacement divided by the true peak. `1.0` means the sharp
  near-adhesion peak survived; below `1.0` means window or grid averaging washed it into a blob.
- **Off-cell noise**: RMS spurious displacement outside the cell, in micrometres, where the true
  motion is essentially zero. This is the floor the force step will amplify.

A single tuned setting would leave open whether a baseline was simply under-tuned, so the benchmark
reports a frontier rather than a point. It sweeps each method's main smoothing knob across its full
range and tunes the coupled secondary knob at every step. What is shown is each method's best
achievable trade-off between the two numbers, with no method handicapped. The top-left corner of each
panel is best: high peak, low noise.

![Peak recovery versus off-cell noise for five methods, at four displacement regimes. In each panel the
curves trace each method's achievable frontier as its smoothing knob is swept; points nearer the top-left
are better. At small motion the curves bunch near peak 1.0 and PIV sits leftmost. As the displacement
grows the curves fall and spread, and the purple FFD-pyr curve rises above the rest.](../images/displacement-frontier.png)

Best peak recovery each method reached inside a matched low-noise budget:

| displacement (peak) | best method | order by peak recovery |
| --- | --- | --- |
| ×1 (~3 px) | PIV, FFD | PIV 0.997, FFD 0.997, iLK 0.98, FFD-pyr 0.97, TV-L1 0.81 |
| ×5 (~15 px) | three-way | PIV 0.84, FFD-pyr 0.82, iLK 0.82, FFD 0.73, TV-L1 0.57 |
| ×10 (~30 px) | FFD-pyr | FFD-pyr 0.76, PIV 0.73, iLK 0.70, FFD 0.65, TV-L1 0.42 |
| ×25 (~75 px) | FFD-pyr | FFD-pyr 0.72, FFD 0.60, PIV 0.59, iLK 0.59, TV-L1 0.42 |

The ordering changed with motion, and that change is what the ship list follows.

- **PIV stayed quietest.** It held the leftmost frontier at small motion and degraded gracefully out
  to 75 px. iLK, an optical-flow method, tracked it within a few percent at small motion.
- **FFD-pyr led under large deformation.** Its image pyramid gives it capture range and its coarse
  control grid keeps the noise low, so it pulled ahead once the motion passed about 15 px.
- **TV-L1 trailed at every scale, for a reason specific to this build.** The reference TV-L1 has no
  image pyramid, so it runs out of capture range under large motion. TV-L1 and other optical-flow
  variants with proper multiscale handling are reported elsewhere as competitive with or better than
  PIV; the poor showing here reflects this build and this regime, which is why we carry iLK rather
  than this TV-L1.

The crossover sat near 15 px. Below it PIV led on noise; above it FFD-pyr led on capture. Where the
crossover sits exactly, and whether it holds, depends on imaging conditions this test did not vary.

## What the benchmark does not cover

The ship decision rests on one slice of the problem, and that slice has not been checked against the
axes most likely to move it. These are the caveats we are shipping on, not open questions that block
the decision.

- **Imaging conditions.** One bead field, one microscope, one pixel size (0.16 µm/px), one bead
  density and point-spread function. Sparser or denser beads, a different objective, or a dimmer
  signal could reorder the methods.
- **Noise level.** The test used the native noise of two real frames and did not sweep
  signal-to-noise. Earlier work in this project found that at low SNR the ordering can flip and
  optical-flow methods can overtake PIV, so the "PIV is quietest" result holds for clean images only.
  This is the single strongest reason iLK stays on the ship list.
- **Motion type.** Uniform in-plane warps of a static field. No out-of-plane bead motion, no
  photobleaching, no drift beyond what registration removed.
- **Sample variety.** Two synthetic force fields rescaled to one cell size. Real cells vary in shape,
  spread, and adhesion pattern, and other force distributions may favour other methods.

The literature is likewise unsettled: several TFM studies report optical-flow methods matching or
beating PIV, and our own iLK tying PIV at small motion is consistent with them.

## Shipping defaults

No method can be set once and left; each has more than one first-order knob, and the right values
depend on the beads and the motion. The values below are the starting points the UI ships, to be
checked on a frame with **Preview Current Frame** before a full run.

- **PIV.** Three knobs are first-order, not one. **Window** sets the peak-versus-noise trade-off:
  `24` px is the ship default, smaller sharpens the peak and raises noise, larger smooths. **Overlap**
  controls how finely the windows are sampled: higher overlap (up to `0.875`) recovers sharp peaks
  better but costs compute and memory, and can exhaust GPU memory on large frames. **Passes** drives
  convergence and capture range: too few, and large motion is missed. The benchmark held passes and
  overlap fixed to compare methods fairly, which says nothing about their importance on real data — all
  three are worth tuning.
- **FFD-pyr.** **Level spacing** of the finest control grid is the main trade-off: `16` to `24` px
  keeps the field smooth and quiet, `8` to `12` sharpens the peak. Pyramid depth sets the capture
  range and matters once motion is large; raise it if big displacements are missed.
- **iLK.** **Radius** acts as a noise aperture: start near `7` to `10` px, larger for noisier images.
  **Warp count** governs refinement and converges by about `16`; raise it if the field looks
  under-converged, not for capture.

### Regime examples

**A typical spread cell.** The beads move three to six pixels between the relaxed and the deformed
frame. PIV with window `24` is a good first try; in this test it reached peak recovery near `1.0` and
the lowest off-cell noise, around `0.012` µm. If the peak looks blunted, drop the window or raise the
overlap and re-preview.

**A soft gel or a large, motile cell.** The beads move thirty pixels or more, and PIV begins to lose
the peak. FFD-pyr with level spacing `16` recovered about a quarter more of the peak than PIV at 30 px
here, and held its lead out to 75 px. If large displacements are still missed, raise the pyramid
depth.

**A dim or sparse-bead image.** Outside what the benchmark tested; the ordering should not be assumed
to hold. Preview PIV and iLK on the same frame and compare the off-cell background directly, since low
SNR is the regime where the ordering is reported to change.

**The noise floor.** Whatever method is chosen, expect roughly `0.01` to `0.02` µm of apparent
displacement outside the cell. This is the floor, not signal: real camera noise that no amount of
smoothing removes without also washing out the peak. It should not be tuned to zero; tune the peak and
accept the floor.

## Reproducing the benchmark

Every point in the figure comes from the public benchmark repository, and the whole study reruns from
three scripts.

- The sweep that produces the data:
  [`study/frontier.py`](https://github.com/ArturRuppel/benchmarkTFM/blob/master/benchmark_evaluations/displacement_measurements/study/frontier.py)
- The figure and the matched-budget table:
  [`study/frontier_figure.py`](https://github.com/ArturRuppel/benchmarkTFM/blob/master/benchmark_evaluations/displacement_measurements/study/frontier_figure.py)
- The coupling analysis that decides which secondary knob to tune per method:
  [`study/sensitivity.py`](https://github.com/ArturRuppel/benchmarkTFM/blob/master/benchmark_evaluations/displacement_measurements/study/sensitivity.py)

The scenarios and metrics live alongside them under
[`benchmark_evaluations/displacement_measurements`](https://github.com/ArturRuppel/benchmarkTFM/tree/master/benchmark_evaluations/displacement_measurements).
