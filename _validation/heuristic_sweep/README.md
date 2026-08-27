# TFM regularization-heuristic sweep

Goal: fit a *heuristic* for the elastic-net regularization (L1 sparsity `l1`,
L2 ridge `l2`) of the traction reconstruction, from ground truth, across a grid
of adhesion footprints and magnitudes and (later) imaging conditions. The bridge
benchmark told us the *shape* — `l1` pins near ~0.11, `l2` tracks inverse SNR —
but three opaque noise points can't fit a curve. This replaces it with a
generative ladder that has clean, analytic ground truth on a common grid.

## The objective: Sabass composite J

Only the recovered **traction** is scored, against an analytic GT traction
rasterized on a common `GT_REFERENCE_SIZE` grid. The ranked objective is the
Sabass et al. (2008) composite (see `sabass.py`):

    J = |DTM| + DTMS + DTA/45

- **DTM** — deviation of traction *magnitude* on the adhesions (signed, the L2/scale error),
- **DTMS** — spurious force in the *surrounding* ring (contour sharpness, the L1/sparsity error),
- **DTA** — angular error in degrees (÷45 to weigh it against the two O(1) terms).

J decomposes recovery error into the three modes the `(l1, l2)` knobs trade off;
a single blended `nRMSE` collapses exactly that tradeoff, so it is **recorded as a
cross-check but never ranked on**. (Caveat: DTMS was defined for sharp-contour disc
adhesions; with the Gaussian profile the GT itself carries a small DTMS tail-floor —
watch whether it biases the L1 optimum. See `sabass.py`.)

Displacement error is *never* scored on its own — it does not propagate
uniformly through the DC-stripped, band-limited Green's operator, so minimizing
it can rank displacement settings wrongly. Let the operator do the spectral
weighting and measure the output.

## Three stages (see `sweep_config.py`)

1. **Displacement caching** (`build_cache.py`): per scene, sweep the *resolution*
   knob. At each resolution, raise the *convergence* knob until the field stops
   changing (`CONV_TOL`) and cache that converged field. Convergence is set by a
   convergence criterion, **not** by force score: an under-converged field is an
   accidental second regularizer, and we want the force-side regularizer to do all
   the regularizing.
2. **Oracle force caching** (`build_force_cache.py`): read the cached fields; for
   each, run *both* regularizer paths over their own grids and keep the map that
   scores best against GT — `FTTC + L2` over `LAMBDA_GRID`, and `FISTA + L1` over
   `FRAC1`. Each cached map is therefore the achievable **ceiling** of that path on
   that displacement field, and the full objective curve is stored alongside it, so
   the cost of a wrong regularization is recoverable without re-solving. Resolution
   stays *in* the search: one cached field per resolution, the objective adjudicates
   resolution jointly with the regularizer.
3. **Rendering and analysis** (`render_cache.py`, then `analyze.py`): the renderer
   bakes one 3×3 comparison card per (condition, scene, displacement input) plus a
   tidy `renders/index.csv`; `analyze.py` reduces that index and the stored curves
   into the report's figures. `browse.py` is a thin viewer over the same cards.

> The earlier design swept the full elastic-net `FRAC1 × FRAC2` grid through a
> `sweep_forces.py` driver into `results/*.csv`. That was retired: the oracle cache
> subsumes it (it answers "what is each path's ceiling?" directly instead of by
> exhaustive search), and the pure-L1 and pure-L2 paths turned out to bracket the
> mixed grid. `scoring.py` keeps the GT and metric primitives that driver owned.

## Why a balanced pair, never a monopole

`forward_l1.py`'s Green's operator zeroes the DC (k=0) Fourier mode
(`# ... DC=0`). A single unipole carries a large nonzero *mean* traction — the
DC mode — which the operator discards by construction. Its error would scale
with footprint×magnitude, i.e. exactly the sweep axes, faking a "heuristic" out
of a DC artifact. So every scene is a **balanced pair**: two equal-and-opposite
uniform-traction discs, net force zero.

## Generator contract  ← the interface your generator must produce

The generator writes one directory per scene under
`$STAGE/scenes/<condition>/<scene_id>/`, each containing:

| file            | meaning                                                             |
|-----------------|---------------------------------------------------------------------|
| `reference.tif` | undeformed realistic bead image, `GT_REFERENCE_SIZE`², float32      |
| `deformed.tif`  | `reference` warped by the GT displacement field, same shape         |
| `scene.toml`    | **authoritative** analytic GT spec (below)                          |

`scene.toml` — the scorer rasterizes GT traction *from this*, at any grid, so it
must fully specify the balanced pair:

```toml
[meta]
condition = "realistic"
scene_id  = "f30_m400"          # footprint / magnitude tag
image_size = 700                 # px, == GT_REFERENCE_SIZE
pixel_size = 0.1                 # µm/px, before binning

[substrate]                      # must match sweep_config forward params
young_modulus = 20000.0
poisson = 0.5

[pair]                           # balanced pair: pole 2 = −pole 1
profile   = "tophat"             # radial traction profile
footprint = 30.0                 # disc radius, µm
magnitude = 400.0                # traction magnitude, Pa
axis_deg  = 0.0                  # pair-axis orientation
separation = 120.0               # center-to-center distance, µm
center    = [0.0, 0.0]           # pair midpoint offset from image center, µm
```

**Consistency requirement (avoids the inverse crime):** the GT *displacement*
the generator applies to warp `reference` → `deformed` must be produced by the
forward Green's operator acting on this exact disc-pair traction, using the same
`substrate` params. The realism (and the noise we study) then comes entirely
from (a) the real bead texture and (b) the displacement method's recovery error
— not from an operator mismatch. Optional `traction_{x,y}.npy` /
`displacement_{x,y}.npy` dumps may be written for sanity checks but are **not**
read by the sweep; `scene.toml` is the single source of GT.

### Cell scenes (the diffuse-field variant, `make_cells.py`)

The dipole grid above isolates one localized source. A real cell is a diffuse
superposition of many contractile stress fibres, so a second scene *kind* stages
whole cells: `make_cells.py` takes each benchmarkTFM synth cell's fitted-fibre
traction as the GT shape, forward-projects it with **this** pipeline's Green's
operator (so `u` and `t_gt` stay a consistent forward/inverse pair), scales to hit
each peak-displacement target, and warps the best-imaging stack (scenario 6) exactly
as the dipole run does. These land in their own condition dir (`cell_s6j1`).

The contract inverts in one key place: **for a cell scene the GT is the stored
`gt_traction.npy` field, not the `scene.toml`.** There is no `[pair]` block and
nothing is rasterized from the toml; the toml carries only metadata, and the scorer
loads `gt_traction.npy` directly (see `build_force_cache.load_gt`, `meta.kind == "cell"`).

| file             | meaning                                                            |
|------------------|--------------------------------------------------------------------|
| `reference.tif`  | frame 0 of the scenario-6 stack (zero-jitter reference)            |
| `deformed.tif`   | `warp(frame 1, u)` — mild registration jitter + photon noise ride along |
| `gt_traction.npy`| **authoritative** GT traction `(2, N, N)` Pa (scaled fibre field)  |
| `cell_mask.npy`  | cell OUTLINE `(N, N)` bool — the honest segmentation prior (looser than the traction support); consumed only by `cell_confinement.py` |
| `scene.toml`     | metadata only (below); **not** the GT for cells                    |

```toml
[meta]
condition   = "cell_s6j1"
scene_id    = "synth00_u3.155"   # <cell>_u<peak |u| px>
image_size  = 512
pixel_size  = 0.1612
kind        = "cell"             # selects the gt_traction.npy path in the scorer
source_cell = "synth00"          # benchmarkTFM synth cell id
n_fibers    = 82                 # fitted contractile fibres in that cell
peak_disp_px = 3.155             # strength target the field was scaled to
rms_traction_pa = 812.0          # RMS |t| over the significant-GT region (bookkeeping)

[substrate]                      # same E / poisson as the dipole run (sweep_config)
young_modulus = 1000.0
poisson = 0.5
```

Because the diffuse field merges adjacent sources and its mean vector cancels, the
per-adhesion Sabass terms (and so `J`) degenerate; cell scenes are therefore ranked
on whole-field **nRMSE** — the blended error that is only a cross-check for dipoles
but is well-defined here. The `field_metrics` (`mag_bias`, `ang_field`, `bg_leak`;
see `field_metrics` in `scoring.py`) are recorded as a diagnostic decomposition
of that error, not the ranking key. Everything downstream (displacement cache, force
cache, resolution search) is identical to the dipole run; only the GT construction and
the ranking metric differ.

## Layout

```
$STAGE/ (= /helix/…/tfm_heuristic, set in env.sh, NOT in the repo)
  images/tif_stacks/{scenario*_dens*_NA*_expo*.tif,manifest.csv}          ← make_stacks.py
  scenes/<condition>/<scene_id>/{reference.tif,deformed.tif,scene.toml}  ← make_scenes.py (dipole)
  scenes/cell_s6j1/<scene_id>/{reference.tif,deformed.tif,gt_traction.npy,cell_mask.npy,scene.toml}  ← make_cells.py
  cache/<condition>/<scene_id>/disp_<method>_res<k>_sm0.npz               ← build_cache.py
  cache/<condition>/<scene_id>/force_<method>_res<k>_sm0.npz              ← build_force_cache.py
  renders/<condition>/<scene_id>/<input>.png + renders/index.csv          ← render_cache.py
  logs/                                                                   ← SLURM out/err
  code/                                                                   ← synced repo (sync_code.sh)
```

Tracked code lives in the repo (`_validation/heuristic_sweep/`); paths come from
`$STAGE` at runtime so no infra path is baked into the public repo.

## Running on Maestro (later — nothing here runs on its own)

```bash
# stage 0 (local): synthesize the 8 imaging-condition bead stacks. Deterministic
# (fixed seeds), needs psfmodels (pip install psfmodels). See make_stacks.py /
# calib_psf.py; the one calibrated constant (flux_per_px) is frozen in calib_psf.py.
python make_stacks.py            # -> $STAGE/images/tif_stacks/{scenario*.tif,manifest.csv}
python make_scenes.py --images-dir "$STAGE/images/tif_stacks" --stage "$STAGE"   # -> scenes/

# then, once scenes/ is populated:
bash jobs/sync_code.sh          # stage repo code + napariTFM pkg -> $STAGE/code
ssh aruppel@maestro.pasteur.fr
source /helix/…/tfm_heuristic/code/env.sh   # sets STAGE, PYTHONPATH
bash jobs/submit.sh pipeline     # displacement cache (GPU) -> oracle force cache, chained
#   or run the stages separately:
#   bash jobs/submit.sh cache               # stage 1 only
#   bash jobs/submit.sh force --after JOBID # stage 2 only
```

Then, back where the stage is readable:

```bash
python render_cache.py --stage "$STAGE" --workers 8   # cards + renders/index.csv
```

## Reproducing the report

The findings are written up in
[`docs/specs/2026-07-20-heuristic-sweep-method-parameter-selection.md`](../../docs/specs/2026-07-20-heuristic-sweep-method-parameter-selection.md).
Every figure in that report is regenerated by `analyze.py` from `renders/index.csv`
and the objective curves stored in the force cache — no re-solving, seconds to run.
The scripts read the stage from `$STAGE` (or `--stage`), so `source env.sh` first;
no path is baked in.

```bash
source env.sh                                     # sets STAGE (see above)
IMG=../../docs/images                             # write straight into the report's images

python analyze.py          --outdir "$IMG"        # all six report figures + sweep_summary.csv
python analyze.py --clip-test                     # + does post-hoc mask clipping change the
                                                  #   L1-vs-L2 verdict? (cells only, ~1 min)
```

**A warning about the ranked objective.** `J` contains `DTMS`, a reward for a clean background,
and group-L1 zeroes the background by construction — so `J` structurally prefers L1. The fourth
panel of the regularization figure re-scores the same configurations on criteria without a
background term, and the verdict changes (L1 wins 89% on `J`, 56% on nRMSE, 19% on angular
error). Do not read `J`-ranked results as a recommendation between the two regularizer paths;
see the report's "What the objective was hiding".

**`cell_confinement.py` is stale and its figure is no longer in the report.** It sweeps
`fwd_mask_strength` 0→100 as if it were the old in-solver soft-penalty dial; commit `468cb38`
turned that parameter into an on/off gate for *post-hoc* clipping in `fttc.py` and deleted the
in-solver penalty from `forward_l1.py`. Its "soft" arm therefore now returns results identical
to the no-mask baseline, and only its post-hoc arm is meaningful (that arm is what the report's
confinement table quotes). Rewrite it against the current design before trusting it again.

Two analyses stand outside the report because they answer questions about the
*solver* rather than about the regimes, and both re-solve rather than reading the
cache: `compare_l2_reg.py` (GCV vs Bayesian vs GT-optimal λ on the L2 path) and
`compare_devices.py` (CPU/GPU parity).

`cell_confinement.py` remains the one analysis that consumes `cell_mask.npy`. It holds the
displacement (cached PIV window-24) and regularization (shipped default `l1=0.05`, `l2=0`)
fixed and varies **only** the mask: `fwd_mask_strength` 0→100 with the cell outline, against
the no-mask baseline, with the GT-support mask as an oracle ceiling. If the stage was
generated before `make_cells.py` saved `cell_mask.npy`, pass
`--scenarios-dir <benchmarkTFM>/benchmarks/scenarios` once to backfill the masks without
touching `reference.tif`/`deformed.tif` (so the displacement cache stays valid).

The diffuse-cell scenes (condition `cell_s6j1`) are staged from the benchmarkTFM synth cells by
`make_cells.py`: it takes each cell's fitted-fibre traction as GT, forward-projects it with this
pipeline's Green's operator, and rewarps the best-imaging bead stack at each strength. Run once before
the sweep (needs the benchmarkTFM scenarios locally):

```bash
python make_cells.py --scenarios-dir <benchmarkTFM>/benchmarks/scenarios \
                     --images-dir "$STAGE/images/tif_stacks" --stage "$STAGE"
# then the usual pipeline, scoped to the cell condition:
SCENE_GLOB="cell_s6j1/*" bash jobs/submit.sh pipeline    # on Maestro, after source env.sh
```

Figure map (script → files in `docs/images/`, and the report section they serve):

| script | figure | report section |
|---|---|---|
| `analyze.py` | `heuristic-sweep-competence.png` | Which method wins where |
| `analyze.py` | `heuristic-sweep-parameters.png` | Sparsity or smoothness? |
| `analyze.py` | `heuristic-sweep-regularization.png` | The cost of a wrong regularization; What the objective was hiding |
| `analyze.py` | `heuristic-sweep-examples.png` | What the ceiling looks like |
| `analyze.py` | `heuristic-sweep-imaging.png` | How imaging parameters set quality |
| `analyze.py` | `heuristic-sweep-cells.png` | Diffuse fields: realistic cells |
| `analyze.py` | `heuristic-sweep-edge-artifact.png` | Where the artifacts actually are |

Omit `--outdir` and each script writes to `figures/` (gitignored) instead, for
scratch runs that should not touch the committed report images.
