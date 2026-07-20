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

## Two-stage decomposition (see `sweep_config.py`)

1. **Displacement caching** (`build_cache.py`): per scene, sweep the *resolution*
   knob. At each resolution, raise the *convergence* knob until the field stops
   changing (`CONV_TOL`) and cache that converged field. Convergence is set by a
   convergence criterion, **not** by force score: an under-converged field is an
   accidental second regularizer, and we want L1+L2 to do all the regularizing.
2. **Force sweep** (`sweep_forces.py`): read the cached fields; for each, invert
   over the full `FRAC1 × FRAC2` grid; upsample each recovery to the common
   reference grid; record J and its components (+ nRMSE, corr). Resolution stays
   *in* this search — one cached field per resolution, J adjudicates resolution
   jointly with `(l1, l2)`.

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

## Layout

```
$STAGE/ (= /helix/…/tfm_heuristic, set in env.sh, NOT in the repo)
  images/tif_stacks/{scenario*_dens*_NA*_expo*.tif,manifest.csv}          ← make_stacks.py
  scenes/<condition>/<scene_id>/{reference.tif,deformed.tif,scene.toml}  ← make_scenes.py
  cache/<condition>/<scene_id>/disp_res<k>.npz                            ← build_cache.py
  results/                                                                ← sweep_forces.py
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
bash jobs/submit.sh cache        # displacement cache (GPU array over scenes)
bash jobs/submit.sh sweep        # force sweep (after cache completes)
sbatch jobs/sweep.sbatch        # force sweep -> results/
```

## Reproducing the report

The findings are written up in
[`docs/specs/2026-07-20-heuristic-sweep-method-parameter-selection.md`](../../docs/specs/2026-07-20-heuristic-sweep-method-parameter-selection.md).
Every figure in that report is regenerated by a script here, from the same
`results/*.csv` shards the sweep produces. The scripts read the stage from
`$STAGE` (or `--stage`), so `source env.sh` first; no path is baked in.

The cross-method sweep caches all three displacement methods per scene
(`disp_<method>_res<k>.npz`), so `results/` carries a `method` column.

```bash
source env.sh                                    # sets STAGE (see above)
IMG=../../docs/images                            # write straight into the report's images

python aggregate.py       --outdir "$IMG"        # competence + winners, parameter heuristics
python imaging_quality.py --outdir "$IMG"        # imaging-parameter drivers, recoverable envelope
python compare_reg.py     --outdir "$IMG"        # L1 sensitivity vs parameter-free Bayesian-L2 (~15 min)
python compare_methods.py --outdir "$IMG"        # illustrative best-J recoveries per method (dipoles)
python cell_aggregate.py  --outdir "$IMG"        # diffuse-cell competence + L1 heuristic transfer
python cell_examples.py   --outdir "$IMG"        # illustrative cell recoveries per method
```

The diffuse-cell scenes (condition `cell_s6j1`) are staged from the benchmarkTFM synth cells by
`make_cells.py`: it takes each cell's fitted-fibre traction as GT, forward-projects it with this
pipeline's Green's operator, and rewarps the best-imaging bead stack at each strength. Run once before
the sweep (needs the benchmarkTFM scenarios locally):

```bash
python make_cells.py --scenarios-dir <benchmarkTFM>/benchmarks/scenarios \
                     --images-dir "$STAGE/images/tif_stacks" --stage "$STAGE"
# then the usual two-stage pipeline, scoped to the cell condition:
SCENE_GLOB="cell_s6j1/*" bash jobs/submit.sh pipeline    # on Maestro, after source env.sh
```

Figure map (script → files in `docs/images/`, and the report section they serve):

| script | figures | report section |
|---|---|---|
| `aggregate.py` | `heuristic-sweep-competence.png`, `heuristic-sweep-parameters.png` | Which method wins where; Regularization |
| `imaging_quality.py` | `heuristic-sweep-imaging-drivers.png`, `heuristic-sweep-imaging-envelope.png` | Imaging parameters; Recoverable envelope |
| `compare_reg.py` | `heuristic-sweep-regularization-sensitivity.png` (+ `reg_compare.csv`) | The cost of a wrong L1 |
| `compare_methods.py` | `heuristic-sweep-examples.png` | What winning looks like |
| `cell_aggregate.py` | `heuristic-sweep-cells-competence.png` | Diffuse fields: realistic cells |
| `cell_examples.py` | `heuristic-sweep-cells-examples.png` | Diffuse fields: realistic cells |

Omit `--outdir` and each script writes to `figures/` (gitignored) instead, for
scratch runs that should not touch the committed report images.
