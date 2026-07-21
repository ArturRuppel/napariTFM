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
loads `gt_traction.npy` directly (see `sweep_forces.py`, `meta.kind == "cell"`).

| file             | meaning                                                            |
|------------------|--------------------------------------------------------------------|
| `reference.tif`  | frame 0 of the scenario-6 stack (zero-jitter reference)            |
| `deformed.tif`   | `warp(frame 1, u)` — mild registration jitter + photon noise ride along |
| `gt_traction.npy`| **authoritative** GT traction `(2, N, N)` Pa (scaled fibre field)  |
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
see `field_metrics` in `sweep_forces.py`) are recorded as a diagnostic decomposition
of that error, not the ranking key. Everything downstream (cache, sweep, resolution
search) is identical to the dipole run; only the GT construction and the ranking
metric differ.

## Layout

```
$STAGE/ (= /helix/…/tfm_heuristic, set in env.sh, NOT in the repo)
  images/tif_stacks/{scenario*_dens*_NA*_expo*.tif,manifest.csv}          ← make_stacks.py
  scenes/<condition>/<scene_id>/{reference.tif,deformed.tif,scene.toml}  ← make_scenes.py (dipole)
  scenes/cell_s6j1/<scene_id>/{reference.tif,deformed.tif,gt_traction.npy,scene.toml}  ← make_cells.py
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
python cell_compare_reg.py --outdir "$IMG"       # L1 vs parameter-free Bayesian-L2 on cells
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
| `cell_compare_reg.py` | `heuristic-sweep-cells-regularization.png` (+ `cell_reg_compare.csv`) | Does smoothness win on diffuse fields? |

Omit `--outdir` and each script writes to `figures/` (gitignored) instead, for
scratch runs that should not touch the committed report images.
