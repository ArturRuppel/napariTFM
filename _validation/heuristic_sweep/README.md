# TFM regularization-heuristic sweep

Goal: fit a *heuristic* for the elastic-net regularization (L1 sparsity `l1`,
L2 ridge `l2`) of the traction reconstruction, from ground truth, across a grid
of adhesion footprints and magnitudes and (later) imaging conditions. The bridge
benchmark told us the *shape* — `l1` pins near ~0.11, `l2` tracks inverse SNR —
but three opaque noise points can't fit a curve. This replaces it with a
generative ladder that has clean, analytic ground truth on a common grid.

## The one objective: force nRMSE

Only the recovered **traction** is scored, against an analytic GT traction
rasterized on a common `GT_REFERENCE_SIZE` grid: `nRMSE = ‖t − t_GT‖ / ‖t_GT‖`.
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
   reference grid; record force nRMSE. Resolution stays *in* this search — one
   cached field per resolution, force nRMSE adjudicates resolution jointly with
   `(l1, l2)`.

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
  scenes/<condition>/<scene_id>/{reference.tif,deformed.tif,scene.toml}  ← generator
  cache/<condition>/<scene_id>/disp_res<k>.npz                            ← build_cache.py
  results/                                                                ← sweep_forces.py
  logs/                                                                   ← SLURM out/err
  code/                                                                   ← synced repo (sync_code.sh)
```

Tracked code lives in the repo (`_validation/heuristic_sweep/`); paths come from
`$STAGE` at runtime so no infra path is baked into the public repo.

## Running on Maestro (later — nothing here runs on its own)

```bash
# from local, once the generator has populated scenes/:
bash jobs/sync_code.sh          # stage repo code + napariTFM pkg -> $STAGE/code
ssh aruppel@maestro.pasteur.fr
source /helix/…/tfm_heuristic/code/env.sh   # sets STAGE, PYTHONPATH
bash jobs/submit.sh cache        # displacement cache (GPU array over scenes)
bash jobs/submit.sh sweep        # force sweep (after cache completes)
sbatch jobs/sweep.sbatch        # force sweep -> results/
```
