"""Central configuration for the TFM regularization-heuristic sweep.

One place to define every axis of the experiment, so `build_cache.py` and
`sweep_forces.py` can never disagree about the grid.

Design (agreed 2026-07-20):
  * Force nRMSE vs an *analytic* ground-truth traction is the ONLY objective.
    Displacement error is never scored on its own -- it does not propagate
    uniformly through the DC-stripped, band-limited Green's operator, so it is
    a misleading proxy. See README.md.
  * Convergence knobs are set by a convergence *criterion* (self-consistency),
    NOT by force score -- an under-converged displacement field would act as a
    second, opaque regularizer on top of L1+L2 and poison the heuristic.
  * The resolution knob DOES stay in the force-scored search: one cached field
    per resolution, force nRMSE adjudicates resolution jointly with (l1, l2).
  * Ground truth is a *balanced pair* (two equal-and-opposite adhesions), never
    a single monopole: forward_l1's Green's operator zeroes the DC (k=0) mode,
    so a net-force monopole is unrecoverable by construction.
"""
from __future__ import annotations

# --- imaging conditions -----------------------------------------------------
# Start with the ceiling only (realistic, undegraded). The jitter+camera-noise
# degradation ladder slots in here later as extra condition labels.
CONDITIONS = ["realistic"]

# --- displacement stage -----------------------------------------------------
DISP_METHOD = "PIV"          # bridge used PIV; ILK/FFD parameterized below

# Resolution knob: swept, one cached field per value, kept in the force search.
RES_KNOB = {"PIV": "piv_window", "ILK": "ilk_radius", "FFD": "ffd_level_spacing"}
RES_VALUES = {
    "PIV": [8, 12, 16, 24, 32],
    "ILK": [5, 7, 11, 15],
    "FFD": [8.0, 12.0, 18.0, 24.0],
}

# Convergence knob: raised along this ladder until the field stops changing;
# the converged setting is cached. Not force-scored.
CONV_KNOB = {"PIV": "piv_passes", "ILK": "ilk_num_warp", "FFD": "ffd_num_iters"}
CONV_LADDER = {
    "PIV": [2, 4, 6, 8, 12, 16],
    "ILK": [4, 8, 12, 20],
    "FFD": [25, 50, 100, 200],
}
CONV_TOL = 0.01              # rel. L2 change between successive settings -> converged
DOWNSCALE_FACTOR = 4         # global binning, held fixed (700 -> ~175 grid)

# --- forward / substrate ----------------------------------------------------
YOUNG_MODULUS = 20000.0      # Pa
POISSON = 0.5
PIXEL_SIZE_UM = 0.1          # camera pixel size before binning

# --- regularization search (elastic net) ------------------------------------
import numpy as np
FRAC1 = [round(x, 4) for x in np.geomspace(0.02, 0.4, 8)]        # L1 sparsity
FRAC2 = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]        # L2 ridge (0 = pure L1)
L1_MAX_ITER = 400

# --- ground-truth traction model (balanced pair) ----------------------------
# Each pole is a uniform-traction disc; the two poles are equal and opposite,
# directed along the pair axis. GT_TRACTION_PROFILE names the radial profile so
# the generator and the scorer rasterize the *same* field.
GT_TRACTION_PROFILE = "tophat"   # {"tophat", "gaussian"}

# Reference grid on which every recovered field is compared to GT, so the
# resolution choice is a fair comparison (not biased toward coarse grids).
GT_REFERENCE_SIZE = 700          # px, matches the source image size
