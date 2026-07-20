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
# The condition ladder is now the synthetic scenario stacks under IMAGES_DIR: 8
# scenarios sampling (bead density x NA/PSF x exposure/SNR), each a (4,512,512)
# TYX stack whose frame 0 is the zero-jitter reference and frames 1-3 add
# registration jitter [0.067, 0.133, 0.2] px. Each (scenario, jitter-frame) pair
# is one condition -> one full footprint x disp sweep. make_scenes enumerates
# them from disk and labels them "s<idx>_j<frame>"; CONDITIONS below is only a
# fallback default for single-condition runs.
CONDITIONS = ["realistic"]
REF_FRAME = 0                    # frame 0 = zero-jitter reference (per stack metadata)
JITTER_FRAMES = [1, 3]           # deform-source frames -> 2 conditions/scenario (mild/severe jitter)

# --- displacement stage -----------------------------------------------------
DISP_METHOD = "PIV"          # bridge used PIV; ILK/FFD parameterized below

# Resolution knob: the displacement method's INTERNAL spatial-resolution control,
# swept at a FIXED output grid (downscale_factor is a fixed pipeline convention,
# never swept). For PIV that is piv_window (smaller = finer/noisier, larger =
# coarser/smoother). At high SNR the window barely matters; at low SNR it is the
# bias-variance tradeoff -- so the axis earns its keep across the noisy scenes.
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
YOUNG_MODULUS = 1000.0       # Pa (1 kPa, calibration substrate)
POISSON = 0.5
PIXEL_SIZE_UM = 0.1612       # synthetic-stack pitch (pitch_um in the TIFF metadata)

# --- regularization search (elastic net) ------------------------------------
import numpy as np
FRAC1 = [round(x, 4) for x in np.geomspace(0.02, 0.4, 8)]        # L1 sparsity
FRAC2 = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]        # L2 ridge (0 = pure L1)
L1_MAX_ITER = 400

# --- ground-truth traction model (balanced pair) ----------------------------
# Each pole is a Gaussian traction spot (sigma = footprint); the two poles are
# equal and opposite (contractile, pointing inward). Gaussian, not tophat, so the
# sub-pixel footprints (0.1 µm = 0.6 px) rasterize to something nonzero.
GT_TRACTION_PROFILE = "gaussian"   # {"tophat", "gaussian"}

# Reference grid on which every recovered field is compared to GT, so the
# resolution choice is a fair comparison (not biased toward coarse grids).
GT_REFERENCE_SIZE = 512          # px, matches the synthetic-stack size

# --- scenario generation (make_scenes.py) -----------------------------------
# Each scene is a CROSS-FRAME pair from one synthetic scenario stack: reference =
# frame 0 (zero jitter), deformed = warp(frame_k, u) for a jitter frame k in
# JITTER_FRAMES. The registration jitter + photon noise between frame 0 and frame k
# rides along as genuine reference-vs-deformed noise; GT stays u. No seed axis.
BEAD_CHANNEL = 0                 # synthetic stacks are single-channel (TYX); channel 0
CROP_SIZE = GT_REFERENCE_SIZE    # synthetic stacks are already 512²; no crop needed
# The grid is (footprint x PEAK DISPLACEMENT): peak displacement is the natural SNR
# axis, and the traction magnitude is DERIVED per cell to hit it (Green's op is
# linear, so one forward solve per footprint sets the scale). E and separation are
# absorbed into the derived magnitude.
FOOTPRINTS_UM = [round(x, 3) for x in np.geomspace(0.1, 5.0, 5)]    # Gaussian sigma, µm
PEAK_DISP_PX = [round(x, 3) for x in np.geomspace(0.5, 50.0, 6)]    # target |u|max, px
SEPARATION_UM = 30.0             # centre-to-centre; on 512 keeps boundary leak ~8% (vs 11% at 40)
AXIS_DEG = 45.0

