"""
SPT-PIV displacement analysis — isolated prototype script.

Algorithm:
  1. Detect beads in all frames (reference + deformed) with TrackPy (sub-pixel).
  2. Compute PIV coarse displacement field for every consecutive frame pair.
  3. Link beads globally with LapTrack using PIV-corrected distances.
  4. Keep only beads with complete tracks (present in reference + ALL deformed frames).
  5. Compute per-bead displacement as position_in_frame - position_in_reference.
  6. Visualise: red/green overlay with displacement arrows.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Callable

import sys
import numpy as np
import tifffile
import trackpy as tp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from laptrack import LapTrack

# Project preprocessing (background suppression + Gaussian smoothing)
sys.path.insert(0, "/home/aruppel/Projects/napariTFM")
from napariTFM.backend.preprocessing import ImageProcessor
_processor = ImageProcessor()

# ---------------------------------------------------------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------------------------------------------------------

# --- Paths -----------------------------------------------------------------
REFERENCE_PATH   = "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/reference.tif"
DEFORMED_PATHS   = [
    "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/deformed.tif",
]
PIXEL_SIZE_UM    = 0.1          # µm per pixel (used for axis labels only)
OUTPUT_DIR       = Path(__file__).parent / "output_mid"

# Optional: ground-truth displacement files for validation (set to None to skip)
GT_DISP_X_PATH   = "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/displacement_x.npy"
GT_DISP_Y_PATH   = "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/displacement_y.npy"

# --- Preprocessing ---------------------------------------------------------
# Percentile clipping before normalisation. Low end removes background floor;
# high end clips saturated clusters. Passed to apply_intensity_scaling().
PREPROCESS_MIN_PCT  = 80        # lower percentile clip
PREPROCESS_MAX_PCT  = 99.9      # upper percentile clip
PREPROCESS_GAUSSIAN = 1         # Gaussian sigma after scaling (0 = off)

# --- Bead detection (TrackPy) ----------------------------------------------
BEAD_DIAMETER    = 7            # px, must be odd; approximate bead size
BEAD_SEPARATION  = 8            # px, minimum centre-to-centre distance
BEAD_MINMASS     = "auto"       # float, or "auto" to estimate from image

# --- PIV -------------------------------------------------------------------
PIV_WIN_LARGE    = 64           # px, first-pass window
PIV_WIN_SMALL    = 32           # px, second-pass (refinement) window
PIV_STEP         = 16           # px, grid step (overlap = window - step)

# --- Linking (LapTrack) ----------------------------------------------------
# Max residual distance after PIV correction, in pixels.
# A bead pair whose PIV-corrected separation exceeds this is never linked.
SPT_RESIDUAL_RADIUS = 8         # px

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load and preprocess an image using the project's standard pipeline.

    Steps:
      1. Percentile-based intensity scaling [PREPROCESS_MIN_PCT, PREPROCESS_MAX_PCT]
         → clips background floor and hot pixels, normalises to [0, 1].
      2. Gaussian smoothing (sigma = PREPROCESS_GAUSSIAN, 0 = off).
    """
    img = tifffile.imread(path)
    processed, _ = _processor.apply_intensity_scaling(
        img.astype(np.float32), PREPROCESS_MIN_PCT, PREPROCESS_MAX_PCT
    )
    if PREPROCESS_GAUSSIAN > 0:
        processed = _processor.apply_gaussian_filter(processed, sigma=PREPROCESS_GAUSSIAN)
    return processed.astype(np.float32)


def auto_minmass(image: np.ndarray, diameter: int) -> float:
    """Estimate a reasonable minmass from image statistics.

    We locate features with minmass=0, look at the mass distribution,
    and return the value at the 30th percentile (keeps real beads, drops noise).
    """
    tp.quiet()
    raw = (image * 65535).astype(np.uint16)
    feats = tp.locate(raw, diameter=diameter, minmass=0, separation=BEAD_SEPARATION)
    if feats.empty:
        return 0.0
    return float(np.percentile(feats["mass"], 30))


def detect_beads(image: np.ndarray, minmass: float) -> pd.DataFrame:
    """Return sub-pixel bead positions as a DataFrame with columns x, y, mass."""
    tp.quiet()
    raw = (image * 65535).astype(np.uint16)
    feats = tp.locate(raw, diameter=BEAD_DIAMETER, minmass=minmass,
                      separation=BEAD_SEPARATION)
    return feats[["x", "y", "mass"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# PIV
# ---------------------------------------------------------------------------

def _xcorr_displacement(win_a: np.ndarray, win_b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel peak of normalised cross-correlation between two windows."""
    fa = np.fft.rfft2(win_a - win_a.mean())
    fb = np.fft.rfft2(win_b - win_b.mean())
    corr = np.fft.irfft2(fa * np.conj(fb))
    corr = np.fft.fftshift(corr)

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    cy, cx = peak

    # Sub-pixel refinement with 3-point Gaussian fit along each axis
    def gauss_peak(arr, idx):
        n = len(arr)
        if idx == 0 or idx == n - 1:
            return float(idx)
        num = np.log(arr[idx - 1] + 1e-9) - np.log(arr[idx + 1] + 1e-9)
        den = 2 * (np.log(arr[idx - 1] + 1e-9) - 2 * np.log(arr[idx] + 1e-9)
                   + np.log(arr[idx + 1] + 1e-9))
        return idx + (num / den if abs(den) > 1e-12 else 0.0)

    cy_sub = gauss_peak(corr[:, cx], cy)
    cx_sub = gauss_peak(corr[cy, :], cx)

    h, w = corr.shape
    dy = cy_sub - h // 2
    dx = cx_sub - w // 2
    return dx, dy


def _piv_pass(img_ref: np.ndarray, img_def: np.ndarray,
              win: int, step: int,
              init_gx: Optional[np.ndarray] = None,
              init_gy: Optional[np.ndarray] = None,
              init_dx: Optional[np.ndarray] = None,
              init_dy: Optional[np.ndarray] = None,
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single PIV pass. Returns (gx, gy, dx, dy) on a regular grid."""
    from scipy.interpolate import RegularGridInterpolator

    H, W = img_ref.shape
    half = win // 2

    xs = np.arange(half, W - half, step)
    ys = np.arange(half, H - half, step)
    gx, gy = np.meshgrid(xs, ys)
    dx_arr = np.zeros_like(gx, dtype=float)
    dy_arr = np.zeros_like(gy, dtype=float)

    # Build interpolators for the initial estimate (from a potentially
    # different grid produced by the first pass)
    if init_dx is not None:
        init_xs = init_gx[0, :]
        init_ys = init_gy[:, 0]
        interp_sx = RegularGridInterpolator(
            (init_ys, init_xs), init_dx,
            method="linear", bounds_error=False, fill_value=0.0)
        interp_sy = RegularGridInterpolator(
            (init_ys, init_xs), init_dy,
            method="linear", bounds_error=False, fill_value=0.0)
    else:
        interp_sx = interp_sy = None

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            # Pre-shift the deformed window by the initial estimate (second pass)
            if interp_sx is not None:
                shift_x = int(round(float(interp_sx([[y, x]])[0])))
                shift_y = int(round(float(interp_sy([[y, x]])[0])))
            else:
                shift_x, shift_y = 0, 0

            # Reference window
            r0, r1 = y - half, y + half
            c0, c1 = x - half, x + half
            win_r = img_ref[r0:r1, c0:c1]

            # Deformed window (shifted)
            dr0 = np.clip(r0 + shift_y, 0, H - win)
            dc0 = np.clip(c0 + shift_x, 0, W - win)
            win_d = img_def[dr0:dr0 + win, dc0:dc0 + win]

            if win_r.shape != (win, win) or win_d.shape != (win, win):
                continue

            ddx, ddy = _xcorr_displacement(win_r, win_d)
            dx_arr[i, j] = shift_x + ddx
            dy_arr[i, j] = shift_y + ddy

    return gx, gy, dx_arr, dy_arr


def compute_piv(img_ref: np.ndarray, img_def: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass PIV: coarse then refined. Returns (gx, gy, dx, dy)."""
    gx1, gy1, dx1, dy1 = _piv_pass(img_ref, img_def, PIV_WIN_LARGE, PIV_STEP)
    gx2, gy2, dx2, dy2 = _piv_pass(img_ref, img_def, PIV_WIN_SMALL, PIV_STEP,
                                    init_gx=gx1, init_gy=gy1,
                                    init_dx=dx1, init_dy=dy1)
    return gx2, gy2, dx2, dy2


def interpolate_piv(gx: np.ndarray, gy: np.ndarray,
                    dx: np.ndarray, dy: np.ndarray,
                    query_x: np.ndarray, query_y: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear interpolation of PIV field at arbitrary (query_x, query_y)."""
    from scipy.interpolate import RegularGridInterpolator

    xs = gx[0, :]
    ys = gy[:, 0]

    interp_dx = RegularGridInterpolator((ys, xs), dx, method="linear",
                                        bounds_error=False, fill_value=None)
    interp_dy = RegularGridInterpolator((ys, xs), dy, method="linear",
                                        bounds_error=False, fill_value=None)

    pts = np.column_stack([query_y, query_x])
    return interp_dx(pts), interp_dy(pts)


# ---------------------------------------------------------------------------
# LINKING
# ---------------------------------------------------------------------------

def build_piv_metric(piv_fields: List[Tuple]) -> Callable:
    """
    Return a LapTrack-compatible pairwise distance metric using PIV correction.

    piv_fields[n] = (gx, gy, dx, dy) for transition frame_n → frame_{n+1}.

    LapTrack calls metric(u, v) where u and v are individual 1-D coordinate
    vectors. We embed the frame index as the third element so the metric can
    look up the correct PIV field: u = [x, y, frame].
    Returns squared PIV-corrected distance (matching LapTrack's default
    sqeuclidean convention).
    """
    def metric(u: np.ndarray, v: np.ndarray) -> float:
        xa, ya, fa = float(u[0]), float(u[1]), int(round(u[2]))
        xb, yb, fb = float(v[0]), float(v[1]), int(round(v[2]))

        # Determine direction: earlier frame → later frame
        if fb - fa == 1:
            frame_from = fa
            x_from, y_from = xa, ya
            x_to, y_to = xb, yb
        elif fa - fb == 1:
            frame_from = fb
            x_from, y_from = xb, yb
            x_to, y_to = xa, ya
        else:
            return 1e12  # non-consecutive — should not happen with gap closing off

        gx, gy, piv_dx, piv_dy = piv_fields[frame_from]
        pred_dx, pred_dy = interpolate_piv(
            gx, gy, piv_dx, piv_dy,
            np.array([x_from]), np.array([y_from])
        )
        x_pred = x_from + pred_dx[0]
        y_pred = y_from + pred_dy[0]

        return (x_pred - x_to) ** 2 + (y_pred - y_to) ** 2

    return metric


def link_beads(all_coords: List[np.ndarray],
               piv_fields: List[Tuple]) -> pd.DataFrame:
    """
    Run LapTrack on per-frame bead coordinates with PIV-guided cost.
    Coordinates passed to LapTrack have shape (N, 3): [x, y, frame_index].
    The frame_index column is used by the metric to look up the PIV field;
    LapTrack's 'coordinate_cols' is restricted to x and y for display.
    Returns a DataFrame with columns: frame, track_id, x, y.
    """
    metric = build_piv_metric(piv_fields)
    cutoff = SPT_RESIDUAL_RADIUS ** 2

    lt = LapTrack(
        track_dist_metric=metric,
        track_cost_cutoff=cutoff,
        gap_closing_cost_cutoff=False,   # no gap closing — strict completeness
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )

    df = _coords_to_df(all_coords)
    track_df, _, _ = lt.predict_dataframe(
        df,
        coordinate_cols=["x", "y", "frame"],
        frame_col="frame",
        only_coordinate_cols=False,
    )
    # LapTrack returns a multi-index (frame, within-frame-index) and
    # renames the frame column to 'frame_y'. Flatten to a clean DataFrame.
    track_df = track_df.reset_index()   # brings both index levels into columns
    # After reset: columns include 'frame' (index level 0) and 'frame_y' (dup column)
    if "frame_y" in track_df.columns:
        track_df = track_df.drop(columns=["frame_y"])
    elif "frame" not in track_df.columns:
        # fallback: the multi-index level was named differently
        track_df = track_df.rename(columns={track_df.columns[0]: "frame"})
    return track_df[["frame", "x", "y", "track_id"]].reset_index(drop=True)


def _coords_to_df(all_coords: List[np.ndarray]) -> pd.DataFrame:
    rows = []
    for frame_idx, coords in enumerate(all_coords):
        for x, y in coords:
            rows.append({"frame": frame_idx, "x": x, "y": y})
    return pd.DataFrame(rows)


def filter_complete_tracks(track_df: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Keep only particles present in ALL n_frames frames, starting from frame 0."""
    counts = track_df.groupby("track_id")["frame"].nunique()
    complete = counts[counts == n_frames].index
    # Also require the track to include frame 0 (reference)
    has_ref = track_df[track_df["frame"] == 0].groupby("track_id").size()
    complete = complete.intersection(has_ref.index)
    return track_df[track_df["track_id"].isin(complete)].copy()


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def overlay_image(ref: np.ndarray, deformed: np.ndarray) -> np.ndarray:
    """
    Compose a red/green overlay:
      - Background (bright in both) → white
      - Beads in reference only      → red
      - Beads in deformed only        → green
      - Beads in both                 → dark (overlap cancels)
    Returns an RGB float image in [0, 1].
    """
    r = np.clip(ref, 0, 1)
    g = np.clip(deformed, 0, 1)
    b = np.minimum(r, g)        # white background: B channel = min(R, G)
    return np.stack([r, g, b], axis=-1)


def make_displacement_plot(ref: np.ndarray, deformed: np.ndarray,
                           bead_x: np.ndarray, bead_y: np.ndarray,
                           disp_x: np.ndarray, disp_y: np.ndarray,
                           title: str, out_path: Path,
                           gt_dx: Optional[np.ndarray] = None,
                           gt_dy: Optional[np.ndarray] = None) -> None:
    """
    Save a figure showing the red/green overlay with displacement arrows.
    If ground-truth is provided, shows a second panel with error map.
    """
    overlay = overlay_image(ref, deformed)

    n_panels = 2 if gt_dx is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: overlay + arrows -----------------------------------------
    ax = axes[0]
    ax.imshow(overlay, origin="upper")
    mag = np.hypot(disp_x, disp_y)
    scale = max(mag.max(), 1e-3)
    ax.quiver(bead_x, bead_y, disp_x, disp_y,
              mag, cmap="plasma", clim=(0, scale),
              angles="xy", scale_units="xy", scale=1,
              width=0.002, headwidth=4, headlength=4)
    ax.set_title(f"{title}\n{len(bead_x)} tracked beads")
    ax.axis("off")

    # --- Panel 2: error relative to ground truth ---------------------------
    if gt_dx is not None:
        from scipy.interpolate import griddata
        H, W = ref.shape

        # Interpolate GT to bead positions
        gy_coords, gx_coords = np.mgrid[0:H, 0:W]
        gt_at_beads_x = griddata(
            (gx_coords.ravel(), gy_coords.ravel()), gt_dx.ravel() / PIXEL_SIZE_UM,
            (bead_x, bead_y), method="linear"
        )
        gt_at_beads_y = griddata(
            (gx_coords.ravel(), gy_coords.ravel()), gt_dy.ravel() / PIXEL_SIZE_UM,
            (bead_x, bead_y), method="linear"
        )
        err = np.hypot(disp_x - gt_at_beads_x, disp_y - gt_at_beads_y)

        ax2 = axes[1]
        sc = ax2.scatter(bead_x, bead_y, c=err, cmap="hot_r", s=4,
                         vmin=0, vmax=np.percentile(err, 95))
        plt.colorbar(sc, ax=ax2, label="displacement error (px)")
        ax2.set_xlim(0, W)
        ax2.set_ylim(H, 0)
        ax2.set_aspect("equal")
        ax2.set_title("Error vs ground truth")
        ax2.axis("off")

        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        median_err = float(np.nanmedian(err))
        print(f"  RMSE vs GT: {rmse:.3f} px  |  median: {median_err:.3f} px")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load all images ---------------------------------------------------
    print("Loading images...")
    ref_img = load_image(REFERENCE_PATH)
    def_imgs = [load_image(p) for p in DEFORMED_PATHS]
    all_imgs = [ref_img] + def_imgs
    n_frames = len(all_imgs)

    # Optional ground truth
    gt_dx = np.load(GT_DISP_X_PATH) if GT_DISP_X_PATH else None
    gt_dy = np.load(GT_DISP_Y_PATH) if GT_DISP_Y_PATH else None

    # --- Bead detection ----------------------------------------------------
    if BEAD_MINMASS == "auto":
        print("Estimating minmass from reference image...")
        minmass = auto_minmass(ref_img, BEAD_DIAMETER)
        print(f"  minmass = {minmass:.1f}")
    else:
        minmass = float(BEAD_MINMASS)

    print("Detecting beads...")
    all_detections = []
    for i, img in enumerate(all_imgs):
        feats = detect_beads(img, minmass)
        all_detections.append(feats)
        label = "reference" if i == 0 else f"deformed[{i-1}]"
        print(f"  {label}: {len(feats)} beads detected")

    all_coords = [df[["x", "y"]].values for df in all_detections]

    # --- PIV ---------------------------------------------------------------
    print("Computing PIV fields...")
    piv_fields = []
    for i in range(n_frames - 1):
        label = f"frame {i} → {i+1}"
        print(f"  PIV {label}...")
        gx, gy, dx, dy = compute_piv(all_imgs[i], all_imgs[i + 1])
        piv_fields.append((gx, gy, dx, dy))
        print(f"    max PIV displacement: {np.hypot(dx, dy).max():.2f} px")

    # --- Linking -----------------------------------------------------------
    print("Linking beads (LapTrack)...")
    track_df = link_beads(all_coords, piv_fields)
    n_before = track_df["track_id"].nunique()

    track_df = filter_complete_tracks(track_df, n_frames)
    n_after = track_df["track_id"].nunique()
    print(f"  Tracks before completeness filter: {n_before}")
    print(f"  Complete tracks (present in all {n_frames} frames): {n_after}")

    # --- Displacement calculation ------------------------------------------
    print("Computing displacements...")
    ref_positions = (track_df[track_df["frame"] == 0]
                     .set_index("track_id")[["x", "y"]])

    results = []   # one row per (track, frame)
    for frame_idx in range(1, n_frames):
        frame_rows = (track_df[track_df["frame"] == frame_idx]
                      .set_index("track_id")[["x", "y"]])
        common = ref_positions.index.intersection(frame_rows.index)
        rx = ref_positions.loc[common, "x"].values
        ry = ref_positions.loc[common, "y"].values
        fx = frame_rows.loc[common, "x"].values
        fy = frame_rows.loc[common, "y"].values
        for tid, bx, by, ddx, ddy in zip(common,
                                          rx, ry,
                                          fx - rx, fy - ry):
            results.append({
                "track_id": tid, "frame": frame_idx,
                "ref_x": bx, "ref_y": by,
                "disp_x": ddx, "disp_y": ddy,
                "disp_mag": np.hypot(ddx, ddy),
            })

    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_path = OUTPUT_DIR / "displacements.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Displacements saved to {csv_path}")

    # --- Visualisation -------------------------------------------------------
    print("Generating plots...")
    for frame_idx in range(1, n_frames):
        frame_res = results_df[results_df["frame"] == frame_idx]
        out_path = OUTPUT_DIR / f"displacement_frame{frame_idx:03d}.png"

        make_displacement_plot(
            ref=ref_img,
            deformed=all_imgs[frame_idx],
            bead_x=frame_res["ref_x"].values,
            bead_y=frame_res["ref_y"].values,
            disp_x=frame_res["disp_x"].values,
            disp_y=frame_res["disp_y"].values,
            title=f"SPT-PIV displacement — frame {frame_idx}",
            out_path=out_path,
            gt_dx=gt_dx,
            gt_dy=gt_dy,
        )

    print("Done.")


if __name__ == "__main__":
    main()
