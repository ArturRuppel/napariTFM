"""
SPT displacement analysis — optical flow + bead refinement.

Algorithm:
  1. Detect beads ONCE in the reference image only.
  2. Compute optical flow (ref → each deformed frame) using OpenCV DIS
     (TV-L1 requires opencv-contrib which is not installed in this env).
  3. For each ref bead, sample the flow locally → predicted position in deformed.
  4. Refine each predicted position against the deformed image using
     TrackPy's iterative centre-of-mass (tp.refine_com) → sub-pixel accuracy.
  5. Quality filter: residual distance, eccentricity, mass.
  6. Resolve duplicate assignments (two beads converging to the same peak).
  7. Displacement = refined_position − reference_position.
  8. Visualise: red/green overlay with displacement arrows.

No second detection step — beads are detected only in the reference and
find their own homes in each deformed frame.
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
sys.path.insert(0, "/home/aruppel/Projects/napariTFM")
from napariTFM.backend.preprocessing import ImageProcessor

_processor = ImageProcessor()
tp.quiet()

# ---------------------------------------------------------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------------------------------------------------------

# --- Paths -----------------------------------------------------------------
REFERENCE_PATH  = "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/reference.tif"
DEFORMED_PATHS  = [
    "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/deformed.tif",
]
PIXEL_SIZE_UM   = 0.1
OUTPUT_DIR      = Path(__file__).parent / "output_spt_mid"

GT_DISP_X_PATH  = "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/displacement_x.npy"
GT_DISP_Y_PATH  = "/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM/mid/displacement_y.npy"

# --- Preprocessing ---------------------------------------------------------
PREPROCESS_MIN_PCT  = 80
PREPROCESS_MAX_PCT  = 99.9
PREPROCESS_GAUSSIAN = 1

# --- Bead detection (reference only) --------------------------------------
BEAD_DIAMETER       = 5       # px, must be odd
BEAD_SEPARATION     = 2       # px, minimum centre-to-centre distance
BEAD_MINMASS        = "auto"  # float or "auto"
TRACKPY_PREPROCESS  = True    # keep TrackPy internal bandpass ON

# --- Optical flow sampling ------------------------------------------------
FLOW_SAMPLE_WINDOW  = 7       # px, size of local averaging window at each bead

# --- Position refinement --------------------------------------------------
REFINE_RADIUS       = 5       # px, half-size of the refinement window in deformed
REFINE_MAX_ITER     = 20      # max iterations for centre-of-mass walk

# --- Quality filters ------------------------------------------------------
RESIDUAL_RADIUS     = 5.0     # px, max |refined − predicted|; else rejected
MAX_ECCENTRICITY    = 0.5     # 0 = circular, 1 = line; beads should be near 0
MIN_MASS_FRACTION   = 0.2     # min mass as fraction of median ref bead mass

# --- Duplicate resolution -------------------------------------------------
DUPLICATE_RADIUS    = 2.0     # px, two refined positions closer than this = duplicate

# ---------------------------------------------------------------------------
# IMAGE LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load and preprocess: percentile intensity scaling + Gaussian smoothing."""
    img = tifffile.imread(path)
    processed, _ = _processor.apply_intensity_scaling(
        img.astype(np.float32), PREPROCESS_MIN_PCT, PREPROCESS_MAX_PCT
    )
    if PREPROCESS_GAUSSIAN > 0:
        processed = _processor.apply_gaussian_filter(processed, PREPROCESS_GAUSSIAN)
    return processed.astype(np.float32)


def to_uint16(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return ((img - lo) / (hi - lo + 1e-9) * 65535).astype(np.uint16)

# ---------------------------------------------------------------------------
# BEAD DETECTION (reference only)
# ---------------------------------------------------------------------------

def auto_minmass(img_u16: np.ndarray) -> float:
    """Estimate minmass from the 30th percentile of detected feature masses."""
    feats = tp.locate(img_u16, diameter=BEAD_DIAMETER, minmass=0,
                      separation=BEAD_SEPARATION, preprocess=TRACKPY_PREPROCESS)
    return float(feats["mass"].quantile(0.30)) if len(feats) else 0.0


def detect_beads(img_u16: np.ndarray, minmass: float) -> pd.DataFrame:
    """Detect beads in reference image. Returns DataFrame with x, y, mass."""
    feats = tp.locate(img_u16, diameter=BEAD_DIAMETER, minmass=minmass,
                      separation=BEAD_SEPARATION, preprocess=TRACKPY_PREPROCESS)
    return feats[["x", "y", "mass"]].reset_index(drop=True)

# ---------------------------------------------------------------------------
# OPTICAL FLOW
# ---------------------------------------------------------------------------

def compute_optical_flow(ref: np.ndarray, deformed: np.ndarray) -> np.ndarray:
    """Dense optical flow using OpenCV DISOpticalFlow (Dense Inverse Search).

    DIS outperforms TV-L1 on sparse bead images — TV-L1's smoothness prior
    over-regularises at the bead scale. MEDIUM preset balances speed/accuracy.

    Returns flow of shape (H, W, 2):
      flow[:,:,0] = dx  (column displacement, pixels)
      flow[:,:,1] = dy  (row displacement, pixels)
    """
    def to_u8(img):
        lo, hi = img.min(), img.max()
        return ((img - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(to_u8(ref), to_u8(deformed), None)


def sample_flow_at_beads(flow: np.ndarray,
                         bead_x: np.ndarray,
                         bead_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Local-average the flow field in a small window around each bead."""
    H, W = flow.shape[:2]
    half = FLOW_SAMPLE_WINDOW // 2
    dx = np.zeros(len(bead_x))
    dy = np.zeros(len(bead_x))
    for i, (x, y) in enumerate(zip(bead_x, bead_y)):
        cx, cy = int(round(float(x))), int(round(float(y)))
        y0, y1 = max(0, cy - half), min(H, cy + half + 1)
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        dx[i] = float(flow[y0:y1, x0:x1, 0].mean())
        dy[i] = float(flow[y0:y1, x0:x1, 1].mean())
    return dx, dy

# ---------------------------------------------------------------------------
# POSITION REFINEMENT
# ---------------------------------------------------------------------------

def refine_positions(deformed_u16: np.ndarray,
                     pred_x: np.ndarray,
                     pred_y: np.ndarray) -> pd.DataFrame:
    """Refine predicted positions against the deformed image.

    Uses tp.refine_com: iterative centre-of-mass walk toward nearest
    intensity peak, then sub-pixel centroid fit.

    Predictions that land on empty/zero-padded regions (e.g. outside the
    warped image boundary) are detected upfront and returned with NaN so
    the quality filter can reject them cleanly.

    Returns DataFrame with columns: x, y, mass, size, ecc, signal, raw_mass.
    Rows correspond 1-to-1 with the input pred_x/pred_y arrays.
    """
    H, W = deformed_u16.shape
    r = REFINE_RADIUS

    clipped_x = np.clip(pred_x, r, W - r - 1)
    clipped_y = np.clip(pred_y, r, H - r - 1)

    # Pre-filter: reject positions whose refinement window is entirely dark
    # (happens at warped-image boundaries where pixel values are padded to 0).
    valid = np.zeros(len(pred_x), dtype=bool)
    for i, (x, y) in enumerate(zip(clipped_x, clipped_y)):
        cx, cy = int(round(x)), int(round(y))
        patch = deformed_u16[cy - r: cy + r + 1, cx - r: cx + r + 1]
        if patch.size > 0 and int(patch.max()) > 0:
            valid[i] = True

    # Build NaN-filled result for the invalid ones
    nan_row = {c: np.nan for c in ["x", "y", "mass", "size", "ecc", "signal", "raw_mass"]}
    result_rows = [dict(nan_row) for _ in range(len(pred_x))]

    if valid.any():
        coords = pd.DataFrame({
            "x": clipped_x[valid],
            "y": clipped_y[valid],
        })
        refined = tp.refine_com(
            raw_image=deformed_u16,
            image=deformed_u16,
            radius=r,
            coords=coords,
            max_iterations=REFINE_MAX_ITER,
            characterize=True,
            engine="python",   # avoids numba zero-division on dark patches
        ).reset_index(drop=True)

        valid_idx = np.where(valid)[0]
        for out_i, row in zip(valid_idx, refined.itertuples(index=False)):
            result_rows[out_i] = row._asdict()

    return pd.DataFrame(result_rows)

# ---------------------------------------------------------------------------
# QUALITY FILTERING
# ---------------------------------------------------------------------------

def quality_filter(ref_beads: pd.DataFrame,
                   refined: pd.DataFrame,
                   pred_x: np.ndarray,
                   pred_y: np.ndarray,
                   ref_median_mass: float) -> np.ndarray:
    """Return boolean mask of accepted beads (True = keep).

    Rejects beads where:
      - refined position moved more than RESIDUAL_RADIUS from prediction
      - eccentricity > MAX_ECCENTRICITY  (non-circular → cluster or noise)
      - mass < MIN_MASS_FRACTION * ref_median_mass  (bead not actually there)
    """
    residual = np.hypot(refined["x"].values - pred_x,
                        refined["y"].values - pred_y)
    keep = (
        (residual <= RESIDUAL_RADIUS) &
        (refined["ecc"].values <= MAX_ECCENTRICITY) &
        (refined["mass"].values >= MIN_MASS_FRACTION * ref_median_mass)
    )
    return keep


def resolve_duplicates(refined_x: np.ndarray,
                       refined_y: np.ndarray,
                       pred_x: np.ndarray,
                       pred_y: np.ndarray) -> np.ndarray:
    """For any two refined positions within DUPLICATE_RADIUS of each other,
    keep the one with smaller |refined − predicted| and reject the other.

    Returns boolean mask of keepers (True = keep).
    """
    residuals = np.hypot(refined_x - pred_x, refined_y - pred_y)
    keep = np.ones(len(refined_x), dtype=bool)

    if len(refined_x) == 0:
        return keep

    tree = cKDTree(np.column_stack([refined_x, refined_y]))
    pairs = tree.query_pairs(DUPLICATE_RADIUS)
    for i, j in pairs:
        if not keep[i] or not keep[j]:
            continue
        # Reject the one further from its prediction
        if residuals[i] <= residuals[j]:
            keep[j] = False
        else:
            keep[i] = False
    return keep

# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def overlay_image(ref: np.ndarray, deformed: np.ndarray) -> np.ndarray:
    """Red/green overlay: white background, coloured where beads diverge."""
    r = np.clip(ref, 0, 1)
    g = np.clip(deformed, 0, 1)
    b = np.minimum(r, g)
    return np.stack([r, g, b], axis=-1)


def make_displacement_plot(ref: np.ndarray,
                           deformed: np.ndarray,
                           bead_x: np.ndarray,
                           bead_y: np.ndarray,
                           disp_x: np.ndarray,
                           disp_y: np.ndarray,
                           title: str,
                           out_path: Path,
                           gt_dx_um: Optional[np.ndarray] = None,
                           gt_dy_um: Optional[np.ndarray] = None) -> None:
    overlay = overlay_image(ref, deformed)
    H, W = ref.shape
    n_panels = 2 if gt_dx_um is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    # Panel 1 — overlay + arrows
    ax = axes[0]
    ax.imshow(overlay, origin="upper")
    mag = np.hypot(disp_x, disp_y)
    if mag.max() > 1e-6:
        ax.quiver(bead_x, bead_y, disp_x, disp_y,
                  mag, cmap="plasma", clim=(0, mag.max()),
                  angles="xy", scale_units="xy", scale=1,
                  width=0.002, headwidth=4, headlength=4)
    ax.set_title(f"{title}\n{len(bead_x)} beads")
    ax.axis("off")

    # Panel 2 — error vs ground truth
    if gt_dx_um is not None:
        from scipy.interpolate import RegularGridInterpolator
        gy = np.arange(H, dtype=float)
        gx = np.arange(W, dtype=float)
        gt_dx_px = gt_dx_um / PIXEL_SIZE_UM
        gt_dy_px = gt_dy_um / PIXEL_SIZE_UM
        interp_x = RegularGridInterpolator((gy, gx), gt_dx_px,
                                           bounds_error=False, fill_value=np.nan)
        interp_y = RegularGridInterpolator((gy, gx), gt_dy_px,
                                           bounds_error=False, fill_value=np.nan)
        pts = np.column_stack([bead_y, bead_x])
        gt_at_x = interp_x(pts)
        gt_at_y = interp_y(pts)
        err = np.hypot(disp_x - gt_at_x, disp_y - gt_at_y)

        ax2 = axes[1]
        sc = ax2.scatter(bead_x, bead_y, c=err, cmap="hot_r", s=5,
                         vmin=0, vmax=np.nanpercentile(err, 95))
        plt.colorbar(sc, ax=ax2, label="error (px)")
        ax2.set_xlim(0, W); ax2.set_ylim(H, 0); ax2.set_aspect("equal")
        ax2.set_title("Error vs ground truth"); ax2.axis("off")

        rmse   = float(np.sqrt(np.nanmean(err ** 2)))
        median = float(np.nanmedian(err))
        print(f"  RMSE vs GT: {rmse:.3f} px  |  median: {median:.3f} px")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load images
    print("Loading images...")
    ref_img  = load_image(REFERENCE_PATH)
    def_imgs = [load_image(p) for p in DEFORMED_PATHS]
    ref_u16  = to_uint16(ref_img)

    gt_dx_um = np.load(GT_DISP_X_PATH) if GT_DISP_X_PATH else None
    gt_dy_um = np.load(GT_DISP_Y_PATH) if GT_DISP_Y_PATH else None

    # Detect beads in reference
    print("Detecting beads in reference...")
    if BEAD_MINMASS == "auto":
        minmass = auto_minmass(ref_u16)
        print(f"  auto minmass = {minmass:.1f}")
    else:
        minmass = float(BEAD_MINMASS)

    ref_beads = detect_beads(ref_u16, minmass)
    ref_median_mass = float(ref_beads["mass"].median())
    print(f"  {len(ref_beads)} beads detected")

    all_results = []

    for frame_idx, (def_img, def_path) in enumerate(zip(def_imgs, DEFORMED_PATHS), start=1):
        print(f"\nProcessing frame {frame_idx} ({Path(def_path).parent.name})...")
        def_u16 = to_uint16(def_img)

        # Optical flow
        print("  Computing optical flow...")
        flow = compute_optical_flow(ref_img, def_img)
        flow_mag = np.hypot(flow[:,:,0], flow[:,:,1])
        print(f"  Flow: mean={flow_mag.mean():.2f} px, max={flow_mag.max():.2f} px")

        # Predict positions
        bead_x = ref_beads["x"].values
        bead_y = ref_beads["y"].values
        flow_dx, flow_dy = sample_flow_at_beads(flow, bead_x, bead_y)
        pred_x = bead_x + flow_dx
        pred_y = bead_y + flow_dy

        # Refine against deformed image
        print("  Refining positions...")
        refined = refine_positions(def_u16, pred_x, pred_y)

        # Quality filter
        keep_q = quality_filter(ref_beads, refined, pred_x, pred_y, ref_median_mass)
        print(f"  Quality filter: {keep_q.sum()} / {len(keep_q)} passed")

        # Resolve duplicates (only among already-kept beads)
        refined_x = refined["x"].values.copy()
        refined_y = refined["y"].values.copy()
        keep_d = resolve_duplicates(
            refined_x[keep_q], refined_y[keep_q],
            pred_x[keep_q],    pred_y[keep_q]
        )
        # Map back to full index
        keep_full = keep_q.copy()
        keep_full[keep_q] &= keep_d
        print(f"  After duplicate removal: {keep_full.sum()} beads")

        # Compute displacements
        disp_x = refined_x[keep_full] - bead_x[keep_full]
        disp_y = refined_y[keep_full] - bead_y[keep_full]

        # Save CSV
        frame_df = pd.DataFrame({
            "frame":    frame_idx,
            "ref_x":    bead_x[keep_full],
            "ref_y":    bead_y[keep_full],
            "refined_x": refined_x[keep_full],
            "refined_y": refined_y[keep_full],
            "disp_x":   disp_x,
            "disp_y":   disp_y,
            "disp_mag": np.hypot(disp_x, disp_y),
            "ecc":      refined["ecc"].values[keep_full],
            "mass":     refined["mass"].values[keep_full],
        })
        all_results.append(frame_df)

        # Plot
        make_displacement_plot(
            ref=ref_img, deformed=def_img,
            bead_x=bead_x[keep_full], bead_y=bead_y[keep_full],
            disp_x=disp_x, disp_y=disp_y,
            title=f"SPT displacement — frame {frame_idx}",
            out_path=OUTPUT_DIR / f"displacement_frame{frame_idx:03d}.png",
            gt_dx_um=gt_dx_um, gt_dy_um=gt_dy_um,
        )

    results_df = pd.concat(all_results, ignore_index=True)
    csv_path = OUTPUT_DIR / "displacements.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nDisplacements saved to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
