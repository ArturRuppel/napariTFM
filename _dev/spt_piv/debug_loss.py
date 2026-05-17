"""
Diagnose bead loss through the detection → refinement pipeline.

For each benchmark level, prints a funnel:
  detected → valid window → refined → passed quality filters (broken down
  by criterion) → after duplicate removal

Also plots:
  1. Spatial map of rejections coloured by reason
  2. Residual distribution vs flow quality (flow magnitude at each bead)
  3. Flow field accuracy vs GT
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from spt_displacement import (
    load_image, to_uint16, auto_minmass, detect_beads,
    compute_optical_flow, sample_flow_at_beads,
    refine_positions, quality_filter, resolve_duplicates,
    BEAD_DIAMETER, BEAD_SEPARATION, TRACKPY_PREPROCESS,
    REFINE_RADIUS, RESIDUAL_RADIUS, MAX_ECCENTRICITY, MIN_MASS_FRACTION,
    DUPLICATE_RADIUS, PIXEL_SIZE_UM,
)

tp.quiet()

LEVELS = ["mid", "high"]
BASE   = Path("/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM")
OUT    = Path(__file__).parent / "debug_loss"
OUT.mkdir(exist_ok=True)


for LEVEL in LEVELS:
    print(f"\n{'='*60}")
    print(f"  LEVEL = {LEVEL}")
    print(f"{'='*60}")
    level_dir = BASE / LEVEL

    ref_img = load_image(str(level_dir / "reference.tif"))
    def_img = load_image(str(level_dir / "deformed.tif"))
    ref_u16 = to_uint16(ref_img)
    def_u16 = to_uint16(def_img)

    gt_dx_um = np.load(level_dir / "displacement_x.npy")
    gt_dy_um = np.load(level_dir / "displacement_y.npy")
    gt_dx_px = gt_dx_um / PIXEL_SIZE_UM
    gt_dy_px = gt_dy_um / PIXEL_SIZE_UM
    H, W = ref_img.shape

    # ── Step 1: Detection ────────────────────────────────────────────────────
    minmass = auto_minmass(ref_u16)
    ref_beads = detect_beads(ref_u16, minmass)
    ref_median_mass = float(ref_beads["mass"].median())
    n_detected = len(ref_beads)
    print(f"\n[1] Detected in reference:   {n_detected}")

    bead_x = ref_beads["x"].values
    bead_y = ref_beads["y"].values

    # ── Step 2: Optical flow ─────────────────────────────────────────────────
    print("    Computing optical flow...")
    flow = compute_optical_flow(ref_img, def_img)
    flow_dx, flow_dy = sample_flow_at_beads(flow, bead_x, bead_y)
    pred_x = bead_x + flow_dx
    pred_y = bead_y + flow_dy
    flow_mag_at_beads = np.hypot(flow_dx, flow_dy)

    # Flow accuracy vs GT (sample GT at each bead position)
    gy_flat = np.arange(H, dtype=float)
    gx_flat = np.arange(W, dtype=float)
    interp_gtx = RegularGridInterpolator((gy_flat, gx_flat), gt_dx_px,
                                          bounds_error=False, fill_value=np.nan)
    interp_gty = RegularGridInterpolator((gy_flat, gx_flat), gt_dy_px,
                                          bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([bead_y, bead_x])
    gt_dx_at_beads = interp_gtx(pts)
    gt_dy_at_beads = interp_gty(pts)
    flow_error_at_beads = np.hypot(flow_dx - gt_dx_at_beads,
                                   flow_dy - gt_dy_at_beads)

    print(f"    Flow vs GT — median: {np.nanmedian(flow_error_at_beads):.2f} px, "
          f"RMSE: {np.sqrt(np.nanmean(flow_error_at_beads**2)):.2f} px, "
          f"max: {np.nanmax(flow_error_at_beads):.2f} px")

    # ── Step 3: Zero-window pre-filter ────────────────────────────────────────
    r = REFINE_RADIUS
    clipped_x = np.clip(pred_x, r, W - r - 1)
    clipped_y = np.clip(pred_y, r, H - r - 1)
    valid_window = np.zeros(n_detected, dtype=bool)
    for i, (x, y) in enumerate(zip(clipped_x, clipped_y)):
        cx, cy = int(round(x)), int(round(y))
        patch = def_u16[cy - r: cy + r + 1, cx - r: cx + r + 1]
        if patch.size > 0 and int(patch.max()) > 0:
            valid_window[i] = True
    n_valid_window = valid_window.sum()
    n_dark_window = n_detected - n_valid_window
    print(f"\n[2] Valid window (non-dark):  {n_valid_window}  "
          f"(lost {n_dark_window} to dark/boundary)")

    # ── Step 4: Refinement ────────────────────────────────────────────────────
    print("    Running tp.refine_com...")
    refined = refine_positions(def_u16, pred_x, pred_y)
    n_nan = refined["x"].isna().sum()
    print(f"\n[3] After refine_com NaN:    {n_detected - n_nan}  (NaN: {n_nan})")

    # ── Step 5: Quality filter — breakdown ───────────────────────────────────
    residual = np.hypot(refined["x"].values - pred_x,
                        refined["y"].values - pred_y)
    ecc  = refined["ecc"].values
    mass = refined["mass"].values

    fail_nan       = refined["x"].isna().values
    fail_residual  = (~fail_nan) & (residual > RESIDUAL_RADIUS)
    fail_ecc       = (~fail_nan) & (ecc > MAX_ECCENTRICITY)
    fail_mass      = (~fail_nan) & (mass < MIN_MASS_FRACTION * ref_median_mass)
    # a bead can fail multiple criteria; "fail any" = rejected
    fail_any       = fail_nan | fail_residual | fail_ecc | fail_mass
    keep_q         = ~fail_any

    print(f"\n[4] Quality filter breakdown  (total rejected: {fail_any.sum()}):")
    print(f"    NaN (dark window or refine fail): {fail_nan.sum()}")
    print(f"    Residual > {RESIDUAL_RADIUS} px:            {fail_residual.sum()}")
    print(f"    Eccentricity > {MAX_ECCENTRICITY}:            {fail_ecc.sum()}")
    print(f"    Mass < {MIN_MASS_FRACTION}×median:              {fail_mass.sum()}")
    print(f"    Passed:                           {keep_q.sum()}")

    # ── Step 6: Duplicate removal ─────────────────────────────────────────────
    refined_x_q = refined["x"].values[keep_q]
    refined_y_q = refined["y"].values[keep_q]
    keep_d = resolve_duplicates(refined_x_q, refined_y_q,
                                pred_x[keep_q], pred_y[keep_q])
    keep_full = keep_q.copy()
    keep_full[keep_q] &= keep_d
    n_dupes = keep_q.sum() - keep_full.sum()
    print(f"\n[5] After duplicate removal: {keep_full.sum()}  (lost {n_dupes} duplicates)")

    # Final summary
    print(f"\n    TOTAL:  {n_detected} → {keep_full.sum()} "
          f"({100 * keep_full.sum() / n_detected:.1f}% yield)")

    # ── PLOTS ─────────────────────────────────────────────────────────────────

    # --- Plot A: spatial map of rejection reasons ---
    # Categories (in priority order for colouring):
    #   0 = kept (grey)
    #   1 = dark window (purple)
    #   2 = NaN only (after window was ok — shouldn't be many)
    #   3 = residual fail (red)
    #   4 = eccentricity fail (orange)
    #   5 = mass fail (blue)
    #   6 = duplicate (cyan)
    reason_label = ["kept", "dark/boundary", "refine NaN", "residual",
                    "eccentricity", "mass", "duplicate"]
    reason_color = ["#aaaaaa", "#9B59B6", "#E74C3C", "#E74C3C",
                    "#E67E22", "#3498DB", "#1ABC9C"]

    category = np.zeros(n_detected, dtype=int)  # default: kept
    # Apply in reverse priority so highest-priority reason wins
    category[fail_nan]              = 2   # NaN / dark
    # Dark-window is a subset of NaN; overwrite those specifically
    nan_but_dark = fail_nan & ~valid_window
    nan_not_dark = fail_nan & valid_window
    category[nan_but_dark]          = 1
    category[nan_not_dark]          = 2
    category[fail_residual]         = 3
    category[fail_ecc]              = 4
    category[fail_mass]             = 5
    # duplicates: beads that passed quality but then got removed
    dup_mask_full = keep_q & ~keep_full
    category[dup_mask_full]         = 6
    # kept = still 0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.imshow(ref_img, cmap="gray", vmin=0, vmax=0.3)
    cmap_list = [reason_color[c] for c in category]
    sc = ax.scatter(bead_x, bead_y, c=cmap_list, s=4, alpha=0.8)
    patches = [mpatches.Patch(color=reason_color[i], label=reason_label[i])
               for i in range(len(reason_label))]
    ax.legend(handles=patches, fontsize=7, loc="upper right")
    ax.set_title(f"{LEVEL}: rejection map  ({keep_full.sum()}/{n_detected} kept)")
    ax.axis("off")

    # --- Plot B: flow error vs residual (only non-NaN beads) ---
    ax = axes[1]
    valid_mask = ~fail_nan
    kept_mask  = keep_full
    sc1 = ax.scatter(flow_error_at_beads[valid_mask & kept_mask],
                     residual[valid_mask & kept_mask],
                     s=3, alpha=0.4, c="steelblue", label="kept")
    sc2 = ax.scatter(flow_error_at_beads[valid_mask & ~kept_mask],
                     residual[valid_mask & ~kept_mask],
                     s=3, alpha=0.4, c="red", label="rejected")
    ax.axhline(RESIDUAL_RADIUS, color="red", linestyle="--", lw=1,
               label=f"residual threshold ({RESIDUAL_RADIUS} px)")
    ax.set_xlabel("flow error vs GT (px)")
    ax.set_ylabel("refine residual |refined − predicted| (px)")
    ax.set_title(f"{LEVEL}: flow error vs refinement residual")
    ax.legend(fontsize=8)
    ax.set_xlim(0, np.nanpercentile(flow_error_at_beads, 99) * 1.1)
    ax.set_ylim(0, RESIDUAL_RADIUS * 2.5)

    plt.tight_layout()
    fig.savefig(OUT / f"{LEVEL}_A_loss_map.png", dpi=150)
    plt.close(fig)
    print(f"    Saved {LEVEL}_A_loss_map.png")

    # --- Plot C: residual distribution by flow-error quartile ---
    valid = ~fail_nan
    flow_err_v = flow_error_at_beads[valid]
    resid_v    = residual[valid]
    q25, q50, q75 = np.nanpercentile(flow_err_v, [25, 50, 75])
    labels_q = [f"flow err ≤{q25:.1f}px", f"{q25:.1f}–{q50:.1f}px",
                f"{q50:.1f}–{q75:.1f}px", f">{q75:.1f}px"]
    masks_q  = [
        flow_err_v <= q25,
        (flow_err_v > q25) & (flow_err_v <= q50),
        (flow_err_v > q50) & (flow_err_v <= q75),
        flow_err_v > q75,
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "mediumseagreen", "darkorange", "firebrick"]
    bins = np.linspace(0, RESIDUAL_RADIUS * 2, 60)
    for m, lbl, col in zip(masks_q, labels_q, colors):
        if m.sum() > 0:
            ax.hist(resid_v[m], bins=bins, alpha=0.6, label=lbl, color=col)
    ax.axvline(RESIDUAL_RADIUS, color="black", linestyle="--", lw=1.5,
               label=f"threshold = {RESIDUAL_RADIUS} px")
    ax.set_xlabel("refinement residual |refined − predicted| (px)")
    ax.set_ylabel("count")
    ax.set_title(f"{LEVEL}: residual distributions split by flow accuracy quartile")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / f"{LEVEL}_B_residual_by_flow_quartile.png", dpi=150)
    plt.close(fig)
    print(f"    Saved {LEVEL}_B_residual_by_flow_quartile.png")

    # --- Plot D: flow error map (spatial) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    sc = ax.scatter(bead_x, bead_y, c=flow_error_at_beads,
                    cmap="hot_r", s=4,
                    vmin=0, vmax=np.nanpercentile(flow_error_at_beads, 95))
    plt.colorbar(sc, ax=ax, label="flow error vs GT (px)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_title(f"{LEVEL}: optical flow error at each bead")
    ax.axis("off")

    ax = axes[1]
    # Show final kept beads error vs GT
    disp_x = refined["x"].values[keep_full] - bead_x[keep_full]
    disp_y = refined["y"].values[keep_full] - bead_y[keep_full]
    gt_at_kept_x = gt_dx_at_beads[keep_full]
    gt_at_kept_y = gt_dy_at_beads[keep_full]
    final_err = np.hypot(disp_x - gt_at_kept_x, disp_y - gt_at_kept_y)
    sc2 = ax.scatter(bead_x[keep_full], bead_y[keep_full], c=final_err,
                     cmap="hot_r", s=4,
                     vmin=0, vmax=np.nanpercentile(final_err, 95))
    plt.colorbar(sc2, ax=ax, label="final displacement error vs GT (px)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_title(f"{LEVEL}: final error  "
                 f"(RMSE={np.sqrt(np.nanmean(final_err**2)):.3f} px, "
                 f"median={np.nanmedian(final_err):.3f} px)")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(OUT / f"{LEVEL}_C_error_maps.png", dpi=150)
    plt.close(fig)
    print(f"    Saved {LEVEL}_C_error_maps.png")

    # --- Plot E: flow vector field vs GT (dense) ---
    step = 20
    ys, xs = np.mgrid[step//2:H:step, step//2:W:step]
    pts_grid = np.column_stack([ys.ravel(), xs.ravel()])
    gt_dx_grid = interp_gtx(pts_grid).reshape(ys.shape)
    gt_dy_grid = interp_gty(pts_grid).reshape(ys.shape)
    flow_dx_grid = flow[ys, xs, 0]
    flow_dy_grid = flow[ys, xs, 1]
    err_grid = np.hypot(flow_dx_grid - gt_dx_grid, flow_dy_grid - gt_dy_grid)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    vmax = np.nanpercentile(np.hypot(gt_dx_grid, gt_dy_grid), 99)
    for ax, ddx, ddy, title in zip(
            axes,
            [gt_dx_grid,   flow_dx_grid,  flow_dx_grid - gt_dx_grid],
            [gt_dy_grid,   flow_dy_grid,  flow_dy_grid - gt_dy_grid],
            ["Ground truth", "DIS flow", "Residual (flow − GT)"]):
        mag = np.hypot(ddx, ddy)
        ax.quiver(xs, ys, ddx, ddy, mag,
                  cmap="plasma", clim=(0, vmax if "Residual" not in title else None),
                  angles="xy", scale_units="xy", scale=1, width=0.003)
        ax.set_title(title + f"\nmax={mag.max():.1f} px")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(OUT / f"{LEVEL}_D_flow_vs_gt.png", dpi=150)
    plt.close(fig)
    print(f"    Saved {LEVEL}_D_flow_vs_gt.png")

print("\nAll done. Outputs in:", OUT)
