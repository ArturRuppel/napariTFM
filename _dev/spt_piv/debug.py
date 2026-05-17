"""Diagnostic script — probe bead detection, PIV accuracy, and assignment quality."""

import sys
from pathlib import Path
import numpy as np
import tifffile
import trackpy as tp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))
from spt_piv_displacement import (
    load_image, detect_beads, auto_minmass, compute_piv, interpolate_piv,
    BEAD_DIAMETER, BEAD_SEPARATION, PIXEL_SIZE_UM,
)

LEVEL = "mid"   # change to "high" or "low"

BASE   = Path("/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM") / LEVEL
OUT    = Path(__file__).parent / f"debug_{LEVEL}"
OUT.mkdir(exist_ok=True)

ref_img  = load_image(str(BASE / "reference.tif"))
def_img  = load_image(str(BASE / "deformed.tif"))
gt_dx_um = np.load(BASE / "displacement_x.npy")   # in µm
gt_dy_um = np.load(BASE / "displacement_y.npy")
gt_dx_px = gt_dx_um / PIXEL_SIZE_UM
gt_dy_px = gt_dy_um / PIXEL_SIZE_UM
H, W = ref_img.shape

# ── 1. BEAD DETECTION ──────────────────────────────────────────────────────

tp.quiet()
minmass = auto_minmass(ref_img, BEAD_DIAMETER)
print(f"minmass = {minmass:.1f}")

ref_beads = detect_beads(ref_img, minmass)
def_beads = detect_beads(def_img, minmass)
print(f"Reference beads: {len(ref_beads)}")
print(f"Deformed beads:  {len(def_beads)}")

# --- 1a. Detection overlay on reference ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, img, beads, title in zip(
        axes,
        [ref_img, def_img],
        [ref_beads, def_beads],
        [f"Reference — {len(ref_beads)} beads", f"Deformed — {len(def_beads)} beads"]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=0.3)
    ax.scatter(beads["x"], beads["y"], s=4, c="lime", linewidths=0, alpha=0.7)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
fig.savefig(OUT / "1_bead_detection.png", dpi=150)
plt.close()
print("Saved 1_bead_detection.png")

# --- 1b. Where are beads MISSING in the deformed image? ---
# For each ref bead, find closest def bead; if >3 px away, flag as missing.
from scipy.spatial import cKDTree
tree = cKDTree(def_beads[["x", "y"]].values)
dists, _ = tree.query(ref_beads[["x", "y"]].values, k=1)
missing_mask = dists > 3.0

fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(def_img, cmap="gray", vmin=0, vmax=0.3)
ax.scatter(ref_beads.loc[~missing_mask, "x"], ref_beads.loc[~missing_mask, "y"],
           s=4, c="lime", alpha=0.5, label=f"detected ({(~missing_mask).sum()})")
ax.scatter(ref_beads.loc[missing_mask, "x"], ref_beads.loc[missing_mask, "y"],
           s=10, c="red", alpha=0.8, label=f"missing in deformed ({missing_mask.sum()})")
ax.set_title("Reference bead positions plotted on deformed image\n(red = no close match in deformed)")
ax.legend(loc="upper right", fontsize=7)
ax.axis("off")
fig.savefig(OUT / "2_missing_beads.png", dpi=150)
plt.close()
print("Saved 2_missing_beads.png")

# ── 2. PIV ACCURACY ────────────────────────────────────────────────────────

print("Computing PIV...")
gx, gy, piv_dx, piv_dy = compute_piv(ref_img, def_img)
print(f"PIV grid: {gx.shape}, step={gx[0,1]-gx[0,0]:.0f} px")

# Build GT sampled at PIV grid points
gy_flat = np.arange(H)
gx_flat = np.arange(W)
interp_gt_dx = RegularGridInterpolator((gy_flat, gx_flat), gt_dx_px,
                                        bounds_error=False, fill_value=np.nan)
interp_gt_dy = RegularGridInterpolator((gy_flat, gx_flat), gt_dy_px,
                                        bounds_error=False, fill_value=np.nan)
pts = np.column_stack([gy.ravel(), gx.ravel()])
gt_dx_at_piv = interp_gt_dx(pts).reshape(gx.shape)
gt_dy_at_piv = interp_gt_dy(pts).reshape(gx.shape)

piv_err = np.hypot(piv_dx - gt_dx_at_piv, piv_dy - gt_dy_at_piv)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
vmax = np.nanpercentile(np.hypot(gt_dx_at_piv, gt_dy_at_piv), 99)

ax = axes[0]
q = ax.quiver(gx, gy, gt_dx_at_piv, gt_dy_at_piv,
              np.hypot(gt_dx_at_piv, gt_dy_at_piv), cmap="plasma", clim=(0, vmax),
              angles="xy", scale_units="xy", scale=1, width=0.003)
ax.set_title("Ground truth displacement")
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")

ax = axes[1]
ax.quiver(gx, gy, piv_dx, piv_dy,
          np.hypot(piv_dx, piv_dy), cmap="plasma", clim=(0, vmax),
          angles="xy", scale_units="xy", scale=1, width=0.003)
ax.set_title("PIV estimated displacement")
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")

ax = axes[2]
sc = ax.scatter(gx.ravel(), gy.ravel(), c=piv_err.ravel(), cmap="hot_r", s=8,
                vmin=0, vmax=np.nanpercentile(piv_err, 95))
plt.colorbar(sc, ax=ax, label="PIV error (px)")
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
ax.set_title(f"PIV error  (RMSE={np.nanmean(piv_err**2)**0.5:.2f} px, "
             f"median={np.nanmedian(piv_err):.2f} px)")

plt.tight_layout()
fig.savefig(OUT / "3_piv_accuracy.png", dpi=150)
plt.close()
print("Saved 3_piv_accuracy.png")

# ── 3. EXPECTED BEAD POSITIONS (GT-shifted) vs DETECTED ────────────────────
# For every ref bead, compute where it SHOULD be in the deformed image using GT
ref_x = ref_beads["x"].values
ref_y = ref_beads["y"].values
gt_at_ref_dx, gt_at_ref_dy = interpolate_piv(gx, gy, gt_dx_at_piv, gt_dy_at_piv,
                                              ref_x, ref_y)
expected_x = ref_x + gt_at_ref_dx
expected_y = ref_y + gt_at_ref_dy

# How close is the nearest detected def bead to the GT-predicted position?
dists_gt, idx_gt = tree.query(np.column_stack([expected_x, expected_y]), k=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.imshow(def_img, cmap="gray", vmin=0, vmax=0.3)
ax.scatter(def_beads["x"], def_beads["y"], s=3, c="lime", alpha=0.5, label="detected")
ax.scatter(expected_x, expected_y, s=4, c="red", alpha=0.5, label="GT-predicted")
ax.set_title("GT-predicted bead positions (red) vs detected in deformed (green)")
ax.legend(fontsize=7)
ax.axis("off")

ax = axes[1]
sc = ax.scatter(ref_x, ref_y, c=dists_gt, cmap="hot_r", s=5,
                vmin=0, vmax=np.percentile(dists_gt, 95))
plt.colorbar(sc, ax=ax, label="distance to nearest detected bead (px)")
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
ax.set_title(f"How far is nearest detected bead from GT-predicted?\n"
             f"median={np.median(dists_gt):.2f} px, "
             f">{2:.0f}px: {(dists_gt>2).sum()} beads ({100*(dists_gt>2).mean():.1f}%)")
plt.tight_layout()
fig.savefig(OUT / "4_gt_predicted_vs_detected.png", dpi=150)
plt.close()
print("Saved 4_gt_predicted_vs_detected.png")

# ── 4. MASS DISTRIBUTION ────────────────────────────────────────────────────
tp.quiet(False)
raw_ref = (ref_img * 65535).astype(np.uint16)
raw_def = (def_img * 65535).astype(np.uint16)
all_ref = tp.locate(raw_ref, diameter=BEAD_DIAMETER, minmass=0, separation=BEAD_SEPARATION)
all_def = tp.locate(raw_def, diameter=BEAD_DIAMETER, minmass=0, separation=BEAD_SEPARATION)
tp.quiet()

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, np.percentile(all_ref["mass"], 99), 100)
ax.hist(all_ref["mass"], bins=bins, alpha=0.6, label=f"reference (n={len(all_ref)})", color="blue")
ax.hist(all_def["mass"], bins=bins, alpha=0.6, label=f"deformed (n={len(all_def)})", color="orange")
ax.axvline(minmass, color="red", linestyle="--", label=f"minmass={minmass:.0f}")
ax.set_xlabel("TrackPy mass")
ax.set_ylabel("count")
ax.legend()
ax.set_title("Bead mass distribution — reference vs deformed")
fig.savefig(OUT / "5_mass_distribution.png", dpi=150)
plt.close()
print("Saved 5_mass_distribution.png")

# ── 5. SUMMARY ──────────────────────────────────────────────────────────────
print(f"\n=== SUMMARY ({LEVEL}) ===")
print(f"Ref beads detected:     {len(ref_beads)}")
print(f"Def beads detected:     {len(def_beads)}")
print(f"Ref beads w/ no nearby def bead (<3px): {missing_mask.sum()} ({100*missing_mask.mean():.1f}%)")
print(f"GT-predicted → nearest def bead:")
print(f"  median dist:  {np.median(dists_gt):.3f} px")
print(f"  >2 px:  {(dists_gt>2).sum()}  ({100*(dists_gt>2).mean():.1f}%)")
print(f"  >5 px:  {(dists_gt>5).sum()}  ({100*(dists_gt>5).mean():.1f}%)")
print(f"  >10 px: {(dists_gt>10).sum()}  ({100*(dists_gt>10).mean():.1f}%)")
print(f"PIV RMSE: {np.nanmean(piv_err**2)**0.5:.2f} px, median: {np.nanmedian(piv_err):.2f} px")
gt_max = float(np.nanmax(np.hypot(gt_dx_at_piv, gt_dy_at_piv)))
piv_max = float(np.nanmax(np.hypot(piv_dx, piv_dy)))
print(f"GT max displacement: {gt_max:.2f} px,  PIV max: {piv_max:.2f} px")
