"""
Explore bead detection on the preprocessed reference image.
Shows the effect of preprocessing vs raw, then sweeps detection parameters.
"""

from pathlib import Path
import numpy as np
import tifffile
import trackpy as tp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/aruppel/Projects/napariTFM")
from spt_piv_displacement import load_image, BEAD_DIAMETER, BEAD_SEPARATION
from napariTFM.backend.preprocessing import ImageProcessor

tp.quiet()
_proc = ImageProcessor()

LEVEL   = "mid"
BASE    = Path("/home/aruppel/Projects/napariTFM/_validation/benchmark_TFM") / LEVEL
OUT     = Path(__file__).parent / "debug_detection"
OUT.mkdir(exist_ok=True)

# ── Load images ──────────────────────────────────────────────────────────────
raw    = tifffile.imread(BASE / "reference.tif").astype(np.float32)
lo, hi = raw.min(), raw.max()
raw_norm = (raw - lo) / (hi - lo)

preprocessed = load_image(str(BASE / "reference.tif"))

# ── 1. Raw vs preprocessed side by side ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
regions = [
    ("full",    (0,   700, 0,   700)),
    ("sparse",  (0,   200, 0,   200)),
    ("dense",   (250, 450, 250, 450)),
]
for col, (label, (y0, y1, x0, x1)) in enumerate(regions):
    vmax_raw  = np.percentile(raw_norm, 99.5)
    vmax_pre  = np.percentile(preprocessed, 99.5)
    axes[0, col].imshow(raw_norm[y0:y1, x0:x1], cmap="gray",
                        vmin=0, vmax=vmax_raw, interpolation="nearest")
    axes[0, col].set_title(f"Raw — {label}", fontsize=9)
    axes[0, col].axis("off")
    axes[1, col].imshow(preprocessed[y0:y1, x0:x1], cmap="gray",
                        vmin=0, vmax=vmax_pre, interpolation="nearest")
    axes[1, col].set_title(f"Preprocessed — {label}", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle("Raw vs preprocessed (percentile scaling 80–99.9 + Gaussian σ=1)", fontsize=11)
plt.tight_layout()
fig.savefig(OUT / "P1_raw_vs_preprocessed.png", dpi=150)
plt.close()
print("Saved P1_raw_vs_preprocessed.png")

# ── 2. Mass distributions: raw vs preprocessed for a range of diameters ─────
raw16  = (raw_norm * 65535).astype("uint16")
pre16  = (preprocessed * 65535).astype("uint16")

diameters = [5, 7, 9, 11]
fig, axes = plt.subplots(2, len(diameters), figsize=(5 * len(diameters), 8))

for col, diam in enumerate(diameters):
    sep = diam + 1
    for row, (img16, label, color) in enumerate([
            (raw16,  "raw",          "steelblue"),
            (pre16,  "preprocessed", "darkorange")]):
        feats = tp.locate(img16, diameter=diam, minmass=0, separation=sep)
        ax = axes[row, col]
        ax.hist(feats["mass"], bins=80, color=color, log=True)
        ax.set_title(f"{label}  diam={diam}  n={len(feats)}", fontsize=8)
        ax.set_xlabel("mass")
        ax.set_ylabel("count (log)")
        # Mark the valley — use Otsu-like approach on log-mass
        log_mass = np.log10(feats["mass"].clip(1))
        hist, edges = np.histogram(log_mass, bins=100)
        # Find first local minimum after the first peak (noise)
        from scipy.signal import argrelmin
        mins = argrelmin(hist, order=3)[0]
        if len(mins):
            valley = 10 ** edges[mins[0]]
            ax.axvline(valley, color="red", linestyle="--", lw=1,
                       label=f"valley≈{valley:.0f}")
            ax.legend(fontsize=7)

plt.suptitle("Mass distributions — raw vs preprocessed × diameter", fontsize=11)
plt.tight_layout()
fig.savefig(OUT / "P2_mass_distributions.png", dpi=150)
plt.close()
print("Saved P2_mass_distributions.png")

# ── 3. Detection visualisation: preprocessed, sweep diam + minmass ───────────
# Use a representative 250×250 crop that includes both sparse and dense areas
y0, y1, x0, x1 = 50, 300, 50, 300
crop = preprocessed[y0:y1, x0:x1]
crop16 = pre16[y0:y1, x0:x1]
full16 = pre16

configs = [
    (5,  6,     0),
    (5,  6,  2000),
    (7,  8,     0),
    (7,  8,  2000),
    (9, 10,     0),
    (9, 10, 10000),
    (11, 12,    0),
    (11, 12, 50000),
]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, (diam, sep, mm) in zip(axes.ravel(), configs):
    feats_full = tp.locate(full16, diameter=diam, minmass=mm, separation=sep)
    in_crop = feats_full[
        (feats_full.x >= x0) & (feats_full.x < x1) &
        (feats_full.y >= y0) & (feats_full.y < y1)
    ]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=np.percentile(crop, 99.5),
              interpolation="nearest")
    ax.scatter(in_crop.x - x0, in_crop.y - y0,
               s=20, facecolors="none", edgecolors="lime", linewidths=0.8, alpha=0.9)
    ax.set_title(
        f"diam={diam} sep={sep} minmass={mm}\n"
        f"{len(in_crop)} in crop / {len(feats_full)} total",
        fontsize=8
    )
    ax.axis("off")

plt.suptitle("Detection sweep on preprocessed image — 250×250 crop", fontsize=11)
plt.tight_layout()
fig.savefig(OUT / "P3_detection_sweep.png", dpi=150)
plt.close()
print("Saved P3_detection_sweep.png")

# ── 4. Best candidate: zoom in on both sparse and dense sub-regions ───────────
# Pick the config that looks most promising from above to zoom in further
diam, sep, mm = 7, 8, 2000
feats_best = tp.locate(pre16, diameter=diam, minmass=mm, separation=sep)
print(f"\nBest candidate (diam={diam} sep={sep} minmass={mm}): {len(feats_best)} beads")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
zoom_regions = [
    ("sparse 150×150", (10,  160,  10, 160)),
    ("medium 150×150", (275, 425, 100, 250)),
    ("dense 150×150",  (275, 425, 275, 425)),
]
for ax, (label, (y0, y1, x0, x1)) in zip(axes, zoom_regions):
    in_r = feats_best[
        (feats_best.x >= x0) & (feats_best.x < x1) &
        (feats_best.y >= y0) & (feats_best.y < y1)
    ]
    crop = preprocessed[y0:y1, x0:x1]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=np.percentile(crop, 99.5),
              interpolation="nearest")
    ax.scatter(in_r.x - x0, in_r.y - y0,
               s=30, facecolors="none", edgecolors="lime", linewidths=0.9)
    ax.set_title(f"{label}  n={len(in_r)}", fontsize=9)
    ax.axis("off")

plt.suptitle(f"Preprocessed — diam={diam} sep={sep} minmass={mm}", fontsize=11)
plt.tight_layout()
fig.savefig(OUT / "P4_zoom_best.png", dpi=150)
plt.close()
print("Saved P4_zoom_best.png")
