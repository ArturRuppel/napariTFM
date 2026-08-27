#!/usr/bin/env python3
"""Ground truth and scoring primitives shared by every stage of the sweep.

This module owns two things, and nothing else:

  * `rasterize_gt` -- the analytic dipole GT traction. Scene *generation*
    (make_scenes) and every *scorer* call it, so the GT can never drift between
    the field that was staged and the field that is scored.
  * `metrics` / `field_metrics` -- the recovery metrics. The objective is the
    Sabass (2008) composite J = |DTM| + DTMS + DTA/45 for isolated dipoles, which
    splits recovery error into magnitude-on-adhesions / spurious-background /
    direction; diffuse cells rank on whole-field nRMSE instead, because a
    centripetal field's per-adhesion terms degenerate. Both are computed here so
    the cache, the renderer and the analysis all score identically.

It is a library, not a driver -- there is no `main`. The force stage is
`build_force_cache.py` (GT-tuned oracle FTTC+L2 and FISTA+L1 per displacement
field); the elastic-net l1xl2 grid sweep this module used to drive was retired
once the oracle cache subsumed it.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

import sweep_config as C
import sabass


def _add_pair(tx, ty, xx, yy, cx, cy, sep, foot, axis_deg, mag, profile):
    """Add one balanced contractile pair (two equal-and-opposite poles along the
    axis, each pulling toward the pair centre) to the traction accumulators, in px
    units. Net force of the pair is zero, so any sum of pairs is DC-free too."""
    ax = np.radians(axis_deg)
    ux, uy = np.cos(ax), np.sin(ax)
    for sgn in (+1.0, -1.0):
        px = cx + sgn * (sep / 2) * ux
        py = cy + sgn * (sep / 2) * uy
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        if profile == "gaussian":
            w = np.exp(-r2 / (2 * foot ** 2))
        else:  # tophat
            w = (r2 < foot ** 2).astype(np.float32)
        tx += -sgn * mag * ux * w      # contractile: pole pulls inward (-sgn·axis)
        ty += -sgn * mag * uy * w


def rasterize_gt(scene, N):
    """Analytic GT traction of the balanced dipole pair on an N x N grid ->
    ((2, N, N) Pa, significant mask). Two equal-and-opposite contractile poles,
    net force zero (required: the Green's operator zeroes the DC mode). Generation
    and scoring both call this, so the dipole GT never drifts between them.

    Cell scenes do NOT go through here: their GT traction is a stored field
    (gt_traction.npy, the benchmarkTFM fitted-fibre geometry), loaded directly by
    whoever scores them. See make_cells.py."""
    meta = scene["meta"]
    ps = meta["pixel_size"] * (meta["image_size"] / N)         # µm per GT px
    pair = scene["pair"]
    yy, xx = np.mgrid[0:N, 0:N]
    tx = np.zeros((N, N), np.float32)
    ty = np.zeros((N, N), np.float32)
    cx = N / 2 + pair.get("center", [0.0, 0.0])[0] / ps
    cy = N / 2 + pair.get("center", [0.0, 0.0])[1] / ps
    _add_pair(tx, ty, xx, yy, cx, cy, pair["separation"] / ps, pair["footprint"] / ps,
              pair.get("axis_deg", 0.0), pair["magnitude"],
              pair.get("profile", C.GT_TRACTION_PROFILE))
    return np.stack([tx, ty], 0), (tx ** 2 + ty ** 2) > 0


def field_metrics(t, gt, frac=0.05):
    """Whole-field recovery metrics that stay well-defined on a DIFFUSE field,
    where the per-adhesion Sabass terms degrade (one merged component, a mean
    traction vector that cancels). Scored over the significant-GT region.

    - mag_bias  : magnitude-weighted scale error, sum‖t‖/sum‖t_gt‖ − 1 (signed).
    - ang_field : magnitude-weighted mean angular error in degrees (per-pixel
                  cosine, so it survives cancellation that kills a mean vector).
    - bg_leak   : spurious energy off the source, sum‖t‖(bg) / sum‖t‖(sig).
    """
    magt, magg = np.hypot(t[0], t[1]), np.hypot(gt[0], gt[1])
    if magg.max() <= 0:
        return dict(mag_bias=float("nan"), ang_field=float("nan"), bg_leak=float("nan"))
    sig = magg > frac * magg.max()
    bg = ~sig
    sum_g = float(magg[sig].sum()) or 1.0
    mag_bias = float(magt[sig].sum() / sum_g - 1.0)
    dot = (t[0] * gt[0] + t[1] * gt[1])
    denom = np.where((magt > 0) & (magg > 0), magt * magg, 1.0)
    cospix = np.clip(dot / denom, -1.0, 1.0)
    w = magg[sig]
    cos_w = float((w * cospix[sig]).sum() / (w.sum() or 1.0))
    ang_field = float(np.degrees(np.arccos(np.clip(cos_w, -1.0, 1.0))))
    on = float(magt[sig].sum()) or 1.0
    bg_leak = float(magt[bg].sum() / on)
    return dict(mag_bias=mag_bias, ang_field=ang_field, bg_leak=bg_leak)


def metrics(t, gt, do_sabass=True):
    """Whole-field metrics (well-suited to diffuse cells) + nRMSE/corr, plus the
    Sabass composite J and its components when `do_sabass` (isolated dipoles).

    The RANKING metric is chosen at analysis time per scene kind: J for dipoles,
    the whole-field terms for cells. Sabass is skipped for cells: its per-adhesion
    loop is O(n_adh · pixels) and blows up on an 82-fibre field, and DTA/DTM are
    ill-defined once the significant region merges -- exactly why the whole-field
    terms exist. Skipped fields are recorded as NaN so the schema stays uniform."""
    magt, magg = np.hypot(t[0], t[1]), np.hypot(gt[0], gt[1])
    gnorm = float(np.sqrt((gt ** 2).sum())) or 1.0
    out = dict(
        J=float("nan"), dtm=float("nan"), dtms=float("nan"), dta=float("nan"), n_adh=0,
        nrmse=float(np.sqrt(((t - gt) ** 2).sum()) / gnorm),
        corr=float(np.corrcoef(magt.ravel(), magg.ravel())[0, 1]),
        **field_metrics(t, gt),
    )
    if do_sabass:
        fa = sabass.significant_mask(gt, frac=0.1)
        s = sabass.sabass_metrics(t, gt, fa)
        out.update(J=float(sabass.objective(s["dtm"], s["dtms"], s["dta"])),
                   dtm=s["dtm"], dtms=s["dtms"], dta=s["dta"], n_adh=s["n_adh"])
    return out
