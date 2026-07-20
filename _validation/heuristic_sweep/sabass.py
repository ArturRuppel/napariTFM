"""Sabass et al. (2008) TFM quality metrics + the composite objective J.

VENDORED from ~/Projects/benchmarkTFM/benchmarktfm/metrics.py (Sabass section) and
benchmarktfm/sweep.py (objective). Kept as a lean, dependency-free copy (numpy + scipy
only) so the cluster sweep imports it without pulling benchmarktfm's napari backend.
If the canonical definitions change there, update here.

Fields are (2, H, W), [0]=x, [1]=y. J = |DTM| + DTMS + DTA/45 is the objective the
regularization heuristic is fit against -- it decomposes recovery error into the three
failure modes (magnitude on adhesions / spurious background / direction) that the L1+L2
knobs trade off, which a single blended nRMSE cannot separate.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

OBJECTIVE_WEIGHTS = (1.0, 1.0, 1.0 / 45.0)   # (|DTM|, DTMS, DTA); DTA in degrees -> /45


def _mag(field):
    return np.hypot(field[0], field[1])


def significant_mask(field, frac=0.1):
    """Where |field| >= frac * peak -- the adhesions, for a GT traction field."""
    m = _mag(np.asarray(field, float))
    mx = m.max()
    return m > frac * mx if mx > 0 else np.zeros(m.shape, bool)


def adhesions_from_mask(fa_mask, min_area_px=4):
    """One adhesion per connected component: {'center': (cy,cx), 'radius': r_px}."""
    lbl, n = ndimage.label(np.asarray(fa_mask) > 0)
    out = []
    for i in range(1, n + 1):
        area = int((lbl == i).sum())
        if area < min_area_px:
            continue
        cy, cx = ndimage.center_of_mass(lbl == i)
        out.append({"center": (float(cy), float(cx)), "radius": float(np.sqrt(area / np.pi))})
    return out


def dtm(t, t_gt, threshold_frac=0.1):
    """Deviation of Traction Magnitude: mean over the significant-GT region of
    (‖t‖−‖t_gt‖)/‖t_gt‖. Signed: 0=perfect, <0 under-, >0 over-estimate. Restricted to
    ‖t_gt‖ > threshold_frac·peak so soft GT tails don't blow the relative deviation up."""
    mt, mg = _mag(np.asarray(t, float)), _mag(np.asarray(t_gt, float))
    if mg.max() <= 0:
        return float("nan")
    valid = (mg > threshold_frac * mg.max()) & np.isfinite(mt) & np.isfinite(mg)
    if not valid.any():
        return float("nan")
    return float(np.mean((mt[valid] - mg[valid]) / mg[valid]))


def dtms(t, adhesions, ring_width_px=15):
    """Deviation of Traction Magnitude in the Surrounding: mean over adhesions of
    (mean ‖t‖ in a ring just outside) / (mean ‖t‖ on the adhesion). 0=sharp, ->1=leaky."""
    mt = _mag(np.asarray(t, float))
    h, w = mt.shape
    yy, xx = np.ogrid[:h, :w]
    vals = []
    for a in adhesions:
        cy, cx = a["center"]; r = a["radius"]
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        on = d <= r
        ring = (d > r) & (d <= r + ring_width_px)
        if on.any() and ring.any():
            m_on = mt[on].mean()
            if m_on > 0:
                vals.append(mt[ring].mean() / m_on)
    return float(np.mean(vals)) if vals else float("nan")


def dta(t, t_gt, adhesions):
    """Deviation of Traction Angle (degrees): angle between mean reconstructed and mean GT
    traction vector on each adhesion, averaged. 0=perfect direction."""
    t = np.asarray(t, float); t_gt = np.asarray(t_gt, float)
    h, w = t.shape[1:]
    yy, xx = np.ogrid[:h, :w]
    cosines = []
    for a in adhesions:
        cy, cx = a["center"]; r = a["radius"]
        on = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) <= r
        if not on.any():
            continue
        cv = np.array([t[0][on].mean(), t[1][on].mean()])
        gv = np.array([t_gt[0][on].mean(), t_gt[1][on].mean()])
        cn, gn = np.linalg.norm(cv), np.linalg.norm(gv)
        if cn > 0 and gn > 0:
            cosines.append(np.clip(np.dot(cv, gv) / (cn * gn), -1, 1))
    if not cosines:
        return float("nan")
    return float(np.degrees(np.arccos(np.mean(cosines))))


def sabass_metrics(t, t_gt, fa_mask, ring_width_px=15):
    """The three Sabass measures keyed for tidy records, plus the adhesion count."""
    adh = adhesions_from_mask(fa_mask)
    return {
        "dtm": dtm(t, t_gt),
        "dtms": dtms(t, adh, ring_width_px),
        "dta": dta(t, t_gt, adh),
        "n_adh": len(adh),
    }


def objective(dtm_v, dtms_v, dta_v, weights=OBJECTIVE_WEIGHTS):
    """J = w0·|DTM| + w1·DTMS + w2·DTA. NaN-propagating: a failed adhesion match is a
    failed config, not a silently dropped one."""
    return weights[0] * abs(dtm_v) + weights[1] * dtms_v + weights[2] * dta_v
