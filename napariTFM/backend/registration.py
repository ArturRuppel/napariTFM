"""Parameter-free rigid (translation-only) registration for stage-drift removal.

Before any displacement analysis, every bead frame is registered to a common
anchor (the first bead frame), so the only motion the displacement method sees is
the cell-induced deformation, not bulk stage drift. This matters most for the
capture-limited methods (Lucas-Kanade, FFD): a large drift can exceed their
capture range, so a method that never measures the drift cannot have it removed
after the fact. Registering up front keeps the residual small for every method,
which is why it is done here rather than folded into any one backend.

Registration is translation only and estimated by Hann-windowed phase
cross-correlation: the window suppresses the spectral leakage that non-periodic
image borders would otherwise inject, and the method has no tuning knobs. The
cell channel is moved by the same per-frame drift downstream (see
``batch_analysis._cells_for_overlay``) so overlays sit in the same anchor frame
as the traction field.

Convention: ``drift`` is ``(u_x, u_y)`` in pixels, the bulk translation of the
image content relative to the anchor (positive = right/down), matching
``DisplacementResult.drift_pixels``. ``apply_drift`` undoes it.
"""
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.filters import window as _window
from skimage.registration import phase_cross_correlation

_UPSAMPLE = 20       # subpixel precision ~1/20 px; fixed, not a user knob
_DEADZONE = 1e-2     # px: below this, skip the resample so drift-free data is not blurred
_KERNEL_MARGIN = 2   # px: the cubic resampling kernel reaches ~2 px past the integer shift


def estimate_drift(anchor: np.ndarray, image: np.ndarray, upsample: int = _UPSAMPLE) -> np.ndarray:
    """Bulk translation ``(u_x, u_y)`` of ``image`` relative to ``anchor``, in pixels.

    Hann-windowed phase cross-correlation (translation only, parameter-free).
    Positive ``u_x``/``u_y`` means the image content sits right/down of the anchor.
    Returns ``(0, 0)`` when the estimate is non-finite (blank/degenerate frame).
    """
    a = np.nan_to_num(np.asarray(anchor, dtype=np.float64))
    b = np.nan_to_num(np.asarray(image, dtype=np.float64))
    w = _window("hann", a.shape)
    shift_yx, _, _ = phase_cross_correlation(a * w, b * w, upsample_factor=upsample)
    dy, dx = float(shift_yx[0]), float(shift_yx[1])
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return np.zeros(2, dtype=np.float32)
    # phase_cross_correlation returns the shift that registers `image` onto `anchor`
    # via ndimage.shift(image, shift_yx); the DRIFT (how far the content moved off the
    # anchor) is its negative, re-ordered to (u_x, u_y).
    return np.array([-dx, -dy], dtype=np.float32)


def apply_drift(image: np.ndarray, drift: np.ndarray, order: int = 3) -> np.ndarray:
    """Resample ``image`` into the anchor frame, undoing bulk ``drift`` ``(u_x, u_y)``.

    A near-zero drift returns the input unchanged, so drift-free data is never
    needlessly blurred by interpolation. ``mode='nearest'`` avoids wrap-around at
    the borders.
    """
    u_x, u_y = float(drift[0]), float(drift[1])
    if abs(u_x) < _DEADZONE and abs(u_y) < _DEADZONE:
        return np.asarray(image)
    # ndimage.shift is (row, col) == (y, x); move content by -drift to undo the shift.
    return ndimage.shift(np.asarray(image, dtype=np.float64),
                         shift=(-u_y, -u_x), order=order, mode="nearest")


def valid_region(ref_drift: np.ndarray, frame_drifts: np.ndarray, shape,
                 margin: int = _KERNEL_MARGIN):
    """Interior box ``(r0, r1, c0, c1)`` common to every registered frame's real data.

    Registering a frame by a shift pulls a strip of fabricated pixels in at the
    border (``apply_drift`` fills with ``mode='nearest'`` edge replication). Its
    width on a side is the shift into that side, rounded up, plus the resampling
    kernel's reach (``margin``). This returns the largest interior box that no
    registered frame (nor the registered reference) fills with fabricated pixels,
    so the displacement method only ever measures real, co-observed content.

    A side with no drift into it is not cropped, so a drift-free stack keeps its
    full frame. If a pathological drift would crop everything away, the full frame
    is returned unchanged rather than an empty box.
    """
    H, W = int(shape[0]), int(shape[1])
    ux = np.concatenate([[float(ref_drift[0])], np.asarray(frame_drifts, dtype=float)[:, 0]])
    uy = np.concatenate([[float(ref_drift[1])], np.asarray(frame_drifts, dtype=float)[:, 1]])

    def _crop(vals: np.ndarray) -> int:
        # vals are the shifts INTO one side; the strip fabricated there is the
        # largest such shift, ceil'd, plus the kernel reach. Zero if none exceeds
        # the deadzone (that frame was never resampled, so it has no fake border).
        raw = max(0.0, float(np.max(vals)))
        return int(np.ceil(raw)) + margin if raw > _DEADZONE else 0

    left, right = _crop(-ux), _crop(ux)
    top, bottom = _crop(-uy), _crop(uy)
    r0, r1, c0, c1 = top, H - bottom, left, W - right
    if r1 - r0 < 1 or c1 - c0 < 1:  # pathological drift: keep the full frame
        return 0, H, 0, W
    return r0, r1, c0, c1


# --- drift cache -----------------------------------------------------------
#
# Registration drift depends only on the reference and every bead frame (the
# anchor is the first bead frame, and all drifts are measured relative to it) --
# NOT on the displacement method or any of its knobs. So a parameter tune-loop
# that re-runs the same folder recomputes byte-identical drift every time. These
# helpers persist the estimate to a human-readable CSV sidecar and read it back
# when the inputs are unchanged, so only the first run pays the estimation cost.
# The resample (`apply_drift`) still runs every time -- it feeds the method -- so
# the cache saves the estimate, not the whole registration step.

_CACHE_HEADER = "# napariTFM drift cache v1"


def drift_fingerprint(reference: np.ndarray, target: np.ndarray,
                      upsample: int = _UPSAMPLE) -> str:
    """A cheap content hash of the exact inputs the drift estimate depends on.

    Covers the reference, every bead frame (strided pixel sums, so a change to any
    frame invalidates), both shapes, and the subpixel factor. Not cryptographic --
    a cache-validity heuristic to catch a swapped/edited dataset, computed in a few
    ms so it never eats the saving it guards.
    """
    r = np.nan_to_num(np.asarray(reference, dtype=np.float64))
    t = np.nan_to_num(np.asarray(target, dtype=np.float64))
    if t.ndim == 2:
        t = t[None]
    parts = [str(r.shape), str(t.shape), str(int(upsample)),
             f"{r[::16, ::16].sum():.6e}"]
    parts += [f"{frame.sum():.6e}" for frame in t[:, ::16, ::16]]
    return hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()


def save_drift_csv(path, ref_drift: np.ndarray, drift_pixels: np.ndarray,
                   fingerprint: str) -> None:
    """Persist the reference + per-frame drift to a CSV sidecar (see `load_drift_csv`).

    Row 0 is the reference's drift (labelled ``reference``); the rest are one row
    per bead frame in order. Columns are ``u_x,u_y`` in pixels. The fingerprint of
    the inputs is stored in the header so a stale cache is detected on load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_drift = np.asarray(ref_drift, dtype=np.float32)
    lines = [f"{_CACHE_HEADER} fingerprint={fingerprint}",
             "frame,u_x,u_y",
             f"reference,{ref_drift[0]:.9g},{ref_drift[1]:.9g}"]
    for i, (ux, uy) in enumerate(np.asarray(drift_pixels, dtype=np.float32)):
        lines.append(f"{i},{ux:.9g},{uy:.9g}")
    path.write_text("\n".join(lines) + "\n")


def load_drift_csv(path, expected_frames: int,
                   fingerprint: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a drift sidecar written by `save_drift_csv`, or ``None`` if unusable.

    Returns ``None`` (so the caller recomputes) when the file is missing, its
    header fingerprint does not match ``fingerprint`` (inputs changed), the
    reference row is absent, or the per-frame rows do not cover exactly
    ``expected_frames`` -- i.e. any mismatch fails safe to recomputation.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        text = path.read_text().splitlines()
    except OSError:
        return None
    if not text or not text[0].startswith(_CACHE_HEADER):
        return None
    if f"fingerprint={fingerprint}" not in text[0]:
        return None
    ref_drift = None
    frames: dict = {}
    for line in text[1:]:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("frame,"):
            continue
        try:
            key, ux, uy = line.split(",")
            vec = np.array([float(ux), float(uy)], dtype=np.float32)
        except ValueError:
            return None
        if key == "reference":
            ref_drift = vec
        else:
            frames[int(key)] = vec
    if ref_drift is None or len(frames) != expected_frames:
        return None
    try:
        drift_pixels = np.stack([frames[i] for i in range(expected_frames)])
    except KeyError:
        return None
    return ref_drift, drift_pixels
