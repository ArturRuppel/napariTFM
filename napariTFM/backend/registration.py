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
