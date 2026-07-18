"""GPU port of the rigid stage-drift registration -- torch phase cross-correlation
and grid_sample resampling. Imported **lazily** by
:mod:`napariTFM.backend.displacement_analysis` only when the resolved displacement
device is CUDA; a torch-free / CPU install keeps using the scikit-image reference
in :mod:`napariTFM.backend.registration`.

Two hot per-frame operations move to the GPU:

* :class:`TorchDriftEstimator` -- Hann-windowed phase cross-correlation with the
  same upsampled-DFT subpixel refinement (Guizar-Sicairos) scikit-image uses, so
  the drift it returns is **bit-for-bit identical** to
  :func:`registration.estimate_drift` (validated to 0 px in
  ``tests/test_registration.py``). It is *stateful on the anchor*: the anchor's
  windowed spectrum is the same for every frame in a stack, so it is computed once
  at construction and reused, which is most of the speed-up (~24x over the CPU
  reference at 2048^2). float32 is exact here -- the drift is quantized to
  ``1/upsample`` px and the single-precision FFT resolves the correlation peak well
  inside that -- so it is the default (this laptop GPU throttles float64).

* :func:`apply_drift_torch` -- the drift-undoing resample as a constant-translation
  ``grid_sample`` (``border`` padding == the reference's ``mode='nearest'`` edge
  replication). Unlike the drift estimate this is *not* bit-identical to SciPy's
  spline resample (grid_sample's cubic is Catmull-Rom, not a prefiltered cubic
  B-spline), but on the small (few-px) stage drifts here the difference is far
  below the displacement tolerance -- same "numerically equivalent GPU port"
  contract as the PIV/iLK backends.

Convention matches :mod:`registration`: ``drift`` is ``(u_x, u_y)`` in pixels
(positive = content sits right/down of the anchor); ``apply_drift`` undoes it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import window as _window

_DEADZONE = 1e-2   # px: below this, skip the resample (matches registration._DEADZONE)


def _hann(shape, device, dtype):
    """Hann window matching ``registration.estimate_drift`` (``skimage.filters.window``)."""
    return torch.as_tensor(_window("hann", tuple(shape)), device=device, dtype=dtype)


def _upsampled_dft(data, ups_size, upsample_factor, offsets):
    """Faithful port of ``skimage.registration._upsampled_dft`` for 2D input.

    Small ``(ups_size, ups_size)`` upsampled inverse DFT of ``data`` (a complex
    ``(H, W)`` cross-power spectrum) around ``offsets``, via per-axis kernel
    matmuls -- the subpixel-refinement core of phase cross-correlation.
    """
    im2pi = 1j * 2 * np.pi
    rdtype = torch.float64 if data.dtype == torch.complex128 else torch.float32
    # Iterate axes in reverse (skimage contracts the last axis each pass), so the
    # two ups_size axes end up first: data (H,W) -> (ups,H) -> (ups,ups).
    for n, off in [(data.shape[0], offsets[0]), (data.shape[1], offsets[1])][::-1]:
        freq = torch.fft.fftfreq(n, upsample_factor, device=data.device).to(rdtype)
        k = (torch.arange(ups_size, device=data.device, dtype=rdtype) - off)[:, None] * freq[None, :]
        kernel = torch.exp(-im2pi * k.to(data.dtype))              # (ups_size, n)
        data = torch.tensordot(kernel, data, dims=([1], [data.ndim - 1]))
    return data


class TorchDriftEstimator:
    """Anchor-fixed GPU phase cross-correlation; call per frame for its ``(u_x, u_y)`` drift.

    Reproduces ``registration.estimate_drift(anchor, image, upsample)`` exactly.
    Build once per stack (the anchor's windowed spectrum is cached); call for each
    frame and the reference.
    """

    def __init__(self, anchor, upsample=20, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.upsample = float(upsample)
        anchor = np.nan_to_num(np.asarray(anchor, dtype=np.float64))
        self.shape = anchor.shape
        self.window = _hann(anchor.shape, self.device, dtype)
        a = torch.as_tensor(anchor, device=self.device, dtype=dtype) * self.window
        self.anchor_freq = torch.fft.fft2(a)                       # cached across frames

    def __call__(self, image) -> np.ndarray:
        H, W = self.shape
        b = torch.as_tensor(np.nan_to_num(np.asarray(image, dtype=np.float64)),
                            device=self.device, dtype=self.dtype) * self.window
        prod = self.anchor_freq * torch.fft.fft2(b).conj()
        # Phase normalization (skimage default): divide by magnitude -> pure phase corr.
        prod = prod / torch.clamp(prod.abs(), min=100 * torch.finfo(self.dtype).eps)
        cc = torch.fft.ifft2(prod)
        peak = int(torch.argmax(cc.abs()))
        my, mx = peak // W, peak % W
        shift = np.array([my - H if my > H // 2 else my,
                          mx - W if mx > W // 2 else mx], dtype=float)
        if not np.all(np.isfinite(shift)):
            return np.zeros(2, dtype=np.float32)
        # Upsampled-DFT subpixel refinement around the integer peak.
        shift = np.round(shift * self.upsample) / self.upsample
        ups_size = int(np.ceil(self.upsample * 1.5))
        dftshift = float(ups_size // 2)
        cc2 = _upsampled_dft(prod.conj(), ups_size, self.upsample,
                             dftshift - shift * self.upsample).conj()
        p2 = int(torch.argmax(cc2.abs()))
        maxima = np.array([p2 // ups_size, p2 % ups_size], dtype=float) - dftshift
        shift = shift + maxima / self.upsample
        dy, dx = shift[0], shift[1]
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return np.zeros(2, dtype=np.float32)
        # skimage returns the shift registering image onto anchor; drift is its
        # negative, reordered to (u_x, u_y) -- identical to registration.estimate_drift.
        return np.array([-dx, -dy], dtype=np.float32)


def apply_drift_torch(image, drift, device="cuda", order=1) -> np.ndarray:
    """Resample ``image`` into the anchor frame, undoing bulk ``drift`` ``(u_x, u_y)``.

    GPU ``grid_sample`` counterpart of ``registration.apply_drift``: ``border``
    padding replicates edge pixels (SciPy ``mode='nearest'``), and a near-zero
    drift returns the input unchanged so drift-free data is never blurred.
    ``order`` 1 -> bilinear, 3 -> bicubic (Catmull-Rom).
    """
    u_x, u_y = float(drift[0]), float(drift[1])
    image = np.asarray(image)
    if abs(u_x) < _DEADZONE and abs(u_y) < _DEADZONE:
        return image
    dev = torch.device(device)
    img = torch.as_tensor(image, device=dev, dtype=torch.float32)
    H, W = img.shape
    rr = torch.arange(H, device=dev, dtype=torch.float32)[:, None].expand(H, W)
    cc = torch.arange(W, device=dev, dtype=torch.float32)[None, :].expand(H, W)
    # Undo the shift: sample content from (r + u_y, c + u_x).
    gx = 2.0 * (cc + u_x) / (W - 1) - 1.0
    gy = 2.0 * (rr + u_y) / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    mode = "bicubic" if int(order) >= 3 else "bilinear"
    out = F.grid_sample(img[None, None], grid, mode=mode,
                        padding_mode="border", align_corners=True)
    return out[0, 0].cpu().numpy()
