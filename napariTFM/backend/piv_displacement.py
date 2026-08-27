"""Multi-pass PIV (particle image velocimetry) displacement backend.

Estimates the in-plane displacement between a reference and a moving image by
FFT cross-correlation of interrogation windows, coarse-to-fine with window
deformation: the classic PIV algorithm, and the most accurate displacement front
end on dense bead images in our benchmarks.

**One implementation, run on either device.** The whole multipass scheme lives in
:func:`_piv_torch` (torch) and runs on the CPU (``disp_device="cpu"``) or a CUDA
device (``"cuda"``, ~100x faster; ``"auto"`` picks CUDA when present, else CPU).
Because it is literally the same code, the CPU and CUDA results are the same
algorithm -- they differ only by device-level floating-point noise (~1e-4 px:
cuFFT vs pocketfft rounding, non-associative parallel reductions, and PIV's integer
peak-location argmax occasionally picking a neighbouring pixel on a near-tie), never
by a different method. torch is therefore a hard dependency of this backend (there
is no longer an openpiv CPU fallback).

The scheme: percentile-normalize both frames; then per coarse->fine pass, split the
accumulated flow symmetrically over the two frames and bicubic-warp them
(``deformation_method="symmetric"`` equivalent), FFT-cross-correlate Hann-tapered
interrogation windows, Gaussian-fit the sub-pixel peak, reject outliers with the
normalized-median test, optionally smooth the sparse vector field
(``piv_smooth``), and add the edge-extrapolated dense residual to the accumulator.

Knobs (all read from :class:`DisplacementParameters`): ``piv_window`` (final
interrogation window, px), ``piv_overlap`` (window overlap fraction), ``piv_passes``
(coarse->fine passes), ``piv_smooth`` (per-pass Gaussian sigma on the sparse vector
grid, in grid cells; 0 disables). ``calculate_flow`` returns ``(reference, moving)
-> (H, W, 2) float32`` displacement in **pixels** at full native resolution,
``[..., 0] = u_x`` (columns), ``[..., 1] = u_y`` (rows); positive =
rightward/downward. Downscaling and the pixel->µm conversion happen downstream in
:func:`napariTFM.backend.displacement_analysis.calculate_displacement_field`.
"""
from typing import Optional

import numpy as np

from napariTFM.backend._displacement_base import BaseDisplacementAnalyzer, resolve_gpu_device
from napariTFM.backend.parameter_dataclasses import DisplacementParameters


def _norm01(image: np.ndarray) -> np.ndarray:
    """Percentile stretch to [0, 1] (1st..99.5th percentile), NaNs zeroed. Also imported by
    the scikit-image iLK CPU path so every method sees identically-scaled input."""
    img = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0)
    lo, hi = np.percentile(img, [1.0, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _window_schedule(window: int, passes: int, coarse_factor: float, H: int, W: int):
    """Geometric coarse->fine window sizes from a capped top window down to ``window``.

    ``coarse_factor`` is fixed at 2.0 (the classic ×2 windowsize schedule); it is not a
    user knob.
    """
    cap = max(window, min(H, W) // 3)
    if passes == 1:
        return [window]
    top = max(window, min(int(round(window * coarse_factor ** (passes - 1))), cap))
    return [int(round(top * (window / top) ** (p / (passes - 1)))) for p in range(passes)]


# ======================================================================== #
#  torch backend -- runs on CPU or CUDA, the sole PIV implementation        #
# ======================================================================== #
_HAN: dict = {}

# Peak-memory budget (bytes) for a single PIV pass's correlation. The grid of
# interrogation windows and its FFT buffers are processed in row-blocks sized to
# stay under this, so a 2048^2 / 4096^2 image no longer materialises every window
# at once and OOMs. Tiling is exact -- each window's correlation is independent,
# so a blocked pass is bit-identical to the unblocked one. 256 MiB leaves room
# for a napari viewer sharing the card; raise it to trade memory for fewer,
# larger kernel launches.
_PIV_TILE_BYTES = 256 * 1024 * 1024


def _piv_torch(ref, dfm, device, window=16, overlap=0.75, passes=8, coarse_factor=2.0,
               smooth=1.0, nmt_thresh=2.0, nmt_eps=0.1):
    """Whole multipass PIV resident on ``device`` (``torch.device`` -- CPU or CUDA).

    ``smooth`` is the per-pass Gaussian sigma (in sparse-grid cells) applied to the
    window-vector field after the normalized-median outlier test, before densification;
    0 disables it. ``nmt_thresh``/``nmt_eps`` tune that outlier test. Returns u_px
    (2, H, W) numpy, ``[0]=x/col``, ``[1]=y/row``, positive right/down.
    """
    import torch
    import torch.nn.functional as F

    dt = torch.float32

    def norm(a):
        a = torch.nan_to_num(a, nan=0.0)
        lo = torch.quantile(a, 0.01); hi = torch.quantile(a, 0.995)
        return torch.clamp((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    def han(win):
        key = (win, str(device))
        if key not in _HAN:
            h = torch.hann_window(win, periodic=False, device=device, dtype=dt)
            _HAN[key] = h[:, None] * h[None, :]
        return _HAN[key]

    def warp(img, u):
        # bicubic sub-pixel deformation (spline-order-3 equivalent); border padding
        # holds the edge value (map_coordinates mode='nearest' equivalent).
        H, W = img.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dt),
                                torch.arange(W, device=device, dtype=dt), indexing="ij")
        gx = 2.0 * (xx + u[0]) / max(W - 1, 1) - 1.0
        gy = 2.0 * (yy + u[1]) / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], -1)[None]
        return F.grid_sample(img[None, None], grid, mode="bicubic",
                             padding_mode="border", align_corners=True)[0, 0]

    def subpix(c, p, r):
        l = c.clamp_min(1e-6); pp = p.clamp_min(1e-6); rr = r.clamp_min(1e-6)
        den = l.log() - 2 * pp.log() + rr.log()
        return torch.where(den.abs() < 1e-9, torch.zeros_like(den),
                           0.5 * (l.log() - rr.log()) / den)

    def one_pass(refi, dfmi, win, step):
        # unfold() returns strided *views* (no copy); the (Ny,Nx,win,win) grid is only
        # materialised a block of rows at a time, inside the loop, to bound peak memory.
        Au = refi.unfold(0, win, step).unfold(1, win, step)       # (Ny,Nx,win,win) view
        Bu = dfmi.unfold(0, win, step).unfold(1, win, step)
        Ny, Nx = Au.shape[:2]
        h = han(win)
        ix = torch.arange(Nx, device=device)[None, :]
        du = torch.empty((Ny, Nx), device=device, dtype=dt)
        dv = torch.empty((Ny, Nx), device=device, dtype=dt)
        # ~32*win^2 bytes/window covers the real A,B blocks, the rfft2 complex buffers,
        # and the irfft2 output held at once; pick the row-block height from the budget.
        rows = max(1, _PIV_TILE_BYTES // max(1, 32 * win * win * Nx))
        for y0 in range(0, Ny, rows):
            y1 = min(Ny, y0 + rows)
            A = Au[y0:y1]; B = Bu[y0:y1]
            A = (A - A.mean((-2, -1), keepdim=True)) * h
            B = (B - B.mean((-2, -1), keepdim=True)) * h
            R = torch.fft.irfft2(torch.conj(torch.fft.rfft2(A)) * torch.fft.rfft2(B), s=(win, win))
            nb = R.shape[0]
            flat = R.reshape(nb, Nx, -1).argmax(-1)
            py = flat // win; px = flat % win
            iy = torch.arange(nb, device=device)[:, None]
            def g(ay, ax):
                return R[iy, ix, ay % win, ax % win]
            dvb = torch.where(py >= win // 2, py - win, py).to(dt)    # Nyquist unwrap
            dvb = dvb + subpix(g(py - 1, px), g(py, px), g(py + 1, px))
            dub = torch.where(px >= win // 2, px - win, px).to(dt)
            dub = dub + subpix(g(py, px - 1), g(py, px), g(py, px + 1))
            du[y0:y1] = dub; dv[y0:y1] = dvb
        ys = np.arange(0, (Ny - 1) * step + 1, step) + win // 2
        xs = np.arange(0, (Nx - 1) * step + 1, step) + win // 2
        return ys, xs, du, dv

    def med3(x):
        pad = F.pad(x[None, None], (1, 1, 1, 1), mode="replicate")
        nb = pad.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*x.shape, 9)
        return nb.median(-1).values

    def nmt(f, thresh=nmt_thresh, eps=nmt_eps):
        # Normalized median test (Westerweel & Scarano 2005): replace a vector whose
        # residual to the local median, normalized by the local median residual,
        # exceeds ``thresh``. A stronger outlier detector than a plain absolute-median
        # test, which keeps the field close to ground truth on stress tests (large
        # displacement + noise). ``eps`` is the fluctuation noise floor.
        med = med3(f)
        res = (f - med).abs()
        bad = res / (med3(res) + eps) > thresh
        return torch.where(bad, med, f)

    def gauss(f, sigma):
        # Separable Gaussian blur of the sparse vector grid (reflect pad). This is the
        # per-pass displacement-field smoother (``piv_smooth``): the coarse passes carry
        # the large-scale motion and it regularizes the noisy fine residual. Clamp the
        # radius to the grid size -- the coarsest multipass grids are only a few windows
        # across, where a large sigma would overflow reflect padding (harmless: the field
        # is near-uniform over its own footprint there).
        if sigma <= 0:
            return f
        rad = min(max(1, int(round(3 * sigma))), min(f.shape) - 1)
        if rad < 1:
            return f
        x = torch.arange(-rad, rad + 1, device=f.device, dtype=f.dtype)
        k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
        fp = F.pad(f[None, None], (rad, rad, rad, rad), mode="reflect")
        fp = F.conv2d(fp, k.view(1, 1, 1, -1))
        fp = F.conv2d(fp, k.view(1, 1, -1, 1))
        return fp[0, 0]

    def to_dense(ys, xs, du, dv, H, W):
        # Per pass: outlier-reject (NMT), smooth (piv_smooth), then resample the sparse
        # vector grid to full res, LINEAR-extrapolating past the grid edge (clamped cell
        # index, unclamped interpolation weight) so the border half-window is filled
        # rather than clamped to the edge value.
        if len(ys) < 2 or len(xs) < 2:
            return torch.zeros((2, H, W), device=device, dtype=dt)
        du = gauss(nmt(du), smooth); dv = gauss(nmt(dv), smooth)
        Ny, Nx = du.shape
        y0, x0 = float(ys[0]), float(xs[0]); sy = float(ys[1] - ys[0]); sx = float(xs[1] - xs[0])
        fy = (torch.arange(H, device=device, dtype=dt) - y0) / sy
        fx = (torch.arange(W, device=device, dtype=dt) - x0) / sx
        i0 = fy.floor().clamp(0, Ny - 2).long(); wy = (fy - i0)[:, None]
        j0 = fx.floor().clamp(0, Nx - 2).long(); wx = (fx - j0)[None, :]

        def bilin(fld):
            top = fld[i0][:, j0] * (1 - wx) + fld[i0][:, j0 + 1] * wx
            bot = fld[i0 + 1][:, j0] * (1 - wx) + fld[i0 + 1][:, j0 + 1] * wx
            return top * (1 - wy) + bot * wy

        return torch.stack([bilin(du), bilin(dv)])

    refi = norm(torch.as_tensor(np.asarray(ref), dtype=dt, device=device))
    dfmi = norm(torch.as_tensor(np.asarray(dfm), dtype=dt, device=device))
    H, W = refi.shape
    u = torch.zeros((2, H, W), device=device, dtype=dt)
    for win in _window_schedule(window, passes, coarse_factor, H, W):
        win = max(8, min(win, min(H, W) // 3)); win -= win % 2
        step = max(4, int(round(win * (1.0 - overlap))))
        # Symmetric window deformation: split the accumulated flow over both frames
        # (each warped by half), correlate the residual, then add it to the accumulator.
        ys, xs, du, dv = one_pass(warp(refi, -0.5 * u), warp(dfmi, 0.5 * u), win, step)
        u = u + to_dense(ys, xs, du, dv, H, W)
    return u.cpu().numpy()


# ======================================================================== #
#  analyzer                                                                #
# ======================================================================== #
class PIVDisplacementAnalyzer(BaseDisplacementAnalyzer):
    """Estimate displacement by multi-pass FFT cross-correlation PIV.

    A single torch implementation (:func:`_piv_torch`) runs on either device:
    ``disp_device="cpu"`` on the CPU, ``"cuda"`` on a CUDA GPU, ``"auto"`` on CUDA
    when present else CPU. torch is required (no openpiv fallback). Reads
    ``piv_window``/``piv_overlap``/``piv_passes``/``piv_smooth``.
    """

    algorithm_name = "PIV"
    smoothing_param_name = "piv_smooth"
    smoothing_candidates = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0)
    convergence_param_name = "piv_passes"
    convergence_candidates = (1, 2, 4, 6, 8, 10, 12)

    def __init__(self, params: Optional[DisplacementParameters] = None):
        super().__init__(params)
        try:
            import torch  # torch is a hard dependency of the PIV backend (CPU path too)
        except ImportError as exc:
            raise ImportError(
                "The PIV backend requires PyTorch, a core dependency of napariTFM. "
                "Reinstall napariTFM, or `pip install torch`."
            ) from exc
        # resolve_gpu_device returns a CUDA device for "cuda"/"auto"(+CUDA), or None for
        # "cpu"/"auto"(no CUDA); the sole PIV path is torch, so map None -> the CPU device.
        dev = resolve_gpu_device(str(self.params.disp_device), method="PIV")
        self._device = dev if dev is not None else torch.device("cpu")
        self._backend = "torch"

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray,
                       weight: np.ndarray | None = None) -> np.ndarray:
        """Cross-correlate ``moving`` against ``reference`` and return the full-res flow
        as ``(H, W, 2)`` float32 in pixels ([...,0]=u_x, [...,1]=u_y).

        ``weight`` is ignored (PIV's confinement is the upstream crop); it is
        accepted only to satisfy the shared analyzer interface."""
        u = _piv_torch(
            reference, moving, self._device,
            window=max(8, int(self.params.piv_window)),
            overlap=float(self.params.piv_overlap),
            passes=max(1, int(self.params.piv_passes)),
            smooth=max(0.0, float(self.params.piv_smooth)),
        )
        H, W = np.asarray(reference).shape
        return self._pack(u, H, W)
