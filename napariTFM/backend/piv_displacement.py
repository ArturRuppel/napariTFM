"""Multi-pass PIV (particle image velocimetry) displacement backend.

Estimates the in-plane displacement between a reference and a moving image by
FFT cross-correlation of interrogation windows, coarse-to-fine with window
deformation: the classic PIV algorithm, and the most accurate displacement front
end on dense bead images in our benchmarks.

Two numerically-equivalent implementations behind one parameter set. The default
**CPU** path is `openpiv <https://github.com/OpenPIV/openpiv-python>`_, the
community-standard reference used as the credibility anchor for the benchmark's
PIV baseline. When ``torch`` is installed (the ``[gpu]`` extra) and a CUDA device
is available/selected, a **GPU** port runs the same multipass window-deformation
scheme ~100x faster; it is at measured parity with openpiv on dense-bead data (the
regime real TFM images live in), not bit-identical: the two use different FFT
libraries, and PIV's integer peak-location argmax can pick a different pixel on a
near-tie, so exact equality is neither expected nor claimed.

Both paths read the **same three knobs** -- ``piv_window`` (final interrogation
window, px), ``piv_overlap`` (window overlap fraction), ``piv_passes``
(coarse->fine passes) -- so switching ``disp_device`` changes only the compute
backend, not the meaning of any parameter. ``calculate_flow`` returns
``(reference, moving) -> (H, W, 2) float32`` displacement in **pixels** at full
native resolution, ``[..., 0] = u_x`` (columns), ``[..., 1] = u_y`` (rows);
positive = rightward/downward. Downscaling and the pixel->µm conversion happen
downstream in :func:`napariTFM.backend.displacement_analysis.calculate_displacement_field`.
"""
from typing import Optional

import numpy as np

from napariTFM.backend._displacement_base import BaseDisplacementAnalyzer, resolve_gpu_device
from napariTFM.backend.parameter_dataclasses import DisplacementParameters


def _norm01(image: np.ndarray) -> np.ndarray:
    """Percentile stretch to [0, 1] (1st..99.5th percentile), NaNs zeroed. Shared by both
    backends so the CPU and GPU paths see identically-scaled inputs."""
    img = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0)
    lo, hi = np.percentile(img, [1.0, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _window_schedule(window: int, passes: int, coarse_factor: float, H: int, W: int):
    """Geometric coarse->fine window sizes from a capped top window down to ``window``.

    Shared by both backends. ``coarse_factor`` is fixed at 2.0 (matching openpiv's ×2
    windowsize schedule); it is not a user knob, so the CPU and GPU schedules agree.
    """
    cap = max(window, min(H, W) // 3)
    if passes == 1:
        return [window]
    top = max(window, min(int(round(window * coarse_factor ** (passes - 1))), cap))
    return [int(round(top * (window / top) ** (p / (passes - 1)))) for p in range(passes)]


# ======================================================================== #
#  openpiv CPU reference -- the default backend                            #
# ======================================================================== #
def _piv_openpiv(ref, dfm, window=16, overlap=0.75, passes=8, coarse_factor=2.0):
    """Multi-pass window-deformation PIV via openpiv (the trusted CPU reference).

    ``window`` is the final (finest) interrogation window; a coarse->fine schedule of
    ``passes`` windows ends at it (openpiv's own ×2 schedule). ``overlap`` is the
    fractional window overlap. Returns u_px (2,H,W) [0]=x/col [1]=y/row.

    openpiv returns a sparse vector grid in its own convention (x ascending cols, y
    ascending BOTTOM->TOP, v positive UP). We interpolate to a dense (H,W) field and flip
    the y-component so the sign/axis match this module's contract.
    """
    from openpiv import windef
    from scipy.interpolate import RegularGridInterpolator

    a = _norm01(ref); b = _norm01(dfm)
    H, W = a.shape
    n = max(1, int(passes)); w = int(window)
    # Same capped coarse->fine schedule as the GPU port (coarse_factor 2.0), so the
    # two backends agree on the windows, not just the three knobs. Coerce to even
    # sizes >= 8 (openpiv/FFT windows); overlap < window always for overlap < 1.
    sizes = _window_schedule(w, n, coarse_factor, H, W)
    windowsizes = tuple(max(8, s - s % 2) for s in sizes)
    overlaps = tuple(max(0, min(ws - 2, int(ws * float(overlap)))) for ws in windowsizes)
    s = windef.PIVSettings()
    s.windowsizes = windowsizes
    s.overlap = overlaps
    s.num_iterations = n
    s.deformation_method = "symmetric"
    s.interpolation_order = 3
    x, y, u, v, _ = windef.simple_multipass(a, b, s)

    xs = x[0, :]
    ys = y[:, 0]
    if ys[0] > ys[-1]:
        ys = ys[::-1]; u = u[::-1]; v = v[::-1]
    gi_u = RegularGridInterpolator((ys, xs), np.asarray(u, float), bounds_error=False, fill_value=None)
    gi_v = RegularGridInterpolator((ys, xs), np.asarray(v, float), bounds_error=False, fill_value=None)
    rr, cc = np.mgrid[0:H, 0:W]
    pts = np.stack([rr.ravel(), cc.ravel()], -1)
    ux = gi_u(pts).reshape(H, W)
    uy = -gi_v(pts).reshape(H, W)          # openpiv v is +up; contract wants +down
    # openpiv's grid is Cartesian-y (0 at the image BOTTOM, ascending upward), so the
    # dense sampling above lands vertically mirrored vs this module's array contract
    # (row 0 = top) and vs the torch backend. Flip rows to match. Verified against the
    # torch path (flipud both components -> corr 0.99); without it the *default* CPU
    # backend returns upside-down displacement, silently corrupting recovered traction.
    return np.stack([ux, uy])[:, ::-1].copy()


# ======================================================================== #
#  torch / GPU backend -- lazily imported, numerically equiv on dense data #
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


def _piv_torch(ref, dfm, device, window=16, overlap=0.75, passes=8, coarse_factor=2.0):
    """Whole multipass PIV resident on ``device`` (GPU). Returns u_px (2,H,W) numpy."""
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
        H, W = img.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dt),
                                torch.arange(W, device=device, dtype=dt), indexing="ij")
        gx = 2.0 * (xx + u[0]) / max(W - 1, 1) - 1.0
        gy = 2.0 * (yy + u[1]) / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], -1)[None]
        return F.grid_sample(img[None, None], grid, mode="bilinear",
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

    def nmt(f, thresh=2.0, eps=0.1):
        med = med3(f)
        res = (f - med).abs()
        bad = res / (med3(res) + eps) > thresh
        return torch.where(bad, med, f)

    def to_dense(ys, xs, du, dv, H, W):
        if len(ys) < 2 or len(xs) < 2:
            return torch.zeros((2, H, W), device=device, dtype=dt)
        du = nmt(du); dv = nmt(dv)
        Ny, Nx = du.shape
        y0, x0 = float(ys[0]), float(xs[0]); sy = float(ys[1] - ys[0]); sx = float(xs[1] - xs[0])
        ii = torch.arange(H, device=device, dtype=dt); jj = torch.arange(W, device=device, dtype=dt)
        gy = 2.0 * ((ii - y0) / sy) / max(Ny - 1, 1) - 1.0        # resample at TRUE grid coords
        gx = 2.0 * ((jj - x0) / sx) / max(Nx - 1, 1) - 1.0
        gyy, gxx = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([gxx, gyy], -1)[None]
        field = torch.stack([du, dv])[None]
        return F.grid_sample(field, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)[0]

    refi = norm(torch.as_tensor(np.asarray(ref), dtype=dt, device=device))
    dfmi = norm(torch.as_tensor(np.asarray(dfm), dtype=dt, device=device))
    H, W = refi.shape
    u = torch.zeros((2, H, W), device=device, dtype=dt)
    for win in _window_schedule(window, passes, coarse_factor, H, W):
        win = max(8, min(win, min(H, W) // 3)); win -= win % 2
        step = max(4, int(round(win * (1.0 - overlap))))
        ys, xs, du, dv = one_pass(refi, warp(dfmi, u), win, step)
        u = u + to_dense(ys, xs, du, dv, H, W)
    return u.cpu().numpy()


# ======================================================================== #
#  analyzer                                                                #
# ======================================================================== #
class PIVDisplacementAnalyzer(BaseDisplacementAnalyzer):
    """Estimate displacement by multi-pass FFT cross-correlation PIV.

    openpiv (CPU) by default; transparently GPU-accelerated via the torch port when
    torch and CUDA are available and ``disp_device`` allows it. Both backends read
    ``piv_window``/``piv_overlap``/``piv_passes`` identically.
    """

    algorithm_name = "PIV"

    def __init__(self, params: Optional[DisplacementParameters] = None):
        super().__init__(params)
        self._device = resolve_gpu_device(str(self.params.disp_device), method="PIV")
        self._backend = "torch" if self._device is not None else "openpiv"

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray,
                       weight: np.ndarray | None = None) -> np.ndarray:
        """Cross-correlate ``moving`` against ``reference`` and return the full-res flow
        as ``(H, W, 2)`` float32 in pixels ([...,0]=u_x, [...,1]=u_y).

        ``weight`` is ignored (PIV's confinement is the upstream crop); it is
        accepted only to satisfy the shared analyzer interface."""
        kw = dict(
            window=max(8, int(self.params.piv_window)),
            overlap=float(self.params.piv_overlap),
            passes=max(1, int(self.params.piv_passes)),
        )
        if self._backend == "torch":
            u = _piv_torch(reference, moving, self._device, **kw)
        else:
            u = _piv_openpiv(reference, moving, **kw)

        H, W = np.asarray(reference).shape
        return self._pack(u, H, W)
