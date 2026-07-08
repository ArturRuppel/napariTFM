"""Multi-pass PIV (particle image velocimetry) displacement backend.

Estimates the in-plane displacement between a reference and a moving image by
FFT cross-correlation of interrogation windows, coarse-to-fine with window
deformation (Hanning taper, normalised-median outlier rejection, 3-point
Gaussian subpixel peak) -- the classic PIV algorithm, and the most accurate
displacement front end on dense bead images in our benchmarks.

PIV has a **torch-free numpy core**, so it runs on a plain install with no extra
dependencies. When ``torch`` is installed and a CUDA device is available it
transparently uses a GPU backend that is ~100x faster and numerically equivalent
on well-posed (dense-bead) data -- the regime real TFM images live in.
``piv_device`` selects: ``"auto"`` (GPU if available, else numpy), ``"cuda"``
(require GPU), or ``"cpu"`` (force the numpy core).

``calculate_flow`` returns ``(reference, moving) -> (H, W, 2) float32``
displacement in **pixels** at full native resolution, with ``[..., 0] = u_x``
(columns) and ``[..., 1] = u_y`` (rows); positive = rightward/downward.
Downscaling and the pixel->µm conversion happen downstream in
:func:`napariTFM.backend.displacement_analysis.calculate_displacement_field`.
"""
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sp_fft, ndimage
from scipy.interpolate import RegularGridInterpolator

from napariTFM.backend.parameter_dataclasses import DisplacementParameters


# ======================================================================== #
#  numpy core -- torch-free, always available                              #
# ======================================================================== #
def _norm(a: np.ndarray) -> np.ndarray:
    """Percentile stretch to [0, 1] (1st..99.5th percentile), NaNs zeroed."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
    lo, hi = np.percentile(a, [1.0, 99.5])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _warp(img: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample the moving image at (x + u_x, y + u_y) so it aligns back onto the
    reference by the current estimate, leaving only the residual for the next pass."""
    H, W = img.shape
    ii, jj = np.mgrid[0:H, 0:W]
    return ndimage.map_coordinates(img, [ii + u[1], jj + u[0]], order=1, mode="nearest")


def _subpixel(c, cpk, r):
    """3-point Gaussian peak offset along one axis (left, peak, right samples).

    Vectorised: ``c``/``cpk``/``r`` may be scalars or arrays. Returns 0 where the
    log-curvature is degenerate."""
    l = np.maximum(c, 1e-6); p = np.maximum(cpk, 1e-6); rr = np.maximum(r, 1e-6)
    denom = np.log(l) - 2 * np.log(p) + np.log(rr)
    # np.where evaluates both branches, so the division runs where denom~0 too;
    # its inf/nan is masked out below -- silence the spurious divide warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = 0.5 * (np.log(l) - np.log(rr)) / denom
    return np.where(np.abs(denom) < 1e-9, 0.0, offset)


def _pass(ref: np.ndarray, dfm: np.ndarray, win: int, step: int):
    """One PIV pass on already-aligned images. Returns centre grid (ys, xs) and
    residual (du, dv) per window via **batched** FFT cross-correlation + subpixel
    Gaussian.

    Every interrogation window is extracted as a single strided view and
    correlated in one batched real-FFT call (``scipy.fft``, multithreaded) — a
    numpy mirror of the GPU backend's ``one_pass``, numerically identical to it.
    The window stack is processed in row-bands so the materialised windows and
    FFT temporaries stay bounded (~a few hundred MB) regardless of image size."""
    swr = sliding_window_view(ref, (win, win))[::step, ::step]     # (Ny,Nx,win,win) view
    swd = sliding_window_view(dfm, (win, win))[::step, ::step]
    Ny, Nx = swr.shape[:2]
    han = np.hanning(win)[:, None] * np.hanning(win)[None, :]       # taper -> less edge leakage
    du = np.zeros((Ny, Nx)); dv = np.zeros((Ny, Nx))

    band = max(1, (256 << 20) // max(Nx * win * win * 4, 1))       # ~256 MB float32 per band
    for y0 in range(0, Ny, band):
        y1 = min(y0 + band, Ny)
        a = swr[y0:y1].astype(np.float32); b = swd[y0:y1].astype(np.float32)
        a = (a - a.mean((-2, -1), keepdims=True)) * han
        b = (b - b.mean((-2, -1), keepdims=True)) * han
        # circular cross-correlation of the whole band at once; peak (vs origin)
        # = shift of b w.r.t. a = +u. Un-shifted (no fftshift): the peak sits
        # near index 0 and Nyquist-unwraps below, exactly as the GPU path does.
        r = sp_fft.irfft2(np.conj(sp_fft.rfft2(a, workers=-1)) * sp_fft.rfft2(b, workers=-1),
                          s=(win, win), workers=-1)
        nb = y1 - y0
        flat = r.reshape(nb, Nx, -1).argmax(-1)
        py = flat // win; px = flat % win
        iy = np.arange(nb)[:, None]; ix = np.arange(Nx)[None, :]
        def g(ay, ax):
            return r[iy, ix, ay % win, ax % win]                   # modular = circular neighbours
        dv[y0:y1] = np.where(py >= win // 2, py - win, py) + _subpixel(g(py - 1, px), g(py, px), g(py + 1, px))
        du[y0:y1] = np.where(px >= win // 2, px - win, px) + _subpixel(g(py, px - 1), g(py, px), g(py, px + 1))

    ys = np.arange(Ny) * step + win // 2
    xs = np.arange(Nx) * step + win // 2
    return ys, xs, du, dv


def _nmt_replace(f: np.ndarray, thresh: float = 2.0, eps: float = 0.1) -> np.ndarray:
    """Normalised median test: replace outlier vectors by the local median (3x3)."""
    med = ndimage.median_filter(f, size=3, mode="nearest")
    res = np.abs(f - med)
    fluct = ndimage.median_filter(res, size=3, mode="nearest")
    bad = res / (fluct + eps) > thresh
    out = f.copy(); out[bad] = med[bad]
    return out


def _to_dense(ys, xs, du, dv, H, W):
    """Sparse window vectors on a regular grid -> dense (2,H,W)."""
    if len(ys) < 2 or len(xs) < 2:                  # too few windows to interpolate
        return np.zeros((2, H, W))
    du = _nmt_replace(du)
    dv = _nmt_replace(dv)
    gy, gx = np.mgrid[0:H, 0:W]
    pts = np.stack([gy.ravel(), gx.ravel()], -1)
    fu = RegularGridInterpolator((ys, xs), du, bounds_error=False, fill_value=None)
    fv = RegularGridInterpolator((ys, xs), dv, bounds_error=False, fill_value=None)
    return np.stack([fu(pts).reshape(H, W), fv(pts).reshape(H, W)])


def _window_schedule(window: int, passes: int, coarse_factor: float, H: int, W: int):
    """Geometric coarse->fine window sizes from a capped top window down to ``window``."""
    cap = max(window, min(H, W) // 3)
    if passes == 1:
        return [window]
    top = max(window, min(int(round(window * coarse_factor ** (passes - 1))), cap))
    return [int(round(top * (window / top) ** (p / (passes - 1)))) for p in range(passes)]


def _piv_numpy(ref, dfm, window=16, overlap=0.75, passes=8, coarse_factor=2.0):
    """Multi-pass window-deformation PIV (numpy). Returns u_px (2,H,W) [0]=x/col [1]=y/row."""
    ref, dfm = _norm(ref), _norm(dfm)
    H, W = ref.shape
    u = np.zeros((2, H, W))
    for win in _window_schedule(window, passes, coarse_factor, H, W):
        win = max(8, min(win, min(H, W) // 3))
        win -= win % 2                                            # even
        step = max(4, int(round(win * (1.0 - overlap))))
        warped = _warp(dfm, u)                                    # align by current estimate
        ys, xs, du, dv = _pass(ref, warped, win, step)
        u = u + _to_dense(ys, xs, du, dv, H, W)                   # accumulate the residual
    return u


# ======================================================================== #
#  torch / GPU backend -- lazily imported, numerically equiv on dense data #
# ======================================================================== #
_HAN: dict = {}


def _piv_torch(ref, dfm, device, window=16, overlap=0.75, passes=8,
               coarse_factor=2.0):
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
        A = refi.unfold(0, win, step).unfold(1, win, step)        # (Ny,Nx,win,win) view
        B = dfmi.unfold(0, win, step).unfold(1, win, step)
        h = han(win)
        A = (A - A.mean((-2, -1), keepdim=True)) * h
        B = (B - B.mean((-2, -1), keepdim=True)) * h
        R = torch.fft.irfft2(torch.conj(torch.fft.rfft2(A)) * torch.fft.rfft2(B), s=(win, win))
        Ny, Nx = R.shape[:2]
        flat = R.reshape(Ny, Nx, -1).argmax(-1)
        py = flat // win; px = flat % win
        iy = torch.arange(Ny, device=device)[:, None]
        ix = torch.arange(Nx, device=device)[None, :]
        def g(ay, ax):
            return R[iy, ix, ay % win, ax % win]
        dv = torch.where(py >= win // 2, py - win, py).to(dt)     # Nyquist unwrap
        dv = dv + subpix(g(py - 1, px), g(py, px), g(py + 1, px))
        du = torch.where(px >= win // 2, px - win, px).to(dt)
        du = du + subpix(g(py, px - 1), g(py, px), g(py, px + 1))
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
class PIVDisplacementAnalyzer:
    """Estimate displacement by multi-pass FFT cross-correlation PIV.

    Torch-free by default (numpy core); transparently GPU-accelerated when torch
    and CUDA are available.
    """

    def __init__(self, params: Optional[DisplacementParameters] = None):
        """Initialise the PIV analyzer.

        Args:
            params: Algorithm parameters. The PIV-specific fields consumed are
                ``piv_window`` (final interrogation window, px), ``piv_overlap``
                (window overlap fraction), ``piv_passes`` (coarse->fine passes),
                and ``piv_device`` (``"auto"``/``"cuda"``/``"cpu"``).

        Raises:
            ImportError: only if ``piv_device="cuda"`` and torch is missing.
                ``"auto"`` and ``"cpu"`` never require torch.
        """
        self.params = params or DisplacementParameters()
        self.algorithm_name = "PIV"
        self._backend, self._device = self._resolve_backend(str(self.params.piv_device))

    @staticmethod
    def _resolve_backend(request: str) -> Tuple[str, object]:
        """Pick ("numpy", None) or ("torch", torch.device). "auto" prefers CUDA,
        falls back to numpy; "cuda" requires it; "cpu" forces the numpy core."""
        request = (request or "auto").lower()
        if request == "cpu":
            return "numpy", None
        if request == "cuda":
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
                raise ImportError(
                    "piv_device='cuda' needs PyTorch, which is not installed. "
                    "Install the optional extra with `pip install napariTFM[piv]` "
                    "(it provides torch), or set piv_device='auto'/'cpu'."
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "piv_device='cuda' but no CUDA device is available; "
                    "use piv_device='auto' or 'cpu'."
                )
            return "torch", torch.device("cuda")
        # "auto": GPU if torch+CUDA are both present, else numpy.
        try:
            import torch
            if torch.cuda.is_available():
                return "torch", torch.device("cuda")
        except ImportError:
            pass
        return "numpy", None

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Cross-correlate ``moving`` against ``reference`` and return the full-res flow.

        Returns:
            np.ndarray: Displacement field, shape ``(H, W, 2)``, ``float32``, in
            **pixels**, full native resolution. ``[..., 0] = u_x`` (columns),
            ``[..., 1] = u_y`` (rows); positive = rightward/downward. A
            degenerate/blank pair returns a zero field.
        """
        window = max(8, int(self.params.piv_window))
        overlap = float(self.params.piv_overlap)
        passes = max(1, int(self.params.piv_passes))
        kw = dict(window=window, overlap=overlap, passes=passes)

        if self._backend == "torch":
            u = _piv_torch(reference, moving, self._device, **kw)
        else:
            u = _piv_numpy(reference, moving, **kw)

        H, W = np.asarray(reference).shape
        flow = np.stack([u[0], u[1]], axis=-1).astype(np.float32, copy=False)  # (H,W,2): u_x,u_y
        if not np.isfinite(flow).all():
            import warnings
            warnings.warn(
                "PIV produced a non-finite displacement field for one frame pair "
                "(degenerate/blank input?); returning zeros for it.",
                RuntimeWarning,
            )
            flow = np.zeros((H, W, 2), dtype=np.float32)
        return flow

    def downscale_flow(self, flow: np.ndarray, factor: int) -> np.ndarray:
        """Downscale the dense flow field by block-mean averaging.

        Reduces flow-field resolution while preserving vector magnitudes by
        averaging displacement vectors within non-overlapping ``factor x factor``
        tiles. ``factor <= 1`` returns the input unchanged.
        """
        if factor <= 1:
            return flow

        h, w = flow.shape[:2]
        new_h, new_w = h // factor, w // factor

        # Block-mean over non-overlapping factor x factor tiles (any remainder
        # rows/cols beyond new_h*factor / new_w*factor are dropped). Reshaping to
        # (new_h, factor, new_w, factor, 2) and averaging the two block axes is
        # the vectorized equivalent of the per-block loop.
        trimmed = flow[:new_h * factor, :new_w * factor]
        return (
            trimmed.reshape(new_h, factor, new_w, factor, 2)
            .mean(axis=(1, 3))
            .astype(np.float64)
        )
