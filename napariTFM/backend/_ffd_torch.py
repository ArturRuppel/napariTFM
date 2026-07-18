"""Grid-pyramid FFD (free-form deformation) -- GPU-only displacement backend.

Vendored, unchanged in behaviour, from benchmarkTFM's
``benchmarktfm.displacement.ffd_pyr`` (plus the control-grid and metric helpers it
imports from ``ffd_gpu``). This is napariTFM's own registration method: a B-spline
control grid whose spacing is CONSTANT in *level* pixels over a coarse-to-fine
image pyramid, so it is genuinely coarse on the coarse image (few control points,
bulk motion) and fine on the fine image (detail). Capture range comes from image
downsampling; the field is carried between levels by pre-warping the deformed
image (residual composition).

The knob that earns the method its keep is ``level_spacing`` -- where refinement
stops getting finer -- which IS the bias-variance dial: fine (~8 px) recovers sharp
peaks on clean data, coarse (~24 px) is the noise regularizer. Its optimum scales
with noise. ``num_levels`` sets the pyramid depth (capture range for large motion).

This module imports :mod:`torch` at top level and is imported **lazily** by
:class:`napariTFM.backend.ffd_displacement.FFDDisplacementAnalyzer`. FFD has no
CPU implementation: without the ``[gpu]`` extra it is unavailable, by design (the
LBFGS control-grid fit is impractical on CPU).

Contract: ``ffd_pyr(ref, dfm, **params) -> u_px (2,H,W)`` float32, ``[0]=x/col``,
``[1]=y/row``; positive = rightward/downward.
"""
import numpy as np
import torch
import torch.nn.functional as F

from napariTFM.backend._flow_common import _base_grid, _device, _norm01, _pyramid, _warp

_LNCC_WIN = 9          # local-CC window (px); the scale over which structure is correlated


# ---------------------------------------------------------------- control grid #
def _bspline_basis(n, s, device):
    """1-D cubic B-spline sampling weights for an axis of length ``n`` at spacing ``s``.

    Returns ``(idx, w)``: ``idx`` (n,) long is the base control index ``floor(x/s)`` for each
    pixel x, and ``w`` (n,4) the four cubic weights B0..B3(frac) for control points
    ``idx, idx+1, idx+2, idx+3``. The four weights sum to 1 (partition of unity). Computed in
    float64 for a clean fraction, the weights returned float64 (the caller casts)."""
    x = torch.arange(n, device=device, dtype=torch.float64)
    t = x / s
    idx = torch.floor(t).long()
    u = t - idx.double()
    u2 = u * u
    u3 = u2 * u
    w = torch.stack([
        (1.0 - u) ** 3,
        3.0 * u3 - 6.0 * u2 + 4.0,
        -3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0,
        u3,
    ], dim=1) / 6.0
    return idx, w


def _basis_matrix(n, s, G, device, dtype):
    """Dense ``(n, G)`` cubic B-spline weight matrix for an axis: row ``x`` holds that pixel's
    four cubic weights at its supporting control columns ``floor(x/s)..+3``, zero elsewhere.

    Lets the field be evaluated as a pair of matmuls (see :func:`_field_from_grid`) whose
    backward is a GEMM rather than the atomic scatter-add that the equivalent gather's backward
    becomes -- the same field, but the scatter dominated the optimiser at large images."""
    idx, w = _bspline_basis(n, s, device)        # idx (n,) base control index, w (n,4) weights
    M = torch.zeros(n, G, device=device, dtype=dtype)
    cols = idx[:, None] + torch.arange(4, device=device)   # (n,4), guaranteed in [0, G-1]
    M.scatter_(1, cols, w.to(dtype))
    return M


def _field_from_grid(C, My, Mx):
    """Dense field (2,H,W) from control grid C (2,Gh,Gw) as a separable cubic B-spline.

    ``field[c] = My @ C[c] @ Mx^T`` with ``My`` (H,Gh), ``Mx`` (W,Gw) from :func:`_basis_matrix`.
    Numerically identical to the per-pixel tensor-product sum
    ``sum_{a,b} wy[h,a] wx[w,b] C[c, iy[h]+a, ix[w]+b]``, but evaluated as two matmuls so its
    backward is two GEMMs, not a scatter-add over C -- which was the single largest cost in the
    LBFGS loop at 2048^2. Fully differentiable in C. Units follow C (full-res pixels)."""
    tmp = torch.matmul(My, C)                    # (2,H,Gw)  -- contract row neighbours
    return torch.matmul(tmp, Mx.transpose(0, 1))  # (2,H,W)  -- contract col neighbours


# ------------------------------------------------------------------- metrics #
def _mse(a, b):
    return ((a - b) ** 2).mean()


def _boxmean(x, w):
    """Reflect-padded w x w box mean of a 2D tensor -- the local-window aggregator for LNCC.
    Kept as a single dense w x w conv, not a separable pair: the LBFGS loop is launch-bound,
    so the second conv's launch + pad cost more than the arithmetic a separable form saves."""
    k = torch.ones(1, 1, w, w, device=x.device, dtype=x.dtype) / (w * w)
    return F.conv2d(F.pad(x[None, None], (w // 2,) * 4, mode="reflect"), k)[0, 0]


def _lncc(a, b):
    """Local normalised cross-correlation LOSS (1 - mean squared local CC), ANTs-style. Unlike
    MSE, every w x w window contributes equally regardless of its intensity or the region's area,
    so the sharp high-motion peak is not drowned by the low-motion bulk. Differentiable through
    the warp. Lower is better; 0 = perfect local alignment everywhere."""
    w = _LNCC_WIN
    ma, mb = _boxmean(a, w), _boxmean(b, w)
    va = _boxmean(a * a, w) - ma * ma
    vb = _boxmean(b * b, w) - mb * mb
    cov = _boxmean(a * b, w) - ma * mb
    cc = cov * cov / (va * vb + 1e-5)
    return 1.0 - cc.mean()


def _elastic_energy(C, s, nu=0.45):
    """Linear-elastic (Navier) strain energy of the deformation, on the control lattice.

    The physically correct prior for a TFM gel: the substrate's displacement genuinely minimises
    this energy under its traction boundary conditions, so penalising it biases the fit toward
    fields a real gel can produce. It penalises FIRST derivatives (strain) rather than second
    (curvature), so it permits concentrated strain at localised force points while forbidding
    physically implausible roughness. ``C[0]``=row(y) disp, ``C[1]``=col(x) disp; control points
    ``s`` px apart. ``nu`` is the substrate Poisson ratio (mu=1; lambda from plane-strain Lame);
    keep nu<0.5 (0.5 is singular)."""
    d = 1.0 / (2.0 * s)
    exx = (C[1, 1:-1, 2:] - C[1, 1:-1, :-2]) * d        # du_x/dx
    eyy = (C[0, 2:, 1:-1] - C[0, :-2, 1:-1]) * d        # du_y/dy
    dux_dy = (C[1, 2:, 1:-1] - C[1, :-2, 1:-1]) * d
    duy_dx = (C[0, 1:-1, 2:] - C[0, 1:-1, :-2]) * d
    exy = 0.5 * (dux_dy + duy_dx)
    tr = exx + eyy
    mu = 1.0
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    return (0.5 * lam * tr ** 2 + mu * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2)).mean()


# ------------------------------------------------------------------ driver #
def _resize_field(f, Hl, Wl):
    """Resize a displacement field (2,Ha,Wa in that level's px) to (Hl,Wl), scaling magnitude by
    the per-axis resolution ratio so it stays a correct displacement at the new resolution."""
    Ha, Wa = f.shape[1:]
    r = F.interpolate(f[None], size=(Hl, Wl), mode="bilinear", align_corners=True)[0].clone()
    r[0] *= (Hl - 1) / (Ha - 1) if Ha > 1 else 1.0     # row/y
    r[1] *= (Wl - 1) / (Wa - 1) if Wa > 1 else 1.0     # col/x
    return r


def ffd_pyr(ref, dfm, level_spacing=12.0, num_levels=6, downscale=2.0, min_size=16,
            num_iters=50, metric="lncc", elastic=0.0, tol=0.0, interp="bicubic",
            device=None, dtype=torch.float32, verbose=False, init_field=None,
            return_loss=False, early_stop=0.0):
    """Coarse-to-fine GRID pyramid over an image pyramid. ``level_spacing`` = control spacing in
    LEVEL pixels (constant across levels -> coarse grid on coarse image, fine on fine); it is the
    bias-variance dial (fine = sharp peaks, coarse = noise-regularized). Each level pre-warps dfm
    by the carried field and fits a fresh grid to the residual. ``tol>0`` stops early when a
    level's data-loss gain < tol.

    ``init_field`` warm-starts the fit: pass the previous frame's result (same ``u_px (2,H,W)``
    ``[0]=x/col,[1]=y/row`` layout) and the pyramid starts from it instead of zero, so each level
    only fits the small frame-to-frame delta. Valid because every frame is registered to the same
    fixed reference; the full pyramid is kept, so a bad guess (discontinuity) just costs iterations,
    never divergence. Returns u_px (2,H,W) float32, [0]=x/col [1]=y/row.

    ``early_stop`` is the per-level LBFGS convergence tolerance (``tolerance_change``): >0 lets a
    level's LBFGS quit before ``num_iters`` once its loss stops improving by that much, which is
    where warm-started frames recoup most of their speed-up (a converged level burns no further
    iterations chasing a near-zero delta). ``0`` (default) uses LBFGS's own default tolerance,
    i.e. the pre-existing behaviour, bit-for-bit.

    ``return_loss=True`` additionally returns the finest reached level's data-loss (the fit's
    alignment quality, lower = better) as ``(u_px, data_loss)``. The default ``False`` keeps the
    plain ``u_px`` return so existing callers are unaffected."""
    dev = _device(device)
    I0 = _norm01(torch.as_tensor(np.asarray(ref), dtype=dtype, device=dev))
    I1 = _norm01(torch.as_tensor(np.asarray(dfm), dtype=dtype, device=dev))
    H, W = I0.shape
    loss_fn = _lncc if metric == "lncc" else _mse
    p0 = _pyramid(I0, downscale, nlevel=num_levels, min_size=min_size)   # coarsest first
    p1 = _pyramid(I1, downscale, nlevel=num_levels, min_size=min_size)

    if init_field is None:
        field = torch.zeros((2, H, W), device=dev, dtype=dtype)         # accumulated, full-res px
    else:
        ext = torch.as_tensor(np.asarray(init_field), dtype=dtype, device=dev)  # (2,H,W) [x/col,y/row]
        field = torch.stack([ext[1], ext[0]])                          # -> internal [row/y, col/x]
    prev = None
    for J0, J1 in zip(p0, p1):
        Hl, Wl = J0.shape
        base = _base_grid(Hl, Wl, dev, dtype)
        field_l = _resize_field(field, Hl, Wl)                         # carried field, level px
        with torch.no_grad():
            J1w = _warp(J1, field_l, base, mode=interp)                # pre-warp by carried field

        Gh = int(np.floor((Hl - 1) / level_spacing)) + 4
        Gw = int(np.floor((Wl - 1) / level_spacing)) + 4
        My = _basis_matrix(Hl, float(level_spacing), Gh, dev, dtype)
        Mx = _basis_matrix(Wl, float(level_spacing), Gw, dev, dtype)
        C = torch.zeros((2, Gh, Gw), device=dev, dtype=dtype, requires_grad=True)
        if early_stop > 0:
            opt = torch.optim.LBFGS([C], max_iter=int(num_iters), line_search_fn="strong_wolfe",
                                     tolerance_change=early_stop)
        else:
            opt = torch.optim.LBFGS([C], max_iter=int(num_iters), line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            inc = _field_from_grid(C, My, Mx)
            warped = _warp(J1w, inc, base, mode=interp)
            loss = loss_fn(warped, J0)
            if elastic:
                loss = loss + elastic * _elastic_energy(C, float(level_spacing))
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            inc = _field_from_grid(C, My, Mx)
            total_l = field_l + inc                                    # level px
            dl = float(loss_fn(_warp(J1, total_l, base, mode=interp), J0))
            field = _resize_field(total_l, H, W)                       # carry up to full res
        if verbose:
            print(f"    level {Hl:3}px spacing{level_spacing:g} (G {Gh}x{Gw}): data {dl:.4f} "
                  f"|field|max {float(field.abs().max()):.1f}")
        if tol > 0 and prev is not None and prev - dl < tol:
            break
        prev = dl

    v, u = field[0], field[1]
    # float32, not float64: the analyzer packs this straight into a float32 field, so the
    # upcast only doubled the device->host copy (same as the iLK path).
    out = torch.stack([u, v]).cpu().numpy()
    if return_loss:
        return out, dl
    return out
