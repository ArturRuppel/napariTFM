"""Shared torch scaffolding for the GPU displacement backends (iLK, FFD).

Vendored, unchanged in behaviour, from the benchmarkTFM project's
``benchmarktfm.displacement._gpu_common`` (the public benchmark that validated
these ports numerically). napariTFM does not depend on that package: the
algorithms are copied here so the plugin stands alone.

This module imports :mod:`torch` at top level, so it is imported **lazily** by
the analyzers (only on the GPU path). A plain ``pip install napariTFM`` without
the ``[gpu]`` extra never touches it.

skimage's ``optical_flow_ilk`` shares its pyramid/warp/gradient machinery with
``optical_flow_tvl1``; both are ``_coarse_to_fine(I0, I1, solver)`` differing
only in the per-level solver. So everything except the solver lives here. The
helpers are deliberately faithful to skimage (matched by the benchmark's
``verify_parity`` scripts) rather than idiomatic: where a docstring names the
skimage function it mirrors, that correspondence is the specification.
"""
import math

import torch
import torch.nn.functional as F


def _device(dev=None):
    if dev is not None:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _norm01(a):
    """Percentile stretch to [0,1] (1st..99.5th), NaNs zeroed -- identical to the CPU
    methods' _norm01 so every method sees the same input."""
    a = torch.nan_to_num(a, nan=0.0)
    lo = torch.quantile(a, 0.01)
    hi = torch.quantile(a, 0.995)
    return torch.clamp((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _symmetric_pad(im, r, dim):
    """Pad `im` by `r` along `dim` with scipy.ndimage's mode='reflect' -- SYMMETRIC padding
    (d c b a | a b c d | d c b a), which duplicates the edge sample.

    torch's F.pad(mode='reflect') is *not* this: it is ndi's mode='mirror'
    (d c b | a b c d | c b a), which does not duplicate the edge. Using F.pad here silently
    mismatched skimage's Gaussian at every border. Index-gather instead of pad, so any
    radius works regardless of axis length.
    """
    n = im.shape[dim]
    idx = torch.cat([
        torch.arange(r - 1, -1, -1),
        torch.arange(n),
        torch.arange(n - 1, n - r - 1, -1),
    ]).clamp_(0, n - 1).to(im.device)
    return im.index_select(dim, idx)


def _gaussian_blur(img, sigma, truncate=4.0):
    """Separable Gaussian blur matching ``scipy.ndimage.gaussian_filter(sigma, mode='reflect')``,
    which is what skimage's ``pyramid_reduce`` -> ``_smooth`` -> ``gaussian`` calls.

    Both the kernel radius and the boundary handling are load-bearing: ndi uses
    ``lw = int(truncate*sigma + 0.5)`` with truncate=4.0 (radius 3 at the pyramid's
    sigma=2/3, not the 2 that ceil(2*sigma) gives), and symmetric 'reflect' padding.
    """
    r = int(truncate * sigma + 0.5)
    if r < 1:
        return img
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x ** 2) / (sigma ** 2))
    k = k / k.sum()
    im = img[None, None]
    im = _symmetric_pad(im, r, 3)                     # cols
    im = F.conv2d(im, k.view(1, 1, 1, -1))
    im = _symmetric_pad(im, r, 2)                     # rows
    im = F.conv2d(im, k.view(1, 1, -1, 1))
    return im[0, 0]


def _pyramid(img, downscale=2.0, nlevel=10, min_size=16):
    """Coarse-to-fine list (coarsest first), mirroring skimage _get_pyramid: Gaussian
    pre-smooth (sigma = 2*downscale/6) then bilinear resize to ceil(dim/downscale)."""
    pyr = [img]
    size = min(img.shape)
    sigma = 2 * downscale / 6.0
    count = 1
    while count < nlevel and size > downscale * min_size:
        prev = pyr[-1]
        sm = _gaussian_blur(prev, sigma)
        out_h = int(math.ceil(prev.shape[0] / downscale))
        out_w = int(math.ceil(prev.shape[1] / downscale))
        red = F.interpolate(sm[None, None], size=(out_h, out_w), mode="bilinear",
                            align_corners=False)[0, 0]
        pyr.append(red)
        size = min(red.shape)
        count += 1
    return pyr[::-1]


def pyramid_num_levels(shape, downscale=2.0, min_size=16):
    """Number of levels ``_pyramid`` builds for an image of ``shape`` (h, w) when the
    only stop is the size floor: keep shrinking by ``downscale`` until the short side
    would fall to ``downscale * min_size`` or below. Mirrors ``_pyramid``'s size gate
    exactly (same ``ceil`` per axis), so a caller can report/size the pyramid depth
    without building it. Depends only on the shape and these two knobs -- there is no
    separate level-count cap."""
    h, w = int(shape[0]), int(shape[1])
    n = 1
    while min(h, w) > downscale * min_size:
        h = math.ceil(h / downscale)
        w = math.ceil(w / downscale)
        n += 1
    return n


def _resize_flow(flow, shape):
    """Rescale a (2,H,W) flow field to `shape`, scaling the vector values by the same
    factor. Mirrors skimage's ``_resize_flow``, i.e.
    ``ndi.zoom(flow, [1]+scale, order=0, mode='nearest', prefilter=False)``.

    The sampling convention is the subtle part and ``F.interpolate(mode='nearest')`` gets
    it wrong. ndi.zoom with the default grid_mode=False maps output index o to input
    ``round(o * (n_in - 1) / (n_out - 1))`` -- an align_corners=True convention -- whereas
    torch's nearest uses ``floor(o * n_in / n_out)``. The two agree when the ratio is
    exactly 2 (so a 512-px pyramid never exposed it) and diverge otherwise: a 500-px image
    pyramids to [500,250,125,63,32], and the 32->63 step then shifts the flow by up to a
    pixel. Rounding is ``floor(x + 0.5)`` to match scipy's nearest rather than torch's
    round-half-to-even. flow[0]=row, flow[1]=col.
    """
    H0, W0 = flow.shape[1:]
    H1, W1 = shape
    dev = flow.device
    zr = (H0 - 1) / (H1 - 1) if H1 > 1 else 1.0
    zc = (W0 - 1) / (W1 - 1) if W1 > 1 else 1.0
    ri = torch.floor(torch.arange(H1, device=dev, dtype=torch.float64) * zr + 0.5)
    ci = torch.floor(torch.arange(W1, device=dev, dtype=torch.float64) * zc + 0.5)
    ri = ri.long().clamp_(0, H0 - 1)
    ci = ci.long().clamp_(0, W0 - 1)
    r = flow.index_select(1, ri).index_select(2, ci).clone()
    r[0] *= H1 / H0
    r[1] *= W1 / W0
    return r


def _gradient(a):
    """np.gradient-style central differences (one-sided at edges), for axis 0 (row) and
    axis 1 (col). Returns (g_row, g_col)."""
    gr = torch.empty_like(a)
    gr[1:-1] = (a[2:] - a[:-2]) * 0.5
    gr[0] = a[1] - a[0]
    gr[-1] = a[-1] - a[-2]
    gc = torch.empty_like(a)
    gc[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gc[:, 0] = a[:, 1] - a[:, 0]
    gc[:, -1] = a[:, -1] - a[:, -2]
    return gr, gc


def _base_grid(H, W, device, dtype):
    """Absolute (row, col) coordinate grids, the `base` argument to _warp."""
    rr = torch.arange(H, device=device, dtype=dtype)[:, None].expand(H, W)
    cc = torch.arange(W, device=device, dtype=dtype)[None, :].expand(H, W)
    return rr, cc


def _warp(img, flow, base, mode="bilinear"):
    """Warp of img by flow (2,H,W)=[row,col], border padding. ``mode='bilinear'`` is
    skimage warp order=1 (what the iLK port needs for parity); ``mode='bicubic'`` is a
    higher-order resample that preserves sharp peaks better (FFD uses it, matching
    elastix's cubic-B-spline interpolation). base=(rows,cols) absolute-coord grids."""
    H, W = img.shape
    rows = base[0] + flow[0]
    cols = base[1] + flow[1]
    gx = 2.0 * cols / (W - 1) - 1.0          # grid_sample: x=col, y=row, normalized [-1,1]
    gy = 2.0 * rows / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(img[None, None], grid, mode=mode,
                        padding_mode="border", align_corners=True)
    return out[0, 0]
