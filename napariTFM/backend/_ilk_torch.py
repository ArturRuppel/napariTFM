"""GPU iterative Lucas-Kanade dense optical flow -- torch port of skimage's iLK.

Vendored, unchanged in behaviour, from benchmarkTFM's
``benchmarktfm.displacement.ilk_gpu`` (validated there against the CPU skimage
original to a few % relL2, so the CPU and GPU paths are the same algorithm). This
module imports :mod:`torch` at top level and is imported **lazily** by
:class:`napariTFM.backend.ilk_displacement.ILKDisplacementAnalyzer`, only when the
GPU path is selected.

Same algorithm as skimage ``optical_flow_ilk`` (Le Besnerais & Champagnat 2005,
Table 2): at every pixel a windowed local least-squares solve, iterated over a
coarse-to-fine warp schedule; ``radius`` is the half-window. The Gaussian pyramid
(``downscale``, ``min_size``) reproduces skimage's fixed internal pyramid, so the
two user knobs -- ``radius`` and ``num_warp`` -- carry the same meaning on CPU and
GPU.

Contract: ``ilk_gpu(ref, dfm, **params) -> u_px (2,H,W)``, ``[0]=x/col``,
``[1]=y/row``; positive = rightward/downward.
"""
import numpy as np
import torch
import torch.nn.functional as F

from napariTFM.backend._flow_common import (
    _base_grid, _device, _gradient, _norm01, _pyramid, _resize_flow, _warp,
)


def _reflect_axis(im, r, dim):
    """Reflect-pad `im` by `r` along spatial `dim` (2=row, 3=col), iterating when the
    window exceeds the array so it matches ndi 'mirror' for radius >= dim (torch's single
    F.pad reflect requires pad < dim)."""
    while r > 0:
        n = im.shape[dim]
        if n < 2:
            break
        step = min(r, n - 1)
        pad = [0, 0, 0, 0]
        pad[0:2] = [step, step] if dim == 3 else pad[0:2]
        pad[2:4] = [step, step] if dim == 2 else pad[2:4]
        im = F.pad(im, pad, mode="reflect")
        r -= step
    return im


def _box(a, radius):
    """Uniform (box) filter, window 2*radius+1, reflect padding -- matches ndi.uniform_filter
    mode='mirror' for any radius. Separable running mean via conv."""
    size = 2 * radius + 1
    k = torch.full((size,), 1.0 / size, device=a.device, dtype=a.dtype)
    im = a[None, None]
    im = _reflect_axis(im, radius, 3)          # cols
    im = _reflect_axis(im, radius, 2)          # rows
    im = F.conv2d(im, k.view(1, 1, 1, -1))
    im = F.conv2d(im, k.view(1, 1, -1, 1))
    return im[0, 0]


def _solve_level(I0, I1, flow, radius, num_warp):
    """skimage _ilk inner solver on one pyramid level. flow (2,H,W)=[row,col]; returned
    flow replaces the input each warp (total-flow formulation, not incremental)."""
    H, W = I0.shape
    base = _base_grid(H, W, I0.device, I0.dtype)
    for _ in range(num_warp):
        warp1 = _warp(I1, flow, base)
        gr, gc = _gradient(warp1)
        err = gr * flow[0] + gc * flow[1] + I0 - warp1
        Arr = _box(gr * gr, radius)
        Acc = _box(gc * gc, radius)
        Arc = _box(gr * gc, radius)
        br = _box(gr * err, radius)
        bc = _box(gc * err, radius)
        det = Arr * Acc - Arc * Arc
        bad = det.abs() < 1e-14
        det_s = torch.where(bad, torch.ones_like(det), det)
        fr = torch.where(bad, torch.zeros_like(det), (Acc * br - Arc * bc) / det_s)
        fc = torch.where(bad, torch.zeros_like(det), (Arr * bc - Arc * br) / det_s)
        flow = torch.stack([fr, fc])
    return flow


def ilk_gpu(ref, dfm, radius=7, num_warp=10, downscale=2.0, min_size=16, device=None):
    """GPU iterative Lucas-Kanade flow between reference and deformed image.

    Mirrors ``skimage.registration.optical_flow_ilk`` (uniform-kernel, no prefilter):
    Gaussian pyramid, then the iLK solver at each level with the flow upsampled between
    levels. Returns u_px (2,H,W) float32, [0]=x/col, [1]=y/row.
    """
    dev = _device(device)
    I0 = _norm01(torch.as_tensor(np.asarray(ref), dtype=torch.float32, device=dev))
    I1 = _norm01(torch.as_tensor(np.asarray(dfm), dtype=torch.float32, device=dev))
    radius = max(1, int(radius))
    num_warp = max(1, int(num_warp))

    p0 = _pyramid(I0, downscale, min_size=min_size)
    p1 = _pyramid(I1, downscale, min_size=min_size)

    flow = torch.zeros((2,) + p0[0].shape, dtype=torch.float32, device=dev)
    flow = _solve_level(p0[0], p1[0], flow, radius, num_warp)
    for J0, J1 in zip(p0[1:], p1[1:]):
        flow = _solve_level(J0, J1, _resize_flow(flow, J0.shape), radius, num_warp)

    v, u = flow[0], flow[1]                    # row/y, col/x
    # float32, not float64: the analyzer packs this straight into a float32 field
    # (_pack -> astype float32), so upcasting here only doubled the device->host copy.
    return torch.stack([u, v]).cpu().numpy()
