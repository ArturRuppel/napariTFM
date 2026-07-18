"""Iterative Lucas-Kanade (iLK) dense optical-flow displacement backend.

Dense local-least-squares flow: at every pixel a windowed (``radius``) Lucas-Kanade
solve, iterated over a coarse-to-fine warp schedule. The registration-family
counterpart to PIV's correlation windows: ``ilk_radius`` plays the role PIV's
window plays, so it is the primary knob.

The default **CPU** path is `scikit-image
<https://scikit-image.org>`_'s ``optical_flow_ilk``. When ``torch`` is installed
(the ``[gpu]`` extra) and a CUDA device is available/selected, a **GPU** port runs
the identical algorithm ~25x faster; unlike PIV, the two are numerically
equivalent (matched to a few % relL2, no argmax step to flip), so ``disp_device``
is invisible to the result here. Both read the same two knobs, ``ilk_radius`` and
``ilk_num_warp``; the Gaussian pyramid that gives iLK its capture range is
scikit-image's fixed internal one (reproduced by the port), not a user knob.

``calculate_flow`` returns ``(H, W, 2) float32`` in pixels, ``[..., 0] = u_x``
(cols), ``[..., 1] = u_y`` (rows); positive = right/down.
"""
from typing import Optional

import numpy as np

from napariTFM.backend._displacement_base import BaseDisplacementAnalyzer, resolve_gpu_device
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.piv_displacement import _norm01


def _ilk_cpu(ref, dfm, radius=7, num_warp=10):
    """iLK flow via scikit-image (the trusted CPU reference). Returns u_px (2,H,W)
    [0]=x/col [1]=y/row. skimage returns (v, u) = (row/y, col/x); we reorder to [x, y]."""
    from skimage.registration import optical_flow_ilk

    v, u = optical_flow_ilk(_norm01(ref), _norm01(dfm),
                            radius=max(1, int(radius)), num_warp=max(1, int(num_warp)),
                            gaussian=False, prefilter=False)
    return np.stack([u, v]).astype(np.float64)


class ILKDisplacementAnalyzer(BaseDisplacementAnalyzer):
    """Estimate displacement by iterative Lucas-Kanade optical flow.

    scikit-image (CPU) by default; transparently GPU-accelerated via the torch port
    when torch and CUDA are available and ``disp_device`` allows it. Both backends
    read ``ilk_radius``/``ilk_num_warp`` identically.
    """

    algorithm_name = "Lucas-Kanade"

    def __init__(self, params: Optional[DisplacementParameters] = None):
        super().__init__(params)
        self._device = resolve_gpu_device(str(self.params.disp_device), method="Lucas-Kanade")
        self._backend = "torch" if self._device is not None else "skimage"

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        radius = max(1, int(self.params.ilk_radius))
        num_warp = max(1, int(self.params.ilk_num_warp))

        if self._backend == "torch":
            from napariTFM.backend._ilk_torch import ilk_gpu
            u = ilk_gpu(reference, moving, radius=radius, num_warp=num_warp, device=self._device)
        else:
            u = _ilk_cpu(reference, moving, radius=radius, num_warp=num_warp)

        H, W = np.asarray(reference).shape
        return self._pack(u, H, W)
