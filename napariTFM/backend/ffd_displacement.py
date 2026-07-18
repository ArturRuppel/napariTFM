"""Free-form deformation (FFD) displacement backend -- GPU-only.

napariTFM's own registration method: a cubic B-spline control grid whose spacing is
constant in *level* pixels over a coarse-to-fine image pyramid, fit by LBFGS to a
local-normalised-cross-correlation objective. It is the strongest option under
large deformation in our benchmarks (its image pyramid gives capture range while
its coarse control grid keeps off-cell noise low).

The primary knob is ``ffd_level_spacing``, the finest control spacing: it is the
bias-variance dial (fine ~8 px recovers sharp peaks on clean data, coarse ~24 px
is the noise regularizer). The pyramid depth (capture range) is derived from
``ffd_downscale`` and ``ffd_min_size`` rather than a separate count; ``ffd_metric``
chooses the image-match objective (``lncc`` preserves the peak better than ``mse``).

FFD has **no CPU implementation** by design: the control-grid optimisation is
impractical without a GPU. It requires the ``[gpu]`` extra (torch) and a CUDA
device; without them the analyzer refuses to construct with a clear message, and
the UI greys the option out.

``calculate_flow`` returns ``(H, W, 2) float32`` in pixels, ``[..., 0] = u_x``
(cols), ``[..., 1] = u_y`` (rows); positive = right/down.
"""
from typing import Optional

import numpy as np

from napariTFM.backend._displacement_base import BaseDisplacementAnalyzer, resolve_gpu_device
from napariTFM.backend.parameter_dataclasses import DisplacementParameters

_UNAVAILABLE_MSG = (
    "FFD is GPU-only and needs the optional GPU extra (a CUDA device + PyTorch). "
    "Install it with `pip install napariTFM[gpu]` and select a CUDA device, or "
    "choose the PIV or Lucas-Kanade method, which run on CPU by default."
)


def ffd_available() -> bool:
    """True when FFD can run here: torch importable and a CUDA device present.

    Used by the UI to grey out the FFD method when it cannot run. Never raises.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class FFDDisplacementAnalyzer(BaseDisplacementAnalyzer):
    """Estimate displacement by grid-pyramid free-form deformation (GPU-only)."""

    algorithm_name = "FFD"

    def __init__(self, params: Optional[DisplacementParameters] = None):
        super().__init__(params)
        # "cpu" (or "auto" with no CUDA device) has no FFD backend to fall back to.
        self._device = resolve_gpu_device(str(self.params.disp_device), method="FFD")
        if self._device is None:
            raise RuntimeError(_UNAVAILABLE_MSG)

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray,
                       weight: Optional[np.ndarray] = None) -> np.ndarray:
        from napariTFM.backend._ffd_torch import ffd_pyr

        H, W = np.asarray(reference).shape
        u = ffd_pyr(
            reference, moving,
            level_spacing=float(self.params.ffd_level_spacing),
            num_iters=max(1, int(self.params.ffd_num_iters)),
            downscale=float(self.params.ffd_downscale),
            min_size=max(1, int(self.params.ffd_min_size)),
            metric=str(self.params.ffd_metric),
            elastic=float(self.params.ffd_elastic),
            interp=str(self.params.ffd_interp),
            device=self._device,
            early_stop=float(self.params.ffd_early_stop),
            weight=weight,
        )
        return self._pack(u, H, W)
