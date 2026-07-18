"""Shared plumbing for the displacement backends (PIV, iLK, FFD).

Each backend is a thin analyzer with the same two-method interface the pipeline
expects: ``calculate_flow(reference, moving) -> (H, W, 2) float32`` in **pixels**
(``[..., 0] = u_x`` columns, ``[..., 1] = u_y`` rows; positive = right/down), and
``downscale_flow(flow, factor)``. The device story is uniform: a trusted CPU
reference implementation by default (openpiv for PIV, scikit-image for iLK), with
a numerically-equivalent torch GPU port used when it is both installed (the
``[gpu]`` extra) and selected. FFD is GPU-only.
"""
from __future__ import annotations

import warnings

import numpy as np

from napariTFM.backend.parameter_dataclasses import DisplacementParameters


def resolve_gpu_device(request: str, *, method: str):
    """Resolve the shared device request to a torch CUDA device, or ``None`` for CPU.

    ``request`` is ``"auto"`` (GPU if torch+CUDA are present, else the CPU
    reference), ``"cuda"`` (require a CUDA device), or ``"cpu"`` (force the CPU
    reference). Returns a ``torch.device`` when the GPU port should run, or
    ``None`` when the CPU reference should. ``method`` only sharpens the error
    messages. Never imports torch on the ``"cpu"`` path, so a torch-free install
    stays torch-free by default.
    """
    request = (request or "auto").lower()
    if request == "cpu":
        return None
    if request == "cuda":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                f"device='cuda' needs PyTorch, which is not installed. Install the "
                f"optional GPU extra with `pip install napariTFM[gpu]`, or set "
                f"device='auto'/'cpu' to use the CPU {method} implementation."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' but no CUDA device is available; use device='auto' or 'cpu'."
            )
        return torch.device("cuda")
    # "auto": GPU when torch + CUDA are both present, else the CPU reference.
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except ImportError:
        pass
    return None


class BaseDisplacementAnalyzer:
    """Common interface + shared field post-processing for the displacement backends."""

    algorithm_name = "displacement"

    def __init__(self, params: DisplacementParameters | None = None):
        self.params = params or DisplacementParameters()

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray,
                       weight: np.ndarray | None = None) -> np.ndarray:
        """Estimate the dense displacement field for one frame pair.

        ``weight`` (optional, ``(H, W)`` in ``[0, 1]``) confines the fit to a
        foreground region: only FFD honours it (it masks its loss); PIV and iLK,
        being local estimators, accept and ignore it (their confinement is the
        upstream crop). ``None`` = fit the whole input, the default for all methods.
        """
        raise NotImplementedError

    @staticmethod
    def _pack(u_xy: np.ndarray, H: int, W: int) -> np.ndarray:
        """Pack a ``(2, H, W)`` ``[u_x, u_y]`` field into the pipeline's ``(H, W, 2)``
        float32 layout, zeroing a non-finite (degenerate/blank input) field with a warning."""
        flow = np.stack([u_xy[0], u_xy[1]], axis=-1).astype(np.float32, copy=False)
        if not np.isfinite(flow).all():
            warnings.warn(
                "Displacement backend produced a non-finite field for one frame pair "
                "(degenerate/blank input?); returning zeros for it.",
                RuntimeWarning,
            )
            flow = np.zeros((H, W, 2), dtype=np.float32)
        return flow

    def downscale_flow(self, flow: np.ndarray, factor: int) -> np.ndarray:
        """Downscale the dense flow by block-mean averaging over ``factor x factor`` tiles.

        Preserves vector magnitudes (mean, not decimation); ``factor <= 1`` returns the
        input unchanged. Remainder rows/cols beyond the last whole tile are dropped.
        """
        if factor <= 1:
            return flow
        h, w = flow.shape[:2]
        new_h, new_w = h // factor, w // factor
        trimmed = flow[:new_h * factor, :new_w * factor]
        return (
            trimmed.reshape(new_h, factor, new_w, factor, 2)
            .mean(axis=(1, 3))
            .astype(np.float64)
        )
