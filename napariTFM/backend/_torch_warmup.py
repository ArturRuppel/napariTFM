"""Best-effort background warmup of torch, to hide its first-use latency.

The GPU displacement backends (PIV, iLK, FFD) import torch lazily, so the very
first run or preview pays three one-time, shape-independent costs on the worker
thread while the user waits: ``import torch`` (~1-3 s), CUDA context creation
(~1-2 s), and the first conv2d / grid_sample / FFT / autograd-backward, which
loads cuDNN and selects kernels. :func:`warm_up_torch` pays those costs on a
daemon thread at plugin construction instead, while the user is still looking at
an empty project.

It is deliberately fire-and-forget: it touches no Qt/napari objects (so a plain
thread beats napari's ``thread_worker`` here), runs at most once per process, and
swallows every error so a torch-free or CPU-only install just no-ops. Set the
``NAPARITFM_NO_WARMUP`` environment variable to any truthy value to disable it.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_warm_lock = threading.Lock()
_warmed = False


def warm_up_torch() -> None:
    """Kick off a one-shot background torch warmup, unless already started.

    Returns immediately. Safe to call from multiple widget instances: only the
    first call spawns a thread. Honors ``NAPARITFM_NO_WARMUP`` as an opt-out.
    """
    global _warmed
    if os.environ.get("NAPARITFM_NO_WARMUP"):
        return
    with _warm_lock:
        if _warmed:
            return
        _warmed = True
    threading.Thread(target=_warm, name="napariTFM-torch-warmup", daemon=True).start()


def _warm() -> None:
    """Import torch and run throwaway ops so the one-time costs are paid here.

    Runs on the background thread. Any failure (no torch, no CUDA, driver
    hiccup) is logged at debug and swallowed -- warmup is purely an optimization.
    """
    try:
        import torch
        import torch.nn.functional as F

        # Mirror resolve_gpu_device's "auto" branch: CUDA when present, else CPU.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Tiny tensors: the costs we're hiding (import, context, cuDNN load) are
        # shape-independent, so a small shape warms the same machinery as a large
        # one. Touch each primitive the real backends use.
        img = torch.zeros((1, 1, 32, 32), device=device)
        weight = torch.ones((1, 1, 3, 3), device=device) / 9.0

        # conv2d -- shared blur/box filters (iLK, FFD LNCC).
        F.conv2d(img, weight, padding=1)

        # grid_sample -- every backend warps through _flow_common._warp.
        grid = torch.zeros((1, 32, 32, 2), device=device)
        F.grid_sample(img, grid, align_corners=False)

        # FFT round-trip -- PIV cross-correlation.
        torch.fft.irfft2(torch.fft.rfft2(img), s=(32, 32))

        # A tiny backward pass -- warms the autograd/cuDNN backward kernels that
        # FFD's LBFGS optimization loop needs.
        x = img.clone().requires_grad_(True)
        F.conv2d(x, weight, padding=1).sum().backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        logger.debug("torch warmup complete on %s", device)
    except Exception as exc:  # noqa: BLE001 -- warmup must never raise
        logger.debug("torch warmup skipped: %s", exc)
