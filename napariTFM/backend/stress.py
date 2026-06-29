"""Engine-agnostic stress result shared by the two stress backends.

napariTFM infers the monolayer's internal stress from the traction field with
one of two interchangeable engines, selected per experiment by
``params.stress_method``:

* **MSM** (:mod:`napariTFM.backend.msm`) — the FEM-based Monolayer Stress
  Microscopy of the published pipeline. Needs a Young's modulus / Poisson ratio
  and a triangular mesh; carries mesh nodes/elements and solver diagnostics.
* **BISM** (:mod:`napariTFM.backend.bism`) — Bayesian Inversion Stress
  Microscopy. Needs no material parameters and no mesh (a single sparse solve on
  the regular grid); carries a traction-reconstruction R² and, optionally,
  per-pixel posterior uncertainty.

Both produce a :class:`StressResult` so every downstream consumer — the viewer
stream, the ``.ntfm`` writer, the tidy-table export, the stage-status dots — is
engine-agnostic and only reads the common fields (``stress_tensor`` +
``physical_scale``). The engine-specific fields are optional and ``None`` for the
backend that has no analog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class StressResult:
    """Inferred internal stress field, from either the MSM or BISM engine.

    Common fields (always populated, read by every downstream consumer):
        stress_tensor: ``(nt, ny, nx, 2, 2)`` Cauchy stress (mN/m). Components are
            ``[..., 0, 0] = sigma_xx``, ``[..., 1, 1] = sigma_yy`` and
            ``[..., 0, 1] = [..., 1, 0] = sigma_xy`` (symmetric).
        parameters: the :class:`MSMParameters` the stage ran with.
        physical_scale: pixel size / grid spacing / units dict.
        original_shape / stress_shape: ``(ny, nx)`` of the force grid.
        method: ``"MSM"`` or ``"BISM"`` — which engine produced this result.

    MSM (FEM) diagnostics — ``None`` for a BISM result:
        nodes / elements: per-frame triangular mesh (drives the mesh overlay).
        condition_number / residual: FEM solver stability metrics.

    BISM diagnostics — ``None`` for an MSM result:
        r2_traction: traction-reconstruction R² (≈1 for a good solve).
    """

    stress_tensor: np.ndarray
    parameters: object
    physical_scale: dict
    original_shape: tuple
    stress_shape: tuple
    method: str = "MSM"

    # MSM (FEM) diagnostics
    nodes: Optional[List[np.ndarray]] = None
    elements: Optional[List[np.ndarray]] = None
    condition_number: Optional[float] = None
    residual: Optional[float] = None

    # BISM diagnostics
    r2_traction: Optional[float] = None
