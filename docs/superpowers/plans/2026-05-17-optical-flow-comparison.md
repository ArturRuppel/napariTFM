# Optical Flow Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark harness in `_dev/optical_flow_comparison/` that runs DIS, Farneback, and Lucas-Kanade on the three TFM benchmark scenarios and writes a comparison report.

**Architecture:** A single CLI runner orchestrates `(scenario × algorithm)` pairs. Each algorithm is a thin adapter behind a uniform protocol — `(ref, deformed, query_points) → (displacements, valid_mask)`. Shared modules own preprocessing, bead detection, metrics, and reporting. Output is a tidy CSV plus per-algorithm and combined PNGs.

**Tech Stack:** Python, OpenCV (DIS / Farneback / LK), TrackPy, NumPy, pandas, matplotlib, scipy. All present in the `napariTFMv2` conda environment.

**Spec:** `docs/superpowers/specs/2026-05-17-optical-flow-comparison-design.md`

**Conventions:**
- All paths absolute from repo root `/home/aruppel/Projects/napariTFM/`.
- All commands run from repo root unless stated otherwise.
- Tests live alongside the code in `_dev/optical_flow_comparison/tests/` and are run with `conda run -n napariTFMv2 python -m pytest <path> -q`.
- Use `conda run -n napariTFMv2` for every Python invocation — the project's pytest baseline confirms this is the working environment.
- Each task ends with a single commit.

---

## File Structure (created by this plan)

```
_dev/optical_flow_comparison/
  __init__.py
  runner.py                       # Task 8
  detection.py                    # Task 3
  preprocessing.py                # Task 2
  metrics.py                      # Task 6
  reporting.py                    # Task 7
  adapters/
    __init__.py
    base.py                       # Task 1 (FlowAdapter Protocol + sample_dense_at_points)
    dis.py                        # Task 4
    farneback.py                  # Task 4
    lucas_kanade.py               # Task 5
  tests/
    __init__.py
    test_base.py                  # Task 1
    test_preprocessing.py         # Task 2
    test_detection.py             # Task 3
    test_dis_adapter.py           # Task 4
    test_farneback_adapter.py     # Task 4
    test_lk_adapter.py            # Task 5
    test_metrics.py               # Task 6
    test_reporting.py             # Task 7
    test_runner_smoke.py          # Task 8
  output/                         # Generated at runtime; not committed
```

Add `_dev/optical_flow_comparison/output/` to `.gitignore` in Task 0.

---

## Task 0: Scaffold the package and gitignore output

**Files:**
- Create: `_dev/optical_flow_comparison/__init__.py`
- Create: `_dev/optical_flow_comparison/adapters/__init__.py`
- Create: `_dev/optical_flow_comparison/tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create package directories with empty `__init__.py` files**

```bash
mkdir -p _dev/optical_flow_comparison/adapters _dev/optical_flow_comparison/tests
: > _dev/optical_flow_comparison/__init__.py
: > _dev/optical_flow_comparison/adapters/__init__.py
: > _dev/optical_flow_comparison/tests/__init__.py
```

- [ ] **Step 2: Add the output directory to `.gitignore`**

Read the current `.gitignore` first. If `_dev/optical_flow_comparison/output/` is not already covered, append:

```
_dev/optical_flow_comparison/output/
```

- [ ] **Step 3: Verify the package imports**

Run: `conda run -n napariTFMv2 python -c "import _dev.optical_flow_comparison; import _dev.optical_flow_comparison.adapters"`
Expected: no output, exit code 0.

If the import fails because `_dev/` is not on sys.path when invoked this way, that's expected — the runner will be invoked as `python _dev/optical_flow_comparison/runner.py`, not as a module import. Skip this verification and rely on Task 8's smoke test instead.

- [ ] **Step 4: Commit**

```bash
git add _dev/optical_flow_comparison .gitignore
git commit -m "Scaffold optical flow comparison package"
```

---

## Task 1: Adapter base protocol + dense-field sampler

**Files:**
- Create: `_dev/optical_flow_comparison/adapters/base.py`
- Test: `_dev/optical_flow_comparison/tests/test_base.py`

The base module defines the protocol all adapters implement and provides one shared utility: bilinear sampling of a dense `(H, W, 2)` flow field at `(N, 2)` query points. DIS and Farneback adapters both need this.

- [ ] **Step 1: Write the failing test**

Create `_dev/optical_flow_comparison/tests/test_base.py`:

```python
import numpy as np
import pytest

from _dev.optical_flow_comparison.adapters.base import sample_dense_at_points


def test_sample_dense_at_points_returns_exact_values_at_integer_coords():
    # Flow field where dx = column index, dy = row index. Sampling at integer
    # points must return those indices exactly.
    h, w = 10, 12
    flow = np.zeros((h, w, 2), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    flow[..., 0] = xx
    flow[..., 1] = yy

    pts = np.array([[0.0, 0.0], [5.0, 3.0], [11.0, 9.0]], dtype=np.float32)
    out = sample_dense_at_points(flow, pts)

    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], [0.0, 5.0, 11.0], atol=1e-5)
    np.testing.assert_allclose(out[:, 1], [0.0, 3.0, 9.0], atol=1e-5)


def test_sample_dense_at_points_interpolates_between_pixels():
    h, w = 4, 4
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = np.arange(w)[None, :]
    flow[..., 1] = 0.0

    pts = np.array([[1.5, 2.0]], dtype=np.float32)  # x=1.5 → dx should be 1.5
    out = sample_dense_at_points(flow, pts)

    np.testing.assert_allclose(out[0, 0], 1.5, atol=1e-5)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-5)


def test_sample_dense_at_points_clamps_out_of_bounds_points():
    h, w = 4, 4
    flow = np.ones((h, w, 2), dtype=np.float32)
    pts = np.array([[-1.0, -1.0], [100.0, 100.0]], dtype=np.float32)
    out = sample_dense_at_points(flow, pts)

    # Clamped to the edge; edge value of the constant field is 1.0.
    np.testing.assert_allclose(out, np.ones((2, 2)), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_base.py -q`
Expected: collection error or `ModuleNotFoundError: No module named '_dev.optical_flow_comparison.adapters.base'`.

- [ ] **Step 3: Implement `base.py`**

Create `_dev/optical_flow_comparison/adapters/base.py`:

```python
from typing import Protocol

import cv2
import numpy as np


class FlowAdapter(Protocol):
    """Uniform interface every algorithm wrapper exposes.

    Implementations are responsible for any algorithm-specific preprocessing
    (e.g., float→uint8 conversion). The runner passes float32 images in
    [0, 1] and the same set of query points to every adapter, and treats the
    return value as opaque.
    """

    name: str

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict displacements at the given query points.

        Args:
            reference: float32, shape (H, W), range [0, 1].
            deformed:  float32, shape (H, W), range [0, 1].
            query_points: float32, shape (N, 2), columns (x, y) in pixels.

        Returns:
            displacements: float32, shape (N, 2), columns (dx, dy) in pixels.
            valid_mask:    bool,    shape (N,). True where the prediction is
                           trustworthy. Dense methods return all-True.
        """
        ...


def sample_dense_at_points(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear sample a dense flow field at scattered points.

    Args:
        flow:   shape (H, W, 2), dtype convertible to float32.
        points: shape (N, 2), columns (x, y).

    Returns:
        shape (N, 2), columns (dx, dy). Points outside the image are clamped
        to the nearest edge value.
    """
    h, w = flow.shape[:2]
    fx = np.clip(points[:, 0].astype(np.float32), 0.0, w - 1.0)
    fy = np.clip(points[:, 1].astype(np.float32), 0.0, h - 1.0)

    dx = cv2.remap(
        flow[..., 0].astype(np.float32),
        fx.reshape(-1, 1),
        fy.reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()
    dy = cv2.remap(
        flow[..., 1].astype(np.float32),
        fx.reshape(-1, 1),
        fy.reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()

    return np.column_stack([dx, dy]).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_base.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add _dev/optical_flow_comparison/adapters/base.py _dev/optical_flow_comparison/tests/test_base.py
git commit -m "Add flow adapter protocol and dense-field sampler"
```

---

## Task 2: Preprocessing wrapper

**Files:**
- Create: `_dev/optical_flow_comparison/preprocessing.py`
- Test: `_dev/optical_flow_comparison/tests/test_preprocessing.py`

A thin wrapper over `napariTFM.backend.preprocessing.ImageProcessor` with the canonical parameters baked in (`percentile=[80, 99.9]`, `gaussian_sigma=1`). Centralizing this keeps every algorithm seeing the same input.

- [ ] **Step 1: Write the failing test**

Create `_dev/optical_flow_comparison/tests/test_preprocessing.py`:

```python
import numpy as np

from _dev.optical_flow_comparison.preprocessing import preprocess


def test_preprocess_returns_float32_in_unit_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4096, size=(64, 64), dtype=np.uint16).astype(np.float32)

    out = preprocess(img)

    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_preprocess_is_deterministic_for_same_input():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 4096, size=(32, 32), dtype=np.uint16).astype(np.float32)

    a = preprocess(img)
    b = preprocess(img)

    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_preprocessing.py -q`
Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `preprocessing.py`**

Create `_dev/optical_flow_comparison/preprocessing.py`:

```python
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from napariTFM.backend.preprocessing import ImageProcessor

MIN_PERCENTILE = 80.0
MAX_PERCENTILE = 99.9
GAUSSIAN_SIGMA = 1.0

_processor = ImageProcessor()


def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply the project-standard TFM preprocessing pipeline.

    Steps: percentile intensity scaling [80, 99.9], then Gaussian filter
    (sigma=1). Matches `_validation/benchmark_TFM/validate_TFM.py`.

    Returns: float32 array in [0, 1], same shape as input.
    """
    scaled, _ = _processor.apply_intensity_scaling(
        image.astype(np.float32, copy=False), MIN_PERCENTILE, MAX_PERCENTILE
    )
    filtered = _processor.apply_gaussian_filter(scaled, sigma=GAUSSIAN_SIGMA)
    return filtered.astype(np.float32, copy=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_preprocessing.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add _dev/optical_flow_comparison/preprocessing.py _dev/optical_flow_comparison/tests/test_preprocessing.py
git commit -m "Add shared preprocessing wrapper"
```

---

## Task 3: Bead detection helper

**Files:**
- Create: `_dev/optical_flow_comparison/detection.py`
- Test: `_dev/optical_flow_comparison/tests/test_detection.py`

Wraps TrackPy with the parameters locked in the spec: `diameter=7, separation=8, minmass="auto"`. The auto-minmass estimator follows `_dev/spt_piv/spt_piv_displacement.py`: locate with `minmass=0`, then return the 30th-percentile mass.

- [ ] **Step 1: Write the failing test**

Create `_dev/optical_flow_comparison/tests/test_detection.py`:

```python
import numpy as np
import pytest

from _dev.optical_flow_comparison.detection import detect_beads


def _synthetic_beads(shape=(128, 128), positions=((20, 30), (70, 80), (100, 50))):
    """Float32 [0, 1] image with Gaussian bead-like peaks at the given (x, y)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w), dtype=np.float32)
    sigma = 1.5
    for (x, y) in positions:
        img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    img = img / img.max()
    return img


def test_detect_beads_finds_all_synthetic_peaks():
    positions = [(20, 30), (70, 80), (100, 50)]
    img = _synthetic_beads(positions=positions)

    points = detect_beads(img)

    assert points.shape[1] == 2
    assert len(points) >= len(positions)
    # Every synthetic peak should have a detected point within 1 px.
    for (x, y) in positions:
        d = np.hypot(points[:, 0] - x, points[:, 1] - y)
        assert d.min() < 1.0, f"no detection near ({x}, {y}); nearest = {d.min()} px"


def test_detect_beads_returns_float32_xy_columns():
    img = _synthetic_beads()
    points = detect_beads(img)
    assert points.dtype == np.float32
    assert points.shape[1] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_detection.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `detection.py`**

Create `_dev/optical_flow_comparison/detection.py`:

```python
import numpy as np
import trackpy as tp

DIAMETER = 7
SEPARATION = 8
AUTO_MINMASS_PERCENTILE = 30.0

tp.quiet()


def _to_uint16(image: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] image to uint16 for TrackPy."""
    return np.clip(image * 65535.0, 0, 65535).astype(np.uint16)


def _auto_minmass(image_u16: np.ndarray) -> float:
    """Estimate minmass: locate everything, take the 30th-percentile mass.

    Mirrors `_dev/spt_piv/spt_piv_displacement.py`.
    """
    feats = tp.locate(image_u16, diameter=DIAMETER, minmass=0, separation=SEPARATION)
    if feats.empty:
        return 0.0
    return float(np.percentile(feats["mass"].values, AUTO_MINMASS_PERCENTILE))


def detect_beads(image: np.ndarray) -> np.ndarray:
    """Detect beads using TrackPy with the project-standard parameters.

    Args:
        image: float32, shape (H, W), range [0, 1].

    Returns:
        float32 array of shape (N, 2), columns (x, y) in pixels.
    """
    image_u16 = _to_uint16(image)
    minmass = _auto_minmass(image_u16)
    feats = tp.locate(image_u16, diameter=DIAMETER, minmass=minmass, separation=SEPARATION)
    if feats.empty:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([feats["x"].values, feats["y"].values]).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_detection.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add _dev/optical_flow_comparison/detection.py _dev/optical_flow_comparison/tests/test_detection.py
git commit -m "Add TrackPy bead detection helper"
```

---

## Task 4: DIS and Farneback adapters

Both dense adapters share the same shape: compute a full flow field, then sample at the query points. Implementing them together keeps the symmetry visible.

**Files:**
- Create: `_dev/optical_flow_comparison/adapters/dis.py`
- Create: `_dev/optical_flow_comparison/adapters/farneback.py`
- Test: `_dev/optical_flow_comparison/tests/test_dis_adapter.py`
- Test: `_dev/optical_flow_comparison/tests/test_farneback_adapter.py`

- [ ] **Step 1: Write the failing tests**

Create `_dev/optical_flow_comparison/tests/test_dis_adapter.py`:

```python
import numpy as np

from _dev.optical_flow_comparison.adapters.dis import DISAdapter


def _shifted_pair(shape=(96, 96), shift=(3.0, 2.0), n_beads=20, seed=0):
    """Synthetic ref/deformed pair: random Gaussian beads, shifted by `shift`.

    Returns (ref, deformed, bead_positions_in_reference, shift).
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    margin = 10
    positions = rng.uniform(margin, min(h, w) - margin, size=(n_beads, 2)).astype(np.float32)

    def render(centers):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w), dtype=np.float32)
        sigma = 1.5
        for (x, y) in centers:
            img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        return img / max(img.max(), 1e-6)

    ref = render(positions)
    deformed = render(positions + np.array(shift, dtype=np.float32))
    return ref, deformed, positions, np.array(shift, dtype=np.float32)


def test_dis_adapter_has_name():
    assert DISAdapter().name == "DIS"


def test_dis_adapter_recovers_uniform_shift_within_tolerance():
    ref, deformed, beads, shift = _shifted_pair()
    adapter = DISAdapter()

    displacements, valid = adapter.displacements_at(ref, deformed, beads)

    assert displacements.shape == (len(beads), 2)
    assert valid.shape == (len(beads),)
    assert valid.all()
    mean = displacements.mean(axis=0)
    # DIS is approximate; allow 1 px tolerance per axis on this synthetic case.
    np.testing.assert_allclose(mean, shift, atol=1.0)
```

Create `_dev/optical_flow_comparison/tests/test_farneback_adapter.py` with the same `_shifted_pair` helper (copy it; the engineer may read tasks out of order) and:

```python
import numpy as np

from _dev.optical_flow_comparison.adapters.farneback import FarnebackAdapter


def _shifted_pair(shape=(96, 96), shift=(3.0, 2.0), n_beads=20, seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape
    margin = 10
    positions = rng.uniform(margin, min(h, w) - margin, size=(n_beads, 2)).astype(np.float32)

    def render(centers):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w), dtype=np.float32)
        sigma = 1.5
        for (x, y) in centers:
            img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        return img / max(img.max(), 1e-6)

    ref = render(positions)
    deformed = render(positions + np.array(shift, dtype=np.float32))
    return ref, deformed, positions, np.array(shift, dtype=np.float32)


def test_farneback_adapter_has_name():
    assert FarnebackAdapter().name == "Farneback"


def test_farneback_adapter_recovers_uniform_shift_within_tolerance():
    ref, deformed, beads, shift = _shifted_pair()
    adapter = FarnebackAdapter()

    displacements, valid = adapter.displacements_at(ref, deformed, beads)

    assert displacements.shape == (len(beads), 2)
    assert valid.all()
    mean = displacements.mean(axis=0)
    np.testing.assert_allclose(mean, shift, atol=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_dis_adapter.py _dev/optical_flow_comparison/tests/test_farneback_adapter.py -q`
Expected: `ModuleNotFoundError` for both.

- [ ] **Step 3: Implement the DIS adapter**

Create `_dev/optical_flow_comparison/adapters/dis.py`:

```python
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from napariTFM.backend.displacement_analysis import DisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters

from _dev.optical_flow_comparison.adapters.base import sample_dense_at_points


class DISAdapter:
    """OpenCV DIS optical flow, sampled at query points."""

    name = "DIS"

    def __init__(self) -> None:
        self._analyzer = DisplacementAnalyzer(DisplacementParameters())

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        flow = self._analyzer.calculate_flow(reference, deformed)
        displacements = sample_dense_at_points(flow, query_points)
        valid = np.ones(len(query_points), dtype=bool)
        return displacements, valid
```

- [ ] **Step 4: Implement the Farneback adapter**

Create `_dev/optical_flow_comparison/adapters/farneback.py`:

```python
import cv2
import numpy as np

from _dev.optical_flow_comparison.adapters.base import sample_dense_at_points


def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32, copy=False)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo <= 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - lo) / (hi - lo) * 255.0
    return np.ascontiguousarray(scaled.astype(np.uint8))


class FarnebackAdapter:
    """OpenCV Farneback dense optical flow, sampled at query points."""

    name = "Farneback"

    PYR_SCALE = 0.5
    LEVELS = 3
    WIN_SIZE = 15
    ITERATIONS = 3
    POLY_N = 5
    POLY_SIGMA = 1.2
    FLAGS = 0

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_u8 = _to_uint8(reference)
        def_u8 = _to_uint8(deformed)
        flow = cv2.calcOpticalFlowFarneback(
            ref_u8, def_u8, None,
            self.PYR_SCALE, self.LEVELS, self.WIN_SIZE,
            self.ITERATIONS, self.POLY_N, self.POLY_SIGMA, self.FLAGS,
        ).astype(np.float32, copy=False)
        displacements = sample_dense_at_points(flow, query_points)
        valid = np.ones(len(query_points), dtype=bool)
        return displacements, valid
```

- [ ] **Step 5: Run the adapter tests**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_dis_adapter.py _dev/optical_flow_comparison/tests/test_farneback_adapter.py -q`
Expected: `4 passed`.

If a test fails because the synthetic-shift tolerance is too tight, do NOT loosen it silently — debug the adapter first. The synthetic case is generous; if the mean recovered shift is off by more than 1 px on a uniform 3-px shift, something is wrong with the adapter wiring (e.g., x/y swapped, wrong sign).

- [ ] **Step 6: Commit**

```bash
git add _dev/optical_flow_comparison/adapters/dis.py _dev/optical_flow_comparison/adapters/farneback.py \
        _dev/optical_flow_comparison/tests/test_dis_adapter.py _dev/optical_flow_comparison/tests/test_farneback_adapter.py
git commit -m "Add DIS and Farneback flow adapters"
```

---

## Task 5: Lucas-Kanade adapter

Sparse OF — feed the query points in, get back tracked positions and a per-point status flag.

**Files:**
- Create: `_dev/optical_flow_comparison/adapters/lucas_kanade.py`
- Test: `_dev/optical_flow_comparison/tests/test_lk_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `_dev/optical_flow_comparison/tests/test_lk_adapter.py`:

```python
import numpy as np

from _dev.optical_flow_comparison.adapters.lucas_kanade import LucasKanadeAdapter


def _shifted_pair(shape=(96, 96), shift=(3.0, 2.0), n_beads=20, seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape
    margin = 10
    positions = rng.uniform(margin, min(h, w) - margin, size=(n_beads, 2)).astype(np.float32)

    def render(centers):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w), dtype=np.float32)
        sigma = 1.5
        for (x, y) in centers:
            img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        return img / max(img.max(), 1e-6)

    ref = render(positions)
    deformed = render(positions + np.array(shift, dtype=np.float32))
    return ref, deformed, positions, np.array(shift, dtype=np.float32)


def test_lk_adapter_has_name():
    assert LucasKanadeAdapter().name == "Lucas-Kanade"


def test_lk_adapter_recovers_uniform_shift_with_high_accuracy():
    ref, deformed, beads, shift = _shifted_pair()
    adapter = LucasKanadeAdapter()

    displacements, valid = adapter.displacements_at(ref, deformed, beads)

    assert displacements.shape == (len(beads), 2)
    assert valid.shape == (len(beads),)
    assert valid.sum() >= len(beads) - 2  # allow up to 2 dropouts on edges
    # LK should be sub-pixel accurate on this clean synthetic case.
    np.testing.assert_allclose(displacements[valid].mean(axis=0), shift, atol=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_lk_adapter.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the LK adapter**

Create `_dev/optical_flow_comparison/adapters/lucas_kanade.py`:

```python
import cv2
import numpy as np


def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32, copy=False)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo <= 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - lo) / (hi - lo) * 255.0
    return np.ascontiguousarray(scaled.astype(np.uint8))


class LucasKanadeAdapter:
    """OpenCV pyramidal Lucas-Kanade sparse optical flow."""

    name = "Lucas-Kanade"

    WIN_SIZE = (15, 15)
    MAX_LEVEL = 3
    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_u8 = _to_uint8(reference)
        def_u8 = _to_uint8(deformed)
        # OpenCV expects shape (N, 1, 2) and float32 for point arrays.
        pts = query_points.astype(np.float32).reshape(-1, 1, 2)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            ref_u8, def_u8, pts, None,
            winSize=self.WIN_SIZE,
            maxLevel=self.MAX_LEVEL,
            criteria=self.CRITERIA,
        )
        new_pts = new_pts.reshape(-1, 2)
        displacements = (new_pts - query_points.astype(np.float32)).astype(np.float32)
        valid = status.reshape(-1).astype(bool)
        # Zero-out displacements where LK failed so downstream consumers
        # cannot accidentally read stale numbers; the valid mask is the truth.
        displacements[~valid] = 0.0
        return displacements, valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_lk_adapter.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add _dev/optical_flow_comparison/adapters/lucas_kanade.py _dev/optical_flow_comparison/tests/test_lk_adapter.py
git commit -m "Add Lucas-Kanade sparse flow adapter"
```

---

## Task 6: Metrics

Compute RMSE / median / p95 / coverage / magnitude-binned signed bias from `(predicted, gt, valid)`.

**Files:**
- Create: `_dev/optical_flow_comparison/metrics.py`
- Test: `_dev/optical_flow_comparison/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `_dev/optical_flow_comparison/tests/test_metrics.py`:

```python
import numpy as np

from _dev.optical_flow_comparison.metrics import compute_metrics, MAGNITUDE_BINS


def test_compute_metrics_zero_error_case():
    gt = np.array([[1.0, 0.5], [2.0, 0.0], [0.1, 0.1]], dtype=np.float32)
    pred = gt.copy()
    valid = np.array([True, True, True])

    m = compute_metrics(pred, gt, valid)

    assert m["n_beads"] == 3
    assert m["coverage"] == 1.0
    assert m["rmse_px"] == 0.0
    assert m["median_px"] == 0.0
    assert m["p95_px"] == 0.0


def test_compute_metrics_ignores_invalid_beads():
    gt = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    pred = np.array([[1.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    valid = np.array([True, False, False])

    m = compute_metrics(pred, gt, valid)

    assert m["n_beads"] == 3
    assert m["coverage"] == 1 / 3
    assert m["rmse_px"] == 0.0  # only the valid bead is scored, and it is exact


def test_compute_metrics_magnitude_binned_bias_keys_present():
    gt = np.array([[0.2, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    pred = np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]], dtype=np.float32)  # all undershoot
    valid = np.array([True, True, True])

    m = compute_metrics(pred, gt, valid)

    # MAGNITUDE_BINS = [0, 1, 3, inf] → keys bias_low/mid/high
    for key in ("bias_low", "bias_mid", "bias_high"):
        assert key in m
    # The "high" bin (>3 px GT) should have a clearly negative signed error along x.
    assert m["bias_high"] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_metrics.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `metrics.py`**

Create `_dev/optical_flow_comparison/metrics.py`:

```python
from typing import Dict

import numpy as np

# Edges in pixels: [0, 1, 3, inf] → three bins: low, mid, high.
MAGNITUDE_BINS: tuple[float, ...] = (0.0, 1.0, 3.0, float("inf"))
BIN_LABELS: tuple[str, ...] = ("low", "mid", "high")


def compute_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    """Compute per-(scenario, algorithm) summary metrics.

    Errors are evaluated only over beads where `valid` is True.

    Args:
        predicted:    shape (N, 2), columns (dx, dy) in pixels.
        ground_truth: shape (N, 2), columns (dx, dy) in pixels.
        valid:        shape (N,) bool.

    Returns:
        dict with keys: n_beads, coverage, rmse_px, median_px, p95_px,
                        bias_low, bias_mid, bias_high.
        Bias is the mean signed error magnitude (|pred| - |gt|) in pixels
        within each GT-magnitude bin. Negative = undershoot, positive = overshoot.
        Bins with no beads return np.nan.
    """
    n = len(predicted)
    coverage = float(valid.sum()) / n if n > 0 else 0.0

    err = predicted - ground_truth
    err_mag = np.hypot(err[:, 0], err[:, 1])
    valid_mag = err_mag[valid]

    if len(valid_mag) == 0:
        rmse = median = p95 = float("nan")
    else:
        rmse = float(np.sqrt(np.mean(valid_mag ** 2)))
        median = float(np.median(valid_mag))
        p95 = float(np.percentile(valid_mag, 95))

    gt_mag = np.hypot(ground_truth[:, 0], ground_truth[:, 1])
    pred_mag = np.hypot(predicted[:, 0], predicted[:, 1])
    signed = pred_mag - gt_mag

    bin_idx = np.digitize(gt_mag, MAGNITUDE_BINS) - 1  # 0, 1, 2
    bias: dict[str, float] = {}
    for i, label in enumerate(BIN_LABELS):
        in_bin = (bin_idx == i) & valid
        bias[f"bias_{label}"] = float(np.mean(signed[in_bin])) if in_bin.any() else float("nan")

    return {
        "n_beads": n,
        "coverage": coverage,
        "rmse_px": rmse,
        "median_px": median,
        "p95_px": p95,
        **bias,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_metrics.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add _dev/optical_flow_comparison/metrics.py _dev/optical_flow_comparison/tests/test_metrics.py
git commit -m "Add comparison metrics"
```

---

## Task 7: Reporting (plots and CSV writers)

Per-algorithm PNG, per-scenario combined PNG, summary CSV, summary bar chart, tidy per-bead CSV. All matplotlib calls use the `Agg` backend (no display required).

**Files:**
- Create: `_dev/optical_flow_comparison/reporting.py`
- Test: `_dev/optical_flow_comparison/tests/test_reporting.py`

- [ ] **Step 1: Write the failing test**

Create `_dev/optical_flow_comparison/tests/test_reporting.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from _dev.optical_flow_comparison.reporting import (
    write_per_algorithm_png,
    write_combined_scenario_png,
    write_summary_csv,
    write_summary_bar_chart,
    write_tidy_displacements_csv,
)


def _result_rows():
    rng = np.random.default_rng(0)
    beads = rng.uniform(10, 90, size=(15, 2)).astype(np.float32)
    pred = np.full((15, 2), 2.0, dtype=np.float32)
    gt = pred + rng.normal(scale=0.2, size=(15, 2)).astype(np.float32)
    valid = np.ones(15, dtype=bool)
    return beads, pred, gt, valid


def test_write_per_algorithm_png_creates_file(tmp_path: Path):
    beads, pred, gt, valid = _result_rows()
    img = np.zeros((100, 100), dtype=np.float32)
    out = tmp_path / "low" / "DIS.png"

    write_per_algorithm_png(
        out_path=out,
        reference_image=img,
        bead_positions=beads,
        predicted=pred,
        ground_truth=gt,
        valid=valid,
        title="low / DIS",
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_write_combined_scenario_png_handles_multiple_algorithms(tmp_path: Path):
    beads, pred, gt, valid = _result_rows()
    img = np.zeros((100, 100), dtype=np.float32)
    per_algo = {
        "DIS": (pred, gt, valid),
        "Farneback": (pred, gt, valid),
        "Lucas-Kanade": (pred, gt, valid),
    }
    out = tmp_path / "low" / "combined.png"

    write_combined_scenario_png(
        out_path=out,
        reference_image=img,
        bead_positions=beads,
        per_algorithm=per_algo,
        scenario_name="low",
    )

    assert out.exists()


def test_write_summary_csv_round_trips(tmp_path: Path):
    rows = [
        {"scenario": "low", "algorithm": "DIS", "rmse_px": 0.5, "n_beads": 10,
         "coverage": 1.0, "median_px": 0.4, "p95_px": 0.9,
         "bias_low": 0.0, "bias_mid": -0.1, "bias_high": -0.2},
    ]
    out = tmp_path / "summary.csv"

    write_summary_csv(out, rows)

    df = pd.read_csv(out)
    assert list(df.columns) == [
        "scenario", "algorithm", "n_beads", "coverage",
        "rmse_px", "median_px", "p95_px",
        "bias_low", "bias_mid", "bias_high",
    ]
    assert df.iloc[0]["rmse_px"] == 0.5


def test_write_summary_bar_chart_creates_file(tmp_path: Path):
    rows = [
        {"scenario": "low", "algorithm": "DIS", "rmse_px": 0.5},
        {"scenario": "low", "algorithm": "Lucas-Kanade", "rmse_px": 0.3},
        {"scenario": "mid", "algorithm": "DIS", "rmse_px": 0.8},
        {"scenario": "mid", "algorithm": "Lucas-Kanade", "rmse_px": 0.6},
    ]
    out = tmp_path / "summary.png"

    write_summary_bar_chart(out, rows)

    assert out.exists()


def test_write_tidy_displacements_csv_columns(tmp_path: Path):
    beads, pred, gt, valid = _result_rows()
    out = tmp_path / "displacements.csv"

    write_tidy_displacements_csv(
        out,
        scenario="low",
        algorithm="DIS",
        bead_positions=beads,
        predicted=pred,
        ground_truth=gt,
        valid=valid,
        append=False,
    )

    df = pd.read_csv(out)
    expected = [
        "scenario", "algorithm", "bead_id",
        "ref_x", "ref_y",
        "pred_dx", "pred_dy",
        "gt_dx", "gt_dy",
        "err_x", "err_y", "err_mag", "valid",
    ]
    assert list(df.columns) == expected
    assert len(df) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_reporting.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `reporting.py`**

Create `_dev/optical_flow_comparison/reporting.py`:

```python
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_COLUMNS: tuple[str, ...] = (
    "scenario", "algorithm", "n_beads", "coverage",
    "rmse_px", "median_px", "p95_px",
    "bias_low", "bias_mid", "bias_high",
)

TIDY_COLUMNS: tuple[str, ...] = (
    "scenario", "algorithm", "bead_id",
    "ref_x", "ref_y",
    "pred_dx", "pred_dy",
    "gt_dx", "gt_dy",
    "err_x", "err_y", "err_mag", "valid",
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _render_panels(
    ax_quiver,
    ax_err,
    reference_image: np.ndarray,
    bead_positions: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    err_vmax: float | None = None,
) -> None:
    ax_quiver.imshow(reference_image, cmap="gray", origin="upper")
    pts = bead_positions[valid]
    ax_quiver.quiver(
        pts[:, 0], pts[:, 1],
        ground_truth[valid, 0], ground_truth[valid, 1],
        color="lime", angles="xy", scale_units="xy", scale=1,
        width=0.003, label="GT",
    )
    ax_quiver.quiver(
        pts[:, 0], pts[:, 1],
        predicted[valid, 0], predicted[valid, 1],
        color="red", angles="xy", scale_units="xy", scale=1,
        width=0.003, label="pred",
    )
    ax_quiver.axis("off")
    ax_quiver.legend(loc="upper right", fontsize=7)

    err = predicted - ground_truth
    err_mag = np.hypot(err[:, 0], err[:, 1])
    vmax = err_vmax if err_vmax is not None else (float(np.nanpercentile(err_mag[valid], 95)) if valid.any() else 1.0)
    sc = ax_err.scatter(
        bead_positions[valid, 0], bead_positions[valid, 1],
        c=err_mag[valid], cmap="hot_r", s=8, vmin=0, vmax=max(vmax, 1e-6),
    )
    ax_err.set_xlim(0, reference_image.shape[1])
    ax_err.set_ylim(reference_image.shape[0], 0)
    ax_err.set_aspect("equal")
    ax_err.axis("off")
    plt.colorbar(sc, ax=ax_err, fraction=0.046, pad=0.02, label="error (px)")


def write_per_algorithm_png(
    out_path: Path,
    reference_image: np.ndarray,
    bead_positions: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    title: str,
) -> None:
    """Two-panel PNG: quiver overlay + per-bead error heatmap."""
    _ensure_parent(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    _render_panels(ax1, ax2, reference_image, bead_positions, predicted, ground_truth, valid)
    ax1.set_title(f"{title} — {int(valid.sum())} / {len(valid)} beads")
    ax2.set_title("error vs GT")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_combined_scenario_png(
    out_path: Path,
    reference_image: np.ndarray,
    bead_positions: np.ndarray,
    per_algorithm: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scenario_name: str,
) -> None:
    """One row per algorithm; shared error color scale across rows."""
    _ensure_parent(out_path)
    algos = list(per_algorithm.keys())
    n_rows = len(algos)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    # Shared color scale: max of 95th-percentile error across algorithms.
    vmax = 0.0
    for pred, gt, valid in per_algorithm.values():
        if valid.any():
            err_mag = np.hypot((pred - gt)[:, 0], (pred - gt)[:, 1])
            vmax = max(vmax, float(np.nanpercentile(err_mag[valid], 95)))
    vmax = max(vmax, 1e-6)

    for row, algo in enumerate(algos):
        pred, gt, valid = per_algorithm[algo]
        _render_panels(axes[row, 0], axes[row, 1], reference_image, bead_positions,
                       pred, gt, valid, err_vmax=vmax)
        axes[row, 0].set_title(f"{algo} — quivers")
        axes[row, 1].set_title(f"{algo} — error (shared scale, vmax={vmax:.2f} px)")

    fig.suptitle(f"Scenario: {scenario_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(out_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write the summary CSV with a fixed column order."""
    _ensure_parent(out_path)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})


def write_summary_bar_chart(out_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Grouped bar chart of RMSE per (scenario, algorithm)."""
    _ensure_parent(out_path)
    df = pd.DataFrame(list(rows))
    pivot = df.pivot(index="scenario", columns="algorithm", values="rmse_px")

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("RMSE (px)")
    ax.set_title("Displacement RMSE by scenario × algorithm")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_tidy_displacements_csv(
    out_path: Path,
    scenario: str,
    algorithm: str,
    bead_positions: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    append: bool,
) -> None:
    """Tidy long-format per-bead rows. `append=False` writes a header."""
    _ensure_parent(out_path)
    err = predicted - ground_truth
    err_mag = np.hypot(err[:, 0], err[:, 1])
    n = len(bead_positions)
    rows = [
        {
            "scenario": scenario,
            "algorithm": algorithm,
            "bead_id": i,
            "ref_x": float(bead_positions[i, 0]),
            "ref_y": float(bead_positions[i, 1]),
            "pred_dx": float(predicted[i, 0]),
            "pred_dy": float(predicted[i, 1]),
            "gt_dx": float(ground_truth[i, 0]),
            "gt_dy": float(ground_truth[i, 1]),
            "err_x": float(err[i, 0]),
            "err_y": float(err[i, 1]),
            "err_mag": float(err_mag[i]),
            "valid": bool(valid[i]),
        }
        for i in range(n)
    ]
    mode = "a" if append else "w"
    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TIDY_COLUMNS))
        if not append:
            writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_reporting.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add _dev/optical_flow_comparison/reporting.py _dev/optical_flow_comparison/tests/test_reporting.py
git commit -m "Add comparison reporting and plotting"
```

---

## Task 8: Runner

The CLI that ties everything together.

**Files:**
- Create: `_dev/optical_flow_comparison/runner.py`
- Test: `_dev/optical_flow_comparison/tests/test_runner_smoke.py`

The runner:
1. Parses CLI args (`--scenarios`, `--algorithms`, `--benchmark-root`, `--output-dir`).
2. For each scenario: loads + preprocesses reference and deformed images; detects beads in the reference; loads and converts GT.
3. For each `(scenario, algorithm)` pair: calls the adapter, computes metrics, writes per-algorithm PNG, appends to summary rows, appends to tidy CSV.
4. After all pairs in a scenario: writes the combined PNG.
5. After all scenarios: writes the summary CSV and bar chart.

- [ ] **Step 1: Write the failing smoke test**

Create `_dev/optical_flow_comparison/tests/test_runner_smoke.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from _dev.optical_flow_comparison.runner import run


def _make_synthetic_scenario(scenario_dir: Path, shift_px=(2.0, 1.0)):
    """Create a fake benchmark scenario directory with reference + deformed
    TIFs and matching ground-truth .npy fields (in microns; pixel_size_um=0.1)."""
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    h, w = 128, 128

    positions = rng.uniform(15, min(h, w) - 15, size=(30, 2)).astype(np.float32)

    def render(centers):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w), dtype=np.float32)
        sigma = 1.5
        for (x, y) in centers:
            img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        img = img / max(img.max(), 1e-6) * 4000.0
        return img.astype(np.uint16)

    tifffile.imwrite(scenario_dir / "reference.tif", render(positions))
    tifffile.imwrite(scenario_dir / "deformed.tif",
                     render(positions + np.array(shift_px, dtype=np.float32)))

    # GT fields in microns. pixel_size_um=0.1 → 2 px shift = 0.2 µm.
    dx_um = np.full((h, w), shift_px[0] * 0.1, dtype=np.float32)
    dy_um = np.full((h, w), shift_px[1] * 0.1, dtype=np.float32)
    np.save(scenario_dir / "displacement_x.npy", dx_um)
    np.save(scenario_dir / "displacement_y.npy", dy_um)


def test_runner_smoke_end_to_end(tmp_path: Path):
    bench_root = tmp_path / "bench"
    _make_synthetic_scenario(bench_root / "tiny", shift_px=(2.0, 1.0))

    out_dir = tmp_path / "out"

    run(
        benchmark_root=bench_root,
        scenarios=["tiny"],
        algorithms=["DIS", "Farneback", "Lucas-Kanade"],
        output_dir=out_dir,
    )

    # Summary artifacts
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "summary.png").exists()
    assert (out_dir / "displacements.csv").exists()

    # Per-algorithm and combined plots
    for algo in ("DIS", "Farneback", "Lucas-Kanade"):
        assert (out_dir / "tiny" / f"{algo}.png").exists()
    assert (out_dir / "tiny" / "combined.png").exists()

    # Summary CSV sanity
    df = pd.read_csv(out_dir / "summary.csv")
    assert set(df["algorithm"]) == {"DIS", "Farneback", "Lucas-Kanade"}
    assert (df["rmse_px"] >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_runner_smoke.py -q`
Expected: `ModuleNotFoundError: No module named '_dev.optical_flow_comparison.runner'`.

- [ ] **Step 3: Implement `runner.py`**

Create `_dev/optical_flow_comparison/runner.py`:

```python
"""Optical flow comparison runner.

Usage:
    python _dev/optical_flow_comparison/runner.py \
        --scenarios low mid high \
        --algorithms DIS Farneback Lucas-Kanade \
        --benchmark-root _validation/benchmark_TFM \
        --output-dir _dev/optical_flow_comparison/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _dev.optical_flow_comparison.adapters.base import FlowAdapter, sample_dense_at_points
from _dev.optical_flow_comparison.adapters.dis import DISAdapter
from _dev.optical_flow_comparison.adapters.farneback import FarnebackAdapter
from _dev.optical_flow_comparison.adapters.lucas_kanade import LucasKanadeAdapter
from _dev.optical_flow_comparison.detection import detect_beads
from _dev.optical_flow_comparison.metrics import compute_metrics
from _dev.optical_flow_comparison.preprocessing import preprocess
from _dev.optical_flow_comparison.reporting import (
    write_combined_scenario_png,
    write_per_algorithm_png,
    write_summary_bar_chart,
    write_summary_csv,
    write_tidy_displacements_csv,
)

PIXEL_SIZE_UM = 0.1

ADAPTERS: dict[str, type[FlowAdapter]] = {
    "DIS": DISAdapter,
    "Farneback": FarnebackAdapter,
    "Lucas-Kanade": LucasKanadeAdapter,
}


def _load_scenario(scenario_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load + preprocess ref and deformed; load GT as a (H, W, 2) field in px."""
    ref = preprocess(tifffile.imread(scenario_dir / "reference.tif").astype(np.float32))
    deformed = preprocess(tifffile.imread(scenario_dir / "deformed.tif").astype(np.float32))
    gt_dx_um = np.load(scenario_dir / "displacement_x.npy")
    gt_dy_um = np.load(scenario_dir / "displacement_y.npy")
    gt_px = np.stack([gt_dx_um / PIXEL_SIZE_UM, gt_dy_um / PIXEL_SIZE_UM], axis=-1).astype(np.float32)
    return ref, deformed, gt_px


def run(
    benchmark_root: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    output_dir: Path,
) -> None:
    """Run every (scenario, algorithm) pair and write all artifacts."""
    benchmark_root = Path(benchmark_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy_csv = output_dir / "displacements.csv"
    if tidy_csv.exists():
        tidy_csv.unlink()  # fresh run each invocation

    summary_rows: list[dict] = []

    for scenario in scenarios:
        scenario_dir = benchmark_root / scenario
        print(f"[{scenario}] loading...")
        ref, deformed, gt_field = _load_scenario(scenario_dir)
        beads = detect_beads(ref)
        gt_at_beads = sample_dense_at_points(gt_field, beads)
        print(f"[{scenario}] detected {len(beads)} beads")

        per_algo: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for algo_name in algorithms:
            adapter_cls = ADAPTERS[algo_name]
            adapter = adapter_cls()
            print(f"[{scenario}/{algo_name}] running...")
            pred, valid = adapter.displacements_at(ref, deformed, beads)
            metrics = compute_metrics(pred, gt_at_beads, valid)
            print(f"[{scenario}/{algo_name}] RMSE={metrics['rmse_px']:.3f} px  "
                  f"median={metrics['median_px']:.3f} px  coverage={metrics['coverage']:.2%}")

            write_per_algorithm_png(
                out_path=output_dir / scenario / f"{algo_name}.png",
                reference_image=ref,
                bead_positions=beads,
                predicted=pred,
                ground_truth=gt_at_beads,
                valid=valid,
                title=f"{scenario} / {algo_name}",
            )
            write_tidy_displacements_csv(
                out_path=tidy_csv,
                scenario=scenario,
                algorithm=algo_name,
                bead_positions=beads,
                predicted=pred,
                ground_truth=gt_at_beads,
                valid=valid,
                append=tidy_csv.exists(),
            )

            summary_rows.append({"scenario": scenario, "algorithm": algo_name, **metrics})
            per_algo[algo_name] = (pred, gt_at_beads, valid)

        write_combined_scenario_png(
            out_path=output_dir / scenario / "combined.png",
            reference_image=ref,
            bead_positions=beads,
            per_algorithm=per_algo,
            scenario_name=scenario,
        )

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_bar_chart(output_dir / "summary.png", summary_rows)
    print(f"\nDone. Artifacts in {output_dir}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare optical flow algorithms on TFM benchmarks.")
    p.add_argument("--scenarios", nargs="+", default=["low", "mid", "high"])
    p.add_argument("--algorithms", nargs="+", default=list(ADAPTERS.keys()),
                   choices=list(ADAPTERS.keys()))
    p.add_argument("--benchmark-root", type=Path,
                   default=_REPO_ROOT / "_validation" / "benchmark_TFM")
    p.add_argument("--output-dir", type=Path,
                   default=_HERE / "output")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        benchmark_root=args.benchmark_root,
        scenarios=args.scenarios,
        algorithms=args.algorithms,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test to verify it passes**

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests/test_runner_smoke.py -q`
Expected: `1 passed`. The synthetic scenario uses a uniform 2-px shift and ~30 beads; all three adapters should produce valid (if not necessarily great) results on it.

- [ ] **Step 5: Commit**

```bash
git add _dev/optical_flow_comparison/runner.py _dev/optical_flow_comparison/tests/test_runner_smoke.py
git commit -m "Add optical flow comparison runner"
```

---

## Task 9: Run the real benchmark and verify DIS reproduces baseline

The point of this whole plan: actually produce the comparison.

**Files:**
- No code changes. Just run the benchmark and inspect outputs.

- [ ] **Step 1: Run the benchmark on the real scenarios**

Run:

```bash
conda run -n napariTFMv2 python _dev/optical_flow_comparison/runner.py \
    --scenarios low mid high \
    --algorithms DIS Farneback Lucas-Kanade
```

Expected: prints per-scenario, per-algorithm RMSE / median / coverage. Writes `_dev/optical_flow_comparison/output/summary.csv`, `summary.png`, per-scenario subdirectories with per-algorithm and combined PNGs, and `displacements.csv`.

- [ ] **Step 2: Verify the summary artifacts exist**

Run:

```bash
ls -la _dev/optical_flow_comparison/output/
ls -la _dev/optical_flow_comparison/output/low/
ls -la _dev/optical_flow_comparison/output/mid/
ls -la _dev/optical_flow_comparison/output/high/
cat _dev/optical_flow_comparison/output/summary.csv
```

Expected: `summary.csv`, `summary.png`, `displacements.csv` at top level; `{DIS,Farneback,Lucas-Kanade}.png` and `combined.png` in each scenario directory.

- [ ] **Step 3: Verify DIS reproduces the existing baseline within numerical noise**

The existing `_validation/benchmark_TFM/validate_TFM.py` overrides DIS parameters per scenario, so don't expect a bit-exact match. What to compare:

```bash
conda run -n napariTFMv2 python _validation/benchmark_TFM/validate_TFM.py
```

Then compare the printed RMSE numbers from `validate_TFM.py` against the DIS rows in `summary.csv`. They should be in the same ballpark (same order of magnitude per scenario). If they differ by more than ~30%, the adapter wiring is probably reading default parameters where it should not; investigate before declaring the benchmark trustworthy.

- [ ] **Step 4: Run the full test suite to confirm no regressions**

Run: `conda run -n napariTFMv2 python -m pytest tests -q`
Expected: `16 passed` (the baseline from `master`).

Run: `conda run -n napariTFMv2 python -m pytest _dev/optical_flow_comparison/tests -q`
Expected: all tests pass.

- [ ] **Step 5: Final commit (if any output was inadvertently staged)**

Run: `git status` and confirm only intended files are tracked. The `output/` directory should be gitignored from Task 0; nothing in it should be staged.

If everything is clean, no commit needed.

---

## What this plan does NOT do

These are intentional out-of-scope items captured in the spec and reproduced here so they don't get sneaked into "while I'm here" edits:

- No PIV or PTV adapters. Add later as a new file in `adapters/` if the comparison motivates it.
- No grid-interpolation evaluation surface. The runner is structured so this can be added without restructuring (sample dense methods on a grid; scatter-interpolate LK onto the same grid; compare). Out of scope this round.
- No production integration. Nothing in `napariTFM/backend/` is modified.
- No per-scenario parameter tuning. All adapters use defaults documented in their source files. Tuning is a follow-up if the comparison motivates it.

---

## Self-Review (run by plan author)

**Spec coverage:**
- DIS / Farneback / LK adapters → Tasks 4, 5 ✓
- Shared runner with plug-in adapters → Task 8 ✓
- Bead-position evaluation surface → Tasks 3, 6 ✓
- Preprocessing locked to `[80, 99.9]` + σ=1 → Task 2 ✓
- TrackPy `diameter=7, separation=8, minmass="auto"` with 30th-percentile estimator → Task 3 ✓
- Float32 [0, 1] adapter input contract → Tasks 1, 4, 5 ✓
- Metrics: RMSE, median, p95, coverage, bias-by-magnitude → Task 6 ✓
- Per-algorithm PNG, combined PNG per scenario, summary CSV, summary bar chart, tidy CSV → Task 7 ✓
- Extensible to a fourth adapter without runner changes → Task 8 `ADAPTERS` dict ✓
- DIS reproduces baseline RMSE → Task 9 step 3 ✓

**Placeholder scan:** no TBDs, no "add error handling", no "similar to Task N", no missing code blocks. Pass.

**Type consistency:** adapter signature `(reference, deformed, query_points) → (displacements, valid_mask)` used identically across `base.py`, `dis.py`, `farneback.py`, `lucas_kanade.py`, runner. `sample_dense_at_points(flow, points)` signature consistent across `base.py` and adapter call sites. `compute_metrics(predicted, ground_truth, valid)` consistent across `metrics.py` and runner. Reporting function signatures match between definitions and runner call sites. Pass.
