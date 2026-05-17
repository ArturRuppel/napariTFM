# Optical Flow Algorithm Comparison for TFM

## Goal

Benchmark optical flow algorithms on the existing TFM ground-truth benchmarks
(`_validation/benchmark_TFM/{low,mid,high}/`) to inform whether the production
displacement backend (currently OpenCV DIS) should be replaced or supplemented.
The output of this work is a reproducible comparison report, not a production
change.

## First-Principles Motivation

TFM images are sparse bright bead patterns on dark background. Three properties
shape the algorithm choice:

1. **Image information is localized.** Only pixels at or near bead centers
   carry displacement-relevant signal. Most pixels are background.
2. **Beads are conserved.** The same population of beads is present in
   reference and deformed frames (modulo z-drift). No birth/death in plane.
3. **Sub-pixel accuracy is essential.** Typical displacements are 0.1–3 px;
   a 0.1 px error is a meaningful fraction of the signal.

Dense optical flow computes a vector at every pixel. At background pixels
that vector is regularization extrapolated from bead pixels, not measurement.
Sparse optical flow solves only where there is structure, which is a better
impedance match. Whether that theoretical advantage translates to lower RMSE
on real benchmarks is what this experiment tests.

## Scope

**In scope (this round):**
- Three algorithms: DIS (current baseline), Farneback (alternative dense OF),
  Lucas-Kanade pyramidal (sparse OF).
- Evaluation at bead positions only — the most defensible scientific
  comparison surface.
- All three benchmark scenarios: low, mid, high.

**Out of scope (deferred):**
- PIV (cross-correlation) — community-standard comparator; valuable but
  bigger lift, comes next round if the dense/sparse comparison motivates it.
- PTV (TrackPy detection + LapTrack linking) — already prototyped in
  `_dev/spt_piv/spt_piv_displacement.py`; integrate after the OF-only round.
- TV-L1, RLOF, ML-based methods.
- Grid-interpolation evaluation for downstream FTTC propagation — the *final*
  verdict will require this, but it is a follow-up step once the bead-position
  comparison is in place. The runner is designed so the grid evaluator can be
  added without restructuring.
- Production integration of any winning algorithm.

## Architecture

### Layout

```
_dev/optical_flow_comparison/
  runner.py                 # CLI entry: orchestrate scenarios × algorithms
  adapters/
    __init__.py
    base.py                 # Adapter protocol + shared utilities
    dis.py                  # DIS adapter (wraps DisplacementAnalyzer)
    farneback.py            # Farneback adapter (cv2.calcOpticalFlowFarneback)
    lucas_kanade.py         # LK adapter (cv2.calcOpticalFlowPyrLK)
  detection.py              # TrackPy bead detection helper (shared)
  preprocessing.py          # Thin wrapper over napariTFM ImageProcessor
  metrics.py                # RMSE / median / coverage / bias-by-magnitude
  reporting.py              # Per-algorithm and combined PNG generation
  output/                   # Generated artifacts (gitignored)
    low/
    mid/
    high/
    summary.csv
    summary.png
```

### Adapter Protocol

All algorithms expose the same minimal interface:

```python
class FlowAdapter(Protocol):
    name: str  # e.g. "DIS", "Farneback", "Lucas-Kanade"

    def displacements_at(
        self,
        reference: np.ndarray,      # float32, shape (H, W), range [0, 1]
        deformed: np.ndarray,       # float32, shape (H, W), range [0, 1]
        query_points: np.ndarray,   # float32, shape (N, 2), columns (x, y)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (displacements, valid_mask).

        displacements: shape (N, 2), columns (dx, dy), in pixels.
        valid_mask:    shape (N,), bool. True where the algorithm reports a
                       trustworthy result. Dense methods return all-True;
                       LK returns False for points where it failed to converge
                       or fell off the image during pyramid tracking.
        """
```

Internal behavior of each adapter:

- **DIS** — wraps the existing `napariTFM.backend.displacement_analysis.DisplacementAnalyzer`.
  Computes the dense flow once, then samples at `query_points` via bilinear
  interpolation. Uses default `DisplacementParameters`. All-True valid mask.
- **Farneback** — calls `cv2.calcOpticalFlowFarneback` with documented defaults
  (`pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2`).
  Same bilinear sampling pattern. All-True valid mask.
- **Lucas-Kanade** — calls `cv2.calcOpticalFlowPyrLK` with `query_points` as the
  input feature set, returning displaced positions directly. Defaults:
  `winSize=(15, 15), maxLevel=3, criteria=(EPS|COUNT, 20, 0.03)`. Valid mask
  is the `status` array returned by OpenCV.

Each adapter is fully self-contained: it owns its preprocessing/conversion
(e.g., float→uint8), its parameter defaults, and its internal calls. The runner
treats them as opaque.

### Runner

`runner.py` is the single entry point:

```bash
python _dev/optical_flow_comparison/runner.py \
    --scenarios low mid high \
    --algorithms DIS Farneback Lucas-Kanade \
    --output-dir _dev/optical_flow_comparison/output
```

For each (scenario, algorithm) pair:
1. Load reference + deformed images. Apply shared preprocessing
   (percentile `[80, 99.9]`, Gaussian σ=1) once per scenario; cache.
2. Detect beads on the preprocessed reference (TrackPy: `diameter=7,
   separation=8, minmass="auto"`). Cache per scenario. The `"auto"` minmass
   estimator follows the prototype in `_dev/spt_piv/spt_piv_displacement.py`:
   locate with `minmass=0`, then return the 30th percentile of the resulting
   mass distribution (keeps real beads, drops noise peaks).
3. Load dense ground-truth displacement fields (`displacement_x.npy`,
   `displacement_y.npy`), convert µm → px (`/0.1`), sample at bead positions.
4. Call the adapter to get predicted displacements at bead positions.
5. Compute metrics; write per-bead rows to the tidy CSV; render the
   per-(scenario, algorithm) PNG.

After all pairs finish, render the per-scenario combined PNGs and the summary
artifacts.

### Data Flow

```
benchmark TIFs ──► preprocessing ──► preprocessed (cached per scenario)
                                          │
                                          ├──► TrackPy detection ──► bead positions (cached)
                                          │                                │
                                          └──► adapter.displacements_at ──┴──► predicted displacements
                                                                                       │
GT *.npy ──► µm→px conversion ──► GT field ──► sample at bead positions ──► GT displacements
                                                                                       │
                                          metrics + per-bead CSV ◄──────────────────┘
                                                       │
                                                       ▼
                                         per-PNG, combined PNG, summary CSV + bar chart
```

## Metrics

Per (scenario, algorithm):

| Metric | Definition |
|---|---|
| `n_beads` | Number of beads detected in the reference. |
| `coverage` | Fraction of beads with `valid=True` from the adapter. |
| `rmse_px` | √(mean of `err_x² + err_y²` over valid beads). Primary scalar. |
| `median_px` | Median of `√(err_x² + err_y²)` over valid beads. Robust. |
| `p95_px` | 95th percentile of the same. Tail behavior. |
| `bias_low/mid/high` | Mean signed error magnitude in three GT-magnitude bins (0–1 px, 1–3 px, >3 px). Detects systematic under-/over-shoot. |

Errors are computed only over beads with `valid=True`. Coverage is reported
separately so the comparison is not skewed by a method silently dropping
hard-to-track beads.

## Reports

Per `(scenario, algorithm)`:
- `output/<scenario>/<algorithm>.png` — two panels: reference image with
  GT and predicted displacement quivers overlaid; error heatmap at bead
  positions.

Per scenario:
- `output/<scenario>/combined.png` — one row per algorithm, same column
  layout and shared color scale.

Overall:
- `output/summary.csv` — one row per (scenario, algorithm) with all metrics.
- `output/summary.png` — grouped bar chart of RMSE by scenario × algorithm.
- `output/displacements.csv` — tidy per-bead long-format table:
  `scenario, algorithm, bead_id, ref_x, ref_y, pred_dx, pred_dy,
   gt_dx, gt_dy, err_x, err_y, err_mag, valid`.

## Testing Approach

This is exploratory benchmark code in `_dev/`, not production. Validation is
the benchmark itself: if `DIS` reproduces numbers consistent with the existing
`validate_TFM.py` run on the same scenarios, the adapter wiring is correct.
No dedicated unit tests for the runner; small smoke check that each adapter
returns a `(N, 2)` array of the right shape on a tiny synthetic image.

## Dependencies

All already present in the `napariTFMv2` environment:

- `opencv-python` (DIS, Farneback, LK)
- `trackpy` (bead detection)
- `numpy`, `pandas`, `tifffile`, `matplotlib`, `scipy`

No new packages.

## Success Criteria

1. Single command produces `summary.csv` and `summary.png` for all three
   scenarios × three algorithms.
2. DIS adapter RMSE on each scenario matches (within numerical noise) the
   numbers produced by `_validation/benchmark_TFM/validate_TFM.py` — confirms
   the wiring is honest.
3. The summary report supports a clear written conclusion: which algorithm
   is best by RMSE, by tail behavior, by coverage, and whether differences
   are large enough to motivate a production change.
4. The codebase is structured so adding a fourth adapter (PIV is the likely
   candidate) requires only a new file in `adapters/`, with no changes to
   the runner.
