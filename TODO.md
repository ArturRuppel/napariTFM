# napariTFM — Open Worklist

> Accomplished items are pruned (see git history for completed UI-redesign
> slices P0–P8, the UI-Coherence roadmap, the BISM port, the 2026-06-29
> batch/cancel/colorbar/sink work, the 2026-06-30 unified-logging (§1) +
> per-position Export-to-CSV (§2) work, and the §3 streaming-follows-active-
> position work). What remains below is **open work only**, ranked
> easy-wins-first.

---

## EPIC: one-shot TFM backend — pipeline collapse + benchmark (2026-07-06)

The big rock, and it reframes most of the UI worklist below. We prototyped a
**one-shot (physics-as-prior) TFM solver** that recovers surface traction directly
from a reference/deformed image pair in a **single differentiable solve**
(unknown traction on a compact basis → `u = G·t/E` [2D Boussinesq, FFT] → warp ref →
ZNCC photometric loss; one autograd graph, L-BFGS coarse-to-fine over an image
pyramid; a rigid-translation DOF absorbs stage drift). On the published dipole
benchmark, scored with our own Sabass metrics, it **matches TV-L1+FTTC on easy tiers
and beats it on the hard (large-displacement) tier**, with better magnitude accuracy
(DTM) and force confinement (DTMS) — and it does so without a separate displacement
stage or preprocessing stage.

**Why this is strategic, not just another backend:** the one-shot **absorbs
preprocessing (bg-subtraction + drift) AND displacement AND traction into one step.**
Our 3–4 stage pipeline (preprocess → displacement → traction → stress) collapses to
**one solve, plus an optional second stress stage**. That invalidates a large fraction
of the multi-stage UI/UX (per-stage panels, intermediate-layer plumbing, stage
resume/merge logic) — but the tool comes out simpler and better. Read this epic before
investing further in per-stage UI polish below.

### Where the prototype lives (all in the `napariTFM2.5D` working copy, branch `ffd-displacement-backend`)
These are **untracked** files there (`_dev` is gitignored; the two source files are `??`),
so step one of integration is bringing them under version control **in this repo**.
- `napariTFM/backend/oneshot.py` — `OneShotSolver`, `OneShotParameters`, `OneShotResult`.
  Lazy-imports torch (optional `[ffd]` extra), same pattern as the FFD backend. The whole
  fused solver. **This is the thing to port.**
- `napariTFM/widgets/oneshot_widget.py` — magicgui playground dock widget (the trimmed
  UX: E, ν, pixel size, **Smoothness σ**, a single **Mask-confinement** dial, preprocess
  toggle, device). Run today via `_dev/oneshot_playground_launch.py`; not yet in
  `napari.yaml`. This is the seed for the production panel, not the production panel.
- `tests/test_oneshot.py` — 13 backend tests (port alongside the backend).
- `_validation/benchmark_TFM/compare_oneshot.py` — the head-to-head harness (one-shot vs
  TV-L1+FTTC on the published `low/mid/high` dipole tiers, scored via the tracked
  `validate_TFM.py` metrics). Untracked; the honest comparison lives here.
- `_dev/oneshot_bench2d/` — the 2D research harness (synthetic dipole/keratocyte cases,
  real-singlet sample in `data/singlet20_*.tif`, method bake-offs). gitignored scratch.
- `_dev/bench_generator/` — the **scenario-sweep generator** (SPEC.md, fields.py, beads.py,
  psf.py, render.py, noise.py, writer.py, cli.py) with vendored analytic physics in
  `_dev/vendor/DirectMethod/`. This is the seed for TASK 2.

### Settled findings — bake these in, don't relitigate
- **Basis:** beads-as-nodes (one Gaussian traction node per detected reference bead) +
  soft mask. `node_sigma` is *the* smoothness knob.
- **Mask confinement:** one log-scaled soft-β dial (0 = none → strong confinement). The
  literal hard gate was **retired** — it clips genuine near-edge forces (|t|r 0.95 vs 0.99).
- **Preprocess default OFF:** bg+drift *slightly degrade* the force map on every tier we
  tried (the rigid DOF handles drift without resample blur; bead detection high-passes
  internally regardless). Keep it as an opt-in checkbox for pathologically bright backgrounds.
- **Pyramid depth:** the current default `iterations="50,70,150"` (3 levels, coarsest /4)
  **under-captures large displacement** — on the high-force tier one corner adhesion locked
  onto the wrong minimum (DTA 18°). A 4–5 level pyramid fixes it at no time cost (DTA 0.6°).
  **Make the default ≥4 levels.**
- **Dead ends (already refuted, don't retry):** zero-net-force projection (acts only on the
  null-space DC mode — useless), point-matching / Chamfer data terms (ill-posed, per-bead
  hotspots — photometric ZNCC is the robust choice), TV vs L2 reg (negligible unless noise
  makes the regularizer binding). The backend still carries some of these as dormant params
  — strip on port.

### TASK 1 — integrate into the production pipeline
- **Architecture:** expose the one-shot as a **traction backend that reaches back to the
  image pair** and **emits its displacement byproduct (`u = G·t`) as the normal displacement
  layer**, so MSM / strain-energy / BISM downstream still get a displacement field. This is
  the FFD-backend toggle pattern applied one stage over; preferred over a parallel "one-shot
  mode" tab (which duplicates UI).
- **UI collapse:** most per-stage panels fold into a single solve panel with the routine
  dials above. Preprocess and displacement panels lose their reason to exist in this path.
  Keep the two-step (TV-L1+FTTC) selectable as the legacy/alternative backend.
- **Data model:** keep the displacement layer as the contract between the (now fused)
  traction step and the stress step. Stress (MSM/BISM) stays a second stage.
- **Plumbing:** port the untracked files → tracked; torch stays an optional `[ffd]` extra
  (lazy import, manifest scan stays torch-free); set defaults from TASK 2 (esp. pyramid
  depth); wire into `napari.yaml`.
- **Open decision for Artur:** confirm "traction-backend-that-emits-displacement" vs a
  standalone mode before building the panel.

### TASK 2 — a solid, fair TVL1+FTTC vs one-shot benchmark
- **The cardinal rule (learned the hard way):** tune **each method's parameters per
  scenario** and report each at its *own* best. A benchmark with hand-picked FTTC reg is
  worthless — the committed `validate_TFM.py` uses `regularization=1e-6`, which
  under-regularizes (over-estimates, DTMS ~0.97) and does **not** reproduce the paper's
  Fig 3; ~1e-4 is the balanced operating point. FTTC gets GCV + a reg sweep; the one-shot
  gets a small grid over σ / pyramid depth / mask. (Sidebar: the committed validation script
  not reproducing the paper is itself worth a fix — see the standalone note if we split it out.)
- **Scenario axes:** noise level, bead density, displacement magnitude, force geometry
  (dipole / realistic cell / keratocyte), substrate stiffness, and **mask available vs not**
  (the one-shot's real edge — the dipole benchmark can't show it). Build on
  `_dev/bench_generator` sweeps.
- **Metrics:** the Sabass four (correlation, DTM, DTMS, DTA — from `validate_TFM.py`) +
  strain energy + wall-clock.
- **Deliverable:** sweep curves + per-scenario best-config tables + a summary figure, plus
  the resulting production **default parameters** (feeds TASK 1).

### PHASE 3 (separate, focused) — stress benchmark incl. BISM
Extend the benchmark to intercellular **stress** so MSM and **BISM** (already in-repo:
`napariTFM/backend/bism.py`, `stress.py`) can be evaluated head-to-head. Needs different
ground truth (the paper's FEM active-cell model, or the Mavi sims) and a stress front-end.
Deliberately **not** bundled into TASK 2 — different machinery, would muddy both.

### TASK 4 (optimization) — swap the confined forward solver to preconditioned CG
The β>0 confined path in `forward_tfm.py::_solve_iterative` currently uses L-BFGS +
autograd (torch). The loss is a **convex quadratic**, so its minimizer solves the normal
equations `A t = b` with SPD `A` — the textbook case for **preconditioned Conjugate
Gradient**, using the existing Fourier closed-form (`_solve_closed_form`) as the
preconditioner. Expected ~5–10× wall-clock and — because CG needs no autograd — it
**drops the hard torch dependency** on this path (torch/cupy become an optional GPU
accelerator, not a requirement). One caveat: hand-rolled CG needs the exact adjoint (add
a dot-product test). Also the reusable inner-solve engine if the localization prior is
later upgraded L2 → L1. Full rationale, math, and acceptance criteria:
**[`docs/specs/forward-solver-pcg.md`](docs/specs/forward-solver-pcg.md)**.

---

## Open after the round-2 review merge (2026-07-04)

Leftovers from PR #3 (`CODE_REVIEW_FINDINGS_2026-07-03.md`, now in-repo) that
were *not* landed with the merge. The Tier-1 bugs (B-1 odd-grid FFT, B-2 stress
error-swallow, B-4/B-5/B-6), the Tier-2 dead-code deletions, and the A-1/A-3/A-4
refactors are all done and test-green (625 passed). What remains, ranked
easy-wins-first:

### Vectorize `downscale_flow`  ·  DONE (2026-07-04)
`displacement_analysis.py::downscale_flow` was an O(H·W) Python double loop
computing a pure block-mean, on every displacement calc. Replaced with
`flow[:nh*f,:nw*f].reshape(nh,f,nw,f,2).mean(axis=(1,3))`. Measured **~11×**
faster on a 1000×1000 flow at factor 4 (the finding's "~100×" was optimistic),
and the output is **bit-identical** to the loop (max|diff| = 0). Locked with an
equivalence test against a reference reimplementation of the old loop, plus
factor=1 passthrough and an exact-block-average test. 633 passed.

### Split compute-critical from viz-only validation in `validate_fttc_parameters`  ·  DONE (2026-07-04)
`validate_fttc_parameters` is the pre-compute gate for `calculate_force_field` /
`find_optimal_regularization`, but it was failing the whole force computation on
visualization-only params (`force_arrow_scale`, `f_max`, `force_vector_stride` —
none of which enter the traction solve) and enforcing `regularization > 0` even
under `auto_gcv=True`, where the manual value is ignored. **Fix:** dropped the
three viz-only checks from the compute gate (they belong at the rendering layer,
not here — not shuffled into a new uncalled validator, which would just recreate
dead code) and gated the regularization check on `not auto_gcv`. Test-locked:
viz-only params no longer block, reg≤0 is fine under auto-GCV but still rejected
with it off, and the existing compute-critical checks are unchanged.

### BUG: interactive upstream re-run resurrects a stale downstream stage on disk  ·  DONE (2026-07-04, B-3)
CONFIRMED bug, interactive-path only: re-running **displacement** after
displacement→force were on disk left `disp_v2` paired with the old `force_v1`
(computed from `disp_v1`), because `merge_arrays` restored the force that was
absent-from-the-write (invalidated in memory) but present-in-old.
**Fix (the `_DOWNSTREAM`-aware merge option, not `merge_existing=False`):**
`merge_arrays` (`utilities/ntfm.py`) now treats a stage *present* in the write as
proof it was recomputed, so its downstream stages (`_DOWNSTREAM_ARRAY_KEYS`) are
**not** resurrected from disk even when absent from the write. This distinguishes
an upstream re-run (displacement present → drop stale force) from a legitimate
force-only resume (displacement absent → preserve it) purely from the arrays
being merged — no call-site knowledge, so it fixes the batch path too. Chose this
over a blunt `merge_existing=False` because the latter would erase displacement
during a real force-only resume. Test-locked in `test_ntfm_merge.py`: the
buggy-direction unit test was flipped to assert non-resurrection, plus symmetric
stress-under-fresh-force, combined preserve-upstream-drop-downstream, and an
end-to-end `results_to_ntfm` disk regression. 628 passed.

### Collapse the vector-stage Widget/Controller triplication + one blessed lifecycle  ·  DONE (2026-07-04, A-2) — NEEDS MANUAL IN-APP VERIFICATION BEFORE MERGE
Landed on `claude/a2-stage-controller-unification`. What changed:
- **One sealed run/cancel lifecycle** in `BaseAnalysisController`. `run()`/`cancel()`
  are *non-overridable* — sealed via `__init_subclass__` **and** a name-mangled
  `__run` (both verified to hold under this Qt binding). Subclasses fill hooks
  (`_validate` / `_run_params` / `_begin_stream` / `_build_worker` /
  `_on_frame_processed` / `_finalize`). The UI unfreezes on **every** terminal
  path via the worker's `finished` signal (the single chokepoint) — a stage can
  never again forget to freeze on run (B-6) because the base owns the sequence.
- **One cooperative cancel.** Verified against napari 0.7.0: `@thread_worker`
  workers expose **only** `quit()` — no `wait`/`isRunning`/`terminate`/`deleteLater`.
  So the old `terminate()`/`wait(500)` in all three cancels was **dead code**
  (swallowed `AttributeError`); the only thing that ever ran was `quit()`. New
  cancel is `quit()` + disconnect `yielded` (kills the late-frame race) and
  defers teardown/cleanup to `finished` — GUI thread never blocks. The stray
  `QApplication.processEvents()` is gone.
- **disp/force collapsed** into `VectorStageController` (spec = `STAGE_KIND` +
  `RESULT_SETTER` + a few hooks); `DisplacementController`/`FTTCController` are
  thin subclasses (names kept — tests import them). Stress fills the hooks
  directly (its streaming/finalize differ). Preprocessing untouched (bespoke).
- **All previews + GCV made synchronous** (were: force async, disp/stress sync).
  Chosen deliberately given the untestable-GUI constraint — sync has zero
  cross-thread surface, and sync→async later is a cheap additive change onto the
  now-proven run lifecycle; async→sync after a field heisenbug is not.
- Net −84 lines of production code; new `tests/test_stage_lifecycle.py`
  (7 tests) pins freeze-on-run / unfreeze-on-every-terminal-path /
  finalize-on-returned / cooperative-cancel-defers-teardown / seal. 640 passed.

**⚠️ Still needs manual in-app verification before merge.** No test drives these
controllers through a *real* Qt background worker (the suite fakes them), so a
genuine threading regression would pass CI. Drive the real app: run each of
displacement/force/stress (watch Run/Preview disable then re-enable), **cancel
mid-run** (watch it stop without hanging and the UI re-enable), and preview a
heavy frame on each. Only merge after that passes.

### Minor cleanups (low priority, do opportunistically)
- **Mask resized twice per stress folder** and the resize logic is duplicated
  verbatim between `batch_analysis.py:1261-1273` and `stress.py:73-85` — resize
  once, share one helper.
- **Three near-identical `physical_scale` dicts** (`displacement_analysis.py:213`,
  `fttc.py:100`, `bism.py:365`) differ only in a unit-name string — one helper.

**Considered and deliberately dropped** (don't re-open without a reason): the
dead params `tfm_folder`/`folder`/`preprocessed_data` (leftover signatures, low
value); the fttc GCV micro-cleanups (`_interp_vec2grid` NaN branch, `np.copy`,
`np.max([minGi,0])` — fttc is numerically sensitive, not worth the risk for
cosmetics). The `metrics_calculator.py` polarization fix (`eigvals`→`eigvalsh`,
centroid-not-origin moment) is real but stays parked under the existing "#9:
wire up metrics later" decision — fix it *when* it's wired up.

---

## New feature ideas (2026-07-04) — research phase

Four ideas from the owner, ranked easy-wins-first. All are **researched but not
yet designed** — the notes below capture intent + open questions; a design spec
comes before any code. Research findings will be appended as they land.

### 1. Auto-threshold the beads images (e.g. Otsu)  ·  S  ·  cheap win, trivial
Add an optional automatic threshold to the preprocessing path so the background
floor isn't set by hand. Today `preprocess_frame`
(`backend/preprocessing.py:47`) does Gaussian blur → **percentile** intensity
scaling (`apply_intensity_scaling`, manual `min/max_intensity_percentile`) →
optional registration. `skimage.filters.threshold_otsu` (already available via
napari's deps) auto-picks the background level. **Owner: trivial, no research
needed** — just implement. (Note when building: prefer using Otsu to set the
intensity-scaling floor rather than hard-zeroing sub-threshold pixels before
Farneback, which would strip subpixel texture — but this is an implementation
detail, not a blocker.)

### 2. GPU-accelerated displacement  ·  ✅ DONE (2026-07-06)
**Resolved: base method switched Farneback → multi-pass cross-correlation PIV,
with a PyTorch GPU backend.** The Farneback optical-flow front end is gone —
`DisplacementAnalyzer`, the `cv2.calcOpticalFlowFarneback` call, and all the
`nscales`/`inner_iterations`/`pyr_scale`/`poly_*`/`use_gaussian_window` params/UI
were ripped out. The displacement path is now `PIVDisplacementAnalyzer`
(`backend/piv_displacement.py`, ported from the `napariTFM2.5D` prototype):
FFT windowed cross-correlation, Hanning taper, 3-point Gaussian subpixel peak,
normalized-median outlier rejection, coarse→fine window deformation. It has a
**torch-free numpy core** (no new hard dep) and transparently uses a **PyTorch
CUDA** backend when available (~100× faster, numerically equivalent on dense
beads), selected by `piv_device` (`auto`/`cuda`/`cpu`). Torch is an optional
extra: `pip install napariTFM[piv]`. This settles the base-method question below
in favour of PIV and delivers the portable GPU path in one move. Everything from
here down is the historical investigation that led to that decision.

**Findings (2026-07-04):**

⚠️ **The base-method question is bigger than the GPU question — settle it first.**
Farneback (what we use) is *not* the TFM-standard displacement method. The
canonical TFM pipeline (Butler 2002; Sabass/Gardel/Waterman/Schwarz 2008 — the
FTTC paper itself; Danuser-lab TFM; PIVlab; Style 2014; Schwarz-Soiné 2015
review) uses **cross-correlation PIV** (block-matching + Gaussian/parabolic
subpixel peak fit), *not* Farneback. Farneback is a generic-CV convenience with
**no TFM-specific validation**. The evidence on optical-flow-vs-PIV for beads is
genuinely mixed: one high-res-TFM paper (PMC5292691) reports optical flow beats
PIV on noise for deformation measured *at* beads, but a 2025 benchmark
(bioRxiv 2025.07.04.663196) finds optical flow wins *only* in the deep-subpixel
regime and is the least accurate over a broad displacement range, while
cross-correlation handles larger displacements and lower image quality better.
**For a paper in revision this is a correctness question that dwarfs speed** —
worth deciding whether the base method should be cross-correlation PIV (CPU
`openpiv-python`, actively maintained) before investing in a GPU path.

On the GPU path itself (if we keep/optimize dense flow):
- **`cv2.cuda.FarnebackOpticalFlow` is a portability trap.** pip `opencv-python`
  is CPU-only; CUDA OpenCV means a source build or a single-maintainer community
  wheel channel (`cudawarped/opencv-python-cuda-wheels`, installed by URL, not
  PyPI) — not `pip`-able for a non-coding user, and NVIDIA-only (no Mac). At best
  an opportunistic runtime-detect ("use cv2.cuda iff present"), never a dep.
- **The PyTorch GPU path only exists if we move to PIV — it does NOT accelerate
  Farneback.** There is **no torch Farneback** (searched: only NumPy ports like
  `ericPrince/optical-flow` exist; Farneback is per-pixel polynomial expansion +
  pyramid + iterative least-squares, not an FFT). To GPU Farneback you'd need
  `cv2.cuda.FarnebackOpticalFlow` (install trap, above) or the CUDA packages
  `farneback3d`/`OpticalFlow3d` — CUDA-only. **Cross-correlation PIV, by contrast,
  maps cleanly onto torch** (FFT/`Conv2d` NCC), and working torch PIV libs already
  exist: `TorchPIV` (PyPI), `erfanhamdi/torch_PIV`, and a peer-reviewed
  GPU-PIV framework (Comput. Phys. Commun. 2024). PyTorch is the best *portable*
  GPU path — CUDA on Win/Linux, **MPS on Apple Silicon**, CPU fallback, all from
  the same `torch` (mirrors Cellpose's `cuda → mps → cpu` dispatch in
  `cellpose/core.py`); CuPy strands Mac (CUDA-only, ROCm Linux-only/immature).
  **But this whole path is unlocked by the PIV switch, not by torch per se** — so
  "add a torch backend" and "switch base method to PIV" are one decision, not two.
- **`GCpu_OpticalFlow` (chabibchabib)** is still a good *pattern* reference
  (CuPy+cuCIM with automatic NumPy fallback in one code path) but inherits CuPy's
  no-Mac limitation if used as-is.
- **`OpticalFlow3d` (yongxb)** — CUDA Farneback/Lucas-Kanade for dense *3D* flow.
  Relevant to (3): getting `u_z` from bead z-stacks is a 3D-flow problem.
- **Ruled out:** NVIDIA hardware optical flow (NVOFA / `cv2.cuda_NvidiaOpticalFlow`)
  — hard **0.25 px** subpixel floor (S10.5 vectors, only 2 fractional bits
  populated per NVIDIA's own docs), scientifically invalid for TFM. torchvision
  **RAFT** — no TFM validation, documented low-SNR/out-of-distribution failure on
  speckle/PIV data; off-the-shelf weights inappropriate for beads.
- **GPU-PIV Python packages are immature:** `openpiv-python-gpu` is pre-beta,
  no pip/conda pkg, NVIDIA-only; `quickPIV` (Julia) is CPU-only anyway and stale;
  `pyGPUreg` is any-GPU (incl. Mac) but global single-shift registration, not
  windowed PIV. If we go PIV, the safe target is CPU `openpiv-python`.
- **Recommendation:** (1) decide base method — seriously consider cross-correlation
  PIV over Farneback, validate against synthetic ground truth; (2) write the
  displacement kernel against an array namespace (`xp`) so one path runs on
  NumPy (CPU) or a GPU backend; (3) if adding GPU, use a **PyTorch** backend
  (CUDA→MPS→CPU), optional extra, never a hard dep.

**The broader displacement-method space (2026-07-04) — it's not just Farneback vs PIV.**
The displacement step is "dense subpixel deformation field between two bead
images"; TFM has used ≥4 algorithm families. The right choice depends on the
**bead regime** (density/resolvability and displacement magnitude), per the
"Field Guide to TFM" (PMC11082129) and the retracking paper (PMC9216574):

| Family | Method(s) | Python tooling | Best when | Notes |
|---|---|---|---|---|
| **Correlation, windowed** | PIV (FFT xcorr + Gaussian peak fit) | `openpiv-python`, `TorchPIV` | dense/speckle beads, small-moderate disp | field standard; multi-pass for larger disp |
| **Correlation, subset+warp** | **DIC** (IC-GN, affine/quadratic subset shape fn) | `muDIC`, `DICe`, `DICLab2D`, `Pyvale`, `GCpu_OpticalFlow` (GPU DIC) | dense beads, smooth deformation | **higher subpixel accuracy than plain PIV**; DVC = its 3D form, used for 3D/2.5D |
| **Optical flow (differential)** | Farneback (current), **DIS**, KLT/Lucas-Kanade, TV-L1 | `cv2` (all CPU, in the pip wheel) | small/subpixel disp | KLT validated for high-res TFM (PMC5292691); **`cv2.DISOpticalFlow` is a near-free, faster, more accurate drop-in for Farneback** |
| **Particle tracking (PTV/SPT)** | detect+localize beads (radial-symmetry/Gaussian) → track → interpolate to grid | `trackpy` | **resolvable/sparse beads**; large deformation | highest *per-bead* accuracy; **cPTVR retracking handles large disp where PIV xcorr fails** (>90% vs PIV); needs scatter→grid interp before FTTC |
| **Deformable registration** | B-spline free-form deform, Demons | `SimpleITK`/`elastix`, `bUnwarpJ` | whole-field smooth warp | a *generic* smoothness prior is physically motivated (surface displacement IS elastic-constrained) but is not *the* Boussinesq prior and correlates noise vs FTTC's GCV — see model-based note below |

**Cheapest real improvement:** swap Farneback → **`cv2.DISOpticalFlow`** (already
in the pip OpenCV, CPU, faster + more accurate, no new dep, no method-family
commitment). **Highest-accuracy upgrades, regime-dependent:** if beads are
resolvable/sparse → **single-particle tracking** (`trackpy` + radial-symmetry
localization → interpolate → FTTC), best per-bead accuracy and the answer to
large deformation via retracking; if beads are dense/speckle → **DIC (IC-GN)**
over PIV/Farneback for subpixel accuracy. DIC also generalizes to **DVC** for the
3D displacement that (3) 2.5D needs — one method family could serve both the 2D
and the 2.5D displacement step.

**Model-based / one-step TFM — the principled way to couple bead physics
(owner's insight, 2026-07-04).** The beads sit in an elastic gel, so the true
surface displacement field is *not* arbitrary — it lives in the range of the
Boussinesq operator applied to some traction field, i.e. it is genuinely
elastic-constrained. That makes physics-coupled regularization a **feature**, not
the "double-counting" liability first noted. The clean realization is not
"generic deformable registration → FTTC" (two mismatched smoothers: a generic
bending-energy/diffusion prior is *some* smoothness but not *the* Boussinesq
smoothness, and it correlates the displacement noise in a way FTTC's GCV λ does
not model). It's to make the elastic Green's function itself the registration
model — solve **one** inverse problem for the traction field whose predicted
displacement best warps reference→deformed bead image. One prior, the true gel
physics. Same philosophy as the direct method (idea 3+4) and the existing BISM
engine: physics in the operator, not smoothing twice around it. Refs:
model-based TFM (Soiné et al., PMC4352062); FEM-direct (displacements as BCs →
tractions, no deconvolution); Bayesian/GP-TFM (Huang & Sabass, Sci Rep 2018,
elastic prior explicit). Schwarz review (arXiv 1506.02394): this class
"abolishes the need for regularization." **Caveat matching prior owner calls:**
the Bayesian/GP flavor sets λ by evidence/auto-tuning — that's the same
auto-parameter machinery deliberately removed with BISM's MAP auto-λ ("fragile
cleverness for a knob the user can set by hand"). Model-based/FEM-direct realizes
the physics-coupling *without* reintroducing an auto-tuned Bayesian knob — the
better fit.

**PROTOTYPE VALIDATED (2026-07-04)** — `scratchpad/proto_onestep_tfm.py`, a
self-contained synthetic recovery test on GPU (napari conda env, torch 2.10+cu128):
- **The whole pipeline works end-to-end and is differentiable.** Parametrize by
  traction `t`; forward `u = G·t` reuses **the exact Boussinesq kernel from
  `fttc.py::_calculate_greens_function`** (ported to torch); warp `I_ref` by `u`
  with `F.grid_sample`; minimize photometric MSE to `I_def`. Autograd flows
  through FFT → warp → loss, no hand-derived gradients; runs on CUDA (→ MPS/CPU
  unchanged). This is the case where torch genuinely shines (unlike Farneback).
- **It recovers traction directly — this is the "no second reconstruction".**
  `t` is the optimization variable; `u` is only an internal quantity used to
  predict the warped image, never a separate output that then gets inverted.
  At convergence you already hold the traction. Result (128², 400 beads,
  2.5 px peak displacement, 2% camera noise): **displacement 0.073 px RMSE /
  0.98 cosine, traction 0.90 cosine** vs ground truth.
- **Two honest corrections banked from the prototype:**
  (a) **Non-dimensionalize** — with real `E`≈10 kPa the target traction is
  ~hundreds and fixed-lr Adam can't travel there; solving with `E=1` (a pure
  rescale of `t`) makes it O(1) and well-conditioned. Use L-BFGS (curvature-scaled
  steps) + coarse-to-fine, not Adam.
  (b) **It still needs Tikhonov regularization on `t`** — the physics
  parametrization guarantees the displacement is *admissible* (`u ∈ range G`) but
  NOT that the traction is *stable*: high-k tractions barely move beads (Ĝ∼1/k),
  stay nearly unconstrained, and blow up (rel-L2 → 5.7 at λ≈0). A λ‖t‖² sweep
  recovers t_cos 0.90 at λ=1e-4. So "abolishes regularization" (the review's
  phrase) is really about the *model-based* variant that adds structural
  (cytoskeleton/adhesion) priors — the bare image-registration version still
  regularizes, same as FTTC. Correct my earlier overclaim.
- **BENCHMARKED on `_validation/benchmark_TFM` (2026-07-04)** —
  `scratchpad/proto_benchmark.py`, real 700² dipole pairs (low/mid/high =
  0.33/3.3/33 px peak displacement), scored with the repo's own Sabass metrics.
  Sanity check first: FTTC inverse on the **ground-truth** displacement round-trips
  to **corr 1.000** (kernel + inverse verified exactly consistent). Head-to-head,
  both from RAW IMAGES — repo TV-L1→FTTC (the shipping pipeline) vs my one-step,
  traction correlation:
  | scenario | repo TV-L1→FTTC | one-step (best λ) | one-step DTA |
  |---|---|---|---|
  | low 0.33px  | **0.931** | 0.76 | 1.0° |
  | mid 3.3px   | **0.974** | 0.88–0.96 | 0.7–0.9° |
  | high 33px   | **0.670** | 0.53 | 13.4° |
  **Honest verdict: the first-draft prototype does NOT beat the tuned pipeline.**
  Competitive at mid (0.96 vs 0.97, direction as good ~0.8°), behind at
  deep-subpixel (too little photometric signal in 0.33px) and large-motion (33px
  needs better multiscale). A clear reg(λ)/amplitude tradeoff is visible (small λ
  → right amplitude, low corr; larger λ → high corr, ~20% amplitude loss) — needs
  a principled λ selector, not a hand pick. So the value is NOT "beats FTTC on 2D
  in-plane today"; it's (a) a from-scratch joint method already in the ballpark
  with far simpler machinery, and (b) the strategic prize below.
- **Next steps, by leverage:** (1) **the real reason to pursue it** — extend the
  2×2 in-plane kernel to the 3×3 2.5D Boussinesq-Cerruti kernel (idea 3): the
  SAME solver then does 2.5D, which in-plane FTTC *cannot*. That's the capability
  FTTC doesn't have, vs competing where it's already good. (2) close the 2D gap:
  NCC/robust photometric loss (bead illumination drift), better coarse-to-fine +
  more iters for large motion, principled λ (L-curve; NOT auto-Bayesian per the
  MAP-removal call), preprocessing parity with the repo path. (3) then decide
  whether it graduates from scratchpad into `_validation/`.

### 3 + 4. 2.5D TFM and the direct method are ONE body of work  ·  L  ·  substantial
**Key research finding (2026-07-04): ideas 3 and 4 are the same paper and the
same codebase.** Blumberg & Schwarz, *"Comparison of direct and inverse methods
for 2.5D traction force microscopy,"* PLoS One 17(1):e0262773 (2022), DOI
10.1371/journal.pone.0262773 — **same group whose FTTC method this tool already
implements** — presents both a 2.5D *inverse* (FTTC) solver and the *direct*
method side by side, with open Python code at
`github.com/usschwarz/DirectMethod` (numpy/scipy/numba; owner has the 2.5D refs
already).

Both require the **same new input: a 3-component displacement field including
out-of-plane `u_z`** (z-stack beads → 3D flow — see idea 2, `OpticalFlow3d`).
That shared prerequisite is the real cost and the real blocker; once `u_z`
exists, the two solvers are cheap-ish additions:

- **2.5D inverse (extends today's FTTC).** Blumberg-Schwarz derive a *closed-form*
  2.5D Fourier kernel from Boussinesq–Cerruti potentials — a 3×3 Green's function
  `Ĝ(kx,ky,z)` (their Eq. 24) giving all three traction components (incl. normal
  `t_z`) from the 3D surface displacement, no numerical integral inversion. This
  is a **generalization of the existing 2×2 FTTC kernel in `backend/fttc.py`**
  (`_calculate_fourier_modes` / `calculate_traction_2d`) — reuses the FFT
  machinery and GCV regularization, just a bigger kernel.
- **Direct method (new, but small).** Compute the strain tensor by
  differentiating the measured 3D displacement (they recommend a 3×3×3 local
  linear-polynomial patch fit, not raw finite differences), get stress via the
  linear-elastic constitutive law, and read traction off the surface stress
  components at z=0. **No explicit regularization** — the differentiation
  filters noise inherently; their tested divergence correction did *not* help.
  Trade-off vs FTTC: direct is worse at low noise, comparable/better at high
  noise, benefits more from marker density, higher *local* SNR. Conceptually
  adjacent to the existing BISM stress path (strain→stress→traction), so it may
  share utilities there.

**Feasibility read:** the honest sequencing is (a) get 3D displacement first
(the hard part, ties to idea 2's GPU 3D flow), then (b) 2.5D-FTTC as a kernel
generalization of `fttc.py`, then (c) the direct method as a real-space
add-on — porting/adapting from `usschwarz/DirectMethod` rather than deriving
from scratch. Check its license before vendoring any code.

---

## Ranked open work (2026-06-29)

### Remove the green/red input/output-file status icons  ·  DONE (2026-07-02)
Redundant with the colormap-spine rail, which already shows per-stage status —
drop the separate green/red icons that indicate input/output file presence.
**Done.** Removed the `StageFileStatusRow` widget (the per-*artifact* red→green
dot row under each stage header) and its logic: deleted
`widgets/_stage_file_status.py`, the `FILE_STATUS_COLORS`/`file_status_color`/
`file_status_state` helpers in `_ui_style.py`, the `_build_*_specs` builders +
`_stage_status_panels_by_key` construction + `status_panel` plumbing in
`_widget.py`/`_stage_section.py`. The colormap-spine rail (`StageSpine`) and the
experiments-list rail (`MiniRail`) — the per-*stage* status nodes — are
untouched. **One coupling preserved:** the spine node's in-memory status when no
experiment is selected used to come from `panel.refresh()`; `refresh_stage_statuses`
now calls `compute_stage_status(data_manager, STAGE_DATA_ARTIFACTS[key])` directly
(kept `STAGE_DATA_ARTIFACTS` + `_stage_data_status.py` for exactly this).
**One capability removed with the dots:** clicking a red input dot was the *only*
UI trigger for "assign the active napari layer as this input"
(`load_active_layer`/`load_result_artifact`) — no button or shortcut for it
survives. In the experiments-list-driven workflow inputs load from disk on row
selection, so this was a legacy manual override; the widget/controller methods
still exist (just unreachable from the UI) if we want to re-expose them via a
dedicated control later. Tests: deleted `test_stage_file_status.py` + the
dot-routing/embedding tests in `test_workflow_shell.py` and the file-status-color
tests in `test_ui_style.py`; adapted the status-transition test to assert the
spine node (not the dots). 623 passed (full suite).

### Replace "Run all" with "Run selected"  ·  DONE (2026-07-02)
Scope batch runs to the experiments-list's existing row-selection mechanism
(the same one "Delete selected" already uses) instead of always running
every committed row. Design spec:
[`docs/superpowers/specs/2026-07-01-run-selected-design.md`](docs/superpowers/specs/2026-07-01-run-selected-design.md).
**Done as specced** — pure widget-layer change, `BatchAnalysis` untouched.
`_run_selected_experiments` (`_widget.py`) filters `experiment_records()` down
to `ExperimentsList.selected_rows()` (row order) before `build_run_config`, so
`root_folders` carries only the selected paths. The button is now
selection-driven (enabled iff `_selected_paths`, recomputed via the new
`_update_run_btn` folded into `_update_delete_btn`; no more `n > 0`), text is
"Run selected", and a new `ExperimentsList.select_all()` (Ctrl+A, committed
rows only — preview rows excluded) covers "run everything". All `run_all_*`
identifiers/signals/strings renamed to `run_selected_*` across `_widget.py`,
`_experiments_list.py`, `viewer_sink.py`, `queue_progress_sink.py`,
`batch_analysis.py`. Tests updated + added (partial-selection config, Ctrl+A
select-all, preview-exclusion, no-selection no-op) in `test_experiments_list.py`
and `test_workflow_shell.py`; 243 passed in the affected suites, 630 in the full
run (the only 5 failures are a pre-existing imageio/tifffile `fps`-kwarg env
drift in `test_batch_visualizations.py`, unrelated).

### BUG: Run All stops before finishing all queued tasks  ·  LIKELY FALSE ALARM (2026-07-02)
Per the owner (2026-07-02): probably **not** a queue/loop bug — the run most
likely crashed on **bad input images** and aborted, rather than terminating
early through a faulty loop condition. Leave un-fixed pending a real repro; if
it resurfaces, look at the failing position's images first, not the queue logic.

### BUG: Clicking the preprocessing rail circle wouldn't load its output  ·  DONE (2026-07-01)
Clicking the first (preprocessing) icon on either rail claimed "no output" even
when `preprocessed_beads.tif`/`preprocessed_reference.tif` existed on disk, and
on the mini rail the viewer was left showing raw input data instead.
**Root cause:** `_load_stage_results` (`widgets/_widget.py`) filtered every
requested stage through `_NTFM_STAGES = ("displacement", "force", "stress")` —
a stale hardcode from before preprocessing persisted its own output — so a
"preprocessing" click was silently dropped before any disk read happened. The
status dot (`_experiment_stage_status`) correctly checked for the TIFFs and
said "done"; only the click-to-load path disagreed. The mini rail's "loads
input instead" symptom was downstream of the same no-op: selecting the row
loads raw inputs, and the failed preprocessing load never overwrote them.
**Fix:** added `_apply_preprocessing_result`, a load path for preprocessing
that reads the persisted TIFFs from `experiment_output_dir` directly (no
`.ntfm`/tidy-table involved) and binds them via
`visualization_manager.begin_preprocessing_stream()` — mirrors the live
interactive-run path. `_load_stage_results` now handles preprocessing
separately from the `_NTFM_STAGES` filter instead of dropping it. Test-locked
in `test_reload_on_selection.py` (load, no-op-when-missing, and status-line
regression tests). 624 passed.

### 3b. Per-stage layer isolation during streaming  ·  DONE (2026-06-30)
The `120d0a0` machinery only ever ran on the **run-all** path (`ViewerSink`
calls `isolate_layers` per stage) — and it was provably correct there (drove the
real `VisualizationManager` through the sink; isolation held at every
transition). The gap was the **interactive per-stage Run buttons**: they stream
via `begin_*_stream` + `stream_*_frame` and **never isolated at all**, so a
direct stage run left every other layer visible ("everything bleeds in"). Only
the previews isolated, which is why preview looked right and the run didn't.
**Fix:** moved the takeover into the three streaming entry points themselves —
`begin_vector_field_stream`, `begin_stress_stream`, `begin_preprocessing_stream`
now hide non-stage layers, so both the interactive path and the sink isolate for
free. Used a new `hide_other_layers()` (hides unrelated layers but, unlike
`isolate_layers`, does **not** force-show the stage's own layers) so the
deliberate "preserve per-layer visibility across a re-run" behavior survives — a
magnitude layer the user hid stays hidden. Test-locked in
`test_analysis_streaming.py`, `test_preprocessing_streaming.py`, and
`test_preprocessing_ui_redesign.py`. **Distinct from preview** — preview still
takes over via its own end-of-render `isolate_layers`, untouched.

### 3c. Persist the preprocessed images  ·  DONE (2026-06-30)
The actual gap was **interactive-path signal wiring**, not the batch write (the
TODO's old "optional `save_cache`" diagnosis was stale — `batch_analysis.py`
already writes the preprocessed TIFFs unconditionally since `e12731c`). The
interactive persist machinery (`_widget.py::_persist_preprocessed_tiffs` + the
`preprocessing` branch of `_persist_active_experiment`) was fully written **and
test-locked** (`test_interactive_preprocessing_persists_tiffs`), but
`connect_signals()` wired `preprocessing_completed` only to `refresh()` with a
stale "nothing of its own to persist" comment — so a GUI preprocessing run wrote
nothing to disk. The unit test hid it by calling `_on_stage_persisted` directly,
never exercising the signal. **Fix:** connect `preprocessing_completed` →
`_on_stage_persisted("preprocessing")`; added
`test_preprocessing_completed_signal_persists_tiffs` driving the real signal as a
regression guard. Verified end-to-end: by the time the signal fires the streamed
arrays are filled in place in the data manager, so the persist reads real data.
Closely related to the backlog "Load processed `.ntfm` back into memory on
selection."

### 5. BISM as a selectable stress engine  ·  DONE (2026-06-30), superseded (2026-06-30)
~~Replace MSM with BISM~~ → originally **added BISM alongside MSM** behind a
**Stress Method** dropdown. **Superseded same day**: MSM was later ripped out
entirely (see below) — BISM is now the only stress engine, no dropdown.

### 6. Rip out MSM and BISM's MAP auto-λ  ·  DONE (2026-06-30)
**Both MSM and BISM's MAP machinery are fully removed** (user's call: MSM was
never coming back as a real option, and MAP was "too much fragile cleverness for
a knob the user can set by hand").

- **MSM gone**: `backend/msm.py`, `msm_numba_functions.py`, `mesh_generator.py`,
  and `_validation/benchmark_MSM/` deleted outright; `gmsh`/`solidspy` dropped
  from `pyproject.toml`. `widgets/msm_widget.py` → `widgets/stress_widget.py`
  (`MSMWidget`/`MSMController` → `StressWidget`/`StressController`), stripped of
  every MSM branch (`preview_mesh`, the mesh header glyph, the MSM half of every
  `use_bism` conditional — those calls are now unconditional BISM). The mesh-only
  `StressResult` fields (`nodes`/`elements`/`condition_number`/`residual`) are
  gone; `method` defaults to `"BISM"`. `MSMParameters` → `StressParameters`
  (mesh/material fields dropped); `MSMResult` alias gone, everything imports
  `StressResult` from `backend/stress.py` directly (which also now hosts
  `process_mask_data`, relocated out of `msm.py`). The Stress parameter-panel
  section collapsed to a flat list (`bism_regularization` + `max_stress`) — no
  more `WHEN`/`AND`-gated engine choice, since there's only one engine.
- **MAP gone**: `_estimate_lambda_map`, `noise_value_map`, `lam_method`/`use_map`
  threading, the `bism_lambda_method` field, and the **λ Method** dropdown +
  AND-gate sentinel are all deleted from `bism.py`/`parameter_dataclasses.py`/
  `_widget.py`. BISM always uses the fixed `bism_regularization` slider. The
  L-curve idea (`meth_Lambda==2`) is genuinely moot now, not just "moot per a
  note" — no auto-λ plumbing survives to extend.
- Test suite updated to match (`test_msm_analysis.py` deleted, `process_mask_data`
  test ported to `test_bism_stress.py`; MAP-specific tests in `test_bism_stress.py`
  deleted; `test_stress_ownership.py`/`test_workflow_shell.py`/
  `test_reload_on_selection.py` fixtures and dropdown tests updated/removed).
  548 passed.

### 7. visualization engine — napari live viewer + matplotlib export  ·  DONE (2026-06-30)
**Final decision (2026-06-30): the export renders with matplotlib; napari stays
the live interactive viewer only.** The original §7 goal was one renderer (napari
for both), and that was built and worked — but it forced a GL canvas, which on a
desktop pops a window, and going windowless needs a virtual display (xvfb /
subprocess), which is **Linux-only and not pip-installable**. For a tool shipping
to PLOS/JOSS users (mostly Mac/Windows) that's the wrong foundation. A
side-by-side also showed napari's "punch" was largely **additive blending
oversaturating the magnitude map** (arrow brightness summed into the colormap it's
meant to encode) — matplotlib, done well, is the cleaner *and* more faithful
publication artifact, plus vector-grade and fully portable. So the napari-export
detour (offscreen viewer → vendored movie-maker → xvfb subprocess) was torn out.
- **`backend/batch_visualizations.py`** (`BatchVisualizationSaver`) renders each
  product with matplotlib's **Agg** backend — windowless, no display, no GL, no
  xvfb — straight to **`.mp4`** (libx264/yuv420p via `imageio-ffmpeg`, which
  bundles ffmpeg on every OS; canvas padded to even dims). Same per-stage `save_*`
  surface, renders inline (no subprocess/flush). Products: displacement_map
  (viridis + white arrows), force_map (inferno + white arrows), force_cell_overlay
  (gray inverted cells + magnitude-coloured arrows), sigma_xx/yy/normal_stress
  (seismic). Sleek inline vertical colorbar mirrors the viewer's look.
- **Consistency via shared geometry, not a shared renderer.** Arrows come from the
  same `utilities/vector_field.py: build_frame_vectors`/`upscale_field` the live
  `VisualizationManager` uses (verified: same directions/scale as napari on the
  same field), with the same colormaps + contrast + arrow-scale convention. Only
  the final raster differs (Agg vs GL).
- **Mesh dropped** (user's call): FE-mesh GIF was an MSM-only diagnostic. Removed
  from `_run_config.py` + `_handle_visualization`. The **interactive** "Preview
  mesh" button (`msm_widget`/`_icons`/`_widget`) is untouched.
- **New dep:** `imageio-ffmpeg` (added to pyproject; portable, bundles ffmpeg).
  **No system deps** — xvfb is *not* required (the whole napari-export path that
  needed it was removed). `matplotlib` was already a dep.
- Test-locked in `test_batch_visualizations.py` (Agg → no display needed): shared
  vector math + every product writes an mp4 with right frame count / distinct
  frames / stress-component gating / single-frame inputs. Live-path streaming /
  run-config / batch suites green.
- **Arrow-colour default is white** on the magnitude maps (cleanest/most faithful
  per the comparison), magnitude-coloured on the gray cell overlay. The
  additive-glow look is reproducible in matplotlib (additive RGB compositing) if
  ever wanted — not the default.

### 8. Parallel batch workers  ·  L  ·  DONE (2026-07-01)
Batch config gains a **number-of-workers** spinbox (experiments-list toolbar,
1..`os.cpu_count()`, default 1 = today's unchanged behavior); positions process
in parallel, **top positions first** (FIFO submission order into the pool).
- **Workers compute, viewer follows one**, as decided: `num_workers > 1` runs
  each position headlessly (`sink=None`) on a real `ProcessPoolExecutor`
  (`spawn` context — forking a GUI process with live Qt/BLAS/OpenMP threads is
  a deadlock hazard). `start_parallel`/`poll_parallel_progress`
  (`backend/batch_analysis.py`) are the non-blocking pair a `QTimer` drives
  from `_widget.py`, so the GUI thread never freezes for the run's duration.
  The viewer follows the selected row, else the topmost folder, by reusing the
  existing "load `.ntfm` on selection" path once that position's worker
  reports done — no live cross-process frame streaming. Cancellation cancels
  only not-yet-started futures; in-flight workers finish naturally (no
  force-kill, no torn `.ntfm` writes).
- Manual row clicks during a parallel run go through the existing, unmodified
  selection path — clicking any row, finished or not, shows current disk
  truth ("scrub through existing data"), confirmed as the intended UX.
- `num_workers <= 1` (default) is the exact pre-existing synchronous/live-
  streaming path, verified byte-identical — zero regression risk for ordinary
  batch runs.
- Known, deliberately-scoped trade-offs: the per-stage progress bar doesn't
  fill frame-by-frame during a parallel run (only reconciles when the
  followed position completes or the run ends); no executor/timer teardown if
  the widget is closed mid-run (no existing teardown hook to mirror);
  `num_workers` is in-memory GUI state only, not persisted to the experiment-
  series file (unlike `disabled_stages`/`processed_root`).
- Tests: mocked-executor unit tests (`test_batch_parallel.py`) plus one real,
  unmocked `ProcessPoolExecutor` integration test
  (`test_batch_parallel_real_pool.py`) proving the actual multiprocessing
  round-trip (spawn, pickling, subprocess execution, result hand-off) works,
  not just the mocked simulation of it. 575 passed.

---

## Backlog (ranked, quick wins first)

### Adding rows to an empty list should preload the first row  ·  S  ·  DONE (2026-06-30)
When the `ExperimentsList` is empty and the user adds rows, the first added row
should be **preloaded/selected automatically** (rather than leaving the list with
no active selection). Saves a click and gives an active position for downstream
actions to target.

### Remove the export icon from the experiments list (now just a copy-file button)  ·  S  ·  DONE (2026-06-30)
The per-row Export button no longer does anything beyond copying the position's
OME-TIFF elsewhere, so it's redundant — remove the control **and its logic**:
- `widgets/_experiments_list.py`: the `export_btn` (lines ~233-243), the
  `_EXPORT_W` spacer column (`export` placeholder at ~992-994), the
  `export_requested` signals (row + list, ~191/303) and their re-emit
  (~1011), and the enable-on-done line (~277).
- `widgets/_widget.py`: the `export_requested.connect(...)` wiring (~565) and the
  `_on_export_experiment_data` handler (~1295).
- Drop the `"export"` entry from `stage_action_icon` if nothing else uses it, and
  any test that asserts `experiment_row_export_button`.

### Polish the colorbar legend  ·  S  ·  DONE (2026-06-30)
Spacing, label alignment, endpoint-number placement — the `viewer_colorbar.py`
knobs `COLORBAR_HEIGHT_FRACTION` / `LABEL_INSET_FRACTION` are the levers. Pure
visual tuning, no plumbing.

### Output results to `TFM_data/` next to the input, not `processed/`  ·  S/M  ·  DONE (2026-06-30)
TFM results should **not** land in a folder called `processed`. They should go
in a folder called `TFM_data` sitting **right next to the input data** — i.e. as
a sibling of the input's containing folder. When an output-folder variable is
set, the input folder structure is cloned into that output directory, and the
`TFM_data` folder lives where the input data *would* be inside that cloned tree.
Also **rename the artifact**: the single multi-series OME-TIFF holding all the
results should be called `TFM_results.ome.tif` — not `<experiment_name>.ome.tif`
(currently `batch_output.py::experiment_output_path`, line ~104).

### Apply-mask-on-save option in the batch config  ·  M  ·  DONE (2026-06-30)
Opt-in `apply_mask_on_save` flag: when a mask layer is present, **zero every map
pixel where the mask is background (label 0)** before writing the `.ntfm` —
`u_x, u_y, F_x, F_y` (and stress, if present) set to `0.0` wherever `mask == 0`.
The `mask` column records which pixels were zeroed (self-documenting).
- **Why.** Off-cell substrate signal is noise; zeroing it cleans the field and
  compresses ~8× (measured `Ctrl/pos_00`, 8.3% on-cell: 177 MB → 20.5 MB). Long
  runs of exact zeros crush under snappy/zstd.
- **Scope.** Opt-in, default **off** (deliberately lossy — background values are
  discarded irreversibly; only the `mask` column survives). No-op without a mask.
- **Where.** Batch write step (`backend/batch_analysis.py` →
  `ntfm.results_to_ntfm`), a pre-write array op. Interactive/preview unaffected.

### Preprocessing param-panel layout cleanup  ·  M  ·  DONE (2026-06-30)
Tidy the preprocessing widget (`napariTFM/widgets/preprocessing_widget.py`):
- **Shorten the double sliders** a bit (they're wider than they need to be).
- **Remove rolling-ball radius** entirely — both the front-end control and the
  back-end parameter/usage.
- **Regroup the params** into rows: 1 row Intensity, 1 row Cell Intensity, 1 row
  sigma + cell sigma side by side, 1 row registration method.
- **Reorder the input-file rows** to: bead stack (top), reference stack, cell
  stack, Masks. **Rename the layers** to `Beads`, `Reference`, `Cells`, `Masks`.

### Dedup preprocessed-TIFF persistence (batch vs. interactive)  ·  M  ·  DONE (2026-06-30)
The preprocessed-image save lives as **two independent implementations** that
only share the low-level `save_calibrated_tiff` helper:
`backend/batch_analysis.py::_execute_preprocessing` (lines ~724-742) and
`widgets/_widget.py::_persist_preprocessed_tiffs`. Parity is currently held by
hand — which is exactly how the §3c bug happened (one path was wired, the other
sat dead). Collapse the interactive path so it calls into the batch's
preprocessing-save orchestration rather than reimplementing it, so there's one
place that knows how a position's preprocessed TIFFs get written. Pure tidy-up,
no behaviour change.

### Load processed `.ntfm` back into memory on selection  ·  M  ·  DONE (2026-06-30)
Follow-up from the "stage runners weren't saving" fix: selecting an
already-processed experiment reads "done" from disk, but the viewer layers stay
empty until a stage re-runs. On selection, **load its `.ntfm` back into memory**
so the viewer shows the stored result.

### Make preview toggle-vs-one-shot legible in the icons  ·  M  ·  DONE (2026-06-30)
Preview is inconsistent across stages: some stages **toggle** preview (on/off,
persistent state) while others fire it as a **one-shot** action. The icons don't
distinguish the two, so the control's behavior isn't predictable from looking at
it. Make toggle-style previews **render as toggles** in the icon set (a
pressed/active state that reflects the on/off), distinct from the one-shot
(momentary action) icons — so the UI tells the user which kind of preview each
stage offers before they click.

### Progressive per-stage loading bar  ·  L (scope first)  ·  DONE (2026-06-30)
For both **live** mode and **batch** mode: each status circle/node should **fill
up progressively** as its stage runs (not just flip empty→done). One
implementation that serves both modes. **Investigate complexity first** —
driving a smooth per-stage fill needs intra-stage progress signals from the
pipeline (the sink currently emits stage-level start/finish, not fractional
progress), so scope what granularity is actually available before committing to
a design. Biggest unknown in the backlog — do not start without scoping.