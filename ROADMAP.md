# napariTFM Roadmap — Design Decisions

Forward-looking design decisions for napariTFM. These are direction-setting,
not yet scheduled implementation. UI *coherence* work (the CellFlow → napariTFM
port) is tracked separately in [`TODO.md`](TODO.md) and the
`docs/superpowers/plans/` slices; this document captures the larger structural
moves that change the data model, scope, and workflow.

> Status legend: 🔵 decided · 🟡 needs design spike · ⚪ open question

---

## Execution order

The five decisions form a near-linear chain. Recommended sequence:

| Phase | Item                              | Why here                                                        | Status |
|-------|-----------------------------------|----------------------------------------------------------------|--------|
| **0** | §2 Drop mask creation             | Cheap deletion; de-scopes batch + the `.ntfm` mask column      | ✅ done |
| **1** | §1 `.ntfm` + tidy converter       | Foundation — every data-shaped item below consumes it          | 🔵 designed |
| **2** | §4 Batch-only data production      | Writes `.ntfm`; needs the container to exist                   | 🔵 |
| **3** | §3 Toolbar UI                     | Final shape depends on which buttons §4 leaves behind          | 🔵 |
| **4** | §5 Aggregator → `.iris`           | Consumes `.ntfm` series + structured batch output              | 🟡 |

**Parallelizable now:** §2 and the in-flight `TODO.md` UI-coherence slices both
run independently of the format work — start them without waiting on §1.

**Critical path:** §1 is the keystone. §4 and §5 are blocked on it; §3's final
form is blocked on §4. The `.ntfm`/tidy format is now designed (single canonical
grid, Parquet payload + JSON sidecar) — only the `.npy` migration path is open.

---

## §1 — Output formats: tidy tables + a native `.ntfm` container  🔵 designed

**Decision.** A single **tidy, long-format table** is the canonical data
representation, with one row per `(t, y, x)` grid sample. **Everything lives in
the table in physical units, and the unit is carried inline in the column name**
(the `name[unit]` convention) so the table is fully self-describing and the
sidecar is never needed for analysis. Columns:

| column         | role    | meaning                                   |
|----------------|---------|-------------------------------------------|
| `t[min]`       | id      | time                                      |
| `y[µm]`        | id      | sample y position (physical)              |
| `x[µm]`        | id      | sample x position (physical)              |
| `row`          | id      | grid row index (int) — for array pivot    |
| `col`          | id      | grid col index (int) — for array pivot    |
| `u_x[µm]`      | measure | displacement, x                           |
| `u_y[µm]`      | measure | displacement, y                           |
| `F_x[nN]`      | measure | traction force, x                         |
| `F_y[nN]`      | measure | traction force, y                         |
| `sigma_xx[mN/m]` | measure | stress tensor component                 |
| `sigma_yy[mN/m]` | measure | stress tensor component                 |
| `sigma_xy[mN/m]` | measure | stress tensor component                 |
| `sigma_yx[mN/m]` | measure | stress (= `sigma_xy`, symmetric; derived)|
| `mask`         | measure | region label: 0 = background, 1..N = ROI  |

`(t, y, x)` are **identifiers** (Iris vocabulary), the rest are **measures** —
the table is the full 2D field, one frame = all rows sharing a `t`. No summary,
no loss.

**The table is self-contained for analysis.** Per-sample data — coordinates,
indices, `mask`, and every measure — are all columns, so analysis never touches
the sidecar: `df[df['mask']==1].groupby('t[min]').mean()` just works. `row`/`col`
are kept as explicit integer columns so the table `pivot`s straight back to a 2D
array without recomputing indices from physical µm. The **only** thing in the
sidecar is the handful of experiment-wide *constants* (grid origin/spacing/shape,
processing params, provenance) — values identical on every row, which would be
pure repetition as columns and buy analysis nothing.

**Why this is lossless (verified against the code).** Displacement, force, and
stress are all co-registered on one **downscaled analysis grid**: force inherits
the displacement-field grid (`fttc.py` `force_shape = displacement_field
.shape[1:4]`), and MSM interpolates nodal stresses back onto that same grid
(`msm.py` `_interpolate_stress_field`), both reporting `grid_spacing =
pixel_size · downscale_factor`. The triangular FE mesh is an internal MSM compute
detail and is **not** persisted, so there is no mesh topology or multi-grid
problem — one grid holds everything.

**Notes baked into the schema**
- `u_x, u_y` are included (primary measurement, same grid) so the native
  container round-trips fully. The CSV/Iris export may drop them if only
  results are wanted.
- `sigma_yx[mN/m]` is a symmetric duplicate of `sigma_xy` (MSM computes only
  σxx, σyy, σxy). Kept for explicitness as a `derived` column.
- Off-mask grid nodes still emit rows: valid `u_*`/`F_*`, `NaN` stress,
  `mask = 0`. NaN measures are normal in long format — no special-casing.

**Serialization: Parquet canonical, CSV exporter.** Parquet is the canonical
on-disk encoding (preserves dtypes, NaN, compression, fast columnar reads). One
writer builds the Arrow table; CSV is an additional lossy export for
human/Excel/portability use. The same long table feeds the **Iris** handoff
(§5) — Iris consumes a typed data table directly.

**Converter / round-trip.** Build one converter between this tidy table and the
in-memory native representation so **both import and export** run through one
code path. Round-trip constraint: native → tidy → native is lossless for every
column above; array reconstruction is trivial since `row`/`col` are explicit
columns (no need to recompute indices from physical µm). **Import is
metadata-lossy and that is acceptable** — an external tidy table lacks
acquisition/parameter metadata; populate what's present, leave the rest unset,
never fail.

**Native container — `.ntfm` = a `.zip`:**

```
experiment.ntfm  (zip)
├── samples.parquet   # the tidy table above
└── metadata.json     # everything needed to reconstruct + interpret
```

`metadata.json` holds **only experiment-wide constants** (everything per-sample,
including units, is already in the table):
- `format_version`, `software_version`
- **grid**: `origin_um [x0, y0]`, `spacing_um`, `shape [ny, nx]`,
  `downscale_factor` — the grid descriptor (redundant with `x/y/row/col` but
  explicit; cheap insurance if rows are ever filtered)
- `pixel_size_um`, `dt_min`
- gel/substrate params (Young's modulus, Poisson ratio, gel height, …)
- full per-stage parameter snapshot (`UnifiedParameters`: preprocessing /
  displacement / FTTC / MSM)
- source-file / channel provenance
- `derived` column flags (e.g. `sigma_yx`)

This retires the scattered per-artifact `.npy` files (`displacement_results
.npy`, `force_results.npy`, `stress_results.npy` in
`utilities/data_manager.py`); one `.ntfm` per experiment is the unit users move.

**Remaining open item** ⚪
- **Migration.** One-shot `.npy` → `.ntfm` converter, or a transparent
  read-compat shim for existing projects? (Lower-stakes now the schema is fixed.)

---

## §2 — Drop mask creation entirely  ✅ done (Phase 0)

**Decision.** Mask *creation* is **out of scope** for napariTFM. It is scope
creep against the tool's core (displacement → force → stress).

- Masks must be **provided externally** as an input layer.
- Point users to **CellFlow** (the sibling project) for general segmentation
  tooling when they need to produce masks.
- Remove the in-plugin mask-generation step from the workflow and from batch
  analysis (`backend/batch_analysis.py` "Mask Creation" stage). Mesh generation
  (`mesh_generator.generate_mesh`) and MSM continue to *consume* a supplied
  binary mask — that stays; only the step that *generates* masks from images
  goes away.

**Implication.** The stress/MSM stage gains a hard precondition: "mask layer
required." UI and batch should validate presence of an external mask up front
rather than silently creating one. This also simplifies §1's `mask` column —
it's a pure external input, never a napariTFM-produced artifact.

**Why Phase 0:** it's a deletion, low-risk, independent of everything, and it
shrinks the surface §1 and §4 have to cover. Do it first.

---

## §3 — UI: toolbar-style stages, not button walls  🔵 (Phase 3)

**Decision.** Collapse each stage's control surface to a **thin line with a
small set of action buttons** — a toolbar idiom — instead of the current
stacked panels of buttons and ugly per-stage "load" buttons everywhere.

- Every stage (preprocessing, displacement, force/FTTC, stress/MSM) renders as
  one compact row: stage name + status + a few inline action buttons.
- Eliminate the duplicated/ad-hoc **load buttons** scattered across stages;
  loading becomes a single, structured entry point (see §4).
- The `TODO.md` coherence slices (single section primitive, controls built once,
  shared grid vocabulary) are the plumbing toward this; they run in parallel.

**Why Phase 3 (after §4):** §4 decides which actions survive (per-stage buttons
become preview-only, loading consolidates), so the toolbar's *final* button set
is determined by §4. The TODO.md plumbing doesn't have to wait, but the final
toolbar shape does.

---

## §4 — Enforce structured I/O: batch is the only path to real data  🔵 (Phase 2)

**Decision.** Separate **parameter tuning** from **data production**.

- **Batch analysis becomes the only way to produce persisted, real output
  data.** Its output is structured, organized, and the authoritative artifact —
  written as `.ntfm` (§1).
- The **intermediate per-stage action buttons** become **preview-only**: they
  run a stage and *display* results for parameter tuning, but do **not** write
  canonical data to disk.
- This removes the current ambiguity where ad-hoc single-stage runs and batch
  runs both yield "data" in inconsistent layouts.

**Workflow framing.** *Tune interactively on a representative sample → commit
the tuned parameters → run batch to generate the organized dataset.* Output
organization (directory layout, naming, the `.ntfm` container) is owned by the
batch path.

**Open question** ⚪
- On-disk organization for a batch run of many positions/series — one `.ntfm`
  per series vs. a parent index. Coordinate with §1 and §5.

**Why Phase 2:** depends on §1's container existing; reframes the workflow §3
then presents.

---

## §5 — Aggregator: browse a TFM series, export to `.iris`  🟡 (Phase 4)

**Decision.** Add an **aggregator** that operates one level above a single
analysis run.

- **Input:** the structured batch outputs from §4 — many runs / positions /
  conditions across a **whole TFM series experiment** — consuming `.ntfm`
  series directly.
- **Browse:** let the user navigate and inspect the aggregated dataset across
  the series.
- **Export to `.iris`:** emit **`.iris` documents** for the Iris app
  (`~/Projects/Iris`).

**What `.iris` actually is** (verified against the Iris repo). A `.iris` file is
**not** a bundle of pre-rendered images — it is a **declarative analysis spec**:
a typed data table + a grammar-of-graphics figure spec + an optional statistical
test, from which the Iris engine (scipy / statsmodels / pingouin + matplotlib
vector output) re-renders publication figures and citable statistics. Every
computed value is a function of the spec. This aligns cleanly with §1: Iris
consumes a typed table, which is exactly the tidy long-format export.

So the aggregator's job is **not** "render plots" — it is:
1. Collect the series into one typed data table (the tidy format, aggregated /
   reshaped to the analysis grain — per-cell, per-condition, per-timepoint).
2. Attach a **pre-authored declarative spec**: column → plot mappings and chosen
   stat tests for the standard TFM summaries ("premade analyses").
3. Write the table + spec as a `.iris` document; Iris renders it.

**Open questions** 🟡
- Pin the exact `.iris` document schema against the Iris engine's save/load
  format (`engine/iris_engine/` — `build_info.py`, `render.py`, the
  `/document/save` contract) before authoring specs.
- Which aggregate summaries ship as the "premade" spec set (per-condition
  force/stress summaries, time-course curves, distributions).
- Aggregation grain — what one row of the exported table represents.

---

## Dependency graph

```
§2 (drop masks) ──┐
                  ├─► §1 (.ntfm + tidy) ──► §4 (batch-only) ──► §3 (toolbar UI)
TODO.md UI slices ┘                    └──► §5 (aggregator → .iris)
                                              (also needs §4 output)
```

§1 is the keystone. §2 and the TODO.md UI slices are the only work that can
start before it.
