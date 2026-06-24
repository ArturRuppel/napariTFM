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
| **1** | §1 `.ntfm` + tidy converter       | Foundation — every data-shaped item below consumes it          | ✅ built |
| **2** | §4 Batch-only data production      | Writes `.ntfm`; needs the container to exist                   | ✅ done — shared param object *is* the bridge; labels moved to §5 |
| **3** | §3 Toolbar UI                     | Final shape depends on which buttons §4 leaves behind          | ✅ done — all `TODO.md` slices landed (see §3) |
| **4** | §5 Aggregator → `.iris`           | Consumes `.ntfm` series + structured batch output              | 🔵 designed |

**Parallelizable now:** §2 and the in-flight `TODO.md` UI-coherence slices both
run independently of the format work — start them without waiting on §1.

**Critical path:** §1 is the keystone. §4 and §5 are blocked on it; §3's final
form is blocked on §4. The `.ntfm`/tidy format is now **built** (single canonical
grid, Parquet payload + JSON sidecar) in `napariTFM/utilities/ntfm.py` with a
lossless round-trip converter — §4 can now wire batch output to it. Only the
`.npy` migration path remains open.

---

## §1 — Output formats: tidy tables + a native `.ntfm` container  ✅ built

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
| `F_x[Pa]`      | measure | traction stress, x                        |
| `F_y[Pa]`      | measure | traction stress, y                        |
| `sigma_xx[mN/m]`    | measure | normal stress, x                     |
| `sigma_yy[mN/m]`    | measure | normal stress, y                     |
| `sigma_shear[mN/m]` | measure | shear stress (σxy = σyx, symmetric)  |
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
- Stress is three independent components (MSM computes only σxx, σyy, σxy; the
  Cauchy tensor is symmetric). Stored as `sigma_xx`, `sigma_yy`, and a single
  `sigma_shear` (= σxy = σyx) — no redundant `sigma_yx` column.
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

`metadata.json` holds **run-level provenance + the exact config** — no
cherry-picked constants; the config *is* the source of truth for parameters, and
everything per-sample (coords, indices, units) is already in the table:
- `format_version` — the `.ntfm` schema version, independent of code version
- **code provenance** (reproducibility):
  - `git_commit` — full hash of the analysis code at run time
  - `git_dirty` — `true` if the worktree had uncommitted changes (the run is
    *not* fully reproducible from the commit alone); optionally store the diff
    when dirty
  - `package_version` — human-readable napariTFM version
- **`config`** — the full, **resolved per-experiment** run config (the effective
  `UnifiedParameters` actually applied to *this* experiment: pixel size, dt,
  gel/substrate params, `downscale_factor`, every per-stage parameter). For a
  batch run, embed the resolved config for this experiment, **not** the whole
  sweep — each `.ntfm` is reproducible standalone. Supersedes any separately
  listed constants.
- **inputs** — source file(s) / channel provenance
- **`labels`** (added by §4) — free-form experiment-design tags
  (`{condition, replicate, position, …}`) the aggregator groups by; supplied
  per-experiment in the batch config. Not yet in the built writer.

The **grid descriptor is not stored** — it's derivable from `config`
(`spacing = pixel_size · downscale_factor`) plus the table (`row`/`col`, shape).
Config and table are the two sources of truth.

This retires the scattered per-artifact `.npy` files (`displacement_results
.npy`, `force_results.npy`, `stress_results.npy` in
`utilities/data_manager.py`); one `.ntfm` per experiment is the unit users move.

**Implemented** (`napariTFM/utilities/ntfm.py`, tested in
`tests/test_ntfm_format.py`):
- `arrays_to_tidy` / `tidy_to_arrays` — the lossless round-trip converter (one
  canonical grid; `row`/`col` explicit; symmetric stress stored once as
  `sigma_shear`; absent measures → NaN columns, missing columns → `None` on read).
- `write_ntfm` / `read_ntfm` — the zip container (`samples.parquet` +
  `metadata.json`); `write_csv` — the lossy CSV exporter.
- `build_metadata` / `git_provenance` / `package_version` — run-level provenance
  plus the resolved `config`; grid descriptor intentionally not stored.
- `dataframe_from_results` / `results_to_ntfm` — adapter from the pipeline's
  `DisplacementResult` / `FTTCResult` / `MSMResult` + `UnifiedParameters`.

Traction is stored in **Pa** (`F_x[Pa]`, `F_y[Pa]`) — the native pipeline unit;
the earlier `nN` label was a schema error.

**Remaining open item** ⚪
- **Migration.** One-shot `.npy` → `.ntfm` converter, or a transparent
  read-compat shim for existing projects? (Lower-stakes now the schema is fixed.)
- **Wiring.** ✅ Batch writes one `.ntfm` per experiment (§4 backend) and it is now
  the *sole* persisted result: the interactive path is **preview-only** (writes
  nothing; `DataManager` availability follows the in-memory value) and batch no
  longer caches the scattered `.npy` files — stage-resume reads displacement/force
  back from the `.ntfm`. Only the preprocessed `.tif` survives as an opt-in cache.

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

## §3 — UI: toolbar-style stages, not button walls  ✅ done (Phase 3)

**Implemented.** All four `TODO.md` slices landed (commits `8f0afe6`, `0d5ce79`,
`fd6a358`, `e95074b`, `2e70f5a`): inner scroll areas + the 360px lock removed;
`StageSection` rebuilt on CellFlow's `CollapsibleSection` with a glyph-pill header
(🔍 files · ⚙ params · ▷ preview · ▶/■ run), no status dot; the `_ActionStateSync`
proxy machinery retired for a single-instance signal-driven action model; and the
`section_grid` vocabulary now backs both `WorkflowParameterPanel` and the Project
px/dt controls (flat, two-up, no `QGroupBox`). The duplicated per-stage load/save
buttons are gone — loading consolidated when §4 made the interactive path
preview-only. Locked by `tests/test_stage_section_header.py`,
`tests/test_ui_style.py`, `tests/test_preprocessing_ui_redesign.py`,
`tests/test_project_section.py`, `tests/test_collapsible_section.py`.

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

## §4 — Enforce structured I/O: batch is the only path to real data  🟡 (Phase 2)

**Backend wired.** Batch now produces the structured artifact: one
`<experiment>.ntfm` per experiment, written into the `processed/` bucket with the
mirroring rules below (`napariTFM/utilities/batch_output.py`), carrying the
resolved per-experiment `config`, `inputs`, and per-experiment `labels`
(ROADMAP §4 → §5). `figures/` and `batch.log` join it in the same folder; the old
`.npy`/`.tif` files survive only as a stage-resume cache. The
`metrics_results.csv` batch output is **retired** (derived metrics are §5's job).
Tested in `tests/test_batch_output.py` + `tests/test_batch_ntfm_output.py`.

**Preview-only — done.** Per-stage interactive buttons now write nothing to disk:
results live in memory and chain displacement→force→stress in memory; the Save
buttons and the prior-stage `.npy` Load buttons are gone (the mask Load stays —
it's an external input). `.ntfm` is the only persisted artifact; batch caches
just the opt-in preprocessed `.tif`, and stage-resume reads results back from the
`.ntfm`.

**Bridge resolved — shared parameter object.** The interactive stage widgets and
the batch widget hold the **same** `ParameterManager` instance
(`widgets/_widget.py` constructs one and passes it to every stage *and* to the
batch widget). Tuning a stage mutates exactly the object batch serializes via
`get_all_parameters()`, and each `.ntfm` already stamps the resolved
`asdict(UnifiedParameters)`. So "tune → commit → run" is realized by the shared
object — no separate commit step or duplicate param set. (A future explicit
*snapshot/freeze* affordance is optional polish, folded into §3's redesign.)

**Labels relocated to §5.** Experiment-design tags
(`condition`/`replicate`/`position`) are **no longer** entered in the batch UI or
baked into each `.ntfm`. They are assigned in the **aggregator** (§5), where the
grouping is actually used, and persisted in the `.iris`. The `mask` *id* (region
label `1..N`) is a separate thing — it comes straight from the mask file and is
already a `.ntfm` column. (The batch backend's `config['labels']` read is now
dead/optional; removable when §5 lands.)

**Decision.** Separate **parameter tuning** from **data production**.

- **Batch analysis is the only way to produce persisted data.** Its output is
  structured, organized, and the authoritative artifact — one `.ntfm` (§1) per
  experiment.
- The **per-stage action buttons** become **preview-only**: they run a stage and
  *display* results in napari for parameter tuning, but write **nothing** to
  disk.
- This kills the ambiguity where ad-hoc single-stage runs and batch runs both
  yield "data" in inconsistent layouts.

### Tune → commit → run bridge

The link between §3's interactive UI and batch: the interactive session holds
the working `UnifiedParameters`. A **"commit parameters"** action serializes
them into a batch config; batch runs that config over the experiment list and
stamps the resolved params into each `.ntfm`'s `config` block (§1). Tuning and
production share one parameter object — tune on a representative experiment,
commit, batch reuses exactly those numbers.

### Input — unchanged

Batch input keeps the current model: a **list of folders** + a **filename per
input** (beads / reference / cells / `masks.tif`). One input folder = one time
series = one `.ntfm`.

### Output — a mirrored `processed/` bucket

All derived output goes into a bucket named **`processed/`**, never mixed with
raw inputs:

- **Processed root configured** → the input folder tree is **mirrored** under it;
  each experiment's derived files land in the mirrored folder.
- **Processed root empty → in-place**: a `processed/` subfolder is created
  **inside each input folder**, so derived files still never mix with raw.

```
processed root SET:                 processed root EMPTY (in-place):
<processed_root>/                   <input_folder>/
  exp_A/                              beads.tif, reference.tif, masks.tif  (raw)
    exp_A.ntfm                        processed/
    figures/                            <input_folder>.ntfm
    batch.log                           figures/
  exp_B/ …                             batch.log
```

Each experiment's output folder holds: **`<experiment>.ntfm`** (the sole data
artifact), **`figures/`** (per-stage PNG/GIF previews — human-facing, not data),
and **`batch.log`** (run log). Preprocessed `.tif`s, if retained, are an optional
compute cache for stage-resume — not deliverables.

**Mirroring base** 🔵 — the mirror base is the **longest common parent** of all
input folders; each folder's path is reproduced relative to that base under the
processed root. When the folders are **genuinely disconnected** (no shared
parent — e.g. one on `C:`, another on `D:`), **emit a warning and fall back to
folder basenames** flat under the processed root. Single folder likewise falls
back to its basename. (Basename collisions across disconnected roots are the
warning's job to surface.)

### Derived metrics are not batch output

Strain energy, polarization, and all aggregate/derived quantities are **not**
cached in `.ntfm` and **not** written as batch CSVs — they are computed by the
aggregator (§5) and live in `.iris` (with optional CSV export from there). The
old `metrics_results.csv` batch output is **retired**; `.ntfm` stays strictly
single-grain (per-sample). Derived = downstream, always.

### Series organization — labels assigned in the aggregator

The experimental *design* (which experiment is which condition / replicate /
position) is the new info the aggregator needs. **Decision (revised): assign it
in the aggregator (§5), not the batch.** When the aggregator scans a tree of
`.ntfm`s, the user tags/groups experiments there — that is where the grouping is
consumed, so it belongs there. The mapping is persisted in the exported `.iris`
(and optionally a regenerable `series.json` cache), never in a central registry.

This deliberately trades the earlier "self-contained `.ntfm`" goal (a `.ntfm`
alone won't state its condition) for a far simpler batch UI — no per-folder
labels grid. The `.ntfm` still carries everything *intrinsic* (per-sample data,
resolved config, provenance); only the extrinsic experiment-design tags move
downstream to where they're used.

**Why Phase 2:** depends on §1's container (now built); reframes the workflow §3
then presents.

---

## §5 — Aggregator: browse a TFM series, export to `.iris`  🔵 designed (Phase 4)

**Decision.** Add an **aggregator** one level above a single analysis run: it
ingests a **series** (a tree of `.ntfm` files from §4), reduces them to a tidy
summary table, lets the user browse it, and exports a ready-to-render `.iris`
document.

**What `.iris` actually is** (pinned against `Iris/engine/iris_engine/
document.py`, format `2.0`). A `.iris` is **not** pre-rendered images — it is the
**same container shape as `.ntfm`**: a ZIP of a Parquet table + JSON sidecars,
from which the Iris engine (scipy / statsmodels / pingouin + matplotlib vector
output) re-renders figures and citable stats. Derived stats fields are *dropped*
on save and recomputed on open, so the stored file is purely declarative intent.
Contents:

```
document.iris  (zip)
├── manifest.json          # format_version, modified, engine identity + snapshot
├── data/table.parquet     # the summary table (rows)
├── data/schema.json       # columns: [{name, type, label, unit?, levels?}]
│                          #   type ∈ identifier | categorical | numeric
├── analyses/NN-<id>.json  # one grammar-of-graphics spec per premade analysis
└── provenance.json        # where the data came from
```

**The aggregator is a reduction layer.** Each `.ntfm` is per-sample `(t,y,x)` —
far too granular for figures/stats. The aggregator groups each `.ntfm`'s tidy
table by **`(region = mask label, frame = t)`** and computes per-region-per-frame
**derived scalar metrics** (the quantities §4 deferred). **Metric set = exactly
what `metrics_calculator.py` already computes** — no new physics for now:
- `total_strain_energy` (over the region) — `calculate_total_strain_energy`
- `polarization_index`, `lambda1`, `lambda2` — `calculate_polarization` of the
  `calculate_moment_tensor` result

(Extensible later — stress/traction aggregates can be added when needed.)
Stacking all experiments yields one long **summary table**:

> **grain = one row per `(experiment_id, region_id, frame)`** — the
> "regions-by-frame" analog of Iris's own `cells_by_frame` sample.

Each mask label `1..N` is a **distinct cell**, so `region_id` is a meaningful
biological identifier (per-cell), not a connected-component artifact.

Columns map to Iris schema types:
- **identifier** — `experiment_id`, `region_id` (cell), `frame`
- **categorical** — experiment-design tags (`condition`, `replicate`,
  `position`), **assigned in the aggregator UI** (no longer carried in the
  `.ntfm`); promoted one column per label key
- **numeric** — every derived metric (with `unit` + `label`)

**Let Iris do the statistical reduction, not the aggregator.** Iris aggregates
across a grain via the **data hierarchy `spine`** (the coarsest spine level is
the inferential unit — `reduce.py`). So the aggregator emits the *fine*
(per-cell-per-frame) grain and each premade spec sets the spine to the
**replicate unit** (`experiment_id` by default, or a `replicate` label if
biological replicates span experiments). Cells within one experiment are
**pseudo-replicates**, so the spine ensures tests treat each experiment/replicate
as **n = 1**, not each cell or pixel — statistically honest by construction.

**Premade analyses** — ship the authored grammar-of-graphics specs backed by the
current metric set (`spec_version` `2.1`; shape: `{encodings, hierarchy:{spine,
fn}, layers:[{geom}], stats:{family,test,alpha}}`):
1. Strain energy by condition — box/swarm, `group_comparison` (Welch *t* / ANOVA)
2. Polarization index by condition — box/swarm, `group_comparison`
3. Strain-energy **time course** per condition — line over `frame`, per-replicate mean

Each carries `spine: ["experiment_id"]` and `alpha: 0.05`. Iris's `specnorm`
normalizes legacy specs, so loose version coupling is safe.

**Browse + export.** The aggregator UI loads a series, shows the summary table,
lets the user filter/inspect and pick which premade analyses to include, then
**writes the `.iris` zip directly** (the format is small and inspectable — no
hard dependency on the Iris engine package). Optional CSV export of the summary
table. `provenance.json` records the contributing `.ntfm` paths + their
`git_commit`/`config`, plus the napariTFM producer version.

**Resolved**
- **Manifest `engine` identity** 🔵 — napariTFM writes its producer identity into
  `provenance.json` and a minimal manifest; Iris **re-stamps** its
  `build_identity()` on first open/save. (Still worth a one-time round-trip check
  on the Iris side, but the approach is settled.)
- **Metric set** 🔵 — for now, exactly the existing `metrics_calculator.py`
  outputs: `total_strain_energy`, `polarization_index`, `lambda1`, `lambda2`.
  Extensible later.
- **Multi-region semantics** 🔵 — mask labels `1..N` are **distinct cells**;
  `region_id` is a real per-cell identifier (nested under `experiment_id` for
  stats).

**Remaining risk** ⚪
- **Iris format is moving.** Pinned to format `2.0` / spec `2.1` today, but an
  active redesign is in flight (`Iris/docs/superpowers/specs/
  2026-06-24-iris-file-format-redesign-design.md`). Re-verify the container +
  spec grammar against Iris **at build time** — not a design blocker now.

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
