# TFM validation benchmark (analytic Hertz dipoles)

Frozen ground-truth data plus `validate_TFM.py`, which runs napariTFM's displacement and FTTC
backends against it and writes the figures in this directory. Not part of the test suite:
`pyproject.toml` sets `norecursedirs = ["_dev", "_validation"]`, so this is a manual check.

## What the data is

Two counteracting tangential force dipoles on an elastic half-space, evaluated **analytically**
(no simulation, no solver), so the ground truth carries no discretization error of its own.

| | |
|---|---|
| grid | 700 x 700 at `spacing_xy` = 0.1 µm/px |
| substrate | `E` = 20000 Pa, `nu` = 0.5 |
| adhesions | two dipoles at `phi` = +45 deg and -45 deg, both centred at [0, 0] |
| geometry | `d` = 50 µm between counteracting sites, `a` = 3 µm site radius |

The three tiers are a pure **force ladder**; nothing else differs between them:

| tier | `F` per site (µN) | max \|u\| (µm) | max \|u\| (px) | regime |
|---|---|---|---|---|
| `low` | 0.0025 | 0.023 | 0.23 | deep sub-pixel |
| `mid` | 0.025 | 0.234 | 2.3 | few-pixel |
| `high` | 0.25 | 2.342 | 23 | tens-of-pixels |

Each tier holds `dipole_config.toml` (the spec), `displacement_x/y.npy` and `traction_x/y.npy`
(ground truth, µm and Pa), and `reference.tif` / `deformed.tif` (the image pair).

**`dipole_config.toml` is the authoritative record.** It is sufficient to regenerate a tier
against upstream DirectMethod, and it is what to trust if anything below drifts.

## How it was produced

The generating script (`generate_benchmarks_code/generate_benchmark_data.py`) was removed on
2026-07-16; see below. What it did, recorded here so the arrays are not orphaned:

1. Read the tier's `dipole_config.toml` through DirectMethod's `loadDataDescription`,
   `loadSimulationData` and `loadAdheasionSites`.
2. Built analytic displacement and traction fields from DirectMethod's Hertz pattern
   (`inputSim.fields.HertzBuilder.get_u_hertz_pattern` / `get_q_hertz_pattern`), with
   `mu = E / (2 * (1 + nu))`, sampled on a grid centred on the origin spanning
   `n_points_xy * spacing_xy` µm.
3. Saved `displacement_x/y.npy` (µm) and `traction_x/y.npy` (Pa). The out-of-plane components
   were computed but never saved: this is a 2D benchmark.
4. Made `deformed.tif` by warping `reference.tif` with
   `coord_map = [row - u_y/px, col - u_x/px]` via `skimage.transform.warp(..., mode="constant")`,
   renormalised to uint16. That is the **`dfm(q) = ref(q - u)`** convention, the same one
   benchmarkTFM pins with its known-translation smoke tests.

Upstream: DirectMethod by Usschwarz (https://github.com/usschwarz/DirectMethod, MIT), already
cited in the top-level README for the FTTC implementation.

## Why the generator was removed

It could not run, and had not been able to for some time:

- It hardcoded `/home/aruppel/projects/DirectMethod` and
  `/home/aruppel/projects/napariTFM/_validation/benchmark_TFM/` (note the lowercase `projects`;
  the tree is `~/Projects`). Neither path resolves.
- DirectMethod is not vendored in this repo and is not present on the machine, so the imports
  fail at module load.
- It `chdir`-ed into the DirectMethod checkout and symlinked the config as `description.toml`
  there, mutating an external repo to run.
- `main()` only ever generated the `low` tier; `mid` and `high` required hand-editing the path.

A script that cannot execute is not provenance, it only looks like it. This file is the
provenance; the TOML is the spec. Nothing else referenced the generator.

## Relationship to benchmarkTFM

New displacement/force benchmarking lives in the standalone `benchmarkTFM` repo. It does **not**
supersede this data: it uses a different geometry (a crossbow cell with focal-adhesion dipoles,
plus realistic bead rendering) and cannot reproduce these analytic Hertz dipole scenarios. The
two answer different questions. This one is a small, exact, fast regression check on the shipped
code; benchmarkTFM is where method and parameter choices are researched.
