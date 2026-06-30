"""Build a batch-run config straight from the shared config table (P4).

The experiments list is the single source of truth for what gets run: its
per-row records (folder + input file names + free-form columns) plus the shared
``ParameterManager`` values are all a :class:`BatchAnalysis` needs. This pure
function is the sole producer of the run config now that the standalone batch
widget is retired — the run is driven straight from the table.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

# Bump when the on-disk shape of the experiment-series file changes.
SERIES_FORMAT_VERSION = 1

# Every visualization is produced; per-stage viz selection returns as a header
# glyph (P5). Bead overlay was removed entirely.
_VISUALIZATIONS = {
    "displacement_map": True,
    "force_map": True,
    "force_cell_overlay": True,
    "sigma_xx": True,
    "sigma_yy": True,
    "normal_stress": True,
}

# Masks are always supplied externally, so no mask-creation parameters here.
_METRICS_PARAMETERS = {
    "calculate_strain_energy": True,
    "calculate_polarization": True,
    "export_eigenvalues": True,
}


def build_run_config(
    records: Sequence[Mapping],
    parameters: Mapping,
    *,
    disabled_stages: Iterable[str] = (),
    save_cache: bool = False,
    processed_root: object = None,
) -> dict:
    """Assemble the run config from the table's ``records`` + shared ``parameters``.

    ``records`` are :meth:`ExperimentsList.experiment_records` entries (``path`` /
    ``input_files`` / ``columns``); folders run in record order and the input
    file names are taken from the first record (the table applies them
    uniformly). Every pipeline step is mandatory except stress, which is skipped
    when ``"stress"`` is in ``disabled_stages`` (the MSM on/off glyph, D1). Each
    row's free-form columns ride along as ``experiment_metadata`` for the saved
    config and the §5 aggregator.
    """
    disabled = set(disabled_stages)

    root_folders = [record["path"] for record in records]
    input_files = dict(records[0]["input_files"]) if records else {}

    analysis_steps = {
        "preprocessing": True,
        "displacement": True,
        "force": True,
        "stress": "stress" not in disabled,
        "calculate_metrics": True,
    }

    experiment_metadata = {
        record["path"]: dict(record.get("columns", {})) for record in records
    }

    return {
        "root_folders": root_folders,
        "input_files": input_files,
        "analysis_steps": analysis_steps,
        "visualizations": dict(_VISUALIZATIONS),
        "parameters": dict(parameters),
        "metrics_parameters": dict(_METRICS_PARAMETERS),
        "save_cache": bool(save_cache),
        "experiment_metadata": experiment_metadata,
        # Where derived output lands; ``None`` means in-place (a ``processed/``
        # bucket inside each input folder). The status dots and interactive
        # persist resolve against this same value, so all three agree.
        "processed_root": str(processed_root) if processed_root else None,
    }


def build_series_config(
    records: Sequence[Mapping],
    *,
    disabled_stages: Iterable[str] = (),
    processed_root: object = None,
) -> dict:
    """The portable *experiment-series* file: *what* to run, never *how*.

    Holds the dataset (folders + the shared input-file names) and each row's
    free-form design tags, plus run options (which stages to skip, where
    processed output should land). The analysis knobs live in the separate
    ``tfm_params`` preset; the two halves are folded back together only at run
    time and inside each ``.ntfm`` for provenance. Mirrors
    :func:`build_run_config` minus ``parameters`` so the same ``records`` feed
    both.
    """
    input_files = dict(records[0]["input_files"]) if records else {}
    return {
        "format_version": SERIES_FORMAT_VERSION,
        "input_files": input_files,
        "root_folders": [record["path"] for record in records],
        "experiment_metadata": {
            record["path"]: dict(record.get("columns", {})) for record in records
        },
        "run_options": {
            "disabled_stages": list(disabled_stages),
            "processed_root": str(processed_root) if processed_root else None,
        },
    }


def series_records(config: Mapping) -> List[dict]:
    """Rebuild :meth:`ExperimentsList.set_records` rows from a series file.

    The inverse of :func:`build_series_config`: the shared ``input_files`` map is
    copied onto every row and each path picks up its own design tags.
    """
    input_files = config.get("input_files", {}) or {}
    metadata = config.get("experiment_metadata", {}) or {}
    return [
        {
            "path": path,
            "input_files": dict(input_files),
            "columns": dict(metadata.get(path, {})),
        }
        for path in (config.get("root_folders", []) or [])
    ]
