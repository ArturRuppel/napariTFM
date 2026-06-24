"""The config table is the single source of a batch run config (P4.1)."""

from napariTFM.widgets._run_config import build_run_config


_RECORDS = [
    {
        "path": "/data/exp_a",
        "input_files": {"beads": "beads.tif", "reference": "ref.tif", "cells": "cells.tif"},
        "columns": {"condition": "soft"},
    },
    {
        "path": "/data/exp_b",
        "input_files": {"beads": "beads.tif", "reference": "ref.tif", "cells": "cells.tif"},
        "columns": {"condition": "stiff"},
    },
]


def test_root_folders_follow_record_order():
    cfg = build_run_config(_RECORDS, {})
    assert cfg["root_folders"] == ["/data/exp_a", "/data/exp_b"]


def test_input_files_come_from_the_first_record():
    cfg = build_run_config(_RECORDS, {})
    assert cfg["input_files"]["beads"] == "beads.tif"
    assert cfg["input_files"]["reference"] == "ref.tif"
    assert cfg["input_files"]["cells"] == "cells.tif"


def test_all_pipeline_steps_run_by_default():
    cfg = build_run_config(_RECORDS, {})
    steps = cfg["analysis_steps"]
    assert steps["preprocessing"] and steps["displacement"]
    assert steps["force"] and steps["stress"]


def test_disabled_stress_is_skipped():
    # MSM keeps its on/off glyph as the disable path (D1).
    cfg = build_run_config(_RECORDS, {}, disabled_stages=["stress"])
    assert cfg["analysis_steps"]["stress"] is False
    assert cfg["analysis_steps"]["force"] is True


def test_parameters_pass_through():
    cfg = build_run_config(_RECORDS, {"downscale_factor": 4})
    assert cfg["parameters"] == {"downscale_factor": 4}


def test_experiment_metadata_maps_path_to_columns():
    cfg = build_run_config(_RECORDS, {})
    assert cfg["experiment_metadata"] == {
        "/data/exp_a": {"condition": "soft"},
        "/data/exp_b": {"condition": "stiff"},
    }


def test_save_cache_flag_is_carried():
    assert build_run_config(_RECORDS, {})["save_cache"] is False
    assert build_run_config(_RECORDS, {}, save_cache=True)["save_cache"] is True


def test_visualizations_and_metrics_are_constant_and_present():
    cfg = build_run_config(_RECORDS, {})
    assert cfg["visualizations"]["displacement_map"] is True
    assert cfg["metrics_parameters"]["calculate_strain_energy"] is True


def test_empty_records_yield_empty_folders_and_inputs():
    cfg = build_run_config([], {})
    assert cfg["root_folders"] == []
    assert cfg["input_files"] == {}
    assert cfg["experiment_metadata"] == {}
