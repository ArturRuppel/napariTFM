"""Tests for the ``processed/`` output-bucket resolver (ROADMAP §4)."""

from pathlib import Path

from napariTFM.utilities import batch_output
from napariTFM.utilities.batch_output import PROCESSED_BUCKET, resolve_output_plan


def test_in_place_when_no_processed_root():
    folders = ["/data/exp_A", "/data/exp_B"]
    plan = resolve_output_plan(folders, processed_root=None)

    assert plan.output_dirs["/data/exp_A"] == Path("/data/exp_A") / PROCESSED_BUCKET
    assert plan.output_dirs["/data/exp_B"] == Path("/data/exp_B") / PROCESSED_BUCKET
    assert plan.warnings == []


def test_in_place_when_processed_root_empty_string():
    folders = ["/data/exp_A"]
    plan = resolve_output_plan(folders, processed_root="")
    assert plan.output_dirs["/data/exp_A"] == Path("/data/exp_A") / PROCESSED_BUCKET


def test_mirror_tree_under_processed_root():
    folders = ["/data/cond1/exp_A", "/data/cond2/exp_B"]
    plan = resolve_output_plan(folders, processed_root="/out")

    # Longest common parent is /data; tree reproduced relative to it.
    assert plan.output_dirs["/data/cond1/exp_A"] == Path("/out/cond1/exp_A")
    assert plan.output_dirs["/data/cond2/exp_B"] == Path("/out/cond2/exp_B")
    assert plan.warnings == []


def test_single_folder_falls_back_to_basename():
    folders = ["/data/cond1/exp_A"]
    plan = resolve_output_plan(folders, processed_root="/out")

    assert plan.output_dirs["/data/cond1/exp_A"] == Path("/out/exp_A")
    assert plan.warnings == []


def test_disconnected_roots_fall_back_to_basenames_with_warning(monkeypatch):
    folders = ["C:/a/exp_A", "D:/b/exp_B"]

    def _raise(_paths):
        raise ValueError("paths don't have the same drive")

    monkeypatch.setattr(batch_output.os.path, "commonpath", _raise)
    plan = resolve_output_plan(folders, processed_root="/out")

    assert plan.output_dirs["C:/a/exp_A"] == Path("/out/exp_A")
    assert plan.output_dirs["D:/b/exp_B"] == Path("/out/exp_B")
    assert any("disconnected" in w for w in plan.warnings)


def test_basename_collision_is_warned(monkeypatch):
    folders = ["C:/a/exp", "D:/b/exp"]
    monkeypatch.setattr(
        batch_output.os.path,
        "commonpath",
        lambda _paths: (_ for _ in ()).throw(ValueError()),
    )
    plan = resolve_output_plan(folders, processed_root="/out")

    assert any("collide" in w for w in plan.warnings)
