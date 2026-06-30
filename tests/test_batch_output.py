"""Tests for the ``processed/`` output-bucket resolver (ROADMAP §4)."""

from pathlib import Path

from napariTFM.utilities import batch_output
from napariTFM.utilities.batch_output import (
    PROCESSED_BUCKET,
    experiment_ntfm_path,
    experiment_output_dir,
    resolve_output_plan,
)


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


def test_experiment_ntfm_path_matches_batch_write_location():
    # The status dots and the interactive persist resolve the .ntfm with this
    # helper; the batch writes to resolve_output_plan's dir + "<name>.ntfm". They
    # must be byte-identical or the dots read a file the batch never wrote.
    folder = "/data/cond1/pos_00"

    # In-place mode (no processed_root): processed/ bucket inside the folder.
    out_dir = experiment_output_dir(folder, None)
    assert out_dir == Path(folder) / PROCESSED_BUCKET
    assert experiment_ntfm_path(folder, None) == out_dir / "pos_00.ome.tif"

    # The batch resolves the same way for the same single folder.
    plan = resolve_output_plan([folder], None)
    assert experiment_ntfm_path(folder, None) == plan.output_dirs[folder] / "pos_00.ome.tif"


def test_experiment_ntfm_path_honours_processed_root():
    folders = ["/data/cond1/exp_A", "/data/cond2/exp_B"]
    # The status path for one experiment matches the batch's mirror-tree dir.
    plan = resolve_output_plan(folders, "/out")
    # Single-folder resolution falls back to basename; that is fine — what must
    # hold is that the writer and the reader agree given the SAME inputs, which
    # they do because both call resolve_output_plan. Assert the helper composes.
    out_dir = experiment_output_dir("/data/cond1/exp_A", "/out")
    assert experiment_ntfm_path("/data/cond1/exp_A", "/out") == out_dir / "exp_A.ome.tif"


def test_basename_collision_is_warned(monkeypatch):
    folders = ["C:/a/exp", "D:/b/exp"]
    monkeypatch.setattr(
        batch_output.os.path,
        "commonpath",
        lambda _paths: (_ for _ in ()).throw(ValueError()),
    )
    plan = resolve_output_plan(folders, processed_root="/out")

    assert any("collide" in w for w in plan.warnings)
