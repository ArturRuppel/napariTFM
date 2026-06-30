"""Real-process integration test for parallel batch-worker processing.

Every other test that touches ``process_all_folders``/``process_folder``
mocks or monkeypatches something between this process and the actual work
(``process_folder`` itself in ``test_batch_progress.py``; the
``ProcessPoolExecutor``/``_run_position_headless`` in ``test_batch_parallel.py``).
This file deliberately does neither: it runs the real, unmocked
``BatchAnalysis.process_all_folders()`` end to end, once with
``num_workers=1`` (in-process sequential loop) and once with
``num_workers=2`` (a genuine ``ProcessPoolExecutor`` using the ``spawn``
start method -- real OS process spawn, real pickling of the config dict
across the process boundary, a real fresh import of ``napariTFM`` in each
child, real execution of ``process_folder``, and a real result handed back
through a ``Future``), and asserts the two modes agree.

All analysis steps are disabled, so each ``process_folder`` call is a real
but minimal run: it creates the output directory and ``batch.log``
(``_initialize_folder``), and every analysis stage and ``_load_mask``
return ``None`` immediately because there is no work configured / no mask
file present -- both are normal, exercised-elsewhere code paths, not special
cases for this test. No real bead/cell/reference TIFFs are needed.
"""

from pathlib import Path

from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.utilities.batch_output import resolve_output_plan


def _minimal_config(root_folders, num_workers):
    return {
        "root_folders": root_folders,
        "num_workers": num_workers,
        "analysis_steps": {
            "preprocessing": False,
            "displacement": False,
            "force": False,
            "stress": False,
        },
        "visualizations": {},
        "parameters": {},
        "input_files": {},
    }


def _run(tmp_path, run_name, num_workers, position_names=("posA", "posB")):
    """Run the real pipeline over a fresh set of folders, in-place output
    (no ``processed_root``), and return (folders, events)."""
    folders = [str(tmp_path / run_name / name) for name in position_names]
    events = []
    config = _minimal_config(folders, num_workers)
    analysis = BatchAnalysis(config, progress_callback=lambda f, s: events.append((f, s)))
    analysis.process_all_folders()
    return folders, events


def test_sequential_and_real_parallel_pool_agree(tmp_path):
    # Two entirely separate sets of input folders (different parent dirs,
    # same basenames) so the two runs' in-place TFM_data output can never
    # collide on disk.
    folders_seq, events_seq = _run(tmp_path, "run1_sequential", num_workers=1)
    folders_par, events_par = _run(tmp_path, "run2_parallel", num_workers=2)

    # No exception escaped either run (we'd already have failed above), and
    # neither run reports an error status for any folder.
    assert [s for _, s in events_seq if s == "error"] == []
    assert [s for _, s in events_par if s == "error"] == []

    # Every folder in both runs reaches "running" then "done".
    for folder in folders_seq:
        assert (folder, "running") in events_seq
        assert (folder, "done") in events_seq
    for folder in folders_par:
        assert (folder, "running") in events_par
        assert (folder, "done") in events_par

    # Parity: same number/shape of lifecycle events in both modes (the real
    # pool may complete folders in a different order than submission, but
    # the multiset of statuses must match the sequential run's).
    assert sorted(s for _, s in events_seq) == sorted(s for _, s in events_par)
    assert len(events_seq) == len(events_par) == 2 * len(folders_seq)

    # Parity: both modes produce the same set of resolved output directories
    # on disk (in-place TFM_data/<position> sibling of each input folder).
    plan_seq = resolve_output_plan(folders_seq, None)
    plan_par = resolve_output_plan(folders_par, None)
    for folder in folders_seq:
        assert Path(plan_seq.output_dirs[folder]).is_dir()
        assert (Path(plan_seq.output_dirs[folder]) / "batch.log").exists()
    for folder in folders_par:
        assert Path(plan_par.output_dirs[folder]).is_dir()
        assert (Path(plan_par.output_dirs[folder]) / "batch.log").exists()
