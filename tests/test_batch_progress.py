"""BatchAnalysis reports per-folder lifecycle to a progress callback (P4.2)."""

from pathlib import Path

from napariTFM.backend.batch_analysis import BatchAnalysis


def _analysis(callback, root_folders):
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {"root_folders": root_folders}
    analysis._progress_callback = callback
    return analysis


def test_constructor_accepts_progress_callback():
    def cb(folder, status):
        pass

    analysis = BatchAnalysis({"root_folders": []}, progress_callback=cb)
    assert analysis._progress_callback is cb


def test_each_folder_reports_running_then_done(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    events = []
    analysis = _analysis(lambda folder, status: events.append((folder, status)), [a, b])

    # Resolve output in-place; skip the real per-folder work.
    monkeypatch.setattr(analysis, "process_folder", lambda folder, output_dir: None)
    analysis.process_all_folders()

    assert events == [
        (a, "running"), (a, "done"),
        (b, "running"), (b, "done"),
    ]


def test_failure_reports_error_and_continues(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    events = []
    analysis = _analysis(lambda folder, status: events.append((folder, status)), [a, b])

    def _maybe_fail(folder, output_dir):
        if folder == a:
            raise RuntimeError("boom")

    monkeypatch.setattr(analysis, "process_folder", _maybe_fail)
    analysis.process_all_folders()

    # The first folder errors, the run continues to the second.
    assert events == [
        (a, "running"), (a, "error"),
        (b, "running"), (b, "done"),
    ]


def test_no_callback_is_a_silent_noop(tmp_path, monkeypatch):
    a = str(tmp_path / "a")
    analysis = _analysis(None, [a])
    monkeypatch.setattr(analysis, "process_folder", lambda folder, output_dir: None)
    analysis.process_all_folders()  # must not raise
