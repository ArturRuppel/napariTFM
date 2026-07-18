"""Run-selected cancellation must stop an in-flight experiment, not just prevent
the next folder from starting.

Before this fix ``_cancelled`` was only read at the top of the per-folder loop,
so a single running experiment (one folder) ran every stage and every frame to
completion no matter when Cancel was clicked — the flag was set (the click is
delivered via the sink's per-frame ``processEvents`` pump) but never read again.
These tests pin the cooperative checkpoints that make cancel bite mid-folder:

  * ``_raise_if_cancelled`` raises the internal sentinel iff a cancel is pending;
  * a stream loop that checks it stops within one frame of the flag being set;
  * ``_guard_stage`` re-raises the cancel instead of recording it as a stage
    failure (a cancel is not an error);
  * the folder loop reports ``"cancelled"`` (not ``"error"``) and stops.
"""

import types

import numpy as np
import pytest

from napariTFM.backend import batch_analysis as ba
from napariTFM.backend.batch_analysis import BatchAnalysis, _BatchCancelled


def _analysis(root_folders=None, callback=None):
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {
        "root_folders": root_folders or [],
        "analysis_steps": {"displacement": True, "force": True},
    }
    analysis._progress_callback = callback
    analysis._cancelled = False
    analysis._stage_failures = []
    return analysis


def test_raise_if_cancelled_fires_only_when_pending():
    analysis = _analysis()
    analysis._raise_if_cancelled()          # no-op while not cancelled
    analysis.request_cancel()
    with pytest.raises(_BatchCancelled):
        analysis._raise_if_cancelled()


def test_stream_loop_stops_within_one_frame_of_the_click():
    """The real failure: a click delivered mid-stream (here, after frame 0, as the
    sink's per-frame pump would deliver it) must stop the very next frame."""
    analysis = _analysis()
    processed = []

    def streaming_body():
        for frame in range(10):
            analysis._raise_if_cancelled()      # checkpoint at the top of the loop
            processed.append(frame)
            if frame == 0:
                analysis.request_cancel()       # stand-in for the pumped Cancel click
        return {"unreached": True}

    with pytest.raises(_BatchCancelled):
        analysis._guard_stage("displacement", streaming_body)

    assert processed == [0]                      # stopped before frame 1, not all 10
    assert analysis._stage_failures == []        # a cancel is not a stage failure


def test_guard_stage_cancels_at_the_boundary_without_running_the_body():
    analysis = _analysis()
    analysis.request_cancel()                    # already pending before the stage starts
    ran = []
    with pytest.raises(_BatchCancelled):
        analysis._guard_stage("force", lambda: ran.append("body"))
    assert ran == []                             # never entered the stage
    assert analysis._stage_failures == []


def test_folder_loop_reports_cancelled_not_error(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    events = []
    analysis = _analysis([a, b], lambda folder, status: events.append((folder, status)))

    def _cancelled_folder(folder, output_dir):
        # process_folder unwinds via _BatchCancelled when a mid-folder click lands.
        raise _BatchCancelled()

    monkeypatch.setattr(analysis, "process_folder", _cancelled_folder)

    class _Plan:
        output_dirs = {a: tmp_path / "a", b: tmp_path / "b"}
        warnings = []

    monkeypatch.setattr(
        "napariTFM.backend.batch_analysis.resolve_output_plan", lambda *a, **k: _Plan()
    )
    analysis._emit = lambda *a, **k: None
    analysis.process_all_folders()

    # First folder cancelled (not "error"); the run stops — folder b never starts.
    assert events == [(a, "running"), (a, "cancelled")]


def test_real_displacement_executor_stops_mid_stream(monkeypatch):
    """Drive the actual ``_execute_displacement_analysis`` loop: the sink's
    per-frame hook flips ``_cancelled`` (as the live ``processEvents`` pump
    delivers the click), and the next loop iteration's checkpoint must unwind
    after exactly one streamed frame — not run all five."""
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {}
    analysis._cancelled = False

    streamed = []

    class _CancelOnFirstFrameSink:
        def stage_started(self, *a, **k):
            pass

        def stage_frame(self, stage, frame_index, frame):
            streamed.append(frame_index)
            analysis.request_cancel()      # the pumped Cancel click lands here

        def stage_finished(self, *a, **k):
            raise AssertionError("stage must not finish after a cancel")

    analysis._sink = _CancelOnFirstFrameSink()
    monkeypatch.setattr(analysis, "_create_displacement_parameters", lambda: types.SimpleNamespace(
        d_max=5.0, disp_vector_stride=8, disp_arrow_scale=2.0, downscale_factor=4,
    ))

    def fake_gen(reference, beads, params, **kwargs):
        for frame in range(1, 6):          # five frames available (1-based)
            yield np.zeros((2, 2, 2)), frame, 5
        return object()                    # never reached

    monkeypatch.setattr(ba, "calculate_displacement_field", fake_gen)

    preprocessed = {"beads": np.zeros((5, 4, 4)), "reference": np.zeros((4, 4))}
    with pytest.raises(_BatchCancelled):
        analysis._execute_displacement_analysis(None, preprocessed)

    assert streamed == [0]                 # one frame streamed, then cancelled
