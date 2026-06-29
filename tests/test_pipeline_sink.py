"""The batch orchestrator notifies an optional PipelineSink at stage/frame
boundaries (worklist §5: one code path for headless and in-napari runs).

These tests pin the hook contract the live ``ViewerSink`` relies on — which
stage keys fire, the (0-based) frame indices, and the per-stage ``info`` — while
proving that with no sink attached the run is a silent no-op (headless is
byte-for-byte unchanged).
"""

import types

import numpy as np
import pytest

from napariTFM.backend import batch_analysis as ba
from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.backend.pipeline_sink import NullSink, PipelineSink


class RecordingSink(PipelineSink):
    """Captures every hook call as a flat event list for assertions."""

    def __init__(self):
        self.events = []

    def experiment_started(self, path):
        self.events.append(("experiment", path))

    def stage_started(self, stage, num_frames, info=None):
        self.events.append(("start", stage, num_frames, info))

    def stage_frame(self, stage, frame_index, frame):
        self.events.append(("frame", stage, frame_index, frame))

    def stage_finished(self, stage, result):
        self.events.append(("finish", stage, result))


def _bare(sink=None):
    """A BatchAnalysis with just the attributes the _emit helper touches."""
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {}
    analysis._sink = sink
    return analysis


# --- the _emit helper itself ---------------------------------------------

def test_emit_with_no_sink_is_silent_noop():
    analysis = _bare(sink=None)
    analysis._emit("stage_started", "force", 3, {})  # must not raise


def test_emit_forwards_to_sink():
    sink = RecordingSink()
    analysis = _bare(sink=sink)
    analysis._emit("stage_frame", "force", 2, "payload")
    assert sink.events == [("frame", "force", 2, "payload")]


def test_emit_swallows_sink_errors(capsys):
    class Boom(PipelineSink):
        def stage_frame(self, *a):
            raise RuntimeError("kaboom")

    analysis = _bare(sink=Boom())
    analysis._emit("stage_frame", "force", 0, None)  # must not propagate
    assert "Pipeline sink error" in capsys.readouterr().out


def test_null_sink_accepts_every_hook():
    sink = NullSink()
    sink.experiment_started("/data/pos_00")
    sink.stage_started("force", 1, {})
    sink.stage_frame("force", 0, None)
    sink.stage_finished("force", None)


def test_process_all_folders_emits_experiment_started_per_folder(tmp_path, monkeypatch):
    """Each folder fires one experiment_started before its stages (§3)."""
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    sink = RecordingSink()
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {"root_folders": [a, b]}
    analysis._progress_callback = None
    analysis._sink = sink
    analysis._cancelled = False

    # Skip the real per-folder work; we only care about the experiment hook here.
    monkeypatch.setattr(analysis, "process_folder", lambda folder, output_dir: None)
    analysis.process_all_folders()

    assert sink.events == [("experiment", a), ("experiment", b)]


# --- displacement / force share one generator-driven shape ----------------

def _params(**kw):
    return types.SimpleNamespace(**kw)


def test_displacement_emits_start_frames_finish(monkeypatch):
    sink = RecordingSink()
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {}
    analysis._sink = sink
    monkeypatch.setattr(analysis, "_create_displacement_parameters", lambda: _params(
        d_max=5.0, disp_vector_stride=8, disp_arrow_scale=2.0, downscale_factor=4,
    ))

    sentinel = object()

    def fake_gen(reference, beads, params):
        # 1-based frame numbers, two frames, then a final result.
        yield np.zeros((2, 2, 2)), 1, 2
        yield np.zeros((2, 2, 2)), 2, 2
        return sentinel

    monkeypatch.setattr(ba, "calculate_displacement_field", fake_gen)

    preprocessed = {"beads": np.zeros((2, 4, 4)), "reference": np.zeros((4, 4))}
    result = analysis._execute_displacement_analysis(None, preprocessed)

    assert result is sentinel
    kinds = [(e[0], e[1]) for e in sink.events]
    assert kinds == [
        ("start", "displacement"),
        ("frame", "displacement"),
        ("frame", "displacement"),
        ("finish", "displacement"),
    ]
    start = sink.events[0]
    assert start[2] == 2  # num_frames from beads.shape[0]
    assert start[3] == {
        "v_max": 5.0, "vector_stride": 8, "arrow_scale": 2.0, "downscale_factor": 4,
    }
    # Frame indices are 0-based (1-based backend frame minus one).
    assert [e[2] for e in sink.events if e[0] == "frame"] == [0, 1]


def test_force_emits_zero_based_frames_and_fmax_info(monkeypatch):
    sink = RecordingSink()
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {}
    analysis._sink = sink
    monkeypatch.setattr(analysis, "_create_fttc_parameters", lambda: _params(
        f_max=100.0, force_vector_stride=6, force_arrow_scale=3.0, downscale_factor=2,
    ))

    def fake_gen(field, params):
        yield np.zeros((2, 2, 2)), 1, 1
        return "force-result"

    monkeypatch.setattr(ba, "calculate_force_field", fake_gen)

    disp = types.SimpleNamespace(displacement_field=np.zeros((1, 4, 4, 2)))
    result = analysis._execute_force_analysis(None, disp)

    assert result == "force-result"
    assert sink.events[0] == ("start", "force", 1, {
        "v_max": 100.0, "vector_stride": 6, "arrow_scale": 3.0, "downscale_factor": 2,
    })
    assert sink.events[1] == ("frame", "force", 0, sink.events[1][3])
    assert sink.events[-1][:2] == ("finish", "force")


def test_no_sink_means_no_emission_during_force(monkeypatch):
    """Headless (sink=None) drives the same loop without any hook calls."""
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {}
    analysis._sink = None
    monkeypatch.setattr(analysis, "_create_fttc_parameters", lambda: _params(
        f_max=1.0, force_vector_stride=1, force_arrow_scale=1.0, downscale_factor=1,
    ))

    def fake_gen(field, params):
        yield np.zeros((2, 2, 2)), 1, 1
        return "ok"

    monkeypatch.setattr(ba, "calculate_force_field", fake_gen)
    disp = types.SimpleNamespace(displacement_field=np.zeros((1, 4, 4, 2)))
    # Just has to run cleanly with no sink wired.
    assert analysis._execute_force_analysis(None, disp) == "ok"
