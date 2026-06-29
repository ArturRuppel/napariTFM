"""ViewerSink translates orchestrator hooks into VisualizationManager streaming
calls (worklist §5), so an in-napari run-all walks the rail exactly as the
interactive per-stage controllers do — without re-implementing the pipeline.

A fake VisualizationManager / DataManager records the calls; the tests pin the
translation per stage so the live viewer path can't silently drift from the
streaming API the controllers already rely on.
"""

import numpy as np

from napariTFM.utilities.viewer_sink import ViewerSink


class FakeVis:
    def __init__(self):
        self.calls = []

    def begin_vector_field_stream(self, kind, num_frames, vis_params):
        self.calls.append(("begin_vector", kind, num_frames, vis_params))

    def stream_vector_field_frame(self, kind, frame_index, frame):
        self.calls.append(("stream_vector", kind, frame_index))

    def begin_stress_stream(self, num_frames, max_stress, downscale_factor):
        self.calls.append(("begin_stress", num_frames, max_stress, downscale_factor))

    def stream_stress_frame(self, frame_index, frame):
        self.calls.append(("stream_stress", frame_index))

    def begin_preprocessing_stream(self):
        self.calls.append(("begin_preproc",))

    def stream_preprocessing_frame(self, data_type, frame_index, image):
        self.calls.append(("stream_preproc", data_type, frame_index))


class FakeData:
    def __init__(self):
        self.calls = []
        self.preprocessed_bead_stack = None
        self.preprocessed_reference = None
        self.preprocessed_cell_stack = None

    def set_preprocessed_bead_stack(self, data, source="", dirty=False, path=None):
        self.preprocessed_bead_stack = data
        self.calls.append(("bead_stack", data.shape))

    def set_preprocessed_reference(self, data, source="", dirty=False, path=None):
        self.preprocessed_reference = data
        self.calls.append(("reference", data.shape))

    def set_preprocessed_cell_stack(self, data, source="", dirty=False, path=None):
        self.preprocessed_cell_stack = data
        self.calls.append(("cell_stack", data.shape))

    def set_displacement_results(self, results, source="", dirty=False, path=None):
        self.calls.append(("displacement_results", results))

    def set_force_results(self, results, source="", dirty=False, path=None):
        self.calls.append(("force_results", results))

    def set_stress_results(self, results, source="", dirty=False, path=None):
        self.calls.append(("stress_results", results))


def _sink(pump=None, on_experiment=None):
    vis, data = FakeVis(), FakeData()
    return ViewerSink(data, vis, pump=pump, on_experiment=on_experiment), vis, data


# --- displacement / force ------------------------------------------------

def test_vector_stage_starts_and_streams():
    sink, vis, data = _sink()
    info = {"v_max": 5.0, "vector_stride": 8, "arrow_scale": 2.0, "downscale_factor": 4}
    sink.stage_started("displacement", 2, info)
    sink.stage_frame("displacement", 0, np.zeros((2, 2, 2)))
    sink.stage_frame("displacement", 1, np.zeros((2, 2, 2)))
    sink.stage_finished("displacement", "disp-result")

    assert vis.calls[0] == ("begin_vector", "displacement", 2, info)
    assert vis.calls[1] == ("stream_vector", "displacement", 0)
    assert vis.calls[2] == ("stream_vector", "displacement", 1)
    assert ("displacement_results", "disp-result") in data.calls


def test_force_stage_routes_to_force_kind():
    sink, vis, data = _sink()
    info = {"v_max": 100.0, "vector_stride": 6, "arrow_scale": 3.0, "downscale_factor": 2}
    sink.stage_started("force", 1, info)
    sink.stage_frame("force", 0, np.zeros((2, 2, 2)))
    sink.stage_finished("force", "force-result")

    assert vis.calls[0] == ("begin_vector", "force", 1, info)
    assert vis.calls[1] == ("stream_vector", "force", 0)
    assert ("force_results", "force-result") in data.calls


# --- stress --------------------------------------------------------------

def test_stress_stage_starts_and_streams():
    sink, vis, data = _sink()
    sink.stage_started("stress", 3, {"max_stress": 10.0, "downscale_factor": 2})
    sink.stage_frame("stress", 0, np.zeros((2, 2, 2, 2)))
    sink.stage_finished("stress", "stress-result")

    assert vis.calls[0] == ("begin_stress", 3, 10.0, 2)
    assert vis.calls[1] == ("stream_stress", 0)
    assert ("stress_results", "stress-result") in data.calls


# --- preprocessing -------------------------------------------------------

def test_preprocessing_allocates_stacks_then_streams_each_channel():
    sink, vis, data = _sink()
    sink.stage_started("preprocessing", 2, {
        "beads_shape": (2, 4, 4),
        "reference_shape": (4, 4),
        "cells_shape": (2, 4, 4),
    })
    # Allocation happens before any frame, and the layers get bound.
    assert ("bead_stack", (2, 4, 4)) in data.calls
    assert ("reference", (4, 4)) in data.calls
    assert ("cell_stack", (2, 4, 4)) in data.calls
    assert ("begin_preproc",) in vis.calls
    assert data.preprocessed_bead_stack.dtype == np.float32

    sink.stage_frame("preprocessing", 0, {"beads": np.zeros((4, 4))})
    sink.stage_frame("preprocessing", 0, {"reference": np.zeros((4, 4))})
    sink.stage_frame("preprocessing", 1, {"cells": np.zeros((4, 4))})

    assert ("stream_preproc", "beads", 0) in vis.calls
    assert ("stream_preproc", "reference", 0) in vis.calls
    assert ("stream_preproc", "cells", 1) in vis.calls


def test_preprocessing_skips_absent_cell_channel():
    sink, vis, data = _sink()
    sink.stage_started("preprocessing", 1, {
        "beads_shape": (1, 4, 4),
        "reference_shape": (4, 4),
        "cells_shape": None,
    })
    assert all(c[0] != "cell_stack" for c in data.calls)


# --- pump + robustness ---------------------------------------------------

def test_pump_fires_after_each_frame():
    pumps = []
    sink, vis, data = _sink(pump=lambda: pumps.append(1))
    sink.stage_started("force", 1, {
        "v_max": 1.0, "vector_stride": 1, "arrow_scale": 1.0, "downscale_factor": 1,
    })
    sink.stage_frame("force", 0, np.zeros((2, 2, 2)))
    # One pump for stage_started, one for the frame.
    assert len(pumps) == 2


def test_stage_finished_with_none_result_is_noop():
    sink, vis, data = _sink()
    sink.stage_finished("force", None)
    assert data.calls == []


# --- experiment tracking (§3) --------------------------------------------

def test_experiment_started_forwards_path_to_callback():
    seen = []
    sink, vis, data = _sink(on_experiment=seen.append)
    sink.experiment_started("/data/pos_03")
    assert seen == ["/data/pos_03"]


def test_experiment_started_without_callback_is_noop():
    sink, vis, data = _sink()
    sink.experiment_started("/data/pos_03")  # must not raise


def test_experiment_started_pumps_the_event_loop():
    pumps = []
    sink, vis, data = _sink(pump=lambda: pumps.append(1))
    sink.experiment_started("/data/pos_03")
    assert pumps == [1]
