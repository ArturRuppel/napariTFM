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
        # Visibility-takeover (worklist §4): isolate calls are recorded
        # separately so per-stage ordering assertions on ``calls`` stay stable.
        self.isolations = []
        self.visibility = {}
        self.restored = None

    def isolate_layers(self, keep_names):
        self.isolations.append(list(keep_names))

    def capture_layer_visibility(self):
        return dict(self.visibility)

    def restore_layer_visibility(self, snapshot):
        self.restored = dict(snapshot)

    def begin_vector_field_stream(self, kind, num_frames, vis_params):
        self.calls.append(("begin_vector", kind, num_frames, vis_params))

    def stream_vector_field_frame(self, kind, frame_index, frame):
        self.calls.append(("stream_vector", kind, frame_index))

    def begin_stress_stream(self, num_frames, max_stress, downscale_factor):
        self.calls.append(("begin_stress", num_frames, max_stress, downscale_factor))

    def stream_stress_frame(self, frame_index, frame):
        self.calls.append(("stream_stress", frame_index))


class FakeData:
    def __init__(self):
        self.calls = []

    def set_displacement_results(self, results, source="", dirty=False, path=None):
        self.calls.append(("displacement_results", results))

    def set_force_results(self, results, source="", dirty=False, path=None):
        self.calls.append(("force_results", results))

    def set_stress_results(self, results, source="", dirty=False, path=None):
        self.calls.append(("stress_results", results))


def _sink(pump=None, on_experiment=None, on_stage_progress=None):
    vis, data = FakeVis(), FakeData()
    return (
        ViewerSink(
            data, vis, pump=pump, on_experiment=on_experiment,
            on_stage_progress=on_stage_progress,
        ),
        vis,
        data,
    )


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


# --- per-stage layer isolation (§4) --------------------------------------

def test_displacement_stage_isolates_displacement_layers():
    sink, vis, data = _sink()
    sink.stage_started("displacement", 1, {
        "v_max": 1.0, "vector_stride": 1, "arrow_scale": 1.0, "downscale_factor": 1,
    })
    assert vis.isolations == [["Displacement Magnitude", "Displacement Vectors"]]


def test_force_stage_isolates_force_layers():
    sink, vis, data = _sink()
    sink.stage_started("force", 1, {
        "v_max": 1.0, "vector_stride": 1, "arrow_scale": 1.0, "downscale_factor": 1,
    })
    assert vis.isolations == [["Force Magnitude", "Force Vectors"]]


def test_stress_stage_isolates_stress_layers():
    sink, vis, data = _sink()
    sink.stage_started("stress", 1, {"max_stress": 1.0, "downscale_factor": 1})
    assert vis.isolations == [
        ["Normal Stress XX", "Normal Stress YY", "Average Normal Stress"],
    ]


# --- run-boundary visibility restore (§4) --------------------------------

def test_begin_run_snapshots_and_end_run_restores_visibility():
    sink, vis, data = _sink()
    vis.visibility = {"Preprocessed Beads": True, "Force Magnitude": False}
    sink.begin_run()
    sink.end_run()
    assert vis.restored == {"Preprocessed Beads": True, "Force Magnitude": False}


def test_end_run_without_begin_is_noop():
    sink, vis, data = _sink()
    sink.end_run()  # must not raise
    assert vis.restored is None


def test_end_run_is_idempotent():
    sink, vis, data = _sink()
    vis.visibility = {"Force Magnitude": True}
    sink.begin_run()
    sink.end_run()
    vis.restored = None
    sink.end_run()  # second restore is a no-op
    assert vis.restored is None


# --- per-stage progress (item #10, progressive loading bar) --------------

def test_stage_started_reports_running_at_zero():
    seen = []
    sink, vis, data = _sink(on_stage_progress=lambda *a: seen.append(a))
    sink.stage_started("displacement", 4, {
        "v_max": 1.0, "vector_stride": 1, "arrow_scale": 1.0, "downscale_factor": 1,
    })
    assert seen == [("displacement", "running", 0.0)]


def test_stage_frame_reports_growing_fraction():
    seen = []
    sink, vis, data = _sink(on_stage_progress=lambda *a: seen.append(a))
    sink.stage_started("displacement", 4, {
        "v_max": 1.0, "vector_stride": 1, "arrow_scale": 1.0, "downscale_factor": 1,
    })
    sink.stage_frame("displacement", 0, np.zeros((2, 2, 2)))
    sink.stage_frame("displacement", 1, np.zeros((2, 2, 2)))
    sink.stage_frame("displacement", 3, np.zeros((2, 2, 2)))
    assert seen[1:] == [
        ("displacement", "running", 0.25),
        ("displacement", "running", 0.5),
        ("displacement", "running", 1.0),
    ]


def test_stage_finished_reports_done_with_no_fraction():
    seen = []
    sink, vis, data = _sink(on_stage_progress=lambda *a: seen.append(a))
    sink.stage_finished("force", "force-result")
    assert seen == [("force", "done", None)]


def test_no_progress_callback_is_noop():
    sink, vis, data = _sink()
    sink.stage_started("force", 1, {
        "v_max": 1.0, "vector_stride": 1, "arrow_scale": 1.0, "downscale_factor": 1,
    })
    sink.stage_frame("force", 0, np.zeros((2, 2, 2)))
    sink.stage_finished("force", "force-result")  # must not raise
