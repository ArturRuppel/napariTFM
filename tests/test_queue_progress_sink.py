"""QueueProgressSink: the process-boundary counterpart to ViewerSink.

A parallel Run-all worker (backend/batch_analysis.py's _run_position_headless)
attaches one of these instead of a ViewerSink, so its stage/frame lifecycle
hooks reach the parent process via a plain queue instead of a Qt signal.
"""

import queue

from napariTFM.backend.pipeline_sink import PipelineSink
from napariTFM.backend.queue_progress_sink import QueueProgressSink


def test_stage_started_enqueues_running_at_zero():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")

    sink.stage_started("displacement", 4)

    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 0.0)


def test_stage_frame_enqueues_growing_fraction():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")
    sink.stage_started("displacement", 4)
    q.get_nowait()  # discard the stage_started message

    sink.stage_frame("displacement", 0, None)
    sink.stage_frame("displacement", 1, None)
    sink.stage_frame("displacement", 3, None)

    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 0.25)
    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 0.5)
    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 1.0)


def test_stage_frame_falls_back_to_one_frame_when_num_frames_is_zero():
    """Mirrors ViewerSink's max(self._stage_num_frames, 1) guard: a
    zero-frame stage still reports a defined fraction instead of dividing by
    zero."""
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")
    sink.stage_started("stress", 0)
    q.get_nowait()

    sink.stage_frame("stress", 0, None)

    assert q.get_nowait() == ("/data/pos_00", "stress", "running", 1.0)


def test_stage_finished_enqueues_done_with_no_fraction():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")

    sink.stage_finished("force", object())

    assert q.get_nowait() == ("/data/pos_00", "force", "done", None)


def test_folder_is_stamped_on_every_message():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_07")

    sink.stage_started("preprocessing", 1)

    folder, *_ = q.get_nowait()
    assert folder == "/data/pos_07"


def test_is_a_pipeline_sink():
    assert issubclass(QueueProgressSink, PipelineSink)
