"""A PipelineSink that reports stage/frame progress across a process boundary.

Used by parallel Run-all workers (see ``batch_analysis._run_position_headless``):
each worker process attaches one ``QueueProgressSink`` wrapping a
``multiprocessing.Queue`` shared with the parent process, so the parent's
150ms poll timer (``BatchAnalysis.poll_parallel_progress``) can drain real
per-stage, per-frame progress instead of only learning a folder's terminal
``done``/``error`` status when its worker fully returns.

Mirrors ``ViewerSink``'s fraction math exactly (see
``napariTFM.utilities.viewer_sink.ViewerSink``) so both sinks compute "how far
into this stage are we" identically -- one delivers it via a Qt signal
in-process, this one via a multiprocessing queue.
"""

from typing import Any, Optional

from napariTFM.backend.pipeline_sink import PipelineSink


class QueueProgressSink(PipelineSink):
    """Puts ``(folder, stage, status, fraction)`` tuples onto a shared queue.

    Parameters
    ----------
    queue
        A ``multiprocessing.Queue`` created by the parent process (spawn
        context matching the ``ProcessPoolExecutor``) and passed into this
        worker's task. Multiple worker processes ``put()`` onto it
        concurrently; that is the queue's designed usage.
    folder
        This worker's experiment folder path, stamped onto every message so
        the parent can route it to the right row.
    """

    def __init__(self, queue, folder: str):
        self._queue = queue
        self._folder = folder
        self._stage_num_frames = 0

    def stage_started(self, stage: str, num_frames: int, info: Optional[dict] = None) -> None:
        self._stage_num_frames = num_frames
        self._queue.put((self._folder, stage, "running", 0.0))

    def stage_frame(self, stage: str, frame_index: int, frame: Any) -> None:
        fraction = (frame_index + 1) / max(self._stage_num_frames, 1)
        self._queue.put((self._folder, stage, "running", fraction))

    def stage_finished(self, stage: str, result: Any) -> None:
        self._queue.put((self._folder, stage, "done", None))
