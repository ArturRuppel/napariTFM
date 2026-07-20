"""A PipelineSink that reports stage/frame progress across a process boundary.

Used by parallel Run-selected workers (see ``batch_analysis._run_position_headless``):
each worker process attaches one ``QueueProgressSink`` wrapping a queue
(typically a ``multiprocessing.Manager().Queue()`` proxy -- see
``BatchAnalysis.start_parallel``) shared with the parent process, so the
parent's 150ms poll timer (``BatchAnalysis.poll_parallel_progress``) can
drain real per-stage, per-frame progress instead of only learning a folder's
terminal ``done``/``error`` status when its worker fully returns.

This is the only concrete ``PipelineSink``: it computes "how far into this stage
are we" from the stage's frame count and delivers that fraction to the parent
via a cross-process queue.
"""

from typing import Any, Optional

from napariTFM.backend.pipeline_sink import PipelineSink


class QueueProgressSink(PipelineSink):
    """Puts ``(folder, stage, status, fraction)`` tuples onto a shared queue.

    Parameters
    ----------
    queue
        Any object exposing ``put()`` -- in the parallel-batch worker path
        (``batch_analysis._run_position_headless``/``start_parallel``) this is
        typically a ``multiprocessing.Manager().Queue()`` proxy created by the
        parent process (spawn context matching the ``ProcessPoolExecutor``),
        not a plain ``multiprocessing.Queue`` -- a raw ``Queue`` can't be handed
        to an already-running pool via ``submit()`` (only a Manager proxy
        survives that pickling trip). Multiple worker processes ``put()`` onto
        it concurrently; that is the queue's designed usage. Duck-typed on
        ``.put()`` only, so any queue-like object with that method works
        (e.g. a plain ``queue.Queue()`` in tests).
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
        # In-order stages use frame_index as the authoritative position.
        fraction = (frame_index + 1) / max(self._stage_num_frames, 1)
        self._queue.put((self._folder, stage, "running", fraction))

    def stage_finished(self, stage: str, result: Any) -> None:
        self._queue.put((self._folder, stage, "done", None))
