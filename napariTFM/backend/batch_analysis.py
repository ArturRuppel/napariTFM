import os
import sys

# Set Qt to offscreen mode for headless/console execution (must be before any Qt imports)
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, wait as futures_wait
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from queue import Empty
from time import sleep
from time import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import tifffile
import yaml
from skimage.transform import resize

from napariTFM.backend.batch_visualizations import BatchVisualizationSaver
from napariTFM.backend.displacement_analysis import (
    DisplacementResult,
    calculate_displacement_field,
)
from napariTFM.backend.fttc import FTTCResult, calculate_force_field
from napariTFM.backend.bism import calculate_bism_stresses
from napariTFM.backend.ntfm_writer import write_experiment_ntfm
from napariTFM.backend.parameter_dataclasses import DisplacementParameters, FTTCParameters, StressParameters, PreprocessingParameters, UnifiedParameters
from napariTFM.backend.preprocessing import preprocess_frame, preprocess_stack
from napariTFM.backend.queue_progress_sink import QueueProgressSink
from napariTFM.utilities import ntfm
from napariTFM.utilities.batch_output import RESULTS_FILENAME, resolve_output_plan


def save_calibrated_tiff(data: np.ndarray, filepath: Path, pixel_size: float,
                         frame_interval: float) -> None:
    """Save *data* as a calibrated TIFF with ImageJ-compatible metadata.

    This is the shared writer used by both the batch pipeline and the
    interactive widget so that preprocessed images are byte-identical
    regardless of how they were produced.

    Written as float32 with the pixels untouched. Preprocessing already
    normalizes each stack to [0, 1] (``apply_intensity_scaling``), so the old
    second min-max rescale to uint16 both quantized the data and silently
    re-stretched each stack to its own range — the reloaded image no longer
    matched what preprocessing produced, and batch resume fed rescaled uint16
    back into a pipeline that fresh runs feed float [0, 1]. float32 keeps disk
    byte-consistent with memory (and stays Fiji-openable via the ImageJ
    metadata below).

    Args:
        data: numpy array to save
        filepath: path where to save the file
        pixel_size: spatial calibration in µm/pixel
        frame_interval: temporal calibration in minutes/frame
    """
    if data is None:
        return

    data_out = np.asarray(data, dtype=np.float32)

    # Create ImageJ-compatible metadata
    imagej_metadata = {
        'ImageJ': '1.53c',
        'spacing': pixel_size,
        'unit': 'um',
        'frame_interval': frame_interval,
        'frame_interval_unit': 'minute'
    }

    # For Z-stacks or time series, specify dimensions
    if data.ndim > 2:
        imagej_metadata.update({
            'frames': data.shape[0],
            'slices': 1,
            'channels': 1
        })

    # Combine metadata for compatibility
    metadata = {
        'PhysicalSizeX': pixel_size,
        'PhysicalSizeXUnit': 'um',
        'PhysicalSizeY': pixel_size,
        'PhysicalSizeYUnit': 'um',
        'TimeIncrement': frame_interval,
        'TimeIncrementUnit': 'min',
        **imagej_metadata
    }

    # Save with metadata using tifffile
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(filepath),
        data_out,
        imagej=True,
        metadata=metadata,
        resolution=(1 / pixel_size, 1 / pixel_size),  # resolution in pixels per unit
        photometric='minisblack'
    )

    print(f"Saved calibrated TIFF: {filepath}")


def save_preprocessed_tiffs(output_dir: Path, pixel_size: float, frame_interval: float, *,
                            beads: Optional[np.ndarray] = None,
                            reference: Optional[np.ndarray] = None,
                            cells: Optional[np.ndarray] = None) -> None:
    """Write a position's preprocessed bead/reference/cell TIFFs to *output_dir*.

    Single place that knows the canonical filenames
    (``preprocessed_beads.tif`` / ``preprocessed_reference.tif`` /
    ``preprocessed_cells.tif``), used by both the batch pipeline and the
    interactive widget so a position's saved TIFFs stay in lockstep no matter
    which path produced them. Any array left as ``None`` is silently skipped
    (delegated to :func:`save_calibrated_tiff`).
    """
    output_dir = Path(output_dir)
    save_calibrated_tiff(beads, output_dir / "preprocessed_beads.tif", pixel_size, frame_interval)
    save_calibrated_tiff(reference, output_dir / "preprocessed_reference.tif", pixel_size, frame_interval)
    save_calibrated_tiff(cells, output_dir / "preprocessed_cells.tif", pixel_size, frame_interval)


def _run_position_headless(
    config: dict, folder: str, output_dir: str, queue,
) -> tuple[str, str, Optional[str]]:
    """Process one position headlessly, in its own (spawned) worker process.

    Module-level (not a method or closure) so it is picklable for
    ``ProcessPoolExecutor`` under the ``spawn`` start method, which has to
    ship the callable itself to the worker. Constructs a ``BatchAnalysis``
    wired with a ``QueueProgressSink`` -- so this worker's real per-stage/
    per-frame progress reaches the parent process via *queue* -- and runs the
    same per-position pipeline as the sequential path (``process_folder``).

    Args:
        config: the run's plain-dict config (pickled across the process
            boundary; must not contain Qt objects -- see
            ``widgets/_run_config.build_run_config``).
        folder: the input folder path for this position.
        output_dir: the resolved ``TFM_data/`` output directory for this
            position (one entry of ``OutputPlan.output_dirs``).
        queue: the shared progress queue -- a ``multiprocessing.Manager().Queue()``
            proxy, not a plain ``multiprocessing.Queue`` (see
            :meth:`BatchAnalysis.start_parallel` for why) -- created once by
            ``start_parallel`` and shared by every worker this run submitted,
            for this worker's ``QueueProgressSink`` to report progress on.

    Returns:
        ``(folder, status, error_message)`` where ``status`` is ``"done"``
        or ``"error"``, and ``error_message`` is ``None`` on success or the
        stringified exception on failure. Never raises: a failure inside
        ``process_folder`` is caught and reported as the ``"error"`` status
        instead, so a single bad position can't take down the worker (which
        would otherwise surface as a ``BrokenProcessPool`` for every other
        future in the pool).
    """
    sink = QueueProgressSink(queue, folder)
    analysis = BatchAnalysis(config, sink=sink)
    try:
        analysis.process_folder(folder, output_dir)
    except Exception as e:
        return folder, "error", str(e)
    return folder, "done", None


# TODO black image when only one frame for cell-force overlay visualization
class TeeLogger:
    """Custom logger that captures print statements and logging output to both console and file."""

    def __init__(self, filename: Path, config: dict = None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = open(filename, 'w', encoding='utf-8')
        self.start_time = datetime.now()

        # Write header to log file
        self.print_banner()
        self.log.write(f"Processing started at: {self.start_time}\n")
        self.log.write("-" * 50 + "\n\n")
        self.terminal.write(f"Processing started at: {self.start_time}\n")
        self.terminal.write("-" * 50 + "\n\n")

        # Log configuration parameters if provided
        if config:
            self.log.write("Analysis Parameters:\n")
            self.log.write("-" * 20 + "\n")
            self.terminal.write("Analysis Parameters:\n")
            self.terminal.write("-" * 20 + "\n")

            # Log analysis steps
            self.log.write("\nEnabled Analysis Steps:\n")
            self.terminal.write("\nEnabled Analysis Steps:\n")
            for step, enabled in config.get('analysis_steps', {}).items():
                self.log.write(f"- {step}: {'Yes' if enabled else 'No'}\n")
                self.terminal.write(f"- {step}: {'Yes' if enabled else 'No'}\n")

            # Log key parameters
            self.log.write("\nKParameters:\n")
            self.terminal.write("\nParameters:\n")
            params = config.get('parameters', {})
            for key, value in sorted(params.items()):
                self.log.write(f"- {key}: {value}\n")
                self.terminal.write(f"- {key}: {value}\n")

            self.log.write("\n" + "-" * 50 + "\n\n")
            self.terminal.write("\n" + "-" * 50 + "\n\n")

    def print_banner(self):
        separator = "-" * 80
        banner = '''
 ----------------------------------------------------------------------------- 
|                                                                             |
|                                         ,--.,--------.,------.,--.   ,--.   |
|   ,--,--,  ,--,--. ,---.  ,--,--.,--.--.`--''--.  .--'|  .---'|   `.'   |   |
|   |      \| ,-.  || .-. |' ,-.  ||  .--',--.   |  |   |  `--, |  |'.'|  |   |
|   |  ||  |\ '-'  || '-' '\ '-'  ||  |   |  |   |  |   |  |`   |  |   |  |   |
|   `--''--' `--`--'|  |-'  `--`--'`--'   `--'   `--'   `--'    `--'   `--'   |
|                   `--'                                                      |
|                                                                             |
|                   Traction Force Microscopy Analysis Tool                   |
 ----------------------------------------------------------------------------- '''

        contact_info = '''                 
            For comments, questions or bug reports, please contact:
                           artur.ruppel@pasteur.fr
                    https://github.com/ArturRuppel/napariTFM'''

        # Combined output for both terminal and log file
        output = (
            f"\n{separator}\n"
            f"{banner}\n"
            f"{contact_info}\n"
            f"{separator}\n\n"
        )

        # Write to both outputs
        self.terminal.write(output)
        self.log.write(output)

        # Flush both outputs to ensure immediate writing
        self.flush()

        # Only pause the terminal output
        sleep(2)

    def write(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        self.terminal.write(formatted_message)
        self.log.write(formatted_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        end_time = datetime.now()
        duration = end_time - self.start_time

        # Write footer with timing information
        self.log.write("\n" + "-" * 50 + "\n")
        self.log.write("Analysis Summary:\n")
        self.log.write(f"Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"Duration: {duration}\n")

        self.terminal.write("\n" + "-" * 50 + "\n")
        self.terminal.write("Analysis Summary:\n")
        self.terminal.write(f"Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.terminal.write(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.terminal.write(f"Duration: {duration}\n")

        # Calculate hours, minutes, seconds for more readable format
        total_seconds = int(duration.total_seconds())  # Convert to integer for second resolution
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        time_str = ""
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        self.log.write(f"Total time: {time_str}\n")
        self.terminal.write(f"Total time: {time_str}\n")

        self.log.close()
        sys.stdout = self.terminal


class _BatchCancelled(Exception):
    """Raised internally to unwind out of an in-flight folder when the user
    cancels a Run-selected. Kept distinct from stage failures so it is never
    recorded as an ``error`` — the folder loop catches it and reports
    ``"cancelled"`` instead. See :meth:`BatchAnalysis._raise_if_cancelled`.
    """


class BatchAnalysis:
    """Handles batch analysis of TFM data using service layer components."""

    def __init__(self, config: dict, progress_callback=None, sink=None):
        self.config = config
        self._tee_logger = None
        # Optional per-folder lifecycle hook (P4): called as
        # ``callback(folder_path, status)`` with status in
        # {"running", "done", "error"} so a live UI can walk the rail.
        self._progress_callback = progress_callback
        # Optional stage/frame observation surface (worklist §5): a live napari
        # ``ViewerSink`` streams each stage into the viewer as it runs, so an
        # in-napari run-all and a headless run share this one code path. A
        # headless run leaves this ``None`` and the hooks are silent no-ops.
        self._sink = sink
        self._cancelled = False
        # Parallel-mode pool state (populated by start_parallel).
        self._executor = None
        self._pending_futures = {}
        self._progress_manager = None
        self._progress_queue = None
        # Per-folder record of stages that raised. process_folder resets this at
        # the start of each folder and raises at the end if it is non-empty, so a
        # swallowed stage exception surfaces as "error" instead of a silent "done".
        self._stage_failures = []

    def _record_stage_failure(self, stage: str, error: Exception) -> None:
        """Note that ``stage`` raised, so process_folder can report the folder
        as failed rather than green-with-nothing-on-disk. The exception is still
        caught (other stages' partial results are written) but not forgotten."""
        if not hasattr(self, "_stage_failures"):
            self._stage_failures = []
        self._stage_failures.append(f"{stage}: {error}")

    def _emit(self, method: str, *args) -> None:
        """Notify the optional sink; never let it break a run (worklist §5).

        Mirrors :meth:`_report_progress`: the sink is purely additive
        observation, so a hook that raises is logged and swallowed rather than
        aborting the compute.
        """
        sink = getattr(self, "_sink", None)
        if sink is None:
            return
        try:
            getattr(sink, method)(*args)
        except Exception as e:
            print(f"Pipeline sink error in {method}: {str(e)}")

    def request_cancel(self) -> None:
        """Ask the batch to stop as soon as possible (cooperative).

        Set by a Cancel control during a Run-selected. In the sequential path the
        flag is checked at every stage boundary and at the top of every per-frame
        streaming loop (:meth:`_raise_if_cancelled`), so an in-flight folder stops
        within roughly one frame rather than running to completion — the fix for
        "a single running experiment can't be cancelled". In the parallel path it
        still only prevents not-yet-started folders from being submitted (a
        running subprocess is not killed).
        """
        self._cancelled = True

    def _raise_if_cancelled(self) -> None:
        """Cooperative cancellation checkpoint: raise :class:`_BatchCancelled` if a
        cancel has been requested. Called at stage boundaries and inside the
        per-frame loops; the folder loop catches it and reports ``"cancelled"``.
        """
        if getattr(self, "_cancelled", False):
            raise _BatchCancelled()

    def _format_duration(self, seconds: float) -> str:
        """Format duration in appropriate units (seconds or minutes)."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        return f"{seconds/60:.1f} minutes"

    def process_all_folders(self) -> None:
        """
        Process all folders specified in the configuration for TFM analysis.

        This is the main entry point for batch processing multiple experiment folders.
        It iterates through each folder path specified in config['root_folders'] and
        processes them either one at a time (``config['num_workers'] <= 1``, the
        default -- see :meth:`_process_all_folders_sequential`) or in parallel
        across a process pool (``config['num_workers'] > 1`` -- see
        :meth:`_process_all_folders_parallel`).

        The method handles:
        - Preprocessing of bead and cell images
        - Displacement field calculation
        - Force analysis
        - Stress analysis (consumes externally supplied masks)
        - Visualization generation

        Each experiment's derived output is saved under the resolved ``TFM_data/``
        bucket: one ``.ntfm`` data artifact plus ``figures/`` and a
        ``batch.log`` capturing the processing steps and any issues encountered.

        Raises:
            FileNotFoundError: If a specified folder doesn't exist
            RuntimeError: If processing fails for any folder
        """
        plan = resolve_output_plan(
            self.config['root_folders'],
            self.config.get('processed_root'),
        )
        for warning in plan.warnings:
            print(f"WARNING: {warning}")

        num_workers = int(self.config.get('num_workers', 1) or 1)
        if num_workers <= 1:
            self._process_all_folders_sequential(plan)
        else:
            self._process_all_folders_parallel(plan, num_workers)

    def _process_all_folders_sequential(self, plan) -> None:
        """Process every folder one at a time, in-process (the original,
        unchanged ``process_all_folders`` loop, extracted verbatim).

        The ``ViewerSink`` (constructed by the caller, not this class) still
        drives live frame-by-frame streaming exactly as before, via
        ``self._emit(...)``.
        """
        for folder in self.config['root_folders']:
            if getattr(self, "_cancelled", False):
                self._report_progress(folder, "cancelled")
                break
            self._report_progress(folder, "running")
            # Tell a live sink which position is now streaming so the viewer +
            # experiments-list selection follow the rail (worklist §3).
            self._emit('experiment_started', folder)
            try:
                self.process_folder(folder, plan.output_dirs[folder])
            except _BatchCancelled:
                print(f"Folder {folder} cancelled")
                self._report_progress(folder, "cancelled")
                break
            except Exception as e:
                print(f"Folder {folder} failed: {str(e)}")
                self._report_progress(folder, "error")
                continue
            self._report_progress(folder, "done")

    def _process_all_folders_parallel(self, plan, num_workers: int) -> None:
        """Blocking parallel variant for CLI/headless/test callers.

        Reuses :meth:`start_parallel` + :meth:`poll_parallel_progress` -- the
        same non-blocking primitives the GUI's run-all flow consumes -- so
        there is exactly one code path that knows how to run a process pool
        over the folders. This method just polls that path to exhaustion
        instead of returning right after submission, and so also inherits
        its cancellation handling for free.
        """
        self.start_parallel(plan, num_workers)
        finished = False
        while not finished:
            _events, _stage_events, finished = self.poll_parallel_progress()
            if not finished:
                sleep(0.05)

    def start_parallel(self, plan, num_workers: int) -> None:
        """Submit one headless task per folder to a process pool and return
        immediately (non-blocking; for GUI callers that poll progress via
        :meth:`poll_parallel_progress` from e.g. a Qt timer).

        Uses a ``ProcessPoolExecutor`` (spawn start method -- see module
        docstring/plan rationale: forking a GUI process with live Qt/BLAS/
        OpenMP threads is a deadlock hazard) rather than threads, because the
        pipeline is CPU-bound (numba/numpy/scipy) and would not parallelize
        across a single GIL.

        Futures are submitted in ``self.config['root_folders']`` order (top
        positions first). A pool with ``max_workers=num_workers`` starts the
        first N immediately and pulls the next queued folder in FIFO order as
        a slot frees, which alone satisfies "top positions first" -- no
        explicit priority queue is needed.

        Also creates one shared queue -- via a ``multiprocessing.Manager``
        (same spawn context as the pool), not a plain ``ctx.Queue()`` -- for
        every worker this run submitted, and passes it into each
        ``_run_position_headless`` call so a worker's ``QueueProgressSink``
        can report real per-stage/per-frame progress back across the process
        boundary; :meth:`poll_parallel_progress` drains it every call. A
        plain ``ctx.Queue()`` cannot be handed to an already-running pool
        this way: ``ProcessPoolExecutor`` ships submitted args to workers
        through its own internal call queue, which re-pickles them outside
        the normal process-launch "inheritance" a raw ``Queue`` requires --
        that raises ``RuntimeError: Queue objects should only be shared
        between processes through inheritance``. A ``Manager().Queue()`` is
        a proxy (a thin client to a small server process the ``Manager``
        owns) and pickles like any other object, so it survives that trip.

        Stores the executor, a ``{Future: folder}`` map, the manager, and the
        progress queue on ``self._executor`` / ``self._pending_futures`` /
        ``self._progress_manager`` / ``self._progress_queue`` for
        :meth:`poll_parallel_progress` to drain (and, once every future is
        accounted for, to shut the manager down alongside the executor).
        """
        ctx = multiprocessing.get_context("spawn")
        self._executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
        self._pending_futures: dict[Future, str] = {}
        self._progress_manager = ctx.Manager()
        self._progress_queue = self._progress_manager.Queue()
        for folder in self.config['root_folders']:
            output_dir = str(plan.output_dirs[folder])
            future = self._executor.submit(
                _run_position_headless, self.config, folder, output_dir, self._progress_queue,
            )
            self._pending_futures[future] = folder
            # "running" at submission time is an acceptable approximation (a
            # queued-but-not-yet-started task reports "running" slightly
            # early), matching the "running" semantics used elsewhere in this
            # file -- the queue above reports real per-stage progress once the
            # worker actually starts.
            self._report_progress(folder, "running")

    def poll_parallel_progress(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, Optional[float]]], bool]:
        """Drain whatever parallel-mode futures have completed, and whatever
        per-stage progress messages have arrived, since the last call --
        without blocking. Intended to be called repeatedly (e.g. from a Qt
        timer) until ``finished`` is ``True``.

        Returns:
            ``(events, stage_events, finished)`` --

            - ``events``: a list of ``(folder, status)`` pairs that newly
              completed during *this* call, where ``status`` is one of
              ``"done"``, ``"error"``, or ``"cancelled"``. Each pair is also
              reported through :meth:`_report_progress` (the same hook the
              sequential path already uses), so a caller only needs to
              listen on one channel; the returned list is for callers (e.g.
              the GUI) that want to react inline without a callback, such as
              reloading a finished position's ``.ntfm`` from disk.
            - ``stage_events``: a list of ``(folder, stage, status, fraction)``
              tuples drained from the shared progress queue (one entry per
              ``QueueProgressSink`` message a worker put since the last poll)
              -- ``status`` is ``"running"`` (with a growing ``fraction``) or
              ``"done"`` (``fraction`` is then ``None``), mirroring
              ``ViewerSink.on_stage_progress``'s shape. Always drained fully,
              even on a call where ``events``/``finished`` report nothing new.
            - ``finished``: ``True`` once every submitted folder has been
              accounted for (no futures left pending after this call's
              bookkeeping). When it flips to ``True`` the executor is shut
              down via ``shutdown(wait=False)`` (non-blocking -- workers are
              already done by construction at that point).

        Cancellation: checks ``self._cancelled`` (set by
        :meth:`request_cancel`) on every call. When set, every not-yet-started
        future has ``.cancel()`` called on it -- this only succeeds for
        futures still queued (``Future.cancel()`` returns ``True``/``False``)
        -- and those folders are reported/returned with status
        ``"cancelled"``. There is no new submission after :meth:`start_parallel`
        runs (everything is submitted up front), so "stop submitting" is
        automatic. Already-running workers are *not* forcefully terminated;
        they finish naturally and report their real ``done``/``error`` status
        on a later poll.
        """
        stage_events: list[tuple[str, str, str, Optional[float]]] = []
        progress_queue = getattr(self, "_progress_queue", None)
        if progress_queue is not None:
            while True:
                try:
                    stage_events.append(progress_queue.get_nowait())
                except Empty:
                    break

        events: list[tuple[str, str]] = []

        if getattr(self, "_cancelled", False):
            for future in list(self._pending_futures):
                if future.cancel():
                    folder = self._pending_futures.pop(future)
                    self._report_progress(folder, "cancelled")
                    events.append((folder, "cancelled"))

        if self._pending_futures:
            done, _pending = futures_wait(list(self._pending_futures), timeout=0)
            for future in done:
                folder = self._pending_futures.pop(future)
                try:
                    _folder, status, _err = future.result()
                except Exception as e:
                    # A worker process crashed outright (e.g. BrokenProcessPool)
                    # rather than returning its usual (folder, status, err)
                    # tuple -- still report this folder as failed instead of
                    # raising out of a non-blocking poll.
                    status = "error"
                    print(f"Parallel worker for {folder} failed: {str(e)}")
                self._report_progress(folder, status)
                events.append((folder, status))

        finished = not self._pending_futures
        if finished and getattr(self, "_executor", None) is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if finished and getattr(self, "_progress_manager", None) is not None:
            # Every future is accounted for, so every worker has already
            # returned -- any ``QueueProgressSink.put()`` it was going to do
            # already happened, and the drain above at the top of this call
            # (or a prior one) collected it. Safe to tear down the Manager's
            # server process now rather than leak it past this run.
            self._progress_manager.shutdown()
            self._progress_manager = None

        return events, stage_events, finished

    def _report_progress(self, folder: str, status: str) -> None:
        """Notify the optional progress callback; never let it break a run."""
        callback = getattr(self, "_progress_callback", None)
        if callback is None:
            return
        try:
            callback(folder, status)
        except Exception as e:
            print(f"Progress callback error: {str(e)}")

    def process_folder(self, folder_path: str, output_dir=None) -> None:
        """
        Process a single folder containing TFM experiment data.

        This method executes the complete TFM analysis pipeline on a single
        experimental dataset. Derived output (the ``.ntfm`` data artifact,
        ``figures/`` previews, ``batch.log``, and any stage-resume cache) is
        written under ``output_dir`` — the ``TFM_data/`` bucket resolved by
        :func:`resolve_output_plan`. When ``output_dir`` is not
        supplied it is resolved in-place (a ``TFM_data/`` sibling of the
        input folder).

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the raw experimental data.
            Must include the files specified in config['input_files'].

        Processing Steps
        ---------------
        1. Preprocessing:
            - Processes bead and cell images if available
            - Applies background subtraction and filtering
            - Optionally caches preprocessed images as calibrated TIFFs

        2. Displacement Analysis:
            - Calculates displacement fields from bead images

        3. Force Analysis:
            - Computes traction forces using FTTC

        4. Stress Analysis:
            - Calculates internal stress fields

        Displacement, force, and stress are persisted only in the experiment's
        ``.ntfm``, not as standalone ``.npy`` files.

        Each step is conditional on the corresponding flag in config['analysis_steps']
        being True. Visualizations are generated based on config['visualizations'].

        Results
        -------
        Writes derived output under the resolved ``TFM_data/`` bucket:
            - <experiment>.ntfm: the sole data artifact (tidy table + metadata)
            - figures/: per-stage PNG/GIF previews (human-facing, not data)
            - batch.log: detailed processing log
            - preprocessed_*.tif: the one stage-resume cache item not held by the
              .ntfm (images upstream of the analysis grid), always written.
              Displacement/force/stress are not re-cached as .npy — resume
              reads them from the .ntfm. The external mask is an input read from
              the input folder.

        Raises
        ------
        FileNotFoundError
            If required input files are missing
        RuntimeError
            If any processing step fails
        ValueError
            If input data is invalid or corrupted
        """
        folder = Path(folder_path)
        if output_dir is None:
            plan = resolve_output_plan([folder_path], self.config.get('processed_root'))
            output_dir = plan.output_dirs[folder_path]
        output_dir = Path(output_dir)

        tfm_folder = self._initialize_folder(output_dir)
        viz_saver = BatchVisualizationSaver(output_dir)
        self._stage_failures = []

        try:
            print(f"Processing folder: {folder_path}")
            print("=" * 50)

            # Handle preprocessing
            preprocessed_data = self._handle_preprocessing_execution(folder, tfm_folder)
            self._handle_visualization(tfm_folder, viz_saver, 'preprocessing', preprocessed_data)

            # Handle displacement
            displacement_data = self._handle_displacement_execution(tfm_folder, preprocessed_data)
            self._handle_visualization(tfm_folder, viz_saver, 'displacement', displacement_data)

            # Handle force analysis
            force_data = self._handle_force_execution(tfm_folder, folder, displacement_data)
            self._handle_visualization(tfm_folder, viz_saver, 'force', force_data)

            # Masks are supplied externally as an input layer:
            # loaded from the input folder, never generated by napariTFM.
            mask_data = self._load_mask(folder)

            # Handle stress analysis
            stress_data = self._handle_stress_execution(tfm_folder, folder, force_data, mask_data)
            self._handle_visualization(tfm_folder, viz_saver, 'stress', stress_data)

            # Write the sole data artifact: one .ntfm per experiment.
            # Written *before* the failure check below so any stages that did
            # succeed are still persisted on a partial failure.
            self._write_experiment_ntfm(
                output_dir, folder, displacement_data, force_data, stress_data, mask_data
            )

            # A stage that raised was caught so downstream stages and the write
            # could still run, but a folder with any failed stage did not fully
            # succeed and must not be reported "done".
            if self._stage_failures:
                raise RuntimeError(
                    "Folder processing failed for stage(s): "
                    + "; ".join(self._stage_failures)
                )

            print("Folder processing completed successfully!")
            print("=" * 50)

        finally:
            self._cleanup()

    def _guard_stage(self, stage: str, body) -> Optional[dict]:
        """Run a stage's ``body`` iff enabled; record any exception as a failure.

        Returns ``body()``'s result, ``None`` if the stage is disabled, or
        ``None`` after recording a failure if it raised. Centralizing the
        enabled-check + failure-recording in one place is what guarantees every
        enabled stage that raises is reported as an ``error`` (not a silent
        "done") — the four handlers used to each re-implement this, and the one
        that swallowed its own exception (stress) reintroduced that regression. A graceful ``return None`` from ``body``
        (disabled upstream, no mask, nothing to resume) is a deliberate skip and
        is left untouched.
        """
        self._raise_if_cancelled()          # stage-boundary cancel: don't start this stage
        if not self.config['analysis_steps'][stage]:
            return None
        try:
            return body()
        except _BatchCancelled:
            raise                            # a cancel is not a stage failure — let it unwind
        except Exception as e:
            print(f"{stage.capitalize()} failed: {str(e)}")
            self._record_stage_failure(stage, e)
            return None

    def _handle_preprocessing_execution(self, folder: Path, tfm_folder: Path) -> Optional[dict]:
        """Handle preprocessing execution. Always runs if enabled."""
        return self._guard_stage(
            "preprocessing", lambda: self._execute_preprocessing(folder, tfm_folder)
        )

    def _handle_displacement_execution(self, tfm_folder: Path, preprocessed_data: Optional[dict]) -> Optional[dict]:
        """Handle displacement analysis execution. Always runs if enabled.

        ``preprocessed_data`` is ignored: displacement always consumes the
        *persisted* preprocessed tiffs, never the in-session float arrays.
        Preprocessing always writes the calibrated tiffs (see
        ``_execute_preprocessing``), so reading them back here means a fresh run
        and a stage-resume feed Farneback byte-identical inputs — consistent with
        the project-wide "calcs always read from disk" invariant. A missing/unreadable tiff raises and is
        recorded as a displacement failure by ``_guard_stage``.
        """
        def body():
            print("Loading preprocessed images from file...")
            preprocessed = {
                'beads': tifffile.imread(str(tfm_folder / "preprocessed_beads.tif")),
                'reference': tifffile.imread(str(tfm_folder / "preprocessed_reference.tif")),
            }
            return self._execute_displacement_analysis(tfm_folder, preprocessed)

        return self._guard_stage("displacement", body)

    def _handle_force_execution(self, tfm_folder: Path, folder: Path, displacement_data: Optional[dict]) -> Optional[dict]:
        """Handle force analysis execution. Always runs if enabled."""
        def body():
            dd = displacement_data
            if dd is None:
                # Stage-resume: read the displacement field back from the prior
                # run's .ntfm (no intermediate .npy). Force analysis
                # consumes only ``.displacement_field``, so a one-field shim is
                # sufficient.
                print("Resuming displacement from existing .ntfm...")
                field = self._resume_field_from_ntfm(tfm_folder, folder, "displacement_field")
                if field is None:
                    return None
                dd = SimpleNamespace(displacement_field=field)
            return self._execute_force_analysis(tfm_folder, dd)

        return self._guard_stage("force", body)

    def _handle_stress_execution(self, tfm_folder: Path, folder: Path, force_data: Optional[dict], mask_data: Optional[np.ndarray]) -> Optional[dict]:
        """Handle stress analysis execution. Always runs if enabled."""
        def body():
            fd = force_data
            if fd is None:
                # Stage-resume: read the force field back from the prior run's
                # .ntfm. Stress analysis consumes only ``.force_field``.
                print("Resuming force from existing .ntfm...")
                field = self._resume_field_from_ntfm(tfm_folder, folder, "force_field")
                if field is None:
                    return None
                fd = SimpleNamespace(force_field=field)

            if mask_data is None:
                print("Stress analysis requires an external mask; napariTFM does "
                      "not generate masks. No mask was found for this experiment, "
                      "skipping stress analysis.")
                return None

            return self._execute_stress_analysis(tfm_folder, mask_data, fd)

        return self._guard_stage("stress", body)

    def _resume_field_from_ntfm(self, tfm_folder: Path, folder: Path, field_key: str) -> Optional[np.ndarray]:
        """Read a grid field back from the experiment's ``.ntfm`` for stage-resume.

        The ``.ntfm`` is the sole persisted result; when an upstream
        stage was skipped this run, the field is reconstructed from a prior run's
        container. ``field_key`` is one of :func:`ntfm.read_series_ntfm`'s array
        keys (e.g. ``displacement_field``, ``force_field``). Returns ``None``
        (with a message) if the container or the field is missing.
        """
        ntfm_path = tfm_folder / RESULTS_FILENAME
        if not ntfm_path.exists():
            print(f"No existing container to resume from at {ntfm_path}.")
            return None
        try:
            arrays, _, _, _ = ntfm.read_series_ntfm(ntfm_path)
            field = arrays.get(field_key)
        except Exception as e:
            print(f"Could not read {field_key} from {ntfm_path}: {str(e)}")
            return None
        # The writer emits every measure column (NaN when a stage wasn't run), so
        # an all-NaN field means the stage was never computed — treat as absent.
        if field is None or np.all(np.isnan(field)):
            print(f"{field_key} not available in {ntfm_path}.")
            return None
        return field

    def _load_mask(self, folder: Path) -> Optional[np.ndarray]:
        """Load the externally supplied mask from the input folder, or ``None``.

        Masks are a pure external input — napariTFM never generates
        them. The filename defaults to ``masks.tif`` and can be overridden via
        ``config['input_files']['masks']``.
        """
        mask_name = self.config.get('input_files', {}).get('masks', 'masks.tif')
        mask_path = folder / mask_name
        if not mask_path.exists():
            print(f"No external mask found at {mask_path}; stress analysis will be skipped.")
            return None
        try:
            return tifffile.imread(str(mask_path))
        except Exception as e:
            print(f"Could not load mask data from {mask_path}: {str(e)}")
            return None

    def _write_experiment_ntfm(
        self,
        output_dir: Path,
        folder: Path,
        displacement_result: Optional[DisplacementResult],
        force_result: Optional[FTTCResult],
        stress_result,
        mask_data: Optional[np.ndarray],
    ) -> None:
        """Write the experiment's sole data artifact: ``<experiment>.ntfm``.

        One container holds the tidy long-format table (displacement / force /
        stress on the shared analysis grid) plus run-level provenance, the
        resolved config, and the per-experiment ``labels``. This is the sole
        persisted form of the results — the scattered ``.npy`` files are gone;
        stage-resume reads displacement/force back from this container.
        """
        labels = (self.config.get('labels') or {}).get(str(folder), {})
        ntfm_path = output_dir / RESULTS_FILENAME
        # Delegate to the one shared writer (also used by interactive per-stage
        # runs), so batch- and live-saved containers are identical.
        try:
            written = write_experiment_ntfm(
                ntfm_path,
                parameters=self.config.get('parameters', {}),
                displacement_result=displacement_result,
                force_result=force_result,
                stress_result=stress_result,
                mask=mask_data,
                folder=folder,
                input_files=self.config.get('input_files', {}),
                labels=labels,
                apply_mask_on_save=bool(self.config.get('apply_mask_on_save', False)),
            )
        except Exception as e:
            # The .ntfm is the sole persisted artifact: if it cannot
            # be written the folder produced nothing durable, so this must not be
            # swallowed. Re-raise so process_all_folders reports the folder as
            # "error" (red dot) instead of "done" with no output on disk.
            print(f"Could not write .ntfm container: {str(e)}")
            raise
        if written is None:
            print("No analysis results produced; skipping .ntfm write.")
            return
        print(f"Saved data artifact: {ntfm_path}")

    def _execute_preprocessing(self, folder: Path, tfm_folder: Path) -> Optional[dict]:
        """
        Execute the preprocessing step of the TFM analysis pipeline.

        This method handles the initial processing of raw microscopy images,
        including both bead and cell images if available.

        Parameters
        ----------
        folder : Path
            Path to the input folder containing raw data files
        tfm_folder : Path
            Path to the output folder where processed files will be saved

        Returns
        -------
        Optional[dict]
            Dictionary containing:
            - 'beads': Preprocessed bead image stack (np.ndarray)
            - 'reference': Preprocessed reference image (np.ndarray)
            - 'cells': Preprocessed cell image stack (np.ndarray, optional)
            - 'parameters': Preprocessing parameters used
            Returns None if preprocessing fails

        Processing Steps
        ---------------
        1. Loads raw bead images and reference image
        2. Optionally loads cell images if specified in config
        3. Applies preprocessing pipeline:
            - Gaussian filtering
            - Intensity normalization
            - Image registration (for bead images)
        4. Saves results as calibrated TIFF files with metadata

        The preprocessing parameters are taken from the config:
            - min_intensity_percentile
            - max_intensity_percentile
            - gaussian_sigma
            - registration_mode
            Plus additional parameters for cell image processing

        Raises
        ------
        FileNotFoundError
            If input files are not found
        RuntimeError
            If preprocessing operations fail
        """

        print("Starting Preprocessing...")
        start_time = time()
        params = self._create_preprocessing_parameters()

        # Process bead images
        bead_stack = tifffile.imread(str(folder / self.config['input_files']['beads']))
        reference = tifffile.imread(str(folder / self.config['input_files']['reference']))

        # Load and process cell images if available
        cell_stack = None
        if 'cells' in self.config['input_files'] and self.config['input_files']['cells']:
            try:
                cell_stack = tifffile.imread(str(folder / self.config['input_files']['cells']))
                print("Found cell image stack, will process alongside beads")
            except FileNotFoundError:
                print(f"Warning: Cell image file specified but not found: {self.config['input_files']['cells']}")

        # Tell a live sink the channel shapes up front so it can pre-allocate
        # the Preprocessed* stacks and bind their layers before frames stream in.
        # The announced frame total is the whole preprocessing workload — beads +
        # the single reference frame + cells — so the progress bar (which the sink
        # now advances monotonically across the three channels) reaches 100% only
        # when every channel is done, instead of filling on beads and snapping back.
        total_preproc_frames = (
            int(bead_stack.shape[0])
            + 1
            + (int(cell_stack.shape[0]) if cell_stack is not None else 0)
        )
        self._emit('stage_started', 'preprocessing', total_preproc_frames, {
            'beads_shape': tuple(bead_stack.shape),
            'reference_shape': tuple(reference.shape),
            'cells_shape': tuple(cell_stack.shape) if cell_stack is not None else None,
        })

        bead_results = []
        for result, frame, total in preprocess_stack(bead_stack, params, reference):
            self._raise_if_cancelled()
            bead_results.append(result)
            print(f"Progress (beads): {(frame / total) * 100:.1f}%, Frame {frame}/{total}")
            # ``preprocess_stack`` yields a 0-based frame (the stack index).
            self._emit('stage_frame', 'preprocessing', frame, {'beads': result.processed_image})

        reference_result = preprocess_frame(reference, params)
        self._emit('stage_frame', 'preprocessing', 0, {'reference': reference_result.processed_image})

        # Process cell images if available
        cell_results = []
        if cell_stack is not None:
            print("Processing cell images...")
            for result, frame, total in preprocess_stack(cell_stack, params, reference_image=None, is_cell=True):
                self._raise_if_cancelled()
                cell_results.append(result)
                print(f"Progress (cells): {(frame / total) * 100:.1f}%, Frame {frame}/{total}")
                self._emit('stage_frame', 'preprocessing', frame, {'cells': result.processed_image})

        # Save results with calibration
        preprocessed = {
            'beads': np.stack([r.processed_image for r in bead_results]),
            'reference': reference_result.processed_image,
            'parameters': params.__dict__
        }

        if cell_results:
            preprocessed['cells'] = np.stack([r.processed_image for r in cell_results])

        # Preprocessed images give preprocessing the same persistence guarantee
        # every downstream stage has: they are always written so that the
        # preprocessing dot reliably shows "done" on reload and stage-resume can
        # skip straight to displacement without re-running preprocessing.
        pixel_size = self.config['parameters']['pixel_size']
        frame_interval = self.config['parameters']['frame_interval']

        save_preprocessed_tiffs(
            tfm_folder,
            pixel_size,
            frame_interval,
            beads=preprocessed['beads'],
            reference=preprocessed['reference'],
            cells=preprocessed.get('cells'),
        )

        self._emit('stage_finished', 'preprocessing', preprocessed)
        print(f"Preprocessing completed in {self._format_duration(time() - start_time)}")
        return preprocessed

    def _execute_displacement_analysis(self, tfm_folder: Path, preprocessed_data: dict) -> Optional[DisplacementResult]:
        """
        Execute the displacement analysis step of the TFM analysis pipeline.

        This method calculates displacement fields from preprocessed bead images
        using optical flow techniques.

        Parameters
        ----------
        tfm_folder : Path
            Path to the output folder where processed files will be saved
        preprocessed_data : dict
            Dictionary containing preprocessed images:
            - 'beads': Preprocessed bead image stack (np.ndarray)
            - 'reference': Preprocessed reference image (np.ndarray)

        Returns
        -------
        Optional[DisplacementResult]
            Object containing:
            - displacement_field: Calculated displacement vectors (np.ndarray)
            - parameters: Displacement calculation parameters used
            Returns None if displacement analysis fails

        Processing Steps
        ---------------
        1. Loads raw bead images and reference image
        2. Optionally loads cell images if specified in config
        3. Applies preprocessing pipeline:
            - Background subtraction
            - Optical flow calculation (Farneback)
            - Optional downscaling and filtering
        4. Returns the displacement field (persisted later in the .ntfm)

        The displacement parameters are taken from the config:
            - nscales, inner_iterations, median_filtering (Farneback parameters)
            - downscale_factor, pixel_size

        Raises
        ------
        FileNotFoundError
            If input files are not found
        RuntimeError
            If displacement calculation fails
        """
        print("Starting Displacement Analysis...")
        start_time = time()

        disp_params = self._create_displacement_parameters()
        beads = preprocessed_data['beads']
        self._emit('stage_started', 'displacement', int(beads.shape[0]), {
            'v_max': disp_params.d_max,
            'vector_stride': disp_params.disp_vector_stride,
            'arrow_scale': disp_params.disp_arrow_scale,
            'downscale_factor': disp_params.downscale_factor,
        })

        displacement_field_generator = calculate_displacement_field(
            preprocessed_data['reference'],
            beads,
            disp_params,
        )

        # Initialize result container
        try:
            while True:
                self._raise_if_cancelled()
                # Get next intermediate result
                displacement_field, frame, total = next(displacement_field_generator)
                self._log_stage_progress(displacement_field, frame, total, "displacement", "µm")
                # The backend yields a 1-based frame number; the viewer (and its
                # slider) want it 0-based.
                self._emit('stage_frame', 'displacement', frame - 1, displacement_field)
        except StopIteration as e:
            # Retrieve final result from generator's return value
            displacement_result = e.value

        if displacement_result is None:
            raise RuntimeError("Displacement calculation failed")

        # No intermediate .npy: the sole persisted result is the experiment's
        # .ntfm, written once all stages finish. Stage-resume reads
        # the displacement field back from that .ntfm.
        self._emit('stage_finished', 'displacement', displacement_result)
        print(f"Displacement analysis completed in {self._format_duration(time() - start_time)}")
        return displacement_result

    def _execute_force_analysis(self, tfm_folder: Path, displacement_data: DisplacementResult) -> Optional[dict]:
        """
        Execute the force analysis step of the TFM analysis pipeline.

        This method implements Fourier Transform Traction Cytometry (FTTC) to
        calculate traction forces from displacement fields.

        Parameters
        ----------
        tfm_folder : Path
            Path to the output folder where results will be saved
        displacement_data : DisplacementResult
            Object containing:
            - displacement_field: Displacement vectors
            - parameters: Displacement calculation parameters

        Returns
        -------
        Optional[dict]
            Dictionary containing:
            - force_field: Calculated traction forces (np.ndarray)
            - parameters: Force calculation parameters
            Returns None if analysis fails

        Processing Steps
        ---------------
        1. Prepares displacement data for FTTC
        2. Performs force calculation:
            - Fourier transform of displacement field
            - Application of Green's function
            - Regularization
            - Inverse transform
        3. Returns the force field (persisted later in the .ntfm)

        The force calculation parameters are taken from the config:
            - young_modulus
            - poisson_ratio_substrate
            - gel_height
            - regularization
            And other FTTC parameters

        Notes
        -----
        Progress updates are logged during processing, including:
            - Frame-by-frame completion status
            - Mean and max force values
            - Processing time

        Raises
        ------
        RuntimeError
            If force calculation fails
        ValueError
            If input data is invalid or parameters are out of range
        """
        print("Starting Force Analysis...")
        start_time = time()

        fttc_params = self._create_fttc_parameters()
        self._emit('stage_started', 'force',
                   int(displacement_data.displacement_field.shape[0]), {
                       'v_max': fttc_params.f_max,
                       'vector_stride': fttc_params.force_vector_stride,
                       'arrow_scale': fttc_params.force_arrow_scale,
                       'downscale_factor': fttc_params.downscale_factor,
                   })

        force_generator = calculate_force_field(
            displacement_data.displacement_field,
            fttc_params
        )

        # Initialize result container
        try:
            while True:
                self._raise_if_cancelled()
                # Get next intermediate result
                force_field, frame, total = next(force_generator)
                self._log_stage_progress(force_field, frame, total, "force", "Pa")
                # 1-based from the backend; the viewer wants it 0-based.
                self._emit('stage_frame', 'force', frame - 1, force_field)
        except StopIteration as e:
            # Retrieve final result from generator's return value
            force_result = e.value

        if force_result is None:
            raise RuntimeError("Force calculation failed")

        # No intermediate .npy: the .ntfm is the sole persisted
        # result; stage-resume reads the force field back from it.
        self._emit('stage_finished', 'force', force_result)
        print(f"Force analysis completed in {self._format_duration(time() - start_time)}")
        return force_result

    def _execute_stress_analysis(self, tfm_folder: Path, mask_data: np.ndarray, force_data: FTTCResult) -> Optional[dict]:
        """
        Execute the stress analysis step of the TFM analysis pipeline.

        Uses Bayesian Inversion Stress Microscopy (BISM, mesh-free) to infer
        internal stress fields within cell monolayers from the traction field.

        Parameters
        ----------
        tfm_folder : Path
            Path to the output folder where results will be saved
        mask_data : np.ndarray
            Binary masks defining cell regions
        force_data : FTTCResult
            Object containing:
            - force_field: Traction forces
            - parameters: Force calculation parameters

        Returns
        -------
        Optional[dict]
            Dictionary containing:
            - stress_tensor: Calculated stress tensors (np.ndarray)
            - parameters: Stress calculation parameters
            Returns None if analysis fails

        Notes
        -----
        Progress updates are logged during processing, including:
            - Frame-by-frame completion status
            - Mean and max stress values
            - Processing time

        Raises
        ------
        RuntimeError
            If stress calculation fails
        ValueError
            If input data is invalid
        """
        print("Starting Stress Analysis...")
        start_time = time()

        params = self._create_stress_parameters()

        # Ensure mask_data is 3D (t, y, x)
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, ...]

        # Resize masks to exactly match force field shape
        force_shape = force_data.force_field.shape[1:3]  # (height, width)
        mask_shape = mask_data.shape[1:3]  # (height, width)

        if mask_shape != force_shape:
            print(f"Resizing masks from {mask_shape} to {force_shape} to match force field...")
            mask_data = np.stack([
                resize(
                    mask.astype(float),
                    force_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5
                for mask in mask_data
            ])
            print(f"After resize - Mask pixels > 0: {np.sum(mask_data > 0)}")

        # NOTE: errors propagate (like force/displacement) so _handle_stress_execution
        # records the failure and process_folder reports 'error'. An inner
        # `except Exception: return None` here previously swallowed even the
        # RuntimeError below, so a real BISM failure was reported as success —
        # a silent recurrence of the swallowed-failure bug for the stress stage.
        fr_params = getattr(force_data, 'parameters', None)
        downscale = getattr(fr_params, 'downscale_factor', 1) if fr_params is not None else 1
        self._emit('stage_started', 'stress',
                   int(force_data.force_field.shape[0]), {
                       'max_stress': params.max_stress,
                       'downscale_factor': downscale,
                   })

        stress_generator = calculate_bism_stresses(
            force_field=force_data.force_field,
            masks=mask_data,
            params=params,
        )

        final_result = None
        try:
            while True:
                self._raise_if_cancelled()
                stress_result, frame, total = next(stress_generator)
                self._log_stage_progress(stress_result.stress_tensor, frame, total, "stress", "mN/m")
                # Cumulative stack; newest frame is its last slice. 1-based → 0.
                self._emit('stage_frame', 'stress', frame - 1,
                           stress_result.stress_tensor[-1])
        except StopIteration as e:
            final_result = e.value

        if final_result is None:
            raise RuntimeError("Stress calculation failed")

        # No intermediate .npy: the .ntfm is the sole persisted
        # result.
        self._emit('stage_finished', 'stress', final_result)
        print(f"Stress analysis completed in {self._format_duration(time() - start_time)}")
        return final_result

    def _unified_parameters(self) -> UnifiedParameters:
        """Rebuild the unified parameter set from the config dict.

        Reconstructing UnifiedParameters and delegating to its to_*_parameters
        keeps a single source of truth for field mapping: unknown keys from
        older configs are ignored and missing keys fall back to defaults.
        """
        valid = {field.name for field in fields(UnifiedParameters)}
        raw = self.config.get('parameters', {})
        return UnifiedParameters(**{k: v for k, v in raw.items() if k in valid})

    def _create_preprocessing_parameters(self) -> PreprocessingParameters:
        """Create preprocessing parameters from config."""
        return self._unified_parameters().to_preprocessing_parameters()

    def _create_displacement_parameters(self) -> DisplacementParameters:
        """Create displacement parameters from config."""
        return self._unified_parameters().to_displacement_parameters()

    def _create_fttc_parameters(self) -> FTTCParameters:
        """Create FTTC parameters from config."""
        return self._unified_parameters().to_fttc_parameters()

    def _create_stress_parameters(self) -> StressParameters:
        """Create stress (BISM) parameters from config."""
        return self._unified_parameters().to_stress_parameters()

    def _log_stage_progress(self, array, frame, total, quantity, unit):
        """Print per-frame mean/max magnitude for a streamed stage array.

        One helper for all three measure stages (displacement / force / stress),
        which differ only in the printed quantity name and unit.
        """
        magnitude = np.sqrt(np.sum(array ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean {quantity}: {np.mean(magnitude):.2f} {unit}, "
              f"Max {quantity}: {np.max(magnitude):.2f} {unit}")

    def _initialize_folder(self, output_dir: Path) -> Path:
        """Set up the TFM_data/ output folder and logging.

        Derived output — the ``.ntfm`` artifact, ``figures/``, ``batch.log`` and
        the stage-resume cache — all live under ``output_dir``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / "batch.log"
        self._tee_logger = TeeLogger(log_file, self.config)
        sys.stdout = self._tee_logger

        return output_dir

    def _handle_visualization(self, tfm_folder: Path, viz_saver: BatchVisualizationSaver, step: str,
                              current_data: Optional[dict] = None) -> None:
        """
        Handle visualizations for each analysis step, loading data from files if needed.

        Parameters
        ----------
        tfm_folder : Path
            Path to the TFM data folder
        viz_saver : BatchVisualizationSaver
            Visualization saver instance
        step : str
            Current analysis step ('preprocessing', 'displacement', 'force', 'stress')
        current_data : Optional[dict]
            Data from the current analysis step, if available
        """
        # Map analysis steps to their visualization flags. Preprocessing has no
        # visualization (the bead overlay was removed).
        viz_map = {
            'displacement': 'displacement_map',
            'force': ['force_map', 'force_cell_overlay'],
            'stress': ['sigma_xx', 'sigma_yy', 'normal_stress']
        }

        viz_flags = viz_map.get(step, [])
        if isinstance(viz_flags, str):
            viz_flags = [viz_flags]

        # Check if any visualization is enabled for this step
        if not any(self.config['visualizations'].get(flag, False) for flag in viz_flags):
            return

        try:
            data = current_data
            if data is None:
                # Only preprocessing has an on-disk cache to fall back to (the
                # opt-in preprocessed .tif). Displacement/force/stress results are
                # not cached as .npy — if the stage produced nothing
                # this run there is nothing to visualize.
                if step == 'preprocessing':
                    try:
                        data = {
                            'beads': tifffile.imread(str(tfm_folder / "preprocessed_beads.tif")),
                            'reference': tifffile.imread(str(tfm_folder / "preprocessed_reference.tif"))
                        }
                    except Exception as e:
                        print(f"Could not load preprocessed files for visualization: {str(e)}")
                        return

            if data is None:
                print(f"No data available for {step} visualization")
                return

            # Generate visualizations based on enabled flags
            if step == 'displacement' and self.config['visualizations']['displacement_map']:
                print("Generating displacement map visualization...")
                viz_saver.save_displacement_visualization(data)

            elif step == 'force':
                if self.config['visualizations']['force_map']:
                    print("Generating force map visualization...")
                    viz_saver.save_force_visualization(data)

                if self.config['visualizations']['force_cell_overlay']:
                    print("Generating force-cell overlay visualization...")
                    try:
                        cell_images = tifffile.imread(str(tfm_folder / "preprocessed_cells.tif"))
                        viz_saver.save_force_cell_overlay(data, cell_images)
                    except Exception as e:
                        print(f"Could not generate force-cell overlay: {str(e)}")

            elif step == 'stress':
                if any(self.config['visualizations'][flag] for flag in ['sigma_xx', 'sigma_yy', 'normal_stress']):
                    print("Generating stress visualization...")
                    viz_saver.save_stress_visualization(
                        data,
                        plot_sigma_xx=self.config['visualizations']['sigma_xx'],
                        plot_sigma_yy=self.config['visualizations']['sigma_yy'],
                        plot_normal_stress=self.config['visualizations']['normal_stress']
                    )

        except Exception as e:
            print(f"Error generating {step} visualization: {str(e)}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BatchAnalysis':
        """Create BatchAnalysis instance from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._tee_logger:
            self._tee_logger.close()
            self._tee_logger = None


if __name__ == "__main__":
    analyzer = BatchAnalysis.from_yaml("config.yaml")
    analyzer.process_all_folders()
