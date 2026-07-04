"""A live :class:`PipelineSink` that streams a batch run into the napari viewer.

This is the viewer half of worklist §5: the *same* :class:`BatchAnalysis`
orchestrator drives both a headless run and the in-napari "Run selected". Headless
attaches no sink; the in-napari run attaches a :class:`ViewerSink`, which turns
the orchestrator's stage/frame notifications into the very same
``VisualizationManager`` streaming calls the interactive per-stage controllers
use — so a run walks the rail in the viewer, filling each stage's layers
frame by frame, exactly as if each stage button were pressed in turn.

No background thread is introduced: the batch runs synchronously on the GUI
thread (as it already did), and the sink pumps the event loop after each frame
so the viewer repaints live. That keeps the layer mutations on the GUI thread —
where napari requires them — without any cross-thread marshalling.
"""

import numpy as np

from napariTFM.backend.pipeline_sink import PipelineSink


class ViewerSink(PipelineSink):
    """Translate orchestrator hooks into ``VisualizationManager`` streaming.

    Parameters
    ----------
    data_manager, visualization_manager
        The same managers the interactive widgets share, so a run-all leaves the
        viewer in the same state a manual stage run would.
    pump
        Optional zero-arg callable invoked after each streamed frame to repaint
        the viewer (typically ``QApplication.processEvents``). ``None`` disables
        pumping — handy in tests.
    on_experiment
        Optional one-arg callable ``callback(path)`` invoked when the run enters
        a new experiment folder (worklist §3), so the shell can make the
        experiments-list selection follow the position being processed. ``None``
        disables tracking.
    on_stage_progress
        Optional callable ``callback(stage, status, fraction)`` invoked as each
        stage starts (``status="running"``, ``fraction=0.0``), streams a frame
        (``status="running"``, ``fraction`` growing 0..1), and finishes
        (``status="done"``, ``fraction=None``) — item #10's progressive
        per-stage loading bar. ``None`` disables this (e.g. headless runs).
    """

    # Per-stage active-layer sets (worklist §4). While a stage streams, the sink
    # takes over layer visibility and shows only its own layers, so a run-all
    # never blends the stage in flight with the previous stage's overlay.
    # Preprocessing is two-phase: beads + reference stream first, then cells.
    _STAGE_LAYERS = {
        'displacement': ['Displacement Magnitude', 'Displacement Vectors'],
        'force': ['Force Magnitude', 'Force Vectors'],
        'stress': ['Normal Stress XX', 'Normal Stress YY', 'Average Normal Stress'],
    }
    _PREPROC_BEADS_REF = ['Preprocessed Beads', 'Preprocessed Reference']
    _PREPROC_CELLS = ['Preprocessed Cells']

    def __init__(
        self, data_manager, visualization_manager, pump=None, on_experiment=None,
        on_stage_progress=None,
    ):
        self.data_manager = data_manager
        self.vis = visualization_manager
        self._pump = pump
        self._on_experiment = on_experiment
        self._on_stage_progress = on_stage_progress
        # Set once the preprocessing stage has flipped to its cell-only phase, so
        # the flip fires on the first cell frame and never repeats mid-stage.
        self._preproc_cells_isolated = False
        # Pre-run visibility snapshot, held between begin_run/end_run; ``None``
        # outside a run so end_run is a safe no-op (and idempotent).
        self._restore_visibility = None
        # Frame count of the stage currently streaming, for fraction = (i+1)/n.
        self._stage_num_frames = 0

    # --- run boundary (§4) ------------------------------------------------

    def begin_run(self):
        """Snapshot layer visibility before the run takes the viewer over.

        The shell calls this once before the orchestrator starts so the
        per-stage isolation below is reversible: ``end_run`` puts every
        pre-existing layer back the way the user had it.
        """
        self._restore_visibility = self.vis.capture_layer_visibility()

    def end_run(self):
        """Hand visibility back to the user (restore the ``begin_run`` snapshot).

        A no-op if no snapshot is held (no ``begin_run``, or already restored),
        so the shell can call it unconditionally in its ``finally``.
        """
        if self._restore_visibility is None:
            return
        self.vis.restore_layer_visibility(self._restore_visibility)
        self._restore_visibility = None

    # --- lifecycle hooks --------------------------------------------------

    def experiment_started(self, path):
        # Drive the experiments-list selection to the streaming position (§3).
        # The viewer itself already follows because we stream its frames below;
        # this only moves the list's row highlight to match.
        if self._on_experiment is not None:
            self._on_experiment(path)
        self._repaint()

    def stage_started(self, stage, num_frames, info=None):
        info = info or {}
        self._stage_num_frames = num_frames
        if self._on_stage_progress is not None:
            self._on_stage_progress(stage, 'running', 0.0)
        if stage == 'preprocessing':
            self._begin_preprocessing(info)
            # Start in the beads+reference phase; the first cell frame flips it.
            self._preproc_cells_isolated = False
            # Preprocessing streams three channels (beads, reference, cells) each
            # with their own 0-based per-channel frame_index, so frame_index can't
            # drive a monotonic bar. Count emitted frames instead, against the
            # announced total work (beads + reference + cells).
            self._preproc_frames_seen = 0
            self.vis.isolate_layers(self._PREPROC_BEADS_REF)
        elif stage in ('displacement', 'force'):
            self.vis.begin_vector_field_stream(stage, num_frames, {
                'v_max': info['v_max'],
                'vector_stride': info['vector_stride'],
                'arrow_scale': info['arrow_scale'],
                'downscale_factor': info['downscale_factor'],
            })
            self.vis.isolate_layers(self._STAGE_LAYERS[stage])
        elif stage == 'stress':
            self.vis.begin_stress_stream(
                num_frames=num_frames,
                max_stress=info['max_stress'],
                downscale_factor=info['downscale_factor'],
            )
            self.vis.isolate_layers(self._STAGE_LAYERS['stress'])
        self._repaint()

    def stage_frame(self, stage, frame_index, frame):
        if stage == 'preprocessing':
            # ``frame`` is a {channel: image} mapping (one key per yield). The
            # first cell frame flips isolation from beads+reference to cells-only
            # (worklist §4) — beads and reference stream before any cell frame.
            for channel, image in frame.items():
                self.vis.stream_preprocessing_frame(channel, frame_index, image)
                if channel == 'cells' and not self._preproc_cells_isolated:
                    self._preproc_cells_isolated = True
                    self.vis.isolate_layers(self._PREPROC_CELLS)
        elif stage in ('displacement', 'force'):
            self.vis.stream_vector_field_frame(stage, frame_index, frame)
        elif stage == 'stress':
            self.vis.stream_stress_frame(frame_index, frame)
        if self._on_stage_progress is not None:
            if stage == 'preprocessing':
                # Monotonic across the three channels (see stage_started).
                self._preproc_frames_seen += 1
                fraction = min(1.0, self._preproc_frames_seen / max(self._stage_num_frames, 1))
            else:
                # In-order stages: frame_index is the authoritative position.
                fraction = (frame_index + 1) / max(self._stage_num_frames, 1)
            self._on_stage_progress(stage, 'running', fraction)
        self._repaint()

    def stage_finished(self, stage, result):
        if self._on_stage_progress is not None:
            self._on_stage_progress(stage, 'done', None)
        # Store the full result so interactive frame-scrubbing and any downstream
        # stage see it — mirrors what each per-stage controller does on
        # completion. Preprocessing needs nothing here: its stacks were allocated
        # in the data manager up front and filled in place as frames streamed.
        if result is None:
            return
        if stage == 'displacement':
            self.data_manager.set_displacement_results(result, dirty=True)
        elif stage == 'force':
            self.data_manager.set_force_results(result, dirty=True)
        elif stage == 'stress':
            self.data_manager.set_stress_results(result, dirty=True)

    # --- helpers ----------------------------------------------------------

    def _begin_preprocessing(self, info):
        """Pre-allocate the Preprocessed* stacks, then bind their layers.

        Mirrors ``PreprocessingController._begin_stream``: the stacks are
        registered with the data manager as zeroed float32 arrays (generated,
        dirty) so the layers are backed by the very arrays the run fills in
        place; ``begin_preprocessing_stream`` then creates/reuses the layers.
        """
        beads_shape = info.get('beads_shape')
        reference_shape = info.get('reference_shape')
        cells_shape = info.get('cells_shape')

        if beads_shape is not None:
            self.data_manager.set_preprocessed_bead_stack(
                np.zeros(beads_shape, dtype=np.float32), dirty=True
            )
        if reference_shape is not None:
            self.data_manager.set_preprocessed_reference(
                np.zeros(reference_shape, dtype=np.float32), dirty=True
            )
        if cells_shape is not None:
            self.data_manager.set_preprocessed_cell_stack(
                np.zeros(cells_shape, dtype=np.float32), dirty=True
            )

        self.vis.begin_preprocessing_stream()

    def _repaint(self):
        if self._pump is not None:
            self._pump()
