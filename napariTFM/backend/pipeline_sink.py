"""Optional observation surface for a batch / run-all pass.

The batch orchestrator (:class:`napariTFM.backend.batch_analysis.BatchAnalysis`)
runs the *same* compute whether or not a napari viewer is present — item §5 of
the worklist: "keep a headless mode, but the same code, only without the
viewer". A :class:`PipelineSink` is how the viewer plugs in: the orchestrator
notifies the sink at each stage and frame boundary, and a live ``ViewerSink``
turns those notifications into napari layer updates so a run streams into the
viewer exactly as if each stage button were pressed in turn.

A headless run attaches no sink (or :class:`NullSink`); every hook is then a
silent no-op and the run is byte-for-byte what it was before sinks existed. The
sink is therefore purely additive — it never alters what is computed or written
to disk, only what (optionally) gets shown while it runs.

This module is deliberately free of Qt and napari imports so the backend stays
headless-importable; the viewer-coupled implementation lives in
``napariTFM.utilities.viewer_sink``.

Stage keys are the four pipeline stages: ``"preprocessing"``, ``"displacement"``,
``"force"`` and ``"stress"``.
"""

from typing import Any, Optional


class PipelineSink:
    """Base sink: receives stage / frame lifecycle hooks from the orchestrator.

    Subclass and override the hooks of interest; the defaults do nothing, so a
    sink only pays for the events it cares about. Hooks must never raise — the
    orchestrator isolates them (a misbehaving sink can't break a run), but a
    sink that swallows its own errors keeps the run output clean.
    """

    def experiment_started(self, path: str) -> None:
        """A new experiment folder is about to stream (worklist §3).

        Fires once per folder, before its first ``stage_started``, with the
        folder path the orchestrator is entering. A live sink uses it to make
        the viewer + experiments-list selection *follow* the position being
        processed, so a batch/run-all walks the list as it walks the rail.
        """

    def stage_started(self, stage: str, num_frames: int, info: Optional[dict] = None) -> None:
        """A stage is about to stream ``num_frames`` frames.

        ``info`` carries the per-stage visualization context a live sink needs
        to allocate its layers up front (e.g. ``v_max`` / ``vector_stride`` /
        ``arrow_scale`` / ``downscale_factor`` for displacement & force,
        ``max_stress`` / ``downscale_factor`` for stress, frame shape and channel
        availability for preprocessing).
        """

    def stage_frame(self, stage: str, frame_index: int, frame: Any) -> None:
        """One freshly computed frame of ``stage`` (0-based ``frame_index``).

        ``frame`` is whatever that stage produces per frame: the field array for
        displacement / force, the stress-tensor frame for stress, and a
        ``{channel: image}`` mapping for preprocessing (channel in
        ``{"beads", "reference", "cells"}``).
        """

    def stage_finished(self, stage: str, result: Any) -> None:
        """A stage finished; ``result`` is its final result object (or ``None``)."""


class NullSink(PipelineSink):
    """A sink that does nothing — the explicit form of "no viewer attached".

    Passing ``NullSink()`` is equivalent to passing ``None``; both make every
    hook a no-op. It exists so call sites can hold a non-optional sink reference
    without sprinkling ``if sink is not None`` everywhere.
    """
