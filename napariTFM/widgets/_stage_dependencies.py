"""Dependency and freshness helpers for interactive analysis stages."""

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Callable

from qtpy.QtCore import QObject

import numpy as np


DISPLAY_ONLY_FIELDS = {
    "displacement": {"d_max", "disp_vector_stride", "disp_arrow_scale"},
    "force": {"f_max", "force_vector_stride", "force_arrow_scale"},
    "stress": {"max_stress"},
}


def computational_parameters(stage: str, params: object) -> object:
    """Return a recursively normalized, display-independent parameter value."""
    display_fields = DISPLAY_ONLY_FIELDS.get(stage, set())

    def normalize(value: object) -> object:
        if isinstance(value, Enum):
            return normalize(value.value)
        if isinstance(value, np.generic):
            return normalize(value.item())
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: normalize(getattr(value, field.name))
                for field in fields(value)
                if field.name not in display_fields
            }
        if isinstance(value, Mapping):
            return {
                normalize(key): normalize(item)
                for key, item in value.items()
                if key not in display_fields
            }
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [normalize(item) for item in value]
        return value

    return normalize(params)


def parameters_match(stage: str, stored: object, current: object) -> bool:
    """Return whether stored and current computational parameters are equal."""
    if stored is None or current is None:
        return False
    return computational_parameters(stage, stored) == computational_parameters(
        stage, current
    )


class StaleChoice(Enum):
    """Decision returned by the stale-artifact prompt."""

    RECALCULATE = "recalculate"
    REUSE = "reuse"
    CANCEL = "cancel"


class InteractiveStageCoordinator(QObject):
    """Resolve and execute interactive pipeline stage dependencies.

    Parameters
    ----------
    stages:
        Mapping from ``displacement``, ``force``, and ``stress`` to adapters.
        Each adapter exposes ``preview(completion=..., **inputs)``, ``run()``,
        ``cancel()``, and ``completed``/``failed`` signals.
    artifact_getters:
        Per-stage zero-argument callables returning the stored artifact or
        ``None``. Artifacts expose their computation metadata as ``parameters``.
    parameter_getters:
        Per-stage zero-argument callables returning the parameters currently
        selected in the UI.
    prompt:
        Callable receiving a stale upstream stage name and returning a
        :class:`StaleChoice`.
    source_validator:
        Callable receiving the requested target stage and returning whether its
        non-derived source inputs are available. It is responsible for user
        feedback when validation fails.
    progress:
        Callable receiving human-readable progress and failure messages.
    """

    STAGES = ("displacement", "force", "stress")
    MODES = ("preview", "run")

    def __init__(
        self,
        *,
        stages: Mapping[str, object],
        artifact_getters: Mapping[str, Callable[[], object]],
        parameter_getters: Mapping[str, Callable[[], object]],
        prompt: Callable[[str], StaleChoice],
        source_validator: Callable[[str], bool],
        progress: Callable[[str], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._stages = stages
        self._artifact_getters = artifact_getters
        self._parameter_getters = parameter_getters
        self._prompt = prompt
        self._source_validator = source_validator
        self._progress = progress
        self._token = 0
        self._active_stage: str | None = None
        self._target_stage: str | None = None
        self._target_mode: str | None = None
        self._connections: list[tuple[object, Callable[..., None]]] = []

    def request(self, stage: str, mode: str) -> None:
        """Request a preview or full run of ``stage`` and its prerequisites."""
        if stage not in self.STAGES:
            raise ValueError(f"Unknown pipeline stage: {stage!r}")
        if mode not in self.MODES:
            raise ValueError(f"Unknown operation mode: {mode!r}")

        active = self._active_stage
        self._invalidate()
        if active is not None:
            self._stages[active].cancel()
        token = self._token
        self._target_stage = stage
        self._target_mode = mode

        target_index = self.STAGES.index(stage)
        self._resolve(token, stage, mode, 0, target_index, {})

    def cancel(self, stage: str) -> None:
        """Cancel the active operation belonging to the requested target."""
        if stage not in (self._target_stage, self._active_stage):
            return
        active = self._active_stage
        self._invalidate()
        if active is not None:
            self._stages[active].cancel()

    def _invalidate(self) -> None:
        self._token += 1
        self._disconnect_all()
        self._active_stage = None
        self._target_stage = None
        self._target_mode = None

    def _is_current(self, token: int) -> bool:
        return token == self._token

    def _resolve(
        self,
        token: int,
        target: str,
        mode: str,
        index: int,
        target_index: int,
        transient: dict[str, object],
    ) -> None:
        if not self._is_current(token):
            return
        if index == target_index:
            self._start_target(token, target, mode, transient)
            return

        stage = self.STAGES[index]
        artifact = self._artifact_getters[stage]()
        if artifact is not None:
            stored = getattr(artifact, "parameters", None)
            current = self._parameter_getters[stage]()
            if parameters_match(stage, stored, current):
                self._continue_with(
                    token, target, mode, index, target_index, transient, artifact
                )
                return
            choice = self._prompt(stage)
            if choice is StaleChoice.CANCEL:
                self._invalidate()
                return
            if choice is StaleChoice.REUSE:
                self._continue_with(
                    token, target, mode, index, target_index, transient, artifact
                )
                return

        self._progress(f"{target.title()} {mode}: calculating {stage}.")
        self._start_prerequisite(token, target, mode, index, target_index, transient)

    def _continue_with(
        self,
        token: int,
        target: str,
        mode: str,
        index: int,
        target_index: int,
        transient: dict[str, object],
        result: object,
    ) -> None:
        if mode == "preview":
            transient = {**transient, self.STAGES[index]: result}
        self._resolve(token, target, mode, index + 1, target_index, transient)

    def _start_prerequisite(
        self,
        token: int,
        target: str,
        mode: str,
        index: int,
        target_index: int,
        transient: dict[str, object],
    ) -> None:
        stage = self.STAGES[index]
        if not self._source_validator(stage):
            self._invalidate()
            return
        adapter = self._stages[stage]
        self._active_stage = stage
        self._connect_failure(adapter, token, target, mode, stage)

        if mode == "preview":
            inputs = self._preview_inputs(index, transient)

            def completed(result: object) -> None:
                if not self._is_current(token):
                    return
                self._disconnect_all()
                self._active_stage = None
                self._continue_with(
                    token, target, mode, index, target_index, transient, result
                )

            try:
                started = adapter.preview(completion=completed, **inputs)
                if started is False:
                    self._fail(token, target, mode, stage, "preview did not start")
            except Exception as error:
                self._fail(token, target, mode, stage, error)
            return

        def completed(*_args: object) -> None:
            if not self._is_current(token):
                return
            self._disconnect_all()
            self._active_stage = None
            artifact = self._artifact_getters[stage]()
            if artifact is None:
                self._fail(token, target, mode, stage, "no result was stored")
                return
            self._continue_with(
                token, target, mode, index, target_index, transient, artifact
            )

        self._connect(adapter.completed, completed)
        try:
            adapter.run()
        except Exception as error:
            self._fail(token, target, mode, stage, error)

    def _start_target(
        self,
        token: int,
        target: str,
        mode: str,
        transient: dict[str, object],
    ) -> None:
        if not self._source_validator(target):
            self._invalidate()
            return
        adapter = self._stages[target]
        self._active_stage = target
        self._connect_failure(adapter, token, target, mode, target)
        self._progress(f"{target.title()} {mode}: calculating {target}.")
        if mode == "preview":
            index = self.STAGES.index(target)

            def completed(_result: object) -> None:
                if self._is_current(token):
                    self._finish()

            try:
                started = adapter.preview(
                    completion=completed,
                    **self._preview_inputs(index, transient),
                )
                if started is False:
                    self._fail(token, target, mode, target, "preview did not start")
            except Exception as error:
                self._fail(token, target, mode, target, error)
            return

        def completed(*_args: object) -> None:
            if self._is_current(token):
                self._finish()

        self._connect(adapter.completed, completed)
        try:
            adapter.run()
        except Exception as error:
            self._fail(token, target, mode, target, error)

    def _preview_inputs(
        self, index: int, transient: Mapping[str, object]
    ) -> dict[str, object]:
        if index == 0:
            return {}
        upstream = self.STAGES[index - 1]
        if upstream not in transient:
            return {}
        return {f"{upstream}_result": transient[upstream]}

    def _connect_failure(
        self,
        adapter: object,
        token: int,
        target: str,
        mode: str,
        stage: str,
    ) -> None:
        def failed(error: object = None, *_args: object) -> None:
            self._fail(token, target, mode, stage, error)

        self._connect(adapter.failed, failed)

    def _fail(
        self,
        token: int,
        target: str,
        mode: str,
        stage: str,
        error: object,
    ) -> None:
        if not self._is_current(token):
            return
        detail = f": {error}" if error else ""
        self._progress(
            f"{target.title()} {mode} failed while calculating {stage}{detail}"
        )
        self._invalidate()

    def _finish(self) -> None:
        self._disconnect_all()
        self._active_stage = None
        self._target_stage = None
        self._target_mode = None

    def _connect(self, signal: object, callback: Callable[..., None]) -> None:
        signal.connect(callback)
        self._connections.append((signal, callback))

    def _disconnect_all(self) -> None:
        connections, self._connections = self._connections, []
        for signal, callback in connections:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass
