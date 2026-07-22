from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from napariTFM.widgets._stage_dependencies import (
    InteractiveStageCoordinator,
    StaleChoice,
    computational_parameters,
    parameters_match,
)


class _Method(Enum):
    DIRECT = "direct"


@dataclass
class _Displacement:
    window: object
    d_max: float
    disp_vector_stride: int
    disp_arrow_scale: float


@dataclass
class _Force:
    young_modulus: object
    f_max: float
    force_vector_stride: int = 4
    force_arrow_scale: float = 1.0


@dataclass
class _Stress:
    method: object
    max_stress: float


def test_identical_computational_parameters_match():
    params = _Force(young_modulus=10.0, f_max=100.0)

    assert parameters_match("force", params, params)


def test_visualization_only_changes_do_not_make_results_stale():
    displacement = _Displacement(32, 5.0, 4, 1.0)
    force = _Force(10.0, 100.0)
    stress = _Stress(_Method.DIRECT, 50.0)

    assert parameters_match(
        "displacement",
        displacement,
        replace(
            displacement,
            d_max=10.0,
            disp_vector_stride=8,
            disp_arrow_scale=2.0,
        ),
    )
    assert parameters_match(
        "force",
        force,
        replace(
            force,
            f_max=200.0,
            force_vector_stride=8,
            force_arrow_scale=2.0,
        ),
    )
    assert parameters_match("stress", stress, replace(stress, max_stress=100.0))


def test_solver_parameter_change_makes_result_stale():
    stored = _Force(young_modulus=10.0, f_max=100.0)

    assert not parameters_match("force", stored, replace(stored, young_modulus=20.0))


def test_absent_parameter_metadata_is_stale():
    params = _Force(young_modulus=10.0, f_max=100.0)

    assert not parameters_match("force", None, params)
    assert not parameters_match("force", params, None)


def test_numpy_scalars_normalize_like_python_scalars_recursively():
    stored = {
        "solver": _Stress(_Method.DIRECT, max_stress=50.0),
        "values": [np.int64(4), (np.float32(1.5),)],
    }
    current = {
        "solver": _Stress("direct", max_stress=75.0),
        "values": [4, (1.5,)],
    }

    assert computational_parameters("stress", stored) == {
        "solver": {"method": "direct"},
        "values": [4, [1.5]],
    }
    assert parameters_match("stress", stored, current)


class _Signal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def disconnect(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, *args):
        for callback in list(self._callbacks):
            callback(*args)


@dataclass
class _Artifact:
    name: str
    parameters: object


class _Stage:
    def __init__(self, name, calls, preview_result=None):
        self.name = name
        self.calls = calls
        self.preview_result = preview_result
        self.completed = _Signal()
        self.failed = _Signal()
        self.raise_preview = None
        self.raise_run = None
        self._next_run = 0
        self._cancelled_runs = set()

    def preview(self, *, completion, **inputs):
        self.calls.append((self.name, "preview", inputs))
        if self.raise_preview is not None:
            raise self.raise_preview
        if self.preview_result is not None:
            completion(self.preview_result)

    def run(self):
        self.calls.append((self.name, "run", {}))
        if self.raise_run is not None:
            raise self.raise_run
        self._next_run += 1
        return self._next_run

    def cancel(self):
        self.calls.append((self.name, "cancel", {}))
        self._cancelled_runs.update(range(1, self._next_run + 1))

    def complete_run(self, generation):
        if generation not in self._cancelled_runs:
            self.completed.emit()


def _coordinator(calls, artifacts=None, choices=None, valid=True, preview_results=None):
    artifacts = artifacts or {}
    choices = choices or {}
    preview_results = preview_results or {}
    stages = {
        name: _Stage(name, calls, preview_results.get(name))
        for name in ("displacement", "force", "stress")
    }
    prompts = []
    progress = []
    coordinator = InteractiveStageCoordinator(
        stages=stages,
        artifact_getters={
            name: lambda name=name: artifacts.get(name) for name in stages
        },
        parameter_getters={name: lambda name=name: name + "-params" for name in stages},
        prompt=lambda stage: prompts.append(stage) or choices[stage],
        source_validator=lambda stage: valid,
        progress=progress.append,
    )
    return coordinator, stages, artifacts, prompts, progress


def test_force_preview_computes_missing_displacement_and_passes_it_transiently():
    calls = []
    displacement = _Artifact("frame-displacement", "displacement-params")
    coordinator, _, _, prompts, _ = _coordinator(
        calls, preview_results={"displacement": displacement}
    )
    coordinator.request("force", "preview")
    assert calls == [
        ("displacement", "preview", {}),
        ("force", "preview", {"displacement_result": displacement}),
    ]
    assert prompts == []


def test_stress_run_computes_both_missing_prerequisites_in_order():
    calls = []
    coordinator, stages, artifacts, _, _ = _coordinator(calls)
    coordinator.request("stress", "run")
    artifacts["displacement"] = _Artifact("full-displacement", "displacement-params")
    stages["displacement"].completed.emit()
    artifacts["force"] = _Artifact("full-force", "force-params")
    stages["force"].completed.emit()
    assert calls == [
        ("displacement", "run", {}),
        ("force", "run", {}),
        ("stress", "run", {}),
    ]


def test_matching_artifact_is_reused_without_prompt():
    calls = []
    artifact = _Artifact("full-displacement", "displacement-params")
    coordinator, _, _, prompts, _ = _coordinator(
        calls, artifacts={"displacement": artifact}
    )
    coordinator.request("force", "preview")
    assert calls == [("force", "preview", {"displacement_result": artifact})]
    assert prompts == []


def test_stale_artifact_can_be_recalculated():
    calls = []
    old = _Artifact("old", "old-params")
    new = _Artifact("new", "displacement-params")
    coordinator, _, _, prompts, _ = _coordinator(
        calls,
        artifacts={"displacement": old},
        choices={"displacement": StaleChoice.RECALCULATE},
        preview_results={"displacement": new},
    )
    coordinator.request("force", "preview")
    assert prompts == ["displacement"]
    assert calls[-1] == ("force", "preview", {"displacement_result": new})


def test_stale_artifact_can_be_reused_for_current_chain():
    calls = []
    old = _Artifact("old", "old-params")
    coordinator, _, _, prompts, _ = _coordinator(
        calls,
        artifacts={"displacement": old},
        choices={"displacement": StaleChoice.REUSE},
    )
    coordinator.request("force", "preview")
    assert prompts == ["displacement"]
    assert calls == [("force", "preview", {"displacement_result": old})]


def test_stale_artifact_cancel_starts_nothing():
    calls = []
    coordinator, _, _, _, _ = _coordinator(
        calls,
        artifacts={"displacement": _Artifact("old", "old-params")},
        choices={"displacement": StaleChoice.CANCEL},
    )
    coordinator.request("force", "preview")
    assert calls == []


def test_cancel_forwards_to_active_stage_and_late_completion_does_not_continue():
    calls = []
    coordinator, stages, artifacts, _, _ = _coordinator(calls)
    coordinator.request("stress", "run")
    coordinator.cancel("stress")
    artifacts["displacement"] = _Artifact("late", "displacement-params")
    stages["displacement"].completed.emit()
    assert calls == [
        ("displacement", "run", {}),
        ("displacement", "cancel", {}),
    ]


def test_failure_terminates_chain():
    calls = []
    coordinator, stages, _, _, progress = _coordinator(calls)
    coordinator.request("stress", "run")
    stages["displacement"].failed.emit(RuntimeError("boom"))
    assert calls == [("displacement", "run", {})]
    assert "failed" in progress[-1].lower()


def test_source_validation_failure_starts_nothing():
    calls = []
    coordinator, _, _, _, _ = _coordinator(calls, valid=False)
    coordinator.request("stress", "run")
    assert calls == []


def test_direct_target_action_needs_no_artifact():
    calls = []
    coordinator, _, _, _, _ = _coordinator(calls)
    coordinator.request("displacement", "preview")
    assert calls == [("displacement", "preview", {})]


def test_superseded_run_completion_cannot_disconnect_new_chain():
    calls = []
    coordinator, stages, artifacts, _, _ = _coordinator(calls)
    coordinator.request("stress", "run")
    stale_completion = stages["displacement"].completed._callbacks[0]

    coordinator.request("stress", "run")
    stale_completion()
    artifacts["displacement"] = _Artifact("current", "displacement-params")
    stages["displacement"].completed.emit()

    assert calls == [
        ("displacement", "run", {}),
        ("displacement", "cancel", {}),
        ("displacement", "run", {}),
        ("force", "run", {}),
    ]


def test_superseding_active_run_cancels_old_operation_before_replacement():
    calls = []
    coordinator, stages, artifacts, _, _ = _coordinator(calls)
    coordinator.request("stress", "run")
    old_generation = stages["displacement"]._next_run

    coordinator.request("stress", "run")
    new_generation = stages["displacement"]._next_run
    artifacts["displacement"] = _Artifact("current", "displacement-params")
    stages["displacement"].complete_run(old_generation)
    assert calls[-1] == ("displacement", "run", {})

    stages["displacement"].complete_run(new_generation)
    assert calls[-1] == ("force", "run", {})
    assert calls[:3] == [
        ("displacement", "run", {}),
        ("displacement", "cancel", {}),
        ("displacement", "run", {}),
    ]


def test_preview_startup_exception_is_reported_and_coordinator_is_reusable():
    calls = []
    coordinator, stages, _, _, progress = _coordinator(calls)
    stages["displacement"].raise_preview = RuntimeError("preview startup")

    coordinator.request("displacement", "preview")

    assert "failed" in progress[-1].lower()
    stages["displacement"].raise_preview = None
    coordinator.request("displacement", "preview")
    assert calls == [
        ("displacement", "preview", {}),
        ("displacement", "preview", {}),
    ]


def test_run_startup_exception_is_reported_and_coordinator_is_reusable():
    calls = []
    coordinator, stages, _, _, progress = _coordinator(calls)
    stages["displacement"].raise_run = RuntimeError("run startup")

    coordinator.request("displacement", "run")

    assert "failed" in progress[-1].lower()
    stages["displacement"].raise_run = None
    coordinator.request("displacement", "run")
    assert calls == [
        ("displacement", "run", {}),
        ("displacement", "run", {}),
    ]


def test_new_request_does_not_cancel_normally_completed_operation():
    calls = []
    result = _Artifact("frame", "displacement-params")
    coordinator, _, _, _, _ = _coordinator(
        calls, preview_results={"displacement": result}
    )

    coordinator.request("displacement", "preview")
    coordinator.request("displacement", "preview")

    assert calls == [
        ("displacement", "preview", {}),
        ("displacement", "preview", {}),
    ]
