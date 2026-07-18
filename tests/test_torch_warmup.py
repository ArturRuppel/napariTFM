"""Tests for the best-effort background torch warmup.

The warmup is purely an optimization: it must run at most once per process,
respect the NAPARITFM_NO_WARMUP opt-out, and swallow every failure so a
torch-free / CPU-only install just no-ops.
"""
import sys

import pytest

from napariTFM.backend import _torch_warmup


@pytest.fixture(autouse=True)
def _reset_warmed():
    """Reset the once-per-process guard around each test."""
    _torch_warmup._warmed = False
    yield
    _torch_warmup._warmed = False


def test_warm_runs_without_raising():
    """The real ops run cleanly on this machine's torch (CUDA or CPU)."""
    _torch_warmup._warm()  # runs synchronously; must not raise


def test_warm_is_torch_free_safe(monkeypatch):
    """A torch-free install must no-op, not raise."""
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", None)
    _torch_warmup._warm()  # swallows the ImportError, logs at debug


def test_warm_up_is_idempotent(monkeypatch):
    """Only the first call spawns a thread."""
    calls = []

    class _FakeThread:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("name"))

        def start(self):
            pass

    monkeypatch.delenv("NAPARITFM_NO_WARMUP", raising=False)
    monkeypatch.setattr(_torch_warmup.threading, "Thread", _FakeThread)

    _torch_warmup.warm_up_torch()
    _torch_warmup.warm_up_torch()

    assert len(calls) == 1


def test_warm_up_respects_opt_out(monkeypatch):
    """NAPARITFM_NO_WARMUP disables the warmup entirely."""
    calls = []
    monkeypatch.setattr(
        _torch_warmup.threading, "Thread",
        lambda *a, **k: calls.append(1) or pytest.fail("thread spawned despite opt-out"),
    )
    monkeypatch.setenv("NAPARITFM_NO_WARMUP", "1")

    _torch_warmup.warm_up_torch()

    assert calls == []
