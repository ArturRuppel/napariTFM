"""Shared pytest fixtures for the napariTFM test suite.

The GUI tests here build real ``QWidget``\\s against a single shared
``QApplication`` (via the per-module ``app`` fixture) and, for the most part,
never tear them down.  Live widgets therefore accumulate for the whole run.
Qt only *actually* frees a widget when the ``DeferredDelete`` event posted by
``deleteLater()`` is delivered, which normally happens when the event loop
unwinds -- but these tests never run a loop, so even the widgets that *do* call
``deleteLater()`` stay alive.  Once the cumulative set of live widgets crosses
an internal Qt threshold, teardown inside the Qt/napari path segfaults
non-deterministically (roughly 1 run in 5).

The autouse fixture below bounds the live set: after every test it closes any
top-level widgets, then explicitly flushes the ``DeferredDelete`` queue so the
deletions actually happen instead of piling up.
"""

import os

# Render Qt into an off-screen buffer for the whole suite: the few tests that
# must ``.show()`` a real widget (to exercise ``isVisible()``) otherwise throw
# actual windows onto the display, which flicker during a run and route teardown
# through the real compositor/GL path where the segfault above lives. We force
# it (a desktop/Wayland session exports QT_QPA_PLATFORM, so setdefault would be a
# no-op) unless the developer explicitly opts into visible windows -- run
# ``NAPARITFM_TEST_SHOW_WINDOWS=1 pytest`` to watch a GUI test on screen.
if not os.environ.get("NAPARITFM_TEST_SHOW_WINDOWS"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest


@pytest.fixture(autouse=True)
def _dispose_leaked_qt_widgets():
    """Close top-level widgets and drain Qt's deferred-delete queue post-test."""
    yield

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return

    from qtpy.QtCore import QEvent

    for widget in list(app.topLevelWidgets()):
        widget.close()
        widget.deleteLater()

    # Deliver close()/hide() side effects, then force the DeferredDelete events
    # (from deleteLater above and from any test that called it itself) to run
    # now rather than accumulating until the next time a loop happens to spin.
    app.processEvents()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
