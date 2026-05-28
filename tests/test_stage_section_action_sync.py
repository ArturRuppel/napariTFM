import pytest
from qtpy.QtWidgets import QApplication, QPushButton

from napariTFM.widgets._stage_section import _ActionStateSync


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_sync_is_parented_to_target_not_proxy(app):
    target = QPushButton()
    proxy = QPushButton()
    sync = _ActionStateSync(target, proxy)
    # Parented to the object it filters, so it dies with target (never dangles).
    assert sync.parent() is target


def test_sync_mirrors_enabled_state(app):
    target = QPushButton()
    proxy = QPushButton()
    target.setEnabled(False)
    sync = _ActionStateSync(target, proxy)
    assert proxy.isEnabled() is False
    target.setEnabled(True)
    app.processEvents()
    assert proxy.isEnabled() is True


def test_sync_survives_proxy_destroyed_before_target(app):
    target = QPushButton()
    proxy = QPushButton()
    sync = _ActionStateSync(target, proxy)

    # Destroy the proxy's C++ object while target is still alive.
    proxy.deleteLater()
    app.processEvents()

    # Toggling target dispatches EnabledChange to the filter; must NOT crash.
    target.setEnabled(False)
    app.processEvents()
    target.setEnabled(True)
    app.processEvents()
    # Reaching here without aborting is the assertion.
    assert True
