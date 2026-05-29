import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QLabel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _Source(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.enabled = {"run": False, "preview": False}
        self.ran = 0
        self.previewed = 0

    def states(self):
        return dict(self.enabled)

    def run(self):
        self.ran += 1

    def preview(self):
        self.previewed += 1


def test_header_buttons_disabled_until_states_allow(app):
    from napariTFM.widgets._stage_section import StageSection

    src = _Source()
    sec = StageSection(
        "Stage",
        QLabel("body"),
        actions={"run": src.run, "preview": src.preview},
        action_states=src.states,
        action_states_changed=src.changed,
    )
    assert sec.run_cancel_btn.isEnabled() is False
    assert sec.preview_button.isEnabled() is False

    src.enabled = {"run": True, "preview": True}
    src.changed.emit()
    assert sec.run_cancel_btn.isEnabled() is True
    assert sec.preview_button.isEnabled() is True


def test_header_run_and_preview_invoke_handlers(app):
    from napariTFM.widgets._stage_section import StageSection

    src = _Source()
    src.enabled = {"run": True, "preview": True}
    sec = StageSection(
        "Stage",
        QLabel("body"),
        actions={"run": src.run, "preview": src.preview},
        action_states=src.states,
        action_states_changed=src.changed,
    )
    src.changed.emit()
    sec.run_cancel_btn.click()
    sec.preview_button.click()
    assert src.ran == 1
    assert src.previewed == 1


def test_no_action_state_sync_class():
    import napariTFM.widgets._stage_section as mod

    assert not hasattr(mod, "_ActionStateSync")
