import pytest
from qtpy.QtWidgets import QApplication, QWidget

from napariTFM.widgets._stage_section import StageSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_non_optional_stage_has_no_enable_toggle(app):
    section = StageSection("Displacement", QWidget())
    assert section.enable_btn is None
    assert section.is_enabled is True


def test_optional_stage_exposes_checked_enable_toggle(app):
    section = StageSection("Stress", QWidget(), optional=True)
    assert section.enable_btn is not None
    assert section.enable_btn.isCheckable()
    assert section.enable_btn.isChecked() is True
    assert section.enable_btn.objectName() == "stage_stress_enable_button"
    assert section.is_enabled is True


def test_disabling_optional_stage_marks_spine_off_and_blocks_actions(app):
    section = StageSection(
        "Stress",
        QWidget(),
        optional=True,
        action_states=lambda: {"run": True, "preview": True},
        status="ready",
    )
    section.set_enabled(False)

    assert section.is_enabled is False
    assert section.spine._status == "off"
    assert section.run_cancel_btn.isEnabled() is False
    assert section.preview_button.isEnabled() is False
    # The toggle itself stays live so the stage can be switched back on.
    assert section.enable_btn.isEnabled() is True
    assert section.enable_btn.isChecked() is False


def test_re_enabling_restores_the_underlying_status(app):
    section = StageSection(
        "Stress",
        QWidget(),
        optional=True,
        action_states=lambda: {"run": True},
        status="ready",
    )
    section.set_enabled(False)
    assert section.spine._status == "off"

    section.set_enabled(True)
    assert section.is_enabled is True
    assert section.spine._status == "ready"
    assert section.run_cancel_btn.isEnabled() is True


def test_status_set_while_off_is_remembered_not_shown(app):
    section = StageSection("Stress", QWidget(), optional=True, status="ready")
    section.set_enabled(False)

    section.set_status("done")
    assert section.spine._status == "off"  # still reads as off
    assert section.status == "done"        # but the real status is tracked

    section.set_enabled(True)
    assert section.spine._status == "done"


def test_enabled_changed_signal_fires(app):
    section = StageSection("Stress", QWidget(), optional=True)
    seen = []
    section.enabled_changed.connect(seen.append)

    section.enable_btn.click()
    assert seen == [False]
    section.enable_btn.click()
    assert seen == [False, True]
