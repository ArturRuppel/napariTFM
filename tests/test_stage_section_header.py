import pytest
from qtpy.QtWidgets import QApplication, QPushButton, QWidget

from napariTFM.widgets._stage_section import StageSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_section_exposes_params_btn_with_stable_name(app):
    section = StageSection("Preprocessing", QWidget())

    assert section.params_btn.objectName() == "stage_preprocessing_params_button"
    assert section.params_btn.isCheckable()


def test_section_exposes_run_cancel_btn_with_stable_name(app):
    section = StageSection("Preprocessing", QWidget())

    assert section.run_cancel_btn.objectName() == "stage_preprocessing_run_cancel_button"


def test_section_no_longer_exposes_save_button(app):
    from qtpy.QtWidgets import QToolButton

    section = StageSection("Preprocessing", QWidget())

    # save_button attribute is None per the migration plan
    assert section.save_button is None

    # No QToolButton in the header is labeled or named like a save button
    header_buttons = section.findChildren(QToolButton)
    save_buttons = [b for b in header_buttons if "save" in b.objectName().lower()]
    assert save_buttons == []


def test_config_button_is_alias_of_params_btn(app):
    section = StageSection("Preprocessing", QWidget())

    assert section.config_button is section.params_btn


def test_run_cancel_btn_tooltip_swaps_on_status_running(app):
    section = StageSection("Preprocessing", QWidget(), status="ready")
    assert "Run" in section.run_cancel_btn.toolTip()

    section.set_status("running")
    assert "Cancel" in section.run_cancel_btn.toolTip()

    section.set_status("done")
    assert "Run" in section.run_cancel_btn.toolTip()


def test_run_cancel_btn_clicks_run_target_when_not_running(app):
    run_target = QPushButton()
    cancel_target = QPushButton()
    clicks = {"run": 0, "cancel": 0}
    run_target.clicked.connect(lambda: clicks.__setitem__("run", clicks["run"] + 1))
    cancel_target.clicked.connect(lambda: clicks.__setitem__("cancel", clicks["cancel"] + 1))

    section = StageSection(
        "Preprocessing",
        QWidget(),
        action_targets={"run": run_target, "cancel": cancel_target},
        status="ready",
    )

    section.run_cancel_btn.click()
    assert clicks == {"run": 1, "cancel": 0}

    section.set_status("running")
    section.run_cancel_btn.click()
    assert clicks == {"run": 1, "cancel": 1}
