import pytest
from qtpy.QtWidgets import QApplication, QWidget

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

    assert not hasattr(section, "save_button")

    # No QToolButton in the header is labeled or named like a save button
    header_buttons = section.findChildren(QToolButton)
    save_buttons = [b for b in header_buttons if "save" in b.objectName().lower()]
    assert save_buttons == []


def test_section_no_longer_exposes_deprecated_header_aliases(app):
    section = StageSection("Preprocessing", QWidget())

    for name in ["config_button", "run_button", "cancel_button"]:
        assert not hasattr(section, name)


def test_run_cancel_btn_tooltip_swaps_on_status_running(app):
    section = StageSection("Preprocessing", QWidget(), status="ready")
    assert "Run" in section.run_cancel_btn.toolTip()

    section.set_status("running")
    assert "Cancel" in section.run_cancel_btn.toolTip()

    section.set_status("done")
    assert "Run" in section.run_cancel_btn.toolTip()


def test_run_cancel_btn_invokes_run_handler_when_not_running(app):
    clicks = {"run": 0, "cancel": 0}

    section = StageSection(
        "Preprocessing",
        QWidget(),
        actions={
            "run": lambda: clicks.__setitem__("run", clicks["run"] + 1),
            "cancel": lambda: clicks.__setitem__("cancel", clicks["cancel"] + 1),
        },
        action_states=lambda: {"run": True},
        status="ready",
    )

    section.run_cancel_btn.click()
    assert clicks == {"run": 1, "cancel": 0}

    section.set_status("running")
    section.run_cancel_btn.click()
    assert clicks == {"run": 1, "cancel": 1}


def test_extra_action_adds_named_glyph_button(app):
    section = StageSection(
        "Force Analysis",
        QWidget(),
        extra_actions=[
            {"key": "gcv", "tooltip": "Auto-select regularization", "icon": "gcv"}
        ],
    )

    button = section.extra_buttons["gcv"]
    assert button.objectName() == "stage_force_analysis_gcv_button"
    assert button.text() == ""
    assert not button.icon().isNull()
    assert button.toolTip() == "Auto-select regularization"


def test_extra_action_button_invokes_handler(app):
    clicks = {"gcv": 0}
    section = StageSection(
        "Force Analysis",
        QWidget(),
        extra_actions=[
            {
                "key": "gcv",
                "tooltip": "Auto-select regularization",
                "icon": "gcv",
                "handler": lambda: clicks.__setitem__("gcv", clicks["gcv"] + 1),
            }
        ],
        action_states=lambda: {"gcv": True},
    )

    section.extra_buttons["gcv"].click()
    assert clicks["gcv"] == 1


def test_extra_action_enablement_follows_action_states(app):
    enabled = {"gcv": False}
    section = StageSection(
        "Force Analysis",
        QWidget(),
        extra_actions=[{"key": "gcv", "tooltip": "GCV", "icon": "gcv"}],
        action_states=lambda: {"gcv": enabled["gcv"]},
    )

    assert not section.extra_buttons["gcv"].isEnabled()

    enabled["gcv"] = True
    section._refresh_action_states()
    assert section.extra_buttons["gcv"].isEnabled()


def test_extra_action_button_retints_on_accent_change(app):
    section = StageSection(
        "Force Analysis",
        QWidget(),
        extra_actions=[{"key": "gcv", "tooltip": "GCV", "icon": "gcv"}],
    )
    before = section.extra_buttons["gcv"].icon().cacheKey()

    section.set_accent("#ff00ff")
    assert section.extra_buttons["gcv"].icon().cacheKey() != before


def test_action_buttons_use_vector_icons_not_text(app):
    section = StageSection("Preprocessing", QWidget())

    for button in (section.run_cancel_btn, section.preview_button, section.params_btn):
        assert button.text() == ""
        assert not button.icon().isNull()


def test_run_cancel_icon_swaps_on_running(app):
    section = StageSection("Preprocessing", QWidget(), status="ready")
    run_icon = section.run_cancel_btn.icon().cacheKey()

    section.set_status("running")
    cancel_icon = section.run_cancel_btn.icon().cacheKey()
    assert cancel_icon != run_icon

    section.set_status("done")
    assert section.run_cancel_btn.icon().cacheKey() != cancel_icon


def test_no_status_indicator_dot(app):
    section = StageSection("Preprocessing", QWidget())
    assert not hasattr(section, "status_indicator")


def test_status_is_readable_via_property(app):
    section = StageSection("Preprocessing", QWidget(), status="ready")
    assert section.status == "ready"
    section.set_status("done")
    assert section.status == "done"


def test_status_panel_is_embedded_always_visible(app):
    panel = QWidget()
    section = StageSection("Preprocessing", QWidget(), status_panel=panel)
    section.show()
    app.processEvents()
    # The file-status row is a direct child of the section, shown without a toggle.
    assert panel.parent() is not None
    assert panel.isVisibleTo(section) is True


def test_no_files_toggle_button(app):
    section = StageSection("Preprocessing", QWidget(), status_panel=QWidget())
    # The 🔍 inspector toggle is retired; the status row is always visible.
    assert not hasattr(section, "files_btn")


def test_section_exposes_viz_btn_checkable_and_visible_by_default(app):
    section = StageSection("Preprocessing", QWidget())

    assert section.viz_btn.objectName() == "stage_preprocessing_viz_button"
    assert section.viz_btn.isCheckable()
    assert section.viz_btn.isChecked() is True


def test_viz_btn_toggle_emits_visualization_toggled_with_state(app):
    section = StageSection("Preprocessing", QWidget())
    emitted = []
    section.visualization_toggled.connect(emitted.append)

    section.viz_btn.setChecked(False)
    section.viz_btn.setChecked(True)

    assert emitted == [False, True]


def test_viz_btn_uses_vector_icon_not_text(app):
    section = StageSection("Preprocessing", QWidget())

    assert section.viz_btn.text() == ""
    assert not section.viz_btn.icon().isNull()


def test_viz_btn_tooltip_reflects_visibility_state(app):
    section = StageSection("Preprocessing", QWidget())
    assert "Hide" in section.viz_btn.toolTip()

    section.viz_btn.setChecked(False)
    assert "Show" in section.viz_btn.toolTip()
