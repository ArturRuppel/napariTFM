from qtpy.QtCore import Qt
from qtpy.QtWidgets import QStyle, QToolButton, QWidget


COMPACT_SPACING = 4
ICON_BUTTON_SIZE = 24

STAGE_ACCENTS = {
    "inputs": "#6c757d",
    "preprocessing": "#2f80ed",
    "displacement": "#9b5de5",
    "force": "#2a9d8f",
    "force_analysis": "#2a9d8f",
    "stress": "#e76f51",
    "stress_analysis": "#e76f51",
    "batch": "#f4a261",
    "batch_analysis": "#f4a261",
}

STATUS_COLORS = {
    "not_started": "#8c8c8c",
    "ready": "#2f80ed",
    "running": "#f4a261",
    "done": "#2a9d8f",
    "stale": "#e9c46a",
    "error": "#d62828",
}


def make_icon_button(
    owner: QWidget,
    action: str,
    object_name: str,
    tooltip: str,
    standard_icon: QStyle.StandardPixmap,
) -> QToolButton:
    button = QToolButton(owner)
    button.setAutoRaise(True)
    button.setFixedSize(ICON_BUTTON_SIZE, ICON_BUTTON_SIZE)
    button.setIcon(owner.style().standardIcon(standard_icon))
    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setToolButtonStyle(Qt.ToolButtonIconOnly)
    return button


def status_indicator_style(status: str) -> str:
    color = STATUS_COLORS.get(status, STATUS_COLORS["not_started"])
    return (
        "background-color: "
        f"{color};"
        " border: 1px solid rgba(0, 0, 0, 80);"
        " border-radius: 5px;"
        " min-width: 10px;"
        " max-width: 10px;"
        " min-height: 10px;"
        " max-height: 10px;"
    )
