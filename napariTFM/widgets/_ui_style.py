import colorsys

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QStyle, QToolButton, QWidget


COMPACT_SPACING = 4
ICON_BUTTON_SIZE = 24

MUTED_TEXT_COLOR = "#999"

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

STATUS_GLYPHS = {
    "available": "✓",
    "missing_required": "✗",
    "missing_optional": "○",
    "running": "⟳",
    "stale": "⚠",
    "error": "⚠",
}

ACTION_GLYPHS = {
    "view": "👁",
    "save": "💾",
    "load": "↑",
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


def stage_accent(key: str) -> str:
    """Return the accent hex color for a stage key, falling back to inputs."""
    return STAGE_ACCENTS.get(key, STAGE_ACCENTS["inputs"])


def muted_stage_accent(key: str) -> str:
    """Return a muted (low-saturation, midtone-lightness) variant of a stage accent."""
    hex_value = stage_accent(key).lstrip("#")
    r = int(hex_value[0:2], 16) / 255.0
    g = int(hex_value[2:4], 16) / 255.0
    b = int(hex_value[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s_muted = s * 0.35
    l_muted = 0.5 + (l - 0.5) * 0.6
    r_out, g_out, b_out = colorsys.hls_to_rgb(h, l_muted, s_muted)
    return "#{:02x}{:02x}{:02x}".format(
        round(r_out * 255), round(g_out * 255), round(b_out * 255)
    )


def stage_header_style(accent: str) -> str:
    """Stylesheet for a stage section's accented header label."""
    return (
        f"font-weight: bold; color: {accent}; "
        f"border-left: 3px solid {accent}; padding-left: 6px;"
    )


def title_style() -> str:
    """Stylesheet for the top-level shell title label."""
    return "font-weight: bold; font-size: 14px;"


def section_label_style() -> str:
    """Stylesheet for a bold section label (e.g. a form group heading)."""
    return "font-weight: bold;"


def caption_style() -> str:
    """Stylesheet for a small, muted caption label."""
    return f"color: {MUTED_TEXT_COLOR}; font-size: 9pt;"


def danger_text_style() -> str:
    """Stylesheet for text on a destructive action control."""
    return "color: red;"


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
