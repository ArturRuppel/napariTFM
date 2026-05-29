import colorsys

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QLabel, QSizePolicy, QStyle, QToolButton, QVBoxLayout, QWidget


COMPACT_SPACING = 4
ICON_BUTTON_SIZE = 24
STAGE_ACTION_BUTTON_SIZE = 22
TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
DEFAULT_FIELD_SPACING = 8
DEFAULT_ROW_SPACING = 4

MUTED_TEXT_COLOR = "#999"

# ── Theme palettes ───────────────────────────────────────────────────────
# Each palette maps semantic color names to hex. Stage keys reference these
# names via STAGE_ACCENTS, so switching ACTIVE_PALETTE re-accents the whole UI.
CIVIDIS = {
    "rosewater": "#d6c35d", "pink": "#555c6d", "mauve": "#243c6e",
    "red": "#555c6d", "peach": "#a79d73", "yellow": "#d6c35d",
    "green": "#7d7c78", "teal": "#7d7c78", "sapphire": "#d6c35d",
    "blue": "#555c6d",
}
VIRIDIS = {
    "rosewater": "#9bd93c", "pink": "#31668e", "mauve": "#463480",
    "red": "#31668e", "peach": "#38b977", "yellow": "#9bd93c",
    "green": "#21918c", "teal": "#21918c", "sapphire": "#9bd93c",
    "blue": "#31668e",
}
NORD = {
    "rosewater": "#bf616a", "pink": "#b48ead", "mauve": "#b48ead",
    "red": "#bf616a", "peach": "#d08770", "yellow": "#ebcb8b",
    "green": "#a3be8c", "teal": "#8fbcbb", "sapphire": "#81a1c1",
    "blue": "#5e81ac",
}
DRACULA = {
    "rosewater": "#ffb86c", "pink": "#ff79c6", "mauve": "#bd93f9",
    "red": "#ff5555", "peach": "#ffb86c", "yellow": "#f1fa8c",
    "green": "#50fa7b", "teal": "#8be9fd", "sapphire": "#8be9fd",
    "blue": "#6272a4",
}

THEME_PALETTES = {
    "Cividis": CIVIDIS,
    "Viridis": VIRIDIS,
    "Nord": NORD,
    "Dracula": DRACULA,
}
ACTIVE_THEME_NAME = "Cividis"
ACTIVE_PALETTE = THEME_PALETTES[ACTIVE_THEME_NAME]

# Stage key -> semantic palette color name. Each visible stage gets a
# distinct accent so the workflow reads as ordered, themeable bands.
STAGE_ACCENTS = {
    "inputs": "sapphire",
    "project": "sapphire",
    "preprocessing": "blue",
    "displacement": "mauve",
    "force": "teal",
    "stress": "peach",
    "batch": "yellow",
}

STATUS_COLORS = {
    "not_started": "#8c8c8c",
    "ready": "#2f80ed",
    "running": "#f4a261",
    "done": "#2a9d8f",
    "error": "#d62828",
}

STATUS_GLYPHS = {
    "available": "✓",
    "missing_required": "✗",
    "missing_optional": "○",
    "running": "⟳",
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


def theme_names() -> tuple[str, ...]:
    return tuple(THEME_PALETTES)


def active_theme_name() -> str:
    return ACTIVE_THEME_NAME


def set_active_theme(name: str) -> None:
    global ACTIVE_PALETTE, ACTIVE_THEME_NAME
    ACTIVE_THEME_NAME = name
    ACTIVE_PALETTE = THEME_PALETTES[name]


def stage_accent(key: str) -> str:
    """Resolve a stage key to its accent hex via the active palette."""
    semantic = STAGE_ACCENTS.get(key, STAGE_ACCENTS["inputs"])
    return ACTIVE_PALETTE[semantic]


def muted_accent(hex_value: str) -> str:
    """Return a muted (low-saturation, midtone-lightness) variant of a hex color."""
    hex_value = hex_value.lstrip("#")
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


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.lstrip("#")
    return (
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
    )


def muted_stage_accent(key: str) -> str:
    """Return a muted variant of a stage accent."""
    return muted_accent(stage_accent(key))


def stage_header_style(accent: str) -> str:
    """Accent-tinted pill style for a stage section's header title."""
    return (
        "font-weight: bold; "
        "font-size: 9pt; "
        f"color: {muted_accent(accent)}; "
        f"background-color: {stage_header_pill_background(accent)}; "
        "border-radius: 4px; "
        "padding: 1px 6px;"
    )


def stage_header_pill_background(accent: str, alpha: int = 38) -> str:
    """Translucent rgba background for a stage header pill, from the muted accent."""
    r, g, b = _hex_to_rgb(muted_accent(accent))
    return f"rgba({r}, {g}, {b}, {alpha})"


def stage_header_disabled_action_color(accent: str) -> str:
    """Dimmed glyph color for a disabled stage action button."""
    r, g, b = _hex_to_rgb(muted_accent(accent))
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    s *= 0.55
    l = max(0.0, l * 0.62)
    r_out, g_out, b_out = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(
        round(r_out * 255), round(g_out * 255), round(b_out * 255)
    )


def stage_header_action_button_style(accent: str) -> str:
    """Pill stylesheet for a glyph QToolButton in a stage header."""
    color = muted_accent(accent)
    disabled = stage_header_disabled_action_color(accent)
    return (
        "QToolButton { "
        "font-weight: bold; font-size: 9pt; "
        f"color: {color}; "
        f"background-color: {stage_header_pill_background(accent)}; "
        "border: none; border-radius: 4px; padding: 0; margin: 0; "
        "text-align: center; } "
        "QToolButton:hover { "
        f"background-color: {stage_header_pill_background(accent, alpha=58)}; }} "
        "QToolButton:checked { "
        f"background-color: {stage_header_pill_background(accent, alpha=82)}; }} "
        "QToolButton:disabled { "
        f"color: {disabled}; "
        f"background-color: {stage_header_pill_background(accent, alpha=28)}; }} "
        "QToolButton:disabled:checked { "
        f"color: {disabled}; "
        f"background-color: {stage_header_pill_background(accent, alpha=44)}; }}"
    )


def make_stage_action_button(
    owner, object_name: str, tooltip: str, glyph: str, accent: str, checkable: bool = False
) -> QToolButton:
    """Build a glyph QToolButton styled as a CellFlow-style accent pill."""
    button = QToolButton(owner)
    button.setText(glyph)
    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setCheckable(checkable)
    button.setFixedSize(STAGE_ACTION_BUTTON_SIZE, STAGE_ACTION_BUTTON_SIZE)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    button.setStyleSheet(stage_header_action_button_style(accent))
    return button


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


def section_grid() -> QGridLayout:
    """A 4-column grid (label, field, label, field) where field columns
    stretch — so sliders, combos, and labels fill the available width and
    label columns stay aligned across all sections that share the grid."""
    layout = QGridLayout()
    layout.setHorizontalSpacing(DEFAULT_FIELD_SPACING)
    layout.setVerticalSpacing(DEFAULT_ROW_SPACING)
    layout.setColumnStretch(0, 0)
    layout.setColumnStretch(1, 1)
    layout.setColumnStretch(2, 0)
    layout.setColumnStretch(3, 1)
    return layout


def add_section_header(grid, row, widget):
    """Add a heading widget spanning all 4 columns of a section_grid."""
    grid.addWidget(widget, row, 0, 1, 4)
    return widget


def add_section_full_row(grid, row, widget):
    """Add a widget (separator, button row, …) spanning all 4 columns."""
    grid.addWidget(widget, row, 0, 1, 4)
    return widget


def add_section_pair_row(grid, row, left_label, left_widget, right_label=None, right_widget=None):
    """Add a row with up to two [label][widget] pairs. Widgets keep their
    natural size policy (no fixed-width wrap) so sliders/combos can stretch."""
    left_label_widget = _block_label(left_label)
    _add_section_pair_cell(grid, row, 0, left_label_widget, left_widget)

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        _add_section_pair_cell(grid, row, 2, right_label_widget, right_widget)
    return left_label_widget, left_widget, right_label_widget, right_widget


def _add_section_pair_cell(grid, row, column, label_widget, widget):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
    layout.addWidget(label_widget)
    layout.addWidget(widget)
    grid.addWidget(container, row, column, 1, 2)
    return container


def _block_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label
