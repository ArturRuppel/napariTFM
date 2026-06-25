import colorsys

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QGridLayout, QLabel, QSizePolicy, QStyle, QToolButton, QVBoxLayout, QWidget


COMPACT_SPACING = 4
ICON_BUTTON_SIZE = 24
STAGE_ACTION_BUTTON_SIZE = 22
STAGE_ACTION_ICON_SIZE = 15
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

# ── Ordered perceptual ramps ─────────────────────────────────────────────
# Each theme is an ordered list of hex stops. Stages sample the ACTIVE ramp by
# their pipeline position, so the workflow reads as one colormap sweep instead
# of collapsing to a few muddy palette colors.
THEME_RAMPS = {
    "Viridis": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
    # CellFlow's five ordered cividis stops (inverted interior 15–85% sampling),
    # yellow → deep blue. Stage positions land each pipeline node exactly on a
    # stop: project=yellow (top), stress=deep blue (last).
    "Cividis": ["#d6c35d", "#a79d73", "#7d7c78", "#555c6d", "#243c6e"],
    "Nord":    ["#5e81ac", "#81a1c1", "#8fbcbb", "#a3be8c", "#ebcb8b", "#d08770"],
    "Dracula": ["#6272a4", "#bd93f9", "#8be9fd", "#50fa7b", "#f1fa8c", "#ffb86c"],
}
ACTIVE_RAMP = THEME_RAMPS[ACTIVE_THEME_NAME]

# Stage -> position along the ramp (0 = start, 1 = end). project/inputs anchor
# the start; batch anchors the end; the four pipeline stages spread between.
STAGE_RAMP_POSITION = {
    "inputs": 0.0, "project": 0.0,
    "preprocessing": 0.25, "displacement": 0.50,
    "force": 0.75, "stress": 1.0, "batch": 1.0,
}

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

# File-status dot colours: a stage's input/output artifacts read red (missing)
# → green (present, in cache or on disk), with a quiet grey for absent optionals.
FILE_STATUS_COLORS = {
    "present": "#5e9468",  # muted green — value is in cache or a file is on disk
    "missing": "#b05751",  # muted brick — a required artifact is absent
    "optional": "#5b626d",  # quiet grey — an optional artifact is absent (no alarm)
    "error": "#c2a04e",  # muted amber — the artifact failed to load
}


def file_status_color(state: str) -> str:
    """Colour for a file-status dot; unknown states fall back to the grey."""
    return FILE_STATUS_COLORS.get(state, FILE_STATUS_COLORS["optional"])


def file_status_state(available: bool, required: bool, error: bool) -> str:
    """Classify an artifact into a file-status state (present/missing/optional/error)."""
    if error:
        return "error"
    if available:
        return "present"
    return "missing" if required else "optional"


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
    global ACTIVE_PALETTE, ACTIVE_THEME_NAME, ACTIVE_RAMP
    ACTIVE_THEME_NAME = name
    ACTIVE_PALETTE = THEME_PALETTES[name]
    ACTIVE_RAMP = THEME_RAMPS[name]


def _sample_ramp(ramp: list[str], t: float) -> str:
    """Linear-interpolate a hex color at position t in [0, 1] along an ordered ramp."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    if len(ramp) == 1:
        return ramp[0]
    span = t * (len(ramp) - 1)
    i = int(span)
    if i >= len(ramp) - 1:
        return ramp[-1]
    frac = span - i
    a, b = _hex_to_rgb(ramp[i]), _hex_to_rgb(ramp[i + 1])
    rgb = tuple(round(a[c] + (b[c] - a[c]) * frac) for c in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def stage_accent(key: str) -> str:
    """Resolve a stage key to its accent by sampling the active colormap ramp."""
    position = STAGE_RAMP_POSITION.get(key, STAGE_RAMP_POSITION["inputs"])
    return _sample_ramp(ACTIVE_RAMP, position)


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


def stage_action_button_icon(name: str, accent: str):
    """A theme-tinted QIcon (with a dimmed disabled mode) for a header action."""
    from napariTFM.widgets._icons import stage_action_icon

    return stage_action_icon(
        name,
        muted_accent(accent),
        disabled_color=stage_header_disabled_action_color(accent),
        size=STAGE_ACTION_ICON_SIZE,
    )


def make_stage_action_button(
    owner,
    object_name: str,
    tooltip: str,
    glyph: str,
    accent: str,
    checkable: bool = False,
    icon_name: str | None = None,
) -> QToolButton:
    """Build a stage-header action button styled as a CellFlow-style accent pill.

    Pass ``icon_name`` for a crisp tinted vector icon; ``glyph`` is the legacy
    text fallback used when no icon is given.
    """
    button = QToolButton(owner)
    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setCheckable(checkable)
    button.setFixedSize(STAGE_ACTION_BUTTON_SIZE, STAGE_ACTION_BUTTON_SIZE)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    button.setStyleSheet(stage_header_action_button_style(accent))
    if icon_name is not None:
        button.setIcon(stage_action_button_icon(icon_name, accent))
        button.setIconSize(QSize(STAGE_ACTION_ICON_SIZE, STAGE_ACTION_ICON_SIZE))
    else:
        button.setText(glyph)
    return button


# ── Designed-surface tokens (mockup v2 aggregation layer) ─────────────────
# Theme-agnostic so the designed panels sit on any host background: surfaces
# are translucent white overlays (a "lift"), text uses the mockup's grey ramp.
TEXT_BRIGHT = "#e6edf3"
TEXT_MID = "#aeb6c0"
TEXT_DIM = "#6b7484"
ROW_LIFT_BG = "rgba(255, 255, 255, 13)"   # a selected/raised row surface
HAIRLINE = "rgba(255, 255, 255, 18)"

# Experiment-row status word -> color (amber running, green done, dim queued).
EXPERIMENT_STATUS_COLORS = {
    "run": "#e3b341",
    "done": "#3fb950",
    "queued": TEXT_DIM,
}


def experiment_status_color(label: str) -> str:
    """Color for an experiment row's overall-status word (run/done/queued)."""
    return EXPERIMENT_STATUS_COLORS.get(label, TEXT_DIM)


def experiment_name_color(selected: bool) -> str:
    """Brighten the active row's name; dim the rest."""
    return TEXT_BRIGHT if selected else TEXT_MID


def experiment_row_style(selected: bool, accent: str) -> str:
    """Row container style: a raised, accent-bordered surface when selected."""
    if not selected:
        return (
            "QWidget#experiment_row { background: transparent; "
            "border: 1px solid transparent; border-radius: 8px; }"
        )
    r, g, b = _hex_to_rgb(accent)
    return (
        "QWidget#experiment_row { "
        f"background: {ROW_LIFT_BG}; "
        f"border: 1px solid rgba({r}, {g}, {b}, 130); "
        "border-radius: 8px; }"
    )


def mono_input_style() -> str:
    """Themed pill style for a QLineEdit, so config fields aren't raw Qt."""
    return (
        "QLineEdit { "
        "background: rgba(255, 255, 255, 8); "
        f"border: 1px solid {HAIRLINE}; border-radius: 6px; "
        f"padding: 3px 7px; color: {TEXT_BRIGHT}; }} "
        "QLineEdit:focus { border-color: rgba(255, 255, 255, 38); }"
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


def mono_font() -> QFont:
    """A monospace QFont for physical-value readouts (units stay column-aligned)."""
    font = QFont()
    font.setFamilies(["IBM Plex Mono", "DejaVu Sans Mono", "Menlo", "Consolas", "monospace"])
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


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
