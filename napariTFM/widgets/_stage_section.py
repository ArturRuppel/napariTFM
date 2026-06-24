import re
from typing import Callable

from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    make_stage_action_button,
    stage_accent,
    stage_action_button_icon,
    stage_header_action_button_style,
    stage_header_style,
)
from napariTFM.widgets._collapsible_section import CollapsibleSection
from napariTFM.widgets._stage_spine import StageSpine


class StageSection(QWidget):
    """Workflow stage section with a CellFlow-style glyph-pill header."""

    def __init__(
        self,
        title: str,
        child: QWidget,
        actions: dict[str, Callable] | None = None,
        action_states: Callable[[], dict[str, bool]] | None = None,
        action_states_changed=None,
        status: str = "not_started",
        accent: str | None = None,
        status_panel: QWidget | None = None,
        parameter_panel: QWidget | None = None,
        parameters_expanded: bool = False,
    ):
        super().__init__()
        self._title = title
        self._child = child
        self._actions = actions or {}
        self._action_states = action_states
        self._status = status
        self.status_panel = status_panel
        self.parameter_panel = parameter_panel
        if accent is not None:
            self._accent = accent
        else:
            self._accent = stage_accent(self._slug)
        self._accent_above = self._accent
        self._accent_below = self._accent

        self.spine = StageSpine(self._accent, status=status)

        outer = QHBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setLayout(outer)

        body = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)
        body.setLayout(layout)

        outer.addWidget(self.spine)
        outer.addWidget(body, 1)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(COMPACT_SPACING)

        self.header_label = QLabel(title)
        self.header_label.setStyleSheet(stage_header_style(self._accent))
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()

        self.files_btn = self._create_glyph_button(
            "files", "🔍", f"Show {title} data", "files", checkable=True
        )
        self.files_btn.toggled.connect(self._set_status_panel_expanded)

        self.params_btn = self._create_glyph_button(
            "params", "⚙", f"Toggle {title} parameters", "params", checkable=True
        )
        self.params_btn.toggled.connect(self._set_parameter_panel_expanded)

        self.preview_button = self._create_glyph_button(
            "preview", "▷", f"Preview {title}", "preview"
        )
        preview_handler = self._actions.get("preview")
        if preview_handler is not None:
            self.preview_button.clicked.connect(
                lambda _checked=False, fn=preview_handler: fn()
            )
        self.preview_button.setEnabled(False)

        self.run_cancel_btn = self._create_glyph_button(
            "run_cancel", "▶", f"Run {title}", "run"
        )
        self.run_cancel_btn.clicked.connect(self._on_run_cancel_clicked)
        self.run_cancel_btn.setEnabled(False)

        self._toggle_button = self.params_btn
        self._action_buttons = [
            self.files_btn,
            self.params_btn,
            self.preview_button,
            self.run_cancel_btn,
        ]
        # Static-icon buttons re-tint on theme change; run/cancel re-tints by status.
        self._static_button_icons = {
            self.files_btn: "files",
            self.params_btn: "params",
            self.preview_button: "preview",
        }
        for button in self._action_buttons:
            header_layout.addWidget(button)

        if self.status_panel is not None:
            self._status_section = CollapsibleSection(
                "Data", self.status_panel, expanded=False, accent_color=self._accent
            )
            self._status_section.set_header_visible(False)
        else:
            self._status_section = None

        if self.parameter_panel is not None:
            self._param_section = CollapsibleSection(
                self._title, self.parameter_panel, expanded=False, accent_color=self._accent
            )
            self._param_section.set_header_visible(False)
        else:
            self._param_section = None

        self._content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(COMPACT_SPACING)
        self._content.setLayout(content_layout)
        content_layout.addWidget(child)

        layout.addLayout(header_layout)
        if self._status_section is not None:
            layout.addWidget(self._status_section)
        if self._param_section is not None:
            layout.addWidget(self._param_section)
        layout.addWidget(self._content)

        # Body is always visible; only the params/data sections collapse.
        self._content.setVisible(True)
        self._child.setVisible(True)

        self.set_status(status)

        self.files_btn.setVisible(self.status_panel is not None)
        has_panel = self._param_section is not None
        self.params_btn.setVisible(has_panel)
        self.params_btn.setChecked(parameters_expanded if has_panel else False)
        if has_panel:
            self._set_parameter_panel_expanded(parameters_expanded)

        if action_states_changed is not None:
            action_states_changed.connect(self._refresh_action_states)
        self._refresh_action_states()

    @property
    def _slug(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self._title.lower()).strip("_")
        return slug or "stage"

    @property
    def status(self) -> str:
        return self._status

    def set_status(self, status: str):
        self._status = status
        if status == "running":
            self.run_cancel_btn.setIcon(stage_action_button_icon("cancel", self._accent))
            self.run_cancel_btn.setToolTip(f"Cancel {self._title}")
        else:
            self.run_cancel_btn.setIcon(stage_action_button_icon("run", self._accent))
            self.run_cancel_btn.setToolTip(f"Run {self._title}")
        self.spine.set_status(status)
        self._refresh_action_states()

    def _refresh_action_states(self):
        states = self._action_states() if self._action_states is not None else {}
        running = self._status == "running"
        self.run_cancel_btn.setEnabled(running or states.get("run", False))
        self.preview_button.setEnabled(states.get("preview", False))

    def set_accent(self, accent: str) -> None:
        """Re-accent the header pill + action buttons (used by the theme picker)."""
        self._accent = accent
        self.header_label.setStyleSheet(stage_header_style(accent))
        for button in self._action_buttons:
            button.setStyleSheet(stage_header_action_button_style(accent))
        for button, icon_name in self._static_button_icons.items():
            button.setIcon(stage_action_button_icon(icon_name, accent))
        run_name = "cancel" if self._status == "running" else "run"
        self.run_cancel_btn.setIcon(stage_action_button_icon(run_name, accent))
        if self._param_section is not None:
            self._param_section.set_accent_color(accent)
        if self._status_section is not None:
            self._status_section.set_accent_color(accent)
        self.spine.set_accents(accent, self._accent_above, self._accent_below)

    def set_accents(self, accent: str, above: str | None = None, below: str | None = None) -> None:
        """Set the stage accent plus its neighbours, for the gradient spine."""
        self._accent = accent
        self._accent_above = above or accent
        self._accent_below = below or accent
        self.set_accent(accent)

    def _create_glyph_button(
        self, action: str, glyph: str, tooltip: str, icon_name: str, checkable: bool = False
    ):
        return make_stage_action_button(
            self,
            f"stage_{self._slug}_{action}_button",
            tooltip,
            glyph,
            self._accent,
            checkable=checkable,
            icon_name=icon_name,
        )

    def _on_run_cancel_clicked(self):
        key = "cancel" if self._status == "running" else "run"
        handler = self._actions.get(key)
        if handler is not None:
            handler()

    def _set_parameter_panel_expanded(self, expanded: bool):
        if self._param_section is not None:
            self._param_section._toggle.setChecked(expanded)

    def _set_status_panel_expanded(self, expanded: bool):
        if self._status_section is not None:
            self._status_section._toggle.setChecked(expanded)
        self.files_btn.setToolTip(
            f"{'Hide' if expanded else 'Show'} {self._title} data"
        )
