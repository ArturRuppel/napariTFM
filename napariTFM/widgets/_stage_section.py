import re
from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QStyle, QVBoxLayout, QWidget

from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    make_icon_button,
    stage_accent,
    stage_header_style,
    status_indicator_style,
)
from napariTFM.widgets._collapsible_section import CollapsibleSection


class StageSection(QWidget):
    """Reusable workflow stage section with stable header actions and status."""

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

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(COMPACT_SPACING)

        self.status_indicator = QLabel()
        self.status_indicator.setObjectName(f"stage_{self._slug}_status_indicator")
        self.status_indicator.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header_layout.addWidget(self.status_indicator)

        self.header_label = QLabel(title)
        self.header_label.setStyleSheet(stage_header_style(self._accent))
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()

        self.params_btn = self._create_params_button()
        self.run_cancel_btn = self._create_run_cancel_button()
        self.preview_button = self._create_action_button("preview", QStyle.SP_FileDialogContentsView)

        self._toggle_button = self.params_btn

        for button in [self.params_btn, self.run_cancel_btn, self.preview_button]:
            header_layout.addWidget(button)

        if self.parameter_panel is not None:
            self._param_section = CollapsibleSection(
                self._title,
                self.parameter_panel,
                expanded=False,
                accent_color=self._accent,
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
        if self.status_panel is not None:
            layout.addWidget(self.status_panel)
        if self._param_section is not None:
            layout.addWidget(self._param_section)
        layout.addWidget(self._content)

        # Body (action buttons / status) is always visible.
        self._content.setVisible(True)
        self._child.setVisible(True)

        self.set_status(status)

        # The params button is the ONLY collapsible: it toggles the parameter
        # section. Sections without a panel simply have no params affordance.
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

    def set_status(self, status: str):
        self._status = status
        self.status_indicator.setStyleSheet(status_indicator_style(status))
        self.status_indicator.setToolTip(f"{self._title} status: {status}")
        if status == "running":
            self.run_cancel_btn.setIcon(
                self.style().standardIcon(QStyle.SP_DialogCancelButton)
            )
            self.run_cancel_btn.setToolTip(f"Cancel {self._title}")
        else:
            self.run_cancel_btn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )
            self.run_cancel_btn.setToolTip(f"Run {self._title}")
        self._refresh_action_states()

    def _refresh_action_states(self):
        states = self._action_states() if self._action_states is not None else {}
        running = self._status == "running"
        # While running, the run/cancel button is always live (it cancels).
        self.run_cancel_btn.setEnabled(running or states.get("run", False))
        self.preview_button.setEnabled(states.get("preview", False))

    def set_accent(self, accent: str) -> None:
        """Re-accent this section's header (used by the theme picker)."""
        self._accent = accent
        self.header_label.setStyleSheet(stage_header_style(accent))
        if self._param_section is not None:
            self._param_section.set_accent_color(accent)

    def _create_action_button(self, action: str, standard_icon: QStyle.StandardPixmap):
        button = make_icon_button(
            self,
            action,
            f"stage_{self._slug}_{action}_button",
            f"{action.capitalize()} {self._title}",
            standard_icon,
        )
        handler = self._actions.get(action)
        if handler is not None:
            button.clicked.connect(lambda _checked=False, fn=handler: fn())
        button.setEnabled(False)
        return button

    def _create_params_button(self):
        button = make_icon_button(
            self,
            "params",
            f"stage_{self._slug}_params_button",
            f"Toggle {self._title} parameters",
            QStyle.SP_FileDialogDetailedView,
        )
        button.setCheckable(True)
        button.toggled.connect(self._set_parameter_panel_expanded)
        return button

    def _create_run_cancel_button(self):
        button = make_icon_button(
            self,
            "run_cancel",
            f"stage_{self._slug}_run_cancel_button",
            f"Run {self._title}",
            QStyle.SP_MediaPlay,
        )
        button.setEnabled(False)
        button.clicked.connect(self._on_run_cancel_clicked)
        return button

    def _on_run_cancel_clicked(self):
        key = "cancel" if self._status == "running" else "run"
        handler = self._actions.get(key)
        if handler is not None:
            handler()

    def _set_parameter_panel_expanded(self, expanded: bool):
        self.params_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        if self._param_section is not None:
            self._param_section._toggle.setChecked(expanded)
