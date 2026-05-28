import re

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QStyle, QVBoxLayout, QWidget

from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    make_icon_button,
    muted_stage_accent,
    stage_accent,
    stage_header_style,
    status_indicator_style,
)


class _ActionStateSync(QObject):
    """Keep a header proxy button aligned with its delegated child button.

    Parented to ``target`` (the object it installs an event filter on) so the
    filter can never outlive that object: when ``target`` is destroyed Qt
    destroys this filter and removes it cleanly, avoiding a dangling event
    filter dispatched to a deleted object. ``sync`` is guarded so a proxy that
    was destroyed first (e.g. during teardown) is a no-op rather than a crash.
    """

    def __init__(self, target: QWidget, proxy: QWidget):
        super().__init__(target)
        self._target = target
        self._proxy = proxy
        target.installEventFilter(self)
        self.sync()

    def eventFilter(self, obj, event):
        if obj is self._target and event.type() == QEvent.EnabledChange:
            self.sync()
        return super().eventFilter(obj, event)

    def sync(self):
        try:
            self._proxy.setEnabled(self._target.isEnabled())
        except RuntimeError:
            # proxy (or target) C++ object already deleted during teardown.
            pass


class StageSection(QWidget):
    """Reusable workflow stage section with stable header actions and status."""

    def __init__(
        self,
        title: str,
        child: QWidget,
        expanded: bool = False,
        action_targets: dict[str, QWidget] | None = None,
        status: str = "not_started",
        accent: str | None = None,
        status_panel: QWidget | None = None,
        parameter_panel: QWidget | None = None,
        parameters_expanded: bool = False,
    ):
        super().__init__()
        self._title = title
        self._child = child
        self._action_targets = action_targets or {}
        self._status = status
        self.status_panel = status_panel
        self.parameter_panel = parameter_panel
        if accent is not None:
            self._accent = accent
        else:
            inherited = self._find_ancestor_accent()
            if inherited is not None:
                self._accent = inherited
            else:
                self._accent = stage_accent(self._slug)
        self._action_state_syncs: list[_ActionStateSync] = []

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

        self._parameter_content = QWidget()
        parameter_layout = QVBoxLayout()
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.setSpacing(COMPACT_SPACING)
        self._parameter_content.setLayout(parameter_layout)
        if self.parameter_panel is not None:
            parameter_layout.addWidget(self.parameter_panel)

        self._content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(COMPACT_SPACING)
        self._content.setLayout(content_layout)
        content_layout.addWidget(child)

        layout.addLayout(header_layout)
        if self.status_panel is not None:
            layout.addWidget(self.status_panel)
        layout.addWidget(self._parameter_content)
        layout.addWidget(self._content)
        if self.parameter_panel is None:
            self._parameter_content.setVisible(False)

        self.set_status(status)
        self._set_expanded(expanded)
        if self.parameter_panel is None:
            self.params_btn.setChecked(expanded)
        else:
            self.params_btn.setChecked(parameters_expanded)
            self._set_parameter_panel_expanded(parameters_expanded)

    @property
    def _slug(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self._title.lower()).strip("_")
        return slug or "stage"

    def _find_ancestor_accent(self) -> str | None:
        """Walk up the Qt parent chain for a StageSection ancestor's muted accent.

        Currently dormant: StageSection is always constructed before being
        reparented, so self.parent() is None at __init__. The method is kept
        for callers that re-resolve the accent after reparenting (e.g., a
        future move-section-to-new-parent API). add_inner_section bypasses
        this by passing accent= directly to the inner constructor.
        """
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, StageSection):
                return muted_stage_accent(parent._slug)
            parent = parent.parent()
        return None

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

    def _create_action_button(self, action: str, standard_icon: QStyle.StandardPixmap):
        button = make_icon_button(
            self,
            action,
            f"stage_{self._slug}_{action}_button",
            f"{action.capitalize()} {self._title}",
            standard_icon,
        )

        target = self._action_targets.get(action)
        button.setEnabled(target is not None and target.isEnabled())
        if target is not None:
            button.clicked.connect(target.click)
            self._action_state_syncs.append(_ActionStateSync(target, button))
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
        if self.parameter_panel is None:
            button.toggled.connect(self._set_expanded)
        else:
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
        run_target = self._action_targets.get("run")
        button.setEnabled(run_target is not None and run_target.isEnabled())
        if run_target is not None:
            self._action_state_syncs.append(_ActionStateSync(run_target, button))
        button.clicked.connect(self._on_run_cancel_clicked)
        return button

    def _on_run_cancel_clicked(self):
        if self._status == "running":
            target = self._action_targets.get("cancel")
        else:
            target = self._action_targets.get("run")
        if target is not None:
            target.click()

    def _set_expanded(self, expanded: bool):
        self.params_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._content.setVisible(expanded)
        self._child.setVisible(expanded)

    def _set_parameter_panel_expanded(self, expanded: bool):
        self.params_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._parameter_content.setVisible(expanded)
        if self.parameter_panel is not None:
            self.parameter_panel.setVisible(expanded)

    def add_inner_section(
        self,
        title: str,
        child: QWidget,
        expanded: bool = False,
    ) -> "StageSection":
        """Create a nested StageSection inside this section's content area.

        The inner section is rendered with a muted variant of this section's
        accent (CellFlow-style accent inheritance).
        """
        inner = StageSection(
            title,
            child,
            expanded=expanded,
            accent=muted_stage_accent(self._slug),
        )
        self._content.layout().addWidget(inner)
        return inner
