import re
from typing import Callable

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    SECTION_MARGIN,
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

    enabled_changed = Signal(bool)

    def __init__(
        self,
        title: str,
        child: QWidget,
        actions: dict[str, Callable] | None = None,
        action_states: Callable[[], dict[str, bool]] | None = None,
        action_states_changed=None,
        status: str = "not_started",
        accent: str | None = None,
        parameter_panel: QWidget | None = None,
        parameters_expanded: bool = False,
        optional: bool = False,
        enabled: bool = True,
        extra_actions: list[dict] | None = None,
        preview_is_toggle: bool = False,
    ):
        super().__init__()
        self._title = title
        self._child = child
        self._actions = actions or {}
        self._extra_action_specs = extra_actions or []
        self._action_states = action_states
        self._status = status
        self._optional = optional
        self._enabled = enabled if optional else True
        # Toggle-style preview (persistent on/off) renders the header pill as a
        # checkable toggle so its pressed/active state reflects whether preview
        # is currently on — distinct from the momentary, non-checkable one-shot
        # preview button the stages use. No stage opts into this today (it was
        # the preprocessing live preview), but the machinery stays generic.
        self._preview_is_toggle = preview_is_toggle
        self.enable_btn = None
        self.parameter_panel = parameter_panel
        if accent is not None:
            self._accent = accent
        else:
            self._accent = stage_accent(self._slug)
        self._accent_above = self._accent
        self._accent_below = self._accent

        self.spine = StageSpine(self._accent, status=status, label=self._title)

        outer = QHBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setLayout(outer)

        body = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # No uniform inter-row spacing: a collapsed param panel and an empty body
        # must contribute zero height (and zero gap) so sections sit flush. The
        # only deliberate gap is a hair under the pill, added explicitly below.
        layout.setSpacing(0)
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

        # Stage-specific auxiliary actions (e.g. Bayesian auto-λ, mesh preview)
        # live in the header as glyph pills, alongside the standard controls.
        self.extra_buttons: dict[str, object] = {}
        self._extra_button_icons: dict[object, str] = {}
        for spec in self._extra_action_specs:
            key = spec["key"]
            icon_name = spec["icon"]
            button = self._create_glyph_button(
                key, spec.get("glyph", ""), spec["tooltip"], icon_name
            )
            handler = spec.get("handler")
            if handler is not None:
                button.clicked.connect(
                    lambda _checked=False, fn=handler: fn()
                )
            button.setEnabled(False)
            self.extra_buttons[key] = button
            self._extra_button_icons[button] = icon_name

        self.params_btn = self._create_glyph_button(
            "params", "⚙", f"Toggle {title} parameters", "params", checkable=True
        )
        self.params_btn.toggled.connect(self._set_parameter_panel_expanded)

        self.preview_button = self._create_glyph_button(
            "preview", "▷", self._preview_tooltip(False), "preview",
            checkable=self._preview_is_toggle,
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
        extra_button_list = list(self.extra_buttons.values())
        self._action_buttons = [
            *extra_button_list,
            self.params_btn,
            self.preview_button,
            self.run_cancel_btn,
        ]
        # Static-icon buttons re-tint on theme change; run/cancel re-tints by status.
        self._static_button_icons = {
            self.params_btn: "params",
            self.preview_button: "preview",
            **self._extra_button_icons,
        }
        for button in self._action_buttons:
            header_layout.addWidget(button)

        if self._optional:
            self.enable_btn = self._create_glyph_button(
                "enable", "⏻", f"{'Disable' if self._enabled else 'Enable'} {title}",
                "power", checkable=True
            )
            self.enable_btn.setChecked(self._enabled)
            self.enable_btn.toggled.connect(self._on_enable_toggled)
            self._action_buttons.append(self.enable_btn)
            self._static_button_icons[self.enable_btn] = "power"
            header_layout.insertWidget(0, self.enable_btn)

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
        # The one deliberate gap in this zero-spacing column: a uniform hair of
        # air under every pill so adjacent sections never glue together when
        # their bodies/param panels are collapsed. Applied unconditionally so all
        # stages space identically.
        layout.addSpacing(SECTION_MARGIN)
        if self._param_section is not None:
            layout.addWidget(self._param_section)
        layout.addWidget(self._content)

        # Body is always visible; only the params/data sections collapse.
        self._content.setVisible(True)
        self._child.setVisible(True)

        self.set_status(status)

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

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _effective_status(self) -> str:
        """What the spine shows: 'off' when disabled, else the real status."""
        return self._status if self._enabled else "off"

    def set_enabled(self, enabled: bool) -> None:
        """Turn this (optional) stage on/off; off reads as 'skipped', not missing."""
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if self.enable_btn is not None:
            self.enable_btn.blockSignals(True)
            self.enable_btn.setChecked(enabled)
            self.enable_btn.blockSignals(False)
            self.enable_btn.setToolTip(
                f"{'Disable' if enabled else 'Enable'} {self._title}"
            )
        self.spine.set_status(self._effective_status())
        self._refresh_action_states()
        self.enabled_changed.emit(enabled)

    def _on_enable_toggled(self, checked: bool) -> None:
        self.set_enabled(checked)

    def set_status(self, status: str):
        self._status = status
        if status == "running":
            self.run_cancel_btn.setIcon(stage_action_button_icon("cancel", self._accent))
            self.run_cancel_btn.setToolTip(f"Cancel {self._title}")
        else:
            self.run_cancel_btn.setIcon(stage_action_button_icon("run", self._accent))
            self.run_cancel_btn.setToolTip(f"Run {self._title}")
        self.spine.set_status(self._effective_status())
        self._refresh_action_states()

    def set_progress(self, fraction: float | None) -> None:
        """Forward a stage's fractional completion (0..1) to its spine node."""
        self.spine.set_progress(fraction)

    def _preview_tooltip(self, active: bool) -> str:
        if not self._preview_is_toggle:
            return f"Preview {self._title}"
        return f"{'Stop' if active else 'Start'} live preview of {self._title}"

    def _refresh_action_states(self):
        if not self._enabled:
            self.run_cancel_btn.setEnabled(False)
            self.preview_button.setEnabled(False)
            for button in self.extra_buttons.values():
                button.setEnabled(False)
            return
        states = self._action_states() if self._action_states is not None else {}
        running = self._status == "running"
        self.run_cancel_btn.setEnabled(running or states.get("run", False))
        self.preview_button.setEnabled(states.get("preview", False))
        if self._preview_is_toggle:
            active = states.get("preview_active", False)
            self.preview_button.setChecked(active)
            self.preview_button.setToolTip(self._preview_tooltip(active))
        for key, button in self.extra_buttons.items():
            button.setEnabled(states.get(key, False))

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
            # Hide the whole section when collapsed so it reserves no height or
            # gap — collapsing the inner frame alone leaves a dead stub.
            self._param_section.setVisible(expanded)
