"""Collapsible, accent-aware section primitive (ported from CellFlow)."""
from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._ui_style import (
    SECTION_MARGIN,
    TINY_MARGIN,
    muted_accent,
)


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget."""

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = False,
        parent: QWidget | None = None,
        title_color: str | None = None,
        accent_color: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title
        self._default_title_color: str | None = title_color
        # An explicit accent_color marks this as the OUTER stage anchor: stripe
        # is thicker and the header text uses the full accent hue. Inner sections
        # leave accent_color=None and inherit a muted variant via parent walk.
        self._explicit_accent: str | None = accent_color
        self._effective_accent: str | None = accent_color
        self._is_outer_accent: bool = accent_color is not None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, TINY_MARGIN, 0, TINY_MARGIN)
        layout.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setObjectName("collapsible_toggle")
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(self._qt_display_text(title))
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._toggle.toggled.connect(self._on_toggled)

        self._status: str | None = None

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)
        header_row.addWidget(self._toggle, 1)
        layout.addLayout(header_row)

        self._content_frame = QFrame()
        self._content_frame.setObjectName("collapsible_content")
        self._content_frame.setFrameShape(QFrame.NoFrame)
        frame_layout = QVBoxLayout(self._content_frame)
        frame_layout.setContentsMargins(
            SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN
        )
        frame_layout.setSpacing(TINY_MARGIN)
        frame_layout.addWidget(inner)

        self._content_frame.setVisible(expanded)
        layout.addWidget(self._content_frame)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self._apply_accent_styles()
        if self._explicit_accent is None:
            QTimer.singleShot(0, self._maybe_inherit_accent)

        if expanded:
            QTimer.singleShot(0, self._notify_layout_change)

    def _apply_accent_styles(self) -> None:
        """(Re)apply header + content-frame stylesheets from current accent state."""
        accent = self._effective_accent
        if accent is None:
            title_color = self._default_title_color
            font_size_pt = 10
            frame_qss = (
                "QFrame#collapsible_content { border: 1px solid #666666; "
                "border-radius: 4px; margin: 0px 2px 2px 2px; }"
            )
        else:
            if self._is_outer_accent:
                title_color = accent
                font_size_pt = 11
            else:
                title_color = muted_accent(accent)
                font_size_pt = 9
            frame_qss = (
                "QFrame#collapsible_content { "
                "border: none; "
                f"border-left: 2px solid {title_color}; "
                "border-radius: 0px; "
                "margin: 0px 2px 2px 2px; "
                "}"
            )
        color_rule = f"color: {title_color}; " if title_color else ""
        self._toggle.setStyleSheet(
            "QToolButton#collapsible_toggle { "
            f"font-weight: bold; font-size: {font_size_pt}pt; border: none; "
            f"padding: 2px; {color_rule}"
            "}"
        )
        self._content_frame.setStyleSheet(frame_qss)

    def _maybe_inherit_accent(self) -> None:
        """Walk up the parent chain and pick up the nearest ancestor's accent."""
        if self._explicit_accent is not None:
            return
        ancestor_color = self._find_ancestor_accent_color()
        if ancestor_color is None or ancestor_color == self._effective_accent:
            return
        self._effective_accent = ancestor_color
        self._is_outer_accent = False
        self._apply_accent_styles()

    def set_accent_color(self, accent_color: str | None) -> None:
        """Set this section's explicit accent and refresh inherited child accents."""
        self._explicit_accent = accent_color
        self._effective_accent = accent_color
        self._is_outer_accent = accent_color is not None
        self._apply_accent_styles()
        self._refresh_descendant_inherited_accents()

    def _refresh_descendant_inherited_accents(self) -> None:
        for child in self.findChildren(CollapsibleSection):
            if child._explicit_accent is not None:
                continue
            ancestor_color = child._find_ancestor_accent_color()
            child._effective_accent = ancestor_color
            child._is_outer_accent = False
            child._apply_accent_styles()

    def _find_ancestor_accent_color(self) -> str | None:
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection):
                if parent._effective_accent is not None:
                    return parent._effective_accent
            parent = parent.parent()
        return None

    def set_header_visible(self, visible: bool) -> None:
        """Show or hide the built-in toggle header row."""
        self._toggle.setVisible(visible)

    def set_status(self, status: str | None) -> None:
        self._status = status

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def title(self) -> str:
        return self._base_title

    @property
    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def expand(self) -> None:
        self._toggle.setChecked(True)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content_frame.setVisible(checked)
        QTimer.singleShot(0, self._notify_layout_change)

    @staticmethod
    def _qt_display_text(title: str) -> str:
        """Escape mnemonic markers so literal ampersands render correctly."""
        return title.replace("&", "&&")

    def _notify_layout_change(self) -> None:
        """Propagate geometry changes up the nested collapsible chain."""
        self.updateGeometry()
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection) and parent.is_expanded:
                parent.updateGeometry()
                QTimer.singleShot(0, parent._notify_layout_change)
                return
            parent.updateGeometry()
            parent = parent.parent()
