"""Experiments list (top-of-panel substrate): an editable column table of rows.

Each discovered experiment is one row. The columns are *derived from the folder
nesting* under the chosen discovery root: every nesting level becomes a column
and the folder name at that level is the row's value for it (root ``/data`` and
folder ``/data/Ctrl/pos_00`` → ``Column 1 = Ctrl``, ``Column 2 = pos_00``). The
column *names* are an editable, table-wide header; the *values* are read-only
(they are the folder names). Rows are multi-selectable (Ctrl/Shift-click) and
deletable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional

from qtpy.QtCore import QEvent, QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QDoubleValidator, QPainter, QPen
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._collapsible_section import CollapsibleSection
from napariTFM.widgets._icons import stage_action_icon
from napariTFM.widgets._stage_spine import _STATUS_TOOLTIP, _node_style
from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    TEXT_DIM,
    TEXT_MID,
    experiment_name_color,
    experiment_row_style,
    experiment_status_color,
    mono_input_style,
    muted_accent,
    stage_accent,
)

# The four pipeline stages a mini-rail summarises (project/batch are not dots).
PIPELINE_STAGES = ("preprocessing", "displacement", "force", "stress")

# Project-level calibration, relocated into the aggregation layer. Free-text
# fields (soft validator, not spinbox stepping); bounds feed the validator.
_CALIBRATION_SPECS = [
    ("pixel_size", "Pixel Size (µm)", 0.001, 100.0),
    ("frame_interval", "Frame Length (min)", 0.001, 1000.0),
]
_INPUT_DECIMALS = 6


def _format_value(value) -> str:
    """Compact text for a calibration value — no trailing-zero noise."""
    return f"{float(value):g}"


def _path_ends_with(path: Path, rel: Path) -> bool:
    """True when *path*'s trailing parts equal *rel*'s (a relative-path match)."""
    return path.parts[-len(rel.parts):] == rel.parts


def nesting_columns(folder: str | Path, root: str | Path) -> dict[str, str]:
    """Derive a row's columns from *folder*'s nesting under *root*.

    Every path component of *folder* relative to *root* becomes a column named
    ``Column 1``, ``Column 2`` … (left to right) whose value is that component's
    folder name. A folder that is not actually under *root* (or equals it) falls
    back to a single ``Column 1`` column holding the leaf folder name, so a row
    always carries at least one column.
    """
    folder = Path(folder)
    try:
        parts = folder.resolve().relative_to(Path(root).resolve()).parts
    except (ValueError, OSError):
        parts = ()
    if not parts:
        parts = (folder.name,)
    return {f"Column {i + 1}": part for i, part in enumerate(parts)}


def discover_experiment_folders(
    root: str | Path,
    required_names: Iterable[Optional[str]],
) -> list[str]:
    """Find folders under *root* that contain **every** file in *required_names*.

    Folder-presence discovery (D2): no filename parsing, no metadata read. Each
    name is a bare file name or a path relative to the experiment folder, matched
    recursively under *root*; a folder qualifies only when all required names
    resolve to existing files inside it. Blank/``None`` names are dropped. A
    missing *root* or an empty requirement set yields an empty list. Returns
    absolute folder paths, sorted.
    """
    root = Path(root)
    names = [n for n in required_names if n]
    if not root.is_dir() or not names:
        return []

    found: dict[Path, set[str]] = {}
    for name in names:
        rel = Path(name)
        for match in sorted(root.rglob(rel.name)):
            if not match.is_file():
                continue
            if len(rel.parts) > 1 and not _path_ends_with(match, rel):
                continue
            folder = match
            for _ in rel.parts:
                folder = folder.parent
            found.setdefault(folder.resolve(), set()).add(name)

    required = set(names)
    return [str(folder) for folder in sorted(found) if found[folder] >= required]


class MiniRail(QWidget):
    """A compact horizontal row of per-stage status dots for one experiment."""

    DOT_R = 4
    DOT_GAP = 12

    # Clicking a dot asks to bring that stage's output on screen (the owner
    # selects the row + decodes that one series). Emits the stage name; the
    # row wraps it with its path.
    stage_clicked = Signal(str)

    def __init__(self, stages=PIPELINE_STAGES, parent=None):
        super().__init__(parent)
        self.stages = tuple(stages)
        self._statuses = {key: "not_started" for key in self.stages}
        # Fractional completion (0..1) of an in-flight "running" stage, or None
        # when no per-frame progress is known (mirrors StageSpine._progress) --
        # fed by a parallel Run-all's real per-stage/per-frame events instead of
        # the flat placeholder mark_running() paints at submission time.
        self._progress: dict[str, Optional[float]] = {key: None for key in self.stages}
        # Index of the dot under the cursor (-1 = none), driving the hover halo.
        self._hover_idx = -1
        self.setFixedSize(self.DOT_GAP * len(self.stages), 2 * self.DOT_R + 6)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Track motion so the dots can light up and swap the cursor per-dot; the
        # tiny row is otherwise indistinguishable from a static status readout.
        self.setMouseTracking(True)

    def set_statuses(self, statuses: dict[str, str]) -> None:
        for key in self.stages:
            if key in statuses:
                self._statuses[key] = statuses[key]
                if statuses[key] != "running":
                    # Stale progress must not leak into this stage's next run
                    # (mirrors StageSpine.set_status's same guard).
                    self._progress[key] = None
        self.update()

    def set_stage_progress(self, stage: str, fraction: Optional[float]) -> None:
        """Set the in-flight fractional completion (0..1) of one stage's dot.

        Only visible while that stage's status is ``"running"``; harmless to
        call at other times since :meth:`paintEvent` ignores it then. Pass
        ``None`` to fall back to the plain solid-fill "running" dot.
        """
        self._progress[stage] = None if fraction is None else max(0.0, min(1.0, fraction))
        self.update()

    def appearance(self, stage: str) -> tuple[Optional[str], str]:
        """Return (fill_hex_or_None, ring_hex) for a stage dot — used by tests/paint."""
        fill, ring = _node_style(self._statuses[stage], stage_accent(stage))
        return (fill.name() if fill is not None else None, ring.name())

    def _clickable_idx_at(self, pos) -> int:
        """Index of the dot under *pos*, or -1 if none / it's an 'off' stage.

        An 'off' (disabled) stage has no output to show, so it neither responds
        to a click nor lights up on hover — matching ``StageSpine``.
        """
        idx = int(pos.x() // self.DOT_GAP)
        if 0 <= idx < len(self.stages) and self._statuses[self.stages[idx]] != "off":
            return idx
        return -1

    def _tooltip_for(self, stage: str) -> str:
        phrase = _STATUS_TOOLTIP.get(self._statuses[stage], self._statuses[stage])
        return f"{stage.capitalize()}: {phrase}"

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI event
        idx = self._clickable_idx_at(event.pos())
        if idx >= 0:
            self.stage_clicked.emit(self.stages[idx])

    def mouseMoveEvent(self, event) -> None:  # pragma: no cover - GUI event
        idx = self._clickable_idx_at(event.pos())
        self.setCursor(Qt.PointingHandCursor if idx >= 0 else Qt.ArrowCursor)
        if idx != self._hover_idx:
            self._hover_idx = idx
            self.update()

    def leaveEvent(self, _event) -> None:  # pragma: no cover - GUI event
        if self._hover_idx != -1:
            self._hover_idx = -1
            self.update()

    def event(self, event):  # pragma: no cover - GUI event
        if event.type() == QEvent.ToolTip:
            idx = int(event.pos().x() // self.DOT_GAP)
            if 0 <= idx < len(self.stages):
                QToolTip.showText(
                    event.globalPos(), self._tooltip_for(self.stages[idx]), self
                )
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2.0
        r = self.DOT_R
        for i, stage in enumerate(self.stages):
            cx = self.DOT_GAP * i + self.DOT_GAP / 2.0
            status = self._statuses[stage]
            fill, ring = _node_style(status, stage_accent(stage))
            if status == "off":
                painter.setPen(QPen(ring, 2, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
                continue
            if i == self._hover_idx:
                # Light the dot up under the cursor so it reads as a button.
                halo = QColor(ring)
                halo.setAlpha(70)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(halo))
                hr = r + 3
                painter.drawEllipse(QRectF(cx - hr, cy - hr, 2 * hr, 2 * hr))
            rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            progress = self._progress[stage]
            if status == "running" and progress is not None:
                # A pie wedge growing clockwise from 12 o'clock reads as a fill
                # level (mirrors StageSpine), so a parallel run's dot shows its
                # real per-stage progress instead of a flat "something is
                # happening" dot for the worker's entire runtime.
                painter.setPen(QPen(ring, 1.5))
                painter.setBrush(QBrush(self.palette().color(self.backgroundRole())))
                painter.drawEllipse(rect)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(ring))
                span = -round(360 * 16 * progress)
                painter.drawPie(rect, 90 * 16, span)
                continue
            centre = fill if fill is not None else self.palette().color(self.backgroundRole())
            painter.setPen(QPen(ring, 1.5))
            painter.setBrush(QBrush(centre))
            painter.drawEllipse(rect)
        painter.end()


# Fixed widths of the non-column cells, shared by the header and every data row
# so the editable column headers line up over their value cells.
_SELBAR_W = 3
_RAIL_W = MiniRail.DOT_GAP * len(PIPELINE_STAGES)
_CHIP_W = 52


def overall_status(statuses: dict[str, str]) -> str:
    """Collapse a stage-status map into a single chip label."""
    values = [v for k, v in statuses.items() if v != "off"]
    if any(v == "running" for v in values):
        return "running"
    if values and all(v == "done" for v in values):
        return "done"
    return "queued"


_CHIP_TEXT = {"running": "run", "done": "done", "queued": "queued"}


class ExperimentRow(QWidget):
    """One experiment: accent select-bar, per-column value cells, mini-rail, chip."""

    clicked = Signal(str, int)  # path, modifier flag: 0 plain, 1 ctrl, 2 shift
    stage_clicked = Signal(str, str)  # path, stage — one dot's on-demand load

    def __init__(
        self,
        path: str,
        values: Optional[list[str]] = None,
        parent=None,
        *,
        preview: bool = False,
    ):
        super().__init__(parent)
        self._path = path
        self._selected = False
        self._preview = preview
        # The row paints its own (styled) background — selected rows lift.
        self.setObjectName("experiment_row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout()
        layout.setContentsMargins(7, 5, 10, 5)
        layout.setSpacing(COMPACT_SPACING + 4)
        self.setLayout(layout)

        self._selbar = QFrame()
        self._selbar.setFixedWidth(_SELBAR_W)
        self._selbar.setStyleSheet("background: transparent;")
        layout.addWidget(self._selbar)

        # One value cell per column; falls back to the leaf folder name when the
        # table has no columns yet (e.g. rows added without a discovery root).
        cells = list(values) if values else [self.name]
        self._value_labels: list[QLabel] = []
        for text in cells:
            label = QLabel(text)
            layout.addWidget(label, 1)
            self._value_labels.append(label)
        self._name_label = self._value_labels[0]

        self.mini_rail = MiniRail()
        self.mini_rail.stage_clicked.connect(
            lambda stage: self.stage_clicked.emit(self._path, stage)
        )
        layout.addWidget(self.mini_rail)

        self._chip = QLabel("queued")
        self._chip.setFixedWidth(_CHIP_W)
        self._chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._chip.setStyleSheet(f"color: {experiment_status_color('queued')};")
        layout.addWidget(self._chip)

        if self._preview:
            # Nothing has run for a not-yet-committed folder — no status to show.
            self.mini_rail.setVisible(False)
            self._chip.setVisible(False)
            for value_label in self._value_labels:
                value_label.setStyleSheet(f"color: {TEXT_DIM}; font-style: italic;")

        # Apply the deselected resting style (row + name colors).
        self.set_selected(False)

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return Path(self._path).name

    @property
    def is_preview(self) -> bool:
        return self._preview

    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, on: bool) -> None:
        self._selected = on
        accent = stage_accent("displacement")
        self._selbar.setStyleSheet(
            f"background: {accent};" if on else "background: transparent;"
        )
        self.setStyleSheet(experiment_row_style(on, accent))
        if not self._preview:
            color = experiment_name_color(on)
            for label in self._value_labels:
                label.setStyleSheet(f"color: {color};")

    def set_stage_statuses(self, statuses: dict[str, str]) -> None:
        """Merge in a (possibly partial) set of stage statuses.

        ``statuses`` may name only some stages (e.g. a single row repainted by
        ``apply_row_statuses``) — the chip label is derived from the mini-rail's
        full merged state, not just this call's keys, so a partial update can't
        under-report it.
        """
        self.mini_rail.set_statuses(statuses)
        label = overall_status(self.mini_rail._statuses)
        text = _CHIP_TEXT[label]
        self._chip.setText(text)
        self._chip.setStyleSheet(f"color: {experiment_status_color(text)};")

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI event
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            flag = 1
        elif mods & Qt.ShiftModifier:
            flag = 2
        else:
            flag = 0
        self.clicked.emit(self._path, flag)
        super().mousePressEvent(event)


class ExperimentsList(QWidget):
    """Top-of-panel table of experiments; the shared substrate for all three jobs."""

    experiments_changed = Signal()
    active_changed = Signal(str)
    run_all_requested = Signal()
    cancel_run_all_requested = Signal()
    # Emitted when a row's stage dot is clicked, asking the owner to bring that
    # experiment's stage on screen (select the row + decode that one series).
    stage_load_requested = Signal(str, str)  # path, stage

    def __init__(
        self,
        status_fn: Optional[Callable[[str], dict[str, str]]] = None,
        parameter_manager=None,
        data_manager=None,
        parent=None,
    ):
        super().__init__(parent)
        self._status_fn = status_fn
        self._parameter_manager = parameter_manager
        self._data_manager = data_manager
        self._paths: list[str] = []
        # path -> {"input_files": {...}, "columns": {...}} — the per-row metadata
        # that makes the list the single config table (P0). A row's ``columns``
        # are keyed by the table-wide, editable column names in ``_column_names``.
        self._records: dict[str, dict] = {}
        # Ordered, editable column names shared by every row (the table header).
        self._column_names: list[str] = []
        self._rows: list[ExperimentRow] = []
        self._preview_rows: list[ExperimentRow] = []
        self._active: Optional[str] = None
        # Multi-selection for delete (Ctrl/Shift-click); ``_anchor`` is the
        # range-select pivot (the last plainly-clicked row).
        self._selected_paths: set[str] = set()
        self._anchor: Optional[str] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        # A plain, non-interactive label — the table below is always visible,
        # so there's nothing to fold away (unlike the Setup section above it).
        label = QLabel("Experiments")
        label.setObjectName("experiments_panel_label")
        label.setStyleSheet(f"color: {TEXT_MID}; font-weight: bold;")
        layout.addWidget(label)

        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(COMPACT_SPACING)
        layout.addLayout(body_layout)

        # Setup: calibration, input-file names, optional output dir — one
        # collapsible block, auto-collapsing after the first commit.
        self.setup_section = self._build_setup_section()
        body_layout.addWidget(self.setup_section)

        # Staging for the two-step Discover→Commit flow (D2). The root is kept so
        # committed rows can derive their columns from the nesting under it.
        self._discovered: list[str] = []
        self._discover_root: Optional[str] = None
        # Preview-row selection (separate from the committed-row selection
        # machinery — preview rows have no "active"/tuning concept).
        self._discovered_selected: set[str] = set()

        self._staging_label = QLabel("")
        self._staging_label.setStyleSheet(f"color: {TEXT_DIM};")
        self._staging_label.setVisible(False)
        body_layout.addWidget(self._staging_label)

        # Rows live in a bounded scroll region: a long discovered list scrolls
        # internally instead of pushing the rest of the panel down. The editable
        # column header is the first item inside, so it always aligns with the
        # value cells (and shares the scrollbar gutter).
        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(2)
        rows_container = QWidget()
        rows_container.setLayout(self._rows_box)
        self._rows_scroll = QScrollArea()
        self._rows_scroll.setObjectName("experiments_rows_scroll")
        self._rows_scroll.setWidgetResizable(True)
        self._rows_scroll.setMinimumHeight(300)
        self._rows_scroll.setMaximumHeight(480)
        self._rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._rows_scroll.setWidget(rows_container)
        body_layout.addWidget(self._rows_scroll)
        self._rebuild_table()

        # List actions live at the foot of the list, just above the count, on
        # two rows so neither one needs the full panel width to stay legible:
        # row 1 builds the list (Discover stages folders, Add to list commits
        # them, Delete removes the selected rows); row 2 runs it (worker count,
        # Run all — P4, batch is an action on the list, not a separate card).
        list_actions = QHBoxLayout()
        list_actions.setContentsMargins(0, 0, 0, 0)

        self.add_btn = QToolButton()
        self.add_btn.setObjectName("experiments_add_button")
        self.add_btn.setText("Discover")
        self.add_btn.setIcon(stage_action_icon("plus", muted_accent(stage_accent("displacement"))))
        self.add_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.add_btn.clicked.connect(self._on_add_clicked)
        self._update_discover_tooltip()
        list_actions.addWidget(self.add_btn)

        self.commit_btn = QToolButton()
        self.commit_btn.setObjectName("experiments_commit_button")
        self.commit_btn.setText("Add to list")
        self.commit_btn.setEnabled(False)
        self.commit_btn.clicked.connect(self.commit_discovered)
        list_actions.addWidget(self.commit_btn)

        self.delete_btn = QToolButton()
        self.delete_btn.setObjectName("experiments_delete_button")
        self.delete_btn.setText("Delete selected")
        self.delete_btn.setToolTip(
            "Remove the selected rows (Ctrl/Shift-click to select several)"
        )
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_selected)
        list_actions.addWidget(self.delete_btn)

        list_actions.addStretch()
        body_layout.addLayout(list_actions)

        run_actions = QHBoxLayout()
        run_actions.setContentsMargins(0, 0, 0, 0)

        self._run_all_active = False
        self.run_all_btn = QToolButton()
        self.run_all_btn.setObjectName("experiments_run_all_button")
        self.run_all_btn.setText("Run all")
        self.run_all_btn.setIcon(
            stage_action_icon("run", muted_accent(stage_accent("force")))
        )
        self.run_all_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.run_all_btn.setEnabled(False)
        self.run_all_btn.clicked.connect(self._on_run_all_clicked)
        run_actions.addWidget(self.run_all_btn)

        run_actions.addStretch()

        workers_label = QLabel("Workers:")
        workers_label.setStyleSheet(f"color: {TEXT_DIM};")
        run_actions.addWidget(workers_label)

        self._num_workers_spinbox = QSpinBox()
        self._num_workers_spinbox.setObjectName("experiments_num_workers_spinbox")
        self._num_workers_spinbox.setRange(1, os.cpu_count() or 1)
        self._num_workers_spinbox.setValue(1)
        self._num_workers_spinbox.setToolTip(
            "How many positions Run-all processes in parallel"
        )
        run_actions.addWidget(self._num_workers_spinbox)
        body_layout.addLayout(run_actions)

        self._meta = QLabel("")
        self._meta.setStyleSheet(f"color: {TEXT_DIM};")
        body_layout.addWidget(self._meta)
        self._update_meta()

        if self._parameter_manager is not None:
            self._parameter_manager.parameter_changed.connect(self._sync_parameter)
        if self._data_manager is not None:
            self._data_manager.add_change_callback(self._sync_output_dir)

    # -- setup: calibration + input-file names + optional output dir ------
    def _build_setup_section(self) -> CollapsibleSection:
        """The one-time-per-batch config: calibration, input names, output dir.

        Wrapped in a CollapsibleSection that starts expanded and auto-collapses
        the first time the experiment table goes from empty to non-empty (see
        ``set_experiments``/``set_records``) — these fields rarely change
        between batches, so hiding them declutters the common case while
        staying one click away.
        """
        inner = QWidget()
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)
        box.addLayout(self._build_calibration_row())
        box.addLayout(self._build_config_header())
        box.addLayout(self._build_output_dir_row())
        inner.setLayout(box)
        return CollapsibleSection("Setup", inner, expanded=True, title_color=TEXT_MID)

    def _build_calibration_row(self) -> QHBoxLayout:
        """Pixel size + frame interval, free-text fields with a soft validator."""
        self.calibration_controls: dict[str, QLineEdit] = {}
        cal = QHBoxLayout()
        cal.setContentsMargins(0, 0, 0, 0)
        cal.setSpacing(COMPACT_SPACING + 4)
        for name, label, min_val, max_val in _CALIBRATION_SPECS:
            field = QLineEdit()
            validator = QDoubleValidator(min_val, max_val, _INPUT_DECIMALS, field)
            validator.setNotation(QDoubleValidator.StandardNotation)
            field.setValidator(validator)
            field.setObjectName(f"workflow_parameter_{name}")
            field.setStyleSheet(mono_input_style())
            if self._parameter_manager is not None:
                field.setText(
                    _format_value(self._parameter_manager.get_ui_parameter(name))
                )
                field.editingFinished.connect(
                    lambda n=name, c=field: self._commit_parameter(n, c)
                )
            self.calibration_controls[name] = field

            caption = QLabel(label)
            caption.setStyleSheet(f"color: {TEXT_MID};")
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(1)
            cell.addWidget(caption)
            cell.addWidget(field)
            cal.addLayout(cell, 1)
        return cal

    def _build_output_dir_row(self) -> QHBoxLayout:
        """Optional output-directory override — last in Setup, after inputs."""
        out = QHBoxLayout()
        out.setContentsMargins(0, 0, 0, 0)
        self.choose_output_dir_btn = QToolButton()
        self.choose_output_dir_btn.setObjectName("experiments_output_dir_button")
        self.choose_output_dir_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.choose_output_dir_btn.setIcon(
            stage_action_icon("plus", muted_accent(stage_accent("project")))
        )
        self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)
        self.output_dir_label = QLabel("")
        self.output_dir_label.setObjectName("project_output_dir_label")
        self.output_dir_label.setStyleSheet(f"color: {TEXT_DIM};")
        self.clear_output_dir_btn = QToolButton()
        self.clear_output_dir_btn.setObjectName("experiments_clear_output_dir_button")
        self.clear_output_dir_btn.setText("×")
        self.clear_output_dir_btn.setToolTip("Remove custom output directory")
        self.clear_output_dir_btn.clicked.connect(self._clear_output_dir)
        out.addWidget(self.choose_output_dir_btn)
        out.addWidget(self.output_dir_label, 1)
        out.addWidget(self.clear_output_dir_btn)
        self._sync_output_dir()
        return out

    def _clear_output_dir(self) -> None:
        """Reset to the default per-experiment output location (unset override)."""
        if self._data_manager is None:
            return
        self._data_manager.set_output_dir(None)

    def _commit_parameter(self, name: str, control: QLineEdit) -> None:
        """Parse a free-text calibration field; revert to last good value if junk."""
        if self._parameter_manager is None:
            return
        try:
            value = float(control.text().strip())
        except ValueError:
            control.setText(
                _format_value(self._parameter_manager.get_ui_parameter(name))
            )
            return
        self._parameter_manager.set_ui_parameter(name, value)

    def _sync_parameter(self, name: str, value) -> None:
        control = self.calibration_controls.get(name)
        if control is None:
            return
        control.blockSignals(True)
        try:
            control.setText(_format_value(value))
        finally:
            control.blockSignals(False)

    def _choose_output_dir(self) -> None:  # pragma: no cover - GUI dialog
        if self._data_manager is None:
            return
        current = self._data_manager.output_dir or Path.home()
        path = QFileDialog.getExistingDirectory(
            self, "Select Pipeline Output Directory", str(current)
        )
        if path:
            self._apply_output_dir(path)

    def _apply_output_dir(self, path: str) -> None:
        """Commit a chosen output dir to the data manager and announce it."""
        self._data_manager.set_output_dir(path)

    def _sync_output_dir(self) -> None:
        path = getattr(self._data_manager, "output_dir", None)
        if path is None:
            self.output_dir_label.setText("")
            self.output_dir_label.setVisible(False)
            self.output_dir_label.setToolTip("")
            self.choose_output_dir_btn.setText("Add custom output directory")
            self.choose_output_dir_btn.setToolTip(
                "Optional — overrides the default per-experiment output location"
            )
            self.clear_output_dir_btn.setVisible(False)
            return
        text = str(path)
        self.output_dir_label.setText(text)
        self.output_dir_label.setVisible(True)
        self.output_dir_label.setToolTip(text)
        self.choose_output_dir_btn.setText("Change output directory")
        self.choose_output_dir_btn.setToolTip("Choose a different output directory")
        self.clear_output_dir_btn.setVisible(True)

    # -- input-file config (the discovery requirements) ------------------
    def _build_config_header(self) -> QVBoxLayout:
        """Input file names — the discovery requirements copied to each batch.

        Columns are no longer configured here; they are derived from the folder
        nesting at discovery time and edited in the table header itself.
        """
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)

        files = QFormLayout()
        files.setContentsMargins(0, 0, 0, 0)
        files.setSpacing(2)
        self.file_name_inputs: dict[str, QLineEdit] = {}
        for key, label, default in (
            ("beads", "Beads file", "beads.tif"),
            ("reference", "Reference file", "reference.tif"),
            ("cells", "Cells file (optional)", "cells.tif"),
            ("masks", "Masks file (optional)", "masks.tif"),
        ):
            field = QLineEdit(default)
            field.setStyleSheet(mono_input_style())
            self.file_name_inputs[key] = field
            field.textChanged.connect(lambda _t: self._update_discover_tooltip())
            name = QLabel(label)
            name.setStyleSheet(f"color: {TEXT_MID};")
            files.addRow(name, field)
        box.addLayout(files)
        return box

    def input_file_config(self) -> dict:
        """Current input file names, blanks dropped (cells is optional)."""
        return {
            key: field.text().strip()
            for key, field in self.file_name_inputs.items()
            if field.text().strip()
        }

    def _discover_tooltip_text(self) -> str:
        """Plain-language scan description from the filled input-file names."""
        names = [
            field.text().strip()
            for key in ("beads", "reference", "cells", "masks")
            for field in (self.file_name_inputs[key],)
            if field.text().strip()
        ]
        if not names:
            return "Choose a folder to scan for experiment subfolders."
        if len(names) == 1:
            joined = names[0]
        else:
            joined = f"{', '.join(names[:-1])} and {names[-1]}"
        return (
            f"napariTFM will scan the chosen folder for subfolders containing "
            f"{joined}, and initialize each for analysis."
        )

    def _update_discover_tooltip(self) -> None:
        self.add_btn.setToolTip(self._discover_tooltip_text())

    # -- columns (the editable table header) -----------------------------
    def column_names(self) -> list[str]:
        """The ordered, editable column names shared by every row."""
        return list(self._column_names)

    def rename_column(self, index: int, new_name: str) -> None:
        """Rename column *index* table-wide, carrying each row's value across.

        Editing one header field renames that column for every row — the columns
        are a shared table header, not per-row tags. Blank names are allowed but
        drop out of the metadata emitted by :meth:`experiment_records`.
        """
        if not 0 <= index < len(self._column_names):
            return
        old = self._column_names[index]
        new = new_name.strip()
        if new == old:
            return
        self._column_names[index] = new
        for path in self._paths:
            cols = self._records[path]["columns"]
            if old in cols:
                cols[new] = cols.pop(old)

    def _ensure_columns(self, names: Iterable[str]) -> None:
        """Append any column names not already present, preserving order."""
        for name in names:
            if name not in self._column_names:
                self._column_names.append(name)

    # -- two-step Discover -> Commit (D2) --------------------------------
    def discover(self, root: str | Path) -> list[str]:
        """Step 1: stage the folders under *root* that hold the required inputs.

        Folder-presence only — required inputs are beads + reference (cells is
        optional and excluded from the requirement). The root is remembered so
        committed rows can derive their columns from the nesting under it.
        Staging never mutates the committed list; the second Commit step does.
        The staged set renders immediately as dimmed preview rows in the table;
        a second call to ``discover`` *replaces* the current preview set rather
        than merging into it.
        """
        cfg = self.input_file_config()
        required = [cfg.get("beads"), cfg.get("reference")]
        self._discover_root = str(root)
        self._discovered = discover_experiment_folders(root, required)
        self._discovered_selected = set()
        self._update_staging()
        self._rebuild_table()
        return list(self._discovered)

    def discovered(self) -> list[str]:
        return list(self._discovered)

    def commit_discovered(self) -> None:
        """Step 2: add the staged folders with columns from the folder nesting."""
        if not self._discovered:
            return
        root = self._discover_root
        pairs = [(path, nesting_columns(path, root)) for path in self._discovered]
        self._discovered = []
        self._discovered_selected = set()
        self._add_records(pairs, self.input_file_config())
        self._update_staging()

    def _update_staging(self) -> None:
        n = len(self._discovered)
        self.commit_btn.setEnabled(n > 0)
        self._staging_label.setText(
            f"{n} folder{'s' if n != 1 else ''} discovered — Add to list"
            if n
            else ""
        )
        # Don't reserve a blank line when nothing is staged.
        self._staging_label.setVisible(n > 0)

    # -- queries ---------------------------------------------------------
    def experiments(self) -> list[str]:
        return list(self._paths)

    def experiment_records(self) -> list[dict]:
        """Ordered per-row config records: path + input_files + columns.

        A row's columns are projected onto the shared, ordered column names;
        blank-named columns are dropped so the emitted metadata stays clean.
        """
        return [
            {
                "path": path,
                "input_files": dict(self._records[path]["input_files"]),
                "columns": {
                    name: self._records[path]["columns"].get(name, "")
                    for name in self._column_names
                    if name.strip()
                },
            }
            for path in self._paths
        ]

    def active(self) -> Optional[str]:
        return self._active

    def num_workers(self) -> int:
        """Chosen parallel batch-worker count for Run-all."""
        return self._num_workers_spinbox.value()

    def selected_rows(self) -> list[str]:
        """The paths currently multi-selected for deletion, in row order."""
        return [path for path in self._paths if path in self._selected_paths]

    def input_files_for(self, path: str) -> dict:
        """The discovery-defined input file names for one row (empty if unknown)."""
        record = self._records.get(path)
        return dict(record["input_files"]) if record else {}

    def meta_text(self) -> str:
        return self._meta.text()

    # -- mutation --------------------------------------------------------
    def set_experiments(self, paths: list[str]) -> None:
        was_empty = not self._paths
        self._paths = list(dict.fromkeys(paths))  # de-dup, keep order
        # Preserve metadata for surviving paths; seed empty for the rest.
        self._records = {
            path: self._records.get(path, {"input_files": {}, "columns": {}})
            for path in self._paths
        }
        self._selected_paths &= set(self._paths)
        self._rebuild_table()
        if self._active not in self._paths:
            self._active = None
        self.refresh_statuses()
        self._update_meta()
        self._update_delete_btn()
        self.experiments_changed.emit()
        # Adding rows to a previously-empty list should preload an active
        # position rather than leaving the list with no selection.
        if not self._paths:
            self.setup_section.set_expanded(True)
        elif was_empty:
            self.set_active(self._paths[0])
            self.setup_section.set_expanded(False)

    def _add_records(self, pairs, input_files: Optional[dict] = None) -> None:
        """Append ``(path, columns)`` rows, extending the shared column header.

        Each new row carries its own columns dict (derived from folder nesting,
        or supplied directly); any new column names extend the table-wide header.
        Existing rows keep their own metadata.
        """
        new = [(p, dict(cols)) for p, cols in pairs if p not in self._paths]
        # de-dup within the incoming batch, keeping the first occurrence
        seen: set[str] = set()
        new = [(p, c) for p, c in new if not (p in seen or seen.add(p))]
        if not new:
            return
        for _, cols in new:
            self._ensure_columns(cols.keys())
        for path, cols in new:
            self._records[path] = {
                "input_files": dict(input_files or {}),
                "columns": cols,
            }
        self.set_experiments(self._paths + [p for p, _ in new])

    def add_folders(
        self,
        paths: list[str],
        *,
        input_files: Optional[dict] = None,
        columns: Optional[dict] = None,
    ) -> None:
        """Append new folders, copying the given column config onto each new row.

        The *commit* half of the Discover→Commit flow uses :meth:`commit_discovered`
        (which derives columns from nesting). This entry point applies one shared
        ``columns`` dict to every added row and is used by the saved-state/load
        paths and tests.
        """
        pairs = [(path, dict(columns or {})) for path in dict.fromkeys(paths)]
        self._add_records(pairs, input_files)

    def set_records(self, records: list[dict]) -> None:
        """Rebuild the whole table from saved records (the load path, P4.5).

        Each record is ``{"path", "input_files", "columns"}`` — the same shape
        ``experiment_records`` emits. Replaces every row (paths + per-row
        metadata), rebuilds the shared column header from the union of column
        names (first-seen order), seeds the input-file header from the first
        record, then refreshes statuses.
        """
        self._paths = list(dict.fromkeys(r["path"] for r in records))
        self._records = {}
        self._column_names = []
        self._selected_paths = set()
        for record in records:
            path = record["path"]
            cols = dict(record.get("columns") or {})
            self._ensure_columns(cols.keys())
            self._records[path] = {
                "input_files": dict(record.get("input_files") or {}),
                "columns": cols,
            }
        if records:
            first_files = self._records[records[0]["path"]]["input_files"]
            for key, field in self.file_name_inputs.items():
                if key in first_files:
                    field.setText(first_files[key])
        self._rebuild_table()
        if self._active not in self._paths:
            self._active = None
        self.refresh_statuses()
        self._update_meta()
        self._update_delete_btn()
        self.setup_section.set_expanded(not self._paths)
        self.experiments_changed.emit()

    def set_active(self, path: Optional[str], *, selection=None) -> None:
        """Set the single active (downstream-driving) row.

        Unless *selection* is given, the multi-selection collapses to just the
        active row, so a plain click both activates a row and clears any prior
        multi-selection.
        """
        if path is not None and path not in self._paths:
            return
        # Only a genuine change of the active row drives the downstream
        # clear-and-reload. Re-clicking the already-active row (or a multi-select
        # gesture that leaves the active row unchanged) still refreshes the
        # selection styling below, but must NOT re-emit active_changed —
        # otherwise it would wipe the active experiment's in-memory overlays and
        # reload from disk for no reason (CODE_REVIEW_FINDINGS.md #8).
        active_changed = path != self._active
        self._active = path
        if selection is None:
            self._selected_paths = {path} if path else set()
        else:
            self._selected_paths = {p for p in selection if p in self._paths}
        self._apply_selection_styles()
        self._update_delete_btn()
        if active_changed:
            self.active_changed.emit(path or "")

    def delete_selected(self) -> None:
        """Remove selected rows: preview-staged first, else committed rows."""
        if self._discovered_selected:
            self._discovered = [
                p for p in self._discovered if p not in self._discovered_selected
            ]
            self._discovered_selected = set()
            self._update_staging()
            self._rebuild_table()
            self._update_delete_btn()
            return
        if not self._selected_paths:
            return
        remaining = [p for p in self._paths if p not in self._selected_paths]
        if self._active in self._selected_paths:
            self._active = None
        self._selected_paths = set()
        self.set_experiments(remaining)

    def refresh_statuses(self) -> None:
        """Re-read every row's stage status from disk and repaint (eager).

        ``_status_fn`` reads which measures each row's `TFMresults.ome.tif`
        carries — a header-only walk (no pixel decode), cached by
        ``ntfm.populated_measures`` — so every row's dots show the real on-disk
        status the moment folders land in the list. Cheap enough to run
        synchronously across the whole list.
        """
        if self._status_fn is None:
            return
        for row in self._rows:
            try:
                statuses = self._status_fn(row.path)
            except Exception:
                statuses = {}
            row.set_stage_statuses(statuses)
        self._update_meta()

    def apply_row_statuses(self, path: str, statuses: dict[str, str]) -> None:
        """Paint one row's already-known statuses directly, no re-scan of the list.

        For a caller that already computed one experiment's status — a batch
        folder that just finished — so a single row repaints without walking
        every other row's `.ntfm`.
        """
        for row in self._rows:
            if row.path == path:
                row.set_stage_statuses(statuses)
                break

    def _on_row_stage_clicked(self, path: str, stage: str) -> None:
        """A row's stage dot was clicked: ask the owner to show that stage.

        The dots already carry the eager on-disk status, so nothing needs
        fetching here — this just forwards the request to decode that stage's
        pixels into the viewer (the owner selects the row and loads the series).
        """
        self.stage_load_requested.emit(path, stage)

    def follow_streaming(self, path: str) -> None:
        """Track the position a live run is processing (worklist §3).

        Highlights *path* as the active row so the list follows the streaming
        sink. Unlike :meth:`set_active` this emits **no** ``active_changed`` and
        does no disk reload: the sink already owns the viewer's content during a
        run, and reloading from disk would fight the frames it is streaming in.
        A no-op for a path not in the table.
        """
        if path not in self._paths:
            return
        self._active = path
        self._selected_paths = {path}
        self._apply_selection_styles()
        self._update_delete_btn()

    def mark_running(self, path: str) -> None:
        """Flip one experiment's enabled stage dots to 'running' (live, P4).

        Off stages stay off; other rows are untouched. The shell calls this as
        the batch reaches each folder, then ``refresh_statuses`` re-reads disk
        truth once the folder's ``.ntfm`` is written.
        """
        for row in self._rows:
            if row.path != path:
                continue
            statuses = {
                stage: ("running" if status != "off" else "off")
                for stage, status in row.mini_rail._statuses.items()
            }
            row.set_stage_statuses(statuses)
            return

    def set_row_stage_progress(
        self, path: str, stage: str, status: str, fraction: Optional[float]
    ) -> None:
        """Paint one row's one stage dot with real in-flight progress (P4/#10).

        Fed by a parallel Run-all's per-stage/per-frame events (routed through
        the shell's ``_on_batch_stage_progress``), so a parallel-mode row's dot
        fills the same way the single-experiment detail panel's ``StageSpine``
        already does, instead of sitting on the flat ``mark_running()``
        placeholder for the worker's entire runtime. A no-op for a path not in
        the table.
        """
        for row in self._rows:
            if row.path != path:
                continue
            row.set_stage_statuses({stage: status})
            row.mini_rail.set_stage_progress(stage, fraction)
            return

    def _on_run_all_clicked(self) -> None:
        """One button, two roles: start a Run-all, or cancel the live one."""
        if self._run_all_active:
            self.cancel_run_all_requested.emit()
        else:
            self.run_all_requested.emit()

    def set_run_all_active(self, active: bool) -> None:
        """Toggle the Run-all button between 'Run all' and a live 'Cancel'.

        While active the button stays enabled (independent of row count) so the
        in-flight batch can always be cancelled.
        """
        self._run_all_active = active
        icon = "cancel" if active else "run"
        self.run_all_btn.setText("Cancel" if active else "Run all")
        self.run_all_btn.setIcon(
            stage_action_icon(icon, muted_accent(stage_accent("force")))
        )
        if active:
            self.run_all_btn.setEnabled(True)

    # -- internals -------------------------------------------------------
    def _on_row_clicked(self, path: str, flag: int) -> None:
        """Resolve a row click into the new active row + multi-selection.

        flag 0 = plain (select only), 1 = Ctrl (toggle), 2 = Shift (range from
        the anchor). The clicked row becomes active so the parameter panel
        follows the click even while several rows stay selected for deletion.
        """
        if flag == 1:  # Ctrl-click: toggle this row in/out of the selection
            selection = set(self._selected_paths)
            if path in selection:
                selection.discard(path)
            else:
                selection.add(path)
            self._anchor = path
            active = path if path in selection else next(iter(selection), None)
            self.set_active(active, selection=selection)
        elif flag == 2 and self._anchor in self._paths:  # Shift-click: range
            lo, hi = sorted(
                (self._paths.index(self._anchor), self._paths.index(path))
            )
            self.set_active(path, selection=set(self._paths[lo : hi + 1]))
        else:  # plain click: single select + activate
            self._anchor = path
            self.set_active(path)

    def _on_preview_row_clicked(self, path: str, _flag: int) -> None:
        """Toggle one not-yet-committed row in/out of the delete selection."""
        if path in self._discovered_selected:
            self._discovered_selected.discard(path)
        else:
            self._discovered_selected.add(path)
        for row in self._preview_rows:
            row.set_selected(row.path in self._discovered_selected)
        self._update_delete_btn()

    def _apply_selection_styles(self) -> None:
        for row in self._rows:
            row.set_selected(row.path in self._selected_paths)

    def _update_delete_btn(self) -> None:
        if hasattr(self, "delete_btn"):
            self.delete_btn.setEnabled(
                bool(self._selected_paths) or bool(self._discovered_selected)
            )

    def keyPressEvent(self, event) -> None:  # pragma: no cover - GUI event
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and (
            self._selected_paths or self._discovered_selected
        ):
            self.delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def _build_header_widget(self) -> QWidget:
        """The editable column-name header, aligned over the value cells."""
        widget = QWidget()
        widget.setObjectName("experiments_table_header")
        row = QHBoxLayout()
        row.setContentsMargins(7, 5, 10, 5)
        row.setSpacing(COMPACT_SPACING + 4)
        widget.setLayout(row)

        lead = QWidget()
        lead.setFixedWidth(_SELBAR_W)
        row.addWidget(lead)

        self._header_fields: list[QLineEdit] = []
        if self._column_names:
            for index, name in enumerate(self._column_names):
                field = QLineEdit(name)
                field.setObjectName(f"experiments_column_header_{index}")
                field.setStyleSheet(mono_input_style())
                field.editingFinished.connect(
                    lambda i=index, f=field: self.rename_column(i, f.text())
                )
                row.addWidget(field, 1)
                self._header_fields.append(field)
        else:
            placeholder = QLineEdit("Folder")
            placeholder.setEnabled(False)
            placeholder.setStyleSheet(mono_input_style())
            row.addWidget(placeholder, 1)

        rail = QWidget()
        rail.setFixedWidth(_RAIL_W)
        row.addWidget(rail)
        chip = QWidget()
        chip.setFixedWidth(_CHIP_W)
        row.addWidget(chip)
        return widget

    def _rebuild_table(self) -> None:
        while self._rows_box.count():
            item = self._rows_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = []
        self._preview_rows = []
        self._rows_box.addWidget(self._build_header_widget())
        for path in self._paths:
            values = [
                self._records[path]["columns"].get(name, "")
                for name in self._column_names
            ]
            row = ExperimentRow(path, values or None)
            row.clicked.connect(self._on_row_clicked)
            row.stage_clicked.connect(self._on_row_stage_clicked)
            row.set_selected(path in self._selected_paths)
            self._rows_box.addWidget(row)
            self._rows.append(row)
        for path in self._discovered:
            row = ExperimentRow(path, preview=True)
            row.clicked.connect(self._on_preview_row_clicked)
            row.set_selected(path in self._discovered_selected)
            self._rows_box.addWidget(row)
            self._preview_rows.append(row)
        self._update_table_visibility()

    def _update_table_visibility(self) -> None:
        """Collapse the empty table so the action bar sits flush under the
        input-file form. The bounded scroll region (and its "Folder" header
        placeholder) only earns its 300px once there are committed or
        preview rows to show."""
        self._rows_scroll.setVisible(bool(self._paths) or bool(self._discovered))

    def _update_meta(self) -> None:
        n = len(self._paths)
        self._meta.setText(f"{n} experiment{'s' if n != 1 else ''}")
        self.run_all_btn.setEnabled(n > 0)

    def _on_add_clicked(self) -> None:  # pragma: no cover - GUI dialog
        dialog = QFileDialog(self, "Discover experiments under a root folder")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            roots = dialog.selectedFiles()
            if roots:
                self.discover(roots[0])
