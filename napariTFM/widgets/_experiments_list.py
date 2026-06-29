"""Experiments list (top-of-panel substrate): an editable column table of rows.

Each discovered experiment is one row. The columns are *derived from the folder
nesting* under the chosen discovery root: every nesting level becomes a column
and the folder name at that level is the row's value for it (root ``/data`` and
folder ``/data/Ctrl/pos_00`` → ``Level 1 = Ctrl``, ``Level 2 = pos_00``). The
column *names* are an editable, table-wide header; the *values* are read-only
(they are the folder names). Rows are multi-selectable (Ctrl/Shift-click) and
deletable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from qtpy.QtCore import QRectF, Qt, Signal
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._icons import stage_action_icon
from napariTFM.widgets._stage_spine import _node_style
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
    ``Level 1``, ``Level 2`` … (left to right) whose value is that component's
    folder name. A folder that is not actually under *root* (or equals it) falls
    back to a single ``Level 1`` column holding the leaf folder name, so a row
    always carries at least one column.
    """
    folder = Path(folder)
    try:
        parts = folder.resolve().relative_to(Path(root).resolve()).parts
    except (ValueError, OSError):
        parts = ()
    if not parts:
        parts = (folder.name,)
    return {f"Level {i + 1}": part for i, part in enumerate(parts)}


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

    def __init__(self, stages=PIPELINE_STAGES, parent=None):
        super().__init__(parent)
        self.stages = tuple(stages)
        self._statuses = {key: "not_started" for key in self.stages}
        self.setFixedSize(self.DOT_GAP * len(self.stages), 2 * self.DOT_R + 6)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_statuses(self, statuses: dict[str, str]) -> None:
        for key in self.stages:
            if key in statuses:
                self._statuses[key] = statuses[key]
        self.update()

    def appearance(self, stage: str) -> tuple[Optional[str], str]:
        """Return (fill_hex_or_None, ring_hex) for a stage dot — used by tests/paint."""
        fill, ring = _node_style(self._statuses[stage], stage_accent(stage))
        return (fill.name() if fill is not None else None, ring.name())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2.0
        r = self.DOT_R
        for i, stage in enumerate(self.stages):
            cx = self.DOT_GAP * i + self.DOT_GAP / 2.0
            fill, ring = _node_style(self._statuses[stage], stage_accent(stage))
            if self._statuses[stage] == "off":
                painter.setPen(QPen(ring, 2, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
                continue
            centre = fill if fill is not None else self.palette().color(self.backgroundRole())
            painter.setPen(QPen(ring, 1.5))
            painter.setBrush(QBrush(centre))
            painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
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

    selected = Signal(str)  # legacy single-select (kept for back-compat callers)
    clicked = Signal(str, int)  # path, modifier flag: 0 plain, 1 ctrl, 2 shift

    def __init__(self, path: str, values: Optional[list[str]] = None, parent=None):
        super().__init__(parent)
        self._path = path
        self._selected = False
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
        layout.addWidget(self.mini_rail)

        self._chip = QLabel("queued")
        self._chip.setFixedWidth(_CHIP_W)
        self._chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._chip.setStyleSheet(f"color: {experiment_status_color('queued')};")
        layout.addWidget(self._chip)

        # Apply the deselected resting style (row + name colors).
        self.set_selected(False)

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return Path(self._path).name

    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, on: bool) -> None:
        self._selected = on
        accent = stage_accent("displacement")
        self._selbar.setStyleSheet(
            f"background: {accent};" if on else "background: transparent;"
        )
        self.setStyleSheet(experiment_row_style(on, accent))
        color = experiment_name_color(on)
        for label in self._value_labels:
            label.setStyleSheet(f"color: {color};")

    def set_stage_statuses(self, statuses: dict[str, str]) -> None:
        self.mini_rail.set_statuses(statuses)
        label = overall_status(statuses)
        text = _CHIP_TEXT[label]
        self._chip.setText(text)
        self._chip.setStyleSheet(f"color: {experiment_status_color(text)};")

    def _emit_selected(self) -> None:
        self.selected.emit(self._path)

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI event
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            flag = 1
        elif mods & Qt.ShiftModifier:
            flag = 2
        else:
            flag = 0
        self.clicked.emit(self._path, flag)
        self._emit_selected()
        super().mousePressEvent(event)


class ExperimentsList(QWidget):
    """Top-of-panel table of experiments; the shared substrate for all three jobs."""

    experiments_changed = Signal()
    active_changed = Signal(str)
    run_all_requested = Signal()
    output_dir_changed = Signal()

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
        self._active: Optional[str] = None
        # Multi-selection for delete (Ctrl/Shift-click); ``_anchor`` is the
        # range-select pivot (the last plainly-clicked row).
        self._selected_paths: set[str] = set()
        self._anchor: Optional[str] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel("EXPERIMENTS")
        label.setStyleSheet(f"color: {TEXT_MID}; font-weight: bold;")
        header.addWidget(label)
        header.addStretch()

        layout.addLayout(header)

        # Project-level calibration + output directory (the aggregation layer
        # owns these now; the old Project section is gone).
        layout.addLayout(self._build_project_strip())

        # Staging for the two-step Discover→Commit flow (D2). The root is kept so
        # committed rows can derive their columns from the nesting under it.
        self._discovered: list[str] = []
        self._discover_root: Optional[str] = None

        layout.addLayout(self._build_config_header())

        self._staging_label = QLabel("")
        self._staging_label.setStyleSheet(f"color: {TEXT_DIM};")
        self._staging_label.setVisible(False)
        layout.addWidget(self._staging_label)

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
        layout.addWidget(self._rows_scroll)
        self._rebuild_table()

        # List actions live at the foot of the list, just above the count:
        # Discover stages folders, Add to list commits them, Delete removes the
        # selected rows, Run all walks the whole list (P4 — batch is an action on
        # the list, not a separate card).
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)

        self.add_btn = QToolButton()
        self.add_btn.setObjectName("experiments_add_button")
        self.add_btn.setText("Discover")
        self.add_btn.setIcon(stage_action_icon("plus", muted_accent(stage_accent("displacement"))))
        self.add_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.add_btn.clicked.connect(self._on_add_clicked)
        self._update_discover_tooltip()
        actions.addWidget(self.add_btn)

        self.commit_btn = QToolButton()
        self.commit_btn.setObjectName("experiments_commit_button")
        self.commit_btn.setText("Add to list")
        self.commit_btn.setEnabled(False)
        self.commit_btn.clicked.connect(self.commit_discovered)
        actions.addWidget(self.commit_btn)

        self.delete_btn = QToolButton()
        self.delete_btn.setObjectName("experiments_delete_button")
        self.delete_btn.setText("Delete selected")
        self.delete_btn.setToolTip(
            "Remove the selected rows (Ctrl/Shift-click to select several)"
        )
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_selected)
        actions.addWidget(self.delete_btn)

        actions.addStretch()

        self.run_all_btn = QToolButton()
        self.run_all_btn.setObjectName("experiments_run_all_button")
        self.run_all_btn.setText("Run all")
        self.run_all_btn.setIcon(
            stage_action_icon("run", muted_accent(stage_accent("force")))
        )
        self.run_all_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.run_all_btn.setEnabled(False)
        self.run_all_btn.clicked.connect(self.run_all_requested)
        actions.addWidget(self.run_all_btn)
        layout.addLayout(actions)

        self._meta = QLabel("")
        self._meta.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self._meta)
        self._update_meta()

        if self._parameter_manager is not None:
            self._parameter_manager.parameter_changed.connect(self._sync_parameter)
        if self._data_manager is not None:
            self._data_manager.add_change_callback(self._sync_output_dir)
            self._sync_output_dir()

    # -- project-level calibration + output (the aggregation layer) -------
    def _build_project_strip(self) -> QVBoxLayout:
        """Pixel/frame calibration + an output-directory picker, themed."""
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)

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
        box.addLayout(cal)

        out = QHBoxLayout()
        out.setContentsMargins(0, 0, 0, 0)
        self.choose_output_dir_btn = QToolButton()
        self.choose_output_dir_btn.setObjectName("experiments_output_dir_button")
        self.choose_output_dir_btn.setToolTip("Choose output directory")
        self.choose_output_dir_btn.setIcon(
            stage_action_icon("files", muted_accent(stage_accent("project")))
        )
        self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)
        self.output_dir_label = QLabel("No output directory")
        self.output_dir_label.setObjectName("project_output_dir_label")
        self.output_dir_label.setStyleSheet(f"color: {TEXT_DIM};")
        out.addWidget(self.choose_output_dir_btn)
        out.addWidget(self.output_dir_label, 1)
        box.addLayout(out)
        return box

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
        self.output_dir_changed.emit()

    def _sync_output_dir(self) -> None:
        path = getattr(self._data_manager, "output_dir", None)
        if path is None:
            self.output_dir_label.setText("No output directory")
            self.output_dir_label.setToolTip("")
            return
        text = str(path)
        self.output_dir_label.setText(text)
        self.output_dir_label.setToolTip(text)

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
        Staging never mutates the list; the second Commit step does.
        """
        cfg = self.input_file_config()
        required = [cfg.get("beads"), cfg.get("reference")]
        self._discover_root = str(root)
        self._discovered = discover_experiment_folders(root, required)
        self._update_staging()
        return list(self._discovered)

    def discovered(self) -> list[str]:
        return list(self._discovered)

    def commit_discovered(self) -> None:
        """Step 2: add the staged folders with columns from the folder nesting."""
        if not self._discovered:
            return
        root = self._discover_root
        pairs = [(path, nesting_columns(path, root)) for path in self._discovered]
        self._add_records(pairs, self.input_file_config())
        self._discovered = []
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
        self.experiments_changed.emit()

    def set_active(self, path: Optional[str], *, selection=None) -> None:
        """Set the single active (downstream-driving) row.

        Unless *selection* is given, the multi-selection collapses to just the
        active row, so a plain click both activates a row and clears any prior
        multi-selection.
        """
        if path is not None and path not in self._paths:
            return
        self._active = path
        if selection is None:
            self._selected_paths = {path} if path else set()
        else:
            self._selected_paths = {p for p in selection if p in self._paths}
        self._apply_selection_styles()
        self._update_delete_btn()
        self.active_changed.emit(path or "")

    def delete_selected(self) -> None:
        """Remove every multi-selected row from the table."""
        if not self._selected_paths:
            return
        remaining = [p for p in self._paths if p not in self._selected_paths]
        if self._active in self._selected_paths:
            self._active = None
        self._selected_paths = set()
        self.set_experiments(remaining)

    def refresh_statuses(self) -> None:
        if self._status_fn is None:
            return
        for row in self._rows:
            row.set_stage_statuses(self._status_fn(row.path))
        self._update_meta()

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

    def _apply_selection_styles(self) -> None:
        for row in self._rows:
            row.set_selected(row.path in self._selected_paths)

    def _update_delete_btn(self) -> None:
        if hasattr(self, "delete_btn"):
            self.delete_btn.setEnabled(bool(self._selected_paths))

    def keyPressEvent(self, event) -> None:  # pragma: no cover - GUI event
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self._selected_paths:
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
        self._rows_box.addWidget(self._build_header_widget())
        for path in self._paths:
            values = [
                self._records[path]["columns"].get(name, "")
                for name in self._column_names
            ]
            row = ExperimentRow(path, values or None)
            row.clicked.connect(self._on_row_clicked)
            row.set_selected(path in self._selected_paths)
            self._rows_box.addWidget(row)
            self._rows.append(row)
        self._update_table_visibility()

    def _update_table_visibility(self) -> None:
        """Collapse the empty table so the action bar sits flush under the
        input-file form. The bounded scroll region (and its "Folder" header
        placeholder) only earns its 300px once there are rows to show."""
        self._rows_scroll.setVisible(bool(self._paths))

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
