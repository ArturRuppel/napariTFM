"""Experiments list (top-of-panel substrate): mini-rails + selectable rows."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._icons import stage_action_icon
from napariTFM.widgets._stage_spine import _node_style
from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    muted_accent,
    section_label_style,
    stage_accent,
)

# The four pipeline stages a mini-rail summarises (project/batch are not dots).
PIPELINE_STAGES = ("preprocessing", "displacement", "force", "stress")


def _path_ends_with(path: Path, rel: Path) -> bool:
    """True when *path*'s trailing parts equal *rel*'s (a relative-path match)."""
    return path.parts[-len(rel.parts):] == rel.parts


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
    """One experiment: accent select-bar, name, mini-rail, overall-status chip."""

    selected = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path
        self._selected = False

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        self._selbar = QFrame()
        self._selbar.setFixedWidth(3)
        self._selbar.setStyleSheet("background: transparent;")
        layout.addWidget(self._selbar)

        self._name_label = QLabel(self.name)
        layout.addWidget(self._name_label, 1)

        self.mini_rail = MiniRail()
        layout.addWidget(self.mini_rail)

        self._chip = QLabel("queued")
        layout.addWidget(self._chip)

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

    def set_stage_statuses(self, statuses: dict[str, str]) -> None:
        self.mini_rail.set_statuses(statuses)
        label = overall_status(statuses)
        self._chip.setText(_CHIP_TEXT[label])

    def _emit_selected(self) -> None:
        self.selected.emit(self._path)

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI event
        self._emit_selected()
        super().mousePressEvent(event)


class ExperimentsList(QWidget):
    """Top-of-panel list of experiments; the shared substrate for all three jobs."""

    experiments_changed = Signal()
    active_changed = Signal(str)
    run_all_requested = Signal()

    def __init__(
        self,
        status_fn: Optional[Callable[[str], dict[str, str]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._status_fn = status_fn
        self._paths: list[str] = []
        # path -> {"input_files": {...}, "columns": {...}} — the per-row metadata
        # that makes the list the single config table (P0).
        self._records: dict[str, dict] = {}
        self._rows: list[ExperimentRow] = []
        self._active: Optional[str] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Experiments")
        label.setStyleSheet(section_label_style())
        header.addWidget(label)
        header.addStretch()
        self.add_btn = QToolButton()
        self.add_btn.setObjectName("experiments_add_button")
        self.add_btn.setText("Discover")
        self.add_btn.setIcon(stage_action_icon("plus", muted_accent(stage_accent("displacement"))))
        self.add_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.add_btn.clicked.connect(self._on_add_clicked)
        header.addWidget(self.add_btn)

        self.commit_btn = QToolButton()
        self.commit_btn.setObjectName("experiments_commit_button")
        self.commit_btn.setText("Add to list")
        self.commit_btn.setEnabled(False)
        self.commit_btn.clicked.connect(self.commit_discovered)
        header.addWidget(self.commit_btn)

        # Run all (P4): batch is no longer a separate card — running the whole
        # list is an action on the list itself, walking the rail live.
        self.run_all_btn = QToolButton()
        self.run_all_btn.setObjectName("experiments_run_all_button")
        self.run_all_btn.setText("Run all")
        self.run_all_btn.setIcon(
            stage_action_icon("run", muted_accent(stage_accent("force")))
        )
        self.run_all_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.run_all_btn.setEnabled(False)
        self.run_all_btn.clicked.connect(self.run_all_requested)
        header.addWidget(self.run_all_btn)
        layout.addLayout(header)

        # Staging for the two-step Discover→Commit flow (D2).
        self._discovered: list[str] = []

        layout.addLayout(self._build_config_header())

        self._staging_label = QLabel("")
        layout.addWidget(self._staging_label)

        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(0)
        layout.addLayout(self._rows_box)

        self._meta = QLabel("")
        layout.addWidget(self._meta)
        self._update_meta()

    # -- column config (the table's shared header) -----------------------
    def _build_config_header(self) -> QVBoxLayout:
        """Input file names + free-form columns — the config copied to each batch."""
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
        ):
            field = QLineEdit(default)
            self.file_name_inputs[key] = field
            files.addRow(QLabel(label), field)
        box.addLayout(files)

        # Free-form columns: each is a (name, value) pair copied to every row of
        # the next committed batch (D2).
        self._column_fields: list[tuple[QLineEdit, QLineEdit]] = []
        self._columns_box = QVBoxLayout()
        self._columns_box.setContentsMargins(0, 0, 0, 0)
        self._columns_box.setSpacing(2)
        box.addLayout(self._columns_box)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        self.add_column_btn = QToolButton()
        self.add_column_btn.setObjectName("experiments_add_column_button")
        self.add_column_btn.setText("+ Add column")
        self.add_column_btn.clicked.connect(lambda: self.add_column_field())
        add_row.addWidget(self.add_column_btn)
        add_row.addStretch()
        box.addLayout(add_row)
        return box

    def add_column_field(self, name: str = "", value: str = "") -> None:
        """Append a (name, value) column-config field; the +Add column action."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        name_edit = QLineEdit(name)
        name_edit.setPlaceholderText("column")
        value_edit = QLineEdit(value)
        value_edit.setPlaceholderText("value")
        row.addWidget(name_edit, 1)
        row.addWidget(value_edit, 1)
        self._columns_box.addLayout(row)
        self._column_fields.append((name_edit, value_edit))

    def input_file_config(self) -> dict:
        """Current input file names, blanks dropped (cells is optional)."""
        return {
            key: field.text().strip()
            for key, field in self.file_name_inputs.items()
            if field.text().strip()
        }

    def column_config(self) -> dict:
        """Current free-form columns as name->value, unnamed rows dropped."""
        config: dict[str, str] = {}
        for name_edit, value_edit in self._column_fields:
            name = name_edit.text().strip()
            if name:
                config[name] = value_edit.text().strip()
        return config

    # -- two-step Discover -> Commit (D2) --------------------------------
    def discover(self, root: str | Path) -> list[str]:
        """Step 1: stage the folders under *root* that hold the required inputs.

        Folder-presence only — required inputs are beads + reference (cells is
        optional and excluded from the requirement). Staging never mutates the
        list; the second Commit step does.
        """
        cfg = self.input_file_config()
        required = [cfg.get("beads"), cfg.get("reference")]
        self._discovered = discover_experiment_folders(root, required)
        self._update_staging()
        return list(self._discovered)

    def discovered(self) -> list[str]:
        return list(self._discovered)

    def commit_discovered(self) -> None:
        """Step 2: add the staged folders, copying the column config onto each."""
        if not self._discovered:
            return
        self.add_folders(
            self._discovered,
            input_files=self.input_file_config(),
            columns=self.column_config(),
        )
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

    # -- queries ---------------------------------------------------------
    def experiments(self) -> list[str]:
        return list(self._paths)

    def experiment_records(self) -> list[dict]:
        """Ordered per-row config records: path + input_files + free-form columns."""
        return [
            {
                "path": path,
                "input_files": dict(self._records[path]["input_files"]),
                "columns": dict(self._records[path]["columns"]),
            }
            for path in self._paths
        ]

    def active(self) -> Optional[str]:
        return self._active

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
        self._rebuild_rows()
        if self._active not in self._paths:
            self._active = None
        self.refresh_statuses()
        self._update_meta()
        self.experiments_changed.emit()

    def add_folders(
        self,
        paths: list[str],
        *,
        input_files: Optional[dict] = None,
        columns: Optional[dict] = None,
    ) -> None:
        """Append new folders, copying the given column config onto each new row.

        This is the *commit* half of the two-step Discover→Commit flow (D2): the
        current column config (input file names + free-form name/value pairs) is
        copied — not shared — onto every freshly added row. Existing rows keep
        their own metadata.
        """
        new_paths = [p for p in dict.fromkeys(paths) if p not in self._paths]
        if not new_paths:
            return
        for path in new_paths:
            self._records[path] = {
                "input_files": dict(input_files or {}),
                "columns": dict(columns or {}),
            }
        self.set_experiments(self._paths + new_paths)

    def set_records(self, records: list[dict]) -> None:
        """Rebuild the whole table from saved records (the load path, P4.5).

        Each record is ``{"path", "input_files", "columns"}`` — the same shape
        ``experiment_records`` emits. Replaces every row (paths + per-row
        metadata), seeds the input-file header from the first record so a
        round-tripped config keeps its file names, then refreshes statuses.
        """
        self._paths = list(dict.fromkeys(r["path"] for r in records))
        self._records = {}
        for record in records:
            path = record["path"]
            self._records[path] = {
                "input_files": dict(record.get("input_files") or {}),
                "columns": dict(record.get("columns") or {}),
            }
        if records:
            first_files = self._records[records[0]["path"]]["input_files"]
            for key, field in self.file_name_inputs.items():
                if key in first_files:
                    field.setText(first_files[key])
        self._rebuild_rows()
        if self._active not in self._paths:
            self._active = None
        self.refresh_statuses()
        self._update_meta()
        self.experiments_changed.emit()

    def set_active(self, path: Optional[str]) -> None:
        if path is not None and path not in self._paths:
            return
        self._active = path
        for row in self._rows:
            row.set_selected(row.path == path)
        self.active_changed.emit(path or "")

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
    def _rebuild_rows(self) -> None:
        while self._rows_box.count():
            item = self._rows_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = []
        for path in self._paths:
            row = ExperimentRow(path)
            row.selected.connect(self.set_active)
            self._rows_box.addWidget(row)
            self._rows.append(row)

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
