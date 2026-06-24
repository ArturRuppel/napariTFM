# Experiments-List-at-Top Implementation Plan (Slices 5–7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the experiment/folder picker to the top of the napariTFM panel as one shared list that feeds three jobs — tune (select one → the pipeline rail operates on it), batch (run all → one `.ntfm` each), and aggregate (outputs → `.iris`).

**Architecture:** A new custom-painted `ExperimentsList` widget sits above the pipeline rail. Each row is an input folder showing a **mini-rail** of per-stage status dots (reusing the spine's `_node_style`) plus an overall status chip. Selecting a row sets the *active experiment*; the existing pipeline rail re-labels to operate on it. "Run all" (Slice 6) walks every experiment through the enabled stages via the existing `BatchAnalysis` backend, streaming mini-rail updates live. An aggregate footer (Slice 7) folds every experiment's `.ntfm` into one `.iris`.

**Tech Stack:** PyQt6/qtpy, custom `QPainter` widgets, `QFileDialog` multi-select, pytest with offscreen `QApplication`, existing `_ui_style.py` ramp/accent machinery and `_stage_spine._node_style`.

**Branch:** `ui-redesign`. **CRLF discipline:** every touched file must stay pure-LF — verify with `! git grep -Il $'\r' -- <file>` before each commit (see `feedback-line-endings`).

---

## Decisions locked in (read before building)

- **An "experiment" is an input folder**, matching the existing batch backend: it produces `<folder>/TFM_data/<folder.name>.ntfm` (`batch_analysis.py:472`). Experiments are stored as absolute path strings.
- **Selecting an experiment is light in Slice 5**: it sets `self._active_experiment` and re-labels the pipeline context. It does **not** auto-load heavy image data into napari (interactive runs are already preview-only — commit `f03732d`). Heavy "operate on it" wiring is Slice 6's job.
- **Per-experiment status is coarse in Slice 5** (`.ntfm` exists → enabled stages "done"; inputs present → preprocessing "ready"; stress "off" when disabled). Per-field granularity is a Slice 6 refinement once live runs drive it. The status function is **injected** so unit tests stub it.
- **Slice 5 is additive**: the experiments list is inserted above the stage sections; the existing bottom `BatchAnalysisWidget` folder list stays untouched until Slice 6 retires it.
- **Labels (condition/replicate/position) live in Slice 7's aggregator only**, never in the batch/tune UI.

---

## File Structure

| File | Responsibility |
|---|---|
| `napariTFM/widgets/_icons.py` (modify) | add a `plus` stroked-SVG icon for "Add folders" |
| `napariTFM/widgets/_experiments_list.py` (create) | `MiniRail`, `ExperimentRow`, `ExperimentsList` — the whole top-of-panel substrate |
| `napariTFM/widgets/_widget.py` (modify) | instantiate + insert the list, provide the status function, pipeline context label, signal wiring, state persistence |
| `napariTFM/backend/aggregate.py` (create, Slice 7) | `aggregate_to_iris()` — fold `.ntfm` outputs into one `.iris` |
| `tests/test_icons.py` (modify) | expect `plus` in `ICON_NAMES` |
| `tests/test_experiments_list.py` (create) | MiniRail appearance, row selection, list add/select/meta/refresh |
| `tests/test_workflow_shell.py` (modify) | state round-trip, context label, refresh-on-disable, run-all (Slice 6) |
| `tests/test_aggregate.py` (create, Slice 7) | `.ntfm` → `.iris` aggregation |

---

# SLICE 5 — Experiments list at top (the keystone)

### Task 1: Add a `plus` icon for "Add folders"

**Files:**
- Modify: `napariTFM/widgets/_icons.py` (the `_ICON_BODIES` dict)
- Test: `tests/test_icons.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_icons.py`:

```python
def test_plus_icon_is_registered_and_renders_opaque(app):
    from napariTFM.widgets._icons import ICON_NAMES, stage_action_pixmap

    assert "plus" in ICON_NAMES
    pm = stage_action_pixmap("plus", "#7ad151", size=18)
    img = pm.toImage()
    opaque = sum(
        img.pixelColor(x, y).alpha() > 0
        for x in range(img.width())
        for y in range(img.height())
    )
    assert opaque > 0
```

(Reuse the existing `app` fixture in that file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_icons.py::test_plus_icon_is_registered_and_renders_opaque -v`
Expected: FAIL — `"plus" in ICON_NAMES` is False.

- [ ] **Step 3: Add the icon body**

In `_icons.py`, add one entry to `_ICON_BODIES` (alongside `power`):

```python
    "plus": '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_icons.py -v`
Expected: PASS (all icon tests green; the `ICON_NAMES`-count test, if any, also updates automatically since it derives from `_ICON_BODIES`).

- [ ] **Step 5: Commit**

```bash
git grep -Il $'\r' -- napariTFM/widgets/_icons.py tests/test_icons.py   # expect no output
git add napariTFM/widgets/_icons.py tests/test_icons.py
git commit -m "Add 'plus' stroked-SVG icon for Add-folders control

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `MiniRail` — a row of per-stage status dots

**Files:**
- Create: `napariTFM/widgets/_experiments_list.py`
- Test: `tests/test_experiments_list.py`

The mini-rail reuses the spine's node colour logic so dots read identically to the big rail: filled = done, amber = running, hollow ring = ready, dim = not_started, off-grey = off.

- [ ] **Step 1: Write the failing test**

Create `tests/test_experiments_list.py`:

```python
import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._experiments_list import (
    MiniRail,
    PIPELINE_STAGES,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_minirail_has_a_dot_per_pipeline_stage(app):
    rail = MiniRail()
    assert rail.stages == PIPELINE_STAGES
    assert len(PIPELINE_STAGES) == 4


def test_minirail_done_dot_is_filled_with_stage_accent(app):
    from napariTFM.widgets._ui_style import stage_accent

    rail = MiniRail()
    rail.set_statuses({"force": "done"})
    fill, ring = rail.appearance("force")
    assert fill == stage_accent("force")
    assert ring == stage_accent("force")


def test_minirail_ready_dot_is_hollow_ring(app):
    rail = MiniRail()
    rail.set_statuses({"displacement": "ready"})
    fill, ring = rail.appearance("displacement")
    assert fill is None
    assert ring is not None


def test_minirail_off_dot_is_recessed_and_distinct_from_not_started(app):
    rail = MiniRail()
    rail.set_statuses({"stress": "off"})
    off_fill, off_ring = rail.appearance("stress")
    none_fill, none_ring = rail.appearance("preprocessing")  # not_started default
    assert off_fill is None and none_fill is None
    assert off_ring != none_ring  # off uses the recessed grey, not the dim grey
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_experiments_list.py -v`
Expected: FAIL — `ModuleNotFoundError: napariTFM.widgets._experiments_list`.

- [ ] **Step 3: Write `MiniRail`**

Create `napariTFM/widgets/_experiments_list.py`:

```python
"""Experiments list (top-of-panel substrate): mini-rails + selectable rows."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._stage_spine import _node_style
from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    section_label_style,
    stage_accent,
)

# The four pipeline stages a mini-rail summarises (project/batch are not dots).
PIPELINE_STAGES = ("preprocessing", "displacement", "force", "stress")


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_experiments_list.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git grep -Il $'\r' -- napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py   # expect no output
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Add MiniRail: per-experiment row of stage status dots

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `ExperimentRow` — selectable row (selbar + name + mini-rail + chip)

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py`
- Test: `tests/test_experiments_list.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiments_list.py`:

```python
from napariTFM.widgets._experiments_list import ExperimentRow, overall_status


def test_overall_status_done_when_all_enabled_stages_done():
    statuses = {"preprocessing": "done", "displacement": "done",
                "force": "done", "stress": "off"}
    assert overall_status(statuses) == "done"


def test_overall_status_running_when_any_stage_running():
    statuses = {"preprocessing": "done", "displacement": "running",
                "force": "not_started", "stress": "off"}
    assert overall_status(statuses) == "running"


def test_overall_status_queued_otherwise():
    statuses = {"preprocessing": "ready", "displacement": "not_started",
                "force": "not_started", "stress": "off"}
    assert overall_status(statuses) == "queued"


def test_experiment_row_exposes_path_and_name(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    assert row.path == "/data/Ctrl/pos_00"
    assert row.name == "pos_00"


def test_experiment_row_click_emits_selected_path(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    seen = []
    row.selected.connect(seen.append)
    row._emit_selected()  # proxy for mousePressEvent (offscreen-safe)
    assert seen == ["/data/Ctrl/pos_00"]


def test_experiment_row_set_selected_toggles_state(app):
    row = ExperimentRow("/data/Ctrl/pos_00")
    assert row.is_selected() is False
    row.set_selected(True)
    assert row.is_selected() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_experiments_list.py -v`
Expected: FAIL — `ImportError: cannot import name 'ExperimentRow'`.

- [ ] **Step 3: Implement `overall_status` + `ExperimentRow`**

Append to `_experiments_list.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_experiments_list.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git grep -Il $'\r' -- napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py   # expect no output
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Add ExperimentRow + overall_status chip collapsing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `ExperimentsList` — header, add/remove, single-selection model, meta line

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py`
- Test: `tests/test_experiments_list.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiments_list.py`:

```python
from napariTFM.widgets._experiments_list import ExperimentsList


def test_list_starts_empty(app):
    widget = ExperimentsList()
    assert widget.experiments() == []
    assert widget.active() is None


def test_set_experiments_populates_rows(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b"])
    assert widget.experiments() == ["/data/a", "/data/b"]


def test_add_folders_appends_without_duplicates(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a"])
    widget.add_folders(["/data/a", "/data/b"])
    assert widget.experiments() == ["/data/a", "/data/b"]


def test_selecting_a_row_sets_single_active_and_emits(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b"])
    seen = []
    widget.active_changed.connect(seen.append)

    widget.set_active("/data/b")

    assert widget.active() == "/data/b"
    assert seen == ["/data/b"]
    rows = widget._rows
    assert rows[1].is_selected() is True
    assert rows[0].is_selected() is False


def test_meta_line_counts_experiments(app):
    widget = ExperimentsList()
    widget.set_experiments(["/data/a", "/data/b", "/data/c"])
    assert "3 experiments" in widget.meta_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_experiments_list.py -v`
Expected: FAIL — `ImportError: cannot import name 'ExperimentsList'`.

- [ ] **Step 3: Implement `ExperimentsList`**

Append to `_experiments_list.py`:

```python
class ExperimentsList(QWidget):
    """Top-of-panel list of experiments; the shared substrate for all three jobs."""

    experiments_changed = Signal()
    active_changed = Signal(str)

    def __init__(
        self,
        status_fn: Optional[Callable[[str], dict[str, str]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._status_fn = status_fn
        self._paths: list[str] = []
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
        self.add_btn.setText("Add folders")
        self.add_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.add_btn.clicked.connect(self._on_add_clicked)
        header.addWidget(self.add_btn)
        layout.addLayout(header)

        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(0)
        layout.addLayout(self._rows_box)

        self._meta = QLabel("")
        layout.addWidget(self._meta)
        self._update_meta()

    # -- queries ---------------------------------------------------------
    def experiments(self) -> list[str]:
        return list(self._paths)

    def active(self) -> Optional[str]:
        return self._active

    def meta_text(self) -> str:
        return self._meta.text()

    # -- mutation --------------------------------------------------------
    def set_experiments(self, paths: list[str]) -> None:
        self._paths = list(dict.fromkeys(paths))  # de-dup, keep order
        self._rebuild_rows()
        if self._active not in self._paths:
            self._active = None
        self.refresh_statuses()
        self._update_meta()
        self.experiments_changed.emit()

    def add_folders(self, paths: list[str]) -> None:
        merged = list(dict.fromkeys(self._paths + list(paths)))
        if merged == self._paths:
            return
        self.set_experiments(merged)

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

    def _on_add_clicked(self) -> None:  # pragma: no cover - GUI dialog
        dialog = QFileDialog(self, "Add experiment folders")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            self.add_folders(dialog.selectedFiles())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_experiments_list.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git grep -Il $'\r' -- napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py   # expect no output
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Add ExperimentsList: header, add/remove, single-selection, meta

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire `ExperimentsList` into the main widget + pipeline context label

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (layout near `:391`; new helper methods; `refresh_stage_statuses`)
- Test: `tests/test_workflow_shell.py` (extend the existing `_stub_main_widget` helper)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_shell.py`:

```python
def test_experiments_list_is_present_above_pipeline(monkeypatch):
    widget = _stub_main_widget(monkeypatch)
    assert hasattr(widget, "experiments_list")
    assert widget.experiments_list is not None


def test_selecting_experiment_updates_pipeline_context_label(monkeypatch):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.set_experiments(["/data/Ctrl/pos_00"])
    widget.experiments_list.set_active("/data/Ctrl/pos_00")
    assert "pos_00" in widget._pipeline_context_label.text()


def test_disabling_stress_refreshes_experiment_minirails(monkeypatch):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.set_experiments(["/data/Ctrl/pos_00"])
    section = widget._stage_sections_by_key["stress"]
    section.set_enabled(False)
    row = widget.experiments_list._rows[0]
    fill, ring = row.mini_rail.appearance("stress")
    assert fill is None  # stress dot now reads 'off'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py::test_experiments_list_is_present_above_pipeline -v`
Expected: FAIL — `AttributeError: ... has no attribute 'experiments_list'`.

- [ ] **Step 3: Implement the wiring**

In `_widget.py`, just after `container_layout.addWidget(self.project_section)` (`:391`), insert:

```python
        self.experiments_list = ExperimentsList(
            status_fn=self._experiment_stage_status,
        )
        self.experiments_list.active_changed.connect(
            self._on_active_experiment_changed
        )
        self.experiments_list.experiments_changed.connect(
            self._on_experiments_changed
        )
        container_layout.addWidget(self.experiments_list)

        self._active_experiment: str | None = None
        self._pipeline_context_label = QLabel("Pipeline")
        self._pipeline_context_label.setStyleSheet(section_label_style())
        container_layout.addWidget(self._pipeline_context_label)
```

Add the import at the top of `_widget.py`:

```python
from napariTFM.widgets._experiments_list import ExperimentsList, PIPELINE_STAGES
```

(and `section_label_style` to the existing `_ui_style` import, and `QLabel` if not already imported.)

Add these methods to `napariTFMWidget`:

```python
    def _experiment_stage_status(self, path: str) -> dict[str, str]:
        """Coarse per-stage status for an experiment folder (Slice 5).

        `.ntfm` present -> enabled stages 'done'; inputs present ->
        preprocessing 'ready'; disabled stages 'off'. Slice 6 refines this
        to per-field granularity driven by live runs.
        """
        from pathlib import Path

        folder = Path(path)
        ntfm = folder / "TFM_data" / f"{folder.name}.ntfm"
        inputs_ready = (folder / "beads.tif").exists() and (
            folder / "reference.tif"
        ).exists()
        disabled = set(self._disabled_stages())
        statuses: dict[str, str] = {}
        for stage in PIPELINE_STAGES:
            if stage in disabled:
                statuses[stage] = "off"
            elif ntfm.exists():
                statuses[stage] = "done"
            elif inputs_ready and stage == "preprocessing":
                statuses[stage] = "ready"
            else:
                statuses[stage] = "not_started"
        return statuses

    def _on_active_experiment_changed(self, path: str) -> None:
        self._active_experiment = path or None
        if self._active_experiment is None:
            self._pipeline_context_label.setText("Pipeline")
        else:
            from pathlib import Path

            self._pipeline_context_label.setText(
                f"Pipeline · tuning ▸ {Path(self._active_experiment).name}"
            )
        self._write_config()

    def _on_experiments_changed(self) -> None:
        self._write_config()
```

In `refresh_stage_statuses()` (`:629`), add a final line so mini-rails track stage changes:

```python
        self.experiments_list.refresh_statuses()
```

Because `_on_stage_enabled_changed` already calls `refresh_stage_statuses()`, disabling stress now also refreshes the mini-rails — no extra wiring needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py -v`
Expected: PASS (the three new tests + existing ones).

- [ ] **Step 5: Commit**

```bash
git grep -Il $'\r' -- napariTFM/widgets/_widget.py tests/test_workflow_shell.py   # expect no output
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Wire ExperimentsList above pipeline; add tuning context label

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Persist experiments + active selection in state

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (`get_state` `:645`, `set_state` `:654`)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_shell.py`:

```python
def test_state_round_trips_experiments_and_active(monkeypatch):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.set_experiments(["/data/a", "/data/b"])
    widget.experiments_list.set_active("/data/b")

    state = widget.get_state()
    assert state["experiments"] == ["/data/a", "/data/b"]
    assert state["active_experiment"] == "/data/b"

    fresh = _stub_main_widget(monkeypatch)
    fresh.set_state(state)
    assert fresh.experiments_list.experiments() == ["/data/a", "/data/b"]
    assert fresh.experiments_list.active() == "/data/b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py::test_state_round_trips_experiments_and_active -v`
Expected: FAIL — `KeyError: 'experiments'`.

- [ ] **Step 3: Implement state plumbing**

In `get_state()`, add two keys to the returned dict:

```python
        "experiments": self.experiments_list.experiments(),
        "active_experiment": self.experiments_list.active(),
```

In `set_state()`, inside the `_applying_state` guard (after the `disabled` loop), add:

```python
        self.experiments_list.set_experiments(state.get("experiments") or [])
        active = state.get("active_experiment")
        if active:
            self.experiments_list.set_active(active)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen pytest tests/test_workflow_shell.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + visual check + commit**

```bash
QT_QPA_PLATFORM=offscreen pytest -q          # expect all green (~270+)
```

Render a PNG to eyeball the list (offscreen `grab().save(...)`), then:

```bash
git grep -Il $'\r' -- napariTFM/widgets/_widget.py tests/test_workflow_shell.py   # expect no output
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Persist experiments list + active selection in config state

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

**Slice 5 done** — the experiments list is the live top-of-panel substrate, persisted, with mini-rails that track stage enable/disable and a selection that re-labels the pipeline.

---

# SLICE 6 — Run-all walks the rail (live)

**Goal:** "Run all" sits below the rail, iterates the experiments list through the *enabled* stages via the existing `BatchAnalysis` backend, and updates each row's mini-rail live. Retire the bottom `BatchAnalysisWidget` folder list now that the experiments list is the source of folders.

**Dependency:** Slice 5 landed (`experiments_list.experiments()` is the folder source; `disabled_stages` is the active-stage filter).

### Task 6.1: Run-all bar widget + `run_all_experiments()` orchestration
- **Files:** modify `_widget.py`; reuse `batch_analysis.py:BatchAnalysis.process_all_folders()`.
- Add a `runall` bar (button "Run all N experiments" + hint label) below the stage sections.
- `run_all_experiments()` builds the batch config from `experiments_list.experiments()` + `self._disabled_stages()` (skip disabled stages), then runs `BatchAnalysis(config).process_all_folders()`.
- **Test:** stub `BatchAnalysis` with a recorder; assert it receives exactly the listed folders and that disabled stages are absent from the config.

### Task 6.2: Live mini-rail updates during a run
- **Files:** modify `_widget.py`, `batch_analysis.py` (emit a per-folder/per-stage progress callback).
- Add a progress hook `on_stage(folder, stage, status)` that calls `experiments_list._rows[i].set_stage_statuses(...)` (or a public `set_status(path, stage, status)` added to `ExperimentsList`).
- Refine `_experiment_stage_status` to read **which `.ntfm` fields** exist per stage (replace the coarse "all done" with field-level truth), reusing the `ntfm.read_ntfm` / `tidy_to_arrays` path from `batch_analysis.py:398-422`.
- **Test:** drive the progress hook directly; assert the addressed row's dot changes to running→done.

### Task 6.3: Retire the bottom batch folder list
- **Files:** modify `_widget.py` (drop the `"batch"` StageSection or repoint it), `batch_analysis_widget.py`.
- Remove the now-duplicated `folder_list` UI; keep `BatchAnalysis` backend.
- **Test:** assert there is no second folder-picker widget; the only experiment source is `experiments_list`.

---

# SLICE 7 — Aggregate → .iris (ROADMAP §5 backend)

**Goal:** Fold every experiment's `.ntfm` into one `.iris` file, grouped by condition/replicate/position labels parsed from folder structure. A footer widget surfaces it.

**⚠ Needs a short spec first:** the `.iris` schema is net-new (this is the first `.iris` work). Before Task 7.1, run a brief brainstorming pass to pin: (a) container format (reuse the tidy-dataframe approach from `.ntfm`? one long tidy table with `condition`/`replicate`/`position` columns?), (b) what aggregates (per-condition means? raw concat?), (c) label parsing rule (e.g. `<root>/<condition>/<position>` → condition, position). Capture it as `docs/superpowers/specs/<date>-iris-aggregator.md`, then resume.

### Task 7.1: `aggregate_to_iris()` backend
- **Files:** create `napariTFM/backend/aggregate.py`; create `tests/test_aggregate.py`.
- `aggregate_to_iris(experiment_folders: list[Path], out_path: Path, labels: dict[str, dict]) -> Path`: read each `<folder>/TFM_data/<name>.ntfm` via `ntfm.read_ntfm`, attach condition/replicate/position columns, concat tidy frames, write `.iris`.
- **Test (TDD):** build two tiny synthetic `.ntfm` files in a tmp dir, aggregate, assert the `.iris` has both experiments' rows with correct label columns.

### Task 7.2: Aggregate footer widget
- **Files:** create the footer in `_experiments_list.py` or a sibling; modify `_widget.py` to mount it below the run-all bar.
- Footer: "Aggregate → .iris" title, status line (`N done · M pending · groups by condition`), and an "Aggregate" button that calls `aggregate_to_iris(...)`.
- Status counts derive from `_experiment_stage_status` (how many experiments have a `.ntfm`).
- **Test:** with K experiments done (stubbed), assert the footer status reads "K done"; clicking calls the backend with the listed folders.

### Task 7.3: Label entry (condition/replicate/position)
- **Files:** the footer/aggregator widget.
- Provide the label-editing UI **here only** (not in the batch/tune UI), defaulting to a parse of the folder path; let the user override per experiment before aggregating.
- **Test:** default labels parse from path; an override is passed through to `aggregate_to_iris`.

---

## Self-Review

- **Spec coverage:** experiments-at-top (Slice 5 T2–T6) ✓; per-stage on/off already shipped (Slice 4) and now reflected in mini-rails (T5) ✓; batch-as-auto-run (Slice 6) ✓; aggregate→.iris (Slice 7) ✓; theme-following surfaces reuse existing `stage_accent`/`section_label_style` ✓.
- **Placeholder scan:** Slice 5 is fully concrete (code in every step). Slices 6–7 are task-level outlines by design — Slice 6's exact progress-hook signature and Slice 7's `.iris` schema depend on Slice 5 landing and a dedicated spec, flagged explicitly. Do not execute 6–7 from this document without expanding each into full TDD steps first.
- **Type consistency:** `PIPELINE_STAGES`, `MiniRail.appearance`, `ExperimentRow.set_stage_statuses`, `overall_status`, `ExperimentsList.{experiments,active,set_active,set_experiments,add_folders,refresh_statuses}`, and `_experiment_stage_status` names are used identically across Slice 5 tasks and referenced consistently by Slice 6.
- **Reuse:** dots reuse `_stage_spine._node_style`; accents reuse `stage_accent`; the batch run reuses `BatchAnalysis.process_all_folders`; the `.ntfm` read reuses `ntfm.read_ntfm`/`tidy_to_arrays`.

---

## Execution Handoff

Slice 5 is ready to build task-by-task. Slices 6–7 are scoped outlines to expand into full TDD plans when their turn comes (Slice 7 after the `.iris` spec).
