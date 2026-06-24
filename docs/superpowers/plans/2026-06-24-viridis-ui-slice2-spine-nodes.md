# Viridis UI Redesign — Slice 2: Gradient Spine + Status Nodes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give each workflow stage a left-gutter **spine** (a vertical gradient line that blends between neighbouring stage accents) and a **status node** (filled/glowing when done, hollow ring when ready, amber when running, dim when not started) — so the pipeline reads as one connected colormap and status is visible at a glance again, without expanding anything.

**Architecture:** A new `StageSpine(QWidget)` paints a 2px vertical gradient line (own accent blended with the stage above/below) plus a node circle at the header row whose fill/ring encodes status. `StageSection` gains this gutter as its leftmost column (its existing vertical body is nested unchanged into the right column). The shell wires each section's neighbour accents in pipeline order and sets the stage-stack spacing to 0 so segments connect. The node's hollow centre is painted with the widget's palette window colour, so it follows napari's host theme for free.

**Tech Stack:** Python, qtpy (PyQt6) `QPainter`/`QLinearGradient`/`QPen`, pytest with `QApplication` fixtures.

**Line endings (`feedback-line-endings`):** `_stage_section.py` / `_widget.py` may carry mixed CRLF/LF — touch only target lines, never normalize. After staging verify `git diff --cached --stat` == `git diff --cached -w --stat`.

**Known flake:** `tests/test_napari_compatibility.py::...pyqt6_qtpy_backend` — verify in isolation, don't chase.

---

## File Structure

- **Create** `napariTFM/widgets/_stage_spine.py` — the `StageSpine` gutter widget + the pure `_node_style()` helper.
- **Create** `tests/test_stage_spine.py` — unit tests for node-state mapping, status updates, accent storage.
- **Modify** `napariTFM/widgets/_stage_section.py` — nest the body in a right column, add the `StageSpine` gutter; forward `set_status` and a new `set_accents()` to it; update `set_accent` to re-accent the spine.
- **Modify** `napariTFM/widgets/_widget.py` — pass neighbour accents to each section at build, re-apply on theme change, and set the container spacing to 0.
- **Modify** `tests/test_workflow_shell.py` — add one assertion that sections receive ordered neighbour accents.

---

## Task 1: The `StageSpine` widget

**Files:**
- Create: `napariTFM/widgets/_stage_spine.py`
- Test: `tests/test_stage_spine.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage_spine.py`:

```python
import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._stage_spine import StageSpine, _node_style


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_node_style_done_is_filled_with_accent():
    fill, ring = _node_style("done", "#2a788e")
    assert fill is not None
    assert fill.name() == "#2a788e"
    assert ring.name() == "#2a788e"


def test_node_style_ready_is_hollow_ring():
    fill, ring = _node_style("ready", "#2a788e")
    assert fill is None
    assert ring.name() == "#2a788e"


def test_node_style_running_is_amber():
    fill, ring = _node_style("running", "#2a788e")
    assert fill is not None and ring.name() == "#e3b341"


def test_node_style_not_started_is_dim_hollow():
    fill, ring = _node_style("not_started", "#2a788e")
    assert fill is None
    assert ring.name() != "#2a788e"


def test_spine_set_status_updates_state(app):
    spine = StageSpine("#2a788e")
    spine.set_status("done")
    assert spine._status == "done"


def test_spine_set_accents_stores_neighbours(app):
    spine = StageSpine("#2a788e")
    spine.set_accents("#2a788e", above="#414487", below="#22a884")
    assert spine._accent_above == "#414487"
    assert spine._accent_below == "#22a884"


def test_spine_has_fixed_gutter_width(app):
    spine = StageSpine("#2a788e")
    assert spine.width() == StageSpine.GUTTER_WIDTH
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_stage_spine.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `StageSpine`**

Create `napariTFM/widgets/_stage_spine.py`:

```python
"""Left-gutter spine + status node for a workflow stage (the colormap rail)."""
from __future__ import annotations

from typing import Optional, Tuple

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

# status -> node appearance; muted grey for inert, amber for active.
_RUNNING = "#e3b341"
_ERROR = "#d62828"
_DIM = "#6b7484"


def _node_style(status: str, accent: str) -> Tuple[Optional[QColor], QColor]:
    """Return (fill, ring) for a node; fill None means a hollow ring."""
    if status == "done":
        return QColor(accent), QColor(accent)
    if status == "running":
        return QColor(_RUNNING), QColor(_RUNNING)
    if status == "ready":
        return None, QColor(accent)
    if status == "error":
        return QColor(_ERROR), QColor(_ERROR)
    return None, QColor(_DIM)


class StageSpine(QWidget):
    """A vertical gradient line + a status node, sized to its stage's height."""

    GUTTER_WIDTH = 28
    NODE_Y = 20      # node centre from the top, aligned to the header row
    NODE_R = 6
    LINE_W = 2

    def __init__(self, accent: str, status: str = "not_started", parent=None):
        super().__init__(parent)
        self._accent = accent
        self._accent_above = accent
        self._accent_below = accent
        self._status = status
        self.setFixedWidth(self.GUTTER_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def set_status(self, status: str) -> None:
        self._status = status
        self.update()

    def set_accents(self, accent: str, above: Optional[str] = None, below: Optional[str] = None) -> None:
        self._accent = accent
        self._accent_above = above or accent
        self._accent_below = below or accent
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = self.width() / 2.0
        h = self.height()

        gradient = QLinearGradient(0.0, 0.0, 0.0, float(h))
        gradient.setColorAt(0.0, QColor(self._accent_above))
        gradient.setColorAt(0.5, QColor(self._accent))
        gradient.setColorAt(1.0, QColor(self._accent_below))
        pen = QPen(QBrush(gradient), self.LINE_W)
        pen.setCapStyle(Qt.FlatCap)
        painter.setPen(pen)
        painter.drawLine(int(cx), 0, int(cx), int(h))

        fill, ring = _node_style(self._status, self._accent)
        centre = fill if fill is not None else self.palette().color(self.backgroundRole())
        painter.setPen(QPen(ring, 2))
        painter.setBrush(QBrush(centre))
        r = self.NODE_R
        painter.drawEllipse(QRectF(cx - r, self.NODE_Y - r, 2 * r, 2 * r))
        painter.end()
```

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_stage_spine.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_stage_spine.py tests/test_stage_spine.py
git commit -m "Add StageSpine gutter widget (gradient line + status node)"
```

---

## Task 2: Mount the spine in `StageSection`

**Files:**
- Modify: `napariTFM/widgets/_stage_section.py`
- Test: `tests/test_stage_spine.py` (add an integration test)

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_stage_spine.py`:

```python
def test_stage_section_owns_a_spine_and_forwards_status(app):
    from napariTFM.widgets._stage_section import StageSection
    from qtpy.QtWidgets import QLabel
    section = StageSection("Force", QLabel("body"), status="ready")
    assert isinstance(section.spine, StageSpine)
    assert section.spine._status == "ready"
    section.set_status("done")
    assert section.spine._status == "done"


def test_stage_section_set_accents_forwards_to_spine(app):
    from napariTFM.widgets._stage_section import StageSection
    from qtpy.QtWidgets import QLabel
    section = StageSection("Force", QLabel("body"))
    section.set_accents("#22a884", above="#2a788e", below="#7ad151")
    assert section.spine._accent_above == "#2a788e"
    assert section.spine._accent_below == "#7ad151"
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_stage_spine.py -q`
Expected: FAIL — `StageSection` has no `spine`.

- [ ] **Step 3: Add the gutter to `StageSection`**

In `napariTFM/widgets/_stage_section.py`:

Add the import near the existing `_collapsible_section` import (line ~13):

```python
from napariTFM.widgets._stage_spine import StageSpine
```

Add `QHBoxLayout` to the qtpy.QtWidgets import (line 4) — it already imports `QHBoxLayout`, so no change needed. Confirm it's present.

In `__init__`, the current top-level is a `QVBoxLayout` set on `self` (lines ~45-48):

```python
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)
```

Replace those four lines with an outer HBox holding the spine gutter and a body widget that owns the original vertical `layout`:

```python
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
```

Everything after this point that calls `layout.addLayout(...)` / `layout.addWidget(...)` is unchanged — it now fills the body column.

In `set_status` (after it sets `self._status` and updates the run glyph, around line 155), forward to the spine. Change the end of `set_status` from:

```python
        self._refresh_action_states()
```

to:

```python
        self.spine.set_status(status)
        self._refresh_action_states()
```

Add a `set_accents` method and have `set_accent` re-accent the spine. Locate `set_accent` (line ~163) and, at its end (after the `_status_section` block), add the spine re-accent:

```python
        self.spine.set_accents(accent, self._accent_above, self._accent_below)
```

Then add two instance attributes in `__init__` where `self._accent` is assigned (lines ~40-43). After that block, add:

```python
        self._accent_above = self._accent
        self._accent_below = self._accent
```

And add the new method just after `set_accent`:

```python
    def set_accents(self, accent: str, above: str | None = None, below: str | None = None) -> None:
        """Set the stage accent plus its neighbours, for the gradient spine."""
        self._accent = accent
        self._accent_above = above or accent
        self._accent_below = below or accent
        self.set_accent(accent)
```

Note: `set_accent` already restyles header/buttons/sections and now also calls `self.spine.set_accents(accent, self._accent_above, self._accent_below)` — so `set_accents` updating the neighbour fields *then* calling `set_accent` propagates everything in one path.

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_stage_spine.py tests/test_stage_section_header.py tests/test_stage_section_nesting.py tests/test_stage_section_action_sync.py -q`
Expected: PASS (all existing StageSection tests still green).

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_stage_section.py tests/test_stage_spine.py
git commit -m "Mount StageSpine gutter in StageSection; forward status + accents"
```

---

## Task 3: Wire neighbour accents in the shell

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing test**

The shell builds `self._stage_sections` (an ordered list) and adds them to `container_layout`. Add to `tests/test_workflow_shell.py` (use the existing real-widget construction pattern in that file; if the file builds the shell via `_make_widget()`/fixture, reuse it):

```python
def test_stage_sections_receive_ordered_neighbour_accents(app):
    from napariTFM.widgets import _ui_style
    widget = _make_widget()  # reuse this file's existing shell constructor/fixture
    sections = widget._stage_sections
    # first stage's "above" is its own accent; each section's "below" equals the
    # next section's accent (a continuous ramp down the rail).
    for i, sec in enumerate(sections[:-1]):
        assert sec._accent_below == sections[i + 1]._accent
```

If this file has no `_make_widget()` helper, construct the shell the same way the other real-widget tests in the file do and adapt the variable name.

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_workflow_shell.py::test_stage_sections_receive_ordered_neighbour_accents -q`
Expected: FAIL — neighbours default to each section's own accent.

- [ ] **Step 3: Wire neighbours + collapse stage spacing**

In `napariTFM/widgets/_widget.py`, find where the stage sections are added (lines ~525-529):

```python
        self._stage_sections = list(self._stage_sections_by_key.values())

        for section in self._stage_sections:
            container_layout.addWidget(section)
        container_layout.addStretch()
```

Replace with neighbour wiring + zero spacing so the spine segments connect:

```python
        self._stage_sections = list(self._stage_sections_by_key.values())
        self._apply_spine_neighbours()

        container_layout.setSpacing(0)
        for section in self._stage_sections:
            container_layout.addWidget(section)
        container_layout.addStretch()
```

Add the helper method to the shell class (near `refresh_stage_statuses`):

```python
    def _apply_spine_neighbours(self):
        """Give each stage's spine its neighbours' accents so the rail blends."""
        sections = self._stage_sections
        for i, section in enumerate(sections):
            above = sections[i - 1]._accent if i > 0 else section._accent
            below = sections[i + 1]._accent if i < len(sections) - 1 else section._accent
            section.set_accents(section._accent, above=above, below=below)
```

Finally, re-apply neighbours after a theme change so the gradient endpoints follow the new palette. In `_on_theme_selected` (line ~569), after it re-accents the sections, add a call to `self._apply_spine_neighbours()`. Locate the loop that calls `section.set_accent(...)` inside `_on_theme_selected` and add, immediately after that loop:

```python
        self._apply_spine_neighbours()
```

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: all pass except the known napari-compat flake.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Wire ordered neighbour accents into stage spines; connect the rail"
```

---

## Self-Review

- **Spec coverage:** spine gradient (Task 1 paint + Task 3 neighbours), status node (Task 1 `_node_style` + Task 2 forwarding), theme-following node centre (Task 1 `palette().color` — the deferred Slice-1 "surfaces follow theme" lands here for the node), connected rail (Task 3 spacing 0). Pulse animation for `running` is intentionally omitted (static amber) to avoid timers in tests — a later polish.
- **Placeholder scan:** the only soft reference is `_make_widget()` in Task 3's test — explicitly flagged to reuse the file's existing shell constructor; the executor confirms the real name when writing the test.
- **Type consistency:** `StageSpine(accent, status, parent)`, `set_status(str)`, `set_accents(accent, above, below)`, `_node_style(status, accent) -> (Optional[QColor], QColor)`, and section attrs `_accent`/`_accent_above`/`_accent_below`/`spine` are used consistently across tasks.
