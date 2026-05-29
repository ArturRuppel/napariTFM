# UI Slice 1 — Remove Inner Scroll Areas + Hardcoded Width

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four pipeline stage bodies reflow to the dock width by deleting the scroll-area-inside-a-scroll-area and the hardcoded `setFixedWidth(360)` each stage widget carries — letting the shell's single `QScrollArea` own layout, as in CellFlow.

**Architecture:** Each stage widget (`PreprocessingWidget`, `DisplacementAnalysisWidget`, `FTTCWidget`, `MSMWidget`) builds its own `QScrollArea` with `setFixedWidth(360)` in `_create_content_container`, nested inside `napariTFMWidget`'s outer scroll area. We strip the inner `QScrollArea` from each, return the content container directly, and drop the now-unused imports. No behavior changes — only the widget tree's nesting and width constraint. This is Step 1 of `TODO.md`; it is independent of the later section-primitive and param-layout work.

**Tech Stack:** Python, qtpy/PyQt, pytest (`QT_QPA_PLATFORM=offscreen`).

**Line endings:** Verify each commit with `git diff -w` matching `git diff` (these widget files have a history of CRLF/LF churn). Confirm no CR: `grep -rc $'\r' <file>` returns 0 for each touched file.

---

## File Structure

- `napariTFM/widgets/preprocessing_widget.py` — strip inner scroll from `_create_content_container`; remove `Qt` (now unused) and `QScrollArea` imports.
- `napariTFM/widgets/displacement_analysis_widget.py` — strip inner scroll; remove `QScrollArea` import.
- `napariTFM/widgets/fttc_widget.py` — strip inner scroll; remove `QScrollArea` import.
- `napariTFM/widgets/msm_widget.py` — strip inner scroll; remove `QScrollArea` import.
- `tests/test_preprocessing_ui_redesign.py` — repurpose `test_preprocessing_widget_keeps_parameter_content_in_scroll_area` into a `no-inner-scroll` lock (this is the only UI test that constructs a real stage widget and asserts on its scroll area).

**Testability note:** Only `test_preprocessing_ui_redesign.py` and `test_workflow_shell.py` construct real stage widgets, and only the preprocessing file has working Qt fakes (`_FakeViewer`, `_ParameterManager`, `_FakeVisualizationManager`). `test_workflow_shell.py` builds the shell with *stub* stage widgets, so it won't exercise the real ones. The `test_displacement_analysis.py` / `test_fttc_analysis.py` / `test_msm_analysis.py` files are **backend** tests — they don't touch Qt. Manufacturing UI fixtures for the other three widgets is out of proportion for a mechanical, identical refactor. So: Task 1 (preprocessing) gets a behavioral RED→GREEN lock; Tasks 2–4 are guarded by the full no-regression suite plus the proven-identical change shape. (Extracting the preprocessing fakes into a shared `tests/conftest.py` to enable per-widget UI tests is a reasonable follow-up, deferred out of this slice.)

---

# Task 1: Preprocessing — drop inner scroll + repurpose the scroll test

**Files:**
- Modify: `napariTFM/widgets/preprocessing_widget.py`
- Test: `tests/test_preprocessing_ui_redesign.py`

- [ ] **Step 1: Repurpose the existing scroll test into a no-inner-scroll lock**

In `tests/test_preprocessing_ui_redesign.py`, replace the whole `test_preprocessing_widget_keeps_parameter_content_in_scroll_area` function (currently around line 270) with:

```python
def test_preprocessing_widget_has_no_inner_scroll_area(app):
    widget = PreprocessingWidget(
        _FakeViewer(),
        DataManager(),
        _ParameterManager(),
        _FakeVisualizationManager(),
    )
    widget.resize(360, 220)
    widget.show()
    app.processEvents()

    # The stage widget no longer owns a scroll area or a fixed width; the
    # shell's single scroll area owns layout, so the body reflows to the dock.
    assert widget.findChild(QScrollArea) is None
    assert not hasattr(widget, "data_panel")
```

(`QScrollArea` is already imported at the top of this test file — line 7 — so the `is None` check needs no new import.)

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_preprocessing_ui_redesign.py::test_preprocessing_widget_has_no_inner_scroll_area" -v`
Expected: FAIL — `findChild(QScrollArea)` still returns the inner scroll area, so `assert ... is None` fails.

- [ ] **Step 3: Strip the inner scroll from `_create_content_container`**

In `napariTFM/widgets/preprocessing_widget.py`, replace the method `_create_content_container` (currently around line 389):

Old:
```python
    def _create_content_container(self) -> QWidget:
        """Create the main content container with scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        scroll.setFixedWidth(360)

        container = QWidget()
        layout = QVBoxLayout()

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_frame = self._create_preview_frame()
        self.preview_frame.setVisible(False)
        layout.addWidget(self.preview_frame)
        self.action_frame = self._create_action_frame()
        self.action_frame.setVisible(False)
        layout.addWidget(self.action_frame)
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll
```

New:
```python
    def _create_content_container(self) -> QWidget:
        """Create the main content container.

        The stage widget no longer owns a scroll area or a fixed width — the
        shell's single scroll area owns layout, so the body reflows to the dock
        width (CellFlow model).
        """
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_frame = self._create_preview_frame()
        self.preview_frame.setVisible(False)
        layout.addWidget(self.preview_frame)
        self.action_frame = self._create_action_frame()
        self.action_frame.setVisible(False)
        layout.addWidget(self.action_frame)
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        return container
```

- [ ] **Step 4: Remove the now-unused `QScrollArea` and `Qt` imports**

The deleted method held the file's only use of `QScrollArea` and its only use of `Qt.` (the `Qt.ScrollBarAlwaysOff` line).

- In the `qtpy.QtWidgets` import (line 9), change:
  ```python
      QFrame, QScrollArea, QCheckBox, QApplication,
  ```
  to:
  ```python
      QFrame, QCheckBox, QApplication,
  ```
- In the `qtpy.QtCore` import (line 7), change:
  ```python
  from qtpy.QtCore import Qt, Signal
  ```
  to:
  ```python
  from qtpy.QtCore import Signal
  ```

Verify both are truly unused first:
Run: `grep -nE "QScrollArea|Qt\." napariTFM/widgets/preprocessing_widget.py`
Expected: zero hits (after the Step 3 edit). If any remain, do not remove that import.

- [ ] **Step 5: Run to verify the lock passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest "tests/test_preprocessing_ui_redesign.py::test_preprocessing_widget_has_no_inner_scroll_area" -v`
Expected: PASS.

- [ ] **Step 6: Run the full preprocessing UI test file (no regression)**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_preprocessing_ui_redesign.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add napariTFM/widgets/preprocessing_widget.py tests/test_preprocessing_ui_redesign.py
git commit -m "Drop inner scroll/fixed-width from preprocessing stage body"
```

---

# Task 2: Displacement — drop inner scroll

**Files:**
- Modify: `napariTFM/widgets/displacement_analysis_widget.py`

- [ ] **Step 1: Remove the three scroll lines at the top of `_create_content_container`**

In `napariTFM/widgets/displacement_analysis_widget.py` (`_create_content_container`, around line 388), delete these three lines:

```python
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
```

(The method now begins directly at `container = QWidget()`.)

- [ ] **Step 2: Return the container instead of the scroll**

In the same method, change the closing lines:

Old:
```python
        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll
```
New:
```python
        container.setLayout(layout)
        return container
```

- [ ] **Step 3: Remove the now-unused `QScrollArea` import**

In the `qtpy.QtWidgets` import (line 6), change:
```python
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton, QMessageBox, QSpacerItem,
```
to:
```python
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QSpacerItem,
```

Verify: `grep -nE "QScrollArea" napariTFM/widgets/displacement_analysis_widget.py`
Expected: zero hits.

- [ ] **Step 4: Run the displacement + shell suites (no regression)**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_displacement_analysis.py tests/test_displacement_ownership.py tests/test_workflow_shell.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/displacement_analysis_widget.py
git commit -m "Drop inner scroll/fixed-width from displacement stage body"
```

---

# Task 3: FTTC (Force) — drop inner scroll

**Files:**
- Modify: `napariTFM/widgets/fttc_widget.py`

- [ ] **Step 1: Remove the three scroll lines at the top of `_create_content_container`**

In `napariTFM/widgets/fttc_widget.py` (`_create_content_container`, around line 379), delete these three lines:

```python
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
```

- [ ] **Step 2: Return the container instead of the scroll**

Change the closing lines:

Old:
```python
        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll
```
New:
```python
        container.setLayout(layout)
        return container
```

- [ ] **Step 3: Remove the now-unused `QScrollArea` import**

In the `qtpy.QtWidgets` import (line 6), change:
```python
from qtpy.QtWidgets import (QPushButton, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
```
to:
```python
from qtpy.QtWidgets import (QPushButton, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout,
```

Verify: `grep -nE "QScrollArea" napariTFM/widgets/fttc_widget.py`
Expected: zero hits.

- [ ] **Step 4: Run the force + shell suites (no regression)**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_fttc_analysis.py tests/test_force_ownership.py tests/test_workflow_shell.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/fttc_widget.py
git commit -m "Drop inner scroll/fixed-width from force stage body"
```

---

# Task 4: MSM (Stress) — drop inner scroll

**Files:**
- Modify: `napariTFM/widgets/msm_widget.py`

- [ ] **Step 1: Remove the three scroll lines at the top of `_create_content_container`**

In `napariTFM/widgets/msm_widget.py` (`_create_content_container`, around line 465), delete these three lines:

```python
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
```

- [ ] **Step 2: Return the container instead of the scroll**

Change the closing lines:

Old:
```python
        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll
```
New:
```python
        container.setLayout(layout)
        return container
```

- [ ] **Step 3: Remove the now-unused `QScrollArea` import**

In the `qtpy.QtWidgets` import (line 8), change:
```python
    QLabel, QSizePolicy, QFrame, QScrollArea, QApplication, QSpacerItem,
```
to:
```python
    QLabel, QSizePolicy, QFrame, QApplication, QSpacerItem,
```

Verify: `grep -nE "QScrollArea" napariTFM/widgets/msm_widget.py`
Expected: zero hits.

- [ ] **Step 4: Run the stress + shell suites (no regression)**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_msm_analysis.py tests/test_stress_ownership.py tests/test_workflow_shell.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/msm_widget.py
git commit -m "Drop inner scroll/fixed-width from stress stage body"
```

---

# Task 5: Verification — full suite, clean diff, manual smoke

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest`
Expected: all PASS. Known env flake: `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` may intermittently SIGSEGV in its spawned subprocess — re-run in isolation to confirm it is the flake, not a regression.

- [ ] **Step 2: Confirm no inner scroll areas remain in any stage body**

Run: `grep -rnE "setFixedWidth\(360\)|QScrollArea" napariTFM/widgets/preprocessing_widget.py napariTFM/widgets/displacement_analysis_widget.py napariTFM/widgets/fttc_widget.py napariTFM/widgets/msm_widget.py`
Expected: zero hits. (The shell's own outer `QScrollArea` lives in `_widget.py` and is intentionally untouched.)

- [ ] **Step 3: Confirm no line-ending churn**

Run `git diff -w --stat` for the slice's commit range and compare to `git diff --stat` (must match). Then:
Run: `grep -rc $'\r' napariTFM/widgets/preprocessing_widget.py napariTFM/widgets/displacement_analysis_widget.py napariTFM/widgets/fttc_widget.py napariTFM/widgets/msm_widget.py`
Expected: `0` for each file.

- [ ] **Step 4: Manual smoke (needs napari — owner runs)**

Launch napari, add the napariTFM widget, dock it, and:
1. Narrow and widen the dock — each stage body (Preprocessing, Displacement, Force, Stress) now reflows to the dock width instead of staying pinned at 360px.
2. Confirm there is only one vertical scrollbar (the shell's), not a nested inner scrollbar inside a stage.
3. Confirm the run / preview / cancel header buttons and the per-stage status/progress still work (no behavior change expected).

---

## Self-Review

**Spec coverage (TODO.md Step 1):** strip inner `QScrollArea` + `setFixedWidth(360)` from all four stage widgets — Tasks 1–4, one per widget. Update the affected test — Task 1 Step 1. Verify reflow + no regression — Task 5. ✔

**Placeholder scan:** none — every code step shows the exact old/new text and the exact command.

**Type/name consistency:** every task edits the same method name (`_create_content_container`) and the same closing-line pattern (`return container`). The repurposed test name (`test_preprocessing_widget_has_no_inner_scroll_area`) is used consistently in its run commands. `QScrollArea` import removal is verified by grep in each task.

**Deliberate boundaries:** the shell's outer `QScrollArea` in `_widget.py` is intentionally kept (it is the single scroll area the slice routes layout through). No new UI test fixtures are created for displacement/fttc/msm — flagged in the File Structure note as a deferred conftest follow-up; those three are guarded by the no-regression suite plus the identical, proven preprocessing change.
