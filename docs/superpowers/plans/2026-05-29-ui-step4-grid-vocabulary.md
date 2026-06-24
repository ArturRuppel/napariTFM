# UI Step 4 — Grid Layout Vocabulary + Param Panel Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port CellFlow's `section_grid` layout vocabulary subset into napariTFM's `_ui_style`, then rebuild `WorkflowParameterPanel` and the Project px/dt controls on it so one control idiom rules and parameter rows align uniformly.

**Architecture:** Add a small, self-contained subset of CellFlow's grid family — `section_grid()` (4-column label/field/label/field grid with stretchy field columns), `add_section_header`, `add_section_full_row`, `add_section_pair_row`, plus the private `_add_section_pair_cell`/`_block_label` helpers they lean on — to `_ui_style.py`. Then rebuild the two ad-hoc layouts (`WorkflowParameterPanel._setup_ui` and `ProjectSection._GeneralBody`) on this grid: **two label/field pairs per row**, **flat** (no `QGroupBox`; a bold `section_label_style` header instead). The dense block/sweep/button-row family is intentionally NOT ported (YAGNI — no sweep panel in scope).

**Tech Stack:** Python, qtpy (PyQt6 backend), `QGridLayout`, superqt labeled sliders, pytest.

---

## Constraints (read before touching any file)

- **Mixed line endings:** These files have mixed CRLF/LF. Touch ONLY the lines you intend to change; never normalize a whole file. After staging each commit, verify the change is content-only:
  ```bash
  git diff --cached --stat
  git diff --cached -w --stat
  ```
  The two outputs MUST report the same insertion/deletion counts. If `-w` shows fewer changes, you churned whitespace/line-endings — reset and redo touching only target lines.
- **Commit messages:** NO `Co-Authored-By` trailer. Plain subject + optional body.
- **Branch:** Stay on local `master`. Do NOT push (the owner pushes explicitly).
- **Qt-widget tests:** Any test that constructs a Qt widget MUST take the `app` fixture parameter (holds the `QApplication`; a discarded bare `QApplication([])` segfaults). Use the established fixture:
  ```python
  @pytest.fixture
  def app():
      return QApplication.instance() or QApplication([])
  ```
- **Run the whole suite** after the last task: `pytest -q`. Known flake (verify in isolation, don't chase): `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend`.

---

## File Structure

- `napariTFM/widgets/_ui_style.py` — **MODIFY**: add grid constants + the `section_grid` helper family. New imports: `QGridLayout`, `QLabel`, `QVBoxLayout`.
- `tests/test_ui_style.py` — **MODIFY**: add an `app` fixture + tests for the grid helpers.
- `napariTFM/widgets/_widget.py` — **MODIFY**: rebuild `WorkflowParameterPanel._setup_ui` on `section_grid` (two pairs per row, flat). New imports from `_ui_style`.
- `tests/test_preprocessing_ui_redesign.py` — **MODIFY**: add tests asserting the panel is grid-based and GroupBox-free.
- `napariTFM/widgets/_project_section.py` — **MODIFY**: rebuild `_GeneralBody`'s px/dt controls on `section_grid`. New imports from `_ui_style`.
- `tests/test_project_section.py` — **MODIFY**: add a test asserting the px/dt controls live on a grid and are GroupBox-free.

---

## Task 1: Port the `section_grid` vocabulary subset into `_ui_style.py`

**Files:**
- Modify: `napariTFM/widgets/_ui_style.py`
- Test: `tests/test_ui_style.py`

The ported helpers are a verbatim subset of CellFlow's `src/cellflow/napari/ui_style.py` (lines 428–480). `add_section_pair_row` deliberately lets fields keep their natural size policy (no fixed-width wrap) so sliders/combos stretch into the stretchy field columns.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_style.py`. Add the `app` fixture at the top of the file (after the imports, before the first test) and the four tests at the end:

```python
import pytest
from qtpy.QtWidgets import QApplication, QGridLayout, QLabel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_section_grid_has_four_columns_with_stretchy_fields(app):
    from napariTFM.widgets._ui_style import section_grid

    grid = section_grid()
    assert isinstance(grid, QGridLayout)
    assert grid.columnStretch(0) == 0
    assert grid.columnStretch(1) == 1
    assert grid.columnStretch(2) == 0
    assert grid.columnStretch(3) == 1


def test_add_section_pair_row_places_both_pairs(app):
    from napariTFM.widgets._ui_style import section_grid, add_section_pair_row

    grid = section_grid()
    add_section_pair_row(grid, 0, "Left", QLabel("L"), "Right", QLabel("R"))

    assert grid.itemAtPosition(0, 0) is not None
    assert grid.itemAtPosition(0, 2) is not None


def test_add_section_pair_row_left_only_leaves_right_empty(app):
    from napariTFM.widgets._ui_style import section_grid, add_section_pair_row

    grid = section_grid()
    add_section_pair_row(grid, 0, "Left", QLabel("L"))

    assert grid.itemAtPosition(0, 0) is not None
    assert grid.itemAtPosition(0, 2) is None


def test_add_section_header_spans_all_four_columns(app):
    from napariTFM.widgets._ui_style import section_grid, add_section_header

    grid = section_grid()
    header = add_section_header(grid, 0, QLabel("Title"))

    assert header is not None
    # spanning item occupies column 0 and the spanned columns report the same item
    assert grid.itemAtPosition(0, 0) is grid.itemAtPosition(0, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui_style.py -q`
Expected: FAIL with `ImportError: cannot import name 'section_grid'` (and the other helpers).

- [ ] **Step 3: Add the imports and helpers to `_ui_style.py`**

In `napariTFM/widgets/_ui_style.py`, extend the existing `qtpy.QtWidgets` import. Change:

```python
from qtpy.QtWidgets import QStyle, QToolButton, QWidget
```

to:

```python
from qtpy.QtWidgets import QGridLayout, QLabel, QStyle, QToolButton, QVBoxLayout, QWidget
```

Add these two constants alongside the existing layout constants (after the `TIGHT_SPACING = 4` line):

```python
DEFAULT_FIELD_SPACING = 8
DEFAULT_ROW_SPACING = 4
```

Append the helper family at the end of the file:

```python
def section_grid() -> QGridLayout:
    """A 4-column grid (label, field, label, field) where field columns
    stretch — so sliders, combos, and labels fill the available width and
    label columns stay aligned across all sections that share the grid."""
    layout = QGridLayout()
    layout.setHorizontalSpacing(DEFAULT_FIELD_SPACING)
    layout.setVerticalSpacing(DEFAULT_ROW_SPACING)
    layout.setColumnStretch(0, 0)
    layout.setColumnStretch(1, 1)
    layout.setColumnStretch(2, 0)
    layout.setColumnStretch(3, 1)
    return layout


def add_section_header(grid, row, widget):
    """Add a heading widget spanning all 4 columns of a section_grid."""
    grid.addWidget(widget, row, 0, 1, 4)
    return widget


def add_section_full_row(grid, row, widget):
    """Add a widget (separator, button row, …) spanning all 4 columns."""
    grid.addWidget(widget, row, 0, 1, 4)
    return widget


def add_section_pair_row(grid, row, left_label, left_widget, right_label=None, right_widget=None):
    """Add a row with up to two [label][widget] pairs. Widgets keep their
    natural size policy (no fixed-width wrap) so sliders/combos can stretch."""
    left_label_widget = _block_label(left_label)
    _add_section_pair_cell(grid, row, 0, left_label_widget, left_widget)

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        _add_section_pair_cell(grid, row, 2, right_label_widget, right_widget)
    return left_label_widget, left_widget, right_label_widget, right_widget


def _add_section_pair_cell(grid, row, column, label_widget, widget):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
    layout.addWidget(label_widget)
    layout.addWidget(widget)
    grid.addWidget(container, row, column, 1, 2)
    return container


def _block_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui_style.py -q`
Expected: PASS (all existing tests + the 4 new ones).

- [ ] **Step 5: CRLF audit + commit**

```bash
git add napariTFM/widgets/_ui_style.py tests/test_ui_style.py
git diff --cached --stat
git diff --cached -w --stat   # MUST match the line above
git commit -m "Port section_grid layout vocabulary into _ui_style"
```

---

## Task 2: Rebuild `WorkflowParameterPanel` on `section_grid` (two pairs per row, flat)

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (the `WorkflowParameterPanel._setup_ui` method, around `_widget.py:258-279`)
- Test: `tests/test_preprocessing_ui_redesign.py`

The panel currently builds one `QGroupBox` + `QFormLayout` per section. Replace with: one `section_grid` per displayed section, a bold header row, then params packed **two pairs per row**. `_create_control` (the control factory + `parameter_controls` registration) is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_preprocessing_ui_redesign.py`. (The file already has an `app` fixture — reuse it; do NOT add a second one.)

```python
def test_param_panel_uses_section_grid_not_groupbox(app):
    from qtpy.QtWidgets import QGridLayout, QGroupBox
    from napariTFM.utilities.parameter_manager import ParameterManager
    from napariTFM.widgets._widget import WorkflowParameterPanel

    panel = WorkflowParameterPanel(ParameterManager(), section_titles=("Displacement",))

    assert panel.findChildren(QGroupBox) == []
    assert panel.findChild(QGridLayout) is not None


def test_param_panel_packs_two_pairs_per_row(app):
    from qtpy.QtWidgets import QGridLayout
    from napariTFM.utilities.parameter_manager import ParameterManager
    from napariTFM.widgets._widget import WorkflowParameterPanel

    # Displacement has 7 params; row 0 is the header, row 1 holds the first two.
    panel = WorkflowParameterPanel(ParameterManager(), section_titles=("Displacement",))
    grid = panel.findChild(QGridLayout)

    assert grid.itemAtPosition(1, 0) is not None
    assert grid.itemAtPosition(1, 2) is not None


def test_param_panel_still_registers_controls(app):
    from napariTFM.utilities.parameter_manager import ParameterManager
    from napariTFM.widgets._widget import WorkflowParameterPanel

    panel = WorkflowParameterPanel(ParameterManager(), section_titles=("Displacement",))

    assert "nscales" in panel.parameter_controls
    assert "d_max" in panel.parameter_controls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocessing_ui_redesign.py -q -k "section_grid or two_pairs or registers_controls"`
Expected: FAIL — `test_param_panel_uses_section_grid_not_groupbox` fails because `QGroupBox` children still exist / no `QGridLayout`.

- [ ] **Step 3: Add the `_ui_style` imports**

In `napariTFM/widgets/_widget.py`, extend the existing `_ui_style` import (currently `_widget.py:24`). Change:

```python
from napariTFM.widgets._ui_style import title_style, stage_accent, theme_names, active_theme_name, set_active_theme
```

to:

```python
from napariTFM.widgets._ui_style import title_style, stage_accent, theme_names, active_theme_name, set_active_theme, section_grid, add_section_header, add_section_pair_row, section_label_style, TIGHT_SPACING
```

- [ ] **Step 4: Replace `_setup_ui`**

Replace the entire `_setup_ui` method (`_widget.py:258-279`) with:

```python
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(TIGHT_SPACING)

        for title, specs in self.PARAMETER_SECTIONS:
            if self._section_titles is not None and title not in self._section_titles:
                continue

            grid = section_grid()
            header = QLabel(title)
            header.setStyleSheet(section_label_style())
            add_section_header(grid, 0, header)

            row = 1
            index = 0
            while index < len(specs):
                left_label, left_control = self._control_for_spec(specs[index])
                if index + 1 < len(specs):
                    right_label, right_control = self._control_for_spec(specs[index + 1])
                    add_section_pair_row(grid, row, left_label, left_control, right_label, right_control)
                else:
                    add_section_pair_row(grid, row, left_label, left_control)
                row += 1
                index += 2

            layout.addLayout(grid)

        self.setLayout(layout)

    def _control_for_spec(self, spec):
        name, label, kind, min_val, max_val, step, decimals, choices = spec
        control = self._create_control(name, kind, min_val, max_val, step, decimals, choices)
        return label, control
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_preprocessing_ui_redesign.py -q`
Expected: PASS (the 3 new tests + all pre-existing ones — confirm none of the older panel tests regressed).

- [ ] **Step 6: CRLF audit + commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_preprocessing_ui_redesign.py
git diff --cached --stat
git diff --cached -w --stat   # MUST match
git commit -m "Rebuild WorkflowParameterPanel on section_grid (two-up, flat)"
```

---

## Task 3: Rebuild the Project px/dt controls on `section_grid`

**Files:**
- Modify: `napariTFM/widgets/_project_section.py` (the `_GeneralBody.__init__` layout, `_project_section.py:34-64`)
- Test: `tests/test_project_section.py`

Put the two calibration spinboxes (`pixel_size`, `frame_interval`) on a `section_grid` as a single two-pair row, with the output-dir control and the two button rows added as full-width rows beneath. The save/load/reset/clear buttons and their wiring are unchanged — only their containing layout moves onto the grid.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_section.py` (reuse the existing `app` and `_StubParameterManager`):

```python
def test_general_body_uses_section_grid_not_groupbox(app):
    from qtpy.QtWidgets import QGridLayout, QGroupBox

    section = ProjectSection(_StubParameterManager())

    assert section.body.findChildren(QGroupBox) == []
    grid = section.body.findChild(QGridLayout)
    assert grid is not None
    # pixel_size (col 0) and frame_interval (col 2) share the first row
    assert grid.itemAtPosition(0, 0) is not None
    assert grid.itemAtPosition(0, 2) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_section.py -q -k general_body_uses_section_grid`
Expected: FAIL — `section.body.findChild(QGridLayout)` is `None` (body still uses `QVBoxLayout`/`QHBoxLayout`).

- [ ] **Step 3: Add the `_ui_style` imports**

In `napariTFM/widgets/_project_section.py`, extend the existing `_ui_style` import (`_project_section.py:16`). Change:

```python
from napariTFM.widgets._ui_style import danger_text_style
```

to:

```python
from napariTFM.widgets._ui_style import danger_text_style, section_grid, add_section_pair_row, add_section_full_row
```

- [ ] **Step 4: Rebuild the `_GeneralBody.__init__` layout**

Replace the layout-building block (`_project_section.py:34-64`) — from `layout = QVBoxLayout()` through the `layout.addLayout(output_row)` line — with the grid-based build. The result:

```python
        grid = section_grid()
        grid.setContentsMargins(8, 8, 8, 8)
        self.setLayout(grid)

        controls = []
        for name, label, min_val, max_val, step, decimals in _GENERAL_SPECS:
            control = QDoubleSpinBox()
            control.setRange(min_val, max_val)
            control.setSingleStep(step)
            control.setDecimals(decimals)
            control.setObjectName(f"workflow_parameter_{name}")
            control.setValue(parameter_manager.get_ui_parameter(name))
            control.valueChanged.connect(
                lambda value, n=name: parameter_manager.set_ui_parameter(n, value)
            )
            self.parameter_controls[name] = control
            controls.append((label, control))

        add_section_pair_row(
            grid, 0,
            controls[0][0], controls[0][1],
            controls[1][0], controls[1][1],
        )

        self.output_dir_label = QLabel("No output directory")
        self.output_dir_label.setObjectName("project_output_dir_label")
        self.choose_output_dir_btn = QPushButton("Output Directory")
        self.choose_output_dir_btn.setObjectName("project_choose_output_dir_button")
        self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)

        output_container = QWidget()
        output_row = QHBoxLayout(output_container)
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.addWidget(self.choose_output_dir_btn)
        output_row.addWidget(self.output_dir_label, stretch=1)
        add_section_full_row(grid, 1, output_container)
```

Then the existing save/load/reset/clear button block (`_project_section.py:66-80`) must move onto the grid as two full rows. Replace it with:

```python
        self.save_params_btn = QPushButton("Save Parameters")
        self.load_params_btn = QPushButton("Load Parameters")
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setStyleSheet(danger_text_style())

        button_container1 = QWidget()
        button_row1 = QHBoxLayout(button_container1)
        button_row1.setContentsMargins(0, 0, 0, 0)
        button_row1.addWidget(self.save_params_btn)
        button_row1.addWidget(self.load_params_btn)
        add_section_full_row(grid, 2, button_container1)

        button_container2 = QWidget()
        button_row2 = QHBoxLayout(button_container2)
        button_row2.setContentsMargins(0, 0, 0, 0)
        button_row2.addWidget(self.reset_params_btn)
        button_row2.addWidget(self.clear_data_btn)
        add_section_full_row(grid, 3, button_container2)
```

**Note:** Remove the now-unused `QVBoxLayout` from the `qtpy.QtWidgets` import line (`_project_section.py:5-13`) ONLY if it is no longer referenced anywhere in the file — grep first (`grep -n QVBoxLayout napariTFM/widgets/_project_section.py`). If still used, leave the import untouched (avoid CRLF churn).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_project_section.py -q`
Expected: PASS — the new grid test plus all 7 pre-existing Project tests (controls present, write-through, output-dir sync, buttons, starts-expanded).

- [ ] **Step 6: CRLF audit + commit**

```bash
git add napariTFM/widgets/_project_section.py tests/test_project_section.py
git diff --cached --stat
git diff --cached -w --stat   # MUST match
git commit -m "Rebuild Project px/dt controls on section_grid"
```

---

## Final Verification

- [ ] **Full suite green**

Run: `pytest -q`
Expected: all pass. If `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` fails, re-run it in isolation (`pytest tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend -q`); a pass in isolation = known flake, not a regression.

- [ ] **CRLF audit across the slice**

```bash
git diff --stat <parent-of-task1>..HEAD
git diff -w --stat <parent-of-task1>..HEAD
```
The two MUST report identical counts.

- [ ] **No leftover ad-hoc param layout**

```bash
grep -nE "QGroupBox|QFormLayout" napariTFM/widgets/_widget.py napariTFM/widgets/_project_section.py
```
Expected: no `QGroupBox`/`QFormLayout` remaining in the rebuilt panels (the `QFormLayout`/`QGroupBox` imports in `_widget.py` may remain if used elsewhere — check before removing).

- [ ] **Manual smoke (owner-run, requires napari):** Launch napari, open each stage's ⚙ params panel. Confirm: params render as aligned two-up label/field pairs on a grid (no boxed groups), sliders stretch to fill, the Project section shows Pixel Size / Frame Length side-by-side with the output-dir + button rows beneath, and theme switching still re-accents correctly.
