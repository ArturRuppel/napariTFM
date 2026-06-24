# UI Slice 2+3 — Adopt CollapsibleSection + Retire Proxy Machinery (Keystone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace napariTFM's bespoke `StageSection` param-toggle and `_ActionStateSync`/`action_targets` event-filter proxies with CellFlow's `CollapsibleSection` primitive and a single-instance, signal-driven action model, so each control exists exactly once.

**Architecture:** Port CellFlow's `CollapsibleSection` (header-hideable, accent-inheriting, left-stripe params container) into napariTFM. `StageSection` keeps its always-visible header row (status dot + label + ⚙ params + ▶ run + preview) and body, but the parameter panel now lives inside a header-hidden `CollapsibleSection` driven by `params_btn`. The header action buttons are owned by `StageSection` and built once; the 4 inner stage widgets drop their duplicate action buttons, expose handler callables, and emit a parameterless `action_states_changed` signal whose paired `action_states()` accessor reports per-action enablement. This retires `_ActionStateSync` and `action_targets` entirely.

**Tech Stack:** Python 3.13, qtpy (PyQt backend), superqt sliders, pytest. This is `TODO.md` Steps 2+3, executed as one keystone slice.

**Scope decisions (locked):**
- **Section model:** Adopt CellFlow `CollapsibleSection` outright (not extend `StageSection`).
- **Control wiring:** Signal-based; the header owns the action buttons; inner widgets expose handlers + an `action_states_changed` signal.
- **Status dot** stays — it is a napariTFM feature, hosted in the header row (not part of `CollapsibleSection`).
- **Batch stage** has only a run action and no parameter panel; it follows the same API with an empty preview/params.

**Line-ending discipline (per repo memory `[[feedback_line_endings]]`):** Files carry mixed CRLF/LF. Touch only target lines; never normalize whitespace. After each commit verify `git diff --stat <parent>..HEAD` equals `git diff -w --stat <parent>..HEAD` and that no diff line carries a trailing CR (`git show HEAD | grep -nP '\r$'` returns nothing for newly added lines).

**Test harness note:** `tests/test_workflow_shell.py` stubs `DataManager`/`ParameterManager`/`VisualizationManager` in `sys.modules`; real-manager tests use `importlib.util.spec_from_file_location`. Any test creating a Qt widget MUST take the `app` fixture parameter (discarding a bare `QApplication([])` segfaults). Known flake: `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` — verify in isolation, don't chase.

---

## File Structure

- **Create** `napariTFM/widgets/_collapsible_section.py` — the ported `CollapsibleSection` primitive (one responsibility: a collapsible, accent-aware container).
- **Create** `tests/test_collapsible_section.py` — unit tests for the primitive.
- **Modify** `napariTFM/widgets/_ui_style.py` — add `muted_accent(hex)`, `SECTION_MARGIN`, `TINY_MARGIN`, `TIGHT_SPACING`.
- **Modify** `napariTFM/widgets/_stage_section.py` — host params in a `CollapsibleSection`; remove `_ActionStateSync`/`action_targets`; accept handler callables + an enablement signal; own the header buttons.
- **Modify** `napariTFM/widgets/preprocessing_widget.py`, `displacement_analysis_widget.py`, `fttc_widget.py`, `msm_widget.py` — drop duplicate action buttons, expose `action_states()` + `action_states_changed` + handler callables.
- **Modify** `napariTFM/widgets/_widget.py` — rewire the 5 stage constructions to the new `StageSection` API.
- **Modify/overwrite tests** — `tests/test_stage_section_header.py`, `tests/test_stage_section_action_sync.py`, `tests/test_stage_section_nesting.py`, `tests/test_workflow_shell.py`, and the 4 `*_ownership.py` tests where they assert proxy/`action_targets` behavior.

---

## Phase A — Port the primitive

### Task A1: Add `muted_accent` + layout constants to `_ui_style.py`

**Files:**
- Modify: `napariTFM/widgets/_ui_style.py`
- Test: `tests/test_ui_style.py`

`CollapsibleSection` needs a hex→hex `muted_accent` (the existing `muted_stage_accent` takes a *key*, not a hex), plus three margin/spacing constants CellFlow uses.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ui_style.py`:

```python
def test_muted_accent_desaturates_and_flattens():
    from napariTFM.widgets._ui_style import muted_accent

    out = muted_accent("#3b6fb6")
    assert out.startswith("#") and len(out) == 7
    # idempotent shape: feeding the output back stays a valid hex
    assert muted_accent(out).startswith("#")


def test_layout_constants_present():
    from napariTFM.widgets import _ui_style

    assert _ui_style.TINY_MARGIN == 2
    assert _ui_style.SECTION_MARGIN == 4
    assert _ui_style.TIGHT_SPACING == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_ui_style.py::test_muted_accent_desaturates_and_flattens tests/test_ui_style.py::test_layout_constants_present -v`
Expected: FAIL with `ImportError`/`AttributeError` (`muted_accent`, `TINY_MARGIN` not defined).

- [ ] **Step 3: Add the constants near the top of `_ui_style.py`**

After the existing `COMPACT_SPACING = 4` / `ICON_BUTTON_SIZE = 24` block (around line 7-8), add:

```python
TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
```

- [ ] **Step 4: Add `muted_accent` and refactor `muted_stage_accent` to use it**

`muted_stage_accent` currently inlines the colorsys math (lines 121-133). Extract the hex→hex core into `muted_accent` and have `muted_stage_accent` delegate. Replace the body of `muted_stage_accent` (lines 121-133) with:

```python
def muted_accent(hex_value: str) -> str:
    """Return a muted (low-saturation, midtone-lightness) variant of a hex color."""
    hex_value = hex_value.lstrip("#")
    r = int(hex_value[0:2], 16) / 255.0
    g = int(hex_value[2:4], 16) / 255.0
    b = int(hex_value[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s_muted = s * 0.35
    l_muted = 0.5 + (l - 0.5) * 0.6
    r_out, g_out, b_out = colorsys.hls_to_rgb(h, l_muted, s_muted)
    return "#{:02x}{:02x}{:02x}".format(
        round(r_out * 255), round(g_out * 255), round(b_out * 255)
    )


def muted_stage_accent(key: str) -> str:
    """Return a muted variant of a stage accent."""
    return muted_accent(stage_accent(key))
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_ui_style.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 6: Verify line endings**

Run: `git diff -w --stat napariTFM/widgets/_ui_style.py tests/test_ui_style.py` and confirm it matches `git diff --stat` for the same files.

- [ ] **Step 7: Commit**

```bash
git add napariTFM/widgets/_ui_style.py tests/test_ui_style.py
git commit -m "Add hex muted_accent + layout constants for CollapsibleSection port"
```

---

### Task A2: Port `CollapsibleSection` into napariTFM

**Files:**
- Create: `napariTFM/widgets/_collapsible_section.py`
- Test: `tests/test_collapsible_section.py`

Faithful port of CellFlow `widgets.py:32-232`, rewired to napariTFM's `_ui_style` (drop CellFlow-only imports `icon_button`, `muted_label`, `stage_header_action_button`, `status_label`, `tool_btn` — `CollapsibleSection` itself uses none of them).

- [ ] **Step 1: Write the failing test**

Create `tests/test_collapsible_section.py`:

```python
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from napariTFM.widgets._collapsible_section import CollapsibleSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_starts_collapsed_and_toggles(app):
    inner = QLabel("body")
    sec = CollapsibleSection("Params", inner, expanded=False)
    assert sec.is_expanded is False
    assert sec._content_frame.isVisible() is False
    sec.expand()
    assert sec.is_expanded is True


def test_header_can_be_hidden(app):
    sec = CollapsibleSection("Params", QLabel("body"))
    sec.set_header_visible(False)
    assert sec._toggle.isVisible() is False


def test_outer_accent_styles_header(app):
    sec = CollapsibleSection("Stage", QLabel("body"), accent_color="#3b6fb6")
    assert "#3b6fb6" in sec._toggle.styleSheet()


def test_inner_inherits_ancestor_accent(app):
    outer = CollapsibleSection("Outer", QWidget(), accent_color="#3b6fb6")
    inner = CollapsibleSection("Inner", QLabel("x"))
    inner.setParent(outer)
    inner._maybe_inherit_accent()
    assert inner._effective_accent == "#3b6fb6"


def test_set_accent_color_refreshes_descendants(app):
    outer = CollapsibleSection("Outer", QWidget())
    inner = CollapsibleSection("Inner", QLabel("x"))
    inner.setParent(outer)
    outer.set_accent_color("#aa3344")
    assert inner._effective_accent == "#aa3344"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_collapsible_section.py -v`
Expected: FAIL with `ModuleNotFoundError: napariTFM.widgets._collapsible_section`.

- [ ] **Step 3: Create the primitive**

Create `napariTFM/widgets/_collapsible_section.py`:

```python
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

    def set_title(self, title: str) -> None:
        self._base_title = title
        self._toggle.setText(self._qt_display_text(title))

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

    def collapse(self) -> None:
        self._toggle.setChecked(False)

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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_collapsible_section.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Verify line endings (new file — must be LF-clean and consistent)**

Run: `git add -N napariTFM/widgets/_collapsible_section.py && git diff --stat napariTFM/widgets/_collapsible_section.py` — new file, no CR concern, but confirm `file napariTFM/widgets/_collapsible_section.py` does not report `CRLF`.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_collapsible_section.py tests/test_collapsible_section.py
git commit -m "Port CellFlow CollapsibleSection primitive into napariTFM"
```

---

## Phase B — Reframe StageSection on CollapsibleSection

### Task B1: Host the parameter panel inside a header-hidden `CollapsibleSection`

**Files:**
- Modify: `napariTFM/widgets/_stage_section.py:100-216`
- Test: `tests/test_stage_section_nesting.py` (overwrite), `tests/test_theme_switching.py`

The bespoke `_parameter_content` QWidget + `_set_parameter_panel_expanded` is replaced by a `CollapsibleSection` (header hidden, accent set). `params_btn` drives the section. `set_accent` propagates to the section via `set_accent_color`. The always-visible body and header row are unchanged. **This task does NOT touch the proxy machinery — that is Phase C.**

- [ ] **Step 1: Overwrite `tests/test_stage_section_nesting.py`**

```python
import pytest
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from napariTFM.widgets._collapsible_section import CollapsibleSection
from napariTFM.widgets._stage_section import StageSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _panel():
    p = QWidget()
    return p


def test_param_panel_lives_in_a_collapsible_section(app):
    sec = StageSection("Stage", QLabel("body"), parameter_panel=_panel())
    assert isinstance(sec._param_section, CollapsibleSection)
    # Header of the inner CollapsibleSection is hidden — the stage's own
    # params_btn is the visible toggle.
    assert sec._param_section._toggle.isVisible() is False


def test_body_visible_regardless_of_params_toggle(app):
    sec = StageSection("Stage", QLabel("body"), parameter_panel=_panel())
    assert sec._content.isVisible() is True
    sec.params_btn.setChecked(True)
    assert sec._content.isVisible() is True
    sec.params_btn.setChecked(False)
    assert sec._content.isVisible() is True


def test_params_button_toggles_only_the_collapsible(app):
    sec = StageSection("Stage", QLabel("body"), parameter_panel=_panel())
    sec.params_btn.setChecked(True)
    assert sec._param_section.is_expanded is True
    sec.params_btn.setChecked(False)
    assert sec._param_section.is_expanded is False


def test_no_param_panel_hides_params_button(app):
    sec = StageSection("Stage", QLabel("body"))
    assert sec.params_btn.isVisible() is False
    assert sec._param_section is None


def test_no_inner_section_api(app):
    sec = StageSection("Stage", QLabel("body"))
    assert not hasattr(sec, "add_inner_section")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_stage_section_nesting.py -v`
Expected: FAIL (`_param_section` does not exist).

- [ ] **Step 3: Replace the param-content block in `_stage_section.py`**

Add the import at the top (after the `_ui_style` import block, lines 6-12):

```python
from napariTFM.widgets._collapsible_section import CollapsibleSection
```

Replace lines 100-106 (the `_parameter_content` QWidget construction):

```python
        self._parameter_content = QWidget()
        parameter_layout = QVBoxLayout()
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.setSpacing(COMPACT_SPACING)
        self._parameter_content.setLayout(parameter_layout)
        if self.parameter_panel is not None:
            parameter_layout.addWidget(self.parameter_panel)
```

with:

```python
        if self.parameter_panel is not None:
            self._param_section = CollapsibleSection(
                self._title,
                self.parameter_panel,
                expanded=False,
                accent_color=self._accent,
            )
            self._param_section.set_header_visible(False)
        else:
            self._param_section = None
```

- [ ] **Step 4: Replace the layout + visibility block (old lines 115-134)**

Replace:

```python
        layout.addLayout(header_layout)
        if self.status_panel is not None:
            layout.addWidget(self.status_panel)
        layout.addWidget(self._parameter_content)
        layout.addWidget(self._content)

        # Body (action buttons / status) is always visible.
        self._content.setVisible(True)
        self._child.setVisible(True)

        self.set_status(status)

        # The params button is the ONLY collapsible: it toggles the parameter
        # panel. Sections without a panel simply have no params affordance.
        has_panel = self.parameter_panel is not None
        self.params_btn.setVisible(has_panel)
        self._parameter_content.setVisible(False)
        self.params_btn.setChecked(parameters_expanded if has_panel else False)
        if has_panel:
            self._set_parameter_panel_expanded(parameters_expanded)
```

with:

```python
        layout.addLayout(header_layout)
        if self.status_panel is not None:
            layout.addWidget(self.status_panel)
        if self._param_section is not None:
            layout.addWidget(self._param_section)
        layout.addWidget(self._content)

        # Body (action buttons / status) is always visible.
        self._content.setVisible(True)
        self._child.setVisible(True)

        self.set_status(status)

        # The params button is the ONLY collapsible: it toggles the parameter
        # section. Sections without a panel simply have no params affordance.
        has_panel = self._param_section is not None
        self.params_btn.setVisible(has_panel)
        self.params_btn.setChecked(parameters_expanded if has_panel else False)
        if has_panel:
            self._set_parameter_panel_expanded(parameters_expanded)
```

- [ ] **Step 5: Rewrite `_set_parameter_panel_expanded` (old lines 212-216)**

Replace:

```python
    def _set_parameter_panel_expanded(self, expanded: bool):
        self.params_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._parameter_content.setVisible(expanded)
        if self.parameter_panel is not None:
            self.parameter_panel.setVisible(expanded)
```

with:

```python
    def _set_parameter_panel_expanded(self, expanded: bool):
        self.params_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        if self._param_section is not None:
            self._param_section._toggle.setChecked(expanded)
```

- [ ] **Step 6: Propagate accent into the param section in `set_accent` (lines 156-159)**

Replace:

```python
    def set_accent(self, accent: str) -> None:
        """Re-accent this section's header (used by the theme picker)."""
        self._accent = accent
        self.header_label.setStyleSheet(stage_header_style(accent))
```

with:

```python
    def set_accent(self, accent: str) -> None:
        """Re-accent this section's header (used by the theme picker)."""
        self._accent = accent
        self.header_label.setStyleSheet(stage_header_style(accent))
        if self._param_section is not None:
            self._param_section.set_accent_color(accent)
```

- [ ] **Step 7: Run the section + theme suites**

Run: `python -m pytest tests/test_stage_section_nesting.py tests/test_theme_switching.py -v`
Expected: PASS. (If `test_theme_switching.py` asserts on `_parameter_content`, update it to `_param_section`.)

- [ ] **Step 8: Run the full suite to find shell breakage**

Run: `python -m pytest -q 2>&1 | tail -15`
Expected: `tests/test_workflow_shell.py` and `tests/test_stage_section_header.py` may reference `_parameter_content`; note failures for Phase C. If only those fail on `_parameter_content`, that is expected breakage repaired in Phase C. **Do not commit if `tests/test_stage_section_nesting.py` or `tests/test_theme_switching.py` fail.**

- [ ] **Step 9: Verify line endings, then commit**

```bash
git diff -w --stat napariTFM/widgets/_stage_section.py   # must equal git diff --stat
git add napariTFM/widgets/_stage_section.py tests/test_stage_section_nesting.py tests/test_theme_switching.py
git commit -m "StageSection: host parameter panel in a header-hidden CollapsibleSection"
```

---

## Phase C — Retire the proxy machinery (signal-based, header owns buttons)

**Contract (applies to every inner stage widget):**
- Define a Qt signal `action_states_changed = Signal()` (parameterless).
- Define `action_states(self) -> dict[str, bool]` returning enablement keyed by action name (`"run"`, `"preview"`, `"cancel"`, and `"gcv"` for force).
- Expose handler callables under stable attribute names the host connects to. Reuse the existing controller calls / `_on_*` handlers.
- Replace every `self.<btn>.setEnabled(x)` in the data-availability / freeze methods with updating an internal `self._action_enabled[...] = x` dict, then `self.action_states_changed.emit()`.
- Remove the now-dead `QPushButton` action controls (`process_btn`/`preview_btn`/`cancel_btn` etc.) from the inner body layout. Keep `cancel` reachable via the header's run/cancel toggle. (Stage-specific extras like force's `gcv_btn` that have no header slot stay in the body but still feed `action_states`.)

`Signal` import: `from qtpy.QtCore import Signal`.

---

### Task C1: Rework `StageSection` to own buttons and bind a signal

**Files:**
- Modify: `napariTFM/widgets/_stage_section.py`
- Test: `tests/test_stage_section_action_sync.py` (overwrite), `tests/test_stage_section_header.py`

Replace `action_targets: dict[str, QWidget]` + `_ActionStateSync` with:
- `actions: dict[str, Callable] | None` — click handlers per action name.
- `action_states: Callable[[], dict[str, bool]] | None` — current enablement.
- `action_states_changed` — a `SignalInstance | None` the section connects to in order to re-apply enablement.

- [ ] **Step 1: Overwrite `tests/test_stage_section_action_sync.py`**

```python
import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QLabel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _Source(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.enabled = {"run": False, "preview": False}
        self.ran = 0
        self.previewed = 0

    def states(self):
        return dict(self.enabled)

    def run(self):
        self.ran += 1

    def preview(self):
        self.previewed += 1


def test_header_buttons_disabled_until_states_allow(app):
    from napariTFM.widgets._stage_section import StageSection

    src = _Source()
    sec = StageSection(
        "Stage",
        QLabel("body"),
        actions={"run": src.run, "preview": src.preview},
        action_states=src.states,
        action_states_changed=src.changed,
    )
    assert sec.run_cancel_btn.isEnabled() is False
    assert sec.preview_button.isEnabled() is False

    src.enabled = {"run": True, "preview": True}
    src.changed.emit()
    assert sec.run_cancel_btn.isEnabled() is True
    assert sec.preview_button.isEnabled() is True


def test_header_run_and_preview_invoke_handlers(app):
    from napariTFM.widgets._stage_section import StageSection

    src = _Source()
    src.enabled = {"run": True, "preview": True}
    sec = StageSection(
        "Stage",
        QLabel("body"),
        actions={"run": src.run, "preview": src.preview},
        action_states=src.states,
        action_states_changed=src.changed,
    )
    src.changed.emit()
    sec.run_cancel_btn.click()
    sec.preview_button.click()
    assert src.ran == 1
    assert src.previewed == 1


def test_no_action_state_sync_class():
    import napariTFM.widgets._stage_section as mod

    assert not hasattr(mod, "_ActionStateSync")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_stage_section_action_sync.py -v`
Expected: FAIL (`StageSection` has no `actions`/`action_states` kwargs; `_ActionStateSync` still present).

- [ ] **Step 3: Delete `_ActionStateSync` (lines 15-42) and its import usage**

Remove the entire `class _ActionStateSync(QObject):` block (lines 15-42). Remove `QEvent, QObject` from the `qtpy.QtCore` import on line 3 (keep `Qt`). Add `Callable` import at top: `from typing import Callable`.

- [ ] **Step 4: Rewrite the constructor signature + state wiring (lines 48-70)**

Replace the signature and the `_action_targets`/`_action_state_syncs` lines:

```python
    def __init__(
        self,
        title: str,
        child: QWidget,
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
            self._accent = stage_accent(self._slug)
        self._action_state_syncs: list[_ActionStateSync] = []
```

with:

```python
    def __init__(
        self,
        title: str,
        child: QWidget,
        actions: dict[str, Callable] | None = None,
        action_states: Callable[[], dict[str, bool]] | None = None,
        action_states_changed=None,
        status: str = "not_started",
        accent: str | None = None,
        status_panel: QWidget | None = None,
        parameter_panel: QWidget | None = None,
        parameters_expanded: bool = False,
    ):
        super().__init__()
        self._title = title
        self._child = child
        self._actions = actions or {}
        self._action_states = action_states
        self._status = status
        self.status_panel = status_panel
        self.parameter_panel = parameter_panel
        if accent is not None:
            self._accent = accent
        else:
            self._accent = stage_accent(self._slug)
```

- [ ] **Step 5: Bind the signal + initial enablement at the end of `__init__`**

After the existing `self.set_status(status)` / params-button block (now ending around the `_set_parameter_panel_expanded` call), append:

```python
        if action_states_changed is not None:
            action_states_changed.connect(self._refresh_action_states)
        self._refresh_action_states()
```

- [ ] **Step 6: Rewrite the button factories**

`_create_action_button` (lines 161-175) — drop the target/proxy logic:

```python
    def _create_action_button(self, action: str, standard_icon: QStyle.StandardPixmap):
        button = make_icon_button(
            self,
            action,
            f"stage_{self._slug}_{action}_button",
            f"{action.capitalize()} {self._title}",
            standard_icon,
        )
        handler = self._actions.get(action)
        if handler is not None:
            button.clicked.connect(lambda _checked=False, fn=handler: fn())
        button.setEnabled(False)
        return button
```

`_create_run_cancel_button` (lines 189-202) — drop the proxy/sync:

```python
    def _create_run_cancel_button(self):
        button = make_icon_button(
            self,
            "run_cancel",
            f"stage_{self._slug}_run_cancel_button",
            f"Run {self._title}",
            QStyle.SP_MediaPlay,
        )
        button.setEnabled(False)
        button.clicked.connect(self._on_run_cancel_clicked)
        return button
```

`_on_run_cancel_clicked` (lines 204-210) — call handler callables instead of target `.click()`:

```python
    def _on_run_cancel_clicked(self):
        key = "cancel" if self._status == "running" else "run"
        handler = self._actions.get(key)
        if handler is not None:
            handler()
```

- [ ] **Step 7: Add `_refresh_action_states`**

Add this method (near `set_status`):

```python
    def _refresh_action_states(self):
        states = self._action_states() if self._action_states is not None else {}
        running = self._status == "running"
        # While running, the run/cancel button is always live (it cancels).
        self.run_cancel_btn.setEnabled(running or states.get("run", False))
        self.preview_button.setEnabled(states.get("preview", False))
```

Then call `self._refresh_action_states()` at the end of `set_status` so the run/cancel button re-enables on the running transition. Add to `set_status` (after the existing icon/tooltip if/else):

```python
        self._refresh_action_states()
```

(Guard: `set_status` runs during `__init__` before `run_cancel_btn`/`preview_button` exist? No — they are created at lines 91-93, before the first `set_status(status)` at line 125. Safe.)

- [ ] **Step 8: Run the action-sync + header suites**

Run: `python -m pytest tests/test_stage_section_action_sync.py tests/test_stage_section_header.py -v`
Expected: action-sync PASS. `test_stage_section_header.py` may assert old `action_targets` behavior — update those assertions to the new `actions`/`action_states` API (header still has `status_indicator`, `params_btn`, `run_cancel_btn`, `preview_button`).

- [ ] **Step 9: Commit (shell + inner widgets still broken — expected)**

```bash
git diff -w --stat napariTFM/widgets/_stage_section.py   # must equal git diff --stat
git add napariTFM/widgets/_stage_section.py tests/test_stage_section_action_sync.py tests/test_stage_section_header.py
git commit -m "StageSection: signal-driven actions, header owns buttons; drop _ActionStateSync"
```

---

### Task C2: preprocessing_widget — expose actions, emit states

**Files:**
- Modify: `napariTFM/widgets/preprocessing_widget.py`
- Test: `tests/test_preprocessing_ownership.py`

Preprocessing's "preview" is the `preview_check` checkbox; expose a `toggle_preview()` handler that flips it. `run` = `_on_process_clicked`; `cancel` = `controller.cancel_all_operations`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_preprocessing_ownership.py`:

```python
def test_preprocessing_exposes_action_contract(app, preprocessing_widget):
    w = preprocessing_widget
    assert hasattr(w, "action_states_changed")
    states = w.action_states()
    assert set(states) >= {"run", "preview", "cancel"}
    assert callable(w.run_action)
    assert callable(w.preview_action)
    assert callable(w.cancel_action)
```

(Use the file's existing `preprocessing_widget`/`app` fixtures; if absent, build the widget the way the other tests in this file do.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_preprocessing_ownership.py::test_preprocessing_exposes_action_contract -v`
Expected: FAIL (`action_states_changed` missing).

- [ ] **Step 3: Add the signal + state dict**

At class top add `action_states_changed = Signal()` (import `Signal` from `qtpy.QtCore`). In `__init__` before building controls add:

```python
        self._action_enabled = {"run": False, "preview": False, "cancel": True}
```

- [ ] **Step 4: Expose handlers + accessor**

Add methods:

```python
    def action_states(self):
        return dict(self._action_enabled)

    def run_action(self):
        self._on_process_clicked()

    def preview_action(self):
        self.preview_check.setChecked(not self.preview_check.isChecked())

    def cancel_action(self):
        self.controller.cancel_all_operations()
```

- [ ] **Step 5: Redirect enablement to the state dict (lines 517-533)**

Replace the `setEnabled` calls in the data-availability + freeze methods. Where the code currently does `self.process_btn.setEnabled(self._has_required_data())` and `self.preview_check.setEnabled(has_any_data)`, instead set `self._action_enabled["run"] = ...` / `self._action_enabled["preview"] = ...` and at the end of each method emit:

```python
        self.action_states_changed.emit()
```

Keep `self.preview_check.setEnabled(...)` only if the checkbox remains visible in the body; the header preview button reads `action_states()["preview"]`. (Preview lives in the body as a checkbox AND is mirrored by the header button — acceptable; the duplicate here is a checkbox+button, not two buttons, and the body checkbox shows preview state.)

- [ ] **Step 6: Remove the body run/cancel buttons**

Delete `self.process_btn = QPushButton("Run Preprocessing")` (lines 432-434) and `self.cancel_btn = QPushButton("Cancel Operation")` (lines 438-439) and their `clicked.connect` lines (468-469), since the header now owns run/cancel. Move the `self.controller`-side wiring into `run_action`/`cancel_action`. (If `action_frame` becomes empty, leave the now-empty frame or remove it — but do not change `preview_check`.)

- [ ] **Step 7: Run the preprocessing suite**

Run: `python -m pytest tests/test_preprocessing_ownership.py tests/test_preprocessing_ui_redesign.py -v`
Expected: PASS (update any test asserting `process_btn`/`cancel_btn` existence).

- [ ] **Step 8: Verify line endings + commit**

```bash
git diff -w --stat napariTFM/widgets/preprocessing_widget.py   # must equal git diff --stat
git add napariTFM/widgets/preprocessing_widget.py tests/test_preprocessing_ownership.py tests/test_preprocessing_ui_redesign.py
git commit -m "preprocessing: expose signal-driven action contract; drop body run/cancel"
```

---

### Task C3: displacement_analysis_widget — expose actions, emit states

**Files:**
- Modify: `napariTFM/widgets/displacement_analysis_widget.py`
- Test: `tests/test_displacement_ownership.py`

`run` = `controller.calculate_all_frames`; `preview` = `controller.preview_displacement`; `cancel` = `controller.cancel_operation`.

- [ ] **Step 1: Write the failing test** (mirror C2's contract test in `tests/test_displacement_ownership.py`, asserting `action_states()` keys `{"run","preview","cancel"}` and the three callables).

- [ ] **Step 2: Run to verify it fails.**

Run: `python -m pytest tests/test_displacement_ownership.py -k action_contract -v` → FAIL.

- [ ] **Step 3: Add `action_states_changed = Signal()`, `self._action_enabled = {"run": False, "preview": False, "cancel": True}`.**

- [ ] **Step 4: Add accessor + handlers:**

```python
    def action_states(self):
        return dict(self._action_enabled)

    def run_action(self):
        self.controller.calculate_all_frames()

    def preview_action(self):
        self.controller.preview_displacement()

    def cancel_action(self):
        self.controller.cancel_operation()
```

- [ ] **Step 5: Redirect enablement (lines 482-491):** replace `self.process_btn.setEnabled(can_analyze)` / `self.preview_btn.setEnabled(...)` with `self._action_enabled["run"]`/`["preview"]` assignments; emit `self.action_states_changed.emit()` at method end.

- [ ] **Step 6: Remove body buttons (lines 411-427):** delete `preview_btn`, `process_btn`, `cancel_btn` creation + `row.addWidget` + the `clicked.connect` lines (461-463). The row that held preview/process can be removed if empty.

- [ ] **Step 7: Run** `python -m pytest tests/test_displacement_ownership.py -v` → PASS (update stale button-existence assertions).

- [ ] **Step 8: Verify line endings + commit:**

```bash
git diff -w --stat napariTFM/widgets/displacement_analysis_widget.py
git add napariTFM/widgets/displacement_analysis_widget.py tests/test_displacement_ownership.py
git commit -m "displacement: expose signal-driven action contract; drop body buttons"
```

---

### Task C4: fttc_widget (Force) — expose actions, emit states

**Files:**
- Modify: `napariTFM/widgets/fttc_widget.py`
- Test: `tests/test_force_ownership.py`

`run` = `controller.calculate_forces`; `preview` = `controller.preview_force`; `cancel` = `controller.cancel_operation`. The `gcv_btn` (optimal regularization) has no header slot — keep it in the body but include `"gcv"` in `action_states` and keep its own `setEnabled` for the body button.

- [ ] **Step 1: Write the failing contract test** in `tests/test_force_ownership.py` (keys `{"run","preview","cancel"}`; three callables).

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Add signal + `self._action_enabled = {"run": False, "preview": False, "cancel": True}`.**

- [ ] **Step 4: Add accessor + handlers** (`run_action`→`calculate_forces`, `preview_action`→`preview_force`, `cancel_action`→`cancel_operation`).

- [ ] **Step 5: Redirect enablement (lines 469-480):** set `self._action_enabled["run"/"preview"]` from `has_displacement`/`not frozen`; keep `self.gcv_btn.setEnabled(...)` (body button stays); emit `self.action_states_changed.emit()` at end of each method.

- [ ] **Step 6: Remove body `preview_btn`/`process_btn`/`cancel_btn` (lines 401-422)** and their `clicked.connect` (452-455 except `gcv_btn`). Keep `gcv_btn` and its connect (454).

- [ ] **Step 7: Run** `python -m pytest tests/test_force_ownership.py -v` → PASS.

- [ ] **Step 8: Verify line endings + commit:**

```bash
git diff -w --stat napariTFM/widgets/fttc_widget.py
git add napariTFM/widgets/fttc_widget.py tests/test_force_ownership.py
git commit -m "force: expose signal-driven action contract; drop body run/preview/cancel"
```

---

### Task C5: msm_widget (Stress) — expose actions, emit states

**Files:**
- Modify: `napariTFM/widgets/msm_widget.py`
- Test: `tests/test_stress_ownership.py`

`run` = `controller.start_analysis` (`analyze_btn`); `preview` = `controller.preview_current_frame` (`preview_frame_btn`); `cancel` = `controller.cancel_all_operations`. The `preview_mesh_btn` has no header slot — keep it in the body and in `action_states` as `"mesh"`.

- [ ] **Step 1: Write the failing contract test** in `tests/test_stress_ownership.py`.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Add signal + `self._action_enabled = {"run": False, "preview": False, "cancel": True}`.**

- [ ] **Step 4: Add accessor + handlers** (`run_action`→`start_analysis`, `preview_action`→`preview_current_frame`, `cancel_action`→`cancel_all_operations`).

- [ ] **Step 5: Redirect enablement (lines 607-618):** set `self._action_enabled["run"/"preview"]` from `has_force and has_mask`; keep `preview_mesh_btn` body button as-is; emit `self.action_states_changed.emit()` at end of each method.

- [ ] **Step 6: Remove body `preview_frame_btn`/`analyze_btn`/`cancel_btn` (lines 488-500)** + their `clicked.connect` (530-532). Keep `preview_mesh_btn` (529).

- [ ] **Step 7: Run** `python -m pytest tests/test_stress_ownership.py -v` → PASS.

- [ ] **Step 8: Verify line endings + commit:**

```bash
git diff -w --stat napariTFM/widgets/msm_widget.py
git add napariTFM/widgets/msm_widget.py tests/test_stress_ownership.py
git commit -m "stress: expose signal-driven action contract; drop body run/preview/cancel"
```

---

### Task C6: Rewire the shell (`_widget.py`) to the new StageSection API

**Files:**
- Modify: `napariTFM/widgets/_widget.py:463-516`
- Test: `tests/test_workflow_shell.py`

Replace each `action_targets={...}` block with `actions=`, `action_states=`, `action_states_changed=` drawn from the now-exposed inner-widget contract.

- [ ] **Step 1: Write/extend the failing shell test**

Add to `tests/test_workflow_shell.py` (using the existing manager-stub + `_real_parameter_manager` machinery):

```python
def test_stage_sections_use_signal_action_contract(app, main_widget):
    sec = main_widget._stage_sections_by_key["displacement"]
    assert "run" in sec._actions and "preview" in sec._actions
    assert sec._action_states is not None
    # No proxy machinery remnants
    assert not hasattr(sec, "_action_state_syncs")
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Rewire the five sections (lines 463-516).** For each, replace `action_targets={...}` with the new trio. Example for displacement:

```python
            "displacement": _StageSection(
                "Displacement",
                self.displacement_widget,
                status_panel=self._stage_status_panels_by_key["displacement"],
                parameter_panel=self._stage_parameter_panels_by_key.get("displacement"),
                actions={
                    "run": self.displacement_widget.run_action,
                    "preview": self.displacement_widget.preview_action,
                    "cancel": self.displacement_widget.cancel_action,
                },
                action_states=self.displacement_widget.action_states,
                action_states_changed=self.displacement_widget.action_states_changed,
            ),
```

Apply the same shape to `preprocessing` (`self.preprocessing_widget`), `force` (`self.force_widget`), `stress` (`self.msm_widget`). For `batch` (run-only), use:

```python
            "batch": _StageSection(
                "Batch Analysis",
                self.batch_widget,
                status_panel=self._stage_status_panels_by_key["batch"],
                actions={"run": self.batch_widget.run_analysis_btn.click},
                action_states=lambda: {"run": True},
            ),
```

(Batch keeps its own button; `.click` is a stable bound callable, no proxy. If `run_analysis_btn` enablement matters, give `batch_widget` the same contract in a follow-up — out of scope here.)

- [ ] **Step 4: Run the shell suite**

Run: `python -m pytest tests/test_workflow_shell.py -v`
Expected: PASS (rewrite any test still referencing `action_targets`/`_action_state_syncs`).

- [ ] **Step 5: Verify line endings + commit:**

```bash
git diff -w --stat napariTFM/widgets/_widget.py
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Shell: wire stage sections to signal-driven action contract"
```

---

### Task C7: Full-suite verification, CRLF audit, manual smoke

**Files:** none (verification only).

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q 2>&1 | tail -15`
Expected: all pass (the napari-compat flake may need isolation). Record the count.

- [ ] **Step 2: Confirm the flake passes in isolation**

Run: `python -m pytest tests/test_napari_compatibility.py -v`
Expected: PASS in isolation.

- [ ] **Step 3: Confirm no proxy machinery remains**

Run: `grep -rn "_ActionStateSync\|action_targets" napariTFM/ tests/`
Expected: no matches in `napariTFM/`; tests reference only the new API.

- [ ] **Step 4: CRLF audit across the whole slice**

Run: `git diff --stat <slice-parent>..HEAD` and `git diff -w --stat <slice-parent>..HEAD` — must be identical. Spot-check `git show <each-commit> | grep -nP '\r$'` finds no added line with a trailing CR.

- [ ] **Step 5: Manual smoke (needs napari — owner runs)**

Launch napari, load a dataset, and confirm per stage: header shows status dot + ⚙ params + ▶ run + preview; ⚙ toggles ONLY the `CollapsibleSection` param panel (left-stripe accent, body stays visible); run/preview disable until data is present and re-enable correctly; run→cancel toggle works while running; theme menu (◐) re-accents the param section's stripe live; preprocessing→displacement→force→stress runs end-to-end.

---

## Self-Review

**Spec coverage:**
- TODO Step 2 (one section primitive) → Phase A (port) + Phase B (StageSection hosts CollapsibleSection). ✔
- TODO Step 3 (expose inner controls, retire `_ActionStateSync`/`action_targets`) → Phase C. ✔
- Accent inheritance / live re-accent → A2 (`set_accent_color`) + B1 Step 6 (`set_accent` propagation). ✔

**Type consistency:** `action_states_changed` (signal), `action_states()` (dict accessor), `run_action`/`preview_action`/`cancel_action` (callables) are used identically in C2–C5 (inner widgets) and consumed identically in C1 (StageSection) and C6 (shell). `_param_section` is the single name for the hosted CollapsibleSection (introduced B1, used in B1/C1).

**Open follow-ups (out of scope, do not block this slice):**
- TODO Step 4 (port grid-layout vocabulary; rebuild `WorkflowParameterPanel` on it) — sits on top of this slice.
- Batch stage does not yet emit `action_states_changed`; its run button is always enabled here.
- `tier4-state-architecture.md` (state coherence) remains orthogonal and untracked.
