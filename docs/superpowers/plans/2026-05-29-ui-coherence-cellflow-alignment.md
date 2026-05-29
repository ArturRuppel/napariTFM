# UI Coherence: CellFlow Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring napariTFM's ported UI into coherence with CellFlow by (A) adopting CellFlow's palette-based theme system + trimming the over-specified status states, (B) replacing the QForm/spinbox parameter editor with CellFlow's labeled-slider vocabulary, and (C) removing the "parameters nested as a faux-stage" hack and the duplicate calibration editor.

**Architecture:** Three independent phases, each shippable on its own. Phase A rewrites `_ui_style.py` into a palette-indirection system (`ACTIVE_PALETTE` + `stage_accent(semantic_key)` + `set_active_theme`) and adds a theme picker to the shell footer; it touches no stage behavior. Phase B ports CellFlow's slider factories (`dslider`/`islider`) into a new `_param_controls.py` and rebuilds `WorkflowParameterPanel` to render sliders. Phase C makes `StageSection` mount a parameter panel in a first-class collapsible region (body always visible), deletes `add_inner_section` + the shell's disconnect/reconnect wiring, and drops the duplicate "General" calibration group from the preprocessing stage panel.

**Tech Stack:** Python, qtpy (PyQt6), superqt 0.8.0 (already an env dependency via `qtrangeslider`; `QLabeledSlider`/`QLabeledDoubleSlider` come from `superqt`), pytest with `QApplication` fixtures.

---

## Background & Constraints

**Line endings:** Several napariTFM source files have mixed CRLF/LF. After any edit, verify the *real* diff with `git diff -w -- <file>` (or `git show <sha> -w`) before trusting it. Tell implementers to touch only target lines and never reformat/normalize whitespace.

**Test stubbing gotcha:** `tests/test_workflow_shell.py` replaces `napariTFM.widgets.preprocessing_widget` etc. in `sys.modules` with stub modules at import time, and stubs `ParameterManager`/`DataManager`. Tests that need the *real* `ParameterManager` load it via `importlib.util.spec_from_file_location`. Follow that existing pattern for any new shell-level test that needs a real manager.

**Known env flake (not caused by this work):** `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` intermittently SIGSEGVs in its spawned subprocess. Re-run in isolation to confirm green; do not chase it.

**Reference files (read before starting a phase):**
- CellFlow theme system: `~/Projects/CellFlow/src/cellflow/napari/ui_style.py`
- CellFlow slider factories: `~/Projects/CellFlow/src/cellflow/napari/_widget_helpers.py` (`dslider`, `islider`, `_stack_slider_label_above`, `_patch_label_autosize`)
- CellFlow theme picker: `~/Projects/CellFlow/src/cellflow/napari/main_widget.py:163-208` (`_setup_theme_selector`, `_on_theme_selected`, `_apply_theme_accents`)

**Run the full suite with:** `python -m pytest -q` from the repo root. Baseline is currently green (154 passed, minus the known flake).

---

## File Structure

**Phase A (theme + status trim):**
- Modify: `napariTFM/widgets/_ui_style.py` — add palettes, `ACTIVE_PALETTE`, `THEME_PALETTES`, `theme_names()`, `active_theme_name()`, `set_active_theme()`; change `STAGE_ACCENTS` to semantic color-name values; make `stage_accent()` resolve through the palette; trim `STATUS_COLORS`/`STATUS_GLYPHS` (drop `stale`).
- Modify: `napariTFM/widgets/_stage_section.py` — add `set_accent(accent)` so a live section can be re-themed.
- Modify: `napariTFM/widgets/_widget.py` — add a footer theme picker; re-accent sections on theme change.
- Modify: `tests/test_ui_style.py` — rewrite accent assertions for palette indirection; assert `stale` is gone.
- Create: `tests/test_theme_switching.py` — palette switch changes `stage_accent` output and re-styles sections.

**Phase B (slider parameter vocabulary):**
- Create: `napariTFM/widgets/_param_controls.py` — `dslider`, `islider`, `_stack_slider_label_above`, `_patch_label_autosize` ported from CellFlow.
- Modify: `napariTFM/widgets/_widget.py` — rewrite `WorkflowParameterPanel._create_control` to build sliders for int/float (combo/checkbox unchanged); keep the same `ParameterManager` binding and `_sync_parameter` behavior.
- Create: `tests/test_param_controls.py` — slider factory behavior.
- Modify: `tests/` — any test asserting spinbox types for workflow params (see Task B3 for the grep).

**Phase C (mounting refactor + calibration de-dup):**
- Modify: `napariTFM/widgets/_stage_section.py` — body always visible; `parameter_panel` region is the single collapsible; delete `add_inner_section` and dormant `_find_ancestor_accent`; `params_btn` toggles only the parameter region.
- Modify: `napariTFM/widgets/_widget.py` — pass `parameter_panel=` to each `StageSection`; delete the `add_inner_section` loop and the `params_btn.toggled.disconnect()`/reconnect block; drop `"General"` from the preprocessing stage panel titles.
- Delete/Rewrite: `tests/test_stage_section_nesting.py` — `add_inner_section` is gone; replace with parameter-region tests.
- Modify: `tests/test_workflow_shell.py` — update the inline-parameter-panel toggle test to the new mounting.

---

# PHASE A — Theme system + status-state trim

Independent of stage behavior. Ship first.

### Task A1: Port the palette system into `_ui_style.py`

**Files:**
- Modify: `napariTFM/widgets/_ui_style.py`
- Test: `tests/test_theme_switching.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_theme_switching.py`:

```python
import pytest

from napariTFM.widgets import _ui_style


@pytest.fixture(autouse=True)
def _restore_theme():
    original = _ui_style.active_theme_name()
    yield
    _ui_style.set_active_theme(original)


def test_theme_names_nonempty_and_contains_default():
    names = _ui_style.theme_names()
    assert len(names) >= 2
    assert _ui_style.active_theme_name() in names


def test_stage_accent_resolves_through_active_palette():
    # STAGE_ACCENTS now maps a stage key to a semantic palette color name.
    semantic = _ui_style.STAGE_ACCENTS["preprocessing"]
    expected = _ui_style.ACTIVE_PALETTE[semantic]
    assert _ui_style.stage_accent("preprocessing") == expected


def test_stage_accent_unknown_key_falls_back_to_inputs():
    assert _ui_style.stage_accent("nope") == _ui_style.stage_accent("inputs")


def test_set_active_theme_changes_resolved_accent():
    names = _ui_style.theme_names()
    other = next(n for n in names if n != _ui_style.active_theme_name())
    before = _ui_style.stage_accent("preprocessing")
    _ui_style.set_active_theme(other)
    after = _ui_style.stage_accent("preprocessing")
    assert _ui_style.active_theme_name() == other
    # At least one stage accent differs between two distinct themes.
    differs = any(
        _ui_style.stage_accent(k) != before
        for k in ("preprocessing", "displacement", "force", "stress", "batch")
    ) or after != before
    assert differs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_theme_switching.py -q`
Expected: FAIL (`AttributeError: module ... has no attribute 'active_theme_name'`).

- [ ] **Step 3: Rewrite the palette section of `_ui_style.py`**

Replace the current `STAGE_ACCENTS` block (lines ~12-22) and `stage_accent` (lines ~66-68) with the palette system below. Keep `COMPACT_SPACING`, `ICON_BUTTON_SIZE`, `MUTED_TEXT_COLOR`, `make_icon_button`, `muted_stage_accent`, `stage_header_style`, `title_style`, `section_label_style`, `caption_style`, `danger_text_style` as they are (note `muted_stage_accent` change in Step 4).

```python
# ── Theme palettes ───────────────────────────────────────────────────────
# Each palette maps semantic color names to hex. Stage keys reference these
# names via STAGE_ACCENTS, so switching ACTIVE_PALETTE re-accents the whole UI.
CIVIDIS = {
    "rosewater": "#d6c35d", "pink": "#555c6d", "mauve": "#243c6e",
    "red": "#555c6d", "peach": "#a79d73", "yellow": "#d6c35d",
    "green": "#7d7c78", "teal": "#7d7c78", "sapphire": "#d6c35d",
    "blue": "#555c6d",
}
VIRIDIS = {
    "rosewater": "#9bd93c", "pink": "#31668e", "mauve": "#463480",
    "red": "#31668e", "peach": "#38b977", "yellow": "#9bd93c",
    "green": "#21918c", "teal": "#21918c", "sapphire": "#9bd93c",
    "blue": "#31668e",
}
NORD = {
    "rosewater": "#bf616a", "pink": "#b48ead", "mauve": "#b48ead",
    "red": "#bf616a", "peach": "#d08770", "yellow": "#ebcb8b",
    "green": "#a3be8c", "teal": "#8fbcbb", "sapphire": "#81a1c1",
    "blue": "#5e81ac",
}
DRACULA = {
    "rosewater": "#ffb86c", "pink": "#ff79c6", "mauve": "#bd93f9",
    "red": "#ff5555", "peach": "#ffb86c", "yellow": "#f1fa8c",
    "green": "#50fa7b", "teal": "#8be9fd", "sapphire": "#8be9fd",
    "blue": "#6272a4",
}

THEME_PALETTES = {
    "Cividis": CIVIDIS,
    "Viridis": VIRIDIS,
    "Nord": NORD,
    "Dracula": DRACULA,
}
ACTIVE_THEME_NAME = "Cividis"
ACTIVE_PALETTE = THEME_PALETTES[ACTIVE_THEME_NAME]

# Stage key -> semantic palette color name. Each visible stage gets a
# distinct accent so the workflow reads as ordered, themeable bands.
STAGE_ACCENTS = {
    "inputs": "sapphire",
    "project": "sapphire",
    "preprocessing": "blue",
    "displacement": "mauve",
    "force": "teal",
    "stress": "peach",
    "batch": "yellow",
}


def theme_names() -> tuple[str, ...]:
    return tuple(THEME_PALETTES)


def active_theme_name() -> str:
    return ACTIVE_THEME_NAME


def set_active_theme(name: str) -> None:
    global ACTIVE_PALETTE, ACTIVE_THEME_NAME
    ACTIVE_THEME_NAME = name
    ACTIVE_PALETTE = THEME_PALETTES[name]


def stage_accent(key: str) -> str:
    """Resolve a stage key to its accent hex via the active palette."""
    semantic = STAGE_ACCENTS.get(key, STAGE_ACCENTS["inputs"])
    return ACTIVE_PALETTE[semantic]
```

- [ ] **Step 4: Repoint `muted_stage_accent` at the resolved hex**

The existing `muted_stage_accent(key)` reads `stage_accent(key).lstrip("#")` then muddles in colorsys — that still works because `stage_accent` now returns a hex. No code change needed *unless* the current body referenced `STAGE_ACCENTS[...]` directly. Confirm with:

Run: `grep -n "STAGE_ACCENTS\[" napariTFM/widgets/_ui_style.py`
Expected: only the `stage_accent` fallback line. If `muted_stage_accent` indexes `STAGE_ACCENTS` directly, change it to call `stage_accent(key)`.

- [ ] **Step 5: Run the new test**

Run: `python -m pytest tests/test_theme_switching.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_ui_style.py tests/test_theme_switching.py
git commit -m "Add palette-based theme system to _ui_style"
```

### Task A2: Trim over-specified status states

**Files:**
- Modify: `napariTFM/widgets/_ui_style.py:24-40` (`STATUS_COLORS`, `STATUS_GLYPHS`)
- Test: `tests/test_theme_switching.py`

- [ ] **Step 1: Add the failing assertion**

Append to `tests/test_theme_switching.py`:

```python
def test_status_states_trimmed_to_computed():
    # 'stale' is never produced by StageDataStatusPanel.refresh(); drop it.
    assert "stale" not in _ui_style.STATUS_COLORS
    assert "stale" not in _ui_style.STATUS_GLYPHS
    # The states that ARE produced/used must remain.
    for s in ("not_started", "ready", "running", "done", "error"):
        assert s in _ui_style.STATUS_COLORS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_theme_switching.py::test_status_states_trimmed_to_computed -q`
Expected: FAIL (`stale` present).

- [ ] **Step 3: Remove `stale` from both dicts**

In `_ui_style.py`, delete the `"stale": "#e9c46a",` line from `STATUS_COLORS` and the `"stale": "⚠",` line from `STATUS_GLYPHS`. Leave `error` in both. Result:

```python
STATUS_COLORS = {
    "not_started": "#8c8c8c",
    "ready": "#2f80ed",
    "running": "#f4a261",
    "done": "#2a9d8f",
    "error": "#d62828",
}

STATUS_GLYPHS = {
    "available": "✓",
    "missing_required": "✗",
    "missing_optional": "○",
    "running": "⟳",
    "error": "⚠",
}
```

- [ ] **Step 4: Confirm nothing references the removed key**

Run: `grep -rn '"stale"\|STATUS_COLORS\["stale"\]\|STATUS_GLYPHS\["stale"\]' napariTFM/`
Expected: no matches.

- [ ] **Step 5: Run the suite**

Run: `python -m pytest tests/test_theme_switching.py tests/test_artifact_row.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_ui_style.py tests/test_theme_switching.py
git commit -m "Trim status states to the ones actually computed"
```

### Task A3: Update `test_ui_style.py` for palette indirection

**Files:**
- Modify: `tests/test_ui_style.py`

- [ ] **Step 1: Rewrite the two accent tests**

The current `test_stage_accent_returns_palette_color_for_known_key` and `test_stage_accent_falls_back_to_inputs_for_unknown_key` assert `stage_accent("x") == STAGE_ACCENTS["x"]`, which is now false (`STAGE_ACCENTS` holds semantic names, not hex). Replace them with:

```python
def test_stage_accent_returns_palette_color_for_known_key():
    from napariTFM.widgets._ui_style import ACTIVE_PALETTE
    assert stage_accent("preprocessing") == ACTIVE_PALETTE[STAGE_ACCENTS["preprocessing"]]
    assert stage_accent("displacement") == ACTIVE_PALETTE[STAGE_ACCENTS["displacement"]]


def test_stage_accent_falls_back_to_inputs_for_unknown_key():
    assert stage_accent("nonexistent_stage") == stage_accent("inputs")
```

- [ ] **Step 2: Fix the hue-family test if it breaks under the default palette**

`test_muted_stage_accent_preserves_hue_family` asserts `b > r` for muted preprocessing. Under Cividis, `preprocessing -> blue -> #555c6d` (r=0x55, b=0x6d → b>r holds). Run the test; if a future default palette breaks it, relax it to `assert len(muted) == 6 and muted != stage_accent("preprocessing").lstrip("#")`. For now, keep as-is and verify.

- [ ] **Step 3: Run**

Run: `python -m pytest tests/test_ui_style.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ui_style.py
git commit -m "Update ui_style tests for palette indirection"
```

### Task A4: Add `set_accent` to StageSection

**Files:**
- Modify: `napariTFM/widgets/_stage_section.py`
- Test: `tests/test_theme_switching.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_theme_switching.py`:

```python
def test_stage_section_set_accent_restyles_header(qtbot=None):
    from qtpy.QtWidgets import QApplication, QWidget
    from napariTFM.widgets._stage_section import StageSection

    app = QApplication.instance() or QApplication([])
    section = StageSection("Force Analysis", QWidget(), accent="#111111")
    assert "#111111" in section.header_label.styleSheet()
    section.set_accent("#abcdef")
    assert "#abcdef" in section.header_label.styleSheet()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_theme_switching.py::test_stage_section_set_accent_restyles_header -q`
Expected: FAIL (`AttributeError: 'StageSection' object has no attribute 'set_accent'`).

- [ ] **Step 3: Implement `set_accent`**

In `napariTFM/widgets/_stage_section.py`, add this method to `StageSection` (e.g. right after `set_status`):

```python
    def set_accent(self, accent: str) -> None:
        """Re-accent this section's header (used by the theme picker)."""
        self._accent = accent
        self.header_label.setStyleSheet(stage_header_style(accent))
```

`stage_header_style` is already imported at the top of the file.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/test_theme_switching.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_stage_section.py tests/test_theme_switching.py
git commit -m "Add StageSection.set_accent for live re-theming"
```

### Task A5: Add the theme picker to the shell footer

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_shell.py` (it already builds the shell with stubbed stage widgets — reuse the existing `app` fixture and the `monkeypatch` stub pattern from `test_main_widget_uses_stage_sections_instead_of_tabs`):

```python
def test_shell_theme_button_switches_palette(monkeypatch, app):
    from napariTFM.widgets import _widget
    from napariTFM.widgets import _ui_style
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    original = _ui_style.active_theme_name()
    try:
        widget = _widget.napariTFMWidget(_StubViewer())
        assert hasattr(widget, "theme_btn")
        other = next(n for n in _ui_style.theme_names() if n != original)
        widget._on_theme_selected(other)
        assert _ui_style.active_theme_name() == other
        # Sections were re-accented to the new palette.
        accent = _ui_style.stage_accent("preprocessing")
        assert accent in widget._stage_sections_by_key["preprocessing"].header_label.styleSheet()
    finally:
        _ui_style.set_active_theme(original)
```

> If `_StubViewer` does not already exist in the test module, reuse whatever viewer stub the neighbouring shell tests pass to `napariTFMWidget(...)`; match their construction exactly.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_workflow_shell.py::test_shell_theme_button_switches_palette -q`
Expected: FAIL (`AttributeError: ... 'theme_btn'`).

- [ ] **Step 3: Add imports**

In `_widget.py`, extend the existing `_ui_style` import (currently `from napariTFM.widgets._ui_style import title_style`) to:

```python
from napariTFM.widgets._ui_style import (
    title_style,
    stage_accent,
    theme_names,
    active_theme_name,
    set_active_theme,
)
```

Add `QMenu`, `QToolButton` to the existing `qtpy.QtWidgets` import line.

- [ ] **Step 4: Build the footer picker and re-accent handler**

In `napariTFMWidget.__init__`, *after* `self._stage_sections_by_key` is fully built and the sections are added to `container_layout`, but before `container_layout.addStretch()`, the sections already exist. Add the theme footer to `main_layout` (the outer layout) after `main_layout.addWidget(scroll)`:

```python
        self._setup_theme_selector(main_layout)
```

Then add these methods to the class:

```python
    def _setup_theme_selector(self, layout):
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch()

        self.theme_btn = QToolButton()
        self.theme_btn.setText("◐")
        self.theme_btn.setObjectName("theme_selector_button")
        self.theme_btn.setPopupMode(QToolButton.InstantPopup)

        self.theme_menu = QMenu(self.theme_btn)
        self._theme_actions = {}
        for name in theme_names():
            action = self.theme_menu.addAction(name)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, theme_name=name: self._on_theme_selected(theme_name)
            )
            self._theme_actions[name] = action
        self.theme_btn.setMenu(self.theme_menu)
        self._sync_theme_menu_state()

        footer.addWidget(self.theme_btn)
        layout.addLayout(footer)

    def _on_theme_selected(self, name: str):
        set_active_theme(name)
        for key, section in self._stage_sections_by_key.items():
            section.set_accent(stage_accent(key))
        self._sync_theme_menu_state()

    def _sync_theme_menu_state(self):
        current = active_theme_name()
        for name, action in self._theme_actions.items():
            action.setChecked(name == current)
        self.theme_btn.setToolTip(f"Theme: {current}")
```

> Note: the `ProjectSection` ("project" key) is not in `_stage_sections_by_key`; if you want it re-accented too, also call `self.project_section.set_accent(stage_accent("project"))` in `_on_theme_selected`. ProjectSection inherits `set_accent` from StageSection (Task A4).

- [ ] **Step 5: Run**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS.

- [ ] **Step 6: Full suite + manual sanity**

Run: `python -m pytest -q`
Expected: PASS (modulo the known napari_compatibility flake — re-run it in isolation if it segfaults).

- [ ] **Step 7: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Add theme picker to shell footer with live re-accenting"
```

**Phase A manual check (needs napari):** Launch the plugin, open the theme menu (◐, bottom-right), switch between Cividis/Viridis/Nord/Dracula, and confirm every stage header stripe + label re-colors immediately.

---

# PHASE B — Slider parameter vocabulary

Replaces the QForm/spinbox look with CellFlow's labeled sliders. Depends only on superqt (already available). Ship after A or independently.

### Task B1: Port slider factories into `_param_controls.py`

**Files:**
- Create: `napariTFM/widgets/_param_controls.py`
- Test: `tests/test_param_controls.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_param_controls.py`:

```python
import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._param_controls import dslider, islider


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_islider_roundtrips_value(app):
    s = islider(0, 50, 12)
    assert s.value() == 12
    s.setValue(20)
    assert s.value() == 20


def test_dslider_respects_decimals_and_range(app):
    s = dslider(0.0, 10.0, 1.5, step=0.1, decimals=1)
    assert abs(s.value() - 1.5) < 1e-9
    s.setValue(3.3)
    assert abs(s.value() - 3.3) < 1e-9


def test_sliders_emit_value_changed(app):
    s = islider(0, 10, 1)
    seen = []
    s.valueChanged.connect(seen.append)
    s.setValue(7)
    assert seen and seen[-1] == 7
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_param_controls.py -q`
Expected: FAIL (`ModuleNotFoundError: napariTFM.widgets._param_controls`).

- [ ] **Step 3: Create the module**

Create `napariTFM/widgets/_param_controls.py`. This is a focused port of the slider plumbing from CellFlow's `_widget_helpers.py` (the value label is stacked above the track, with optional ± step buttons). Copy verbatim:

```python
"""Labeled-slider parameter controls, ported from CellFlow's UI vocabulary."""
from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QSize, Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider, QLabeledSlider


def _patch_label_autosize(label) -> None:
    """Size the editable slider label to fit its widest formatted value."""
    def _get_size():
        dec = label.decimals() if hasattr(label, "decimals") else 0

        def _fmt(v):
            return f"{v:.{dec}f}" if dec else f"{int(v)}"

        lo, hi = label.minimum(), label.maximum()
        sample = max((_fmt(lo), _fmt(hi)), key=len)
        if not sample.startswith("-"):
            sample = "-" + sample
        fm = label.fontMetrics()
        prefix = label.prefix() or ""
        suffix = label.suffix() or ""
        w = fm.horizontalAdvance(prefix + sample + suffix) + 18
        h = label.sizeHint().height()
        opt = QStyleOption()
        return label.style().sizeFromContents(
            QStyle.ContentsType.CT_LineEdit, opt, QSize(w, h), label
        )

    label._get_size = _get_size
    label._update_size()


def _slider_step_button(text: str, object_name: str, tooltip: str) -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setAutoRepeat(True)
    button.setFixedSize(18, 18)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return button


class _StepButtonStateSyncer(QObject):
    def __init__(self, sync) -> None:
        super().__init__()
        self._sync = sync

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.EnabledChange:
            self._sync()
        return False


def _connect_step_buttons(slider):
    decrement = _slider_step_button("-", "slider_decrement_button", "Decrease by one step")
    increment = _slider_step_button("+", "slider_increment_button", "Increase by one step")

    def _step(direction: int) -> None:
        if not slider.isEnabled():
            return
        slider.setValue(slider.value() + direction * slider.singleStep())

    def _sync(*_a) -> None:
        enabled = slider.isEnabled()
        decrement.setEnabled(enabled and slider.value() > slider.minimum())
        increment.setEnabled(enabled and slider.value() < slider.maximum())

    decrement.clicked.connect(lambda: _step(-1))
    increment.clicked.connect(lambda: _step(1))
    slider.valueChanged.connect(_sync)
    slider.rangeChanged.connect(_sync)
    syncer = _StepButtonStateSyncer(_sync)
    syncer.setParent(slider)
    slider.installEventFilter(syncer)
    slider._step_button_state_syncer = syncer
    _sync()
    return decrement, increment


def _stack_label_above(slider, *, step_buttons: bool) -> None:
    label = slider._label
    track = slider._slider
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    _patch_label_autosize(label)

    old_layout = slider.layout()
    if old_layout is not None:
        old_layout.removeWidget(label)
        old_layout.removeWidget(track)
        QWidget().setLayout(old_layout)
    label.setParent(slider)
    track.setParent(slider)
    vbox = QVBoxLayout()
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(0)
    vbox.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)
    if step_buttons:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        dec, inc = _connect_step_buttons(slider)
        row.addWidget(dec, alignment=Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(track)
        row.addWidget(inc, alignment=Qt.AlignmentFlag.AlignVCenter)
        vbox.addLayout(row)
    else:
        vbox.addWidget(track)
    slider.setLayout(vbox)


def dslider(lo, hi, val, step=0.1, decimals=2, tooltip="", *, step_buttons=True):
    s = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _stack_label_above(s, step_buttons=step_buttons)
    return s


def islider(lo, hi, val, step=1, tooltip="", *, step_buttons=True):
    s = QLabeledSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _stack_label_above(s, step_buttons=step_buttons)
    return s
```

- [ ] **Step 4: Run**

Run: `python -m pytest tests/test_param_controls.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_param_controls.py tests/test_param_controls.py
git commit -m "Port CellFlow labeled-slider factories to napariTFM"
```

### Task B2: Render int/float workflow params as sliders

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (`WorkflowParameterPanel._create_control`, `_sync_parameter`)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing test**

The shell test module stubs `ParameterManager`. Use the existing real-manager loader pattern (search the file for `spec_from_file_location` and copy the helper). Append:

```python
def test_workflow_parameter_panel_uses_sliders_for_numeric(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel
    from napariTFM.widgets._param_controls import dslider, islider
    pm = _real_parameter_manager()  # reuse the existing importlib-based helper

    panel = WorkflowParameterPanel(pm, section_titles=("Displacement",))
    # nscales is an int param -> islider; disp_arrow_scale is float -> dslider.
    assert type(panel.parameter_controls["nscales"]).__name__ == "QLabeledSlider"
    assert type(panel.parameter_controls["disp_arrow_scale"]).__name__ == "QLabeledDoubleSlider"


def test_workflow_parameter_panel_slider_writes_through(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel
    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Displacement",))
    panel.parameter_controls["nscales"].setValue(7)
    assert pm.get_ui_parameter("nscales") == 7
```

> If a `_real_parameter_manager()` helper does not yet exist in the module, add one mirroring the existing `importlib.util.spec_from_file_location` usage for `DataManager` in `test_data_manager_disk_truth.py` / the shell tests: load `napariTFM/utilities/parameter_manager.py` as a standalone module and instantiate `ParameterManager`.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest "tests/test_workflow_shell.py::test_workflow_parameter_panel_uses_sliders_for_numeric" -q`
Expected: FAIL (control is `QSpinBox`, not `QLabeledSlider`).

- [ ] **Step 3: Rewrite `_create_control`**

In `_widget.py`, add the import near the other widget imports:

```python
from napariTFM.widgets._param_controls import dslider, islider
```

Replace `WorkflowParameterPanel._create_control` (currently `_widget.py:279-307`) with:

```python
    def _create_control(self, name, kind, min_val, max_val, step, decimals, choices):
        if kind == "int":
            control = islider(min_val, max_val, self.parameter_manager.get_ui_parameter(name), step=step)
            control.valueChanged.connect(lambda value, n=name: self.parameter_manager.set_ui_parameter(n, value))
        elif kind == "float":
            control = dslider(min_val, max_val, self.parameter_manager.get_ui_parameter(name), step=step, decimals=decimals)
            control.valueChanged.connect(lambda value, n=name: self.parameter_manager.set_ui_parameter(n, value))
        elif kind == "choice":
            control = QComboBox()
            control.addItems(choices)
            control.currentTextChanged.connect(lambda value, n=name: self.parameter_manager.set_ui_parameter(n, value))
        elif kind == "bool":
            control = QCheckBox()
            control.stateChanged.connect(
                lambda state, n=name: self.parameter_manager.set_ui_parameter(n, state == Qt.Checked)
            )
        else:
            raise ValueError(f"Unsupported parameter control type: {kind}")

        control.setObjectName(f"workflow_parameter_{name}")
        self.parameter_controls[name] = control
        return control
```

> Note: `gel_height`'s old `setSpecialValueText("∞")` does not exist on sliders. Drop it — the slider shows the numeric value. If the ∞ affordance matters to the user, raise it as a follow-up; do not block this task.

- [ ] **Step 4: Make `_sync_parameter` slider-aware**

In `_sync_parameter` (`_widget.py:313-330`), the `else: control.setValue(display_value)` branch already works for sliders (both `QLabeledSlider` and `QLabeledDoubleSlider` expose `setValue`). No change needed — sliders fall through the same branch as the old spinboxes. Verify by reading the method; only `QComboBox`/`QCheckBox` are special-cased, everything else uses `setValue`.

- [ ] **Step 5: Run**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Render numeric workflow parameters as labeled sliders"
```

### Task B3: Sweep for tests/wheel-scroll filters assuming spinbox params

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (`SpinBoxEventFilter` install scan) if needed
- Modify: any test asserting `QSpinBox`/`QDoubleSpinBox` for workflow params

- [ ] **Step 1: Grep for assumptions**

Run:
```bash
grep -rn "QDoubleSpinBox\|QSpinBox\|workflow_parameter_" tests/ napariTFM/widgets/_widget.py
```
Expected findings: (a) `SpinBoxEventFilter` + the `install_filter_on_inputs` scan in `_widget.py` that wheel-guards `QSpinBox/QDoubleSpinBox/QComboBox`; (b) possibly `tests/test_preprocessing_ui_redesign.py` or `test_batch_parameters.py` asserting spinbox types.

- [ ] **Step 2: Extend the wheel-guard to sliders**

The `SpinBoxEventFilter` ignores wheel events on unfocused spin/combo widgets so scrolling the panel doesn't change values. Sliders have the same hazard. In `_widget.py`, update `SpinBoxEventFilter.eventFilter` and the `install_filter_on_inputs` scan to also cover the slider classes:

```python
from superqt import QLabeledDoubleSlider, QLabeledSlider
```
```python
        if (isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox, QLabeledSlider, QLabeledDoubleSlider)) and
                event.type() == event.Wheel):
```
and in `install_filter_on_inputs`:
```python
            for widget in self.window().findChildren(
                (QSpinBox, QDoubleSpinBox, QComboBox, QLabeledSlider, QLabeledDoubleSlider)
            ):
```

- [ ] **Step 3: Fix any spinbox-typed param assertions**

For each test the grep flagged that asserts a *workflow* param control is a spinbox, update it to the slider type (`QLabeledSlider`/`QLabeledDoubleSlider`). Leave batch-widget and ProjectSection (`_GeneralBody`) spinbox assertions ALONE — those are out of scope (see "Deliberately out of scope" below).

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (modulo known flake).

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/
git commit -m "Wheel-guard slider params and fix spinbox-typed assertions"
```

**Phase B manual check (needs napari):** Open each stage's Parameters, confirm sliders render with the value label above the track and ± step buttons, dragging updates the value, and scrolling the panel does NOT change an unfocused slider.

---

# PHASE C — Mounting refactor + calibration de-dup

Removes the "parameters nested as a faux-stage" hack. Body becomes always-visible; the parameter panel is the single collapsible region. Ship last (it reshapes the section the sliders live in).

### Task C1: StageSection — body always visible, params the only collapsible

**Files:**
- Modify: `napariTFM/widgets/_stage_section.py`
- Test: `tests/test_stage_section_nesting.py` (rewrite), `tests/test_stage_section_header.py` (verify)

- [ ] **Step 1: Rewrite the nesting test file into a parameter-region test**

`add_inner_section` is being deleted, so `tests/test_stage_section_nesting.py` (which tests it) must be replaced. Overwrite the file with:

```python
import pytest
from qtpy.QtWidgets import QApplication, QWidget

from napariTFM.widgets._stage_section import StageSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_body_visible_regardless_of_params_toggle(app):
    body = QWidget()
    panel = QWidget()
    section = StageSection("Force Analysis", body, parameter_panel=panel)
    section.show()
    app.processEvents()
    # Stage body (action buttons) is always visible.
    assert body.isVisible()


def test_params_button_toggles_only_the_parameter_panel(app):
    body = QWidget()
    panel = QWidget()
    section = StageSection("Force Analysis", body, parameter_panel=panel)
    section.show()
    app.processEvents()

    assert not panel.isVisible()          # collapsed by default
    section.params_btn.setChecked(True)
    app.processEvents()
    assert panel.isVisible()
    assert body.isVisible()               # body unaffected
    section.params_btn.setChecked(False)
    app.processEvents()
    assert not panel.isVisible()


def test_no_inner_section_api(app):
    # add_inner_section was a faux-stage hack; it is gone.
    assert not hasattr(StageSection("X", QWidget()), "add_inner_section")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_stage_section_nesting.py -q`
Expected: FAIL — `test_body_visible_regardless_of_params_toggle` fails because today the body (`child`) is hidden unless `expanded`, and `add_inner_section` still exists.

- [ ] **Step 3: Refactor `StageSection.__init__` so the body is always visible**

In `_stage_section.py`, change the layout assembly so `self._content` (the child/body) is added **without** being gated by `_set_expanded`, and `params_btn` always controls `_parameter_content`. Concretely:

1. In `__init__`, after building `self._content` and adding widgets, replace the bottom block (currently lines ~121-135) with:

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

2. Change `_create_params_button` so it ALWAYS wires to `_set_parameter_panel_expanded` (delete the `if self.parameter_panel is None` branch):

```python
    def _create_params_button(self):
        button = make_icon_button(
            self,
            "params",
            f"stage_{self._slug}_params_button",
            f"Toggle {self._title} parameters",
            QStyle.SP_FileDialogDetailedView,
        )
        button.setCheckable(True)
        button.toggled.connect(self._set_parameter_panel_expanded)
        return button
```

3. Delete `_set_expanded` (no longer used — body is always visible) and the `expanded` parameter's body-toggling role. Keep the `expanded` kwarg in the signature for call-site compatibility but make it a no-op, OR remove it and update the two call sites (`_widget.py` preprocessing section passes `expanded=True`; `ProjectSection` passes `expanded=True`). Prefer removal — grep first:

Run: `grep -rn "expanded=" napariTFM/widgets/_widget.py napariTFM/widgets/_project_section.py`

Remove `expanded=True`/`expanded=` args at those call sites and drop the parameter from `StageSection.__init__`. (ProjectSection passes a body it wants visible — now always true.)

> Keep `_set_parameter_panel_expanded` as-is; it already shows `_parameter_content` + the panel and flips the arrow.

- [ ] **Step 4: Delete `add_inner_section` and dormant `_find_ancestor_accent`**

Remove the `add_inner_section` method (lines ~238-256) and `_find_ancestor_accent` (lines ~142-156). In `__init__`, the accent resolution currently calls `_find_ancestor_accent`; simplify the accent block to:

```python
        if accent is not None:
            self._accent = accent
        else:
            self._accent = stage_accent(self._slug)
```

- [ ] **Step 5: Run the section tests**

Run: `python -m pytest tests/test_stage_section_nesting.py tests/test_stage_section_header.py tests/test_stage_section_action_sync.py -q`
Expected: PASS. If `test_stage_section_header.py` asserted body-hidden-by-default or `_set_expanded`, update those assertions to the always-visible body model.

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_stage_section.py tests/test_stage_section_nesting.py tests/test_stage_section_header.py
git commit -m "StageSection: body always visible, params panel the sole collapsible"
```

### Task C2: Shell mounts params via `parameter_panel=`; delete the rewire hack

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Update the inline-parameter-panel test**

`tests/test_workflow_shell.py::test_stage_section_params_toggles_inline_parameter_panel_when_provided` exercises the old `parameter_panel` path directly on `StageSection` — that path is now THE path, so the test should still pass after C1. Add a shell-level test that the panel is mounted via the constructor, not `add_inner_section`:

```python
def test_shell_mounts_param_panels_as_section_parameter_panel(monkeypatch, app):
    from napariTFM.widgets import _widget
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(_StubViewer())
    section = widget._stage_sections_by_key["displacement"]
    # The panel is the section's first-class parameter_panel, not a nested section.
    assert section.parameter_panel is widget._stage_parameter_panels_by_key["displacement"]
    assert not hasattr(section, "add_inner_section")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest "tests/test_workflow_shell.py::test_shell_mounts_param_panels_as_section_parameter_panel" -q`
Expected: FAIL (`section.parameter_panel` is None today; panel is mounted via `add_inner_section`).

- [ ] **Step 3: Pass `parameter_panel=` into each StageSection and delete the rewire loop**

In `_widget.py`, in the `_stage_sections_by_key` construction (lines ~466-516), add `parameter_panel=` to the four stages that have a panel. For each, pull from `self._stage_parameter_panels_by_key`:

```python
            "preprocessing": _StageSection(
                "Preprocessing",
                self.preprocessing_widget,
                status_panel=self._stage_status_panels_by_key["preprocessing"],
                parameter_panel=self._stage_parameter_panels_by_key.get("preprocessing"),
                action_targets={
                    "run": self.preprocessing_widget.process_btn,
                    "preview": self.preprocessing_widget.preview_check,
                    "cancel": self.preprocessing_widget.cancel_btn,
                },
            ),
```

Apply the same `parameter_panel=self._stage_parameter_panels_by_key.get("<key>")` addition to `displacement`, `force`, and `stress`. `batch` has no panel — leave it without `parameter_panel`. Remove the now-unused `expanded=True` from the preprocessing section (per Task C1 Step 3).

Then **delete the entire `add_inner_section` mounting loop** (lines ~519-538, the block that builds `_stage_inner_param_sections_by_key`, calls `add_inner_section`, and does `section.params_btn.toggled.disconnect()` / reconnect). Also delete the now-unused `self._stage_inner_param_sections_by_key = {}` line.

- [ ] **Step 4: Confirm the hack is gone**

Run: `grep -n "add_inner_section\|toggled.disconnect\|_stage_inner_param_sections_by_key" napariTFM/widgets/_widget.py`
Expected: no matches.

- [ ] **Step 5: Run shell tests + full suite**

Run: `python -m pytest tests/test_workflow_shell.py -q && python -m pytest -q`
Expected: PASS (modulo known flake). Verify the real diff: `git diff -w -- napariTFM/widgets/_widget.py` shows only the intended changes (watch for CRLF churn).

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Mount stage params via parameter_panel; remove faux-stage rewire hack"
```

### Task C3: De-duplicate calibration (pixel/frame in one place)

**Files:**
- Modify: `napariTFM/widgets/_widget.py` (`_create_stage_parameter_panels`)
- Test: `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing test**

`pixel_size`/`frame_interval` currently appear both in `ProjectSection._GeneralBody` AND in the preprocessing stage panel (because it's built with `("General", "Preprocessing")`). Calibration should live only in the Project section. Append:

```python
def test_preprocessing_panel_excludes_general_calibration(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel
    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Preprocessing",))
    assert "pixel_size" not in panel.parameter_controls
    assert "frame_interval" not in panel.parameter_controls
    # Preprocessing-specific params still present.
    assert "rolling_ball_radius" in panel.parameter_controls
```

- [ ] **Step 2: Run to verify it fails**

This test builds the panel directly with `("Preprocessing",)`, so it passes already — the *bug* is that the shell builds preprocessing with `("General", "Preprocessing")`. Add a shell-level assertion instead:

```python
def test_shell_preprocessing_panel_has_no_calibration_controls(monkeypatch, app):
    from napariTFM.widgets import _widget
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(_StubViewer())
    panel = widget._stage_parameter_panels_by_key["preprocessing"]
    assert "pixel_size" not in panel.parameter_controls
    assert "frame_interval" not in panel.parameter_controls
```

Run: `python -m pytest "tests/test_workflow_shell.py::test_shell_preprocessing_panel_has_no_calibration_controls" -q`
Expected: FAIL (calibration controls present in the preprocessing panel).

- [ ] **Step 3: Drop "General" from the preprocessing stage panel**

In `_widget.py`, `_create_stage_parameter_panels` (lines ~556-567), change the preprocessing entry from `("General", "Preprocessing")` to `("Preprocessing",)`:

```python
        stage_sections = {
            "preprocessing": ("Preprocessing",),
            "displacement": ("Displacement",),
            "force": ("Force",),
            "stress": ("Stress",),
        }
```

Calibration (`pixel_size`, `frame_interval`) now lives solely in `ProjectSection._GeneralBody`, which already edits them and stays synced via `ParameterManager.parameter_changed`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/test_workflow_shell.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "De-duplicate calibration: pixel/frame live only in Project section"
```

### Task C4: Full-suite verification + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: PASS. Re-run the napari_compatibility test in isolation if it segfaults: `python -m pytest tests/test_napari_compatibility.py -q`.

- [ ] **Step 2: CRLF churn audit**

Run: `git diff -w HEAD~6..HEAD -- napariTFM/widgets/` (adjust range to this phase's commits)
Expected: the whitespace-ignoring diff equals the real intent; no phantom line-ending flips. If a file churned, `git checkout <parent> -- <file>`, re-apply the surgical change, recommit.

- [ ] **Step 3: Manual smoke (needs napari — owner runs this)**

Launch the plugin and confirm, per stage:
- Header shows status dot + ⚙ params + ▶ run + preview; clicking ⚙ toggles ONLY the parameter panel (sliders), body stays visible.
- No nested "Parameters" sub-header with its own toggle (the faux-stage is gone).
- Calibration (pixel size / frame length) appears only in the Project section.
- Theme menu (◐) re-accents everything live.
- Run preprocessing → displacement → force → stress end-to-end; sliders drive the computation correctly.

---

## Deliberately Out of Scope

- **Batch widget parameters** (`batch_analysis_widget.py`) keep their existing spinbox/QGroupBox forms and their separate YAML config — that's a batch-*job* spec, a distinct persistence surface. Not touched.
- **ProjectSection `_GeneralBody`** keeps its two calibration spinboxes (`pixel_size`/`frame_interval`) — converting those to sliders is optional polish, not required for coherence.
- **`stale` status detection** — explicitly trimmed (Task A2), not implemented.
- **Replacing `StageSection` with a thin `CollapsibleSection`** — napariTFM's richer header (status dot + action proxies + status panel) is intentional and stays.
- **`gel_height` ∞ affordance** — lost in the slider port; revisit only if the owner asks.

## Self-Review Notes

- **Spec coverage:** Decision 1 (slider params) → Phase B. Decision 2 (full palette + picker) → Phase A Tasks A1/A5. Decision 3 (trim status) → Task A2. Structural incoherences from the assessment: faux-stage nesting → C1/C2; dead `parameter_panel` path → repurposed as THE path in C1; duplicate calibration → C3; dormant `_find_ancestor_accent` + `add_inner_section` → deleted in C1.
- **Type consistency:** `dslider`/`islider` defined in B1, consumed in B2/B3. `set_accent` defined in A4, consumed in A5. `parameter_panel` is an existing `StageSection.__init__` kwarg reused in C2.
- **Phase independence:** A and B share no files except `_widget.py` (A adds a footer + imports; B edits `WorkflowParameterPanel`) — sequence A→B→C to avoid merge friction, but each is independently green.
