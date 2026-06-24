# UI CellFlow Header Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make napariTFM's `StageSection` header match CellFlow's idiom — glyph "pill" action buttons (`⚙ 🔍 ▷ ▶/■`), an accent-tinted pill title, the file-status panel collapsed behind a 🔍 toggle, and no header status dot.

**Architecture:** Port CellFlow's pill-styling helpers into `_ui_style.py` (adapted to take a **hex accent** so the theme picker keeps working), then rewrite `StageSection` to use glyph `QToolButton`s styled as pills, drop the colored status indicator (keep status as stored state + a `status` property), and wrap the existing always-visible `status_panel` in a header-hidden, collapsed `CollapsibleSection` toggled by a 🔍 button. The header still owns the action buttons (keystone decision preserved); only their rendering and the status-panel placement change.

**Tech Stack:** Python, qtpy (PyQt6 backend), `QToolButton`, the already-ported `CollapsibleSection`, pytest.

---

## Constraints (read before touching any file)

- **Mixed line endings:** Files have mixed CRLF/LF. Touch ONLY the lines you change; never normalize. Before each commit, stage and verify:
  ```bash
  git diff --cached --stat
  git diff --cached -w --stat
  ```
  Counts MUST match (a small difference is acceptable ONLY for a genuine blank-line content change, never line-ending churn). Also confirm CR-byte count is unchanged: `git show HEAD:<file> | grep -c $'\r'` vs the staged version.
- **Commit messages:** NO `Co-Authored-By` trailer.
- **Branch:** Stay on local `master`. Do NOT push.
- Qt-widget tests MUST take the `app` fixture parameter.
- Use TDD. Run the whole suite at the end (`pytest -q`). Known flake (verify in isolation only): `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend`.

---

## Reference: CellFlow's idiom (already studied)

- Action buttons: `tool_btn("▶")` + `stage_header_action_button` → 22px rounded `QToolButton`, translucent accent-tinted background, accent bold glyph, hover/checked/disabled alpha variants (`CellFlow ui_style.py:336`).
- Title: `stage_header_label` → rounded accent-tinted pill background (`CellFlow ui_style.py:374`).
- File status: collapsed `CollapsibleSection` revealed by a 🔍 checkable button (`CellFlow widgets.py:235 make_pipeline_files_header`).
- Glyphs: `⚙` params, `🔍` files, `▷` preview, `▶` run, `■` cancel (while running).
- No header status dot — CellFlow stores status without rendering an indicator.

---

## File Structure

- `napariTFM/widgets/_ui_style.py` — **MODIFY**: add pill helpers + glyph-button factory; repurpose `stage_header_style` to the pill title style. New imports: `QSizePolicy`.
- `tests/test_ui_style.py` — **MODIFY**: rewrite the header-style test; add pill-helper tests.
- `napariTFM/widgets/_stage_section.py` — **REWRITE**: glyph-pill header, drop dot, `status` property, 🔍-toggled status section, pill title, `set_accent` restyles pills.
- `tests/test_stage_section_header.py` — **MODIFY**: add glyph/pill assertions (most existing tests survive).
- `tests/test_workflow_shell.py` — **MODIFY**: rewrite the `status_indicator` assertions to read `section.status`.

No `_widget.py` change is required (shell already calls only `set_status`/`set_accent`, passes `status_panel`/`parameter_panel`). `ProjectSection` keeps working: it has no panels (so 🔍/⚙ hidden) and already hides `run_cancel_btn`/`preview_button`.

---

## Task 1: Pill helpers + glyph-button factory in `_ui_style.py`

**Files:**
- Modify: `napariTFM/widgets/_ui_style.py`
- Test: `tests/test_ui_style.py`

CellFlow's pill helpers key off a `stage_key` and internally call `muted_stage_accent(key)`. napariTFM's accent is a resolved **hex** (the theme picker hands `StageSection` a custom hex), so these adapted versions take a hex and mute it via the existing `muted_accent(hex)`.

- [ ] **Step 1: Write/adjust the failing tests**

In `tests/test_ui_style.py`: the existing `test_stage_header_style_embeds_accent` (around line 60) asserts the raw accent appears. The repurposed `stage_header_style` now returns a pill whose color is the **muted** accent. Replace that test and add new ones. Replace the body of `test_stage_header_style_embeds_accent` with:

```python
def test_stage_header_style_is_accent_pill():
    from napariTFM.widgets._ui_style import muted_accent
    accent = stage_accent("preprocessing")
    style = stage_header_style(accent)
    assert muted_accent(accent) in style
    assert "font-weight: bold" in style
    assert "border-radius" in style
    assert "background-color" in style
```

Append these new tests at the end of the file (the `app` fixture already exists in this file from the grid task):

```python
def test_stage_header_pill_background_is_rgba_of_muted_accent():
    from napariTFM.widgets._ui_style import stage_header_pill_background, muted_accent

    accent = stage_accent("force")
    bg = stage_header_pill_background(accent, alpha=38)
    muted = muted_accent(accent).lstrip("#")
    r, g, b = int(muted[0:2], 16), int(muted[2:4], 16), int(muted[4:6], 16)
    assert bg == f"rgba({r}, {g}, {b}, 38)"


def test_stage_header_disabled_action_color_is_hex():
    from napariTFM.widgets._ui_style import stage_header_disabled_action_color

    out = stage_header_disabled_action_color(stage_accent("displacement"))
    assert out.startswith("#") and len(out) == 7


def test_stage_header_action_button_style_has_state_rules():
    from napariTFM.widgets._ui_style import stage_header_action_button_style

    style = stage_header_action_button_style(stage_accent("preprocessing"))
    assert "QToolButton {" in style
    assert "QToolButton:hover" in style
    assert "QToolButton:checked" in style
    assert "QToolButton:disabled" in style
    assert "border-radius" in style


def test_make_stage_action_button_carries_glyph_and_is_fixed(app):
    from qtpy.QtWidgets import QToolButton
    from napariTFM.widgets._ui_style import make_stage_action_button, STAGE_ACTION_BUTTON_SIZE

    btn = make_stage_action_button(None, "stage_x_run_button", "Run", "▶", stage_accent("force"))
    assert isinstance(btn, QToolButton)
    assert btn.text() == "▶"
    assert btn.objectName() == "stage_x_run_button"
    assert btn.width() == STAGE_ACTION_BUTTON_SIZE
    assert btn.isCheckable() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui_style.py -q`
Expected: FAIL — `ImportError`/`AttributeError` for the new names, and the old `test_stage_header_style_embeds_accent` name is gone.

- [ ] **Step 3: Modify `_ui_style.py`**

(a) Add `QSizePolicy` to the qtpy.QtWidgets import. Change:
```python
from qtpy.QtWidgets import QGridLayout, QLabel, QStyle, QToolButton, QVBoxLayout, QWidget
```
to:
```python
from qtpy.QtWidgets import QGridLayout, QLabel, QSizePolicy, QStyle, QToolButton, QVBoxLayout, QWidget
```

(b) Add a size constant next to `ICON_BUTTON_SIZE = 24`:
```python
STAGE_ACTION_BUTTON_SIZE = 22
```

(c) Add a small hex parser helper near `muted_accent` (after `muted_accent`):
```python
def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.lstrip("#")
    return (
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
    )
```

(d) **Replace** the existing `stage_header_style` function (currently the border-left stripe style) with the pill version, and add the new helpers immediately after it:
```python
def stage_header_style(accent: str) -> str:
    """Accent-tinted pill style for a stage section's header title."""
    return (
        "font-weight: bold; "
        "font-size: 9pt; "
        f"color: {muted_accent(accent)}; "
        f"background-color: {stage_header_pill_background(accent)}; "
        "border-radius: 4px; "
        "padding: 1px 6px;"
    )


def stage_header_pill_background(accent: str, alpha: int = 38) -> str:
    """Translucent rgba background for a stage header pill, from the muted accent."""
    r, g, b = _hex_to_rgb(muted_accent(accent))
    return f"rgba({r}, {g}, {b}, {alpha})"


def stage_header_disabled_action_color(accent: str) -> str:
    """Dimmed glyph color for a disabled stage action button."""
    r, g, b = _hex_to_rgb(muted_accent(accent))
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    s *= 0.55
    l = max(0.0, l * 0.62)
    r_out, g_out, b_out = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(
        round(r_out * 255), round(g_out * 255), round(b_out * 255)
    )


def stage_header_action_button_style(accent: str) -> str:
    """Pill stylesheet for a glyph QToolButton in a stage header."""
    color = muted_accent(accent)
    disabled = stage_header_disabled_action_color(accent)
    return (
        "QToolButton { "
        "font-weight: bold; font-size: 9pt; "
        f"color: {color}; "
        f"background-color: {stage_header_pill_background(accent)}; "
        "border: none; border-radius: 4px; padding: 0; margin: 0; "
        "text-align: center; } "
        "QToolButton:hover { "
        f"background-color: {stage_header_pill_background(accent, alpha=58)}; }} "
        "QToolButton:checked { "
        f"background-color: {stage_header_pill_background(accent, alpha=82)}; }} "
        "QToolButton:disabled { "
        f"color: {disabled}; "
        f"background-color: {stage_header_pill_background(accent, alpha=28)}; }} "
        "QToolButton:disabled:checked { "
        f"color: {disabled}; "
        f"background-color: {stage_header_pill_background(accent, alpha=44)}; }}"
    )


def make_stage_action_button(
    owner, object_name: str, tooltip: str, glyph: str, accent: str, checkable: bool = False
) -> QToolButton:
    """Build a glyph QToolButton styled as a CellFlow-style accent pill."""
    button = QToolButton(owner)
    button.setText(glyph)
    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setCheckable(checkable)
    button.setFixedSize(STAGE_ACTION_BUTTON_SIZE, STAGE_ACTION_BUTTON_SIZE)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    button.setStyleSheet(stage_header_action_button_style(accent))
    return button
```
Note on the f-string braces above: the `}}` sequences in the `:hover`/`:checked`/`:disabled` fragments are literal single `}` characters (escaped because those fragments are f-strings). The `{ ` opening braces in non-f-string fragments are plain literals. Keep them exactly as written.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui_style.py -q`
Expected: PASS (existing + new).

- [ ] **Step 5: CRLF audit + commit**
```bash
git add napariTFM/widgets/_ui_style.py tests/test_ui_style.py
git diff --cached --stat
git diff --cached -w --stat
git commit -m "Add CellFlow stage-header pill helpers + glyph-button factory"
```

---

## Task 2: Rewrite `StageSection` to the glyph-pill header

**Files:**
- Rewrite: `napariTFM/widgets/_stage_section.py`
- Test: `tests/test_stage_section_header.py`, `tests/test_workflow_shell.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage_section_header.py`:
```python
def test_action_buttons_use_glyphs(app):
    section = StageSection("Preprocessing", QWidget())

    assert section.run_cancel_btn.text() == "▶"
    assert section.preview_button.text() == "▷"
    assert section.params_btn.text() == "⚙"


def test_run_cancel_glyph_swaps_on_running(app):
    section = StageSection("Preprocessing", QWidget(), status="ready")
    assert section.run_cancel_btn.text() == "▶"

    section.set_status("running")
    assert section.run_cancel_btn.text() == "■"

    section.set_status("done")
    assert section.run_cancel_btn.text() == "▶"


def test_no_status_indicator_dot(app):
    section = StageSection("Preprocessing", QWidget())
    assert not hasattr(section, "status_indicator")


def test_status_is_readable_via_property(app):
    section = StageSection("Preprocessing", QWidget(), status="ready")
    assert section.status == "ready"
    section.set_status("done")
    assert section.status == "done"


def test_files_button_present_only_with_status_panel(app):
    with_panel = StageSection("Preprocessing", QWidget(), status_panel=QWidget())
    assert with_panel.files_btn.isVisible() is True

    without_panel = StageSection("Preprocessing", QWidget())
    assert without_panel.files_btn.isVisible() is False


def test_files_button_toggles_status_section(app):
    section = StageSection("Preprocessing", QWidget(), status_panel=QWidget())
    assert section._status_section.is_expanded is False

    section.files_btn.setChecked(True)
    assert section._status_section.is_expanded is True
    section.files_btn.setChecked(False)
    assert section._status_section.is_expanded is False
```

In `tests/test_workflow_shell.py`, rewrite the dot-based assertions to read `section.status`:
- `test_stage_section_exposes_status_indicator_with_stable_name` (around line 269): replace its body with:
```python
def test_stage_section_tracks_status(app):
    section = _widget.StageSection("Preprocessing", QWidget(), status="ready")
    assert section.status == "ready"
    section.set_status("done")
    assert section.status == "done"
```
  (Keep whatever import/construction style the surrounding tests use — if the file constructs via `_widget.StageSection` or a direct import, match it. Use the same `QWidget` import already present in the file.)
- `test_stage_section_status_indicator_remains_visible_when_collapsed` (around line 311): DELETE this test — the dot is gone and the body is always visible, so the concept no longer exists.
- Lines ~821, ~828, ~838, ~870, ~876: replace each `section.status_indicator.toolTip() == "Preprocessing status: <x>"` with `section.status == "<x>"`, and each `section.status_indicator.toolTip() != "Preprocessing status: done"` with `section.status != "done"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stage_section_header.py tests/test_workflow_shell.py -q`
Expected: FAIL — glyph/`status`/`files_btn`/`_status_section` attrs don't exist yet; the rewritten shell assertions fail against the old dot implementation.

- [ ] **Step 3: Rewrite `napariTFM/widgets/_stage_section.py`**

Replace the ENTIRE file contents with:
```python
import re
from typing import Callable

from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    make_stage_action_button,
    stage_accent,
    stage_header_action_button_style,
    stage_header_style,
)
from napariTFM.widgets._collapsible_section import CollapsibleSection


class StageSection(QWidget):
    """Workflow stage section with a CellFlow-style glyph-pill header."""

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

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(COMPACT_SPACING)

        self.header_label = QLabel(title)
        self.header_label.setStyleSheet(stage_header_style(self._accent))
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()

        self.files_btn = self._create_glyph_button(
            "files", "🔍", f"Show {title} data", checkable=True
        )
        self.files_btn.toggled.connect(self._set_status_panel_expanded)

        self.params_btn = self._create_glyph_button(
            "params", "⚙", f"Toggle {title} parameters", checkable=True
        )
        self.params_btn.toggled.connect(self._set_parameter_panel_expanded)

        self.preview_button = self._create_glyph_button("preview", "▷", f"Preview {title}")
        preview_handler = self._actions.get("preview")
        if preview_handler is not None:
            self.preview_button.clicked.connect(
                lambda _checked=False, fn=preview_handler: fn()
            )
        self.preview_button.setEnabled(False)

        self.run_cancel_btn = self._create_glyph_button("run_cancel", "▶", f"Run {title}")
        self.run_cancel_btn.clicked.connect(self._on_run_cancel_clicked)
        self.run_cancel_btn.setEnabled(False)

        self._toggle_button = self.params_btn
        self._action_buttons = [
            self.files_btn,
            self.params_btn,
            self.preview_button,
            self.run_cancel_btn,
        ]
        for button in self._action_buttons:
            header_layout.addWidget(button)

        if self.status_panel is not None:
            self._status_section = CollapsibleSection(
                "Data", self.status_panel, expanded=False, accent_color=self._accent
            )
            self._status_section.set_header_visible(False)
        else:
            self._status_section = None

        if self.parameter_panel is not None:
            self._param_section = CollapsibleSection(
                self._title, self.parameter_panel, expanded=False, accent_color=self._accent
            )
            self._param_section.set_header_visible(False)
        else:
            self._param_section = None

        self._content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(COMPACT_SPACING)
        self._content.setLayout(content_layout)
        content_layout.addWidget(child)

        layout.addLayout(header_layout)
        if self._status_section is not None:
            layout.addWidget(self._status_section)
        if self._param_section is not None:
            layout.addWidget(self._param_section)
        layout.addWidget(self._content)

        # Body is always visible; only the params/data sections collapse.
        self._content.setVisible(True)
        self._child.setVisible(True)

        self.set_status(status)

        self.files_btn.setVisible(self.status_panel is not None)
        has_panel = self._param_section is not None
        self.params_btn.setVisible(has_panel)
        self.params_btn.setChecked(parameters_expanded if has_panel else False)
        if has_panel:
            self._set_parameter_panel_expanded(parameters_expanded)

        if action_states_changed is not None:
            action_states_changed.connect(self._refresh_action_states)
        self._refresh_action_states()

    @property
    def _slug(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self._title.lower()).strip("_")
        return slug or "stage"

    @property
    def status(self) -> str:
        return self._status

    def set_status(self, status: str):
        self._status = status
        if status == "running":
            self.run_cancel_btn.setText("■")
            self.run_cancel_btn.setToolTip(f"Cancel {self._title}")
        else:
            self.run_cancel_btn.setText("▶")
            self.run_cancel_btn.setToolTip(f"Run {self._title}")
        self._refresh_action_states()

    def _refresh_action_states(self):
        states = self._action_states() if self._action_states is not None else {}
        running = self._status == "running"
        self.run_cancel_btn.setEnabled(running or states.get("run", False))
        self.preview_button.setEnabled(states.get("preview", False))

    def set_accent(self, accent: str) -> None:
        """Re-accent the header pill + action buttons (used by the theme picker)."""
        self._accent = accent
        self.header_label.setStyleSheet(stage_header_style(accent))
        for button in self._action_buttons:
            button.setStyleSheet(stage_header_action_button_style(accent))
        if self._param_section is not None:
            self._param_section.set_accent_color(accent)
        if self._status_section is not None:
            self._status_section.set_accent_color(accent)

    def _create_glyph_button(self, action: str, glyph: str, tooltip: str, checkable: bool = False):
        return make_stage_action_button(
            self,
            f"stage_{self._slug}_{action}_button",
            tooltip,
            glyph,
            self._accent,
            checkable=checkable,
        )

    def _on_run_cancel_clicked(self):
        key = "cancel" if self._status == "running" else "run"
        handler = self._actions.get(key)
        if handler is not None:
            handler()

    def _set_parameter_panel_expanded(self, expanded: bool):
        if self._param_section is not None:
            self._param_section._toggle.setChecked(expanded)

    def _set_status_panel_expanded(self, expanded: bool):
        if self._status_section is not None:
            self._status_section._toggle.setChecked(expanded)
        self.files_btn.setToolTip(
            f"{'Hide' if expanded else 'Show'} {self._title} data"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stage_section_header.py tests/test_workflow_shell.py tests/test_stage_section_nesting.py tests/test_stage_section_action_sync.py tests/test_project_section.py -q`
Expected: PASS. If `test_workflow_shell.py` still references `status_indicator` anywhere you missed, fix those reads to `section.status` (grep: `grep -n status_indicator tests/test_workflow_shell.py` must return nothing).

- [ ] **Step 5: CRLF audit + commit**
```bash
git add napariTFM/widgets/_stage_section.py tests/test_stage_section_header.py tests/test_workflow_shell.py
git diff --cached --stat
git diff --cached -w --stat
git commit -m "Rewrite StageSection header to CellFlow glyph-pill style"
```

---

## Final Verification

- [ ] **Full suite green**

Run: `pytest -q`
Expected: all pass (the napari-compat flake re-verified in isolation if it trips).

- [ ] **No stale dot references**
```bash
grep -rn "status_indicator\|status_indicator_style\|make_icon_button\|SP_MediaPlay\|SP_FileDialog" napariTFM/widgets/_stage_section.py
```
Expected: no matches (all removed from `StageSection`). `status_indicator_style`/`make_icon_button` may still be defined in `_ui_style.py` — that's fine, leave them (out of scope).

- [ ] **Slice-wide CRLF audit**
```bash
git diff --stat <parent-of-task1>..HEAD
git diff -w --stat <parent-of-task1>..HEAD
```
Counts identical (modulo a documented blank-line change).

- [ ] **Manual smoke (owner-run, requires napari):** Each stage header shows an accent-pill title and pill glyph buttons `🔍 ⚙ ▷ ▶`; 🔍 reveals the (now collapsed) data/status section; ⚙ reveals params; run shows `▶`, flips to `■` while running; run/preview disable until inputs present; the theme picker (◐) re-accents the title pill AND the glyph buttons live; no status dot remains.
