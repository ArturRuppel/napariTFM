# Viridis UI Redesign — Slice 1: Colormap Accent Ramp + Mono Readouts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the muddy stage-accent look by resolving each stage's accent as an ordered sample along the active theme's perceptual colormap (viridis/cividis/nord/dracula), default the theme to Viridis, and set parameter value readouts in a monospace font — a pure restyle with no behavior change.

**Architecture:** Replace `stage_accent()`'s `ACTIVE_PALETTE[STAGE_ACCENTS[key]]` lookup (which collapses most stages to 2 colors under Cividis) with `_sample_ramp(active_ramp, position)`, where each stage has a fixed `0..1` position and each theme is an ordered list of hex stops. Keep the legacy palette dicts so unrelated imports/tests stay valid. Add a `mono_font()` helper and apply it to the superqt slider's value label in `_param_controls.py`.

**Tech Stack:** Python, qtpy (PyQt6), superqt labeled sliders, `colorsys`, pytest with `QApplication` fixtures.

**Line endings (repo memory `feedback-line-endings`):** `_ui_style.py` and `_param_controls.py` may carry mixed CRLF/LF. Touch only target lines; never normalize. After staging each commit verify `git diff --cached --stat` == `git diff --cached -w --stat`.

**Known env flake:** `tests/test_napari_compatibility.py::test_widget_constructs_with_pyqt6_qtpy_backend` SIGSEGVs intermittently in a subprocess — verify in isolation, don't chase.

---

## File Structure

- **Modify** `napariTFM/widgets/_ui_style.py` — add `THEME_RAMPS`, `STAGE_RAMP_POSITION`, `_sample_ramp()`, `ACTIVE_RAMP`; rewrite `stage_accent()`; switch default theme to `Viridis`; update `set_active_theme()` to also set `ACTIVE_RAMP`; add `mono_font()`.
- **Modify** `napariTFM/widgets/_param_controls.py` — set `mono_font()` on the slider value label in `_stack_label_above`.
- **Modify** `tests/test_ui_style.py` — rewrite the 1 value-coupled accent test; add ramp + mono_font tests.
- **Modify** `tests/test_theme_switching.py` — rewrite the 1 value-coupled accent test.
- **Create** `tests/test_param_controls_mono.py` — lock the mono value label. *(keeps the new assertion isolated from the existing `test_param_controls.py`)*

---

## Task 1: Colormap ramp data + ordinal `stage_accent`

**Files:**
- Modify: `napariTFM/widgets/_ui_style.py`
- Test: `tests/test_ui_style.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ui_style.py` (top-level, near the other accent tests):

```python
def test_stage_accent_samples_active_ramp_in_pipeline_order():
    from napariTFM.widgets import _ui_style
    _ui_style.set_active_theme("Viridis")
    # project/inputs sit at the ramp start; batch at the end.
    assert _ui_style.stage_accent("project") == _ui_style.THEME_RAMPS["Viridis"][0]
    assert _ui_style.stage_accent("batch") == _ui_style.THEME_RAMPS["Viridis"][-1]
    # adjacent pipeline stages are visibly distinct (the anti-mud guarantee).
    order = ["preprocessing", "displacement", "force", "stress"]
    accents = [_ui_style.stage_accent(k) for k in order]
    assert len(set(accents)) == len(accents)


def test_stage_accent_unknown_key_falls_back_to_ramp_start():
    from napariTFM.widgets import _ui_style
    assert _ui_style.stage_accent("nope") == _ui_style.stage_accent("inputs")
```

Replace the existing value-coupled test `test_stage_accent_returns_palette_color_for_known_key` (around line 22) with:

```python
def test_stage_accent_returns_hex_for_known_keys():
    assert stage_accent("preprocessing").startswith("#")
    assert stage_accent("displacement").startswith("#")
    assert stage_accent("preprocessing") != stage_accent("displacement")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_ui_style.py -q`
Expected: FAIL — `THEME_RAMPS` does not exist / accents not yet distinct.

- [ ] **Step 3: Add ramp data + sampling, rewrite `stage_accent`**

In `napariTFM/widgets/_ui_style.py`, after the `THEME_PALETTES` / `ACTIVE_PALETTE` block (around line 53), add:

```python
# ── Ordered perceptual ramps ─────────────────────────────────────────────
# Each theme is an ordered list of hex stops. Stages sample the ACTIVE ramp by
# their pipeline position, so the workflow reads as one colormap sweep instead
# of collapsing to a few muddy palette colors.
THEME_RAMPS = {
    "Viridis": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
    "Cividis": ["#00204d", "#31446b", "#666970", "#958f78", "#cab969", "#ffea46"],
    "Nord":    ["#5e81ac", "#81a1c1", "#8fbcbb", "#a3be8c", "#ebcb8b", "#d08770"],
    "Dracula": ["#6272a4", "#bd93f9", "#8be9fd", "#50fa7b", "#f1fa8c", "#ffb86c"],
}
ACTIVE_RAMP = THEME_RAMPS[ACTIVE_THEME_NAME]

# Stage -> position along the ramp (0 = start, 1 = end). project/inputs anchor
# the start; batch anchors the end; the four pipeline stages spread between.
STAGE_RAMP_POSITION = {
    "inputs": 0.0, "project": 0.0,
    "preprocessing": 0.18, "displacement": 0.40,
    "force": 0.62, "stress": 0.82, "batch": 1.0,
}


def _sample_ramp(ramp: list[str], t: float) -> str:
    """Linear-interpolate a hex color at position t in [0, 1] along an ordered ramp."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    if len(ramp) == 1:
        return ramp[0]
    span = t * (len(ramp) - 1)
    i = int(span)
    if i >= len(ramp) - 1:
        return ramp[-1]
    frac = span - i
    a, b = _hex_to_rgb(ramp[i]), _hex_to_rgb(ramp[i + 1])
    rgb = tuple(round(a[c] + (b[c] - a[c]) * frac) for c in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)
```

Note: `_sample_ramp` calls `_hex_to_rgb`, which is defined later in the file (line ~142). Module-level function bodies are not executed at import, so the forward reference resolves at call time — no reordering needed.

Then change the default theme name (line ~52) from:

```python
ACTIVE_THEME_NAME = "Cividis"
```

to:

```python
ACTIVE_THEME_NAME = "Viridis"
```

And `ACTIVE_PALETTE = THEME_PALETTES[ACTIVE_THEME_NAME]` (line ~53) stays as-is (now resolves to the Viridis palette — harmless legacy data).

Rewrite `stage_accent()` (lines ~121-124) from:

```python
def stage_accent(key: str) -> str:
    """Resolve a stage key to its accent hex via the active palette."""
    semantic = STAGE_ACCENTS.get(key, STAGE_ACCENTS["inputs"])
    return ACTIVE_PALETTE[semantic]
```

to:

```python
def stage_accent(key: str) -> str:
    """Resolve a stage key to its accent by sampling the active colormap ramp."""
    position = STAGE_RAMP_POSITION.get(key, STAGE_RAMP_POSITION["inputs"])
    return _sample_ramp(ACTIVE_RAMP, position)
```

Update `set_active_theme()` (lines ~115-118) to also swap the active ramp:

```python
def set_active_theme(name: str) -> None:
    global ACTIVE_PALETTE, ACTIVE_THEME_NAME, ACTIVE_RAMP
    ACTIVE_THEME_NAME = name
    ACTIVE_PALETTE = THEME_PALETTES[name]
    ACTIVE_RAMP = THEME_RAMPS[name]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_ui_style.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_ui_style.py tests/test_ui_style.py
git commit -m "Sample stage accents along the active colormap ramp (default Viridis)"
```

---

## Task 2: Repair the theme-switching accent test

**Files:**
- Modify: `tests/test_theme_switching.py`

- [ ] **Step 1: Rewrite the value-coupled test**

Replace `test_stage_accent_resolves_through_active_palette` (around line 19) with:

```python
def test_stage_accent_resolves_through_active_ramp():
    _ui_style.set_active_theme("Viridis")
    assert _ui_style.stage_accent("preprocessing") == _ui_style._sample_ramp(
        _ui_style.THEME_RAMPS["Viridis"], _ui_style.STAGE_RAMP_POSITION["preprocessing"]
    )
```

Leave `test_set_active_theme_changes_resolved_accent` as-is unless it fails.

- [ ] **Step 2: Run the file to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_theme_switching.py -q`
Expected: PASS. If `test_set_active_theme_changes_resolved_accent` fails because a coincidental color match, change its comparison theme pair to `("Viridis", "Dracula")` and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_theme_switching.py
git commit -m "Update theme-switching accent test for ramp sampling"
```

---

## Task 3: Monospace value readouts on sliders

**Files:**
- Modify: `napariTFM/widgets/_ui_style.py`
- Modify: `napariTFM/widgets/_param_controls.py`
- Test: `tests/test_param_controls_mono.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_param_controls_mono.py`:

```python
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication

import pytest

from napariTFM.widgets._param_controls import islider


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_slider_value_label_uses_monospace_font(app):
    s = islider(0, 100, 50)
    assert s._label.font().styleHint() == QFont.StyleHint.Monospace
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_param_controls_mono.py -q`
Expected: FAIL — label uses the default proportional font.

- [ ] **Step 3: Add `mono_font()` and apply it**

In `napariTFM/widgets/_ui_style.py`, add `QFont` to the qtpy.QtGui imports (add the import line near the top imports if absent):

```python
from qtpy.QtGui import QFont
```

Then add the helper (next to the other style helpers, e.g. after `caption_style`):

```python
def mono_font() -> QFont:
    """A monospace QFont for physical-value readouts (units stay column-aligned)."""
    font = QFont()
    font.setFamilies(["IBM Plex Mono", "DejaVu Sans Mono", "Menlo", "Consolas", "monospace"])
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font
```

In `napariTFM/widgets/_param_controls.py`, import the helper and apply it inside `_stack_label_above` right after `label = slider._label` (line ~92):

```python
from napariTFM.widgets._ui_style import mono_font
```

```python
    label = slider._label
    label.setFont(mono_font())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_param_controls_mono.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: all pass except the known napari-compat flake (verify that one in isolation).

- [ ] **Step 6: Commit**

```bash
git add napariTFM/widgets/_ui_style.py napariTFM/widgets/_param_controls.py tests/test_param_controls_mono.py
git commit -m "Set parameter value readouts in a monospace font"
```

---

## Subsequent slices (separate plans, written when reached)

These complete the v2 mockup (`docs/ui-redesign-mockup-v2.html`); each gets its own detailed plan:

- **Slice 2 — Gradient spine + status nodes.** New custom-painted left-gutter widget (`paintEvent` + `QLinearGradient`) and per-stage status nodes bound to `StageSection.set_status`; surfaces start following the napari theme here (`napari.utils.theme` + `viewer.events.theme`).
- **Slice 3 — SVG icon set.** Bundle stroked `.svg`s; replace the emoji glyph buttons built by `make_stage_action_button`.
- **Slice 4 — Per-stage enable toggles + "off ≠ missing".** Stage on/off state; status panels and run gating respect the enabled set; disabled stages render dimmed/dashed.
- **Slices 5–7 (own project plan):** experiments-list-at-top, run-all-walks-the-rail with live preview, and aggregate → `.iris` (§5 backend).

---

## Self-Review

- **Spec coverage:** Slice-1 scope = accent ramp (Task 1) + theme default Viridis (Task 1) + mono readouts (Task 3); the theme-following-surfaces piece was deliberately deferred to Slice 2 (surfaces only matter once we paint the spine/nodes) — noted to the owner.
- **Placeholder scan:** none — every step carries concrete code/commands.
- **Type consistency:** `_sample_ramp(list[str], float) -> str`, `THEME_RAMPS`, `STAGE_RAMP_POSITION`, `ACTIVE_RAMP`, `mono_font() -> QFont` are referenced consistently across tasks and tests.
