# napariTFM UI Coherence — Roadmap

The CellFlow → napariTFM UI port is **macro-complete** (workflow shell, theming,
status dots, labeled sliders) but **meso/micro-incoherent**: the new shell is a
CellFlow-shaped frame wrapped around the old per-tab stage widgets, which still
carry their own scroll areas, fixed widths, and duplicated action controls. This
roadmap closes that visual/structural gap.

**Orthogonal (not tracked here):** state coherence — one parameter system, one
refresh path, config round-trip — is handled by the existing
`docs/superpowers/plans/2026-05-29-tier4-state-architecture.md`. It touches
*state*, not *layout*, and can land independently.

Execution order below reflects real dependencies, not the order the gaps were
found. Steps 2 and 3 are **coupled** — the section-primitive decision changes
what the control-exposure refactor touches — so they are planned and executed as
one keystone slice.

---

## Step 1 — Remove inner scroll areas + hardcoded width  *(first slice)*

Each stage widget (`preprocessing`, `displacement`, `fttc`, `msm`) builds its
**own** `QScrollArea` with `setFixedWidth(360)` inside the shell's single
resizable scroll area — yielding scroll-in-scroll and a stage body locked to
360px while the rest of the panel reflows to the dock. Drop the inner scroll and
fixed width; let the shell's scroll own layout (CellFlow's model).

- Standalone, low-risk, highest visible payoff.
- Independent of the section-primitive decision — safe to do first.
- Touches: the 4 stage widgets' `_create_content_container`; update
  `tests/test_preprocessing_ui_redesign.py::test_preprocessing_widget_keeps_parameter_content_in_scroll_area`
  (now vestigial — params no longer live in the stage widget).

## Step 2 — Unify on one section primitive  *(keystone, with Step 3)*

`StageSection` (flat, always-visible body, only the param panel collapses) and
CellFlow's `CollapsibleSection` (whole-body collapse + accent inheritance for
nested sections) are different interaction models. Pick one: either port
`CollapsibleSection`'s accent-inheritance + whole-body collapse into
`StageSection`, or adopt `CollapsibleSection` outright. This is the decision the
rest hang off — resolve it before Step 3/4.

## Step 3 — Expose inner-widget controls; retire the proxy machinery

Stage widgets build their action buttons (`process_btn`, `preview_btn`,
`cancel_btn`) then `setVisible(False)`; the header buttons proxy-click them via
`_ActionStateSync`. Each control exists twice, kept in sync by an event-filter
shim. Refactor inner widgets to expose controls to the host (CellFlow aliases
*handlers* upward, builds each control once), retiring `_ActionStateSync` /
`action_targets`. Coupled to Step 2 — the section model determines where the
single control instance lives.

## Step 4 — Port the grid layout vocabulary; rebuild the param panel

CellFlow's `ui_style` has a dense grid family (`section_grid`, `block_grid`,
`add_section_pair_row`, `add_block_button_row`, sweep grids); napariTFM's
`_ui_style` is a ~5-helper subset, so each stage falls back to ad-hoc
QHBox/QVBox/QGroupBox and they don't look uniform. Port the grid helpers and
rebuild `WorkflowParameterPanel` (and the Project px/dt controls) on them so one
control idiom rules. Sits on top of Step 2 (where params live depends on the
section model) — do last.
