import json
import logging
from datetime import datetime
from typing import Any

import napari
from qtpy.QtCore import Qt, QObject, QTimer, QSize
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QSizePolicy, QDoubleSpinBox,
    QHBoxLayout, QFrame, QSpinBox, QComboBox, QFileDialog, QCheckBox,
    QMenu, QToolButton, QApplication
)

from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.visualization_manager import VisualizationManager

from napariTFM.widgets.displacement_analysis_widget import DisplacementAnalysisWidget
from napariTFM.widgets.fttc_widget import FTTCWidget
from napariTFM.widgets.stress_widget import StressWidget
from napariTFM.widgets._stage_data_status import DataArtifactSpec, compute_stage_status
from napariTFM.widgets._stage_section import StageSection
from napariTFM.widgets._ui_style import title_style, stage_accent, muted_accent, theme_names, active_theme_name, set_active_theme, section_grid, add_section_header, add_section_pair_row, add_section_labeled_full_row, section_label_style, section_subheader_style, caption_style, TIGHT_SPACING
from napariTFM.widgets._icons import stage_action_icon
from napariTFM.widgets._param_controls import dslider, islider, rslider
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider, QLabeledSlider
from napariTFM.widgets._experiments_list import ExperimentsList, PIPELINE_STAGES
from napariTFM.widgets._run_config import (
    build_run_config,
    build_series_config,
    series_records,
)
from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.backend.ntfm_writer import write_experiment_ntfm
from napariTFM.utilities.batch_output import experiment_ntfm_path, resolve_output_plan
from napariTFM.utilities.viewer_sink import ViewerSink

logger = logging.getLogger(__name__)

PROJECT_FORMAT_VERSION = 2


# Per-stage input/output artifact specs, used only to derive each stage's
# in-memory status (compute_stage_status) for the spine node when no experiment
# is selected. No on_view/on_action callbacks: the old per-artifact status-dot
# row was removed as redundant with the colormap-spine rail.
STAGE_DATA_ARTIFACTS = {
    "displacement": [
        DataArtifactSpec("reference", "Reference", "reference", "input"),
        DataArtifactSpec("bead_stack", "Beads", "bead_stack", "input"),
        DataArtifactSpec("cell_stack", "Cells", "cell_stack", "input", required=False),
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "output"),
    ],
    "force": [
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "input"),
        DataArtifactSpec("force_results", "Traction map", "force_results", "output"),
    ],
    "stress": [
        DataArtifactSpec("force_results", "Traction map", "force_results", "input"),
        DataArtifactSpec("mask_stack", "Mask stack", "mask_stack", "input"),
        DataArtifactSpec("stress_results", "Stress map", "stress_results", "output"),
    ],
}


# Sentinel marking a sub-group heading inside a section's spec list. A spec of
# the form (GROUP, "Advanced") renders a muted sub-header and starts a fresh
# two-per-row run; it is not a parameter control.
GROUP = object()


class WorkflowParameterPanel(QWidget):
    """Single visible parameter editor for the workflow shell."""

    PARAMETER_SECTIONS = [
        ("General", [
            ("pixel_size", "Pixel Size (um)", "float", 0.001, 100.0, 0.1, 3, None),
            ("frame_interval", "Frame Length (min)", "float", 0.001, 1000.0, 0.1, 3, None),
        ]),
        ("Displacement", [
            ("piv_window", "Interrogation Window (px)", "int", 8, 128, 2, 0, None),
            ("piv_passes", "Passes", "int", 1, 12, 1, 0, None),
            ("downscale_factor", "Downscale Factor", "int", 1, 10, 1, 0, None),
            (GROUP, "Visualization"),
            ("disp_vector_stride", "Vector Stride", "int", 1, 100, 1, 0, None),
            ("disp_arrow_scale", "Arrow Scale", "float", 0.1, 50.0, 0.1, 1, None),
            ("d_max", "Max Displacement (um)", "float", 0.1, 200.0, 0.1, 1, None),
        ]),
        ("Force", [
            ("young_modulus", "Young's Modulus (kPa)", "float", 0.1, 1000.0, 0.1, 2, None),
            ("poisson_ratio_substrate", "Poisson Ratio", "float", 0.0, 0.5, 0.01, 2, None),
            ("gel_height", "Gel Height (um)", "float", 0.0, 1000.0, 10.0, 1, None),
            ("lanczos_exp", "Lanczos Exponent", "int", 0, 5, 1, 0, None),
            ("regularization", "Regularization (10^x)", "float", -21.0, 0.0, 0.5, 1, None),
            ("auto_gcv", "Auto-GCV per frame", "bool", None, None, None, None, None),
            (GROUP, "Mask confinement"),
            ("fwd_mask_strength", "Mask Confinement", "float", 0.0, 100.0, 1.0, 0, None),
            ("fwd_smoothness", "Smoothness", "float", 0.0, 1.0, 0.01, 2, None),
            (GROUP, "Visualization"),
            ("force_vector_stride", "Vector Stride", "int", 1, 100, 1, 0, None),
            ("force_arrow_scale", "Arrow Scale", "float", 0.1, 50.0, 0.1, 1, None),
            ("f_max", "Max Force (Pa)", "float", 0.1, 10000.0, 1.0, 1, None),
        ]),
        ("Stress", [
            # BISM (Bayesian, mesh-free): Lambda entered by hand as a base-10
            # exponent, like Force's regularization.
            ("bism_regularization", "Regularization (10^x)", "float", -12.0, 0.0, 0.5, 1, None),
            (GROUP, "Visualization"),
            ("max_stress", "Max Stress (mN/m)", "float", 0.01, 1000.0, 0.1, 2, None),
        ]),
    ]

    # Tooltips keyed by parameter name, applied to both the label and control.
    # The PIV set maps onto napariTFM.backend.piv_displacement (multi-pass FFT
    # cross-correlation, GPU-accelerated when torch + CUDA are available).
    PARAMETER_TOOLTIPS = {
        "fwd_mask_strength": (
            "The master switch for the Force stage. 0 = plain FTTC (regularized "
            "Fourier inversion + Lanczos + GCV). Above 0 = confine traction to the "
            "loaded mask (the same external mask the Stress stage uses) with the "
            "forward solver; higher = the off-mask traction is more strongly "
            "penalized (log-scaled, so every step does something). Needs a mask "
            "loaded — without one it stays on FTTC. The 'Regularization' above is "
            "the Tikhonov λ for both paths."
        ),
        "fwd_smoothness": (
            "Gradient-smoothness on the traction field — the primary regularizer "
            "once confinement is on. Confining forces to the mask, with no smoothness, "
            "lets the in-mask field overfit into artifacts; this term (γ‖∇t‖²) is what "
            "the photometric solver got for free from its coarse basis. Useful band "
            "~0.01..0.3. 0 = off. Inert (greyed) until Mask Confinement > 0."
        ),
        "piv_window": (
            "Final PIV interrogation window, in pixels. Cross-correlation is run "
            "on windows of this size on the last (finest) pass. Smaller windows "
            "resolve finer detail but need denser texture to stay well-posed; "
            "larger windows are more robust to noise but blur small displacements."
        ),
        "piv_passes": (
            "Number of coarse-to-fine PIV passes. Each pass re-warps the moving "
            "image by the running estimate and correlates with a smaller window, "
            "so more passes capture larger displacements and refine subpixel "
            "accuracy at the cost of runtime; gains taper off after a few."
        ),
    }

    def __init__(self, parameter_manager: ParameterManager, section_titles: tuple[str, ...] | None = None):
        super().__init__()
        self.parameter_manager = parameter_manager
        self._section_titles = set(section_titles) if section_titles is not None else None
        self.parameter_controls = {}
        self._setup_ui()
        self._sync_all_controls()
        self._refresh_confinement_enablement()
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)
        self.parameter_manager.parameter_changed.connect(self._refresh_confinement_enablement)

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

            # Lay out specs two-per-row, but a range slider (or a GROUP marker)
            # takes a full row and flushes any scalar still waiting for a partner.
            row = 1
            pending = None  # (label, control, tooltip)

            def flush_pending():
                nonlocal pending, row
                if pending is not None:
                    add_section_pair_row(grid, row, pending[0], pending[1],
                                         left_tooltip=pending[2])
                    row += 1
                    pending = None

            for spec in specs:
                if spec[0] is GROUP:
                    flush_pending()
                    subheader = QLabel(spec[1])
                    subheader.setStyleSheet(section_subheader_style())
                    add_section_header(grid, row, subheader)
                    row += 1
                elif spec[2] == "range":
                    flush_pending()
                    label, control, tooltip = self._control_for_spec(spec)
                    add_section_labeled_full_row(grid, row, label, control, tooltip=tooltip)
                    row += 1
                elif pending is None:
                    pending = self._control_for_spec(spec)
                else:
                    label, control, tooltip = self._control_for_spec(spec)
                    add_section_pair_row(grid, row, pending[0], pending[1], label, control,
                                         left_tooltip=pending[2], right_tooltip=tooltip)
                    row += 1
                    pending = None
            flush_pending()

            layout.addLayout(grid)

        self.setLayout(layout)

    def _control_for_spec(self, spec):
        name, label, kind, min_val, max_val, step, decimals, choices = spec
        control = self._create_control(name, kind, min_val, max_val, step, decimals, choices)
        tooltip = self.PARAMETER_TOOLTIPS.get(name)
        if tooltip:
            control.setToolTip(tooltip)
        return label, control, tooltip

    def _create_control(self, name, kind, min_val, max_val, step, decimals, choices):
        if kind == "range":
            lo_name, hi_name = name
            control = rslider(
                min_val, max_val,
                self.parameter_manager.get_ui_parameter(lo_name),
                self.parameter_manager.get_ui_parameter(hi_name),
                step=step, decimals=decimals,
            )
            control._range_params = (lo_name, hi_name)
            control.valueChanged.connect(
                lambda values, lo=lo_name, hi=hi_name: (
                    self.parameter_manager.set_ui_parameter(lo, values[0]),
                    self.parameter_manager.set_ui_parameter(hi, values[1]),
                )
            )
            control.setObjectName(f"workflow_parameter_{lo_name}")
            # Both names point at this one control so external changes to either
            # bound re-sync the slider (see _sync_parameter).
            self.parameter_controls[lo_name] = control
            self.parameter_controls[hi_name] = control
            return control
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
            # bool(state) rather than `state == Qt.Checked`: stateChanged emits a
            # plain int (0/2), and under PyQt6/PySide6 comparing it to the
            # Qt.CheckState enum silently yields False, leaving the box stuck off.
            control.stateChanged.connect(
                lambda state, n=name: self.parameter_manager.set_ui_parameter(n, bool(state))
            )
        else:
            raise ValueError(f"Unsupported parameter control type: {kind}")

        control.setObjectName(f"workflow_parameter_{name}")
        self.parameter_controls[name] = control
        return control

    def _refresh_confinement_enablement(self, name=None, value=None):
        """Grey out the forward-only Smoothness knob unless Mask Confinement > 0.

        With confinement at 0 the Force stage runs plain FTTC, which never reads
        Smoothness — so it would be a dead control. Driven from parameter_changed
        (any param → cheap no-op when it isn't the confinement dial) and once at
        construction for the initial state.
        """
        if name is not None and name != "fwd_mask_strength":
            return
        control = self.parameter_controls.get("fwd_smoothness")
        if control is None:
            return
        control.setEnabled(self.parameter_manager.get_parameter("fwd_mask_strength") > 0)

    def _sync_all_controls(self):
        for name in self.parameter_controls:
            self._sync_parameter(name, self.parameter_manager.get_ui_parameter(name))

    def _sync_parameter(self, param_name: str, value: Any):
        control = self.parameter_controls.get(param_name)
        if control is None:
            return

        display_value = self.parameter_manager.get_ui_parameter(param_name)
        control.blockSignals(True)
        try:
            if isinstance(control, QLabeledDoubleRangeSlider):
                lo_name, hi_name = control._range_params
                control.setValue((
                    self.parameter_manager.get_ui_parameter(lo_name),
                    self.parameter_manager.get_ui_parameter(hi_name),
                ))
            elif isinstance(control, QComboBox):
                index = control.findText(str(display_value), Qt.MatchFixedString)
                if index >= 0:
                    control.setCurrentIndex(index)
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(display_value))
            else:
                control.setValue(display_value)
        finally:
            control.blockSignals(False)


class SpinBoxEventFilter(QObject):
    def eventFilter(self, obj, event):
        # Check for all spinnable input widgets
        if (isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox, QLabeledSlider, QLabeledDoubleSlider)) and
                event.type() == event.Wheel):
            if not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)


class napariTFMWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self._applying_state = False
        self.viewer = napari_viewer

        # Create and install event filter
        self.spinbox_filter = SpinBoxEventFilter(self)

        # Find and filter all spinboxes in the application
        def install_filter_on_inputs():
            for widget in self.window().findChildren(
                (QSpinBox, QDoubleSpinBox, QComboBox, QLabeledSlider, QLabeledDoubleSlider)
            ):
                widget.installEventFilter(self.spinbox_filter)
                widget.setFocusPolicy(Qt.StrongFocus)

        # Install filters after a short delay to ensure all widgets are created.
        # Import QTimer locally (not the module-level name) so singleShot uses the
        # real Qt timer even when tests monkeypatch the module-level QTimer with a
        # fake poll-timer stand-in.
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, install_filter_on_inputs)

        # Give the dock a comfortable default/minimum width for the panel body.
        self.setMinimumWidth(400)

        # Create scroll area for widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create container widget for scroll area
        container = QWidget()
        container_layout = QVBoxLayout()
        # Right margin keeps content clear of the scroll area's vertical
        # scrollbar instead of butting right up against it.
        container_layout.setContentsMargins(0, 0, 8, 0)
        container.setLayout(container_layout)

        # Brand row + the Project/Parameters toolbar (the front door), all in
        # one row now: icon-only buttons, right-aligned opposite the title,
        # grouped under inline "Project" / "Parameters" captions (the caption
        # disambiguates the group; load/save share an icon across groups on
        # purpose — direction of the icon's arrow disambiguates load vs save).
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("napariTFM")
        title.setStyleSheet(title_style())
        title_row.addWidget(title)
        title_row.addStretch()

        self.new_project_btn = self._make_toolbar_button("new", "Start a new project")
        self.load_project_btn = self._make_toolbar_button("load", "Load a project")
        self.save_project_btn = self._make_toolbar_button("save", "Save project as…")
        self.load_params_btn = self._make_toolbar_button("load", "Load parameters preset")
        self.save_params_btn = self._make_toolbar_button("save", "Save parameters preset")
        self.reset_params_btn = self._make_toolbar_button("reset", "Reset parameters")

        title_row.addWidget(self._toolbar_caption("Project"))
        for button in (self.new_project_btn, self.load_project_btn, self.save_project_btn):
            title_row.addWidget(button)
        title_row.addWidget(self._toolbar_divider())
        title_row.addWidget(self._toolbar_caption("Parameters"))
        for button in (self.load_params_btn, self.save_params_btn, self.reset_params_btn):
            title_row.addWidget(button)

        self._toolbar_icon_buttons = [
            (self.new_project_btn, "new"),
            (self.load_project_btn, "load"),
            (self.save_project_btn, "save"),
            (self.load_params_btn, "load"),
            (self.save_params_btn, "save"),
            (self.reset_params_btn, "reset"),
        ]

        container_layout.addLayout(title_row)

        # Progressive-disclosure gate (G0/G1/G2). No project is open at launch.
        self._project_open = False
        self._dirty = False

        # Initialize managers
        self.data_manager = DataManager()
        self.parameter_manager = ParameterManager()
        self.visualization_manager = VisualizationManager(self.viewer, self.data_manager)

        self.experiments_list = ExperimentsList(
            status_fn=self._experiment_stage_status,
            parameter_manager=self.parameter_manager,
            data_manager=self.data_manager,
        )
        self.experiments_list.active_changed.connect(
            self._on_active_experiment_changed
        )
        self.experiments_list.experiments_changed.connect(
            self._on_experiments_changed
        )
        self.experiments_list.run_selected_requested.connect(self._run_selected_experiments)
        self.experiments_list.cancel_run_selected_requested.connect(self._cancel_run_selected)
        self.experiments_list.pool_requested.connect(self._on_pool_requested)
        self.experiments_list.stage_load_requested.connect(self._on_row_stage_clicked)
        self._active_batch = None
        container_layout.addWidget(self.experiments_list)

        self._active_experiment: str | None = None
        # ntfm-backed stages whose arrays are currently decoded into DataManager
        # + the viewer for the active experiment. Loading is display-only and
        self._pipeline_context_label = QLabel("Pipeline")
        self._pipeline_context_label.setStyleSheet(section_label_style())
        container_layout.addWidget(self._pipeline_context_label)

        # One panel-level status line (P2) — replaces the per-stage labels/bars.
        # Stages report run/skip/done into it, prefixed with the reporting stage.
        self.status_label = QLabel("")
        self.status_label.setObjectName("global_status_label")
        self.status_label.setWordWrap(True)
        container_layout.addWidget(self.status_label)

        self._stage_parameter_panels_by_key = self._create_stage_parameter_panels()

        # Parameter preset toolbar (recipe import/export).
        self.save_params_btn.clicked.connect(self._save_params)
        self.load_params_btn.clicked.connect(self._load_params)
        self.reset_params_btn.clicked.connect(self._reset_parameters)
        # Project toolbar (the front door): handlers defined below.
        self.new_project_btn.clicked.connect(self._new_project)
        self.load_project_btn.clicked.connect(self._load_project)
        self.save_project_btn.clicked.connect(self._save_project)

        # Initialize all widgets with parameter_manager
        self.displacement_widget = DisplacementAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        self.force_widget = FTTCWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        self.stress_widget = StressWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        # Funnel every pipeline stage's progress into the single global status
        # label (P2). Run-selected (the retired batch widget's successor) reports via
        # its own per-folder callback, not a controller progress signal.
        for stage_widget, stage_label in (
            (self.displacement_widget, "Displacement"),
            (self.force_widget, "Force"),
            (self.stress_widget, "Stress"),
        ):
            stage_widget.controller.progress_updated.connect(
                lambda progress, message, label=stage_label: self._relay_stage_status(
                    label, message, progress
                )
            )

        self._stage_sections_by_key = {
            "displacement": StageSection(
                "Displacement",
                self.displacement_widget,
                parameter_panel=self._stage_parameter_panels_by_key.get("displacement"),
                actions={
                    "run": self.displacement_widget.run_action,
                    "preview": self.displacement_widget.preview_action,
                    "cancel": self.displacement_widget.cancel_action,
                },
                action_states=self.displacement_widget.action_states,
                action_states_changed=self.displacement_widget.action_states_changed,
            ),
            "force": StageSection(
                "Force Analysis",
                self.force_widget,
                parameter_panel=self._stage_parameter_panels_by_key.get("force"),
                actions={
                    "run": self.force_widget.run_action,
                    "preview": self.force_widget.preview_action,
                    "cancel": self.force_widget.cancel_action,
                },
                action_states=self.force_widget.action_states,
                action_states_changed=self.force_widget.action_states_changed,
                extra_actions=[
                    {
                        "key": "gcv",
                        "icon": "gcv",
                        "tooltip": "Auto-select regularization (GCV)",
                        "handler": self.force_widget.gcv_action,
                    }
                ],
            ),
            "stress": StageSection(
                "Stress Analysis",
                self.stress_widget,
                parameter_panel=self._stage_parameter_panels_by_key.get("stress"),
                actions={
                    "run": self.stress_widget.run_action,
                    "preview": self.stress_widget.preview_action,
                    "cancel": self.stress_widget.cancel_action,
                },
                action_states=self.stress_widget.action_states,
                action_states_changed=self.stress_widget.action_states_changed,
                optional=True,
                # Stress needs an external mask, so it stays off until the user
                # opts in (D1).
                enabled=False,
            ),
        }
        self._stage_sections = list(self._stage_sections_by_key.values())
        self._apply_stage_accents()
        for key, section in self._stage_sections_by_key.items():
            if section.enable_btn is not None:
                section.enabled_changed.connect(
                    lambda _enabled, k=key: self._on_stage_enabled_changed(k)
                )
            section.spine.clicked.connect(lambda k=key: self._on_stage_node_clicked(k))

        # While a stage's controller has the UI frozen for a long-running op, flip
        # that stage's pill into 'running' so the header's run/cancel button shows
        # the Cancel control (the cancel handler is always wired). On unfreeze,
        # re-read disk truth so the dots settle back to done/ready/off.
        self._freeze_widgets_by_key = {
            "displacement": self.displacement_widget,
            "force": self.force_widget,
            "stress": self.stress_widget,
        }
        for key, stage_widget in self._freeze_widgets_by_key.items():
            stage_widget.controller.ui_frozen.connect(
                lambda frozen, k=key: self._on_stage_freeze(k, frozen)
            )

        container_layout.setSpacing(0)
        for section in self._stage_sections:
            container_layout.addWidget(section)
        container_layout.addStretch()

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Add scroll area to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self._setup_theme_selector(main_layout)
        self.setLayout(main_layout)

        self.connect_signals()
        self.data_manager.add_change_callback(self.refresh)
        self.refresh_stage_statuses()
        self._update_disclosure()

    # Title-row icons: larger and thinner-lined than the stage-header icons,
    # and drawn with the document-set's sharp square caps / miter joins.
    _TOOLBAR_ICON_SIZE = 22
    _TOOLBAR_ICON_STROKE = 1.5
    _TOOLBAR_ICON_CAP = "square"
    _TOOLBAR_ICON_JOIN = "miter"

    def _make_toolbar_button(self, icon_name: str, tooltip: str) -> QToolButton:
        """A compact, icon-only, auto-raised button for the title-row toolbar."""
        button = QToolButton()
        button.setIcon(
            stage_action_icon(
                icon_name,
                muted_accent(stage_accent("project")),
                size=self._TOOLBAR_ICON_SIZE,
                stroke_width=self._TOOLBAR_ICON_STROKE,
                linecap=self._TOOLBAR_ICON_CAP,
                linejoin=self._TOOLBAR_ICON_JOIN,
            )
        )
        button.setIconSize(QSize(self._TOOLBAR_ICON_SIZE, self._TOOLBAR_ICON_SIZE))
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    @staticmethod
    def _toolbar_divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        return divider

    @staticmethod
    def _toolbar_caption(text: str) -> QLabel:
        """A small muted group label preceding a cluster of toolbar buttons."""
        caption = QLabel(text)
        caption.setStyleSheet(caption_style())
        return caption

    def _confirm_discard(self) -> bool:
        """True if it's safe to clear the workspace (clean, no project open, or user said yes)."""
        if not self._dirty or not self._project_open:
            return True
        reply = QMessageBox.question(
            self,
            "Discard changes?",
            "This project has unsaved changes. Discard them?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _new_project(self) -> None:
        """Clear to an empty, open workspace (G1): no rows, default knobs."""
        if not self._confirm_discard():
            return
        self._applying_state = True
        try:
            self.parameter_manager.reset_all_parameters()
            self.experiments_list.set_records([])
            for key, section in self._stage_sections_by_key.items():
                if section.enable_btn is not None:
                    section.set_enabled(False)  # stress is off by default (D1)
            self.data_manager.set_output_dir(None)
        finally:
            self._applying_state = False
        self._project_open = True
        self._dirty = False
        self.refresh()
        self._update_disclosure()

    def _save_project(self) -> None:
        """Save the whole project — dataset + recipe — to one YAML (Save-as)."""
        import yaml

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save project", "project.yaml", "YAML Files (*.yaml *.yml)"
            )
            if not file_path:
                return
            if not file_path.lower().endswith((".yml", ".yaml")):
                file_path += ".yaml"
            config = build_series_config(
                self.experiments_list.experiment_records(),
                disabled_stages=self._disabled_stages(),
                processed_root=self.data_manager.output_dir,
            )
            config["format_version"] = PROJECT_FORMAT_VERSION
            config["parameters"] = self.parameter_manager.get_all_parameters()
            with open(file_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            self._dirty = False
            QMessageBox.information(self, "Success", "Project saved!")
        except Exception as e:
            logger.error(f"Error saving project: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def _load_project(self) -> None:
        """Load a project bundle: dataset + run options + analysis parameters."""
        if not self._confirm_discard():
            return
        import yaml

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load project", "", "YAML Files (*.yaml *.yml)"
            )
            if not file_path:
                return
            with open(file_path) as f:
                config = yaml.safe_load(f) or {}
            self._applying_state = True
            try:
                self._apply_parameters(config.get("parameters", {}) or {})
                run_options = config.get("run_options", {}) or {}
                disabled = set(run_options.get("disabled_stages") or [])
                for key, section in self._stage_sections_by_key.items():
                    if section.enable_btn is not None:
                        section.set_enabled(key not in disabled)
                self.experiments_list.set_records(series_records(config))
            finally:
                self._applying_state = False
            self._project_open = True
            self._dirty = False
            self.refresh()
            self._update_disclosure()
            QMessageBox.information(self, "Success", "Project loaded!")
        except Exception as e:
            logger.error(f"Error loading project: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

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
        self._apply_stage_accents()
        self._sync_theme_menu_state()

    def _apply_stage_accents(self):
        """Accent each stage by its pipeline key (not its title slug), then
        blend neighbours into the spine. Titles like "Force Analysis" don't
        slugify to a ramp key, so keying off the dict is what keeps the rail a
        real gradient instead of collapsing to the fallback colour."""
        for key, section in self._stage_sections_by_key.items():
            section.set_accent(stage_accent(key))
        self._apply_spine_neighbours()
        toolbar_accent = muted_accent(stage_accent("project"))
        for button, icon_name in self._toolbar_icon_buttons:
            button.setIcon(
                stage_action_icon(
                    icon_name,
                    toolbar_accent,
                    size=self._TOOLBAR_ICON_SIZE,
                    stroke_width=self._TOOLBAR_ICON_STROKE,
                    linecap=self._TOOLBAR_ICON_CAP,
                    linejoin=self._TOOLBAR_ICON_JOIN,
                )
            )

    def _apply_spine_neighbours(self):
        """Give each stage's spine its neighbours' accents so the rail blends."""
        sections = self._stage_sections
        for i, section in enumerate(sections):
            above = sections[i - 1]._accent if i > 0 else section._accent
            below = sections[i + 1]._accent if i < len(sections) - 1 else section._accent
            section.set_accents(section._accent, above=above, below=below)

    def _sync_theme_menu_state(self):
        current = active_theme_name()
        for name, action in self._theme_actions.items():
            action.setChecked(name == current)
        self.theme_btn.setToolTip(f"Theme: {current}")

    def _create_stage_parameter_panels(self) -> dict[str, WorkflowParameterPanel]:
        """Create inline workflow parameter editors grouped by pipeline stage."""
        stage_sections = {
            "displacement": ("Displacement",),
            "force": ("Force",),
            "stress": ("Stress",),
        }
        return {
            key: WorkflowParameterPanel(self.parameter_manager, section_titles=titles)
            for key, titles in stage_sections.items()
        }

    def _stage_widgets(self):
        return [
            self.displacement_widget,
            self.force_widget,
            self.stress_widget,
        ]

    def refresh(self):
        """Single reconcile pass: update every stage widget, then statuses."""
        for widget in self._stage_widgets():
            update = getattr(widget, "_update_ui_state", None)
            if callable(update):
                update()
        self.refresh_stage_statuses()

    def _update_disclosure(self) -> None:
        """Reveal only what the current state earns (G0/G1/G2 gated reveal).

        G0 (no project): only the header toolbars + the empty-state hint show.
        G1 (project open, no row selected): the experiments workspace + the
        shared status line. G2 (a row selected): additionally the pipeline
        context label and the four stage pills.
        """
        project_open = self._project_open
        tuning = project_open and self.experiments_list.active() is not None

        self.experiments_list.setVisible(project_open)
        self.status_label.setVisible(project_open)

        self._pipeline_context_label.setVisible(tuning)
        for section in self._stage_sections:
            section.setVisible(tuning)

    def _relay_stage_status(self, stage_label: str, message: str, progress: int = 0) -> None:
        """Render a stage's progress message in the one global status label (P2).

        Interactive runs also echo to the console so live mode prints the same
        progress batch does (worklist §1): batch redirects stdout through its
        TeeLogger and ``print()``s timestamped lines, while live stages report
        only via Qt signals. Mirroring the message to stdout here — in batch's
        ``[timestamp] message`` format — funnels every interactive stage through
        the same console path without disturbing the UI label or the batch
        run-log file. This path never runs under the TeeLogger (run-all reports
        via ``_on_batch_progress``), so there is no double-timestamping.

        ``progress`` (0-100, as emitted by ``progress_updated``) also drives the
        stage's spine node so a running stage's rail fill grows frame by frame
        instead of just sitting on a flat amber dot — a no-op while the stage
        isn't "running" (the spine itself ignores progress then).
        """
        self.status_label.setText(f"{stage_label} — {message}")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {stage_label} — {message}")
        section = self._stage_sections_by_key.get(stage_label.lower())
        if section is not None:
            section.set_progress(progress / 100.0)

    def _run_selected_experiments(self) -> None:
        """Run the selected rows through the pipeline, walking the rail (P4).

        The experiments list is the single run source: its records + the shared
        parameters build the run config, and stress is skipped when its on/off
        glyph is off (D1). A per-folder progress callback drives the live
        mini-rails as the batch advances.

        Only the currently-selected rows are run: the records are filtered down
        to ``ExperimentsList.selected_rows()`` (row order) before building the
        config. The button is disabled when nothing is selected, so this is a
        no-op guard rather than the normal path.

        ``num_workers`` (read once here from the experiments-list spinbox)
        decides which of two code paths runs: ``<= 1`` keeps the original
        synchronous, single-process, live-streaming path unchanged; ``> 1``
        hands the folders to a process pool and polls it from a Qt timer
        instead (no live viewer streaming in that mode -- see
        :meth:`_run_selected_experiments_parallel`).
        """
        selected = set(self.experiments_list.selected_rows())
        records = [
            record
            for record in self.experiments_list.experiment_records()
            if record["path"] in selected
        ]
        if not records:
            return
        num_workers = self.experiments_list.num_workers()
        config = build_run_config(
            records,
            self.parameter_manager.get_all_parameters(),
            disabled_stages=self._disabled_stages(),
            processed_root=self.data_manager.output_dir,
            num_workers=num_workers,
        )
        if num_workers > 1:
            self._run_selected_experiments_parallel(config)
        else:
            self._run_selected_experiments_sequential(config)

    def _run_selected_experiments_sequential(self, config: dict) -> None:
        """Original single-process run path (unchanged behaviour).

        Streams each stage into the live viewer as it runs (worklist §5): the
        same ``BatchAnalysis`` that drives a headless run also walks the rail
        in napari via a ``ViewerSink``. Frames repaint live because the batch
        runs synchronously on the GUI thread and the sink pumps the event loop.
        """
        sink = ViewerSink(
            self.data_manager,
            self.visualization_manager,
            pump=QApplication.processEvents,
            on_experiment=self._on_experiment_streaming,
            on_stage_progress=self._on_run_selected_stage_progress,
        )
        analyzer = BatchAnalysis(
            config,
            progress_callback=self._on_batch_progress,
            sink=sink,
        )
        self._active_batch = analyzer
        self.experiments_list.set_run_selected_active(True)
        # The sink takes over layer visibility per stage while it streams
        # (worklist §4); snapshot now so end_run restores it however the run ends.
        sink.begin_run()
        try:
            analyzer.process_all_folders()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Run-selected failed")
            QMessageBox.critical(self, "Run selected", f"Batch run failed: {exc}")
        finally:
            sink.end_run()
            self._active_batch = None
            self.experiments_list.set_run_selected_active(False)
            self.refresh_stage_statuses()

    def _run_selected_experiments_parallel(self, config: dict) -> None:
        """Process-pool run path: submit once, poll non-blockingly from a timer.

        No ``ViewerSink`` is constructed here -- nothing streams into one in
        parallel mode (workers run headless in separate processes), so
        constructing one would sit idle or half-initialise state expecting
        frames that never come. The per-stage progress rail simply does not
        update live during a parallel run; it catches up via
        ``refresh_stage_statuses`` whenever the followed position completes,
        or at the very end. This is an accepted, deliberate trade-off.

        The viewer follows the currently-selected row, or the topmost folder
        if nothing is selected yet. Cancellation reuses the existing
        ``_cancel_run_selected`` wiring unchanged -- it calls
        ``self._active_batch.request_cancel()``, and ``self._active_batch`` is
        the same analyzer instance used here.
        """
        analyzer = BatchAnalysis(config, progress_callback=self._on_batch_progress, sink=None)
        self._active_batch = analyzer
        self.experiments_list.set_run_selected_active(True)

        try:
            plan = resolve_output_plan(config['root_folders'], config.get('processed_root'))
            for warning in plan.warnings:
                print(f"WARNING: {warning}")

            # Viewer follows the selected row, else the topmost folder (locked
            # product decision) -- only set it if nothing is already selected,
            # so an existing selection is respected.
            if self._active_experiment is None:
                self._active_experiment = config['root_folders'][0]
                self.experiments_list.follow_streaming(self._active_experiment)

            analyzer.start_parallel(plan, config['num_workers'])
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Run-selected failed")
            QMessageBox.critical(self, "Run selected", f"Batch run failed: {exc}")
            self._active_batch = None
            self.experiments_list.set_run_selected_active(False)
            self.refresh_stage_statuses()
            return

        timer = QTimer(self)
        timer.setInterval(150)

        def _poll():
            events, stage_events, finished = analyzer.poll_parallel_progress()
            for folder, stage, status, fraction in stage_events:
                self._on_batch_stage_progress(folder, stage, status, fraction)
            for folder, status in events:
                # self._active_experiment may have changed since the run
                # started (the user can click a different, already-finished
                # row at any time -- that goes through the existing,
                # unchanged active_changed -> _on_active_experiment_changed
                # path and is not this method's concern). Compare against the
                # CURRENT value, not a value captured once at run start, so
                # "follow" tracks whatever the user is looking at right now.
                if status == "done" and folder == self._active_experiment:
                    # The user is following this run, so a finished folder's
                    # results are decoded into the viewer for them (the live
                    # "follow" feature — distinct from manual selection, which
                    # loads no output until a circle is clicked). Status is
                    # already eager; this just brings the pixels on screen.
                    self._load_stage_results(folder, self._NTFM_STAGES)
                    self.refresh_stage_statuses()
            if finished:
                timer.stop()
                self._active_batch = None
                self.experiments_list.set_run_selected_active(False)
                self.refresh_stage_statuses()

        timer.timeout.connect(_poll)
        self._parallel_run_timer = timer  # keep a reference so it isn't garbage-collected
        timer.start()

    def _on_stage_persisted(self, stage_key: str) -> None:
        """A stage finished interactively: persist it, then reconcile the dots.

        Interactive runs are no longer preview-only — a finished stage writes the
        same ``.ntfm`` a batch run would, at the same canonical path, so the row
        dots (top) and the section dots (below) both read the one on-disk truth.
        """
        self._persist_active_experiment(stage_key)
        self.refresh()

    def _persist_active_experiment(self, stage_key: str) -> None:
        """Write the active experiment's results to disk.

        For displacement / force / stress: gathers results from memory and
        writes them through the shared ``.ntfm`` writer.

        A no-op when nothing is selected or the relevant arrays are absent.
        """
        path = self._active_experiment
        if not path:
            return

        try:
            labels = {}
            for record in self.experiments_list.experiment_records():
                if record.get("path") == path:
                    labels = dict(record.get("columns", {}))
                    break
            ntfm_path = experiment_ntfm_path(path, self.data_manager.output_dir)
            write_experiment_ntfm(
                ntfm_path,
                parameters=self.parameter_manager.get_all_parameters(),
                displacement_result=self.data_manager.displacement_results,
                force_result=self.data_manager.force_results,
                stress_result=self.data_manager.stress_results,
                mask=self.data_manager.mask_stack,
                folder=path,
                input_files=self.experiments_list.input_files_for(path),
                labels=labels,
            )
        except Exception as exc:
            logger.exception("Could not persist results for %s", path)
            self.status_label.setText(f"Save failed: {exc}")

    def _cancel_run_selected(self) -> None:
        """Ask the live batch to stop at the next folder boundary (P4 / item 1).

        The batch runs synchronously on the GUI thread; the per-folder progress
        callback pumps the event loop (``processEvents``), so this click is
        delivered between folders and the cooperative flag halts the next one.
        """
        if self._active_batch is not None:
            self._active_batch.request_cancel()
            self.status_label.setText("Batch — cancelling…")

    def _on_experiment_streaming(self, path: str) -> None:
        """Make the UI follow the position a live run is now streaming (§3).

        The sink calls this as the batch enters each folder. We move the
        experiments-list row highlight and the active-experiment pointer to that
        position so the list tracks the rail, and reveal the pipeline context so
        the label names the position being processed. We deliberately do **not**
        go through ``set_active`` / ``_on_active_experiment_changed``: that path
        clears in-memory results and reloads from disk, which would fight the
        very frames the sink is streaming into the viewer.
        """
        from pathlib import Path

        self._active_experiment = path or None
        self.experiments_list.follow_streaming(path)
        if self._active_experiment:
            self._pipeline_context_label.setText(
                f"Pipeline · running ▸ {Path(self._active_experiment).name}"
            )
        self._update_disclosure()

    def _on_run_selected_stage_progress(self, stage: str, status: str, fraction: float | None) -> None:
        """Walk a run's progress onto the matching stage pill's spine node.

        Mirrors the live single-stage path (``_relay_stage_status``) so a
        run's rail fills frame by frame too, instead of sitting static
        until ``refresh_stage_statuses`` reconciles everything at the very end.
        """
        section = self._stage_sections_by_key.get(stage)
        if section is None:
            return
        if section.status != status:
            section.set_status(status)
        if fraction is not None:
            section.set_progress(fraction)

    def _on_batch_stage_progress(
        self, folder: str, stage: str, status: str, fraction: float | None
    ) -> None:
        """Route one parallel-mode worker's real per-stage progress onto its row.

        The parallel poll timer's per-tick stage events land here and go
        straight to that one folder's one mini-rail dot -- the parallel-mode
        sibling of ``_on_run_selected_stage_progress``, which does the equivalent
        for the single-experiment detail panel's ``StageSpine`` during a
        sequential run.
        """
        self.experiments_list.set_row_stage_progress(folder, stage, status, fraction)

    def _on_batch_progress(self, folder: str, status: str) -> None:
        """Live per-folder feedback for Run-selected: walk the rail, then refresh (P4).

        'done'/'error' name the one folder that just changed, so read *that*
        folder's real `.ntfm` status directly instead of rescanning every row
        in the list (the bulk `refresh_statuses` path is input-only/cheap and
        would otherwise paint this just-finished folder back to 'not_started').
        """
        from pathlib import Path

        name = Path(folder).name
        if status == "running":
            self.experiments_list.mark_running(folder)
            self.status_label.setText(f"Batch — running {name}")
        elif status == "done":
            self.experiments_list.apply_row_statuses(folder, self._experiment_stage_status(folder))
            self.status_label.setText(f"Batch — finished {name}")
        elif status == "error":
            self.experiments_list.apply_row_statuses(folder, self._experiment_stage_status(folder))
            self.status_label.setText(f"Batch — failed {name}")
        elif status == "cancelled":
            self.experiments_list.refresh_statuses()
            self.status_label.setText("Batch — cancelled")
        # Keep the rail repainting between folders during the in-process run.
        QApplication.processEvents()

    def _on_stage_freeze(self, key: str, frozen: bool) -> None:
        """Surface the Cancel control while a stage runs, then re-read disk truth.

        A frozen controller means a stage (run *or* preview) is in flight; pinning
        the pill to 'running' is what flips the header's run/cancel button to the
        wired Cancel. On unfreeze we refresh from disk so the dots settle.
        """
        section = self._stage_sections_by_key.get(key)
        if section is None:
            return
        if frozen:
            section.set_status("running")
        else:
            self.refresh_stage_statuses()

    def refresh_stage_statuses(self):
        # Each stage's spine node mirrors the same persisted-.ntfm truth as the
        # experiments-list row dots (top), so the two can never disagree for the
        # active experiment — including treating an all-NaN stage (e.g. a failed
        # force) as not-done in both. Status is eager: reading which measures a
        # `.ntfm` carries is a cheap header read (no pixel decode), so every
        # stage's node shows the on-disk truth without waiting for a click. With
        # no experiment selected there is no disk truth, so the in-memory verdict
        # (computed from the data manager) stands.
        disk_status = (
            self._experiment_stage_status(self._active_experiment)
            if self._active_experiment
            else None
        )
        for key, section in self._stage_sections_by_key.items():
            memory_status = compute_stage_status(
                self.data_manager, STAGE_DATA_ARTIFACTS[key]
            )
            status = disk_status.get(key, memory_status) if disk_status else memory_status
            section.set_status(status)
        self.experiments_list.refresh_statuses()
        self._refresh_aggregate_readiness()

    def _aggregate_ntfm_paths(self):
        """Resolve every committed experiment's ``.ntfm`` path (folder → container).

        Returns a list of ``(folder, ntfm_path)`` in table order. The folder name
        is the human label used in readiness/skip messages.
        """
        from napariTFM.utilities.batch_output import experiment_ntfm_path

        out = []
        for record in self.experiments_list.experiment_records():
            folder = record["path"]
            out.append(
                (folder, experiment_ntfm_path(folder, self.data_manager.output_dir))
            )
        return out

    def _refresh_aggregate_readiness(self) -> None:
        """Recompute how many experiments can pool and push it to the section.

        A header-only walk (``aggregate.partition_ready`` reads OME-XML series
        names, no pixel decode), so it is cheap to run on every status refresh.
        """
        from pathlib import Path

        from napariTFM.backend import aggregate

        pairs = self._aggregate_ntfm_paths()
        paths = [ntfm_path for _folder, ntfm_path in pairs]
        ready, skipped = aggregate.partition_ready(paths)
        # Map each container path back to its folder basename for the message.
        name_by_path = {ntfm_path: Path(folder).name for folder, ntfm_path in pairs}
        skipped_named = [
            (name_by_path.get(p, Path(p).parent.name), reason) for p, reason in skipped
        ]
        self.experiments_list.set_aggregate_readiness(
            len(ready), len(pairs), skipped_named
        )

    def _on_pool_requested(self) -> None:
        """Pool every ready experiment's ``.ntfm`` into one tidy summary table.

        Runs synchronously on the GUI thread (as the sequential batch does): the
        aggregator only reads + reduces containers — no compute — so it finishes
        quickly relative to a run. Writes ``summary.csv`` + ``provenance.json`` +
        ``schema.json`` into the ``TFM_aggregate`` bucket and reports the outcome.
        """
        from pathlib import Path

        from napariTFM.backend import aggregate
        from napariTFM.utilities.batch_output import aggregate_output_dir

        pairs = self._aggregate_ntfm_paths()
        if not pairs:
            return
        paths = [ntfm_path for _folder, ntfm_path in pairs]
        out_dir = aggregate_output_dir(
            [folder for folder, _ in pairs], self.data_manager.output_dir
        )

        self.experiments_list.set_pool_active(True)
        QApplication.processEvents()
        try:
            result = aggregate.pool_experiments(paths, out_dir)
        except ValueError as exc:
            # The one expected, actionable failure: colliding experiment ids.
            self.experiments_list.set_pool_active(False)
            self.experiments_list.set_aggregate_result(str(exc), status="error")
            self._refresh_aggregate_readiness()
            QMessageBox.warning(self, "Pool experiments", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Pool failed")
            self.experiments_list.set_pool_active(False)
            self.experiments_list.set_aggregate_result(f"Pooling failed: {exc}", status="error")
            self._refresh_aggregate_readiness()
            QMessageBox.critical(self, "Pool experiments", f"Pooling failed: {exc}")
            return

        self.experiments_list.set_pool_active(False)
        if result.summary_path is None:
            msg = "No experiments were ready to pool (need force + mask)."
            self.experiments_list.set_aggregate_result(msg, status=None)
        else:
            skipped_note = (
                f"; skipped {len(result.skipped)} "
                f"({', '.join(Path(p).parent.name for p, _ in result.skipped)})"
                if result.skipped
                else ""
            )
            self.experiments_list.set_aggregate_result(
                f"Pooled {result.n_rows} rows → {result.summary_path}{skipped_note}",
                status="done",
            )
        self._refresh_aggregate_readiness()

    def _on_stage_node_clicked(self, key: str) -> None:
        """Decode one stage's series into the viewer (display-only, on demand).

        Clicking a stage's spine circle is what pulls that stage's arrays out of
        the active experiment's `TFMresults.ome.tif` and streams them to the
        viewer — nothing else is loaded. Purely for display: calculations always
        re-read from disk, so no prerequisite stage needs to be resident. Status
        dots are already eager, so no status change is needed here.
        """
        if self._active_experiment:
            self._load_stage_for_display(self._active_experiment, key)

    def _on_row_stage_clicked(self, path: str, stage: str) -> None:
        """A list row's stage dot was clicked: show that experiment's stage.

        Selects the row if it isn't already active (loading its inputs, same as
        any selection), then decodes just the clicked stage's series into the
        viewer. Display-only, one series — the same action as clicking the
        matching spine circle below, reachable straight from the list.
        """
        if path != self._active_experiment:
            self.experiments_list.set_active(path)
        self._load_stage_for_display(path, stage)

    def _load_stage_for_display(self, path: str, stage: str) -> None:
        """Load one stage's output for viewing, narrating it in the status line.

        A circle click reads and decodes an OME-TIFF series on the GUI thread,
        which can take a moment, so it announces "Loading …" (forcing that text
        to paint before the blocking read) and then reports whether the stage
        actually had output to show — a click on a stage with no output is
        otherwise a silent no-op.
        """
        label = stage.capitalize()
        self.status_label.setText(f"Loading {label}…")
        self.status_label.repaint()
        self._load_stage_results(path, [stage])
        # Report by what actually landed in the data manager, not merely that a
        # container existed: a container can hold force but no displacement, and
        # a click on the empty stage must say so rather than claim a load.
        shown = getattr(self.data_manager, f"{stage}_results", None) is not None
        self.status_label.setText(
            f"{label} loaded" if shown else f"No {label} output to show yet"
        )

    def _on_stage_enabled_changed(self, key: str) -> None:
        self.refresh_stage_statuses()

    def _disabled_stages(self) -> list[str]:
        return [
            key
            for key, section in self._stage_sections_by_key.items()
            if not section.is_enabled
        ]

    def _experiment_stage_status(self, path: str) -> dict[str, str]:
        """Per-output stage status for an experiment folder (P3).

        Reads the experiment's `.ntfm` for which measures actually carry data
        (`displacement`/`force`/`stress`), so a stage is 'done' only when *its*
        output is present — not merely because a container exists. Each stage's
        immediate predecessor being done makes it 'ready' (the single run-next
        frontier); upstream of that is 'not_started'. Disabled stages read
        'off' (stress is exempt from auto-skip per D1).

        This is eager and runs for every discovered row: reading which measures
        a `.ntfm` carries is a header-only walk of the OME-TIFF (no pixel decode)
        and is cached by `ntfm.populated_measures`, so painting real dots the
        moment folders land in the list is cheap. Decoding a stage's pixels into
        the viewer is the separate, click-driven step (`_load_stage_results`).
        """
        from pathlib import Path

        from napariTFM.utilities import ntfm as _ntfm
        from napariTFM.utilities.batch_output import RESULTS_FILENAME, experiment_output_dir

        folder = Path(path)
        # Resolve the .ntfm exactly where the batch writes it (and where an
        # interactive run persists it): the shared resolve_output_plan bucket.
        # Reading any other path is how the row dots silently went stale.
        out_dir = experiment_output_dir(path, self.data_manager.output_dir)
        ntfm_path = out_dir / RESULTS_FILENAME
        measures = _ntfm.populated_measures(ntfm_path)
        # Inputs live in the experiment folder under their discovery names.
        input_files = self.experiments_list.input_files_for(path) or {}
        beads_name = input_files.get("beads", "beads.tif")
        reference_name = input_files.get("reference", "reference.tif")
        inputs_ready = (folder / beads_name).exists() and (
            folder / reference_name
        ).exists()

        disabled = set(self._disabled_stages())

        def _status(stage: str) -> str:
            if stage in disabled:
                return "off"
            if stage in measures:
                return "done"
            # 'ready' when this stage's immediate input is available. Displacement
            # is the first stage now, so its input is the raw beads/reference.
            ready_when = {
                "displacement": inputs_ready,
                "force": "displacement" in measures,
                "stress": "force" in measures,
            }
            return "ready" if ready_when.get(stage, False) else "not_started"

        return {stage: _status(stage) for stage in PIPELINE_STAGES}

    # ntfm-backed pipeline stages, in dependency order. Every stage's persisted
    # output is a tidy-table measure in the `.ntfm`, so this is also the full
    # pipeline.
    _NTFM_STAGES = ("displacement", "force", "stress")

    def _read_stage_arrays(self, path: str):
        """Read the active experiment's `.ntfm` once, for on-demand stage loads.

        Returns ``None`` if there's no persisted output yet, or it's
        unreadable. Called on a circle click to bring a stage's pixels into the
        viewer — display-only, so calculations never depend on it being resident.
        """
        import types

        import numpy as np

        from napariTFM.utilities import ntfm as _ntfm
        from napariTFM.utilities.batch_output import RESULTS_FILENAME, experiment_output_dir

        try:
            ntfm_path = (
                experiment_output_dir(path, self.data_manager.output_dir)
                / RESULTS_FILENAME
            )
            if not ntfm_path.exists():
                return None

            # Array-native read: dense stage arrays plus the grid spacing / frame
            # interval recovered from the OME metadata — no tidy table on this
            # display-load path.
            arrays, container_grid_spacing, container_frame_interval, metadata = (
                _ntfm.read_series_ntfm(ntfm_path)
            )

            # Recover physical_scale from stored config (UnifiedParameters asdict).
            config = metadata.get("config", {})

            # Reconstruct each stage's parameter dataclass from the stored config
            # so the viewer's visualize_* methods (which read
            # parameters.downscale_factor, d_max/f_max/max_stress,
            # *_vector_stride, *_arrow_scale) work on a freshly-selected
            # experiment without a re-run. Rebuild UnifiedParameters exactly as
            # config_from_parameters does (filter to valid field names so unknown
            # keys are ignored and missing keys default), then derive per stage.
            # A malformed/empty config falls back to UnifiedParameters defaults.
            from dataclasses import fields as _fields

            from napariTFM.backend.parameter_dataclasses import UnifiedParameters

            try:
                _valid = {f.name for f in _fields(UnifiedParameters)}
                unified = UnifiedParameters(
                    **{k: v for k, v in (config or {}).items() if k in _valid}
                )
            except Exception:
                unified = UnifiedParameters()
            disp_params = unified.to_displacement_parameters()
            force_params = unified.to_fttc_parameters()
            stress_params = unified.to_stress_parameters()

            pixel_size = float(config.get("pixel_size", 0.0))
            downscale_factor = float(config.get("downscale_factor", 0))
            frame_interval = float(config.get("frame_interval", 0.0))

            # Prefer the config's pixel_size * downscale; fall back to the grid
            # spacing / frame interval the container carries in its OME metadata.
            if pixel_size > 0 and downscale_factor > 0:
                grid_spacing = pixel_size * downscale_factor
            else:
                grid_spacing = float(container_grid_spacing)
            if frame_interval <= 0:
                frame_interval = float(container_frame_interval)

            physical_scale = {
                "pixel_size": pixel_size,
                "grid_spacing": grid_spacing,
                "time_interval": frame_interval,
                "grid_spacing_units": "µm",
                "time_interval_units": "min",
            }

            return types.SimpleNamespace(
                arrays=arrays,
                physical_scale=physical_scale,
                disp_params=disp_params,
                force_params=force_params,
                stress_params=stress_params,
            )
        except Exception:
            logger.exception("Failed to load results from .ntfm for %s", path)
            return None

    @staticmethod
    def _array_present(arr) -> bool:
        """True when an array has at least one non-NaN value."""
        import numpy as np

        return arr is not None and not np.all(np.isnan(arr))

    def _apply_displacement_result(self, data) -> None:
        import types

        disp = data.arrays.get("displacement_field")
        if not self._array_present(disp):
            return
        result = types.SimpleNamespace(
            displacement_field=disp,
            physical_scale=data.physical_scale,
            original_shape=disp.shape[1:3],
            displacement_field_shape=disp.shape[1:3],
            parameters=data.disp_params,
        )
        self.data_manager.set_displacement_results(result, dirty=False)
        self.visualization_manager.begin_vector_field_stream(
            'displacement', disp.shape[0],
            {
                'v_max': data.disp_params.d_max,
                'vector_stride': data.disp_params.disp_vector_stride,
                'arrow_scale': data.disp_params.disp_arrow_scale,
                'downscale_factor': data.disp_params.downscale_factor,
            },
        )
        for frame_index in range(disp.shape[0]):
            self.visualization_manager.stream_vector_field_frame(
                'displacement', frame_index, disp[frame_index]
            )

    def _apply_force_result(self, data) -> None:
        import types

        force = data.arrays.get("force_field")
        if not self._array_present(force):
            return
        result = types.SimpleNamespace(
            force_field=force,
            physical_scale=data.physical_scale,
            original_shape=force.shape[1:3],
            force_shape=force.shape[1:3],
            parameters=data.force_params,
        )
        self.data_manager.set_force_results(result, dirty=False)
        self.visualization_manager.begin_vector_field_stream(
            'force', force.shape[0],
            {
                'v_max': data.force_params.f_max,
                'vector_stride': data.force_params.force_vector_stride,
                'arrow_scale': data.force_params.force_arrow_scale,
                'downscale_factor': data.force_params.downscale_factor,
            },
        )
        for frame_index in range(force.shape[0]):
            self.visualization_manager.stream_vector_field_frame(
                'force', frame_index, force[frame_index]
            )

    def _apply_stress_result(self, data) -> None:
        import types

        stress = data.arrays.get("stress_tensor")
        if not self._array_present(stress):
            return
        result = types.SimpleNamespace(
            stress_tensor=stress,
            physical_scale=data.physical_scale,
            original_shape=stress.shape[1:3],
            stress_shape=stress.shape[1:3],
            parameters=data.stress_params,
        )
        self.data_manager.set_stress_results(result, dirty=False)
        # Stress visualization upscales by the force grid's downscale factor.
        # Read it from the parsed file params, not `data_manager.force_results`,
        # so stress displays correctly even when its circle is clicked on its
        # own (force never loaded into memory) — the two share one config.
        stress_downscale = getattr(data.force_params, "downscale_factor", 1)
        self.visualization_manager.begin_stress_stream(
            num_frames=stress.shape[0],
            max_stress=data.stress_params.max_stress,
            downscale_factor=stress_downscale,
        )
        for frame_index in range(stress.shape[0]):
            self.visualization_manager.stream_stress_frame(
                frame_index, stress[frame_index]
            )

    def _load_stage_results(self, path: str, stages) -> list:
        """Decode the requested stages' persisted output into the viewer.

        Returns the list of stage keys actually loaded (used by callers/tests to
        confirm what was decoded).

        Reads the ntfm-backed table once (see `_read_stage_arrays`) but applies
        only the requested stages, so clicking one stage's circle never also
        streams a stage nobody asked to see. Display-only — calculations always
        re-read from disk, so no prerequisite stage is pulled in.
        """
        loaded = []
        ntfm_stages = [s for s in self._NTFM_STAGES if s in stages]
        if ntfm_stages:
            data = self._read_stage_arrays(path)
            if data is not None:
                # Applied in dependency order (displacement → force → stress) so
                # that if a caller does ask for several at once,
                # set_displacement_results' downstream-invalidation never clears
                # a stage this same call just set.
                if "displacement" in ntfm_stages:
                    self._apply_displacement_result(data)
                if "force" in ntfm_stages:
                    self._apply_force_result(data)
                if "stress" in ntfm_stages:
                    self._apply_stress_result(data)
                loaded.extend(ntfm_stages)
        return loaded

    def _on_active_experiment_changed(self, path: str) -> None:
        self._active_experiment = path or None
        # Switching experiments drops the previous one's in-memory results so they
        # can never be persisted into the newly selected experiment's .ntfm. The
        # dots fall back to the new experiment's on-disk truth (which may already
        # read "done" from a prior batch/interactive run).
        self.data_manager.clear_generated_results()
        if self._active_experiment is None:
            self._pipeline_context_label.setText("Pipeline")
            self.data_manager.set_active_inputs(None, {})
        else:
            from pathlib import Path

            self._pipeline_context_label.setText(
                f"Pipeline · tuning ▸ {Path(self._active_experiment).name}"
            )
            # Point the raw-input disk check at the selected experiment so the
            # displacement input dots read green from its discovery files.
            input_files = self.experiments_list.input_files_for(self._active_experiment)
            self.data_manager.set_active_inputs(self._active_experiment, input_files)
            # And actually load those files from disk into memory + the viewer, so
            # Preview and Run (which need the arrays loaded) work on selection.
            # Displacement is the first stage and now owns raw-input loading.
            self.displacement_widget.load_input_files(self._active_experiment, input_files)
            # No output series is decoded on selection: the row/section dots
            # already show the on-disk status eagerly, and a stage's pixels
            # only stream in when its circle is clicked (calculations re-read
            # from disk regardless, so nothing must be resident to run).
            # The mask is an external Stress input; load the discovered masks.tif
            # from disk into memory too, so Stress Run/Preview enable on selection
            # the same way beads/reference do — no manual layer load required.
            mask_name = input_files.get("masks")
            if mask_name:
                # The mask is stored on the downsampled force grid; pass the bead
                # image's xy size (read cheaply from disk — the bead arrays stream
                # in asynchronously and aren't in memory yet) so the mask's
                # visualization layer is scaled to fit the beads in the viewer.
                beads_shape = self.displacement_widget.peek_input_xy_shape(
                    self._active_experiment, input_files, "beads"
                )
                self.stress_widget.load_mask_from_file(
                    Path(self._active_experiment) / mask_name, beads_shape=beads_shape
                )
        self._update_disclosure()

    def _on_experiments_changed(self) -> None:
        if not self._applying_state:
            self._dirty = True
        # Keep the pool-readiness count in step with the table (commit/delete):
        # a header-only walk, so cheap to run whenever the row set changes.
        self._refresh_aggregate_readiness()

    def _reset_parameters(self):
        """Reset parameters to default values and notify all widgets."""
        try:
            reply = QMessageBox.question(
                self,
                "Confirm",
                "Are you sure you want to reset all parameters to default values?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Reset parameters
                self.parameter_manager.reset_all_parameters()
                self.refresh()

        except Exception as e:
            logger.error(f"Error resetting parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to reset parameters: {str(e)}")

    def _apply_parameters(self, params) -> None:
        """Apply a name→value dict of analysis knobs onto the shared manager.

        Unknown names are skipped (forward/backward-compatible files) and
        ``registration_mode`` is normalised.
        """
        if not isinstance(params, dict):
            return
        valid = set(self.parameter_manager.get_all_parameters())
        for name, value in params.items():
            if name not in valid:
                continue
            try:
                if name == "registration_mode" and isinstance(value, str):
                    value = value.lower()
                self.parameter_manager.set_parameter(name, value)
            except Exception as exc:
                logger.warning("Skipped parameter %s: %s", name, exc)

    def _save_params(self):
        """Save the analysis knobs as a portable ``tfm_params`` preset (no paths).

        The preset is the reusable "how to analyze" recipe; the dataset it gets
        applied to lives in the separate experiment-series file.
        """
        import yaml

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save parameters preset",
                "tfm_params.yaml",
                "YAML Files (*.yaml *.yml)",
            )
            if not file_path:
                return
            if not file_path.lower().endswith((".yml", ".yaml")):
                file_path += ".yaml"
            preset = {
                "format_version": 1,
                "parameters": self.parameter_manager.get_all_parameters(),
            }
            with open(file_path, "w") as f:
                yaml.safe_dump(preset, f, default_flow_style=False)
            QMessageBox.information(self, "Success", "Parameters preset saved!")
        except Exception as e:
            logger.error(f"Error saving parameters preset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {str(e)}")

    def _load_params(self):
        """Load a ``tfm_params`` preset and apply it to every stage's knobs."""
        import yaml

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load parameters preset",
                "",
                "YAML Files (*.yaml *.yml)",
            )
            if not file_path:
                return
            with open(file_path) as f:
                preset = yaml.safe_load(f) or {}
            # Accept the versioned {"parameters": {...}} shape and, for old
            # files, a bare flat parameter dict.
            params = preset.get("parameters") if isinstance(preset, dict) else None
            if not isinstance(params, dict):
                params = preset if isinstance(preset, dict) else {}
            self._apply_parameters(params)
            self.refresh()
            QMessageBox.information(self, "Success", "Parameters preset loaded!")
        except Exception as e:
            logger.error(f"Error loading parameters preset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

    def connect_signals(self):
        """Connect signals between components"""
        # A finished stage persists to the active experiment's .ntfm (auto-save),
        # then refreshes so both dot rows reflect the new on-disk truth.
        self.displacement_widget.displacement_calculated.connect(
            lambda *_: self._on_stage_persisted("displacement")
        )
        self.force_widget.force_calculated.connect(
            lambda *_: self._on_stage_persisted("force")
        )
        self.stress_widget.stress_calculated.connect(
            lambda *_: self._on_stage_persisted("stress")
        )

        # Connect parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Propagate parameter edits to the widgets that display them."""
        if not self._applying_state:
            self._dirty = True
        if param_name in ("pixel_size", "frame_interval"):
            # Calibration affects every stage's displayed/derived values.
            for widget in self._stage_widgets():
                update = getattr(widget, "_update_ui_state", None)
                if callable(update):
                    update()
        elif param_name.startswith("force_"):
            self.force_widget._update_ui_state()


