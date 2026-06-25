import json
import logging
from typing import Any

import napari
from qtpy.QtCore import Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QSizePolicy, QDoubleSpinBox,
    QHBoxLayout, QSpinBox, QComboBox, QFileDialog, QCheckBox,
    QMenu, QToolButton, QApplication
)

from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.visualization_manager import VisualizationManager

from napariTFM.widgets.preprocessing_widget import PreprocessingWidget
from napariTFM.widgets.displacement_analysis_widget import DisplacementAnalysisWidget
from napariTFM.widgets.fttc_widget import FTTCWidget
from napariTFM.widgets.msm_widget import MSMWidget
from napariTFM.widgets._stage_data_status import DataArtifactSpec
from napariTFM.widgets._stage_file_status import StageFileStatusRow
from napariTFM.widgets._stage_section import StageSection
from napariTFM.widgets._ui_style import title_style, stage_accent, theme_names, active_theme_name, set_active_theme, section_grid, add_section_header, add_section_pair_row, section_label_style, TIGHT_SPACING
from napariTFM.widgets._param_controls import dslider, islider
from superqt import QLabeledDoubleSlider, QLabeledSlider
from napariTFM.widgets._experiments_list import ExperimentsList, PIPELINE_STAGES
from napariTFM.widgets._run_config import (
    build_run_config,
    build_series_config,
    series_records,
)
from napariTFM.backend.batch_analysis import BatchAnalysis

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "napariTFM_config.json"
STATE_VERSION = 1


STAGE_DATA_ARTIFACTS = {
    "preprocessing": [
        DataArtifactSpec("reference", "Reference image", "reference", "input"),
        DataArtifactSpec("bead_stack", "Bead stack", "bead_stack", "input"),
        DataArtifactSpec("cell_stack", "Cell stack", "cell_stack", "input", required=False),
        DataArtifactSpec("preprocessed_reference", "Preprocessed reference", "preprocessed_reference", "output"),
        DataArtifactSpec("preprocessed_bead_stack", "Preprocessed beads", "preprocessed_bead_stack", "output"),
    ],
    "displacement": [
        DataArtifactSpec("preprocessed_reference", "Preprocessed reference", "preprocessed_reference", "input"),
        DataArtifactSpec("preprocessed_bead_stack", "Preprocessed beads", "preprocessed_bead_stack", "input"),
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "output"),
    ],
    "force": [
        DataArtifactSpec("displacement_results", "Displacement field", "displacement_results", "input"),
        DataArtifactSpec("force_results", "Traction map", "force_results", "output"),
    ],
    "stress": [
        DataArtifactSpec("force_results", "Traction map", "force_results", "input"),
        DataArtifactSpec("mask_stack", "Mask stack", "mask_stack", "input", required=False),
        DataArtifactSpec("stress_results", "Stress map", "stress_results", "output"),
    ],
}

# napari layer names each stage paints, toggled by the header viz glyph (P5).
STAGE_LAYER_NAMES = {
    "preprocessing": (
        "Preprocessed Beads",
        "Preprocessed Reference",
        "Preprocessed Cells",
    ),
    "displacement": ("Displacement Vectors", "Displacement Magnitude"),
    "force": ("Force Vectors", "Force Magnitude"),
    "stress": ("Normal Stress XX", "Normal Stress YY", "Average Normal Stress"),
}


def _build_preprocessing_specs(preprocessing_widget, visualization_manager):
    def assign(role: str):
        return lambda: preprocessing_widget.load_active_layer(role)

    def view(key: str):
        def _show():
            show_artifact = getattr(visualization_manager, "show_artifact", None)
            if show_artifact is not None:
                show_artifact(key)
                return
            if key.startswith("preprocessed_") and hasattr(
                visualization_manager, "update_preprocessing_visualization"
            ):
                visualization_manager.update_preprocessing_visualization()

        return _show

    return [
        DataArtifactSpec(
            "reference",
            "Reference image",
            "reference",
            "input",
            on_view=view("reference"),
            on_action=assign("reference"),
        ),
        DataArtifactSpec(
            "bead_stack",
            "Bead stack",
            "bead_stack",
            "input",
            on_view=view("bead_stack"),
            on_action=assign("beads"),
        ),
        DataArtifactSpec(
            "cell_stack",
            "Cell stack",
            "cell_stack",
            "input",
            required=False,
            on_view=view("cell_stack"),
            on_action=assign("cells"),
        ),
        DataArtifactSpec(
            "preprocessed_reference",
            "Preprocessed reference",
            "preprocessed_reference",
            "output",
            on_view=view("preprocessed_reference"),
        ),
        DataArtifactSpec(
            "preprocessed_bead_stack",
            "Preprocessed beads",
            "preprocessed_bead_stack",
            "output",
            on_view=view("preprocessed_bead_stack"),
        ),
    ]


def _build_displacement_specs(displacement_widget):
    def assign(role: str):
        return lambda: displacement_widget.load_active_layer(role)

    return [
        DataArtifactSpec(
            "preprocessed_reference",
            "Preprocessed reference",
            "preprocessed_reference",
            "input",
            on_action=assign("reference"),
        ),
        DataArtifactSpec(
            "preprocessed_bead_stack",
            "Preprocessed beads",
            "preprocessed_bead_stack",
            "input",
            on_action=assign("beads"),
        ),
        DataArtifactSpec(
            "displacement_results",
            "Displacement field",
            "displacement_results",
            "output",
        ),
    ]


def _build_force_specs(force_widget):
    # Results chain in-memory (ROADMAP §4); the upstream input row is status-only.
    return [
        DataArtifactSpec(
            "displacement_results",
            "Displacement field",
            "displacement_results",
            "input",
        ),
        DataArtifactSpec(
            "force_results",
            "Traction map",
            "force_results",
            "output",
        ),
    ]


def _build_stress_specs(stress_widget):
    return [
        DataArtifactSpec(
            "force_results",
            "Traction map",
            "force_results",
            "input",
        ),
        DataArtifactSpec(
            "mask_stack",
            "Mask stack",
            "mask_stack",
            "input",
            required=False,
            # Masks are an external input (ROADMAP §2): loaded from the active layer.
            on_action=lambda: stress_widget.load_result_artifact("mask_stack"),
        ),
        DataArtifactSpec(
            "stress_results",
            "Stress map",
            "stress_results",
            "output",
        ),
    ]


class WorkflowParameterPanel(QWidget):
    """Single visible parameter editor for the workflow shell."""

    PARAMETER_SECTIONS = [
        ("General", [
            ("pixel_size", "Pixel Size (um)", "float", 0.001, 100.0, 0.1, 3, None),
            ("frame_interval", "Frame Length (min)", "float", 0.001, 1000.0, 0.1, 3, None),
        ]),
        ("Preprocessing", [
            ("rolling_ball_radius", "Rolling Ball Radius", "int", 0, 50, 1, 0, None),
            ("min_intensity_percentile", "Min Intensity (%)", "float", 0.0, 100.0, 0.1, 1, None),
            ("max_intensity_percentile", "Max Intensity (%)", "float", 0.0, 100.0, 0.1, 1, None),
            ("gaussian_sigma", "Gaussian Sigma", "float", 0.0, 10.0, 0.1, 1, None),
            ("cell_min_intensity_percentile", "Cell Min Intensity (%)", "float", 0.0, 100.0, 0.1, 1, None),
            ("cell_max_intensity_percentile", "Cell Max Intensity (%)", "float", 0.0, 100.0, 0.1, 1, None),
            ("cell_gaussian_sigma", "Cell Gaussian Sigma", "float", 0.0, 10.0, 0.1, 1, None),
            ("registration_mode", "Registration Mode", "choice", None, None, None, None,
             ["translation", "rigid", "no registration"]),
        ]),
        ("Displacement", [
            ("nscales", "Farneback Levels", "int", 1, 50, 1, 0, None),
            ("inner_iterations", "Farneback Iterations", "int", 1, 50, 1, 0, None),
            ("median_filtering", "Window Size", "int", 1, 51, 2, 0, None),
            ("downscale_factor", "Downscale Factor", "int", 1, 10, 1, 0, None),
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
            ("force_vector_stride", "Vector Stride", "int", 1, 100, 1, 0, None),
            ("force_arrow_scale", "Arrow Scale", "float", 0.1, 50.0, 0.1, 1, None),
            ("f_max", "Max Force (Pa)", "float", 0.1, 10000.0, 1.0, 1, None),
        ]),
        ("Stress", [
            ("density_factor", "Density Factor", "float", 0.005, 0.1, 0.001, 3, None),
            ("mesh_algorithm", "Mesh Algorithm", "choice", None, None, None, None,
             ["Frontal-Del.", "Delaunay", "MeshAdapt", "BAMG", "FD Quads", "Para. Pack"]),
            ("use_optimization", "Mesh Optimization", "bool", None, None, None, None, None),
            ("poisson_ratio_cells", "Poisson Ratio", "float", 0.0, 0.5, 0.01, 2, None),
            ("max_stress", "Max Stress (mN/m)", "float", 0.01, 1000.0, 0.1, 2, None),
        ]),
    ]

    def __init__(self, parameter_manager: ParameterManager, section_titles: tuple[str, ...] | None = None):
        super().__init__()
        self.parameter_manager = parameter_manager
        self._section_titles = set(section_titles) if section_titles is not None else None
        self.parameter_controls = {}
        self._setup_ui()
        self._sync_all_controls()
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)

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
            if isinstance(control, QComboBox):
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

        # Install filters after a short delay to ensure all widgets are created
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, install_filter_on_inputs)

        # Width is determined by the host dock; no fixed width.

        # Create scroll area for widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create container widget for scroll area
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(container_layout)

        # Title row: brand + the parameter-preset toolbar. These act on the
        # analysis knobs only (the portable tfm_params recipe); the experiment
        # series (folders + tags) has its own Open/Save in the list header.
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("napariTFM")
        title.setStyleSheet(title_style())
        title_row.addWidget(title)
        title_row.addStretch()
        self.save_params_btn = self._make_toolbar_button("Save params", "Save parameters preset")
        self.load_params_btn = self._make_toolbar_button("Load params", "Load parameters preset")
        self.reset_params_btn = self._make_toolbar_button("Reset", "Reset parameters")
        for _btn in (self.save_params_btn, self.load_params_btn, self.reset_params_btn):
            title_row.addWidget(_btn)
        container_layout.addLayout(title_row)

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
        self.experiments_list.run_all_requested.connect(self._run_all_experiments)
        container_layout.addWidget(self.experiments_list)

        self._active_experiment: str | None = None
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

        # Wire the title-bar parameter-preset toolbar; the experiment-series
        # Open/Save lives on the list header instead.
        self.save_params_btn.clicked.connect(self._save_params)
        self.load_params_btn.clicked.connect(self._load_params)
        self.reset_params_btn.clicked.connect(self._reset_parameters)
        self.experiments_list.save_series_requested.connect(self._save_series)
        self.experiments_list.load_series_requested.connect(self._load_series)
        self.experiments_list.output_dir_changed.connect(self._reconcile_to_output_dir)

        # Initialize all widgets with parameter_manager
        self.preprocessing_widget = PreprocessingWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager,

        )
        self.preprocessing_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        #
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

        self.msm_widget = MSMWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        # Funnel every pipeline stage's progress into the single global status
        # label (P2). Run-all (the retired batch widget's successor) reports via
        # its own per-folder callback, not a controller progress signal.
        for stage_widget, stage_label in (
            (self.preprocessing_widget, "Preprocessing"),
            (self.displacement_widget, "Displacement"),
            (self.force_widget, "Force"),
            (self.msm_widget, "Stress"),
        ):
            stage_widget.controller.progress_updated.connect(
                lambda _progress, message, label=stage_label: self._relay_stage_status(
                    label, message
                )
            )

        stage_data_artifacts = dict(STAGE_DATA_ARTIFACTS)
        stage_data_artifacts["preprocessing"] = _build_preprocessing_specs(
            self.preprocessing_widget,
            self.visualization_manager,
        )
        stage_data_artifacts["displacement"] = _build_displacement_specs(
            self.displacement_widget,
        )
        stage_data_artifacts["force"] = _build_force_specs(
            self.force_widget,
        )
        stage_data_artifacts["stress"] = _build_stress_specs(
            self.msm_widget,
        )
        self._stage_status_panels_by_key = {
            key: StageFileStatusRow(key, self.data_manager, artifacts)
            for key, artifacts in stage_data_artifacts.items()
        }

        self._stage_sections_by_key = {
            "preprocessing": StageSection(
                "Preprocessing",
                self.preprocessing_widget,
                status_panel=self._stage_status_panels_by_key["preprocessing"],
                parameter_panel=self._stage_parameter_panels_by_key.get("preprocessing"),
                actions={
                    "run": self.preprocessing_widget.run_action,
                    "preview": self.preprocessing_widget.preview_action,
                    "cancel": self.preprocessing_widget.cancel_action,
                },
                action_states=self.preprocessing_widget.action_states,
                action_states_changed=self.preprocessing_widget.action_states_changed,
            ),
            "displacement": StageSection(
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
            "force": StageSection(
                "Force Analysis",
                self.force_widget,
                status_panel=self._stage_status_panels_by_key["force"],
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
                self.msm_widget,
                status_panel=self._stage_status_panels_by_key["stress"],
                parameter_panel=self._stage_parameter_panels_by_key.get("stress"),
                actions={
                    "run": self.msm_widget.run_action,
                    "preview": self.msm_widget.preview_action,
                    "cancel": self.msm_widget.cancel_action,
                },
                action_states=self.msm_widget.action_states,
                action_states_changed=self.msm_widget.action_states_changed,
                extra_actions=[
                    {
                        "key": "mesh",
                        "icon": "mesh",
                        "tooltip": "Preview mesh",
                        "handler": self.msm_widget.mesh_action,
                    }
                ],
                optional=True,
            ),
        }
        self._stage_sections = list(self._stage_sections_by_key.values())
        self._apply_stage_accents()
        for key, section in self._stage_sections_by_key.items():
            if section.enable_btn is not None:
                section.enabled_changed.connect(
                    lambda _enabled, k=key: self._on_stage_enabled_changed(k)
                )
            section.visualization_toggled.connect(
                lambda visible, k=key: self._set_stage_layers_visible(k, visible)
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

    def _make_toolbar_button(self, text: str, tooltip: str) -> QToolButton:
        """A compact auto-raised text button for the title-bar config toolbar."""
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

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
            "preprocessing": ("Preprocessing",),
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
            self.preprocessing_widget,
            self.displacement_widget,
            self.force_widget,
            self.msm_widget,
        ]

    def refresh(self):
        """Single reconcile pass: update every stage widget, then statuses."""
        for widget in self._stage_widgets():
            update = getattr(widget, "_update_ui_state", None)
            if callable(update):
                update()
        self.refresh_stage_statuses()

    def _relay_stage_status(self, stage_label: str, message: str) -> None:
        """Render a stage's progress message in the one global status label (P2)."""
        self.status_label.setText(f"{stage_label} — {message}")

    def _run_all_experiments(self) -> None:
        """Run the whole config table through the pipeline, walking the rail (P4).

        The experiments list is the single run source: its records + the shared
        parameters build the run config, and stress is skipped when its on/off
        glyph is off (D1). A per-folder progress callback drives the live
        mini-rails as the batch advances.
        """
        records = self.experiments_list.experiment_records()
        if not records:
            return
        config = build_run_config(
            records,
            self.parameter_manager.get_all_parameters(),
            disabled_stages=self._disabled_stages(),
        )
        analyzer = BatchAnalysis(config, progress_callback=self._on_batch_progress)
        try:
            analyzer.process_all_folders()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Run-all failed")
            QMessageBox.critical(self, "Run all", f"Batch run failed: {exc}")
        finally:
            self.refresh_stage_statuses()

    def _on_batch_progress(self, folder: str, status: str) -> None:
        """Live per-folder feedback for Run-all: walk the rail, then refresh (P4)."""
        from pathlib import Path

        name = Path(folder).name
        if status == "running":
            self.experiments_list.mark_running(folder)
            self.status_label.setText(f"Batch — running {name}")
        elif status == "done":
            self.experiments_list.refresh_statuses()
            self.status_label.setText(f"Batch — finished {name}")
        elif status == "error":
            self.experiments_list.refresh_statuses()
            self.status_label.setText(f"Batch — failed {name}")
        # Keep the rail repainting between folders during the in-process run.
        QApplication.processEvents()

    def refresh_stage_statuses(self):
        for key, panel in self._stage_status_panels_by_key.items():
            status = panel.refresh()
            self._stage_sections_by_key[key].set_status(status)
        self.experiments_list.refresh_statuses()

    def _on_stage_enabled_changed(self, key: str) -> None:
        self.refresh_stage_statuses()
        self._write_config()

    def _set_stage_layers_visible(self, key: str, visible: bool) -> None:
        """Show/hide the napari layers a stage paints (header viz glyph, P5)."""
        names = set(STAGE_LAYER_NAMES.get(key, ()))
        layers = getattr(self.viewer, "layers", None)
        if not names or layers is None:
            return
        for layer in layers:
            if layer.name in names:
                layer.visible = visible

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
        'off' (MSM is exempt from auto-skip per D1).
        """
        from pathlib import Path

        from napariTFM.utilities import ntfm as _ntfm

        folder = Path(path)
        tfm_folder = folder / "TFM_data"
        ntfm_path = tfm_folder / f"{folder.name}.ntfm"
        measures = _ntfm.populated_measures(ntfm_path)
        inputs_ready = (folder / "beads.tif").exists() and (
            folder / "reference.tif"
        ).exists()
        # Preprocessing leaves no measure column of its own; a cached image or any
        # downstream measure both prove it ran.
        preproc_cached = (tfm_folder / "preprocessed_beads.tif").exists() and (
            tfm_folder / "preprocessed_reference.tif"
        ).exists()
        preproc_done = preproc_cached or bool(measures)

        disabled = set(self._disabled_stages())

        def _status(stage: str) -> str:
            if stage in disabled:
                return "off"
            if stage == "preprocessing":
                if preproc_done:
                    return "done"
                return "ready" if inputs_ready else "not_started"
            if stage in measures:
                return "done"
            # 'ready' when this stage's immediate input is available.
            ready_when = {
                "displacement": preproc_done,
                "force": "displacement" in measures,
                "stress": "force" in measures,
            }
            return "ready" if ready_when.get(stage, False) else "not_started"

        return {stage: _status(stage) for stage in PIPELINE_STAGES}

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

    def get_state(self) -> dict:
        output_dir = self.data_manager.output_dir
        return {
            "version": STATE_VERSION,
            "parameters": self.parameter_manager.get_all_parameters(),
            "output_dir": str(output_dir) if output_dir else None,
            "disabled_stages": self._disabled_stages(),
            "experiments": self.experiments_list.experiments(),
            "active_experiment": self.experiments_list.active(),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self._applying_state = True
        try:
            self._apply_parameters(state.get("parameters", {}))
            # output_dir is intentionally NOT re-applied: the config lives
            # inside output_dir, so the dir is already known when we load it.
            disabled = set(state.get("disabled_stages") or [])
            for key, section in self._stage_sections_by_key.items():
                if section.enable_btn is not None:
                    section.set_enabled(key not in disabled)
            self.experiments_list.set_experiments(state.get("experiments") or [])
            active = state.get("active_experiment")
            if active:
                self.experiments_list.set_active(active)
        finally:
            self._applying_state = False
        self.refresh()

    def _config_path(self):
        output_dir = self.data_manager.output_dir
        return (output_dir / CONFIG_FILENAME) if output_dir else None

    def _read_config(self):
        path = self._config_path()
        if path is None or not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read config %s: %s", path, exc)
            return None

    def _write_config(self):
        if self._applying_state:
            return
        path = self._config_path()
        if path is None:
            return
        try:
            self.data_manager.ensure_output_dir()
            with open(path, "w") as f:
                json.dump(self.get_state(), f, indent=2)
        except Exception as exc:
            logger.warning("Failed to write config %s: %s", path, exc)

    def _reconcile_to_output_dir(self):
        """On a new output dir: load its config if present, else claim it."""
        state = self._read_config()
        if state is not None:
            self.set_state(state)   # set_state() calls refresh()
        else:
            self._write_config()
            self.refresh()

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
        ``registration_mode`` is normalised, mirroring :meth:`set_state`.
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

    def _save_series(self):
        """Save the experiment series: folders + design tags + run options.

        No analysis knobs ride along — those are a separate ``tfm_params``
        preset, so a series file stays a portable manifest of *what* to run.
        """
        import yaml

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save experiment series",
                "tfm_experiment_series.yaml",
                "YAML Files (*.yaml *.yml)",
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
            with open(file_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            QMessageBox.information(self, "Success", "Experiment series saved!")
        except Exception as e:
            logger.error(f"Error saving experiment series: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save series: {str(e)}")

    def _load_series(self):
        """Load an experiment series: rebuild the table + restore run options."""
        import yaml

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load experiment series",
                "",
                "YAML Files (*.yaml *.yml)",
            )
            if not file_path:
                return
            with open(file_path) as f:
                config = yaml.safe_load(f) or {}

            run_options = config.get("run_options", {}) or {}
            disabled = set(run_options.get("disabled_stages") or [])
            for key, section in self._stage_sections_by_key.items():
                if section.enable_btn is not None:
                    section.set_enabled(key not in disabled)
            # ``processed_root`` is recorded for provenance but intentionally NOT
            # re-applied here: claiming an output dir reloads *its* autosaved
            # state, which would clobber the series we just loaded.
            self.experiments_list.set_records(series_records(config))
            self.refresh_stage_statuses()
            QMessageBox.information(self, "Success", "Experiment series loaded!")
        except Exception as e:
            logger.error(f"Error loading experiment series: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load series: {str(e)}")

    def connect_signals(self):
        """Connect signals between components"""
        self.preprocessing_widget.preprocessing_completed.connect(lambda *_: self.refresh())
        self.displacement_widget.displacement_calculated.connect(lambda *_: self.refresh())
        self.force_widget.force_calculated.connect(lambda *_: self.refresh())
        self.msm_widget.stress_calculated.connect(lambda *_: self.refresh())

        # Connect parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Propagate parameter edits to the widgets that display them."""
        if param_name in ("pixel_size", "frame_interval"):
            # Calibration affects every stage's displayed/derived values.
            for widget in self._stage_widgets():
                update = getattr(widget, "_update_ui_state", None)
                if callable(update):
                    update()
        elif param_name.startswith("force_"):
            self.force_widget._update_ui_state()


