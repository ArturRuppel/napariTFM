import logging
from typing import Any

import napari
from qtpy.QtCore import Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QSizePolicy, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QPushButton, QSpinBox, QComboBox, QFileDialog, QCheckBox,
    QFormLayout
)

from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.visualization_manager import VisualizationManager

from napariTFM.widgets.preprocessing_widget import PreprocessingWidget
from napariTFM.widgets.displacement_analysis_widget import DisplacementAnalysisWidget
from napariTFM.widgets.fttc_widget import FTTCWidget
from napariTFM.widgets.msm_widget import MSMWidget
from napariTFM.widgets.batch_analysis_widget import BatchAnalysisWidget
from napariTFM.widgets._stage_data_status import DataArtifactSpec, StageDataStatusPanel
from napariTFM.widgets._stage_section import StageSection
from napariTFM.widgets._project_section import ProjectSection

logger = logging.getLogger(__name__)


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
    "batch": [
        DataArtifactSpec("batch_outputs", "Batch outputs", None, "output"),
    ],
}


def _build_preprocessing_specs(preprocessing_widget, visualization_manager, save_artifact):
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
            on_action=lambda: save_artifact("preprocessed_reference"),
        ),
        DataArtifactSpec(
            "preprocessed_bead_stack",
            "Preprocessed beads",
            "preprocessed_bead_stack",
            "output",
            on_view=view("preprocessed_bead_stack"),
            on_action=lambda: save_artifact("preprocessed_bead_stack"),
        ),
    ]


def _build_displacement_specs(displacement_widget, save_artifact):
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
            on_action=lambda: save_artifact("displacement_results"),
        ),
    ]


def _build_force_specs(force_widget, save_artifact):
    return [
        DataArtifactSpec(
            "displacement_results",
            "Displacement field",
            "displacement_results",
            "input",
            on_action=lambda: force_widget.load_result_artifact("displacement_results"),
        ),
        DataArtifactSpec(
            "force_results",
            "Traction map",
            "force_results",
            "output",
            on_action=lambda: save_artifact("force_results"),
        ),
    ]


def _build_stress_specs(stress_widget, save_artifact):
    return [
        DataArtifactSpec(
            "force_results",
            "Traction map",
            "force_results",
            "input",
            on_action=lambda: stress_widget.load_result_artifact("force_results"),
        ),
        DataArtifactSpec(
            "mask_stack",
            "Mask stack",
            "mask_stack",
            "input",
            required=False,
            on_action=lambda: stress_widget.load_result_artifact("mask_stack"),
        ),
        DataArtifactSpec(
            "stress_results",
            "Stress map",
            "stress_results",
            "output",
            on_action=lambda: save_artifact("stress_results"),
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
            ("threshold", "Threshold Percentile (%)", "float", 0.0, 100.0, 0.1, 1, None),
            ("dilation", "Mask Dilation (px)", "int", 0, 50, 1, 0, None),
            ("smoothing_sigma", "Boundary Smoothing", "float", 0.0, 40.0, 0.1, 1, None),
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
        layout.setSpacing(4)

        for title, specs in self.PARAMETER_SECTIONS:
            if self._section_titles is not None and title not in self._section_titles:
                continue
            group = QGroupBox(title)
            form = QFormLayout()
            form.setContentsMargins(8, 8, 8, 8)
            form.setSpacing(4)

            for spec in specs:
                name, label, kind, min_val, max_val, step, decimals, choices = spec
                control = self._create_control(name, kind, min_val, max_val, step, decimals, choices)
                form.addRow(label, control)

            group.setLayout(form)
            layout.addWidget(group)

        self.setLayout(layout)

    def _create_control(self, name, kind, min_val, max_val, step, decimals, choices):
        if kind == "int":
            control = QSpinBox()
            control.setRange(min_val, max_val)
            control.setSingleStep(step)
            control.valueChanged.connect(lambda value, n=name: self.parameter_manager.set_ui_parameter(n, value))
        elif kind == "float":
            control = QDoubleSpinBox()
            control.setRange(min_val, max_val)
            control.setSingleStep(step)
            control.setDecimals(decimals)
            if name == "gel_height":
                control.setSpecialValueText("∞")
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


_StageSection = StageSection


class SpinBoxEventFilter(QObject):
    def eventFilter(self, obj, event):
        # Check for all spinnable input widgets
        if (isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox)) and
                event.type() == event.Wheel):
            if not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)


class napariTFMWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer

        # Create and install event filter
        self.spinbox_filter = SpinBoxEventFilter(self)

        # Find and filter all spinboxes in the application
        def install_filter_on_inputs():
            for widget in self.window().findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
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

        # Add title
        title = QLabel("napariTFM")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        container_layout.addWidget(title)

        # Initialize managers
        self.data_manager = DataManager()
        self.parameter_manager = ParameterManager()
        self.visualization_manager = VisualizationManager(self.viewer, self.data_manager)

        self.project_section = ProjectSection(self.parameter_manager, self.data_manager)
        container_layout.addWidget(self.project_section)

        self._stage_parameter_panels_by_key = self._create_stage_parameter_panels()

        # Wire up the Project section's I/O buttons (replaces _create_general_group).
        self.save_params_btn = self.project_section.save_params_btn
        self.load_params_btn = self.project_section.load_params_btn
        self.reset_params_btn = self.project_section.reset_params_btn
        self.clear_data_btn = self.project_section.clear_data_btn
        self.save_params_btn.clicked.connect(self._save_parameters)
        self.load_params_btn.clicked.connect(self._load_parameters)
        self.reset_params_btn.clicked.connect(self._reset_parameters)
        self.clear_data_btn.clicked.connect(self._clear_all_data)

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

        self.batch_widget = BatchAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )
        self._hide_embedded_parameter_panels()
        self._hide_redundant_stage_shell_controls()

        stage_data_artifacts = dict(STAGE_DATA_ARTIFACTS)
        stage_data_artifacts["preprocessing"] = _build_preprocessing_specs(
            self.preprocessing_widget,
            self.visualization_manager,
            self._save_generated_artifact,
        )
        stage_data_artifacts["displacement"] = _build_displacement_specs(
            self.displacement_widget,
            self._save_generated_artifact,
        )
        stage_data_artifacts["force"] = _build_force_specs(
            self.force_widget,
            self._save_generated_artifact,
        )
        stage_data_artifacts["stress"] = _build_stress_specs(
            self.msm_widget,
            self._save_generated_artifact,
        )
        self._stage_status_panels_by_key = {
            key: StageDataStatusPanel(key, self.data_manager, artifacts)
            for key, artifacts in stage_data_artifacts.items()
        }

        self._stage_sections_by_key = {
            "preprocessing": _StageSection(
                "Preprocessing",
                self.preprocessing_widget,
                expanded=True,
                status_panel=self._stage_status_panels_by_key["preprocessing"],
                action_targets=self._find_stage_action_targets(
                    self.preprocessing_widget,
                    run=["process_btn"],
                    preview=["preview_check"],
                    cancel=["cancel_btn"],
                ),
            ),
            "displacement": _StageSection(
                "Displacement",
                self.displacement_widget,
                status_panel=self._stage_status_panels_by_key["displacement"],
                action_targets=self._find_stage_action_targets(
                    self.displacement_widget,
                    run=["process_btn"],
                    preview=["preview_btn"],
                    cancel=["cancel_btn"],
                ),
            ),
            "force": _StageSection(
                "Force Analysis",
                self.force_widget,
                status_panel=self._stage_status_panels_by_key["force"],
                action_targets=self._find_stage_action_targets(
                    self.force_widget,
                    run=["process_btn"],
                    preview=["preview_btn"],
                    cancel=["cancel_btn"],
                ),
            ),
            "stress": _StageSection(
                "Stress Analysis",
                self.msm_widget,
                status_panel=self._stage_status_panels_by_key["stress"],
                action_targets=self._find_stage_action_targets(
                    self.msm_widget,
                    run=["action_panel.analyze_btn", "analyze_btn"],
                    preview=["action_panel.preview_frame_btn", "action_panel.preview_mesh_btn"],
                    cancel=["action_panel.cancel_btn", "cancel_btn"],
                ),
            ),
            "batch": _StageSection(
                "Batch Analysis",
                self.batch_widget,
                status_panel=self._stage_status_panels_by_key["batch"],
                action_targets=self._find_stage_action_targets(
                    self.batch_widget,
                    run=["run_analysis_btn"],
                    preview=[],
                    cancel=[],
                ),
            ),
        }
        self._stage_sections = list(self._stage_sections_by_key.values())

        # Mount per-stage parameter panels as nested "Parameters" sub-sections.
        self._stage_inner_param_sections_by_key = {}
        for key, section in self._stage_sections_by_key.items():
            panel = self._stage_parameter_panels_by_key.get(key)
            if panel is None:
                continue
            inner = section.add_inner_section("Parameters", panel, expanded=False)
            self._stage_inner_param_sections_by_key[key] = inner
            # Reroute the outer section's params_btn to toggle the inner section
            # instead of the legacy overlay parameter content.
            try:
                section.params_btn.toggled.disconnect()
            except (TypeError, RuntimeError):
                pass
            section.params_btn.toggled.connect(
                lambda checked, s=section, i=inner: (
                    s._set_expanded(checked),
                    i._toggle_button.setChecked(checked),
                )
            )

        for section in self._stage_sections:
            container_layout.addWidget(section)
        container_layout.addStretch()

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Add scroll area to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.connect_signals()
        self.data_manager.add_change_callback(self._on_pipeline_data_changed)
        self.refresh_stage_statuses()

    def _hide_embedded_parameter_panels(self):
        """Keep stage-local panels alive for controllers while removing duplicate visible editors."""
        for widget in [
            self.preprocessing_widget,
            self.displacement_widget,
            self.force_widget,
            self.msm_widget,
        ]:
            panel = getattr(widget, "parameter_panel", None)
            if panel is not None:
                panel.setVisible(False)

        for group in self.batch_widget.findChildren(QGroupBox):
            if group.title() in {
                "General Parameters",
                "Preprocessing Parameters",
                "Farneback Displacement Parameters",
                "Force Parameters",
                "Stress Parameters",
            }:
                group.setVisible(False)

    def _hide_redundant_stage_shell_controls(self):
        """Keep controller-owned controls alive while removing duplicated shell surfaces."""
        for widget in [self.msm_widget]:
            for attr in ("data_panel", "action_panel"):
                panel = getattr(widget, attr, None)
                if panel is not None:
                    panel.setVisible(False)

    def _create_stage_parameter_panels(self) -> dict[str, WorkflowParameterPanel]:
        """Create inline workflow parameter editors grouped by pipeline stage."""
        stage_sections = {
            "preprocessing": ("General", "Preprocessing"),
            "displacement": ("Displacement",),
            "force": ("Force",),
            "stress": ("Stress",),
        }
        return {
            key: WorkflowParameterPanel(self.parameter_manager, section_titles=titles)
            for key, titles in stage_sections.items()
        }

    def _find_stage_action_targets(self, widget: QWidget, **action_paths: list[str]) -> dict[str, QWidget]:
        """Find existing child controls that can be triggered from the stage header."""
        targets = {}
        for action, paths in action_paths.items():
            target = self._first_existing_widget(widget, paths)
            if target is not None and hasattr(target, "click"):
                targets[action] = target
        return targets

    def _first_existing_widget(self, widget: QWidget, paths: list[str]) -> QWidget | None:
        for path in paths:
            target = widget
            for attr in path.split("."):
                target = getattr(target, attr, None)
                if target is None:
                    break
            if target is not None:
                return target
        return None

    def refresh_stage_statuses(self):
        for key, panel in self._stage_status_panels_by_key.items():
            status = panel.refresh()
            self._stage_sections_by_key[key].set_status(status)

    def _save_generated_artifact(self, key: str):
        try:
            kwargs = {}
            if key.startswith("preprocessed_"):
                kwargs = {
                    "pixel_size": self.parameter_manager.get_ui_parameter("pixel_size"),
                    "frame_interval": self.parameter_manager.get_ui_parameter("frame_interval"),
                }
            self.data_manager.auto_save_artifact(key, **kwargs)
        except Exception as exc:
            self.data_manager.mark_artifact_error(key, str(exc))
            QMessageBox.warning(self, "Save Failed", str(exc))
        finally:
            self.refresh_stage_statuses()

    def _on_pipeline_data_changed(self):
        for widget in [
            self.preprocessing_widget,
            self.displacement_widget,
            self.force_widget,
            self.msm_widget,
            self.batch_widget,
        ]:
            if hasattr(widget, '_update_ui_state'):
                widget._update_ui_state()
        self.refresh_stage_statuses()

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

                # Update all widget states
                for widget in [
                    self.preprocessing_widget,
                    self.displacement_widget,
                    self.force_widget,
                    self.msm_widget,
                    self.batch_widget
                ]:
                    if hasattr(widget, '_update_ui_state'):
                        widget._update_ui_state()

        except Exception as e:
            logger.error(f"Error resetting parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to reset parameters: {str(e)}")

    def _save_parameters(self):
        """Save parameters using parameter manager."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Parameters",
                "",
                "YAML Files (*.yaml *.yml)"
            )
            if file_path:
                if not file_path.lower().endswith(('.yaml', '.yml')):
                    file_path += '.yaml'
                self.parameter_manager.save_to_file(file_path)
                QMessageBox.information(self, "Success", "Parameters saved successfully!")
        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {str(e)}")

    def _load_parameters(self):
        """Load parameters using parameter manager."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Parameters",
                "",
                "YAML Files (*.yaml *.yml)"
            )
            if file_path:
                self.parameter_manager.load_from_file(file_path)
                QMessageBox.information(self, "Success", "Parameters loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

    def connect_signals(self):
        """Connect signals between components"""
        self.preprocessing_widget.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.displacement_widget.displacement_calculated.connect(self._on_displacement_completed)
        self.force_widget.force_calculated.connect(self._on_force_completed)
        self.msm_widget.stress_calculated.connect(self._on_stress_completed)

        # Connect parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes from parameter manager"""
        # For calibration parameters, we need to update all widgets
        if param_name in ['pixel_size', 'frame_interval']:
            for widget in [
                self.preprocessing_widget,
                self.displacement_widget,
                self.force_widget,
                self.msm_widget,
                self.batch_widget
            ]:
                # Use _update_calibration instead of _update_parameters
                if hasattr(widget, '_update_calibration'):
                    widget._update_calibration()
                # Or just update the UI state if _update_calibration doesn't exist
                elif hasattr(widget, '_update_ui_state'):
                    widget._update_ui_state()

        # Let individual widgets handle their specific parameters if needed
        try:
            if param_name.startswith('preprocessing_'):
                self.preprocessing_widget._update_ui_state()
            elif param_name.startswith('displacement_'):
                self.displacement_widget._update_ui_state()
            elif param_name.startswith('force_'):
                self.force_widget._update_ui_state()
            elif param_name.startswith('stress_'):
                self.msm_widget._update_ui_state()
        except AttributeError:
            pass

    def _clear_all_data(self):
        """
        Clear all data and reset the widget to its initial state.
        This includes clearing the data manager and resetting UI elements.
        """
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Are you sure you want to clear all data? This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Clear data manager
                self.data_manager.__init__()
                self.data_manager.add_change_callback(self._on_pipeline_data_changed)

                # Update UI state in all widgets
                self.preprocessing_widget._update_ui_state()
                self.displacement_widget._update_ui_state()
                self.force_widget._update_ui_state()
                self.msm_widget._update_ui_state()
                self.batch_widget._update_ui_state()
                self.refresh_stage_statuses()

                logger.info("All data cleared successfully")

            except Exception as e:
                logger.error(f"Error clearing data: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to clear data: {str(e)}")

    def _on_preprocessing_completed(self, results):
        """Handle completion of preprocessing"""
        logger.info("Preprocessing completed successfully")

        self.preprocessing_widget._update_ui_state()
        self.displacement_widget._update_ui_state()
        self.force_widget._update_ui_state()
        self.msm_widget._update_ui_state()
        self.batch_widget._update_ui_state()
        self.refresh_stage_statuses()

    def _on_displacement_completed(self, results):
        """Handle completion of displacement analysis"""
        logger.info("Displacement analysis completed successfully")
        self.preprocessing_widget._update_ui_state()
        self.displacement_widget._update_ui_state()
        self.force_widget._update_ui_state()
        self.msm_widget._update_ui_state()
        self.batch_widget._update_ui_state()
        self.refresh_stage_statuses()

    def _on_force_completed(self, results):
        """Handle completion of force calculation"""
        logger.info("Force calculation completed successfully")
        self.preprocessing_widget._update_ui_state()
        self.displacement_widget._update_ui_state()
        self.force_widget._update_ui_state()
        self.msm_widget._update_ui_state()
        self.batch_widget._update_ui_state()
        self.refresh_stage_statuses()

    def _on_stress_completed(self, results):
        """Handle completion of stress calculation"""
        logger.info("Stress calculation completed successfully")
        self.preprocessing_widget._update_ui_state()
        self.displacement_widget._update_ui_state()
        self.force_widget._update_ui_state()
        self.msm_widget._update_ui_state()
        self.batch_widget._update_ui_state()
        self.refresh_stage_statuses()
