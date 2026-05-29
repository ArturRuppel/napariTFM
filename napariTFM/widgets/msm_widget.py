from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal, QObject
from qtpy.QtWidgets import (
    QLabel, QSizePolicy, QFrame, QApplication, QSpacerItem,
    QPushButton, QProgressBar, QFileDialog
)
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMessageBox

from napariTFM.backend.parameter_dataclasses import MSMParameters
from napariTFM.backend.msm import (
    MSMResult,
    calculate_stresses,
    generate_mesh_stack,
    process_mask_data,
)
from napariTFM.widgets._base_widget import BaseAnalysisWidget
from napariTFM.widgets._output_directory import ensure_output_dir_for_generated_artifacts
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.utilities.visualization_manager import VisualizationManager


class MSMController(QObject):
    """Coordinates interactions between UI components, service, and managers."""

    # Define signals as class attributes
    data_updated = Signal(str)
    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)  # Results object
    analysis_failed = Signal(str)  # Error message
    ui_frozen = Signal(bool)

    def __init__(self, viewer: Viewer,
                 data_manager: DataManager, parameter_manager: ParameterManager,
                 visualization_manager: VisualizationManager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.mask_stack is None:
            QMessageBox.warning(None, "Warning", "No mask loaded. Please load a mask first.")
            return False
        if self.data_manager.force_results is None:
            QMessageBox.warning(None, "Warning", "No force data available. Please calculate forces first.")
            return False
        return True

    def start_analysis(self):
        """Start the stress analysis for all frames."""
        try:
            if not self._validate_prerequisites():
                return

            self.analysis_started.emit()
            self._update_progress(0, "Starting analysis...")

            params = self._get_current_parameters()

            # Get mask and force data
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            force_results = self.data_manager.force_results
            if force_results is None:
                raise ValueError("No force data available")

            # First generate all meshes in the main thread
            print("Generating meshes for all frames...")
            mesh_generator = generate_mesh_stack(masks, params)
            mesh_data = []

            # Process mesh generation results in main thread
            try:
                while True:
                    nodes, elements, quality_metrics, frame, total = next(mesh_generator)
                    mesh_data.append((nodes, elements, quality_metrics))
                    self._update_progress(
                        int((frame + 1) / total * 45),  # Use first 45% for mesh generation
                        f"Generating mesh: Frame {frame + 1}/{total}"
                    )
                    # Keep UI responsive
                    QApplication.processEvents()
            except StopIteration as e:
                # Get the final mesh results if returned
                final_mesh_results = e.value
                if final_mesh_results:
                    mesh_data = final_mesh_results

            # Update status for numba compilation
            QApplication.processEvents()

            # Now calculate stress using the pre-generated meshes
            @thread_worker
            def stress_calculation_worker():
                try:
                    # Get stress generator with pre-generated meshes
                    stress_generator = calculate_stresses(
                        force_field=force_results.force_field,
                        masks=masks,
                        params=params,
                        mesh_data=mesh_data
                    )

                    while True:
                        result, current_frame, total_frames = next(stress_generator)
                        progress = 50 + int(current_frame / total_frames * 50)  # Start from 50%
                        yield (progress, f"Calculating stress: Frame {current_frame}/{total_frames}", result)

                except StopIteration as e:
                    return e.value

            worker = stress_calculation_worker()
            worker.running = True
            self.active_workers.append(worker)

            def on_yielded(data):
                progress, status, result = data
                self._update_progress(progress, status)
                if progress == 51:  # First frame after mesh generation
                    # Update visualization for first frame
                    self.visualization_manager.visualize_stress_preview(
                        result.stress_tensor,
                        max_stress=params.max_stress,
                        downscale_factor=force_results.parameters.downscale_factor
                    )

            def on_returned(final_result):
                # Store the complete results
                self.data_manager.set_stress_results(final_result, source="generated", dirty=True)

                # Update visualization with all frames
                self.visualization_manager.visualize_stress_results()

                # Manage layer visibility and ordering
                xx_layer = None
                yy_layer = None
                avg_layer = None

                # First pass: find the stress layers and disable all others
                for layer in self.viewer.layers:
                    if layer.name == 'Normal Stress XX':
                        xx_layer = layer
                        layer.visible = False
                    elif layer.name == 'Normal Stress YY':
                        yy_layer = layer
                        layer.visible = False
                    elif layer.name == 'Average Normal Stress':
                        avg_layer = layer
                        layer.visible = True
                    else:
                        layer.visible = False

                # Second pass: reorder layers
                if xx_layer is not None:
                    current_index = self.viewer.layers.index(xx_layer)
                    if current_index != len(self.viewer.layers) - 3:  # -3 position
                        self.viewer.layers.move(current_index, -3)

                if yy_layer is not None:
                    current_index = self.viewer.layers.index(yy_layer)
                    if current_index != len(self.viewer.layers) - 2:  # -2 position
                        self.viewer.layers.move(current_index, -2)

                if avg_layer is not None:
                    current_index = self.viewer.layers.index(avg_layer)
                    if current_index != len(self.viewer.layers) - 1:  # -1 position (top)
                        self.viewer.layers.move(current_index, -1)

                # Update status with final metrics
                status_msg = (
                    f"Analysis completed successfully\n"
                    f"Average condition number: {final_result.condition_number:.1e}\n"
                    f"Average residual: {final_result.residual:.1e}"
                )
                self._update_progress(100, status_msg)

                self.analysis_completed.emit(final_result)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            def on_errored(error):
                error_msg = f"Analysis failed: {str(error)}"
                self._update_progress(0, error_msg)
                self.analysis_failed.emit(error_msg)
                QMessageBox.critical(None, "Error", error_msg)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            # Connect worker signals
            worker.yielded.connect(on_yielded)
            worker.returned.connect(on_returned)
            worker.errored.connect(on_errored)
            worker.start()

        except Exception as e:
            error_msg = f"Failed to start analysis: {str(e)}"
            self._update_progress(0, error_msg)
            self.analysis_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def preview_current_frame(self):
        """Calculate and display stress field for current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self._update_progress(0, "Generating stress preview...")

            # Get current frame index
            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            params = self._get_current_parameters()

            # Get mask and force data for current frame
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            force_results = self.data_manager.force_results
            if force_results is None:
                raise ValueError("No force data available")

            # Extract current frame data
            mask = masks[current_frame] if masks.ndim > 2 else masks
            force_field = force_results.force_field[current_frame] if force_results.force_field.ndim > 3 else force_results.force_field

            # Calculate stress field for current frame
            stress_generator = calculate_stresses(
                force_field=force_field[np.newaxis, ...],
                masks=mask[np.newaxis, ...],
                params=params,
            )

            try:
                # Get stress results
                result, _, _ = next(stress_generator)

                # Update visualization
                self.visualization_manager.visualize_stress_preview(
                    result.stress_tensor,
                    max_stress=params.max_stress,
                    downscale_factor=force_results.parameters.downscale_factor
                )

                # Manage layer visibility and ordering
                xx_layer = None
                yy_layer = None
                avg_layer = None

                # First pass: find the stress layers and disable all others
                for layer in self.viewer.layers:
                    if layer.name == 'Normal Stress XX':
                        xx_layer = layer
                        layer.visible = False
                    elif layer.name == 'Normal Stress YY':
                        yy_layer = layer
                        layer.visible = False
                    elif layer.name == 'Average Normal Stress':
                        avg_layer = layer
                        layer.visible = True
                    else:
                        layer.visible = False

                # Second pass: reorder layers
                if xx_layer is not None:
                    current_index = self.viewer.layers.index(xx_layer)
                    if current_index != len(self.viewer.layers) - 3:  # -3 position
                        self.viewer.layers.move(current_index, -3)

                if yy_layer is not None:
                    current_index = self.viewer.layers.index(yy_layer)
                    if current_index != len(self.viewer.layers) - 2:  # -2 position
                        self.viewer.layers.move(current_index, -2)

                if avg_layer is not None:
                    current_index = self.viewer.layers.index(avg_layer)
                    if current_index != len(self.viewer.layers) - 1:  # -1 position (top)
                        self.viewer.layers.move(current_index, -1)

                # Create status message with key metrics
                status_msg = (
                    f"Stress preview generated for frame {current_frame}\n"
                    f"Condition number: {result.condition_number:.1e}\n"
                    f"Residual: {result.residual:.1e}"
                )

                self._update_progress(100, status_msg)
                return result

            except StopIteration:
                raise ValueError("Stress calculation failed to produce results")

        except Exception as e:
            error_msg = f"Failed to preview stress field: {str(e)}"
            self._update_progress(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def _get_current_parameters(self) -> MSMParameters:
        """Get current MSM parameters from parameter manager."""
        return self.parameter_manager.get_msm_parameters()

    def freeze_ui(self):
        """Signal the owning widget to disable interactive controls."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal the owning widget to re-enable controls."""
        self.ui_frozen.emit(False)

    def _update_progress(self, progress: int, status: str):
        """Update progress and emit signal."""
        self.progress_updated.emit(progress, status)

    def preview_mesh(self):
        """Generate and display mesh preview for the current frame."""
        try:
            self._update_progress(0, "Generating mesh preview...")

            # Get current frame index
            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            params = self._get_current_parameters()

            # Get mask for current frame
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            mask = masks[current_frame] if masks.ndim > 2 else masks

            mesh_generator = generate_mesh_stack(mask, params)

            try:
                # Get first mesh result
                nodes, elements, quality_metrics, _, _ = next(mesh_generator)

                # Get downscale factor from force results if available
                if self.data_manager.force_results is not None:
                    downscale_factor = self.data_manager.force_results.parameters.downscale_factor
                else:
                    downscale_factor = 1

                # Update visualization
                self.visualization_manager.visualize_mesh(
                    nodes=nodes,
                    elements=elements,
                    downscale_factor=downscale_factor
                )

                # Simplified quality metrics message
                num_elements = len(elements)
                mean_quality = quality_metrics.get('mean_quality', 0.0)
                status_msg = (
                    f"Mesh preview generated for frame {current_frame}\n"
                    f"Elements: {num_elements}\n"
                    f"Mean quality: {mean_quality:.3f}"
                )

                self._update_progress(100, status_msg)

                return nodes, elements, quality_metrics

            except StopIteration:
                raise ValueError("Mesh generation failed to produce results")

        except Exception as e:
            error_msg = f"Failed to preview mesh: {str(e)}"
            self._update_progress(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def cancel_all_operations(self):
        """Cancel all running background operations"""
        for worker in self.active_workers:
            try:
                worker.running = False  # Set cancellation flag
                worker.quit()
                worker.wait(500)  # Wait up to 500ms
                if worker.isRunning():
                    worker.terminate()
                worker.deleteLater()
            except Exception as e:
                pass
        self.active_workers.clear()
        # Update UI status
        self.progress_updated.emit(0, "Operations cancelled")
        QApplication.processEvents()
        self.unfreeze_ui()


class MSMWidget(BaseAnalysisWidget):
    """Widget for Monolayer Stress Microscopy analysis."""
    stress_calculated = Signal(object)  # Emits stress analysis results
    action_states_changed = Signal()  # Emitted when action enablement changes

    def __init__(
            self,
            viewer: Viewer,
            data_manager: DataManager,
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Store managers
        self.parameter_manager = parameter_manager

        # Action enablement consumed by the stage header via the action contract
        self._action_enabled = {"run": False, "preview": False, "cancel": True}

        # Get initial parameters from parameter manager
        self.msm_params = parameter_manager.get_msm_parameters()

        # Initialize controller (owns no panels; emits ui_frozen)
        self.controller = MSMController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
        )

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

        # Keep service parameters synced with the shared parameter manager
        parameter_manager.parameters_reset.connect(self._update_service_parameters)
        parameter_manager.parameter_changed.connect(self._handle_parameter_change)

        self.controller.unfreeze_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_container = self._create_content_container()
        main_layout.addWidget(content_container)

        self.setLayout(main_layout)

    def _create_content_container(self) -> QWidget:
        """Create the main content container."""
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_action_row())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        return container

    def _create_action_row(self) -> QWidget:
        """Build widget-owned action buttons (run/preview/cancel proxied by the stage header)."""
        container = QWidget()
        layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.preview_mesh_btn = QPushButton("Preview Mesh")
        self.preview_mesh_btn.setToolTip("Generate and display a mesh preview for the current frame")
        row1.addWidget(self.preview_mesh_btn)
        layout.addLayout(row1)

        container.setLayout(layout)
        return container

    def _create_status_frame(self) -> QFrame:
        """Create the status display frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _connect_signals(self):
        """Connect all widget signals."""
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_started.connect(self._on_analysis_started)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Wire widget-owned action buttons to controller operations
        self.preview_mesh_btn.clicked.connect(self.controller.preview_mesh)

        # Update enablement when the active layer changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.stress_results is not None:
            self.visualization_manager.update_stress_frame(
                self.viewer.dims.current_step[0]
            )

    def load_result_artifact(self, key: str):
        if key == "mask_stack":
            self._load_mask_stack_from_active_layer()
        else:
            path = self._choose_result_path(key)
            if not path:
                return
            self.data_manager.load_result_artifact(key, path)
            show_force = getattr(self.visualization_manager, "visualize_force_results", None)
            if key == "force_results" and show_force is not None:
                show_force()
        self._update_ui_state()

    def _choose_result_path(self, key: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {key.replace('_', ' ')}",
            "",
            "NumPy Files (*.npy)",
        )
        return path

    def _load_mask_stack_from_active_layer(self):
        active_layer = self.viewer.layers.selection.active
        if active_layer is None or not hasattr(active_layer, "data"):
            QMessageBox.warning(self, "No Layer Selected", "Please select an image layer to load masks from.")
            return
        mask_data = active_layer.data
        if not isinstance(mask_data, np.ndarray):
            QMessageBox.warning(self, "Invalid Layer", "Selected layer is not a valid image layer.")
            return
        force_results = self.data_manager.force_results
        force_field = force_results.force_field if force_results is not None else None
        processed_masks, warnings = process_mask_data(mask_data, force_field)
        for warning in warnings:
            QMessageBox.warning(self, "Warning", warning)
        self.data_manager.set_mask_stack(processed_masks)
        self.visualization_manager.visualize_masks(processed_masks)

    def _update_service_parameters(self, category: ParameterCategory):
        """Update service parameters when parameters are reset."""
        if category == ParameterCategory.STRESS:
            self.msm_params = self.parameter_manager.get_msm_parameters()

    def _handle_parameter_change(self, param_name: str, value: Any):
        """Update service parameters when individual parameters change."""
        stress_params = {
            'density_factor', 'mesh_algorithm', 'use_optimization',
            'poisson_ratio_cells', 'max_stress'
        }
        if param_name in stress_params:
            self.msm_params = self.parameter_manager.get_msm_parameters()

    def action_states(self):
        return dict(self._action_enabled)

    def run_action(self):
        self.controller.start_analysis()

    def preview_action(self):
        self.controller.preview_current_frame()

    def cancel_action(self):
        self.controller.cancel_all_operations()

    def _update_ui_state(self, event=None):
        """Update action button enablement based on available data."""
        has_force = self.data_manager.force_results is not None
        has_mask = self.data_manager.mask_stack is not None

        self.preview_mesh_btn.setEnabled(has_mask)
        self._action_enabled["preview"] = has_force and has_mask
        self._action_enabled["run"] = has_force and has_mask
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        if frozen:
            self.preview_mesh_btn.setEnabled(False)
            self._action_enabled["preview"] = False
            self._action_enabled["run"] = False
            self._action_enabled["cancel"] = True
            self.action_states_changed.emit()
        else:
            self._update_ui_state()

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()

    def _on_analysis_started(self):
        """Handle analysis start event."""
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis started...")

    def _on_analysis_completed(self, results):
        """Handle analysis completion."""
        if ensure_output_dir_for_generated_artifacts(self, self.data_manager):
            try:
                self.data_manager.auto_save_artifact("stress_results")
            except Exception as exc:
                self.data_manager.mark_artifact_error("stress_results", str(exc))
                QMessageBox.warning(self, "Auto-save Failed", str(exc))

        self.stress_calculated.emit(results)
        self._update_ui_state()

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_status(0, f"Analysis failed: {error_msg}")
        self._update_ui_state()
