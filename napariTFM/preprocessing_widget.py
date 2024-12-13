import logging
from typing import Optional

import numpy as np
from napari.utils.events import Event
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QFrame, QLabel, QGroupBox, QProgressBar,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QSizePolicy,
    QFormLayout
)
from qtrangeslider import QRangeSlider

from .base_widget import BaseAnalysisWidget, ProcessingError
from .data_manager import DataManager
from .preprocessing import PreprocessingParameters, ImagePreprocessor
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class PreprocessingWidget(BaseAnalysisWidget):
    """Widget for controlling image preprocessing parameters"""
    """Widget for controlling image preprocessing parameters"""

    preprocessing_completed = Signal(np.ndarray, list)  # Processed stack and info

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: DataManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(
            viewer=viewer,
            data_manager=data_manager,
            visualization_manager=visualization_manager,
        )

        # Initialize preprocessing components
        self.preprocessor = ImagePreprocessor()
        self.preview_enabled = False
        self.original_layer = None
        self.preview_layer = None
        self.current_min_intensity = 0
        self.current_max_intensity = 255

        # Setup UI
        self._setup_ui()
        self._connect_signals()

        # Add layer removal event handler
        self.viewer.layers.events.removed.connect(self._handle_layer_removal)

    def _setup_ui(self):
        # Create a container widget that won't stretch
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Main layout with fixed spacing
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(6, 6, 6, 6)
        container.setLayout(main_layout)

        # Outer layout that can stretch
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(container)
        outer_layout.addStretch()  # This pushes the widget to the top
        self.setLayout(outer_layout)

        # Title
        title = QLabel("Image Preprocessing")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        main_layout.addWidget(title)

        # Intensity Range Group
        intensity_group = self._create_group_box("Intensity Range")
        intensity_layout = QVBoxLayout()
        intensity_layout.setSpacing(5)
        intensity_group.setLayout(intensity_layout)

        # Range slider
        self.intensity_slider = QRangeSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 255)
        self.intensity_slider.setValue((0, 255))
        intensity_layout.addWidget(self.intensity_slider)

        # Spin boxes in horizontal layout with fixed width
        spin_layout = QHBoxLayout()
        spin_layout.setSpacing(5)

        self.min_spin = QSpinBox()
        self.max_spin = QSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setRange(0, 255)
            spin.setFixedWidth(70)
            spin_layout.addWidget(spin)

        intensity_layout.addLayout(spin_layout)
        main_layout.addWidget(intensity_group)

        # Filters Group
        filters_group = self._create_group_box("Filters")
        filters_layout = QFormLayout()
        filters_layout.setSpacing(8)
        filters_group.setLayout(filters_layout)

        # Create filter controls with consistent spacing
        self.median_check, self.median_size_spin, self.median_slider = self._create_filter_control(
            "Median", 3, 15, 3, step=2, odd_only=True
        )
        filters_layout.addRow("Median:", self._wrap_filter_controls(
            self.median_check, self.median_size_spin, self.median_slider
        ))

        self.gaussian_check, self.gaussian_sigma_spin, self.gaussian_slider = self._create_filter_control(
            "Gaussian", 0.1, 10.0, 1.0, step=0.1, double=True
        )
        filters_layout.addRow("Gaussian:", self._wrap_filter_controls(
            self.gaussian_check, self.gaussian_sigma_spin, self.gaussian_slider
        ))

        self.clahe_check, self.clahe_clip_spin, self.clahe_clip_slider = self._create_filter_control(
            "CLAHE", 0.1, 100.0, 16.0, step=0.1, double=True
        )
        filters_layout.addRow("CLAHE:", self._wrap_filter_controls(
            self.clahe_check, self.clahe_clip_spin, self.clahe_clip_slider
        ))

        self.clahe_grid_control, self.clahe_grid_spin, self.clahe_grid_slider = self._create_filter_control(
            "Grid Size", 1, 64, 16, checkbox=False
        )
        filters_layout.addRow("Grid Size:", self._wrap_filter_controls(
            self.clahe_grid_control, self.clahe_grid_spin, self.clahe_grid_slider
        ))

        main_layout.addWidget(filters_group)

        # Preview Section
        preview_frame = QFrame()
        preview_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(5, 5, 5, 5)
        self.preview_check = QCheckBox("Show Preview")
        preview_layout.addWidget(self.preview_check)
        preview_layout.addStretch()
        preview_frame.setLayout(preview_layout)
        main_layout.addWidget(preview_frame)

        # Action Buttons
        button_frame = QFrame()
        button_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 5, 5, 5)
        button_layout.setSpacing(10)

        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.reset_btn = QPushButton("Reset Parameters")

        # Set fixed height for buttons
        for btn in (self.preprocess_btn, self.reset_btn):
            btn.setFixedHeight(30)

        button_layout.addWidget(self.preprocess_btn)
        button_layout.addWidget(self.reset_btn)
        button_frame.setLayout(button_layout)
        main_layout.addWidget(button_frame)

        # Status and Progress
        status_frame = QFrame()
        status_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(5, 5, 5, 5)
        status_layout.setSpacing(5)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(20)
        status_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setFixedHeight(55)
        status_layout.addWidget(self.status_label)

        status_frame.setLayout(status_layout)
        main_layout.addWidget(status_frame)

        # Register controls
        self._register_controls()

    def _create_group_box(self, title: str) -> QGroupBox:
        """Create a group box with consistent styling"""
        group = QGroupBox(title)
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return group

    def _create_filter_control(self, label, min_val, max_val, default_val, step=1.0,
                               checkbox=True, double=False, odd_only=False):
        """Create a consistent set of filter controls"""
        if checkbox:
            check = QCheckBox()
            check.setFixedWidth(20)
        else:
            check = QWidget()
            check.setFixedWidth(20)

        spin = QDoubleSpinBox() if double else QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step if not odd_only else 2)
        spin.setFixedWidth(70)
        if checkbox:
            spin.setEnabled(False)

        slider = QSlider(Qt.Horizontal)
        if odd_only:
            slider_range = int((max_val - min_val) / 2)
            slider.setRange(0, slider_range)
            slider.setValue(int((default_val - min_val) / 2))
        else:
            slider.setRange(int(min_val * (1 / step)), int(max_val * (1 / step)))
            slider.setValue(int(default_val * (1 / step)))
        if checkbox:
            slider.setEnabled(False)

        return check, spin, slider

    def _wrap_filter_controls(self, check, spin, slider):
        """Wrap filter controls in a consistent layout"""
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        layout.addWidget(check)
        layout.addWidget(spin)
        layout.addWidget(slider, 1)  # Give slider a stretch factor of 1

        container.setLayout(layout)
        return container

    def _register_controls(self):
        """Register all controls with the base widget"""
        controls = [
            self.intensity_slider, self.min_spin, self.max_spin,
            self.median_check, self.median_size_spin, self.median_slider,
            self.gaussian_check, self.gaussian_sigma_spin, self.gaussian_slider,
            self.clahe_check, self.clahe_clip_spin, self.clahe_clip_slider,
            self.clahe_grid_spin, self.clahe_grid_slider,
            self.preview_check, self.preprocess_btn, self.reset_btn
        ]

        for control in controls:
            self.register_control(control)

    def _create_filter_controls(self, form_layout: QFormLayout):
        """Create filter parameter controls"""

        def create_parameter_group(
                label: str,
                min_val: float,
                max_val: float,
                default_val: float,
                step: float = 1.0,
                checkbox: bool = True,
                double: bool = False,
                odd_only: bool = False
        ):
            control_layout = QHBoxLayout()

            if checkbox:
                check = QCheckBox("")
                check.setFixedWidth(30)
                control_layout.addWidget(check)
            else:
                check = None
                spacer = QWidget()
                spacer.setFixedWidth(30)
                control_layout.addWidget(spacer)

            spin = QDoubleSpinBox() if double else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default_val)
            spin.setSingleStep(step if not odd_only else 2)
            spin.setFixedWidth(80)
            if checkbox:
                spin.setEnabled(False)
            control_layout.addWidget(spin)

            slider = QSlider(Qt.Horizontal)
            if odd_only:
                slider_range = int((max_val - min_val) / 2)
                slider.setRange(0, slider_range)
                slider.setValue(int((default_val - min_val) / 2))
            else:
                slider.setRange(int(min_val * (1 / step)), int(max_val * (1 / step)))
                slider.setValue(int(default_val * (1 / step)))
            if checkbox:
                slider.setEnabled(False)
            control_layout.addWidget(slider, stretch=1)

            form_layout.addRow(label + ":", control_layout)
            return check, spin, slider

        # Create controls
        self.median_check, self.median_size_spin, self.median_slider = create_parameter_group(
            "Median Filter", 3, 15, 3, step=2, odd_only=True
        )

        self.gaussian_check, self.gaussian_sigma_spin, self.gaussian_slider = create_parameter_group(
            "Gaussian Filter", 0.1, 10.0, 1.0, step=0.1, double=True
        )

        self.clahe_check, self.clahe_clip_spin, self.clahe_clip_slider = create_parameter_group(
            "CLAHE Clip Limit", 0.1, 100.0, 16.0, step=0.1, double=True
        )

        self.clahe_grid_control, self.clahe_grid_spin, self.clahe_grid_slider = create_parameter_group(
            "CLAHE Grid Size", 1, 64, 16, checkbox=False
        )

    def _connect_signals(self):
        """Connect widget signals"""
        # Intensity range
        self.intensity_slider.valueChanged.connect(self._update_from_intensity_slider)
        self.min_spin.valueChanged.connect(self._update_from_intensity_spinboxes)
        self.max_spin.valueChanged.connect(self._update_from_intensity_spinboxes)

        def connect_slider_spin(slider, spinbox, checkbox=None, scale_factor=1.0, odd_only=False):
            def update_from_slider(value):
                spinbox.blockSignals(True)
                if odd_only:
                    actual_value = 3 + (value * 2)
                    spinbox.setValue(actual_value)
                else:
                    spinbox.setValue(value * scale_factor)
                spinbox.blockSignals(False)
                self.update_parameters()

            def update_from_spin(value):
                slider.blockSignals(True)
                if odd_only:
                    slider_value = int((value - 3) / 2)
                    slider.setValue(slider_value)
                else:
                    slider.setValue(int(value / scale_factor))
                slider.blockSignals(False)
                self.update_parameters()

            slider.valueChanged.connect(update_from_slider)
            spinbox.valueChanged.connect(update_from_spin)

            if checkbox:
                checkbox.toggled.connect(slider.setEnabled)
                checkbox.toggled.connect(spinbox.setEnabled)
                checkbox.toggled.connect(self.update_parameters)

        # Connect parameter controls
        connect_slider_spin(self.median_slider, self.median_size_spin, self.median_check, odd_only=True)
        connect_slider_spin(self.gaussian_slider, self.gaussian_sigma_spin, self.gaussian_check, 0.1)
        connect_slider_spin(self.clahe_clip_slider, self.clahe_clip_spin, self.clahe_check, 0.1)
        connect_slider_spin(self.clahe_grid_slider, self.clahe_grid_spin)

        # CLAHE checkbox controls both clip and grid parameters
        self.clahe_check.toggled.connect(self.clahe_grid_spin.setEnabled)
        self.clahe_check.toggled.connect(self.clahe_grid_slider.setEnabled)

        # Preview
        self.preview_check.toggled.connect(self.toggle_preview)

        # Action buttons
        self.preprocess_btn.clicked.connect(self.run_preprocessing)  # Add this line
        self.reset_btn.clicked.connect(self.reset_parameters)  # Add this line

        # Viewer dims change
        if self.viewer is not None:
            self.viewer.dims.events.current_step.connect(self.update_preview_frame)

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        removed_layer = event.value

        if removed_layer == self.preview_layer:
            logger.debug("Preview layer was removed")
            self.preview_layer = None
            self.preview_enabled = False
            self.preview_check.setChecked(False)

        if removed_layer == self.original_layer:
            logger.debug("Original layer was removed")
            self.original_layer = None
            if self.preview_layer is not None:
                self.viewer.layers.remove(self.preview_layer)
                self.preview_layer = None
            self.preview_enabled = False
            self.preview_check.setChecked(False)

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode"""
        self.preview_enabled = enabled

        try:
            if enabled:
                if self.original_layer is None:
                    self.original_layer = self._get_active_image_layer()
                    if self.original_layer is None:
                        raise ProcessingError("No image layer found")

                if self.preview_layer is None:
                    preview_data = (self.original_layer.data[0] if self.original_layer.data.ndim == 3
                                    else self.original_layer.data)
                    self.preview_layer = self.viewer.add_image(
                        np.zeros_like(preview_data),
                        name='Preview',
                        visible=True
                    )

                self.update_preview_frame()
            else:
                if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.preview_layer)
                self.preview_layer = None

        except Exception as e:
            self.preview_check.setChecked(False)
            self.preview_enabled = False
            if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                self.viewer.layers.remove(self.preview_layer)
            self.preview_layer = None
            raise ProcessingError("Preview failed", str(e))

    def update_preview_frame(self, event: Optional[Event] = None):
        """Update the preview for the current frame"""
        if not self.preview_enabled or self.original_layer is None:
            return

        try:
            if self.original_layer.data.ndim == 3:
                current_step = self.viewer.dims.current_step[0]
                frame = self.original_layer.data[current_step].copy()
            else:
                frame = self.original_layer.data.copy()

            if frame.ndim != 2:
                raise ProcessingError(f"Invalid frame dimensions: {frame.shape}")

            # Process frame
            frame_8bit = self.preprocessor.convert_to_8bit(frame)
            processed_frame, info = self.preprocessor.preprocess_frame(frame_8bit)

            # Update preview
            if self.preview_layer is None:
                self.preview_layer = self.viewer.add_image(
                    np.zeros_like(frame_8bit, dtype=np.uint8),
                    name='Preview',
                    visible=True
                )

            self.preview_layer.data = processed_frame
            self.preview_layer.contrast_limits = (0, 255)

            # Update status
            info_text = (
                f"Preview - Original range: ({frame.min():.0f}, {frame.max():.0f})\n"
                f"Original mean: {frame.mean():.1f}, std: {frame.std():.1f}\n"
                f"Final mean: {info['final_mean']:.1f}"
            )
            self._update_status(info_text)

        except Exception as e:
            raise ProcessingError("Preview failed", str(e))

    def update_parameters(self):
        """Update preprocessing parameters from UI controls"""
        try:
            params = PreprocessingParameters(
                min_intensity=self.current_min_intensity,
                max_intensity=self.current_max_intensity,
                enable_median_filter=self.median_check.isChecked(),
                median_filter_size=self.median_size_spin.value(),
                enable_gaussian_filter=self.gaussian_check.isChecked(),
                gaussian_sigma=self.gaussian_sigma_spin.value(),
                enable_clahe=self.clahe_check.isChecked(),
                clahe_clip_limit=self.clahe_clip_spin.value(),
                clahe_grid_size=self.clahe_grid_spin.value()
            )

            params.validate()
            self.preprocessor.update_parameters(params)

            self._update_status("Parameters updated")
            self.parameters_updated.emit()

            if self.preview_enabled:
                self.update_preview_frame()

        except ValueError as e:
            raise ProcessingError("Invalid parameters", str(e))

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        # Reset intensity range
        self.intensity_slider.setValue((0, 255))
        self.min_spin.setValue(0)
        self.max_spin.setValue(255)
        self.current_min_intensity = 0
        self.current_max_intensity = 255

        # Reset filters
        self.median_check.setChecked(False)
        self.median_size_spin.setValue(3)
        self.median_slider.setValue(0)

        self.gaussian_check.setChecked(False)
        self.gaussian_sigma_spin.setValue(1.0)
        self.gaussian_slider.setValue(10)

        self.clahe_check.setChecked(False)
        self.clahe_clip_spin.setValue(16.0)
        self.clahe_clip_slider.setValue(160)
        self.clahe_grid_spin.setValue(16)
        self.clahe_grid_slider.setValue(16)

        self._update_status("Parameters reset to defaults")
        self.update_parameters()

    def _update_from_intensity_slider(self, values):
        """Update spinboxes when intensity range slider changes"""
        min_val, max_val = values
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)

        self.min_spin.setValue(min_val)
        self.max_spin.setValue(max_val)

        self.current_min_intensity = min_val
        self.current_max_intensity = max_val

        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

        self.update_parameters()

    def _update_from_intensity_spinboxes(self):
        """Update intensity range slider when spinboxes change"""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()

        # Ensure min <= max
        if min_val > max_val:
            if self.sender() == self.min_spin:
                max_val = min_val
                self.max_spin.setValue(max_val)
            else:
                min_val = max_val
                self.min_spin.setValue(min_val)

        self.current_min_intensity = min_val
        self.current_max_intensity = max_val

        self.intensity_slider.blockSignals(True)
        self.intensity_slider.setValue((min_val, max_val))
        self.intensity_slider.blockSignals(False)

        self.update_parameters()

    def run_preprocessing(self):
        """Run preprocessing on the entire stack"""
        if self.preview_enabled:
            self.preview_check.setChecked(False)

        try:
            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                raise ProcessingError("No image layer selected")

            # Disable controls during processing
            self._set_controls_enabled(False)
            self._update_status("Starting preprocessing...", 0)

            # Get and validate the image data
            stack = self._ensure_stack_format(active_layer.data)
            if not self._validate_input_data(stack):
                raise ProcessingError("Invalid input data format")

            total_frames = len(stack)
            processed_frames = []
            preprocessing_info = []

            # Process each frame
            for frame_idx in range(total_frames):
                progress = int(5 + (90 * frame_idx / total_frames))
                self._update_status(f"Processing frame {frame_idx + 1}/{total_frames}", progress)

                # Get frame
                frame = stack[frame_idx].copy()

                # Convert to 8-bit and process
                try:
                    frame_8bit = self.preprocessor.convert_to_8bit(frame)
                    processed_frame, frame_info = self.preprocessor.preprocess_frame(frame_8bit)
                    processed_frames.append(processed_frame)
                    preprocessing_info.append(frame_info)
                except Exception as e:
                    raise ProcessingError(
                        f"Error processing frame {frame_idx}",
                        str(e)
                    )

            # Combine processed frames
            processed_stack = np.stack(processed_frames, axis=0)

            # Update visualization
            self._update_status("Updating visualization...", 95)

            # Remove existing preprocessed layer if it exists
            for layer in self.viewer.layers[:]:
                if layer.name == 'Preprocessed':
                    self.viewer.layers.remove(layer)

            # Add new preprocessed layer
            preprocessed_layer = self.viewer.add_image(
                processed_stack,
                name='Preprocessed',
                visible=True,
                metadata={'preprocessing_info': preprocessing_info}
            )

            # Set consistent contrast limits for 8-bit
            preprocessed_layer.contrast_limits = (0, 255)

            # Store results
            self.data_manager.preprocessed_data = processed_stack
            self.data_manager.preprocessing_info = preprocessing_info

            self._update_status("Preprocessing complete", 100)
            self.preprocessing_completed.emit(processed_stack, preprocessing_info)

        except ProcessingError as e:
            self._handle_error(e)
        except Exception as e:
            self._handle_error(ProcessingError(
                "Preprocessing failed",
                str(e),
                self.__class__.__name__
            ))
        finally:
            self._set_controls_enabled(True)

    def cleanup(self):
        """Clean up resources"""
        if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
            self.viewer.layers.remove(self.preview_layer)
        self.preview_layer = None
        self.original_layer = None
        super().cleanup()
