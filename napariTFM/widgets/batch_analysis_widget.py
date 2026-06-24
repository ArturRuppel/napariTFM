import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from qtpy.QtCore import Qt, Signal, QSettings
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QGridLayout, QListView,
    QPushButton, QFrame, QTreeView, QDialog,
    QMessageBox, QListWidget, QCheckBox, QLineEdit, QFileDialog,
)

from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.widgets._base_widget import BaseAnalysisWidget
from napariTFM.widgets._ui_style import section_label_style
from napariTFM.utilities.parameter_manager import ParameterManager


class BatchAnalysisWidget(BaseAnalysisWidget):
    """Widget for running batch analysis on multiple folders."""

    batch_completed = Signal(dict)  # Emits results when batch processing completes
    action_states_changed = Signal()  # Emitted when the header run action's enablement changes

    MESH_ALGORITHMS = {
        "Frontal-Del.": "Frontal-Del.",
        "Delaunay": "Delaunay",
        "MeshAdapt": "MeshAdapt",
        "BAMG": "BAMG",
        "FD Quads": "FD Quads",
        "Para. Pack": "Para. Pack"
    }

    # region === Initialization ===
    def __init__(self, viewer, data_manager, parameter_manager: ParameterManager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        self.parameter_manager = parameter_manager
        self.parameter_spins = {}
        self.parameter_combos = {}
        self.parameter_checks = {}
        self.folder_list = []

        self.blockSignals(True)
        try:
            self._setup_ui()
            self._connect_signals()
            self._update_ui_state()
        finally:
            self.blockSignals(False)

    def _setup_ui(self):
        """Set up the user interface.

        The widget hosts its groups directly; the workflow shell owns the
        single outer scroll area, so this stage no longer nests its own.
        """
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Add new file paths and metadata groups first
        main_layout.addWidget(self._create_file_paths_group())
        main_layout.addWidget(self._create_metadata_group())

        # Batch execution consumes the shared workflow parameters from
        # ParameterManager; this panel only owns batch-specific inputs. All
        # pipeline steps are mandatory and every visualization is produced, so
        # there are no per-step / per-viz checkboxes here anymore.
        main_layout.addWidget(self._create_folder_management_group())
        main_layout.addWidget(self._create_status_frame())

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect widget signals."""
        self.add_folder_btn.clicked.connect(self._add_folder)
        self.clear_folders_btn.clicked.connect(self._clear_folders)
        self.run_analysis_btn.clicked.connect(self._run_batch_analysis)
        self.save_config_btn.clicked.connect(self._save_config_dialog)
        self.load_config_btn.clicked.connect(self._load_config_dialog)

    # endregion === Initialization ===

    # region === UI Creation Groups ===
    def _create_metadata_group(self) -> QGroupBox:
        """Create group for metadata fields."""
        group = QGroupBox("Metadata")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        metadata_fields = [
            ("experiment_date", "Experiment Date:"),
            ("cell_type", "Cell Type:"),
            ("substrate", "Substrate:", "PA gel"),  # Default value from config
            ("notes", "Notes:")
        ]

        self.metadata_inputs = {}
        for key, label, *default in metadata_fields:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            line_edit = QLineEdit()
            if default:  # Set default value if provided
                line_edit.setText(default[0])
            self.metadata_inputs[key] = line_edit
            row.addWidget(line_edit)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_file_paths_group(self) -> QGroupBox:
        """Create group for input/output file paths."""
        group = QGroupBox("File Paths")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Input files section
        input_label = QLabel("Input Files:")
        input_label.setStyleSheet(section_label_style())
        layout.addWidget(input_label)

        input_files = [
            ("beads", "Beads File:", "beads.tif"),
            ("reference", "Reference File:", "reference.tif"),
            ("cells", "Cells File (optional):", "cells.tif")
        ]

        self.file_inputs = {}
        for key, label, default in input_files:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            line_edit = QLineEdit(default)
            self.file_inputs[key] = line_edit
            row.addWidget(line_edit)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_folder_management_group(self) -> QGroupBox:
        """Create folder management group."""
        group = QGroupBox("Folder Management")
        layout = QVBoxLayout()

        # Folder list first
        self.folder_list_widget = QListWidget()
        layout.addWidget(self.folder_list_widget)

        # Create grid layout for buttons
        button_layout = QGridLayout()

        # Add folder management buttons
        self.add_folder_btn = QPushButton("Add Folder")
        self.clear_folders_btn = QPushButton("Clear Folders")
        self.save_config_btn = QPushButton("Save Config")
        self.load_config_btn = QPushButton("Load Config")
        button_layout.addWidget(self.add_folder_btn, 0, 0)
        button_layout.addWidget(self.clear_folders_btn, 0, 1)
        button_layout.addWidget(self.save_config_btn, 1, 0)
        button_layout.addWidget(self.load_config_btn, 1, 1)

        # Opt-in stage-resume cache: write preprocessed images to disk so a later
        # run can resume without re-preprocessing. Off by default — the .ntfm is
        # the deliverable; displacement/force/stress resume from it (ROADMAP §4).
        self.save_cache_checkbox = QCheckBox("Save preprocessed image cache (.tif)")
        self.save_cache_checkbox.setChecked(False)
        layout.addWidget(self.save_cache_checkbox)

        # Add run button — batch always runs in-process (a headless CLI is the
        # future home for out-of-process runs).
        self.run_analysis_btn = QPushButton("Run Analysis")
        layout.addWidget(self.run_analysis_btn)

        layout.addLayout(button_layout)
        group.setLayout(layout)
        return group

    def _create_status_frame(self) -> QFrame:
        """Create status frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    # endregion === UI Creation Groups ===

    # region === Configuration Interface ===

    def _generate_config(self) -> dict:
        """Generate configuration dictionary from UI values"""
        # Get folder paths
        folders = []
        for i in range(self.folder_list_widget.count()):
            folders.append(self.folder_list_widget.item(i).text())

        # Get file names
        input_files = {
            'beads': self.file_inputs['beads'].text(),
            'reference': self.file_inputs['reference'].text()
        }

        if self.file_inputs['cells'].text():
            input_files['cells'] = self.file_inputs['cells'].text()

        # Every pipeline step is mandatory now — the per-step checkboxes are gone.
        # MSM/stress optionality moves to the interactive stage glyph and the
        # unified run path (TODO P3/P4); the batch path runs the full pipeline.
        analysis_steps = {
            'preprocessing': True,
            'displacement': True,
            'force': True,
            'stress': True,
            'calculate_metrics': True,
        }

        # Every visualization is produced; per-stage viz selection returns as a
        # header glyph (TODO P5). Bead overlay was removed entirely.
        visualizations = {
            'displacement_map': True,
            'force_map': True,
            'force_cell_overlay': True,
            'sigma_xx': True,
            'sigma_yy': True,
            'normal_stress': True,
            'mesh': True,
        }

        parameters = self.parameter_manager.get_all_parameters()

        # Add metrics parameters. Masks are always supplied externally (napariTFM
        # does not generate them), so no mask-creation/refinement parameters here.
        # Use default values as these detailed controls are not in the batch UI.
        metrics_parameters = {
            'calculate_strain_energy': True,  # Default
            'calculate_polarization': True,  # Default
            'export_eigenvalues': True  # Default
        }
        config = {
            'root_folders': folders,
            'input_files': input_files,
            'analysis_steps': analysis_steps,
            'visualizations': visualizations,
            'parameters': parameters,
            'metrics_parameters': metrics_parameters,  # New section
            'save_cache': self.save_cache_checkbox.isChecked(),
        }

        return config

    def _save_config_dialog(self):
        """Open a file dialog to save the configuration file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            os.path.expanduser("~"),
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )

        if filepath:
            try:
                self.save_config_to_yaml(filepath)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Configuration saved successfully to:\n{filepath}"
                )
            except IOError as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save configuration:\n{str(e)}"
                )

    def _load_config_dialog(self):
        """Open a file dialog to load a configuration file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            os.path.expanduser("~"),
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )

        if filepath:
            try:
                self.load_config_from_yaml(filepath)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Configuration loaded successfully from:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load configuration:\n{str(e)}"
                )

    def save_config_to_yaml(self, filepath: str) -> None:
        """
        Save the current configuration to a YAML file.

        Parameters:
        -----------
        filepath : str
            The path where the YAML file should be saved. If no extension is provided,
            '.yaml' will be added automatically.

        Raises:
        -------
        IOError: If there is an error writing to the file
        """
        # Add .yaml extension if not present
        if not filepath.lower().endswith(('.yml', '.yaml')):
            filepath += '.yaml'

        try:
            # Generate configuration dictionary
            config = self._generate_config()

            # Save to YAML file
            with open(filepath, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        except Exception as e:
            raise IOError(f"Failed to save configuration to {filepath}: {str(e)}")

    def load_config_from_yaml(self, filepath: str) -> None:
        """
        Load configuration from a YAML file and update the widget state.

        Parameters:
        -----------
        filepath : str
            Path to the YAML configuration file

        Raises:
        -------
        IOError: If there is an error reading the file
        ValueError: If the configuration format is invalid
        """
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)

            # Block signals during update
            self.blockSignals(True)
            try:
                # Update input files
                for key, input_widget in self.file_inputs.items():
                    if key in config.get('input_files', {}):
                        input_widget.setText(config['input_files'][key])

                # Update metadata
                for key, input_widget in self.metadata_inputs.items():
                    if key in config.get('metadata', {}):
                        input_widget.setText(config['metadata'][key])

                failed_updates = self._apply_config_parameters(config.get('parameters', {}))

                # Stage-resume cache toggle
                if 'save_cache' in config:
                    self.save_cache_checkbox.setChecked(bool(config['save_cache']))

                # Analysis steps and visualizations are no longer user-selectable
                # (all steps mandatory, all visualizations produced) — older config
                # files may still carry those sections; they are simply ignored.

                # Clear and update folder list
                self.folder_list.clear()
                self.folder_list_widget.clear()
                for folder in config.get('root_folders', []):
                    if os.path.exists(folder):
                        self.folder_list.append(folder)
                        self.folder_list_widget.addItem(folder)

                if failed_updates:
                    error_msg = "Failed to update the following parameters:\n"
                    for param, error in failed_updates:
                        error_msg += f"- {param}: {error}\n"
                    raise ValueError(error_msg)

            finally:
                self.blockSignals(False)

            # Update UI state
            self._update_ui_state()

        except Exception as e:
            raise IOError(f"Failed to load configuration from {filepath}: {str(e)}")

    # endregion === Configuration Interface ===

    # region === Parameter Management ===
    def _apply_config_parameters(self, params: dict) -> list[tuple[str, str]]:
        """Apply YAML parameters to the shared ParameterManager.

        Unknown keys are ignored so older config files with retired parameters
        such as TV-L1 settings remain loadable.
        """
        failed_updates = []
        valid_parameters = set(self.parameter_manager.get_all_parameters())

        for name, value in params.items():
            if name not in valid_parameters:
                continue
            try:
                if name == 'registration_mode' and isinstance(value, str):
                    value = value.lower()
                self.parameter_manager.set_parameter(name, value)
            except Exception as e:
                failed_updates.append((name, str(e)))

        return failed_updates

    # endregion === Parameter Management ===

    # region === Folder Management ===
    def _add_folder(self):
        """Add multiple folders to analysis queue."""
        # Try to load last directory from QSettings
        settings = QSettings()
        last_dir = settings.value("BatchAnalysis/last_folder", os.path.expanduser("~"))

        # Create file dialog
        dialog = QFileDialog(self, "Select Data Folders", last_dir)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.DontUseNativeDialog)

        # Enable multiple selection
        tree_view = dialog.findChild(QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QTreeView.MultiSelection)

        list_view = dialog.findChild(QListView, "listView")
        if list_view:
            list_view.setSelectionMode(QListView.MultiSelection)

        # Show dialog and process results
        if dialog.exec_() == QDialog.Accepted:
            folders = dialog.selectedFiles()

            if folders:
                # Save the last selected directory
                settings.setValue("BatchAnalysis/last_folder", os.path.dirname(folders[0]))

                # Process each selected folder
                invalid_folders = []
                duplicate_folders = []
                added_folders = []

                for folder in folders:
                    try:
                        # Check if folder is already in the list
                        if folder in self.folder_list:
                            duplicate_folders.append(folder)
                            continue

                        # Validate folder contents
                        missing_files = self._check_folder_contents(folder)
                        if not missing_files:
                            self.folder_list.append(folder)
                            self.folder_list_widget.addItem(folder)
                            added_folders.append(folder)
                        else:
                            invalid_folders.append((folder, missing_files))

                    except Exception as e:
                        invalid_folders.append((folder, str(e)))

                # Show summary message if there are any issues
                if duplicate_folders or invalid_folders:
                    messages = []
                    if added_folders:
                        messages.append(f"Successfully added {len(added_folders)} folder(s)")
                    if duplicate_folders:
                        messages.append(f"Skipped {len(duplicate_folders)} duplicate folder(s)")
                    if invalid_folders:
                        messages.append("The following folders had errors:")
                        for folder, error in invalid_folders:
                            if isinstance(error, list):
                                messages.append(f"- {os.path.basename(folder)}: Missing files: {', '.join(error)}")
                            else:
                                messages.append(f"- {os.path.basename(folder)}: {error}")

                    QMessageBox.information(
                        self,
                        "Folder Selection Results",
                        "\n".join(messages)
                    )

                self._update_ui_state()

    def _clear_folders(self):
        """Clear folder list."""
        self.folder_list.clear()
        self.folder_list_widget.clear()
        self._update_ui_state()

    def _check_folder_contents(self, folder: str) -> list:
        """
        Check folder for required files based on config input file names.

        Parameters:
        -----------
        folder : str
            Path to folder to check

        Returns:
        --------
        list
            List of missing required files
        """
        # Get required file names from input file fields
        required_files = [
            self.file_inputs['beads'].text(),
            self.file_inputs['reference'].text()
        ]

        # Check for optional cell file only if specified
        cell_filename = self.file_inputs['cells'].text()
        if cell_filename and cell_filename != "cells.tif":  # Don't check default value
            required_files.append(cell_filename)

        missing = []
        for file in required_files:
            if not file:  # Skip empty filenames
                continue
            if not os.path.exists(os.path.join(folder, file)):
                missing.append(file)
        return missing

    def _validate_all_folders(self) -> bool:
        """Validate all folders in the list still exist and contain required files."""
        invalid_folders = []
        for folder in self.folder_list[:]:  # Create a copy to iterate
            if not os.path.exists(folder):
                invalid_folders.append((folder, "Folder no longer exists"))
                continue

            missing_files = self._check_folder_contents(folder)
            if missing_files:
                invalid_folders.append((folder, f"Missing files: {', '.join(missing_files)}"))

        if invalid_folders:
            message = "The following folders are invalid:\n\n"
            for folder, reason in invalid_folders:
                message += f"{folder}\n{reason}\n\n"
            message += "Would you like to remove these folders from the list?"

            reply = QMessageBox.question(
                self,
                "Invalid Folders",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                for folder, _ in invalid_folders:
                    self.folder_list.remove(folder)
                    # Find and remove item from QListWidget
                    items = self.folder_list_widget.findItems(folder, Qt.MatchExactly)
                    for item in items:
                        self.folder_list_widget.takeItem(self.folder_list_widget.row(item))
                self._update_ui_state()

            return False

        return True

    # endregion === Folder Management ===

    # region === State Management ===
    def action_states(self) -> dict:
        """Report the stage header's run-action enablement (run only)."""
        return {"run": len(self.folder_list) > 0}

    def _update_ui_state(self):
        """Update UI element states."""
        has_folders = len(self.folder_list) > 0
        self.run_analysis_btn.setEnabled(has_folders)
        self.clear_folders_btn.setEnabled(has_folders)
        self.action_states_changed.emit()

    # endregion === State Management ===

    # region === Execution ===

    def _run_batch_analysis(self):
        """Run batch analysis according to selected console option."""
        if not self.folder_list:
            QMessageBox.warning(self, "No Folders", "Please add folders to analyze first.")
            return

        # Validate folders before proceeding
        if not self._validate_all_folders():
            return

        try:
            # Generate configuration dictionary
            config = self._generate_config()

            # Create temporary YAML file to store configuration
            with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_yaml:
                yaml.safe_dump(config, temp_yaml, default_flow_style=False)
                config_path = temp_yaml.name

            # Run directly in the napari process.
            print("Starting batch analysis...")

            # Create output directories
            for folder in config["root_folders"]:
                tfm_data_dir = Path(folder) / "TFM_data"
                tfm_data_dir.mkdir(exist_ok=True)

            # Run analysis
            analyzer = BatchAnalysis(config)
            analyzer.process_all_folders()

            # Clean up config file
            Path(config_path).unlink()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start batch analysis: {str(e)}"
            )
            # Clean up temporary files
            if 'config_path' in locals():
                try:
                    Path(config_path).unlink()
                except:
                    pass

    # endregion === Execution ===
