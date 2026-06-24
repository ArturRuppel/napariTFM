import os
import sys

# Set Qt to offscreen mode for headless/console execution (must be before any Qt imports)
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from time import sleep
from time import time
from typing import Optional, Dict, List, Any

import numpy as np
import tifffile
import yaml
from skimage.transform import resize
from scipy.ndimage import center_of_mass
import pandas as pd

from napariTFM.backend.batch_analysis_visualizations import BatchVisualizationSaver
from napariTFM.backend.displacement_analysis import (
    DisplacementResult,
    calculate_displacement_field,
)
from napariTFM.backend.fttc import FTTCResult, calculate_force_field
from napariTFM.backend.msm import calculate_stresses, generate_mesh_stack
from napariTFM.backend.parameter_dataclasses import DisplacementParameters, FTTCParameters, MSMParameters, PreprocessingParameters, UnifiedParameters
from napariTFM.backend.preprocessing import preprocess_frame, preprocess_stack
from napariTFM.backend.metrics_calculator import calculate_strain_energy_density, calculate_total_strain_energy, \
    calculate_moment_tensor, calculate_polarization


# TODO black image when only one frame for cell-force overlay visualization
class TeeLogger:
    """Custom logger that captures print statements and logging output to both console and file."""

    def __init__(self, filename: Path, config: dict = None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = open(filename, 'w', encoding='utf-8')
        self.start_time = datetime.now()

        # Write header to log file
        self.print_banner()
        self.log.write(f"Processing started at: {self.start_time}\n")
        self.log.write("-" * 50 + "\n\n")
        self.terminal.write(f"Processing started at: {self.start_time}\n")
        self.terminal.write("-" * 50 + "\n\n")

        # Log configuration parameters if provided
        if config:
            self.log.write("Analysis Parameters:\n")
            self.log.write("-" * 20 + "\n")
            self.terminal.write("Analysis Parameters:\n")
            self.terminal.write("-" * 20 + "\n")

            # Log analysis steps
            self.log.write("\nEnabled Analysis Steps:\n")
            self.terminal.write("\nEnabled Analysis Steps:\n")
            for step, enabled in config.get('analysis_steps', {}).items():
                self.log.write(f"- {step}: {'Yes' if enabled else 'No'}\n")
                self.terminal.write(f"- {step}: {'Yes' if enabled else 'No'}\n")

            # Log key parameters
            self.log.write("\nKParameters:\n")
            self.terminal.write("\nParameters:\n")
            params = config.get('parameters', {})
            for key, value in sorted(params.items()):
                self.log.write(f"- {key}: {value}\n")
                self.terminal.write(f"- {key}: {value}\n")

            self.log.write("\n" + "-" * 50 + "\n\n")
            self.terminal.write("\n" + "-" * 50 + "\n\n")

    def print_banner(self):
        separator = "-" * 80
        banner = '''
 ----------------------------------------------------------------------------- 
|                                                                             |
|                                         ,--.,--------.,------.,--.   ,--.   |
|   ,--,--,  ,--,--. ,---.  ,--,--.,--.--.`--''--.  .--'|  .---'|   `.'   |   |
|   |      \| ,-.  || .-. |' ,-.  ||  .--',--.   |  |   |  `--, |  |'.'|  |   |
|   |  ||  |\ '-'  || '-' '\ '-'  ||  |   |  |   |  |   |  |`   |  |   |  |   |
|   `--''--' `--`--'|  |-'  `--`--'`--'   `--'   `--'   `--'    `--'   `--'   |
|                   `--'                                                      |
|                                                                             |
|                   Traction Force Microscopy Analysis Tool                   |
 ----------------------------------------------------------------------------- '''

        contact_info = '''                 
            For comments, questions or bug reports, please contact:
                           artur.ruppel@pasteur.fr
                    https://github.com/ArturRuppel/napariTFM'''

        # Combined output for both terminal and log file
        output = (
            f"\n{separator}\n"
            f"{banner}\n"
            f"{contact_info}\n"
            f"{separator}\n\n"
        )

        # Write to both outputs
        self.terminal.write(output)
        self.log.write(output)

        # Flush both outputs to ensure immediate writing
        self.flush()

        # Only pause the terminal output
        sleep(2)

    def write(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        self.terminal.write(formatted_message)
        self.log.write(formatted_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        end_time = datetime.now()
        duration = end_time - self.start_time

        # Write footer with timing information
        self.log.write("\n" + "-" * 50 + "\n")
        self.log.write("Analysis Summary:\n")
        self.log.write(f"Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"Duration: {duration}\n")

        self.terminal.write("\n" + "-" * 50 + "\n")
        self.terminal.write("Analysis Summary:\n")
        self.terminal.write(f"Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.terminal.write(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.terminal.write(f"Duration: {duration}\n")

        # Calculate hours, minutes, seconds for more readable format
        total_seconds = int(duration.total_seconds())  # Convert to integer for second resolution
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        time_str = ""
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        self.log.write(f"Total time: {time_str}\n")
        self.terminal.write(f"Total time: {time_str}\n")

        self.log.close()
        sys.stdout = self.terminal


class BatchAnalysis:
    """Handles batch analysis of TFM data using service layer components."""

    def __init__(self, config: dict):
        self.config = config
        self._tee_logger = None

    def _format_duration(self, seconds: float) -> str:
        """Format duration in appropriate units (seconds or minutes)."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        return f"{seconds/60:.1f} minutes"

    def process_all_folders(self) -> None:
        """
        Process all folders specified in the configuration for TFM analysis.

        This is the main entry point for batch processing multiple experiment folders.
        It iterates through each folder path specified in config['root_folders'] and
        processes them sequentially.

        The method handles:
        - Preprocessing of bead and cell images
        - Displacement field calculation
        - Force analysis
        - Stress analysis (consumes externally supplied masks)
        - Visualization generation

        Each folder's results are saved in a 'TFM_data' subdirectory, with detailed
        logs capturing the processing steps and any issues encountered.

        Raises:
            FileNotFoundError: If a specified folder doesn't exist
            RuntimeError: If processing fails for any folder
        """
        for folder in self.config['root_folders']:
            self.process_folder(folder)

    def process_folder(self, folder_path: str) -> None:
        """
        Process a single folder containing TFM experiment data.

        This method executes the complete TFM analysis pipeline on a single experimental
        dataset, creating a 'TFM_data' subdirectory for results.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the raw experimental data.
            Must include the files specified in config['input_files'].

        Processing Steps
        ---------------
        1. Preprocessing:
            - Processes bead and cell images if available
            - Applies background subtraction and filtering
            - Saves preprocessed images as calibrated TIFFs

        2. Displacement Analysis:
            - Calculates displacement fields from bead images
            - Saves displacement data as NumPy arrays

        3. Force Analysis:
            - Computes traction forces using FTTC
            - Saves force fields and related metrics

        4. Stress Analysis:
            - Calculates internal stress fields
            - Saves stress tensors and quality metrics

        Each step is conditional on the corresponding flag in config['analysis_steps']
        being True. Visualizations are generated based on config['visualizations'].

        Results
        -------
        Creates a 'TFM_data' subdirectory containing:
            - preprocessed_beads.tif: Processed bead images
            - preprocessed_reference.tif: Processed reference image
            - preprocessed_cells.tif: Processed cell images (if available)
            - displacements.npy: Displacement field data
            - traction_forces.npy: Force field data
            - masks.tif: Externally supplied binary masks (required input for
              stress analysis; not generated by napariTFM)
            - stress_results.npy: Stress tensor data
            - processing_log.txt: Detailed processing log
            - Various visualization files based on configuration

        Raises
        ------
        FileNotFoundError
            If required input files are missing
        RuntimeError
            If any processing step fails
        ValueError
            If input data is invalid or corrupted
        """
        folder = Path(folder_path)
        tfm_folder = self._initialize_folder(folder)
        viz_saver = BatchVisualizationSaver(folder)

        try:
            print(f"Processing folder: {folder_path}")
            print("=" * 50)

            # Handle preprocessing
            preprocessed_data = self._handle_preprocessing_execution(folder, tfm_folder)
            self._handle_visualization(tfm_folder, viz_saver, 'preprocessing', preprocessed_data)

            # Handle displacement
            displacement_data = self._handle_displacement_execution(tfm_folder, preprocessed_data)
            self._handle_visualization(tfm_folder, viz_saver, 'displacement', displacement_data)

            # Handle force analysis
            force_data = self._handle_force_execution(tfm_folder, displacement_data)
            self._handle_visualization(tfm_folder, viz_saver, 'force', force_data)

            # Masks are supplied externally (loaded from masks.tif by downstream steps)
            mask_data = None

            # Handle stress analysis
            stress_data = self._handle_stress_execution(tfm_folder, force_data, mask_data)
            self._handle_visualization(tfm_folder, viz_saver, 'stress', stress_data)

            # Handle metrics calculation
            if self.config['analysis_steps'].get('calculate_metrics', False):
                self._handle_metrics_execution(tfm_folder, displacement_data, force_data, mask_data)

            print("Folder processing completed successfully!")
            print("=" * 50)

        finally:
            self._cleanup()

    def _handle_preprocessing_execution(self, folder: Path, tfm_folder: Path) -> Optional[dict]:
        """Handle preprocessing execution. Always runs if enabled."""
        if not self.config['analysis_steps']['preprocessing']:
            return None

        try:
            return self._execute_preprocessing(folder, tfm_folder)
        except Exception as e:
            print(f"Preprocessing failed: {str(e)}")
            return None

    def _handle_displacement_execution(self, tfm_folder: Path, preprocessed_data: Optional[dict]) -> Optional[dict]:
        """Handle displacement analysis execution. Always runs if enabled."""
        if not self.config['analysis_steps']['displacement']:
            return None

        try:
            if preprocessed_data is None:
                print("Loading preprocessed images from file...")
                try:
                    preprocessed_bead_stack = tifffile.imread(str(tfm_folder / "preprocessed_beads.tif"))
                    preprocessed_reference = tifffile.imread(str(tfm_folder / "preprocessed_reference.tif"))
                    preprocessed_data = {
                        'beads': preprocessed_bead_stack,
                        'reference': preprocessed_reference,
                    }
                except Exception as e:
                    print(f"Could not load preprocessed files: {str(e)}")
                    return None

            return self._execute_displacement_analysis(tfm_folder, preprocessed_data)
        except Exception as e:
            print(f"Displacement analysis failed: {str(e)}")
            return None

    def _handle_force_execution(self, tfm_folder: Path, displacement_data: Optional[dict]) -> Optional[dict]:
        """Handle force analysis execution. Always runs if enabled."""
        if not self.config['analysis_steps']['force']:
            return None

        try:
            if displacement_data is None:
                print("Loading displacement data from file...")
                try:
                    displacement_data = np.load(str(tfm_folder / "displacements.npy"), allow_pickle=True).item()
                except Exception as e:
                    print(f"Could not load displacement data: {str(e)}")
                    return None
            return self._execute_force_analysis(tfm_folder, displacement_data)
        except Exception as e:
            print(f"Force analysis failed: {str(e)}")
            return None

    def _handle_stress_execution(self, tfm_folder: Path, force_data: Optional[dict], mask_data: Optional[np.ndarray]) -> Optional[dict]:
        """Handle stress analysis execution. Always runs if enabled."""
        if not self.config['analysis_steps']['stress']:
            return None

        try:
            if force_data is None:
                print("Loading force data from file...")
                try:
                    force_data = np.load(str(tfm_folder / "traction_forces.npy"), allow_pickle=True).item()
                except Exception as e:
                    print(f"Could not load force data: {str(e)}")
                    return None

            print("Loading externally supplied mask data...")
            try:
                if mask_data is None:
                    mask_data = tifffile.imread(str(tfm_folder / "masks.tif"))
            except Exception as e:
                print(f"Stress analysis requires an external mask (masks.tif) in "
                      f"{tfm_folder}; napariTFM does not generate masks. "
                      f"Could not load mask data: {str(e)}")
                return None

            return self._execute_stress_analysis(tfm_folder, mask_data, force_data)

        except Exception as e:
            print(f"Stress analysis failed: {str(e)}")
            return None

    def _handle_metrics_execution(self, tfm_folder: Path,
                                  displacement_results: Optional[DisplacementResult],
                                  force_results: Optional[FTTCResult],
                                  mask_data: Optional[np.ndarray]) -> None:
        """Handle metrics calculation (strain energy, polarization)."""
        if not self.config['analysis_steps'].get('calculate_metrics', False):
            print("Metrics calculation skipped.")
            return

        print("Starting Metrics Calculation...")
        start_time = time()

        try:
            # Load displacement data if not provided
            if displacement_results is None:
                print("Loading displacement data from file for metrics...")
                try:
                    displacement_results = np.load(str(tfm_folder / "displacements.npy"), allow_pickle=True).item()
                except Exception as e:
                    print(f"Could not load displacement data for metrics: {str(e)}")
                    return

            # Load force data if not provided
            if force_results is None:
                print("Loading force data from file for metrics...")
                try:
                    force_results = np.load(str(tfm_folder / "traction_forces.npy"), allow_pickle=True).item()
                except Exception as e:
                    print(f"Could not load force data for metrics: {str(e)}")
                    return

            # Load mask data if not provided
            if mask_data is None:
                print("Loading mask data from file for metrics...")
                try:
                    mask_data = tifffile.imread(str(tfm_folder / "masks.tif"))
                except Exception as e:
                    print(f"Could not load mask data for metrics: {str(e)}")
                    return

            displacements_um = displacement_results.displacement_field  # (t, y, x, 2) in µm
            forces_pa = force_results.force_field  # (t, y, x, 2) in Pa

            if displacements_um.shape[0] != forces_pa.shape[0] or \
               (mask_data.ndim == 3 and displacements_um.shape[0] != mask_data.shape[0]):
                print("Mismatch in number of frames between displacements, forces, or masks. Skipping metrics.")
                return

            # Ensure mask_data is 3D (t, y, x)
            if mask_data.ndim == 2:
                mask_data = mask_data[np.newaxis, ...]

            # Resize masks to match force field resolution if needed
            force_shape = forces_pa.shape[1:3]  # (height, width)
            mask_shape = mask_data.shape[1:3]  # (height, width)
            if mask_shape != force_shape:
                print(f"Resizing masks from {mask_shape} to {force_shape} to match force field...")
                mask_data = np.stack([
                    resize(
                        mask.astype(float),
                        force_shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False
                    ) > 0.5
                    for mask in mask_data
                ])

            num_frames = displacements_um.shape[0]
            h, w = displacements_um.shape[1:3]

            pixel_size_config_um = self.config['parameters']['pixel_size']  # µm
            downscale_factor = self.config['parameters']['downscale_factor']
            pixel_size_at_calc_resolution_um = pixel_size_config_um * downscale_factor
            pixel_size_at_calc_resolution_m = pixel_size_at_calc_resolution_um * 1e-6
            pixel_area_m2 = pixel_size_at_calc_resolution_m ** 2

            metrics_results_list: List[Dict[str, Any]] = []

            for i in range(num_frames):
                disp_frame_m = displacements_um[i] * 1e-6  # Convert µm to m
                force_frame_pa = forces_pa[i]
                mask_frame = mask_data[i] if mask_data.ndim == 3 else mask_data # Handle single mask

                # 1. Strain Energy
                sed_jm2 = calculate_strain_energy_density(disp_frame_m, force_frame_pa)
                total_se_j = calculate_total_strain_energy(sed_jm2, mask_frame, pixel_area_m2)

                # 2. Moment Tensor & Polarization
                if np.any(mask_frame): # Calculate centroid if mask is not empty
                    centroid_r, centroid_c = center_of_mass(mask_frame)
                    if np.isnan(centroid_r) or np.isnan(centroid_c): # Fallback if center_of_mass fails
                        centroid_r, centroid_c = (h - 1) / 2.0, (w - 1) / 2.0
                else: # Mask is empty, use image center
                    centroid_r, centroid_c = (h - 1) / 2.0, (w - 1) / 2.0

                yy, xx = np.mgrid[0:h, 0:w]
                pixel_positions_m_x = (xx - centroid_c) * pixel_size_at_calc_resolution_m
                pixel_positions_m_y = (yy - centroid_r) * pixel_size_at_calc_resolution_m
                pixel_positions_m = np.stack((pixel_positions_m_x, pixel_positions_m_y), axis=-1)

                moment_t_nm = calculate_moment_tensor(force_frame_pa, mask_frame, pixel_positions_m, pixel_area_m2)
                polar_idx, eig1_nm, eig2_nm = calculate_polarization(moment_t_nm)

                metrics_results_list.append({
                    'Frame': i,
                    'Total Strain Energy (J)': total_se_j,
                    'Polarization Index': polar_idx,
                    'Eigenvalue1 (N.m)': eig1_nm,
                    'Eigenvalue2 (N.m)': eig2_nm
                })
                print(f"Metrics frame {i+1}/{num_frames}: SE={total_se_j:.2e} J, Polar={polar_idx:.3f}")

            # Save to CSV
            df = pd.DataFrame(metrics_results_list)
            csv_path = tfm_folder / "metrics_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Metrics results saved to {csv_path}")

        except Exception as e:
            print(f"Metrics calculation failed: {str(e)}")
        finally:
            print(f"Metrics calculation completed in {self._format_duration(time() - start_time)}")

    def _execute_preprocessing(self, folder: Path, tfm_folder: Path) -> Optional[dict]:
        """
        Execute the preprocessing step of the TFM analysis pipeline.

        This method handles the initial processing of raw microscopy images,
        including both bead and cell images if available.

        Parameters
        ----------
        folder : Path
            Path to the input folder containing raw data files
        tfm_folder : Path
            Path to the output folder where processed files will be saved

        Returns
        -------
        Optional[dict]
            Dictionary containing:
            - 'beads': Preprocessed bead image stack (np.ndarray)
            - 'reference': Preprocessed reference image (np.ndarray)
            - 'cells': Preprocessed cell image stack (np.ndarray, optional)
            - 'parameters': Preprocessing parameters used
            Returns None if preprocessing fails

        Processing Steps
        ---------------
        1. Loads raw bead images and reference image
        2. Optionally loads cell images if specified in config
        3. Applies preprocessing pipeline:
            - Background subtraction
            - Gaussian filtering
            - Intensity normalization
            - Image registration (for bead images)
        4. Saves results as calibrated TIFF files with metadata

        The preprocessing parameters are taken from the config:
            - rolling_ball_radius
            - min_intensity_percentile
            - max_intensity_percentile
            - gaussian_sigma
            - registration_mode
            Plus additional parameters for cell image processing

        Raises
        ------
        FileNotFoundError
            If input files are not found
        RuntimeError
            If preprocessing operations fail
        """

        print("Starting Preprocessing...")
        start_time = time()
        params = self._create_preprocessing_parameters()

        # Process bead images
        bead_stack = tifffile.imread(str(folder / self.config['input_files']['beads']))
        reference = tifffile.imread(str(folder / self.config['input_files']['reference']))

        # Load and process cell images if available
        cell_stack = None
        if 'cells' in self.config['input_files'] and self.config['input_files']['cells']:
            try:
                cell_stack = tifffile.imread(str(folder / self.config['input_files']['cells']))
                print("Found cell image stack, will process alongside beads")
            except FileNotFoundError:
                print(f"Warning: Cell image file specified but not found: {self.config['input_files']['cells']}")

        bead_results = []
        for result, frame, total in preprocess_stack(bead_stack, params, reference):
            bead_results.append(result)
            print(f"Progress (beads): {(frame / total) * 100:.1f}%, Frame {frame}/{total}")

        reference_result = preprocess_frame(reference, params)

        # Process cell images if available
        cell_results = []
        if cell_stack is not None:
            print("Processing cell images...")
            for result, frame, total in preprocess_stack(cell_stack, params, reference_image=None, is_cell=True):
                cell_results.append(result)
                print(f"Progress (cells): {(frame / total) * 100:.1f}%, Frame {frame}/{total}")

        # Save results with calibration
        preprocessed = {
            'beads': np.stack([r.processed_image for r in bead_results]),
            'reference': reference_result.processed_image,
            'parameters': params.__dict__
        }

        if cell_results:
            preprocessed['cells'] = np.stack([r.processed_image for r in cell_results])

        pixel_size = self.config['parameters']['pixel_size']
        frame_interval = self.config['parameters']['frame_interval']

        self._save_calibrated_tiff(
            preprocessed['beads'],
            tfm_folder / "preprocessed_beads.tif",
            pixel_size,
            frame_interval
        )
        self._save_calibrated_tiff(
            preprocessed['reference'],
            tfm_folder / "preprocessed_reference.tif",
            pixel_size,
            frame_interval
        )
        if 'cells' in preprocessed:
            self._save_calibrated_tiff(
                preprocessed['cells'],
                tfm_folder / "preprocessed_cells.tif",
                pixel_size,
                frame_interval
            )

        print(f"Preprocessing completed in {self._format_duration(time() - start_time)}")
        return preprocessed

    def _execute_displacement_analysis(self, tfm_folder: Path, preprocessed_data: dict) -> Optional[DisplacementResult]:
        """
        Execute the displacement analysis step of the TFM analysis pipeline.

        This method calculates displacement fields from preprocessed bead images
        using optical flow techniques.

        Parameters
        ----------
        tfm_folder : Path
            Path to the output folder where processed files will be saved
        preprocessed_data : dict
            Dictionary containing preprocessed images:
            - 'beads': Preprocessed bead image stack (np.ndarray)
            - 'reference': Preprocessed reference image (np.ndarray)

        Returns
        -------
        Optional[DisplacementResult]
            Object containing:
            - displacement_field: Calculated displacement vectors (np.ndarray)
            - parameters: Displacement calculation parameters used
            Returns None if displacement analysis fails

        Processing Steps
        ---------------
        1. Loads raw bead images and reference image
        2. Optionally loads cell images if specified in config
        3. Applies preprocessing pipeline:
            - Background subtraction
            - Optical flow calculation (Farneback)
            - Optional downscaling and filtering
        4. Saves displacement field as NumPy array

        The displacement parameters are taken from the config:
            - nscales, inner_iterations, median_filtering (Farneback parameters)
            - downscale_factor, pixel_size

        Raises
        ------
        FileNotFoundError
            If input files are not found
        RuntimeError
            If displacement calculation fails
        """
        print("Starting Displacement Analysis...")
        start_time = time()

        displacement_field_generator = calculate_displacement_field(
            preprocessed_data['reference'],
            preprocessed_data['beads'],
            self._create_displacement_parameters(),
        )

        # Initialize result container
        try:
            while True:
                # Get next intermediate result
                displacement_field, frame, total = next(displacement_field_generator)
                self._log_displacement_progress(displacement_field, frame, total)
        except StopIteration as e:
            # Retrieve final result from generator's return value
            displacement_result = e.value

        if displacement_result is None:
            raise RuntimeError("Displacement calculation failed")

        # Save the displacement field
        np.save(str(tfm_folder / "displacements.npy"), displacement_result)

        print(f"Displacement analysis completed in {self._format_duration(time() - start_time)}")
        return displacement_result

    def _execute_force_analysis(self, tfm_folder: Path, displacement_data: DisplacementResult) -> Optional[dict]:
        """
        Execute the force analysis step of the TFM analysis pipeline.

        This method implements Fourier Transform Traction Cytometry (FTTC) to
        calculate traction forces from displacement fields.

        Parameters
        ----------
        tfm_folder : Path
            Path to the output folder where results will be saved
        displacement_data : DisplacementResult
            Object containing:
            - displacement_field: Displacement vectors
            - parameters: Displacement calculation parameters

        Returns
        -------
        Optional[dict]
            Dictionary containing:
            - force_field: Calculated traction forces (np.ndarray)
            - parameters: Force calculation parameters
            Returns None if analysis fails

        Processing Steps
        ---------------
        1. Prepares displacement data for FTTC
        2. Performs force calculation:
            - Fourier transform of displacement field
            - Application of Green's function
            - Regularization
            - Inverse transform
        3. Saves results as NumPy array

        The force calculation parameters are taken from the config:
            - young_modulus
            - poisson_ratio_substrate
            - gel_height
            - regularization
            And other FTTC parameters

        Notes
        -----
        Progress updates are logged during processing, including:
            - Frame-by-frame completion status
            - Mean and max force values
            - Processing time

        Raises
        ------
        RuntimeError
            If force calculation fails
        ValueError
            If input data is invalid or parameters are out of range
        """
        print("Starting Force Analysis...")
        start_time = time()

        force_generator = calculate_force_field(
            displacement_data.displacement_field,
            self._create_fttc_parameters()
        )

        # Initialize result container
        try:
            while True:
                # Get next intermediate result
                force_field, frame, total = next(force_generator)
                self._log_force_progress(force_field, frame, total)
        except StopIteration as e:
            # Retrieve final result from generator's return value
            force_result = e.value

        if force_result is None:
            raise RuntimeError("Force calculation failed")

        np.save(str(tfm_folder / "traction_forces.npy"), force_result)

        print(f"Force analysis completed in {self._format_duration(time() - start_time)}")
        return force_result

    def _execute_stress_analysis(self, tfm_folder: Path, mask_data: np.ndarray, force_data: FTTCResult) -> Optional[dict]:
        """
        Execute the stress analysis step of the TFM analysis pipeline.

        This method implements Monolayer Stress Microscopy (MSM) to calculate
        internal stress fields within cell monolayers.

        Parameters
        ----------
        tfm_folder : Path
            Path to the output folder where results will be saved
        mask_data : np.ndarray
            Binary masks defining cell regions
        force_data : FTTCResult
            Object containing:
            - force_field: Traction forces
            - parameters: Force calculation parameters

        Returns
        -------
        Optional[dict]
            Dictionary containing:
            - stress_tensor: Calculated stress tensors (np.ndarray)
            - mesh_quality: Mesh quality metrics
            - parameters: Stress calculation parameters
            Returns None if analysis fails

        Processing Steps
        ---------------
        1. Generates finite element mesh for each frame
        2. For each frame:
            - Assembles system matrices
            - Applies boundary conditions
            - Solves equilibrium equations
            - Calculates stress tensor field
        3. Saves results as NumPy array

        The stress analysis parameters are taken from the config:
            - poisson_ratio_cells
            - density_factor
            And other MSM parameters
        (The cell Young's modulus is fixed to a constant, not read from config.)

        Notes
        -----
        Progress updates are logged during processing, including:
            - Frame-by-frame completion status
            - Mesh quality metrics
            - Mean and max stress values
            - Processing time

        Raises
        ------
        RuntimeError
            If stress calculation fails
        ValueError
            If input data is invalid or mesh generation fails
        """
        print("Starting Stress Analysis...")
        start_time = time()

        params = self._create_msm_parameters()

        # Ensure mask_data is 3D (t, y, x)
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, ...]

        # Resize masks to exactly match force field shape
        force_shape = force_data.force_field.shape[1:3]  # (height, width)
        mask_shape = mask_data.shape[1:3]  # (height, width)

        if mask_shape != force_shape:
            print(f"Resizing masks from {mask_shape} to {force_shape} to match force field...")
            mask_data = np.stack([
                resize(
                    mask.astype(float),
                    force_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5
                for mask in mask_data
            ])
            print(f"After resize - Mask pixels > 0: {np.sum(mask_data > 0)}")

        try:
            # Initialize mesh generation
            print("Generating meshes for all frames...")
            mesh_generator = generate_mesh_stack(mask_data, params)

            # Store mesh data for all frames
            mesh_data = []

            # Process mesh generation results
            try:
                while True:
                    nodes, elements, quality_metrics, frame, total = next(mesh_generator)
                    mesh_data.append((nodes, elements, quality_metrics))
                    self._log_mesh_progress({
                        'mean_quality': quality_metrics['mean_quality'],
                        'min_angle': quality_metrics['min_angle']
                    }, frame + 1, total)
            except StopIteration as e:
                # Get the final mesh results if returned
                final_mesh_results = e.value
                if final_mesh_results:
                    mesh_data = final_mesh_results

            # Calculate stress for each frame
            print("Calculating stress fields...")
            # Get the generator
            stress_generator = calculate_stresses(
                force_field=force_data.force_field,  # Access forces from the dictionary
                masks=mask_data,
                params=params,
                mesh_data=mesh_data
            )

            # Process stress calculation results
            try:
                while True:
                    stress_result, frame, total = next(stress_generator)
                    self._log_stress_progress(stress_result, frame, total)
            except StopIteration as e:
                # Retrieve final result from generator's return value
                final_result = e.value

            if final_result is None:
                raise RuntimeError("Stress calculation failed")

            # Save results
            np.save(str(tfm_folder / "stress_results.npy"), final_result)

            print(f"Stress analysis completed in {self._format_duration(time() - start_time)}")
            return final_result

        except Exception as e:
            print(f"Error during stress analysis: {str(e)}")
            return None

    def _unified_parameters(self) -> UnifiedParameters:
        """Rebuild the unified parameter set from the config dict.

        Reconstructing UnifiedParameters and delegating to its to_*_parameters
        keeps a single source of truth for field mapping: unknown keys from
        older configs are ignored and missing keys fall back to defaults.
        """
        valid = {field.name for field in fields(UnifiedParameters)}
        raw = self.config.get('parameters', {})
        return UnifiedParameters(**{k: v for k, v in raw.items() if k in valid})

    def _create_preprocessing_parameters(self) -> PreprocessingParameters:
        """Create preprocessing parameters from config."""
        return self._unified_parameters().to_preprocessing_parameters()

    def _create_displacement_parameters(self) -> DisplacementParameters:
        """Create displacement parameters from config."""
        return self._unified_parameters().to_displacement_parameters()

    def _create_fttc_parameters(self) -> FTTCParameters:
        """Create FTTC parameters from config."""
        return self._unified_parameters().to_fttc_parameters()

    def _create_msm_parameters(self) -> MSMParameters:
        """Create MSM parameters from config."""
        return self._unified_parameters().to_msm_parameters()

    def _log_displacement_progress(self, result, frame, total):
        displacement_field_magnitude = np.sqrt(np.sum(result ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean displacement: {np.mean(displacement_field_magnitude):.2f} µm, "
              f"Max displacement: {np.max(displacement_field_magnitude):.2f} µm")

    def _log_force_progress(self, result, frame, total):
        force_magnitude = np.sqrt(np.sum(result ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean force: {np.mean(force_magnitude):.2f} Pa, "
              f"Max force: {np.max(force_magnitude):.2f} Pa")

    def _log_mesh_progress(self, quality, frame, total):
        print(f"Frame {frame}/{total}: "
              f"Average quality: {quality['mean_quality']:.3f}, "
              f"Min angle: {quality['min_angle']:.1f}°")

    def _log_stress_progress(self, result, frame, total):
        magnitude = np.sqrt(np.sum(result.stress_tensor ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean stress: {np.mean(magnitude):.2f} mN/m, "
              f"Max stress: {np.max(magnitude):.2f} mN/m")

    def _initialize_folder(self, folder: Path) -> Path:
        """Set up folder structure and logging."""
        tfm_folder = folder / "TFM_data"
        tfm_folder.mkdir(exist_ok=True)

        log_file = tfm_folder / "processing_log.txt"
        self._tee_logger = TeeLogger(log_file, self.config)
        sys.stdout = self._tee_logger

        return tfm_folder

    def _handle_visualization(self, tfm_folder: Path, viz_saver: BatchVisualizationSaver, step: str,
                              current_data: Optional[dict] = None) -> None:
        """
        Handle visualizations for each analysis step, loading data from files if needed.

        Parameters
        ----------
        tfm_folder : Path
            Path to the TFM data folder
        viz_saver : BatchVisualizationSaver
            Visualization saver instance
        step : str
            Current analysis step ('preprocessing', 'displacement', 'force', 'stress')
        current_data : Optional[dict]
            Data from the current analysis step, if available
        """
        # Map analysis steps to their visualization flags
        viz_map = {
            'preprocessing': 'bead_overlay',
            'displacement': 'displacement_map',
            'force': ['force_map', 'force_cell_overlay'],
            'stress': ['sigma_xx', 'sigma_yy', 'normal_stress', 'mesh']
        }

        viz_flags = viz_map.get(step, [])
        if isinstance(viz_flags, str):
            viz_flags = [viz_flags]

        # Check if any visualization is enabled for this step
        if not any(self.config['visualizations'].get(flag, False) for flag in viz_flags):
            return

        try:
            data = current_data
            if data is None:
                # Try to load data from files based on step
                if step == 'preprocessing':
                    try:
                        data = {
                            'beads': tifffile.imread(str(tfm_folder / "preprocessed_beads.tif")),
                            'reference': tifffile.imread(str(tfm_folder / "preprocessed_reference.tif"))
                        }
                    except Exception as e:
                        print(f"Could not load preprocessed files for visualization: {str(e)}")
                        return

                elif step == 'displacement':
                    try:
                        data = np.load(str(tfm_folder / "displacements.npy"), allow_pickle=True).item()
                    except Exception as e:
                        print(f"Could not load displacement data for visualization: {str(e)}")
                        return

                elif step == 'force':
                    try:
                        data = np.load(str(tfm_folder / "traction_forces.npy"), allow_pickle=True).item()
                    except Exception as e:
                        print(f"Could not load force data for visualization: {str(e)}")
                        return

                elif step == 'stress':
                    try:
                        data = np.load(str(tfm_folder / "stress_results.npy"), allow_pickle=True).item()
                    except Exception as e:
                        print(f"Could not load stress/mask data for visualization: {str(e)}")
                        return

            if data is None:
                print(f"No data available for {step} visualization")
                return

            # Generate visualizations based on enabled flags
            if step == 'preprocessing' and self.config['visualizations']['bead_overlay']:
                print("Generating bead overlay visualization...")
                viz_saver.save_bead_overlay(data['beads'], data['reference'])

            elif step == 'displacement' and self.config['visualizations']['displacement_map']:
                print("Generating displacement map visualization...")
                viz_saver.save_displacement_visualization(data)

            elif step == 'force':
                if self.config['visualizations']['force_map']:
                    print("Generating force map visualization...")
                    viz_saver.save_force_visualization(data)

                if self.config['visualizations']['force_cell_overlay']:
                    print("Generating force-cell overlay visualization...")
                    try:
                        cell_images = tifffile.imread(str(tfm_folder / "preprocessed_cells.tif"))
                        viz_saver.save_force_cell_overlay(data, cell_images)
                    except Exception as e:
                        print(f"Could not generate force-cell overlay: {str(e)}")

            elif step == 'stress':
                if any(self.config['visualizations'][flag] for flag in ['sigma_xx', 'sigma_yy', 'normal_stress']):
                    print("Generating stress visualization...")
                    viz_saver.save_stress_visualization(
                        data,
                        plot_sigma_xx=self.config['visualizations']['sigma_xx'],
                        plot_sigma_yy=self.config['visualizations']['sigma_yy'],
                        plot_normal_stress=self.config['visualizations']['normal_stress']
                    )

                if self.config['visualizations']['mesh']:
                    print("Generating mesh visualization...")
                    viz_saver.save_mesh_visualization(data)

        except Exception as e:
            print(f"Error generating {step} visualization: {str(e)}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BatchAnalysis':
        """Create BatchAnalysis instance from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _save_calibrated_tiff(self, data: np.ndarray, filepath: Path, pixel_size: float,
                              frame_interval: float) -> None:
        """
        Save data as calibrated TIFF file with ImageJ-compatible metadata.

        Args:
            data: numpy array to save
            filepath: path where to save the file
            pixel_size: spatial calibration in µm/pixel
            frame_interval: temporal calibration in minutes/frame
        """
        if data is None:
            return

        # Convert to 16-bit
        data_normalized = data.astype(float)
        data_normalized = (data_normalized - data_normalized.min()) / (
                data_normalized.max() - data_normalized.min())
        data_16bit = (data_normalized * 65535).astype(np.uint16)

        # Create ImageJ-compatible metadata
        imagej_metadata = {
            'ImageJ': '1.53c',
            'spacing': pixel_size,
            'unit': 'um',
            'frame_interval': frame_interval,
            'frame_interval_unit': 'minute'
        }

        # For Z-stacks or time series, specify dimensions
        if data.ndim > 2:
            imagej_metadata.update({
                'frames': data.shape[0],
                'slices': 1,
                'channels': 1
            })

        # Combine metadata for compatibility
        metadata = {
            'PhysicalSizeX': pixel_size,
            'PhysicalSizeXUnit': 'um',
            'PhysicalSizeY': pixel_size,
            'PhysicalSizeYUnit': 'um',
            'TimeIncrement': frame_interval,
            'TimeIncrementUnit': 'min',
            **imagej_metadata
        }

        # Save with metadata using tifffile
        tifffile.imwrite(
            str(filepath),
            data_16bit,
            imagej=True,
            metadata=metadata,
            resolution=(1 / pixel_size, 1 / pixel_size),  # resolution in pixels per unit
            photometric='minisblack'
        )

        print(f"Saved calibrated TIFF: {filepath}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._tee_logger:
            self._tee_logger.close()
            self._tee_logger = None


if __name__ == "__main__":
    analyzer = BatchAnalysis.from_yaml("config.yaml")
    analyzer.process_all_folders()
