import json
import sys
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional, Tuple
from skimage.transform import rescale

import numpy as np
import tifffile
import yaml
from numpy._typing import NDArray

from napariTFM.backend.batch_analysis_visualizations import BatchVisualizationSaver
from napariTFM.services.displacement_service import DisplacementService, DisplacementParameters, DisplacementResult
from napariTFM.services.fttc_service import FTTCService, FTTCParameters
from napariTFM.services.msm_service import MSMService, MSMParameters, MeshResult
from napariTFM.services.preprocessing_service import PreprocessingService, PreprocessingParameters


# TODO remove this warning from console output: UserWarning: <tifffile.TiffWriter 'preprocessed_reference.tif'> not writing description to ImageJ file

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

        # Log configuration parameters if provided
        if config:
            self.log.write("Analysis Parameters:\n")
            self.log.write("-" * 20 + "\n")

            # Log analysis steps
            self.log.write("\nEnabled Analysis Steps:\n")
            for step, enabled in config.get('analysis_steps', {}).items():
                self.log.write(f"- {step}: {'Yes' if enabled else 'No'}\n")

            # Log key parameters
            self.log.write("\nKey Parameters:\n")
            params = config.get('parameters', {})
            for key, value in sorted(params.items()):
                self.log.write(f"- {key}: {value}\n")

            self.log.write("\n" + "-" * 50 + "\n\n")

    def print_banner(self):
        banner = '''
    ╔═════════════════════════════════════════════════════════════════════════════╗
    ║                                                                             ║
    ║                                         ,--.,--------.,------.,--.   ,--.   ║
    ║   ,--,--,  ,--,--. ,---.  ,--,--.,--.--.`--''--.  .--'|  .---'|   `.'   |   ║
    ║   |      \| ,-.  || .-. |' ,-.  ||  .--',--.   |  |   |  `--, |  |'.'|  |   ║
    ║   |  ||  |\ '-'  || '-' '\ '-'  ||  |   |  |   |  |   |  |`   |  |   |  |   ║
    ║   `--''--' `--`--'|  |-'  `--`--'`--'   `--'   `--'   `--'    `--'   `--'   ║
    ║                   `--'                                                      ║
    ║                                                                             ║
    ║                   Traction Force Microscopy Analysis Tool                   ║
    ╚═════════════════════════════════════════════════════════════════════════════╝
        '''

        print(banner)

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
        self.log.write(f"Started:  {self.start_time}\n")
        self.log.write(f"Finished: {end_time}\n")
        self.log.write(f"Duration: {duration}\n")

        # Calculate hours, minutes, seconds for more readable format
        total_seconds = duration.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        if hours > 0:
            self.log.write(f"Total time: {hours}h {minutes}m {seconds:.1f}s\n")
        elif minutes > 0:
            self.log.write(f"Total time: {minutes}m {seconds:.1f}s\n")
        else:
            self.log.write(f"Total time: {seconds:.1f}s\n")

        self.log.close()
        sys.stdout = self.terminal


class BatchAnalysis:
    """Handles batch analysis of TFM data using service layer components."""

    def __init__(self, config: dict):
        self.config = config
        self._tee_logger = None

    def process_all_folders(self) -> None:
        """Process all folders specified in configuration."""
        for folder in self.config['root_folders']:
            self.process_folder(folder)

    def process_folder(self, folder_path: str) -> None:
        """Main processing pipeline for a folder."""
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

            # Handle mask creation
            mask_data = self._handle_mask_creation(tfm_folder)

            # Handle stress analysis
            stress_data = self._handle_stress_execution(tfm_folder, mask_data, force_data)
            self._handle_visualization(tfm_folder, viz_saver, 'stress', stress_data)

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

    def _handle_mask_creation(self, tfm_folder: Path) -> Optional[dict]:
        """Handle mask creation execution. Always runs if enabled."""
        if not self.config['analysis_steps']['create_masks']:
            return None

        try:
            try:
                cell_images = tifffile.imread(str(tfm_folder / "preprocessed_cells.tif"))
            except Exception as e:
                print(f"Could not load cell images: {str(e)}")
                return None

            return self._execute_mask_creation(tfm_folder, cell_images)

        except Exception as e:
            print(f"Mask creation failed: {str(e)}")
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

            print("Loading mask data...")
            try:
                if mask_data is None:
                    mask_data = tifffile.imread(str(tfm_folder / "masks.tif"))
            except Exception as e:
                print(f"Could not load mask data: {str(e)}")
                return None

            return self._execute_stress_analysis(tfm_folder, force_data, mask_data)

        except Exception as e:
            print(f"Stress analysis failed: {str(e)}")
            return None

    def _execute_preprocessing(self, folder: Path, tfm_folder: Path) -> Optional[dict]:
        """Execute preprocessing step using PreprocessingService."""

        print("Starting Preprocessing...")
        start_time = time()
        preprocessing_service = PreprocessingService(self._create_preprocessing_parameters())

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
        for result, frame, total in preprocessing_service.preprocess_stack(bead_stack, reference):
            bead_results.append(result)
            print(f"Progress (beads): {(frame / total) * 100:.1f}%, Frame {frame}/{total}")

        reference_result = preprocessing_service.preprocess_frame(reference)

        # Process cell images if available
        cell_results = []
        if cell_stack is not None:
            print("Processing cell images...")
            for result, frame, total in preprocessing_service.preprocess_stack(cell_stack, reference_image=None, is_cell=True):
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

        print(f"Preprocessing completed in {time() - start_time:.1f} seconds")
        return preprocessed

    def _execute_displacement_analysis(self, tfm_folder: Path, preprocessed_data: Optional[dict]) -> Optional[dict]:
        print("Starting Displacement Analysis...")
        start_time = time()
        displacement_service = DisplacementService(self._create_displacement_parameters())

        # Get the generator
        flow_generator = displacement_service.calculate_flow(
            preprocessed_data['reference'],
            preprocessed_data['beads'],
            yield_intermediates=True
        )

        # Initialize result container
        try:
            while True:
                # Get next intermediate result
                flow, frame, total = next(flow_generator)
                self._log_displacement_progress(flow, frame, total)
        except StopIteration as e:
            # Retrieve final result from generator's return value
            displacement_result = e.value

        if displacement_result is None:
            raise RuntimeError("Displacement calculation failed")

        # Save the flow field
        np.save(str(tfm_folder / "displacements.npy"), displacement_result)

        print(f"Displacement analysis completed in {time() - start_time:.1f} seconds")
        return displacement_result

    def _execute_force_analysis(self, tfm_folder: Path, displacement_data: DisplacementResult) -> Optional[dict]:
        """Execute force analysis step using FTTCService."""
        print("Starting Force Analysis...")
        start_time = time()

        fttc_service = FTTCService(self._create_fttc_parameters())

        # Get the generator
        force_generator = fttc_service.calculate_forces(
            displacement_data.flow,
            yield_intermediates=True
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

        print(f"Force analysis completed in {time() - start_time:.1f} seconds")
        return force_result
    def _execute_mask_creation(self, tfm_folder: Path, cell_images: np.ndarray) -> NDArray[int]:
        """Execute mask creation step using MSMService."""
        print("Starting Mask Creation...")
        start_time = time()

        params = self._create_msm_parameters()
        downscale_factor = self.config['parameters']['downscale_factor']
        masks = []

        # Create masks and generate meshes
        for mask, frame, total in self.msm_service.create_mask_stack(cell_images, params):
            mask = rescale(mask, 1 / downscale_factor, order=0, preserve_range=True, anti_aliasing=False)
            masks.append(mask)
            self._log_mask_progress(mask, frame, total)

        masks = np.array(masks)

        tifffile.imwrite(str(tfm_folder / "masks.tif"), masks.astype("uint8"))

        print(f"Mask creation completed in {time() - start_time:.1f} seconds")
        return masks

    def _execute_stress_analysis(self, tfm_folder: Path, force_data: dict, mask_data: np.ndarray) -> Optional[dict]:
        """
        Execute stress analysis step using MSMService.

        Parameters
        ----------
        tfm_folder : Path
            Path to the TFM data folder
        force_data : dict
            Dictionary containing force field data and parameters
        mask_data : np.ndarray
            3D array of masks (frames, height, width)

        Returns
        -------
        Optional[dict]
            Dictionary containing stress tensors and parameters, or None if analysis fails
        """
        print("Starting Stress Analysis...")
        start_time = time()

        params = self._create_msm_parameters()

        try:
            # Initialize mesh generation
            print("Generating meshes for all frames...")
            mesh_results = []
            for nodes, elements, quality_metrics, frame, total in self.msm_service.generate_mesh_stack(
                    mask_data, params):
                mesh_results.append(MeshResult(
                    nodes=nodes,
                    elements=elements,
                    quality_metrics=quality_metrics
                ))
                self._log_mesh_progress({
                    'mean_quality': quality_metrics['mean_quality'],
                    'min_angle': quality_metrics['min_angle']
                }, frame + 1, total)

            # Calculate stress for each frame
            print("Calculating stress fields...")
            stress_results = []
            for result, frame, total in self.msm_service.calculate_stress_stack(
                    force_data['forces'],
                    params,
                    mesh_results,
                    mask_data
            ):
                stress_results.append(result)
                self._log_stress_progress(result, frame + 1, total)

            # Prepare results dictionary
            stress_data = {
                'stress_tensors': np.stack([r.stress_tensor for r in stress_results]),
                'nodes': [r.nodes for r in stress_results],
                'elements': [r.elements for r in stress_results],
                'condition_numbers': [r.condition_number for r in stress_results],
                'residuals': [r.residual for r in stress_results],
                'parameters': params.__dict__
            }

            # Save results
            np.save(str(tfm_folder / "stress_results.npy"), stress_data)

            print(f"Stress analysis completed in {time() - start_time:.1f} seconds")
            return stress_data

        except Exception as e:
            print(f"Error during stress analysis: {str(e)}")
            return None

    def _create_preprocessing_parameters(self) -> PreprocessingParameters:
        """Create preprocessing parameters from config."""
        return PreprocessingParameters(
            min_intensity_percentile=self.config['parameters']['min_intensity'] / 100,
            max_intensity_percentile=self.config['parameters']['max_intensity'] / 100,
            gaussian_sigma=self.config['parameters']['gaussian_sigma'],
            cell_min_intensity_percentile=self.config['parameters']['cell_min_intensity'] / 100,
            cell_max_intensity_percentile=self.config['parameters']['cell_max_intensity'] / 100,
            cell_gaussian_sigma=self.config['parameters']['cell_gaussian_sigma'],
            registration_mode=self.config['parameters']['registration_mode']
        )

    def _create_displacement_parameters(self) -> DisplacementParameters:
        """Create displacement parameters from config."""
        return DisplacementParameters(
            tau=self.config['parameters']['tau'],
            lambda_=self.config['parameters']['lambda_'],
            theta=self.config['parameters']['theta'],
            nscales=self.config['parameters']['nscales'],
            warps=self.config['parameters']['warps'],
            epsilon=self.config['parameters']['epsilon'],
            inner_iterations=self.config['parameters']['inner_iterations'],
            outer_iterations=self.config['parameters']['outer_iterations'],
            scale_step=self.config['parameters']['scale_step'],
            median_filtering=self.config['parameters']['median_filtering'],
            downscale_factor=self.config['parameters']['downscale_factor'],
            pixel_size=self.config['parameters']['pixel_size'],
            frame_interval=self.config['parameters']['frame_interval'],
            d_max=self.config['parameters']['d_max'],
            disp_vector_stride=self.config['parameters']['disp_vector_stride'],
            disp_arrow_scale=self.config['parameters']['disp_arrow_scale']
        )

    def _create_fttc_parameters(self) -> FTTCParameters:
        """Create FTTC parameters from config."""
        return FTTCParameters(
            young_modulus=self.config['parameters']['young_modulus'],
            poisson_ratio_substrate=self.config['parameters']['poisson_ratio_substrate'],
            gel_height=self.config['parameters'].get('gel_height'),
            lanczos_exp=self.config['parameters']['lanczos_exp'],
            regularization=self.config['parameters']['regularization'],
            auto_gcv=False,
            force_vector_stride=self.config['parameters']['force_vector_stride'],
            force_arrow_scale=self.config['parameters']['force_arrow_scale'],
            f_max=self.config['parameters']['f_max'],
            frame_interval=self.config['parameters']['frame_interval'],
            pixel_size=self.config['parameters']['pixel_size'],
            downscale_factor=self.config['parameters']['downscale_factor']
        )

    def _create_msm_parameters(self) -> MSMParameters:
        """Create MSM parameters from config."""
        return MSMParameters(
            density_factor=self.config['parameters']['density_factor'],
            algorithm=self.config['parameters']['mesh_algorithm'],
            use_optimization=self.config['parameters']['use_optimization'],
            poisson_ratio_cells=self.config['parameters']['poisson_ratio_cells'],
            young_modulus=1.0,
            threshold=self.config['parameters']['threshold'],
            dilation=self.config['parameters']['dilation'],
            smoothing_sigma=self.config['parameters']['smoothing_sigma'],
            max_stress=self.config['parameters']['max_stress'],
            pixel_size=self.config['parameters']['pixel_size'],
            downscale_factor=self.config['parameters']['downscale_factor']
        )

    def _log_displacement_progress(self, result, frame, total):
        flow_magnitude = np.sqrt(np.sum(result ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean displacement: {np.mean(flow_magnitude):.2f} µm, "
              f"Max displacement: {np.max(flow_magnitude):.2f} µm")

    def _log_force_progress(self, result, frame, total):
        force_magnitude = np.sqrt(np.sum(result ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean force: {np.mean(force_magnitude):.2f} Pa, "
              f"Max force: {np.mean(force_magnitude):.2f} Pa")

    def _log_mask_progress(self, mask, frame, total):
        # Calculate surface area (sum of all True/1 pixels)
        surface_area = np.sum(mask)

        # Calculate centroid coordinates
        y_coords, x_coords = np.where(mask)
        if len(x_coords) > 0:  # Check if mask is not empty
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            print(f"Frame {frame}/{total}: "
                  f"Centroid: ({centroid_x:.1f}, {centroid_y:.1f}), "
                  f"Area: {surface_area:.0f} px²")
        else:
            print(f"Frame {frame}/{total}: Empty mask")

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
                        if self.config['visualizations']['mesh']:
                            # Load masks for mesh visualization if needed
                            masks = tifffile.imread(str(tfm_folder / "masks.tif"))
                            data['masks'] = masks
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

                if self.config['visualizations']['mesh'] and 'masks' in data:
                    print("Generating mesh visualization...")
                    viz_saver.save_mesh_visualization(
                        data['masks'],
                        density_factor=self.config['parameters']['density_factor'],
                        algorithm=6 if self.config['parameters']['mesh_algorithm'] == 'Frontal-Del.' else 1,
                        use_optimization=self.config['parameters']['use_optimization']
                    )

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

        # Create description for ImageJ
        description = json.dumps({
            'Info': f'Scale: {pixel_size} um/pixel, Frame interval: {frame_interval} min',
            **imagej_metadata
        })

        # Combine with original metadata for compatibility
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
            description=description,
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
    analyzer = BatchAnalysis.from_yaml("batch_config.yaml")
    analyzer.process_all_folders()
