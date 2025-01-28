import json
import sys
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional, Tuple

import numpy as np
import tifffile
import yaml

from napariTFM.backend.batch_analysis_visualizations import BatchVisualizationSaver
from napariTFM.services.displacement_service import DisplacementService, DisplacementParameters
from napariTFM.services.fttc_service import FTTCService, FTTCParameters
from napariTFM.services.msm_service import MSMService, MSMParameters
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
        banner = """
        ╔═══════════════════════════════════════════════════════╗
        ║                                _ _____ _____ __  __   ║
        ║   _ __   __ _ _ __   __ _ _ __(_)_   _|  ___|  \/  |  ║
        ║  | '_ \ / _` | '_ \ / _` | '__| | | | | |_  | |\/| |  ║
        ║  | | | | (_| | |_) | (_| | |  | | | | |  _| | |  | |  ║
        ║  |_| |_|\__,_| .__/ \__,_|_|  |_| |_| |_|   |_|  |_|  ║
        ║              |_|                                      ║
        ║                                                       ║
        ║        Traction Force Microscopy Analysis Tool        ║
        ╚═══════════════════════════════════════════════════════╝
        """
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
        self.preprocessing_service = PreprocessingService()
        self.displacement_service = DisplacementService()
        self.fttc_service = FTTCService()
        self.msm_service = MSMService()
        self._tee_logger = None

    def process_folder(self, folder_path: str) -> None:
        """Main processing pipeline for a folder."""
        folder = Path(folder_path)
        tfm_folder = self._initialize_folder(folder)
        viz_saver = BatchVisualizationSaver(folder)

        try:
            print(f"\nProcessing folder: {folder_path}")
            print("=" * 50)

            # Handle preprocessing
            preprocessed_data = self._handle_preprocessing_execution(folder, tfm_folder)

            # Handle displacement
            displacement_data = self._handle_displacement_execution(folder, tfm_folder, preprocessed_data)

            # Handle force analysis
            force_data = self._handle_force_execution(folder, tfm_folder, displacement_data)

            # Handle stress analysis
            mesh_data, stress_data = self._handle_stress_execution(folder, tfm_folder, force_data)

            # Generate visualizations
            self._generate_visualizations(viz_saver, preprocessed_data, displacement_data,
                                          force_data, mesh_data, stress_data)

            print("\nFolder processing completed successfully!")
            print("=" * 80)

        finally:
            self._cleanup()

    def _handle_preprocessing_execution(self, folder: Path, tfm_folder: Path) -> Optional[dict]:
        """Handle preprocessing execution. Always runs if enabled."""
        if not self.config['analysis_steps']['preprocessing']:
            return None

        try:
            print("\nExecuting preprocessing step...")
            return self._execute_preprocessing(folder, tfm_folder)
        except Exception as e:
            print(f"Preprocessing failed: {str(e)}")
            return None

    def _handle_displacement_execution(self, folder: Path, tfm_folder: Path, preprocessed_data: Optional[dict]) -> Optional[dict]:
        """Handle displacement analysis execution. Always runs if enabled."""
        if not self.config['analysis_steps']['displacement']:
            return None

        try:
            if preprocessed_data is None:
                print("\nLoading preprocessed images from file...")
                try:
                    preprocessed_bead_stack = tifffile.imread(str(folder / tfm_folder / "preprocessed_beads.tif"))
                    preprocessed_reference = tifffile.imread(str(folder / tfm_folder / "preprocessed_reference.tif"))
                    preprocessed_data = {
                        'beads': preprocessed_bead_stack,
                        'reference': preprocessed_reference,
                    }
                except Exception as e:
                    print(f"Could not load preprocessed files: {str(e)}")
                    return None

            print("\nExecuting displacement analysis step...")
            return self._execute_displacement_analysis(tfm_folder, preprocessed_data)
        except Exception as e:
            print(f"Displacement analysis failed: {str(e)}")
            return None

    def _handle_force_execution(self, folder: Path, tfm_folder: Path, displacement_data: Optional[dict]) -> Optional[dict]:
        """Handle force analysis execution. Always runs if enabled."""
        if not self.config['analysis_steps']['force']:
            return None

        try:
            if displacement_data is None:
                print("\nLoading displacement data from file...")
                try:
                    displacement_data = np.load(str(tfm_folder / "displacements.npy"), allow_pickle=True).item()
                except Exception as e:
                    print(f"Could not load displacement data: {str(e)}")
                    return None

            print("\nExecuting force analysis step...")
            return self._execute_force_analysis(tfm_folder, displacement_data)
        except Exception as e:
            print(f"Force analysis failed: {str(e)}")
            return None

    def _handle_stress_execution(self, folder: Path, tfm_folder: Path, force_data: Optional[dict]) -> Tuple[Optional[dict], Optional[dict]]:
        """Handle stress analysis execution. Always runs if enabled."""
        if not (self.config['analysis_steps']['create_masks'] or self.config['analysis_steps']['stress']):
            return None, None

        try:
            if force_data is None and self.config['analysis_steps']['stress']:
                print("\nLoading force data from file...")
                try:
                    force_data = np.load(str(tfm_folder / "traction_forces.npy"), allow_pickle=True).item()
                except Exception as e:
                    print(f"Could not load force data: {str(e)}")
                    return None, None

            print("\nExecuting mesh and stress analysis steps...")
            return self._execute_stress_analysis(folder, tfm_folder, force_data)
        except Exception as e:
            print(f"Mesh/stress analysis failed: {str(e)}")
            return None, None

    def _initialize_folder(self, folder: Path) -> Path:
        """Set up folder structure and logging."""
        tfm_folder = folder / "TFM_data"
        tfm_folder.mkdir(exist_ok=True)

        log_file = tfm_folder / "processing_log.txt"
        self._tee_logger = TeeLogger(log_file, self.config)
        sys.stdout = self._tee_logger

        return tfm_folder

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
            vector_stride=self.config['parameters']['disp_vector_stride'],
            arrow_scale=self.config['parameters']['disp_arrow_scale']
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
            frame_interval=self.config['parameters']['frame_interval']
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
            max_stress=self.config['parameters']['max_stress']
        )

    def _execute_preprocessing(self, folder: Path, tfm_folder: Path) -> Optional[dict]:
        """Execute preprocessing step using PreprocessingService."""

        print("\nStarting Preprocessing...")
        start_time = time()
        params = self._create_preprocessing_parameters()
        self.preprocessing_service.update_parameters(params)

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
        for result, frame, total in self.preprocessing_service.preprocess_stack(bead_stack, reference):
            bead_results.append(result)
            print(f"Progress (beads): {(frame / total) * 100:.1f}%, Frame {frame}/{total}")

        reference_result = self.preprocessing_service.preprocess_frame(reference)

        # Process cell images if available
        cell_results = []
        if cell_stack is not None:
            print("\nProcessing cell images...")
            for result, frame, total in self.preprocessing_service.preprocess_stack(cell_stack, reference_image=None, is_cell=True):
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
        """Execute displacement analysis step using DisplacementService."""
        print("\nStarting Displacement Analysis...")
        start_time = time()
        params = self._create_displacement_parameters()
        results = []

        for result, frame, total in self.displacement_service.calculate_flow_stack(
                preprocessed_data['reference'],
                preprocessed_data['beads'],
                params
        ):
            results.append(result)
            self._log_displacement_progress(result, frame, total)

        displacement_data = {
            'flows': np.stack([r.flow for r in results]),
            'parameters': params.__dict__
        }
        np.save(str(tfm_folder / "displacements.npy"), displacement_data)

        print(f"Displacement analysis completed in {time() - start_time:.1f} seconds")
        return displacement_data

    def _execute_force_analysis(self, tfm_folder: Path, displacement_data: Optional[dict]) -> Optional[dict]:
        """Execute force analysis step using FTTCService."""

        print("\nStarting Force Analysis...")
        start_time = time()

        params = self._create_fttc_parameters()
        self.fttc_service.initialize_calculator(params)

        force_results = []
        for result, frame, total in self.fttc_service.calculate_force_stack(
                displacement_data['flows'],
                self.config['parameters']['pixel_size'],
                self.config['parameters']['downscale_factor']
        ):
            force_results.append(result)
            self._log_force_progress(result, frame, total)

        force_data = self._prepare_force_data(force_results, params)
        np.save(str(tfm_folder / "traction_forces.npy"), force_data)

        print(f"Force analysis completed in {time() - start_time:.1f} seconds")
        return force_data

    def _execute_stress_analysis(self, folder: Path, tfm_folder: Path, force_data: Optional[dict]) -> Tuple[Optional[dict], Optional[dict]]:
        """Execute mesh generation and stress analysis steps using MSMService."""
        print("\nStarting Mesh and Stress Analysis...")
        start_time = time()

        params = self._create_msm_parameters()
        cell_images = self._load_cell_images(folder, tfm_folder)

        # Generate masks and meshes
        mesh_results = []
        masks = []

        for mask, frame, total in self.msm_service.create_mask_stack(cell_images, params):
            masks.append(mask)
            mesh_result = self.msm_service.generate_mesh(mask, params)
            mesh_results.append(mesh_result)
            self._log_mesh_progress(mesh_result, frame, total)

        # Calculate stress if needed
        stress_results = None
        if self.config['analysis_steps']['stress']:
            stress_results = []
            for result, frame, total in self.msm_service.calculate_stress_stack(
                    force_data['force_field'], params, mesh_results, np.stack(masks)
            ):
                stress_results.append(result)
                self._log_stress_progress(result, frame, total)

        # Save results
        mesh_data = self._save_mesh_data(tfm_folder, mesh_results, masks, params)
        stress_data = self._save_stress_data(tfm_folder, stress_results) if stress_results else None

        print(f"Mesh and stress analysis completed in {time() - start_time:.1f} seconds")
        return mesh_data, stress_data

    def _generate_visualizations(self, viz_saver: BatchVisualizationSaver,
                                 preprocessed_data: Optional[dict],
                                 displacement_data: Optional[dict],
                                 force_data: Optional[dict],
                                 mesh_data: Optional[dict],
                                 stress_data: Optional[dict]) -> None:
        """Generate all requested visualizations using BatchVisualizationSaver."""
        if self.config['visualizations']['bead_overlay'] and preprocessed_data:
            viz_saver.save_bead_overlay(preprocessed_data['beads'], preprocessed_data['reference'])

        if self.config['visualizations']['displacement_map'] and displacement_data:
            viz_saver.save_displacement_visualization(displacement_data)

        if self.config['visualizations']['force_map'] and force_data:
            viz_saver.save_force_visualization(force_data)

        if self.config['visualizations']['mesh'] and mesh_data:
            viz_saver.save_mesh_visualization(mesh_data['masks'], mesh_data['mesh_results'])

        if any([self.config['visualizations']['sigma_xx'],
                self.config['visualizations']['sigma_yy'],
                self.config['visualizations']['normal_stress']]) and stress_data:
            viz_saver.save_stress_visualization(
                stress_data,
                plot_sigma_xx=self.config['visualizations']['sigma_xx'],
                plot_sigma_yy=self.config['visualizations']['sigma_yy'],
                plot_normal_stress=self.config['visualizations']['normal_stress']
            )

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._tee_logger:
            self._tee_logger.close()
            self._tee_logger = None

    def _log_displacement_progress(self, result, frame, total):
        flow_magnitude = np.sqrt(np.sum(result.flow ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean displacement: {np.mean(flow_magnitude):.2f} µm, "
              f"Max displacement: {np.max(flow_magnitude):.2f} µm")

    def _log_force_progress(self, result, frame, total):
        print(f"Frame {frame}/{total}: "
              f"Mean force: {result['mean_force']:.2f} Pa, "
              f"Max force: {result['max_force']:.2f} Pa")

    def _log_mesh_progress(self, result, frame, total):
        quality = result.quality_metrics
        print(f"Frame {frame}/{total}: "
              f"Average quality: {quality['mean_quality']:.3f}, "
              f"Min angle: {quality['min_angle']:.1f}°")

    def _log_stress_progress(self, result, frame, total):
        magnitude = np.sqrt(np.sum(result.stress_tensor ** 2, axis=-1))
        print(f"Frame {frame}/{total}: "
              f"Mean stress: {np.mean(magnitude):.2f} mN/m, "
              f"Max stress: {np.max(magnitude):.2f} mN/m")

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

    def process_all_folders(self) -> None:
        """Process all folders specified in configuration."""
        for folder in self.config['root_folders']:
            self.process_folder(folder)


if __name__ == "__main__":
    analyzer = BatchAnalysis.from_yaml("batch_config.yaml")
    analyzer.process_all_folders()
