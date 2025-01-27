import json

import pyfiglet
import yaml
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Generator
import tifffile
import sys
from datetime import datetime
import logging
import warnings
from time import time

from napariTFM.services.preprocessing_service import PreprocessingService, PreprocessingParameters
from napariTFM.services.displacement_service import DisplacementService, DisplacementParameters
from napariTFM.services.fttc_service import FTTCService, FTTCParameters
from napariTFM.services.msm_service import MSMService, MSMParameters
from napariTFM.backend.batch_analysis_visualizations import BatchVisualizationSaver


def print_banner():
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


class TeeLogger:
    """Custom logger that captures print statements and logging output to both console and file."""

    def __init__(self, filename: Path, config: dict = None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = open(filename, 'w', encoding='utf-8')
        self.start_time = datetime.now()

        # Write header to log file
        print_banner()
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
        """Initialize batch analysis with configuration dictionary."""
        self.config = config
        self._cleanup_internal_storage()
        self._tee_logger = None

        # Initialize services
        self.preprocessing_service = PreprocessingService()
        self.displacement_service = DisplacementService()
        self.fttc_service = FTTCService()
        self.msm_service = MSMService()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BatchAnalysis':
        """Create BatchAnalysis instance from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def process_folder(self, folder_path: str) -> None:
        """Process a single folder according to configuration."""
        folder = Path(folder_path)
        tfm_folder = folder / "TFM_data"
        tfm_folder.mkdir(exist_ok=True)

        # Setup logging
        log_file = folder / "TFM_data/processing_log.txt"
        self._tee_logger = TeeLogger(log_file, self.config)
        sys.stdout = self._tee_logger

        try:
            print(f"\nProcessing folder: {folder_path}")
            print("=" * 80)

            # Initialize visualization saver
            viz_saver = BatchVisualizationSaver(folder)

            #######################
            # Preprocessing Step
            #######################
            preprocessed_data_available = False
            if self.config['analysis_steps']['preprocessing']:
                print("\nStarting Preprocessing...")
                start_time = time()

                try:
                    preproc_params = PreprocessingParameters(
                        min_intensity_percentile=self.config['parameters']['min_intensity'] / 100,
                        max_intensity_percentile=self.config['parameters']['max_intensity'] / 100,
                        gaussian_sigma=self.config['parameters']['gaussian_sigma'],
                        cell_min_intensity_percentile=self.config['parameters']['cell_min_intensity'] / 100,
                        cell_max_intensity_percentile=self.config['parameters']['cell_max_intensity'] / 100,
                        cell_gaussian_sigma=self.config['parameters']['cell_gaussian_sigma'],
                        registration_mode=self.config['parameters']['registration_mode']
                    )

                    self.preprocessing_service.update_parameters(preproc_params)

                    # Load and preprocess bead images
                    bead_stack = tifffile.imread(str(folder / self.config['input_files']['beads']))
                    reference = tifffile.imread(str(folder / self.config['input_files']['reference']))

                    print("Processing bead images...")
                    bead_results = []
                    for progress in self.preprocessing_service.preprocess_stack(bead_stack, reference):
                        result, frame, total = progress
                        bead_results.append(result)
                        print(f"Progress: {(frame / total) * 100:.1f}%, Frame {frame}/{total}")

                    # Save preprocessed bead data
                    preprocessed_beads = np.stack([r.processed_image for r in bead_results])
                    preprocessed_ref = bead_results[0].processed_image
                    tifffile.imwrite(str(tfm_folder / "preprocessed_beads.tif"), preprocessed_beads)
                    tifffile.imwrite(str(tfm_folder / "preprocessed_reference.tif"), preprocessed_ref)

                    # Process cell images if available
                    cell_results = []
                    if 'cells' in self.config['input_files']:
                        print("\nProcessing cell images...")
                        cell_stack = tifffile.imread(str(folder / self.config['input_files']['cells']))
                        for progress in self.preprocessing_service.preprocess_stack(cell_stack, is_cell=True):
                            result, frame, total = progress
                            cell_results.append(result)
                            print(f"Progress: {(frame / total) * 100:.1f}%, Frame {frame}/{total}")

                        # Save preprocessed cell data
                        preprocessed_cells = np.stack([r.processed_image for r in cell_results])
                        tifffile.imwrite(str(tfm_folder / "preprocessed_cells.tif"), preprocessed_cells)

                    preprocessed_data_available = True
                    print(f"\nPreprocessing completed in {time() - start_time:.1f} seconds")

                    # Bead overlay visualization if requested
                    if self.config['visualizations']['bead_overlay']:
                        print("\nGenerating bead overlay visualization...")
                        viz_saver.save_bead_overlay(preprocessed_beads, preprocessed_ref)
                        print("Bead overlay saved successfully")

                except Exception as e:
                    print(f"Preprocessing failed: {str(e)}. Skipping this step.")

            #######################
            # Displacement Analysis
            #######################
            displacement_data_available = False
            if self.config['analysis_steps']['displacement']:
                if not preprocessed_data_available:
                    # Try to load preprocessed data
                    try:
                        preprocessed_beads = tifffile.imread(str(tfm_folder / "preprocessed_beads.tif"))
                        preprocessed_ref = tifffile.imread(str(tfm_folder / "preprocessed_reference.tif"))
                        preprocessed_data_available = True
                    except Exception as e:
                        print(f"Could not load preprocessed data: {str(e)}. Skipping displacement analysis.")

                if preprocessed_data_available:
                    print("\nStarting Displacement Analysis...")
                    start_time = time()

                    try:
                        disp_params = DisplacementParameters(
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

                        displacement_results = []
                        for progress in self.displacement_service.calculate_flow_stack(
                                preprocessed_ref, preprocessed_beads, disp_params
                        ):
                            result, frame, total = progress
                            displacement_results.append(result)

                            # Calculate and display statistics
                            flow_magnitude = np.sqrt(np.sum(result.flow ** 2, axis=-1))
                            print(f"Frame {frame}/{total}: "
                                  f"Mean displacement: {np.mean(flow_magnitude):.2f} µm, "
                                  f"Max displacement: {np.max(flow_magnitude):.2f} µm")

                        # Save displacement results
                        displacement_data = {
                            'flows': np.stack([r.flow for r in displacement_results]),
                            'parameters': disp_params.__dict__
                        }
                        np.save(str(tfm_folder / "displacements.npy"), displacement_data)
                        displacement_data_available = True

                        print(f"\nDisplacement analysis completed in {time() - start_time:.1f} seconds")

                        # Displacement visualization if requested
                        if self.config['visualizations']['displacement_map']:
                            print("\nGenerating displacement visualization...")
                            viz_saver.save_displacement_visualization(displacement_data)
                            print("Displacement visualization saved successfully")

                    except Exception as e:
                        print(f"Displacement analysis failed: {str(e)}. Skipping this step.")

            #######################
            # Force Analysis
            #######################
            force_data_available = False
            if self.config['analysis_steps']['force']:
                if not displacement_data_available:
                    # Try to load displacement data
                    try:
                        displacement_data = np.load(str(tfm_folder / "displacements.npy"),
                                                    allow_pickle=True).item()
                        displacement_data_available = True
                    except Exception as e:
                        print(f"Could not load displacement data: {str(e)}. Skipping force analysis.")

                if displacement_data_available:
                    print("\nStarting Force Analysis...")
                    start_time = time()

                    try:
                        fttc_params = FTTCParameters(
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

                        self.fttc_service.initialize_calculator(fttc_params)

                        force_results = []
                        for progress in self.fttc_service.calculate_force_stack(
                                displacement_data['flows'],
                                self.config['parameters']['pixel_size'],
                                self.config['parameters']['downscale_factor']
                        ):
                            result, frame, total = progress
                            force_results.append(result)

                            # Calculate and display statistics
                            magnitude = np.sqrt(result['tx'] ** 2 + result['ty'] ** 2)
                            print(f"Frame {frame}/{total}: "
                                  f"Mean force: {np.mean(magnitude):.2f} Pa, "
                                  f"Max force: {np.max(magnitude):.2f} Pa")

                        # Save force results
                            force_data = {
                                'force_field': np.stack([np.stack([r['tx'], r['ty']], axis=-1)
                                                         for r in force_results]),
                                'parameters': fttc_params.__dict__
                            }
                            np.save(str(tfm_folder / "traction_forces.npy"), force_data)
                            force_data_available = True

                            print(f"\nForce analysis completed in {time() - start_time:.1f} seconds")

                            # Force visualizations if requested
                            if self.config['visualizations']['force_map']:
                                print("\nGenerating force map visualization...")
                                viz_saver.save_force_visualization(force_data)
                                print("Force map visualization saved successfully")

                            if self.config['visualizations']['force_cell_overlay']:
                                if Path(tfm_folder / "preprocessed_cells.tif").exists():
                                    print("\nGenerating force-cell overlay visualization...")
                                    preprocessed_cells = tifffile.imread(str(tfm_folder / "preprocessed_cells.tif"))
                                    viz_saver.save_force_cell_overlay(force_data, preprocessed_cells)
                                    print("Force-cell overlay visualization saved successfully")
                                else:
                                    print("No cell data available for force-cell overlay visualization")

                    except Exception as e:
                        print(f"Force analysis failed: {str(e)}. Skipping this step.")

            #######################
            # Mesh Generation
            #######################
            mesh_data_available = False
            if self.config['analysis_steps']['create_masks'] or self.config['analysis_steps']['stress']:
                print("\nStarting Mesh Generation...")
                start_time = time()

                try:
                    # Load cell images if not already preprocessed
                    if not Path(tfm_folder / "preprocessed_cells.tif").exists():
                        if 'cells' in self.config['input_files']:
                            print("Processing cell images for mask creation...")
                            cell_stack = tifffile.imread(str(folder / self.config['input_files']['cells']))
                            preproc_params = PreprocessingParameters(
                                cell_min_intensity_percentile=self.config['parameters']['cell_min_intensity'] / 100,
                                cell_max_intensity_percentile=self.config['parameters']['cell_max_intensity'] / 100,
                                cell_gaussian_sigma=self.config['parameters']['cell_gaussian_sigma']
                            )
                            self.preprocessing_service.update_parameters(preproc_params)
                            cell_results = []
                            for progress in self.preprocessing_service.preprocess_stack(cell_stack, is_cell=True):
                                result, frame, total = progress
                                cell_results.append(result)
                                print(f"Progress: {(frame / total) * 100:.1f}%, Frame {frame}/{total}")
                            preprocessed_cells = np.stack([r.processed_image for r in cell_results])
                            tifffile.imwrite(str(tfm_folder / "preprocessed_cells.tif"), preprocessed_cells)
                        else:
                            raise FileNotFoundError("No cell images available for mask creation")
                    else:
                        preprocessed_cells = tifffile.imread(str(tfm_folder / "preprocessed_cells.tif"))

                    # Create MSM parameters
                    msm_params = MSMParameters(
                        density_factor=self.config['parameters']['density_factor'],
                        algorithm=self.config['parameters']['mesh_algorithm'],
                        use_optimization=self.config['parameters']['use_optimization'],
                        poisson_ratio_cells=self.config['parameters']['poisson_ratio_cells'],
                        young_modulus=1.0,  # Unit Young's modulus for stress calculation
                        threshold=self.config['parameters']['threshold'],
                        dilation=self.config['parameters']['dilation'],
                        smoothing_sigma=self.config['parameters']['smoothing_sigma'],
                        max_stress=self.config['parameters']['max_stress']
                    )

                    # Generate masks and meshes
                    print("\nGenerating masks and meshes...")
                    mesh_results = []
                    masks = []
                    total_frames = len(preprocessed_cells)

                    for frame in range(total_frames):
                        # Create mask
                        mask = self.msm_service.create_mask_from_image(
                            preprocessed_cells[frame], msm_params)
                        masks.append(mask)

                        # Generate mesh
                        mesh_preview = self.msm_service.generate_mesh(mask, msm_params)
                        mesh_results.append(mesh_preview)

                        # Calculate mesh quality metrics
                        quality = mesh_preview.quality_metrics
                        print(f"\nFrame {frame + 1}/{total_frames}:")
                        print(f"  Average element quality: {quality['mean_quality']:.3f}")
                        print(f"  Minimum angle: {quality['min_angle']:.1f}°")
                        print(f"  Number of elements: {len(mesh_preview.elements)}")

                    # Save masks
                    masks = np.stack(masks)
                    tifffile.imwrite(str(tfm_folder / "masks.tif"), masks.astype(np.uint8) * 255)

                    # Save mesh data
                    mesh_data = {
                        'mesh_results': mesh_results,
                        'parameters': msm_params.__dict__
                    }
                    np.save(str(tfm_folder / "mesh_data.npy"), mesh_data)
                    mesh_data_available = True

                    print(f"\nMesh generation completed in {time() - start_time:.1f} seconds")

                    # Mesh visualization if requested
                    if self.config['visualizations']['mesh']:
                        print("\nGenerating mesh visualization...")
                        viz_saver.save_mesh_visualization(masks, mesh_results)
                        print("Mesh visualization saved successfully")

                except Exception as e:
                    print(f"Mesh generation failed: {str(e)}. Skipping this step.")

            #######################
            # Stress Analysis
            #######################
            if self.config['analysis_steps']['stress']:
                if not (force_data_available and mesh_data_available):
                    # Try to load required data
                    try:
                        if not force_data_available:
                            force_data = np.load(str(tfm_folder / "traction_forces.npy"),
                                                 allow_pickle=True).item()
                            force_data_available = True
                        if not mesh_data_available:
                            mesh_data = np.load(str(tfm_folder / "mesh_data.npy"),
                                                allow_pickle=True).item()
                            mesh_data_available = True
                    except Exception as e:
                        print(f"Could not load required data: {str(e)}. Skipping stress analysis.")

                if force_data_available and mesh_data_available:
                    print("\nStarting Stress Analysis...")
                    start_time = time()

                    try:
                        stress_results = []
                        for progress in self.msm_service.calculate_stress_stack(
                                force_data['force_field'],
                                mesh_data['parameters'],
                                mesh_data['mesh_results'],
                                masks
                        ):
                            result, frame, total = progress
                            stress_results.append(result)

                            # Calculate and display statistics
                            stress_magnitude = np.sqrt(np.sum(result.stress_tensor ** 2, axis=-1))
                            print(f"\nFrame {frame}/{total}:")
                            print(f"  Mean stress: {np.mean(stress_magnitude):.2f} mN/m")
                            print(f"  Max stress: {np.max(stress_magnitude):.2f} mN/m")
                            print(f"  Condition number: {result.condition_number:.2e}")
                            print(f"  Residual: {result.residual:.2e}")

                        # Save stress results
                        stress_data = {
                            'stress_tensor': np.stack([r.stress_tensor for r in stress_results]),
                            'parameters': msm_params.__dict__,
                            'condition_numbers': [r.condition_number for r in stress_results],
                            'residuals': [r.residual for r in stress_results]
                        }
                        np.save(str(tfm_folder / "stress_tensor.npy"), stress_data)

                        print(f"\nStress analysis completed in {time() - start_time:.1f} seconds")

                        # Stress visualizations if requested
                        if any([self.config['visualizations']['sigma_xx'],
                                self.config['visualizations']['sigma_yy'],
                                self.config['visualizations']['normal_stress']]):
                            print("\nGenerating stress visualizations...")
                            viz_saver.save_stress_visualization(
                                stress_data,
                                plot_sigma_xx=self.config['visualizations']['sigma_xx'],
                                plot_sigma_yy=self.config['visualizations']['sigma_yy'],
                                plot_normal_stress=self.config['visualizations']['normal_stress']
                            )
                            print("Stress visualizations saved successfully")

                    except Exception as e:
                        print(f"Stress analysis failed: {str(e)}. Skipping this step.")

            print("\nFolder processing completed successfully!")
            print("=" * 80)

        finally:
            # Clean up
            self._cleanup_internal_storage()
            if self._tee_logger:
                self._tee_logger.close()
                self._tee_logger = None

    def _cleanup_internal_storage(self):
        """Clean up all internal data storage to free memory."""
        self._preprocessed_data = None
        self._displacement_data = None
        self._force_data = None
        self._stress_data = None

        # Force garbage collection
        import gc
        gc.collect()

    def process_all_folders(self) -> None:
        """Process all folders specified in configuration."""
        for folder in self.config['root_folders']:
            self.process_folder(folder)

if __name__ == "__main__":
    analyzer = BatchAnalysis.from_yaml("batch_config.yaml")
    analyzer.process_all_folders()