import json
import yaml
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Generator
from skimage.transform import resize
import tifffile
import sys
from datetime import datetime

from napariTFM.backend.preprocessing import PreprocessingParameters, ImagePreprocessor
from napariTFM.backend.displacement_analysis import TVL1Parameters, DisplacementAnalyzer
from napariTFM.backend.fttc import FTTC
from napariTFM.backend.msm import MonolayerStressMicroscopy

import logging
import warnings


class TeeLogger:
    """Custom logger that captures print statements and logging output to both console and file."""

    def __init__(self, filename: Path, config: dict = None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = open(filename, 'w', encoding='utf-8')
        self.start_time = datetime.now()

        # Write header to log file
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
        self.terminal.write(message)
        self.log.write(message)
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


def setup_logging(level: str = "INFO", silent: bool = True, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging with configurable level and optional file output.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL")
        silent: If True, suppress all output
        log_file: Optional path to log file

    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)

    if silent:
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add null handler to suppress all output
        logger.addHandler(logging.NullHandler())
        # Set level to CRITICAL+1 to suppress all messages
        logger.setLevel(logging.CRITICAL + 1)
    else:
        # Convert string to logging level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        # Configure logging
        if log_file:
            logging.basicConfig(
                level=numeric_level,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(log_file)
                ]
            )
        else:
            logging.basicConfig(level=numeric_level)

    return logger


# Default to INFO level and not silent
logger = setup_logging()

# Create a warning filter for tifffile warnings
warnings.filterwarnings('ignore', message='.*not writing description to ImageJ file.*',
                        module='tifffile.tifffile')


class BatchAnalysis:
    """Handles batch analysis of TFM data according to YAML configuration."""
    MESH_ALGORITHMS = {
        "Frontal-Del.": 6,
        "Delaunay": 5,
        "MeshAdapt": 1,
        "BAMG": 7,
        "FD Quads": 8,
        "Para. Pack": 9
    }

    def __init__(self, config_path: str):
        """Initialize batch analysis with configuration file."""
        print("Initializing BatchAnalysis")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            logger.debug(f"Loaded configuration from {config_path}")

        # Initialize data containers
        self._initialize_data_containers()
        self._tee_logger = None

    def process_folder(self, folder_path: str) -> None:
        """Process a single folder according to configuration."""
        folder = Path(folder_path)
        log_file = folder / "processing_log.txt"

        # Set up logging to file
        self._tee_logger = TeeLogger(log_file)
        sys.stdout = self._tee_logger
        logger = setup_logging(silent=False, log_file=log_file)

        print(f"Processing folder: {folder_path}")

        # Create output folders
        tfm_folder = folder / "TFM_data"
        tfm_folder.mkdir(exist_ok=True)
        logger.debug(f"Created TFM data folder: {tfm_folder}")

        try:
            # Execute enabled analysis steps
            if self.config['analysis_steps']['preprocessing']:
                print("Starting preprocessing step")
                for progress in self._run_preprocessing(folder):
                    print(f"Progress: {progress['progress']:.1f}%, {progress['message']}")

            if self.config['analysis_steps']['create_masks']:
                print("Starting mask creation")
                self._create_masks(folder)
                print("Mask creation completed")

            if self.config['analysis_steps']['displacement']:
                print("Starting displacement analysis")
                for progress in self._run_displacement_analysis(folder):
                    print(f"Progress: {progress['progress']:.1f}%, {progress['message']}")

            if self.config['analysis_steps']['force']:
                print("Starting force analysis")
                for progress in self._run_force_analysis(folder):
                    print(f"Progress: {progress['progress']:.1f}%, {progress['message']}")

            if self.config['analysis_steps']['stress']:
                # Only proceed with stress analysis if force data exists
                force_path = folder / self.config['output_files']['force']['data']
                if not force_path.exists():
                    logger.warning(f"Force data not found at {force_path}, skipping stress analysis")
                else:
                    self._run_stress_analysis(folder)

        except Exception as e:
            logger.error(f"Error processing folder: {str(e)}", exc_info=True)
            raise

        finally:
            # Clean up worker and internal storage regardless of success or failure
            print(f"Cleaning up after processing folder: {folder_path}")
            self._cleanup_internal_storage()

            # Close and reset the TeeLogger
            if self._tee_logger:
                self._tee_logger.close()
                self._tee_logger = None

            print(f"Finished processing folder: {folder_path}")

    def _initialize_data_containers(self):
        """Initialize or reset all data containers to their default state."""
        logger.debug("Initializing/resetting data containers")

        # Initialize all data containers to None
        self._input_bead_stack: Optional[np.ndarray] = None
        self._input_reference: Optional[np.ndarray] = None
        self._input_cell_stack: Optional[np.ndarray] = None
        self._preprocessed_bead_stack: Optional[np.ndarray] = None
        self._preprocessed_reference: Optional[np.ndarray] = None
        self._preprocessed_cell_stack: Optional[np.ndarray] = None
        self._preprocessing_params: Optional[Dict[str, Any]] = None
        self._displacement_field: Optional[np.ndarray] = None
        self._displacement_params: Optional[Dict[str, Any]] = None
        self._force_field: Optional[np.ndarray] = None
        self._force_params: Optional[Dict[str, Any]] = None
        self._stress_tensor: Optional[np.ndarray] = None
        self._stress_params: Optional[Dict[str, Any]] = None
        self._current_worker = None

    def _cleanup_internal_storage(self):
        """Clean up all internal data storage to free memory."""
        print("Cleaning up internal storage")

        # Clean up worker if it exists
        if self._current_worker is not None:
            self._current_worker.quit()
            self._current_worker = None
            logger.debug("Cleaned up current worker")

        # Reset all data containers
        self._initialize_data_containers()

        # Force garbage collection to free memory
        import gc
        gc.collect()
        logger.debug("Garbage collection completed")

    def process_all_folders(self) -> None:
        """Process all folders specified in configuration."""
        for folder in self.config['root_folders']:
            self.process_folder(folder)

    def _create_masks(self, folder: Path) -> np.ndarray:
        """Create masks from preprocessed cell images using MSM class."""
        print("Creating masks from preprocessed cell images")

        # Load preprocessed cell images if not already in memory
        if self._preprocessed_cell_stack is None:
            cell_path = folder / self.config['output_files']['preprocessing']['cells']
            if not cell_path.exists():
                raise FileNotFoundError(f"Preprocessed cell images not found at {cell_path}")
            logger.debug(f"Loading preprocessed cell images from {cell_path}")
            self._preprocessed_cell_stack = tifffile.imread(str(cell_path))

        # Get mask parameters from config
        mask_params = {
            'threshold_percentile': self.config['parameters']['threshold'],
            'dilation': self.config['parameters']['dilation'],
            'smoothing_sigma': self.config['parameters']['smoothing_sigma']
        }

        logger.debug(f"Creating masks with parameters: {mask_params}")

        # Create masks using MSM class method
        if self._preprocessed_cell_stack.ndim == 2:
            # Single frame
            mask = MonolayerStressMicroscopy.create_mask_from_image(
                self._preprocessed_cell_stack,
                **mask_params
            )
            masks = mask[np.newaxis, ...]
        else:
            # Multiple frames
            masks = np.zeros_like(self._preprocessed_cell_stack, dtype=bool)
            total_frames = len(self._preprocessed_cell_stack)

            for frame in range(total_frames):
                logger.debug(f"Processing frame {frame + 1}/{total_frames}")
                masks[frame] = MonolayerStressMicroscopy.create_mask_from_image(
                    self._preprocessed_cell_stack[frame],
                    **mask_params
                )

        # Save masks
        output_path = folder / self.config['output_files']['masks']['path']
        print(f"Saving masks to {output_path}")

        # Also save as TIFF for visualization
        tiff_path = output_path.with_suffix('.tif')
        self._save_calibrated_tiff(
            masks.astype(np.uint8) * 255,  # Convert to 8-bit
            tiff_path,
            self.config['parameters']['pixel_size'],
            self.config['parameters']['frame_interval']
        )

        return masks

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

        logger.debug(f"Saving calibrated TIFF to {filepath}")

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

    def _load_input_data(self, folder: Path) -> None:
        """Load input data from specified folder."""
        print("Loading input data")

        # Load bead images
        bead_path = folder / self.config['input_files']['beads']
        logger.debug(f"Loading bead images from: {bead_path}")
        self._input_bead_stack = tifffile.imread(str(bead_path))
        logger.debug(f"Loaded bead stack with shape: {self._input_bead_stack.shape}")

        # Load reference image
        ref_path = folder / self.config['input_files']['reference']
        logger.debug(f"Loading reference image from: {ref_path}")
        self._input_reference = tifffile.imread(str(ref_path))
        logger.debug(f"Loaded reference image with shape: {self._input_reference.shape}")

        # Load cell images if available
        if 'cells' in self.config['input_files']:
            cell_path = folder / self.config['input_files']['cells']
            if cell_path.exists():
                logger.debug(f"Loading cell images from: {cell_path}")
                self._input_cell_stack = tifffile.imread(str(cell_path))
                logger.debug(f"Loaded cell stack with shape: {self._input_cell_stack.shape}")

    def _run_thread_worker(self, worker):
        """
        Helper function to run a thread_worker and collect its results.
        """
        logger.debug("Starting thread worker")

        # Clean up any existing worker
        if hasattr(self, '_current_worker') and self._current_worker is not None:
            self._current_worker.quit()
            self._current_worker = None
            logger.debug("Cleaned up previous worker")

        # Store current worker
        self._current_worker = worker

        # Set up result container
        final_result = None

        def _on_returned(result):
            nonlocal final_result
            final_result = result
            logger.debug("Thread worker completed successfully")

        def _on_errored(error):
            logger.error(f"Thread worker error: {str(error)}", exc_info=True)
            raise error

        try:
            # Connect worker signals
            worker.returned.connect(_on_returned)
            worker.errored.connect(_on_errored)

            # Start worker
            worker.start()
            worker.run()  # This will block until completion

            return final_result

        except Exception as e:
            logger.error(f"Error in thread worker: {str(e)}", exc_info=True)
            raise

        finally:
            # Clean up worker
            if self._current_worker is not None:
                self._current_worker.quit()
                self._current_worker = None
                logger.debug("Cleaned up worker in finally block")

    def _run_preprocessing(self, folder: Path) -> Generator:
        """Run preprocessing step."""
        print("Starting preprocessing")

        # Load input data
        try:
            self._load_input_data(folder)
        except Exception as e:
            logger.error(f"Error loading input data: {str(e)}", exc_info=True)
            raise

        # Configure preprocessing parameters
        logger.debug("Configuring preprocessing parameters")
        params = PreprocessingParameters(
            min_intensity_percentile=self.config['parameters']['min_intensity'] / 100,
            max_intensity_percentile=self.config['parameters']['max_intensity'] / 100,
            gaussian_sigma=self.config['parameters']['gaussian_sigma'],
            cell_min_intensity_percentile=self.config['parameters']['cell_min_intensity'] / 100,
            cell_max_intensity_percentile=self.config['parameters']['cell_max_intensity'] / 100,
            cell_gaussian_sigma=self.config['parameters']['cell_gaussian_sigma'],
            registration_mode=self.config['parameters']['registration_mode']
        )

        # Run preprocessing
        try:
            print("Running preprocessing")
            preprocessor = ImagePreprocessor(params)
            results = yield from preprocessor.preprocess_all_generator(self._input_bead_stack, self._input_reference, self._input_cell_stack)

            if results is None:
                raise RuntimeError("Preprocessing did not return any results")

            logger.debug("Processing completed successfully")

            # Store results
            self._preprocessed_bead_stack = results['beads'][0]
            self._preprocessed_reference = results['reference'][0]
            if 'cells' in results:
                self._preprocessed_cell_stack = results['cells'][0]

            self._preprocessing_params = {
                'parameters': params.__dict__,
                'transform_matrices': preprocessor.transform_matrices
            }

            # Create TFM data folder
            tfm_folder = folder / "TFM_data"
            tfm_folder.mkdir(exist_ok=True)

            # Get calibration values from config
            pixel_size = self.config['parameters']['pixel_size']
            frame_interval = self.config.get('parameters', {}).get('frame_interval', 1.0)

            # Save calibrated TIFFs
            if self._preprocessed_bead_stack is not None:
                self._save_calibrated_tiff(
                    self._preprocessed_bead_stack,
                    tfm_folder / "preprocessed_beads.tif",
                    pixel_size,
                    frame_interval
                )

            if self._preprocessed_reference is not None:
                self._save_calibrated_tiff(
                    self._preprocessed_reference,
                    tfm_folder / "preprocessed_reference.tif",
                    pixel_size,
                    frame_interval
                )

            if self._preprocessed_cell_stack is not None:
                self._save_calibrated_tiff(
                    self._preprocessed_cell_stack,
                    tfm_folder / "preprocessed_cells.tif",
                    pixel_size,
                    frame_interval
                )

            print("Preprocessing results saved successfully")

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise

    def _run_displacement_analysis(self, folder: Path) -> Generator[dict, None, None]:
        """Run displacement analysis step with progress reporting."""
        # Load preprocessed data if not in memory
        if self._preprocessed_bead_stack is None:
            try:
                logger.debug("Loading preprocessed data from TIFF files")
                preproc_path = folder / "TFM_data/preprocessed_beads.tif"
                ref_path = folder / "TFM_data/preprocessed_reference.tif"

                if not preproc_path.exists():
                    raise FileNotFoundError(f"Preprocessed bead stack not found at {preproc_path}")
                if not ref_path.exists():
                    raise FileNotFoundError(f"Preprocessed reference not found at {ref_path}")

                self._preprocessed_bead_stack = tifffile.imread(str(preproc_path))
                self._preprocessed_reference = tifffile.imread(str(ref_path))
                print("Successfully loaded preprocessed data from TIFF files")

            except Exception as e:
                logger.error(f"Error loading preprocessed data: {str(e)}", exc_info=True)
                raise

        # Configure displacement parameters
        logger.debug("Configuring displacement parameters")
        params = TVL1Parameters(
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
            downscale_factor=self.config['parameters']['downscale_factor']
        )

        try:
            print("Running displacement analysis")
            analyzer = DisplacementAnalyzer(params)

            # Get visualization parameters from config
            vis_params = {
                'd_max': self.config['parameters']['d_max'],
                'vector_stride': self.config['parameters']['disp_vector_stride'],
                'arrow_scale': self.config['parameters']['disp_arrow_scale']
            }

            # Create the generator
            disp_generator = analyzer.analyze_displacement_generator(
                reference=self._preprocessed_reference,
                bead_stack=self._preprocessed_bead_stack,
                pixel_size=self.config['parameters']['pixel_size'],
                downscale_factor=self.config['parameters']['downscale_factor'],
                visualization_params=vis_params
            )

            # Yield progress from the generator
            try:
                while True:
                    progress = next(disp_generator)
                    yield progress
            except StopIteration as e:
                results = e.value

            # Process final results
            logger.debug("Displacement analysis completed successfully")

            # Format parameters to match widget expectations
            formatted_params = {
                'tvl1_params': {
                    'tau': params.tau,
                    'lambda': params.lambda_,
                    'theta': params.theta,
                    'nscales': params.nscales,
                    'warps': params.warps,
                    'epsilon': params.epsilon,
                    'inner_iterations': params.inner_iterations,
                    'outer_iterations': params.outer_iterations,
                    'scale_step': params.scale_step,
                    'median_filtering': params.median_filtering
                },
                'downscale_factor': self.config['parameters']['downscale_factor'],
                'pixel_size': self.config['parameters']['pixel_size'],
                'frame_interval': self.config['parameters']['frame_interval'],
                'visualization_params': vis_params
            }

            # Store results
            self._displacement_field = np.array(results['flows'])
            self._displacement_params = formatted_params

            # Save results in compatible format
            output_path = folder / self.config['output_files']['displacement']['data']
            np.save(str(output_path), {
                'flows': self._displacement_field,
                'parameters': formatted_params,
                'original_shape': self._preprocessed_reference.shape,
                'flow_shape': self._displacement_field[0].shape[:2],
                'units': 'micrometers'
            })
            print("Displacement analysis results saved successfully")

        except Exception as e:
            logger.error(f"Error during displacement analysis: {str(e)}", exc_info=True)
            raise

    def _run_force_analysis(self, folder: Path) -> Generator[dict, None, None]:
        """Run force analysis step with progress reporting."""
        print("Starting force analysis")

        # Load displacement data if not in memory
        if self._displacement_field is None:
            try:
                logger.debug("Loading displacement data from NPY file")
                disp_path = folder / self.config['output_files']['displacement']['data']

                if not disp_path.exists():
                    raise FileNotFoundError(f"Displacement data not found at {disp_path}")

                loaded = np.load(str(disp_path), allow_pickle=True).item()
                self._displacement_field = loaded['flows']
                self._displacement_params = loaded['parameters']
                print("Successfully loaded displacement data")

            except Exception as e:
                logger.error(f"Error loading displacement data: {str(e)}", exc_info=True)
                raise

        try:
            # Configure FTTC calculator
            logger.debug("Configuring FTTC calculator")
            if self.config['parameters'].get('gel_height', float('inf')) == 0:
                gel_height = None
            else:
                gel_height = self.config['parameters'].get('gel_height', float('inf'))

            fttc = FTTC(
                E=self.config['parameters']['young_modulus'],
                nu=self.config['parameters']['poisson_ratio_substrate'],
                lanczos_exp=self.config['parameters']['lanczos_exp'],
                gel_height=gel_height
            )

            # Initialize results storage
            forces = []
            total_frames = len(self._displacement_field)
            processed_frames = 0

            # Create generator for progress reporting
            def force_generator():
                nonlocal processed_frames, forces
                for frame_idx, displacement in enumerate(self._displacement_field):
                    logger.debug(f"Processing frame {frame_idx + 1}/{total_frames}")

                    # Calculate forces for this frame
                    result = fttc.calculate_traction(
                        displacement,
                        self.config['parameters']['pixel_size'],
                        downscale_factor=self.config['parameters']['downscale_factor'],
                        regularization=10 ** self.config['parameters']['regularization']
                    )

                    # Store results
                    forces.append(result[1])
                    processed_frames += 1

                    # Calculate progress
                    progress = (frame_idx + 1) / total_frames * 100
                    magnitude = np.sqrt(result[1][0] ** 2 + result[1][1] ** 2)

                    # Yield progress update
                    yield {
                        'progress': progress,
                        'message': (f"Frame {frame_idx + 1}/{total_frames} - "
                                    f"Mean: {np.mean(magnitude):.2f} Pa | "
                                    f"Max: {np.max(magnitude):.2f} Pa")
                    }

            # Run through the generator
            generator = force_generator()
            while True:
                try:
                    progress = next(generator)
                    yield progress
                except StopIteration:
                    break

            # Final processing after all frames
            logger.debug("Force calculations completed successfully")

            # Create parameters dictionary
            formatted_params = {
                'young_modulus': self.config['parameters']['young_modulus'],
                'poisson_ratio_substrate': self.config['parameters']['poisson_ratio_substrate'],
                'gel_height': self.config['parameters'].get('gel_height', None),
                'pixel_size': self.config['parameters']['pixel_size'],
                'frame_interval': self.config['parameters']['frame_interval'],
                'regularization': 10 ** self.config['parameters']['regularization'],
                'lanczos_exp': self.config['parameters']['lanczos_exp'],
                'downscale_factor': self.config['parameters']['downscale_factor'],
                'visualization': {
                    'vector_stride': self.config['parameters']['force_vector_stride'],
                    'arrow_scale': self.config['parameters']['force_arrow_scale'],
                    'f_max': self.config['parameters']['f_max']
                }
            }

            # Format and store results
            self._force_field = np.moveaxis(np.array(forces), 1, -1)
            self._force_params = formatted_params

            # Save results
            output_path = folder / self.config['output_files']['force']['data']
            np.save(str(output_path), {
                'force_field': self._force_field,
                'parameters': self._force_params
            })
            print("Force analysis results saved successfully")

        except Exception as e:
            logger.error(f"Error during force analysis: {str(e)}", exc_info=True)
            raise
    def _run_stress_analysis(self, folder: Path) -> None:
        """Run stress analysis step."""
        logger.debug("Loading masks")
        mask_path = folder / self.config['output_files']['masks']['path']
        masks = tifffile.imread(str(mask_path.with_suffix('.tif'))) > 0

        print("Starting stress analysis")
        # Load force data if not in memory
        if self._force_field is None:
            try:
                logger.debug("Loading force data")
                force_path = folder / self.config['output_files']['force']['data']
                loaded = np.load(str(force_path), allow_pickle=True).item()
                self._force_field = loaded['force_field']
                self._force_params = loaded['parameters']
            except Exception as e:
                logger.error(f"Error loading force data: {str(e)}", exc_info=True)
                raise

        # Resize masks to match force field dimensions
        force_shape = self._force_field.shape[1:3]  # Get the spatial dimensions of force field
        resized_masks = np.zeros((len(masks), *force_shape), dtype=bool)

        for i in range(len(masks)):
            # Use resize with order=0 for nearest-neighbor interpolation to preserve binary nature
            resized_masks[i] = resize(masks[i].astype(float),
                                      force_shape,
                                      order=0,
                                      preserve_range=True,
                                      anti_aliasing=False) > 0.5

        masks = resized_masks
        logger.debug(f"Resized masks to match force field shape: {force_shape}")

        try:
            # Calculate stress tensors for each frame
            print("Calculating stress tensors")
            stress_tensors = []
            total_frames = len(self._force_field)

            for frame_idx in range(total_frames):
                logger.debug(f"Processing frame {frame_idx + 1}/{total_frames}")
                msm = MonolayerStressMicroscopy(
                    mask=masks[frame_idx] if masks.ndim > 2 else masks,
                    density_factor=self.config['parameters']['density_factor'],
                    algorithm=self.MESH_ALGORITHMS.get(self.config['parameters']['mesh_algorithm']),
                    use_optimization=self.config['parameters']['use_optimization'],
                    young_modulus=1.0,
                    poisson_ratio=self.config['parameters']['poisson_ratio_cells']
                )

                stress_tensor, cond_num, residual = msm.calculate_stress_field(
                    self._force_field[frame_idx, ..., 0],
                    self._force_field[frame_idx, ..., 1]
                )
                stress_tensors.append(stress_tensor)
                print(f"Completed frame {frame_idx + 1}/{total_frames} (condition number: {cond_num:.2e}, residual: {residual:.2e})")

            self._stress_tensor = np.stack(stress_tensors)
            self._stress_params = {
                'density_factor': self.config['parameters']['density_factor'],
                'use_optimization': self.config['parameters']['use_optimization'],
                'poisson_ratio_cells': self.config['parameters']['poisson_ratio_cells'],
                'max_stress': self.config['parameters']['max_stress']
            }

            # Save results
            output_path = folder / self.config['output_files']['stress']['data']
            np.save(str(output_path), {
                'stress_tensor': self._stress_tensor,
                'parameters': self._stress_params
            })
            print("Stress analysis results saved successfully")

        except Exception as e:
            logger.error(f"Error during stress tensor calculation: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Example usage
    config_path = "batch_config.yaml"
    analyzer = BatchAnalysis(config_path)
    analyzer.process_all_folders()
