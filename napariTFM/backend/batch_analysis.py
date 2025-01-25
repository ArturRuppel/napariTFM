import json

import yaml
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import tifffile
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from preprocessing import PreprocessingParameters, ImagePreprocessor
from displacement_analysis import TVL1Parameters, DisplacementAnalyzer
from fttc import FTTC
from msm import MonolayerStressMicroscopy


class BatchAnalysis:
    """Handles batch analysis of TFM data according to YAML configuration."""

    def __init__(self, config_path: str):
        """Initialize batch analysis with configuration file."""
        logger.info("Initializing BatchAnalysis")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            logger.debug(f"Loaded configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize data containers
        # 1. Input data
        self._input_bead_stack: Optional[np.ndarray] = None
        self._input_reference: Optional[np.ndarray] = None
        self._input_cell_stack: Optional[np.ndarray] = None

        # 2. Preprocessed data
        self._preprocessed_bead_stack: Optional[np.ndarray] = None
        self._preprocessed_reference: Optional[np.ndarray] = None
        self._preprocessed_cell_stack: Optional[np.ndarray] = None
        self._preprocessing_params: Optional[Dict[str, Any]] = None

        # 3. Displacement results
        self._displacement_field: Optional[np.ndarray] = None
        self._displacement_params: Optional[Dict[str, Any]] = None

        # 4. Force results
        self._force_field: Optional[np.ndarray] = None
        self._force_params: Optional[Dict[str, Any]] = None

        # 5. Stress results
        self._stress_tensor: Optional[np.ndarray] = None
        self._stress_params: Optional[Dict[str, Any]] = None

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
        logger.info(f"Saved calibrated TIFF: {filepath}")

    def process_folder(self, folder_path: str) -> None:
        """Process a single folder according to configuration."""
        logger.info(f"Processing folder: {folder_path}")
        folder = Path(folder_path)

        # Create output folders
        tfm_folder = folder / "TFM_data"
        tfm_folder.mkdir(exist_ok=True)
        logger.debug(f"Created TFM data folder: {tfm_folder}")

        # Keep track of active worker
        self._current_worker = None

        try:
            # Execute enabled analysis steps
            if self.config['analysis_steps']['preprocessing']:
                logger.info("Starting preprocessing step")
                self._run_preprocessing(folder)

                # Clean up preprocessing worker
                if hasattr(self, '_current_worker') and self._current_worker is not None:
                    self._current_worker.quit()
                    self._current_worker = None
                    logger.debug("Cleaned up preprocessing worker")

            if self.config['analysis_steps']['displacement']:
                self._run_displacement_analysis(folder)

            if self.config['analysis_steps']['force']:
                self._run_force_analysis(folder)

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
            # Ensure worker is cleaned up even if there's an error
            if hasattr(self, '_current_worker') and self._current_worker is not None:
                self._current_worker.quit()
                self._current_worker = None
                logger.debug("Cleaned up worker in finally block")

    def _load_input_data(self, folder: Path) -> None:
        """Load input data from specified folder."""
        logger.info("Loading input data")

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

    def _run_preprocessing(self, folder: Path) -> None:
        """Run preprocessing step."""
        logger.info("Starting preprocessing")

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
            logger.info("Running preprocessing")
            preprocessor = ImagePreprocessor(params)
            generator = preprocessor.preprocess_all(
                self._input_bead_stack,
                self._input_reference,
                self._input_cell_stack
            )

            results = self._run_thread_worker(generator)

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
            frame_interval = self.config.get('parameters', {}).get('frame_interval', 1.0)  # default 1 min if not specified

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

            logger.info("Preprocessing results saved successfully")

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise

    def _run_displacement_analysis(self, folder: Path) -> None:
        """Run displacement analysis step."""
        logger.info("Starting displacement analysis")

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

                # Load metadata from TIFF if needed
                with tifffile.TiffFile(str(preproc_path)) as tif:
                    if tif.imagej_metadata:
                        self._preprocessing_params = {
                            'parameters': {
                                'pixel_size': tif.imagej_metadata.get('spacing', 1.0),
                                'frame_interval': tif.imagej_metadata.get('frame_interval', 1.0)
                            }
                        }
                logger.info("Successfully loaded preprocessed data from TIFF files")

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
            # Run displacement analysis
            logger.info("Running displacement analysis")
            analyzer = DisplacementAnalyzer(params)
            worker = analyzer.analyze_displacement(
                self._preprocessed_reference,
                self._preprocessed_bead_stack,
                self.config['parameters']['pixel_size'],
                self.config['parameters']['downscale_factor']
            )

            # Run the worker using _run_thread_worker
            results = self._run_thread_worker(worker)

            if results is None:
                raise RuntimeError("Displacement analysis did not return any results")

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
                'visualization_params': {
                    'd_max': self.config['parameters']['d_max'],
                    'vector_stride': self.config['parameters']['disp_vector_stride'],
                    'arrow_scale': self.config['parameters']['disp_arrow_scale']
                }
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
            logger.info("Displacement analysis results saved successfully")

        except Exception as e:
            logger.error(f"Error during displacement analysis: {str(e)}", exc_info=True)
            raise

    def _run_force_analysis(self, folder: Path) -> None:
        """Run force analysis step."""
        logger.info("Starting force analysis")

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
                logger.info("Successfully loaded displacement data")

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
                nu=self.config['parameters']['poisson_ratio'],
                lanczos_exp=self.config['parameters']['lanczos_exp'],
                gel_height=gel_height
            )

            # Run force calculation for each frame
            logger.info("Running force calculations")
            forces = []
            total_frames = len(self._displacement_field)

            for frame_idx, displacement in enumerate(self._displacement_field):
                logger.debug(f"Processing frame {frame_idx + 1}/{total_frames}")

                # Create and run worker for this frame
                worker = fttc.calculate_traction(
                    displacement,
                    self.config['parameters']['pixel_size'],
                    downscale_factor=self.config['parameters']['downscale_factor'],
                    regularization=10 ** self.config['parameters']['regularization']
                )

                results = self._run_thread_worker(worker)

                if results is None:
                    raise RuntimeError(f"Force calculation failed for frame {frame_idx + 1}")

                # Store force field (results[1] contains the forces)
                forces.append(results[1])

                logger.info(f'Calculated forces for frame {frame_idx + 1}/{total_frames} ({(frame_idx + 1) / total_frames * 100:.1f}%)')

            logger.debug("Force calculations completed successfully")

            # Create parameters dictionary in the expected format
            formatted_params = {
                'young_modulus': self.config['parameters']['young_modulus'],
                'poisson_ratio': self.config['parameters']['poisson_ratio'],
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
            self._force_field = np.moveaxis(np.array(forces), 1, -1)   # reformat to make compatible with widgets
            self._force_params = formatted_params


            # Save results in widget-compatible format
            output_path = folder / self.config['output_files']['force']['data']
            np.save(str(output_path), {
                'force_field': self._force_field,
                'parameters': self._force_params
            })
            logger.info("Force analysis results saved successfully")

        except Exception as e:
            logger.error(f"Error during force analysis: {str(e)}", exc_info=True)
            raise

    def _run_stress_analysis(self, folder: Path) -> None:
        """Run stress analysis step."""
        # Load force data if not in memory
        if self._force_field is None:
            force_path = folder / self.config['output_files']['force']['data']
            loaded = np.load(str(force_path), allow_pickle=True).item()
            self._force_field = loaded['data']
            self._force_params = loaded['parameters']

        # Load masks if needed
        if self.config['analysis_steps']['create_masks']:
            mask_path = folder / self.config['output_files']['masks']['path']
            masks = np.load(str(mask_path))
        else:
            # Create simple mask from preprocessed cell images
            masks = self._create_default_masks()

        # Calculate stress tensors for each frame
        stress_tensors = []
        for frame_idx in range(len(self._force_field)):
            msm = MonolayerStressMicroscopy(
                mask=masks[frame_idx],
                density_factor=self.config['parameters']['density_factor'],
                algorithm=2,  # Default to Frontal-Delaunay
                use_optimization=self.config['parameters']['use_optimization'],
                young_modulus=self.config['parameters']['young_modulus'],
                poisson_ratio=self.config['parameters']['poisson_ratio']
            )

            stress_tensor, _, _ = msm.calculate_stress_field(
                self._force_field[frame_idx, ..., 0],
                self._force_field[frame_idx, ..., 1]
            )
            stress_tensors.append(stress_tensor)

        self._stress_tensor = np.stack(stress_tensors)
        self._stress_params = {
            'density_factor': self.config['parameters']['density_factor'],
            'use_optimization': self.config['parameters']['use_optimization'],
            'young_modulus': self.config['parameters']['young_modulus'],
            'poisson_ratio': self.config['parameters']['poisson_ratio']
        }

        # Save results
        output_path = folder / self.config['output_files']['stress']['data']
        np.save(str(output_path), {
            'data': self._stress_tensor,
            'parameters': self._stress_params
        })

    def _create_default_masks(self) -> np.ndarray:
        """Create default masks from preprocessed cell images."""
        if self._preprocessed_cell_stack is None:
            raise ValueError("Cell images required for mask creation")

        from skimage.filters import threshold_otsu
        from scipy.ndimage import binary_dilation, gaussian_filter

        masks = []
        for frame in self._preprocessed_cell_stack:
            # Apply Gaussian smoothing
            smoothed = gaussian_filter(frame, self.config['parameters']['smoothing_sigma'])

            # Threshold
            thresh = np.percentile(smoothed, self.config['parameters']['threshold'])
            mask = smoothed > thresh

            # Dilate
            mask = binary_dilation(mask, iterations=self.config['parameters']['dilation'])
            masks.append(mask)

        return np.stack(masks)

    def process_all_folders(self) -> None:
        """Process all folders specified in configuration."""
        for folder in self.config['root_folders']:
            print(f"Processing folder: {folder}")
            self.process_folder(folder)


if __name__ == "__main__":
    # Example usage
    config_path = "batch_config.yaml"
    analyzer = BatchAnalysis(config_path)
    analyzer.process_all_folders()
