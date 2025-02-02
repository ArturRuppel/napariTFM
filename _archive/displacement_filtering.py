import numpy as np
from scipy import fft
from pathlib import Path
import numpy.typing as npt
from typing import Tuple
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from batch_analysis_visualizations import BatchVisualizationSaver
from napariTFM.services.displacement_service import DisplacementResult


def create_3d_butterworth_filter(shape: Tuple[int, ...], cutoff_freq: float, order: int = 1) -> npt.NDArray:
    """
    Create a 3D Butterworth low-pass filter mask.

    Args:
        shape: Shape of the filter (time, y, x)
        cutoff_freq: Normalized cutoff frequency (0 to 1)
        order: Filter order (steepness of cutoff)

    Returns:
        3D filter mask with same shape as input
    """
    t, y, x = shape
    t_center = t // 2
    y_center = y // 2
    x_center = x // 2

    # Create frequency coordinates
    t_coords, y_coords, x_coords = np.mgrid[:t, :y, :x]
    t_coords = ((t_coords - t_center) / t).astype(float)
    y_coords = ((y_coords - y_center) / y).astype(float)
    x_coords = ((x_coords - x_center) / x).astype(float)

    # Calculate frequency distance from origin
    freq_dist = np.sqrt(t_coords ** 2 + y_coords ** 2 + x_coords ** 2)

    # Create Butterworth filter
    filter_mask = 1 / (1 + (freq_dist / cutoff_freq) ** (2 * order))

    return fft.fftshift(filter_mask)


def calculate_vector_coherence(displacement: npt.NDArray) -> npt.NDArray:
    """
    Calculate local coherence metric combining angle and magnitude information.
    """
    # Calculate magnitudes
    magnitudes = np.sqrt(np.sum(displacement ** 2, axis=-1))

    # Calculate angles
    angles = np.arctan2(displacement[..., 1], displacement[..., 0])

    # Normalize magnitudes to [0, 1] range for each time point
    mag_max = np.maximum(magnitudes.max(axis=(1, 2), keepdims=True), 1e-10)
    norm_magnitudes = magnitudes / mag_max

    # Calculate gradient of both components
    dx = np.gradient(displacement, axis=2)
    dy = np.gradient(displacement, axis=1)
    dt = np.gradient(displacement, axis=0)

    # Combined spatial and temporal gradients
    spatial_grad = np.sqrt(np.sum(dx ** 2 + dy ** 2, axis=-1))
    temporal_grad = np.sqrt(np.sum(dt ** 2, axis=-1))

    # Combine metrics (lower value means more coherent)
    coherence = (spatial_grad + temporal_grad) * (1 + norm_magnitudes)

    return coherence


def filter_displacement_field(
        displacement: npt.NDArray,
        cutoff_freq: float = 0.1,
        filter_order: int = 2,
        kernel_size: int = 5,
) -> npt.NDArray:
    """
    Filter displacement field using 3D frequency domain filtering and vector coherence.
    """
    print(f"Input shape: {displacement.shape}")

    # Calculate vector coherence
    coherence = calculate_vector_coherence(displacement)
    print(f"Coherence range: {coherence.min():.3f} to {coherence.max():.3f}")

    # Process x and y components separately
    filtered_field = np.zeros_like(displacement)

    for component in range(2):
        # Get component data
        data = displacement[..., component]

        # Apply FFT
        fft_data = fft.fftn(data)

        # Create and apply filter
        filter_mask = create_3d_butterworth_filter(
            data.shape,
            cutoff_freq,
            filter_order
        )
        filtered_fft = fft_data * filter_mask

        # Inverse FFT
        filtered_component = np.real(fft.ifftn(filtered_fft))

        # Store filtered component
        filtered_field[..., component] = filtered_component

    return filtered_field


def analyze_frequency_content(displacement: npt.NDArray) -> Tuple[float, float]:
    """
    Analyze frequency content of displacement field to suggest filter parameters.
    """
    # Calculate power spectrum
    fft_data = fft.fftn(displacement[..., 0])  # Analyze x component
    power = np.abs(fft_data) ** 2

    # Calculate cumulative power distribution
    sorted_power = np.sort(power.flatten())
    cumulative_power = np.cumsum(sorted_power) / sorted_power.sum()

    # Find frequency that captures 90% of power
    cutoff_idx = np.searchsorted(cumulative_power, 0.9)
    suggested_cutoff = cutoff_idx / len(cumulative_power)

    # Suggest filter order based on power distribution steepness
    power_gradient = np.gradient(cumulative_power)
    suggested_order = max(1, min(4, int(np.max(power_gradient) * 5)))

    return suggested_cutoff, suggested_order


def visualize_results(original: npt.NDArray, filtered: npt.NDArray, save_path: Path):
    """
    Create visualization comparing original and filtered displacement fields.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot original magnitude
    orig_mag = np.sqrt(np.sum(original[0] ** 2, axis=-1))
    im1 = axes[0, 0].imshow(orig_mag)
    axes[0, 0].set_title('Original Magnitude (t=0)')
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot filtered magnitude
    filt_mag = np.sqrt(np.sum(filtered[0] ** 2, axis=-1))
    im2 = axes[0, 1].imshow(filt_mag)
    axes[0, 1].set_title('Filtered Magnitude (t=0)')
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot difference
    diff_mag = orig_mag - filt_mag
    im3 = axes[1, 0].imshow(diff_mag)
    axes[1, 0].set_title('Difference (Original - Filtered)')
    plt.colorbar(im3, ax=axes[1, 0])

    # Plot frequency spectrum of filtered data
    fft_filtered = np.abs(fft.fft2(filtered[0, ..., 0]))
    im4 = axes[1, 1].imshow(np.log(fft.fftshift(fft_filtered) + 1))
    axes[1, 1].set_title('Frequency Spectrum (Filtered)')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(save_path / 'filter_comparison.png')
    plt.close()


def process_displacement_field(
        displacement: npt.NDArray,
        cutoff_freq: float = None,
        filter_order: int = None,
        kernel_size: int = 5,
        auto_params: bool = True
) -> npt.NDArray:
    """
    Complete processing pipeline with automatic parameter suggestion.
    """
    if auto_params or cutoff_freq is None or filter_order is None:
        suggested_cutoff, suggested_order = analyze_frequency_content(displacement)
        cutoff_freq = cutoff_freq or suggested_cutoff
        filter_order = filter_order or suggested_order
        print(f"Suggested parameters: cutoff_freq={cutoff_freq:.3f}, order={filter_order}")

    return filter_displacement_field(
        displacement,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        kernel_size=kernel_size
    )


def main():
    # File path
    file_path = Path(r"D:\2025-01-31\sample2\position3\TFM_data\displacements.npy")

    try:
        # Load the data
        print("Loading displacement data...")
        result = np.load(file_path, allow_pickle=True).item()

        # Create visualization directory and saver
        viz_path = file_path.parent / "filter_tests"
        viz_path.mkdir(exist_ok=True, parents=True)
        viz_saver = BatchVisualizationSaver(str(viz_path))

        # Get visualization folder path and clean existing files
        viz_folder = Path(viz_saver.viz_folder)
        for existing_file in viz_folder.glob("*.gif"):
            existing_file.unlink()

        # Save visualization of original data
        print("\nGenerating visualization for original data...")
        viz_saver.save_displacement_visualization(result, fps=10)

        # Find and rename the generated file with proper error handling
        generated_files = list(viz_folder.glob("*.gif"))
        if generated_files:
            original_file = generated_files[0]
            new_name = viz_folder / "displacement_original.gif"
            if new_name.exists():
                new_name.unlink()
            original_file.rename(new_name)
            print(f"Renamed original visualization to: {new_name}")

        # Print basic information about the data
        print(f"\nDisplacement field shape: {result.displacement_field.shape}")
        print(f"Original shape: {result.original_shape}")
        print(f"Physical scale: {result.physical_scale}")

        # Calculate and print some statistics before filtering
        magnitude = np.sqrt(np.sum(result.displacement_field ** 2, axis=-1))
        print("\nBefore filtering:")
        print(f"Mean displacement magnitude: {np.mean(magnitude):.3f} µm")
        print(f"Max displacement magnitude: {np.max(magnitude):.3f} µm")
        print(f"Min displacement magnitude: {np.min(magnitude):.3f} µm")

        # Apply filtering pipeline
        print("\nApplying displacement field filtering...")
        # filtered_field = process_displacement_field(
        #     result.displacement_field,
        #     auto_params=True  # Let the algorithm suggest parameters
        # )
        filtered_field = process_displacement_field(
            result.displacement_field,
            cutoff_freq=0.3,
            filter_order=5,
            kernel_size=10
        )

        # Calculate and print statistics after filtering
        filtered_magnitude = np.sqrt(np.sum(filtered_field ** 2, axis=-1))
        print("\nAfter filtering:")
        print(f"Mean displacement magnitude: {np.mean(filtered_magnitude):.3f} µm")
        print(f"Max displacement magnitude: {np.max(filtered_magnitude):.3f} µm")
        print(f"Min displacement magnitude: {np.min(filtered_magnitude):.3f} µm")

        # Create visualization comparing original and filtered data
        print("\nGenerating comparison visualizations...")
        visualize_results(result.displacement_field, filtered_field, viz_path)

        # Create filtered result object
        filtered_result = DisplacementResult(
            displacement_field=filtered_field,
            original_shape=result.original_shape,
            displacement_field_shape=result.displacement_field_shape,
            parameters=result.parameters,
            physical_scale=result.physical_scale
        )

        # Save filtered data
        output_path = file_path.parent / "displacements_filtered.npy"
        print(f"\nSaving filtered results to {output_path}")
        np.save(output_path, filtered_result, allow_pickle=True)

        # Save visualization of filtered data
        print("\nGenerating visualization for filtered data...")
        viz_saver.save_displacement_visualization(filtered_result, fps=10)

        # Find and rename the filtered visualization file with proper error handling
        generated_files = list(viz_folder.glob("*.gif"))
        if generated_files:
            for gen_file in generated_files:
                if gen_file.name != "displacement_original.gif":
                    new_name = viz_folder / "displacement_filtered.gif"
                    if new_name.exists():
                        new_name.unlink()
                    gen_file.rename(new_name)
                    print(f"Renamed filtered visualization to: {new_name}")
                    break

        print("\nProcessing complete!")
        print(f"Visualizations saved to: {viz_path}")
        print("Generated files:")
        print(f"- Original GIF: {viz_folder / 'displacement_original.gif'}")
        print(f"- Filtered GIF: {viz_folder / 'displacement_filtered.gif'}")
        print(f"- Comparison plot: {viz_folder / 'filter_comparison.png'}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()