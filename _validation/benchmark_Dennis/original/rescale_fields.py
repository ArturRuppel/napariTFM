import numpy as np
from scipy.interpolate import interp1d
import os


def resample_and_pad_field(input_file):
    # Read the original data
    data = np.load(input_file)

    # Replace any NaN values with 0
    data = np.nan_to_num(data, nan=0.0)

    # Get original dimensions
    nx, ny = data.shape

    # Create original coordinate arrays
    x_orig = np.linspace(0, 1, nx)
    y_orig = np.linspace(0, 1, ny)

    # Create interpolation function
    interp_func = interp1d(x_orig, data, axis=0, kind='linear')
    interp_func2 = interp1d(y_orig, interp_func(np.linspace(0, 1, 600)), axis=1, kind='linear')

    # Create upsampled data
    upsampled = interp_func2(np.linspace(0, 1, 600))

    # Create padded array with zeros instead of NaN
    padded = np.zeros((800, 800))

    # Insert upsampled data into center of padded array
    start_idx = 100
    end_idx = start_idx + 600
    padded[start_idx:end_idx, start_idx:end_idx] = upsampled

    return padded


def main():
    # List of input files from the screenshot
    input_files = [
        'displacement_x.npy',
        'displacement_y.npy',
        'stress_xx.npy',
        'stress_yy.npy',
        'traction_x.npy',
        'traction_y.npy'
    ]

    # Create output directory if it doesn't exist
    output_dir = 'processed_fields'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each file
    for input_file in input_files:
        # Skip if file doesn't exist
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue

        # Process the file
        print(f"Processing {input_file}...")
        processed_data = resample_and_pad_field(input_file)

        # Create output filepath
        output_file = os.path.join(output_dir, input_file)

        # Save the processed data
        np.save(output_file, processed_data)
        print(f"Saved processed data to {output_file}")


if __name__ == "__main__":
    main()