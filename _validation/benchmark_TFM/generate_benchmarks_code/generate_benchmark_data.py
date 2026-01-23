"""
Generate complete benchmark dataset: dipole fields and deformed images.

This script combines the functionality of generate_dipole_fields.py and 
create_deformed_images.py to create a complete benchmark dataset including 
displacement fields, traction fields, and deformed images from a single TOML 
configuration file.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.transform import warp

# DirectMethod package imports using absolute path
DIRECTMETHOD_ROOT = Path("/home/aruppel/projects/DirectMethod")
DIRECTMETHOD_SRC = DIRECTMETHOD_ROOT / "src"
sys.path.insert(0, str(DIRECTMETHOD_SRC))

from inputSim.fields.HertzBuilder import get_u_hertz_pattern, get_q_hertz_pattern
from utils.tomlLoad import loadSimulationData, loadDataDescription, loadAdheasionSites
from eval.fieldProperties import getDipoleDirect


def generate_complete_benchmark(toml_file, reference_image_path, pixelsize=0.1):
    """
    Generate complete benchmark dataset from TOML configuration.
    
    Args:
        toml_file: Path to TOML configuration file
        reference_image_path: Path to reference TIFF image
        pixelsize: Pixel size in microns for deformed image generation
    """
    import os
    
    benchmark_dir = DIRECTMETHOD_ROOT / "generate_benchmarks"
    
    # Change to the benchmark directory so the framework can find description.toml
    original_cwd = Path.cwd()
    os.chdir(benchmark_dir)
    
    # Create a symlink or copy the TOML file as description.toml
    description_path = benchmark_dir / "description.toml"
    if description_path.exists():
        description_path.unlink()
    description_path.symlink_to(toml_file)
    
    try:
        # Load configuration
        name, E, nu, spacing_xy, _ = loadDataDescription()
        n_points_xy, _ = loadSimulationData(silent=True)
        n_points_xy = int(n_points_xy)  # Ensure it's an integer
        point_list = loadAdheasionSites(silent=True)
        
        print(f"Generating complete benchmark for dataset: {name}")
        print(f"Substrate E: {E} Pa, nu: {nu}")
        print(f"Grid: {n_points_xy}x{n_points_xy} pixels, spacing: {spacing_xy} µm")
        
        mu = E / (2.0 * (1.0 + nu))
        
        # Generate displacement field functions
        u_func, v_func, w_func = get_u_hertz_pattern(point_list)
        
        # Generate traction field functions (proper analytical calculation)
        qx_func, qy_func, qz_func = get_q_hertz_pattern(point_list)
        
        def displacement_field(x, y, z):
            """Function to return displacement vector at given position"""
            return (u_func(x, y, z, mu, nu), 
                    v_func(x, y, z, mu, nu), 
                    w_func(x, y, z, mu, nu))
        
        def traction_field(x, y):
            """Function to return traction vector at given position"""
            return qx_func(x, y), qy_func(x, y), qz_func(x, y)
        
        # Create coordinate grids
        x_size = n_points_xy * spacing_xy  # µm
        y_size = n_points_xy * spacing_xy  # µm
        
        x = np.linspace(-x_size/2, x_size/2, n_points_xy)
        y = np.linspace(-y_size/2, y_size/2, n_points_xy)
        
        X, Y = np.meshgrid(x, y)
        
        # Generate displacement fields
        print("Computing displacement fields...")
        ux = np.zeros_like(X)
        uy = np.zeros_like(X)
        uz = np.zeros_like(X)
        
        for i in range(n_points_xy):
            for j in range(n_points_xy):
                u_val, v_val, w_val = displacement_field(X[i,j], Y[i,j], 0.0)
                ux[i,j] = u_val
                uy[i,j] = v_val
                uz[i,j] = w_val
        
        # Generate analytical traction fields
        print("Computing traction fields (analytical)...")
        fx = np.zeros_like(X)  # qx - traction in x direction
        fy = np.zeros_like(X)  # qy - traction in y direction
        
        for i in range(n_points_xy):
            for j in range(n_points_xy):
                qx_val, qy_val, _ = traction_field(X[i,j], Y[i,j])
                fx[i,j] = qx_val
                fy[i,j] = qy_val
        
        # Save outputs in the same folder as the input files
        output_path = Path(toml_file).parent
        
        # Save displacement and traction arrays as individual NPY files
        print(f"Saving field arrays to {output_path}/")
        np.save(output_path / 'displacement_x.npy', ux)
        np.save(output_path / 'displacement_y.npy', uy)
        np.save(output_path / 'traction_x.npy', fx)
        np.save(output_path / 'traction_y.npy', fy)
        
        # Generate deformed image
        print("Generating deformed image...")
        if Path(reference_image_path).exists():
            reference = tifffile.imread(reference_image_path)
            
            # Convert displacement from µm to pixels
            d_x = ux / pixelsize
            d_y = uy / pixelsize
            
            nr, nc = reference.shape
            row_coords, col_coords = np.meshgrid(
                np.arange(nr), np.arange(nc), indexing="ij")
            
            # Apply deformation
            coord_map = np.array([row_coords - d_y, col_coords - d_x])
            
            # Normalize reference image to [0, 1] for warping
            reference_normalized = reference.astype(np.float64) / np.iinfo(reference.dtype).max
            deformed_normalized = warp(reference_normalized, coord_map, mode="constant")
            
            # Convert back to original data type range
            deformed = (deformed_normalized * np.iinfo(np.uint16).max).astype("uint16")
            
            # Save deformed image
            tifffile.imwrite(output_path / 'deformed.tif', deformed)
            
            # Copy reference image to output directory for completeness
            tifffile.imwrite(output_path / 'reference.tif', reference)
            
            print(f"Deformed image saved to {output_path}/deformed.tif")
        else:
            print(f"Warning: Reference image not found at {reference_image_path}")
        
        # Calculate and print dipole moments for validation
        print("\nDipole analysis:")
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        for i, adhesion in enumerate(point_list):
            if hasattr(adhesion, 'name'):
                getDipoleDirect(ux, uy, dx, dy)
                print(f"Adhesion {adhesion.name}: Expected force ~{adhesion.F} µN")
        
        print("Complete benchmark generation completed!")
        
        return ux, uy, uz, fx, fy, X, Y
        
    finally:
        # Cleanup and restore working directory
        if description_path.exists():
            description_path.unlink()
        os.chdir(original_cwd)


def main():
    """Main function to generate complete benchmark dataset."""
    
    print("Generating complete benchmark dataset using DirectMethod framework...")
    
    # Default paths - can be modified as needed
    base_folder = "/home/aruppel/projects/napariTFM/_validation/benchmark_TFM/"
    toml_file = base_folder + "low/dipole_config.toml"
    reference_image = base_folder + "low/reference.tif"
    
    # Check if TOML config exists
    if not Path(toml_file).exists():
        print(f"Error: {toml_file} not found")
        print("Please create a TOML configuration file.")
        return
    
    # Check if reference image exists
    if not Path(reference_image).exists():
        print(f"Error: Reference image not found at {reference_image}")
        print("Please provide a reference TIFF image.")
        return
    
    try:
        generate_complete_benchmark(toml_file, reference_image, pixelsize=0.1)
    except Exception as e:
        print(f"Error generating benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()