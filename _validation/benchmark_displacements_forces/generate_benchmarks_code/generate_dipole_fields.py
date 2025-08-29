"""
Generate displacement and traction force fields for dipoles using the existing DirectMethod framework.

This script is a wrapper around the DirectMethod inputSim framework that generates
theoretical displacement and traction fields for dipoles using the proper 3D elastic
Green's functions and Hertz-like adhesion models.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# DirectMethod package imports using absolute path
DIRECTMETHOD_ROOT = Path("/home/artur/DirectMethod")
DIRECTMETHOD_SRC = DIRECTMETHOD_ROOT / "src"
sys.path.insert(0, str(DIRECTMETHOD_SRC))

from inputSim.fields.HertzBuilder import get_u_hertz_pattern, get_q_hertz_pattern
from utils.tomlLoad import loadSimulationData, loadDataDescription, loadAdheasionSites
from eval.fieldProperties import getDipoleDirect


def generate_fields_from_toml(toml_file='dipole_config.toml', output_dir='output'):
    """
    Generate dipole fields using the DirectMethod framework with TOML configuration.
    """
    benchmark_dir = DIRECTMETHOD_ROOT / "generate_benchmarks"
    
    # Change to the benchmark directory so the framework can find description.toml
    original_cwd = os.getcwd()
    os.chdir(benchmark_dir)
    
    # Create a symlink or copy the TOML file as description.toml
    description_path = benchmark_dir / "description.toml"
    if description_path.exists():
        description_path.unlink()
    description_path.symlink_to(toml_file)
    
    try:
        # Load configuration
        name, E, nu, spacing_xy, spacing_z = loadDataDescription()
        n_points_xy, n_points_z = loadSimulationData(silent=True)
        n_points_xy = int(n_points_xy)  # Ensure it's an integer
        pointList = loadAdheasionSites(silent=True)
        
        print(f"Generating dipole fields for dataset: {name}")
        print(f"Substrate E: {E} Pa, nu: {nu}")
        print(f"Grid: {n_points_xy}x{n_points_xy} pixels, spacing: {spacing_xy} µm")
        
        mu = E / (2.0 * (1.0 + nu))
        
        # Generate displacement field functions
        u_func, v_func, w_func = get_u_hertz_pattern(pointList)
        
        # Generate traction field functions (proper analytical calculation)
        qx_func, qy_func, qz_func = get_q_hertz_pattern(pointList)
        
        def displacement_field(x, y, z):
            """Function to return displacement vector at given position"""
            return u_func(x, y, z, mu, nu), v_func(x, y, z, mu, nu), w_func(x, y, z, mu, nu)
        
        def traction_field(x, y):
            """Function to return traction vector at given position"""
            return qx_func(x, y), qy_func(x, y), qz_func(x, y)
        
        # Create coordinate grids
        x_size = n_points_xy * spacing_xy  # µm
        y_size = n_points_xy * spacing_xy  # µm
        
        x = np.linspace(-x_size/2, x_size/2, n_points_xy)
        y = np.linspace(-y_size/2, y_size/2, n_points_xy)
        z = np.array([0.0])  # Surface
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
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
        fz = np.zeros_like(X)  # qz - traction in z direction
        
        for i in range(n_points_xy):
            for j in range(n_points_xy):
                qx_val, qy_val, qz_val = traction_field(X[i,j], Y[i,j])
                fx[i,j] = qx_val
                fy[i,j] = qy_val
                fz[i,j] = qz_val
        
        # Create output directory
        output_path = benchmark_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        # Save arrays as individual NPY files
        print(f"Saving arrays to {output_path}/")
        np.save(output_path / 'displacement_x.npy', ux)
        np.save(output_path / 'displacement_y.npy', uy)
        np.save(output_path / 'traction_x.npy', fx)
        np.save(output_path / 'traction_y.npy', fy)
        
        # # Generate plots
        # print(f"Saving plots to {output_path}/")
        # generate_plots(X, Y, ux, uy, fx, fy, output_path)
        
        # Calculate and print dipole moments for validation
        print("\nDipole analysis:")
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        for i, adhesion in enumerate(pointList):
            if hasattr(adhesion, 'name'):
                dipole_moment = getDipoleDirect(ux, uy, dx, dy)
                print(f"Adhesion {adhesion.name}: Expected force ~{adhesion.F} µN")
        
        print("Field generation completed!")
        
        return ux, uy, uz, fx, fy, X, Y
        
    finally:
        # Cleanup and restore working directory
        if description_path.exists():
            description_path.unlink()
        os.chdir(original_cwd)


def generate_plots(X, Y, ux, uy, fx, fy, output_dir):
    """Generate and save visualization plots."""
    
    # Displacement field plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # X displacement
    im1 = ax1.imshow(ux, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                     origin='lower', cmap='RdBu_r')
    ax1.set_title('X Displacement (ux)')
    ax1.set_xlabel('X (µm)')
    ax1.set_ylabel('Y (µm)')
    plt.colorbar(im1, ax=ax1, label='µm')
    
    # Y displacement
    im2 = ax2.imshow(uy, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                     origin='lower', cmap='RdBu_r')
    ax2.set_title('Y Displacement (uy)')
    ax2.set_xlabel('X (µm)')
    ax2.set_ylabel('Y (µm)')
    plt.colorbar(im2, ax=ax2, label='µm')
    
    # X traction
    im3 = ax3.imshow(fx, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                     origin='lower', cmap='RdBu_r')
    ax3.set_title('X Traction (fx)')
    ax3.set_xlabel('X (µm)')
    ax3.set_ylabel('Y (µm)')
    plt.colorbar(im3, ax=ax3, label='Pa')
    
    # Y traction
    im4 = ax4.imshow(fy, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                     origin='lower', cmap='RdBu_r')
    ax4.set_title('Y Traction (fy)')
    ax4.set_xlabel('X (µm)')
    ax4.set_ylabel('Y (µm)')
    plt.colorbar(im4, ax=ax4, label='Pa')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dipole_fields.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Displacement vector field plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Subsample for vector plot
    skip = 8
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    ux_sub = ux[::skip, ::skip]
    uy_sub = uy[::skip, ::skip]
    
    magnitude = np.sqrt(ux**2 + uy**2)
    im = ax.imshow(magnitude, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                   origin='lower', cmap='viridis', alpha=0.7)
    ax.quiver(X_sub, Y_sub, ux_sub, uy_sub, angles='xy', scale_units='xy', 
              scale=np.max(magnitude)/20, color='white', width=0.003)
    
    ax.set_title('Displacement Field Magnitude and Vectors')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    plt.colorbar(im, ax=ax, label='Displacement magnitude (µm)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dipole_displacement_vectors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Force vector field plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Subsample for vector plot
    fx_sub = fx[::skip, ::skip]
    fy_sub = fy[::skip, ::skip]
    
    force_magnitude = np.sqrt(fx**2 + fy**2)
    im = ax.imshow(force_magnitude, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                   origin='lower', cmap='plasma', alpha=0.7)
    ax.quiver(X_sub, Y_sub, fx_sub, fy_sub, angles='xy', scale_units='xy', 
              scale=np.max(force_magnitude)/15, color='white', width=0.003)
    
    ax.set_title('Traction Force Field Magnitude and Vectors')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    plt.colorbar(im, ax=ax, label='Force magnitude (Pa)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dipole_force_vectors.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to generate dipole fields."""
    benchmark_dir = DIRECTMETHOD_ROOT / "generate_benchmarks"
    
    print("Generating dipole fields using DirectMethod framework...")
    
    # Check if TOML config exists
    toml_file = 'dipole_config.toml'
    if not (benchmark_dir / toml_file).exists():
        print(f"Error: {toml_file} not found in {benchmark_dir}")
        print("Please create a TOML configuration file.")
        return
    
    try:
        generate_fields_from_toml(toml_file)
    except Exception as e:
        print(f"Error generating fields: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()