import numpy as np
from scipy.ndimage import sobel


def calculate_traction_from_stress(stress_tensor, mask, pixelsize):
    """
    Calculate traction forces directly from stress tensor using t = σ⋅n

    Args:
        stress_tensor: 4D array (height, width, 2, 2) containing stress components in Pa
        mask: Boolean mask of valid area
        pixelsize: Pixel size in microns

    Returns:
        tuple: (t_x, t_y) traction force components in Pa
    """
    # Convert pixelsize to meters
    pixelsize = pixelsize * 1e-6

    # Extract stress components
    sigma_xx = stress_tensor[:, :, 0, 0]
    sigma_yy = stress_tensor[:, :, 1, 1]
    sigma_xy = stress_tensor[:, :, 0, 1]

    # Calculate gradients of mask to get normal vectors
    ny = sobel(mask.astype(float), axis=0)
    nx = sobel(mask.astype(float), axis=1)

    # Normalize the normal vectors
    norm = np.sqrt(nx ** 2 + ny ** 2)
    mask_nonzero = (norm > 0)
    nx[mask_nonzero] = nx[mask_nonzero] / norm[mask_nonzero]
    ny[mask_nonzero] = ny[mask_nonzero] / norm[mask_nonzero]

    # Calculate traction components using t = σ⋅n
    t_x = sigma_xx * nx + sigma_xy * ny
    t_y = sigma_xy * nx + sigma_yy * ny

    # Convert to force per unit area
    t_x = t_x / (pixelsize ** 2)
    t_y = t_y / (pixelsize ** 2)

    # Zero out values outside mask
    t_x[~mask] = 0
    t_y[~mask] = 0

    return t_x, t_y


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load your stress tensor and mask data
    # stress_tensor = ...
    # mask = ...

    # Calculate traction forces
    pixelsize = 0.8  # microns per pixel
    t_x, t_y = calculate_traction_from_stress(stress_tensor, mask, pixelsize)

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot traction magnitude
    traction_mag = np.sqrt(t_x ** 2 + t_y ** 2)
    im1 = ax1.imshow(traction_mag)
    ax1.set_title('Traction Magnitude')
    plt.colorbar(im1, ax=ax1, label='Traction (Pa)')

    # Plot traction vectors (downsample for clarity)
    spacing = 10
    y, x = np.mgrid[:t_x.shape[0]:spacing, :t_x.shape[1]:spacing]
    ax2.quiver(x, y,
               t_x[::spacing, ::spacing],
               t_y[::spacing, ::spacing])
    ax2.set_title('Traction Vectors')

    plt.tight_layout()
    plt.show()