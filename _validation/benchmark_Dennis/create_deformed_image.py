import tifffile
import numpy as np
from skimage.transform import warp

pixelsize = 0.1

reference = tifffile.imread("reference.tif")
d_x = np.load("displacement_x.npy") / pixelsize
d_y = np.load("displacement_y.npy") / pixelsize
nr, nc = reference.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

deformed_mid = 2 ** 16 * warp(reference, np.array([row_coords - d_y, col_coords - d_x]), mode="constant")
deformed_mid = deformed_mid.astype("uint16")
tifffile.imwrite("deformed.tif", deformed_mid)

# deformed_low = 2 ** 16 * warp(reference, np.array([row_coords - 0.1 * d_y, col_coords - 0.1 * d_x]), mode="constant")
# deformed_low = deformed_low.astype("uint16")
# tifffile.imwrite("deformed_low.tif", deformed_low)

# deformed_high = 2 ** 16 * warp(reference, np.array([row_coords - 10 * d_y, col_coords - 10 * d_x]), mode="constant")
# deformed_high = deformed_high.astype("uint16")
# tifffile.imwrite("deformed_high.tif", deformed_high)
