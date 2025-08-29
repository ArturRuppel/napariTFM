import tifffile
import numpy as np
from skimage.transform import warp

pixelsize = 0.1

base_folder = "/home/artur/napariTFM/_validation/benchmark_displacements_forces"

reference = tifffile.imread(base_folder + "reference.tif")
d_x = np.load(base_folder + "output/displacement_x.npy") / pixelsize
d_y = np.load(base_folder + "output/displacement_y.npy") / pixelsize
nr, nc = reference.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

deformed = 2 ** 16 * warp(reference, np.array([row_coords - d_y, col_coords - d_x]), mode="constant")
deformed = deformed.astype("uint16")
tifffile.imwrite(base_folder + "deformed.tif", deformed)

