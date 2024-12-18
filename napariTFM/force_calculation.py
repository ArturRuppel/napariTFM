# Functions copied from pyTFM.TFM_functions module
# Original source: https://github.com/fabrylab/pyTFM
# License: GNU General Public License v3.0

# The following code is derived from the pyTFM library (https://github.com/fabrylab/pyTFM)
# and adapted for use in the napariTFM project (https://github.com/ArturRuppel/napariTFM).
#
# The original pyTFM library has a dependency on openpiv, which relies on an older version
# of NumPy that is incompatible with napariTFM. However, the specific functionality required
# by napariTFM does not depend on openpiv. Therefore, the relevant parts of the code have been
# copied and adapted here to avoid the dependency conflict.
#
# Both pyTFM and napariTFM are licensed under the GNU General Public License v3.0.
# The original code is Copyright (c) 2021 fabrylab.

# basic functions for traction force microscopy
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyTFM.utilities_TFM import suppress_warnings
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.ndimage.filters import uniform_filter
from scipy.fft import fft2, ifft2


def calculate_traction_stresses(u, v, E, nu, pixelsize, alpha):
    '''This function takes a displacement field u and v, a gel rigidity E, it's poisson ratio nu, the size of a pixel in '''
    M, N = u.shape

    # pad displacement map with zeros until it has a shape of 2^n by 2^n because fourier transform is faster
    n = 2
    while (2 ** n < M) or (2 ** n < N):
        n = n + 1

    M2 = 2 ** n
    N2 = M2

    u_padded = np.zeros((M2, N2))
    v_padded = np.zeros((M2, N2))

    u_padded[:u.shape[0], :u.shape[1]] = u
    v_padded[:v.shape[0], :v.shape[1]] = v

    u_fft = fft2(u_padded)
    v_fft = fft2(v_padded)

    # remove component related to translation
    u_fft[0, 0] = 0
    v_fft[0, 0] = 0

    Kx1 = (2 * np.pi / pixelsize) / N2 * np.arange(int(N2 / 2))
    Kx2 = -(2 * np.pi / pixelsize) / N2 * (N2 - np.arange(int(N2 / 2), N2))
    Ky1 = (2 * np.pi / pixelsize) / M2 * np.arange(int(M2 / 2))
    Ky2 = -(2 * np.pi / pixelsize) / M2 * (M2 - np.arange(int(M2 / 2), M2))

    Kx = np.concatenate((Kx1, Kx2))
    Ky = np.concatenate((Ky1, Ky2))

    kx, ky = np.meshgrid(Kx, Ky)
    k = np.sqrt(kx ** 2 + ky ** 2)
    t_xt = np.zeros((M2, N2), dtype=complex)
    t_yt = np.zeros((M2, N2), dtype=complex)

    for i in np.arange(M2):
        for j in np.arange(N2):
            if i == M2 / 2 or j == N2 / 2:  # Nyquist frequency
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

            elif ~((i == 1) and (j == 1)):
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)
                Gt[0, 1] = - nu * kx[i, j] * ky[i, j]
                Gt[1, 0] = - nu * kx[i, j] * ky[i, j]

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

    t_x = ifft2(t_xt)
    t_y = ifft2(t_yt)
    traction_x = np.real(t_x)
    traction_y = np.real(t_y)

    return traction_x[0:M, 0:N], traction_y[0:M, 0:N]

def ffttc_traction(u, v, pixelsize1, pixelsize2, young, sigma=0.49, spatial_filter="gaussian", fs=None):
    """
    fourier transform based calculation of the traction force. U and v must be given  as deformations in pixel. Size of
    these pixels must be the pixelsize (size of a pixel in the deformation field u or v). Note that thePiv deformation
    returns deformation in pixel of the size of pixels in the images of beads before and after.
    If bf_image is provided this script will return a traction field that is zoomed to the size
     of the bright field image, by interpolation. It is not recommended to use this for any calculations.
    The function can use different filters. Recommended filter is gaussian. Mean filter should yield similar results.

    :param u:deformation field in x direction in pixel of the deformation image
    :param v:deformation field in y direction in pixel of the deformation image
    :param young: Young's modulus in Pa
    :param pixelsize1: pixelsize in m/pixel of the original image, needed because u and v is given as
    displacement of these pixels
    :param pixelsize2: pixelsize of m/pixel the deformation image
    :param sigma: poisson ratio of the gel
    :param spatial_filter: str, values: "mean","gaussian","median". Different smoothing methods for the traction field
    :return: tx_filter,ty_filter: traction forces in x and y direction in Pa
    """

    # 0) subtracting mean(better name for this step)
    u_shift = (u - np.mean(u))  # shifting to zero mean  (translating to pixelsize of u-image is done later)
    v_shift = (v - np.mean(v))

    # Ben's algorithm:
    # 1)Zero padding to get square array with even index number
    ax1_length = np.shape(u_shift)[0]  # u and v must have same dimensions
    ax2_length = np.shape(u_shift)[1]
    max_ind = int(np.max((ax1_length, ax2_length)))
    if max_ind % 2 != 0:
        max_ind += 1

    u_expand = np.zeros((max_ind, max_ind))
    v_expand = np.zeros((max_ind, max_ind))
    u_expand[:ax1_length, :ax2_length] = u_shift
    v_expand[:ax1_length, :ax2_length] = v_shift

    # 2) producing wave vectors
    # form 1:max_ind/2 then -(max_ind/2:1)
    kx1 = np.array([list(range(0, int(max_ind / 2), 1)), ] * int(max_ind))
    kx2 = np.array([list(range(-int(max_ind / 2), 0, 1)), ] * int(max_ind))
    kx = np.append(kx1, kx2, axis=1) * 2 * np.pi  # fourier transform in this case is defined as
    # F(kx)=1/2pi integral(exp(i*kx*x)dk therefore kx must be expressed as a spatial frequency in distance*2*pi
    ky = np.transpose(kx)
    k = np.sqrt(kx ** 2 + ky ** 2) / (pixelsize2 * max_ind)

    # 2.1) calculating angle between k and kx with atan2 function
    alpha = np.arctan2(ky, kx)
    alpha[0, 0] = np.pi / 2

    # 3) calculation of K --> Tensor to calculate displacements from tractions. We calculate inverse of K
    # K⁻¹=[[kix kid],
    #     [kid,kiy]]  ,,, so is "diagonal, kid appears two times
    kix = ((k * young) / (2 * (1 - sigma ** 2))) * (1 - sigma + sigma * np.cos(alpha) ** 2)
    kiy = ((k * young) / (2 * (1 - sigma ** 2))) * (1 - sigma + sigma * np.sin(alpha) ** 2)
    kid = ((k * young) / (2 * (1 - sigma ** 2))) * (sigma * np.sin(alpha) * np.cos(alpha))

    # adding zeros in kid diagonals
    kid[:, int(max_ind / 2)] = np.zeros(max_ind)
    kid[int(max_ind / 2), :] = np.zeros(max_ind)

    # 4) calculate Fourier transform of displacement
    # u_ft=np.fft.fft2(u_expand*pixelsize1*2*np.pi)
    # v_ft=np.fft.fft2(v_expand*pixelsize1*2*np.pi)
    u_ft = scipy.fft.fft2(u_expand * pixelsize1)
    v_ft = scipy.fft.fft2(v_expand * pixelsize1)

    # 4.1) calculate tractions in Fourier space T=K⁻¹*U, U=[u,v] here with individual matrix elements..
    tx_ft = kix * u_ft + kid * v_ft
    ty_ft = kid * u_ft + kiy * v_ft

    # 4.2) go back to real space
    tx = scipy.fft.ifft2(tx_ft).real
    ty = scipy.fft.ifft2(ty_ft).real

    # 5.2) cut back to original shape
    tx_cut = tx[0:ax1_length, 0:ax2_length]
    ty_cut = ty[0:ax1_length, 0:ax2_length]

    # 5.3) using filter
    tx_filter = tx_cut
    ty_filter = ty_cut

    if spatial_filter == "mean":
        fs = fs if isinstance(fs, (float, int)) else int(int(np.max((ax1_length, ax2_length))) / 16)
        tx_filter = uniform_filter(tx_cut, size=fs)
        ty_filter = uniform_filter(ty_cut, size=fs)
    if spatial_filter == "gaussian":
        fs = fs if isinstance(fs, (float, int)) else int(np.max((ax1_length, ax2_length))) / 50
        tx_filter = gaussian_filter(tx_cut, sigma=fs)
        ty_filter = gaussian_filter(ty_cut, sigma=fs)
    if spatial_filter == "median":
        fs = fs if isinstance(fs, (float, int)) else int(int(np.max((ax1_length, ax2_length))) / 16)
        tx_filter = median_filter(tx_cut, size=fs)
        ty_filter = median_filter(ty_cut, size=fs)

    return tx_filter, ty_filter


def ffttc_traction_pure_shear(u, v, pixelsize1, pixelsize2, h, young, sigma=0.49, spatial_filter="mean", fs=None):
    """
    limiting case for h*k==0
    Xavier Trepat, Physical forces during collective cell migration, 2009

    :param u:deformation field in x direction in pixel of the deformation image
    :param v:deformation field in y direction in pixel of the deformation image
    :param young: Young's modulus in Pa
    :param pixelsize1: pixelsize of the original image, needed because u and v is given as displacement of these pixels
    :param pixelsize2: pixelsize of the deformation image
    :param h: height of the membrane the cells lie on, in µm
    :param sigma: Poisson's ratio of the gel
    :param spatial_filter: str, values: "mean","gaussian","median". Different smoothing methods for the traction field.
    :return: tx_filter,ty_filter: traction forces in x and y direction in Pa
    """

    # 0) subtracting mean(better name for this step)
    u_shift = (u - np.mean(u)) * pixelsize1
    v_shift = (v - np.mean(v)) * pixelsize1

    # Ben's algorithm:
    # 1)Zero padding to get square array with even index number
    ax1_length = np.shape(u_shift)[0]  # u and v must have same dimensions
    ax2_length = np.shape(u_shift)[1]
    max_ind = int(np.max((ax1_length, ax2_length)))
    if max_ind % 2 != 0:
        max_ind += 1
    u_expand = np.zeros((max_ind, max_ind))
    v_expand = np.zeros((max_ind, max_ind))
    u_expand[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind] = u_shift
    v_expand[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind] = v_shift

    u_ft = scipy.fft.fft2(u_expand)
    v_ft = scipy.fft.fft2(v_expand)

    # 4.1) calculate tractions in Fourier space
    mu = young / (2 * (1 + sigma))
    tx_ft = mu * u_ft / h
    tx_ft[0, 0] = 0  # zero frequency would represent force everywhere (constant)
    ty_ft = mu * v_ft / h
    ty_ft[0, 0] = 0

    # 4.2) go back to real space
    tx = scipy.fft.ifft2(tx_ft).real
    ty = scipy.fft.ifft2(ty_ft).real

    # 5.2) cut like in script from ben

    tx_cut = tx[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind]
    ty_cut = ty[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind]

    # 5.3) using filter
    tx_filter = tx_cut
    ty_filter = ty_cut
    if spatial_filter == "mean":
        fs = fs if isinstance(fs, (float, int)) else int(int(np.max((ax1_length, ax2_length))) / 16)
        tx_filter = uniform_filter(tx_cut, size=fs)
        ty_filter = uniform_filter(ty_cut, size=fs)
    if spatial_filter == "gaussian":
        fs = fs if isinstance(fs, (float, int)) else int(np.max((ax1_length, ax2_length))) / 50
        tx_filter = gaussian_filter(tx_cut, sigma=fs)
        ty_filter = gaussian_filter(ty_cut, sigma=fs)
    if spatial_filter == "median":
        fs = fs if isinstance(fs, (float, int)) else int(int(np.max((ax1_length, ax2_length))) / 16)
        tx_filter = median_filter(tx_cut, size=fs)
        ty_filter = median_filter(ty_cut, size=fs)

    return tx_filter, ty_filter


def ffttc_traction_finite_thickness(u, v, pixelsize1, pixelsize2, h, young, sigma=0.49, spatial_filter="gaussian", fs=None):
    """
    FTTC with correction for finite substrate thickness according to
    Xavier Trepat, Physical forces during collective cell migration, 2009

    :param u:deformation field in x direction in pixel of the deformation image
    :param v:deformation field in y direction in pixel of the deformation image
    :param young: Young's modulus in Pa
    :param pixelsize1: pixelsize of the original image, needed because u and v is given as displacement of these pixels
    :param pixelsize2: pixelsize of the deformation image
    :param h: height of the membrane the cells lie on, in µm
    :param sigma: Poisson's ratio of the gel
    :param spatial_filter: str, values: "mean","gaussian","median". Different smoothing methods for the traction field.
    :param fs: float, size of the filter (std of gaussian or size of the filter window) in µm
    :return: tx_filter,ty_filter: traction forces in x and y direction in Pa
    """

    # 0) subtracting mean(better name for this step)
    u_shift = (u - np.mean(u)) * pixelsize1
    v_shift = (v - np.mean(v)) * pixelsize1

    # Ben's algortithm:
    # 1)Zero padding to get square array with even index number
    ax1_length = np.shape(u_shift)[0]  # u and v must have same dimensions
    ax2_length = np.shape(u_shift)[1]
    max_ind = int(np.max((ax1_length, ax2_length)))
    if max_ind % 2 != 0:
        max_ind += 1
    u_expand = np.zeros((max_ind, max_ind))
    v_expand = np.zeros((max_ind, max_ind))
    u_expand[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind] = u_shift
    v_expand[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind] = v_shift

    # 2) producing wave vectors (FT-space)
    # form 1:max_ind/2 then -(max_ind/2:1)
    kx1 = np.array([list(range(0, int(max_ind / 2), 1)), ] * int(max_ind), dtype=float)
    kx2 = np.array([list(range(-int(max_ind / 2), 0, 1)), ] * int(max_ind), dtype=float)
    # spatial frequencies: 1/wavelength,in 1/µm in fractions of total length

    kx = np.append(kx1, kx2, axis=1) * 2 * np.pi / (pixelsize2 * max_ind)
    ky = np.transpose(kx)
    k = np.sqrt(kx ** 2 + ky ** 2)  # matrix with "relative" distances??#

    r = k * h
    c = np.cosh(r)
    s = np.sinh(r)
    s_c = np.tanh(r)

    # gamma = ((3 - 4 * sigma) * (c ** 2) + (1 - 2 * sigma) ** 2 + (k * h) ** 2) / (
    #         (3 - 4 * sigma) * s * c + k * h)  ## inf values here because k goes to zero
    gamma = ((3 - 4 * sigma) + (((1 - 2 * sigma) ** 2) / (c ** 2)) + ((r ** 2) / (c ** 2))) / (
            (3 - 4 * sigma) * s_c + r / (c ** 2))

    # 4) calculate fourier transform of displacements
    u_ft = scipy.fft.fft2(u_expand)
    v_ft = scipy.fft.fft2(v_expand)

    """
    #4.0*) approximation for large h according to this paper
    factor3=young/(2*(1-sigma**2)*k)
    factor3[0,0]=factor3[0,1]
    tx_ft=factor3*(u_ft*((k**2)*(1-sigma)+sigma*(kx**2)) + v_ft*kx*ky*sigma)
    ty_ft=factor3*(v_ft*((k**2)*(1-sigma)+sigma*(ky**2)) + u_ft*kx*ky*sigma)
    """

    # 4.1) calculate tractions in Fourier space
    factor1 = (v_ft * kx - u_ft * ky)
    factor2 = (u_ft * kx + v_ft * ky)
    tx_ft = ((-young * ky * c) / (2 * k * s * (1 + sigma))) * factor1 + (
            (young * kx) / (2 * k * (1 - sigma ** 2))) * gamma * factor2
    tx_ft[0, 0] = 0  # zero frequency would represent force everywhere (constant)
    ty_ft = ((young * kx * c) / (2 * k * s * (1 + sigma))) * factor1 + (
            (young * ky) / (2 * k * (1 - sigma ** 2))) * gamma * factor2
    ty_ft[0, 0] = 0

    # 4.2) go back to real space
    tx = scipy.fft.ifft2(tx_ft.astype(np.complex128)).real
    ty = scipy.fft.ifft2(ty_ft.astype(np.complex128)).real

    # 5.2) cut like in script from ben
    tx_cut = tx[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind]
    ty_cut = ty[max_ind - ax1_length:max_ind, max_ind - ax2_length:max_ind]

    # 5.3) using filter
    tx_filter = tx_cut
    ty_filter = ty_cut
    if spatial_filter == "mean":
        fs = fs if isinstance(fs, (float, int)) else int(int(np.max((ax1_length, ax2_length))) / 16)
        tx_filter = uniform_filter(tx_cut, size=fs)
        ty_filter = uniform_filter(ty_cut, size=fs)
    if spatial_filter == "gaussian":
        fs = fs if isinstance(fs, (float, int)) else int(np.max((ax1_length, ax2_length))) / 50
        tx_filter = gaussian_filter(tx_cut, sigma=fs)
        ty_filter = gaussian_filter(ty_cut, sigma=fs)
    if spatial_filter == "median":
        fs = fs if isinstance(fs, (float, int)) else int(int(np.max((ax1_length, ax2_length))) / 16)
        tx_filter = median_filter(tx_cut, size=fs)
        ty_filter = median_filter(ty_cut, size=fs)

    return tx_filter, ty_filter


def TFM_tractions(u, v, pixelsize1, pixelsize2, h, young, sigma=0.49, spatial_filter="gaussian", fs=6):
    """
    height correction breaks down due to numerical reasons at large gel height and small wavelengths of deformations.
    In this case the height corrected ffttc-function returns Nans. THis function falls back
     to the non height-corrected FTTC function if this happens
    :return:
    """
    # translate the filter size to pixels of traction field
    fs = fs / pixelsize2 if isinstance(fs, (int, float)) else None
    if isinstance(h, (int, float)):
        with suppress_warnings(RuntimeWarning):
            tx, ty = ffttc_traction_finite_thickness(u, v, pixelsize1=pixelsize1, pixelsize2=pixelsize2,
                                                     h=h, young=young, sigma=sigma,
                                                     spatial_filter=spatial_filter, fs=fs)  # unit is N/m**2
            if np.any(np.isnan(tx)) or np.any(np.isnan(ty)):
                tx, ty = ffttc_traction(u, v, pixelsize1=pixelsize1, pixelsize2=pixelsize2, young=young, sigma=sigma,
                                        spatial_filter=spatial_filter, fs=fs)

    elif h == "infinite":
        tx, ty = ffttc_traction(u, v, pixelsize1=pixelsize1, pixelsize2=pixelsize2, young=young, sigma=sigma,
                                spatial_filter=spatial_filter, fs=fs)
    else:
        raise ValueError("illegal value for h")
    return tx, ty


def get_xy_for_quiver(u):
    """
    accessory function to calculate grid for plt.quiver. Size of the array will correspond to input u.
    :param u:any array,
    :return:
    """
    xs = np.zeros(np.shape(u))
    for i in range(np.shape(u)[0]):
        xs[i, :] = np.arange(0, np.shape(u)[1], 1)
    ys = np.zeros(np.shape(u))
    for j in range(np.shape(u)[1]):  # is inverted in other skript
        ys[:, j] = np.arange(0, np.shape(u)[0], 1)
    return xs, ys


def contractillity(tx, ty, pixelsize, mask):

    """
    Calculation of contractile force and force epicenter.Contractile force is the sum of all projection of traction
    forces (in N) towards the force epicenter. The force epicenter is the point that maximizes the contractile force.
    :param tx: traction forces in x direction in Pa
    :param ty: traction forces in y direction in Pa
    :param pixelsize: pixelsize of the traction field
    :param mask: mask of which values to use for calculation
    :return: contractile_force,contractile force in N
             proj_x, projection of traction forces towards the force epicenter, x component
             proj_y, projection of traction forces towards the force epicenter, y component
             center, coordinates of the force epicenter
    """

    mask = mask.astype(bool)
    tx_filter = np.zeros(np.shape(tx))
    tx_filter[mask] = tx[mask]

    ty_filter = np.zeros(np.shape(ty))
    ty_filter[mask] = ty[mask]

    tract_abs = np.sqrt(tx_filter ** 2 + ty_filter ** 2)

    area = (pixelsize * (10 ** -6)) ** 2  # in meter
    fx = tx_filter * area  # calculating forces (in Newton) by multiplying with area
    fy = ty_filter * area

    x, y = get_xy_for_quiver(tx)
    bx = np.sum(x * (tract_abs ** 2) + fx * (tx_filter * fx + ty_filter * fy))
    by = np.sum(y * (tract_abs ** 2) + fy * (tx_filter * fx + ty_filter * fy))

    axx = np.sum(tract_abs ** 2 + fx ** 2)
    axy = np.sum(fx * fy)
    ayy = np.sum(tract_abs ** 2 + fy ** 2)
    # ayx=np.sum(tx*ty)

    A = np.array([[axx, axy], [axy, ayy]])
    b = np.array([bx, by]).T

    # solve equation system:
    # center*[bx,by]=[[axx,axy],
    #                [axy,ayy]]
    # given by A*[[1/bx],
    #             [1/by]
    center = np.matmul(np.linalg.inv(A), b)

    # vector projection to origin

    dist_x = center[0] - x
    dist_y = center[1] - y
    dist_abs = np.sqrt(dist_y ** 2 + dist_x ** 2)
    proj_abs = (fx * dist_x + fy * dist_y) / dist_abs
    contractile_force = np.nansum(proj_abs)

    # project_vectors
    proj_x = proj_abs * dist_x / dist_abs
    proj_y = proj_abs * dist_y / dist_abs

    return contractile_force, proj_x, proj_y, center  # unit of contractile force is N


def strain_energy_points(u, v, tx, ty, pixelsize1, pixelsize2):
    pixelsize1 *= 10 ** -6
    pixelsize2 *= 10 ** -6  # conversion to m
    # u is given in pixels/minutes where a pixel is from the original image (pixelsize1)
    # tx is given in forces/pixels**2 where a pixel is from the deformation/traction field (pixelsize2)
    energy_points = 0.5 * (pixelsize2 ** 2) * (tx * u * pixelsize1 + ty * v * pixelsize1)
    # value of a background point
    bg = np.percentile(energy_points, 20)
    energy_points -= bg
    return energy_points


def contractile_projection(proj_x, proj_y, tx, ty, mask, center, contractile_force):
    """
    plotting the projection of traction forces towards the force epicenter
    :param proj_x:
    :param proj_y:
    :param tx:
    :param ty:
    :param mask:
    :param center:
    :param contractile_force:
    :return:
    """

    fig = plt.figure()
    custom_cmap1 = LinearSegmentedColormap.from_list("", ["#DBDC3E", "yellow"])
    im = plt.imshow(np.sqrt((tx / 1000) ** 2 + (ty / 1000) ** 2), vmin=0)
    mask_show = np.zeros(np.shape(mask)) + np.nan
    mask_show[mask] = 1
    plt.imshow(mask_show, alpha=0.5, cmap=custom_cmap1)
    plt.plot(center[0], center[1], "or", color="red")

    x1, y1 = get_xy_for_quiver(tx)
    ratio = 0.2  # ratio of length of the biggest arrow to max axis length
    scale = ratio * np.max(np.shape(proj_x)) / np.max(
        np.sqrt(proj_x ** 2 + proj_y ** 2))  # automatic scaling in dependency of the image size
    plt.quiver(x1, y1, proj_x * scale, proj_y * scale, angles="xy", scale=1, scale_units='xy', width=0.002)

    plt.text(10, 70, "contractile force = " + str(np.round(contractile_force * 10 ** 6, 2)), color="black")

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(mappable=im, cax=cax)
    cbar.set_label('traction forces in kPa')


def gif_mask_overlay(im1, im2, mask):
    im1.setflags(write=1)
    im2.setflags(write=1)
    im1[:, :, 0] += mask.astype("uint8") * 100
    im2[:, :, 0] += mask.astype("uint8") * 100
    images = [im1, im2]
    return images
