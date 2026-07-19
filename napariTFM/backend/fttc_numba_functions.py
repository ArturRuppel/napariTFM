from numba import njit
import numpy as np

# nogil=True is load-bearing, not decoration. These kernels are the per-frame
# hot path of the force stage's streaming worker (a napari @thread_worker on a
# QThreadPool). Without nogil the kernel holds the GIL for the whole frame,
# starving the GUI thread — so a Cancel click can't even be processed until the
# frame finishes, which reads to the user as "cancel is ignored". Releasing the
# GIL lets the GUI thread run the cooperative cancel promptly. Each kernel is
# nopython and writes only its own local arrays (no shared mutable state), so
# releasing the GIL is safe.


@njit(nogil=True, cache=False)
def calculate_2x2_inv(U):
    """Calculate inverse of a 2x2 complex matrix.

    Optimized for 2x2 matrices used in FTTC calculations.

    Args:
        U: Complex 2x2 matrix to invert

    Returns:
        Complex 2x2 inverse matrix
    """
    U_inv = np.empty((2, 2), dtype=np.complex128)
    detU = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    invdetU = 1.0 / detU
    U_inv[0, 0] = invdetU * U[1, 1]
    U_inv[0, 1] = -invdetU * U[0, 1]
    U_inv[1, 0] = -invdetU * U[1, 0]
    U_inv[1, 1] = invdetU * U[0, 0]
    return U_inv


@njit(nogil=True, cache=False)
def blkmul_adj(mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Calculate adjoint multiplication (mat.H) @ v for block matrices.

    Used in the FTTC regularization (SVD block) path, incl. Bayesian λ selection.

    Args:
        mat: Input matrix of shape (a, b, c)
        v: Input vector of shape (a*b,)

    Returns:
        Result vector of shape (a*c,)
    """
    a, b, c = mat.shape
    assert ((a * b,) == v.shape)
    assert (a >= 1)
    MT0 = mat[0].T.conjugate()
    out0 = MT0 @ v[:c]
    out = np.empty(a * c, dtype=out0.dtype)
    out[:c] = out0
    for i in range(1, a):
        MT = mat[i].T.conjugate()
        out[i * c:i * c + c] = MT @ v[i * b:i * b + b]
    return out


@njit(nogil=True, cache=False)
def calculate_traction_2d(FtGmn, L):
    """Calculate Tikhonov regularized inverse of the Green's function in Fourier space.

    Args:
        FtGmn: Fourier transformed Green's function (2x2xMxN complex array)
        L: Tikhonov regularization parameter (scalar)

    Returns:
        Regularized inverse (2x2xMxN complex array)
    """
    M = len(FtGmn[0, 0])
    N = len(FtGmn[0, 0, 0])

    FtGmnInv = np.empty((2, 2, M, N), dtype=np.complex128)
    Tikh = np.zeros((2, 2), dtype=np.complex128)
    Tikh[0, 0] = L
    Tikh[1, 1] = L

    GG = np.empty((2, 2), dtype=np.complex128)
    for i in range(M):
        for j in range(N):
            GG[0, 0] = FtGmn[0, 0, i, j]
            GG[0, 1] = FtGmn[0, 1, i, j]
            GG[1, 0] = FtGmn[1, 0, i, j]
            GG[1, 1] = FtGmn[1, 1, i, j]

            FtGmnInv[:, :, i, j] = np.dot(
                calculate_2x2_inv(np.dot(GG.T, GG) + Tikh), GG.T
            )
    return FtGmnInv