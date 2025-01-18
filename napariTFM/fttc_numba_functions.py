from numba import njit
import numpy as np
@njit(cache=True)
def calculate_2x2_inv(U):
    """Calculates inverse of 2x2 matrix"""
    U_inv = np.empty((2, 2), dtype=np.complex128)
    detU = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    invdetU = 1.0 / detU
    U_inv[0, 0] = invdetU * U[1, 1]
    U_inv[0, 1] = -invdetU * U[0, 1]
    U_inv[1, 0] = -invdetU * U[1, 0]
    U_inv[1, 1] = invdetU * U[0, 0]
    return U_inv

@njit(cache=True)
def blkmul_adj(mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Calculate (mat.H) @ v"""
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

@njit(cache=True)
def calculate_traction_2d(FtGmn, L):
    """Calculates Tikhonov regularized inverse of FTGmn"""
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