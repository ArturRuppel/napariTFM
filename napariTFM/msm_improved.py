"""
Monolayer Stress Microscopy (MSM) module implementing finite element method (FEM) for
calculating stress fields from traction forces in cell monolayers.

Core MSM implementation is based on:
- pyTFM package (https://github.com/fabrylab/pyTFM) - GNU GPL v3.0 License
- Bauer et al. (2021). pyTFM: A tool for traction force and monolayer stress
  microscopy. PLoS Computational Biology, 17(6), e1008364.
  https://doi.org/10.1371/journal.pcbi.1008364

FEM solver and assembly optimizations adapted from:
- SolidsPy package (https://github.com/jpablo/solidspy) - MIT License
- Gómez et al. (2019). SolidsPy: A FEM Implementation in Python for Teaching and Research
  https://doi.org/10.21105/jose.00073

Numerical methods and constraint handling based on:
- Tambe et al. (2011). Collective cell guidance by cooperative intercellular forces
  Nature Materials, 10(6), 469-475.
- Tambe et al. (2013). Monolayer stress microscopy: limitations, artifacts, and
  accuracy of recovered intercellular stresses
PLoS ONE, 8(2), e55172.

The implementation uses Numba-accelerated operations for performance optimization
while maintaining the accuracy of the original MSM method. The solver employs an
optimized LSQR implementation with careful constraint handling for solving the
resulting system of equations.
"""


import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import os
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import solidspy.assemutil as ass
import solidspy.postprocesor as pos
import solidspy.solutil as sol
from numba import jit
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, vstack, diags
from scipy.sparse.linalg import lsqr
from skimage.measure import regionprops

from napariTFM.mesh_generator import AdaptiveTriangleMeshGenerator


@jit(nopython=True)
def shape_quad4_numba(r, s):
    """Shape functions and derivatives for quad4 element"""
    N = np.array([
        (1 - r) * (1 - s) / 4,
        (1 + r) * (1 - s) / 4,
        (1 + r) * (1 + s) / 4,
        (1 - r) * (1 + s) / 4
    ])

    dN = np.array([
        [-(1 - s), (1 - s), (1 + s), -(1 + s)],
        [-(1 - r), -(1 + r), (1 + r), (1 - r)]
    ]) * 0.25

    return N, dN


@jit(nopython=True)
def elast_diff_2d_numba(r, s, coord):
    """Calculate B and H matrices for 2D elasticity"""
    N, dN = shape_quad4_numba(r, s)

    J = np.zeros((2, 2))
    J[0, 0] = np.sum(dN[0] * coord[:, 0])
    J[0, 1] = np.sum(dN[0] * coord[:, 1])
    J[1, 0] = np.sum(dN[1] * coord[:, 0])
    J[1, 1] = np.sum(dN[1] * coord[:, 1])

    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    Jinv = np.array([
        [J[1, 1], -J[0, 1]],
        [-J[1, 0], J[0, 0]]
    ]) / det

    B = np.zeros((3, 8))
    dNx = Jinv[0, 0] * dN[0] + Jinv[0, 1] * dN[1]
    dNy = Jinv[1, 0] * dN[0] + Jinv[1, 1] * dN[1]

    for i in range(4):
        B[0, 2 * i] = dNx[i]
        B[1, 2 * i + 1] = dNy[i]
        B[2, 2 * i] = dNy[i]
        B[2, 2 * i + 1] = dNx[i]

    H = np.zeros((2, 8))
    for i in range(4):
        H[0, 2 * i] = N[i]
        H[1, 2 * i + 1] = N[i]

    return H, B, det


@jit(nopython=True)
def elast_quad4_numba(coord, params):
    """Elastic quadrilateral element calculation"""
    E = params[0]
    nu = params[1]
    dens = 1.0 if len(params) <= 2 else params[2]

    fact = E / (1 - nu * nu)
    C = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ]) * fact

    K = np.zeros((8, 8))
    M = np.zeros((8, 8))

    gpts = np.array([
        [-0.577350269189626, -0.577350269189626],
        [0.577350269189626, -0.577350269189626],
        [0.577350269189626, 0.577350269189626],
        [-0.577350269189626, 0.577350269189626]
    ])
    gwts = np.array([1.0, 1.0, 1.0, 1.0])

    for i in range(4):
        r, s = gpts[i]
        H, B, det = elast_diff_2d_numba(r, s, coord)
        factor = det * gwts[i]
        K += factor * (B.T @ C @ B)
        M += dens * factor * (H.T @ H)

    return K, M


@jit(nopython=True)
def numba_assembler_core(elements, mats, nodes, neq, assem_op):
    """Numba-accelerated core assembly operations for triangular elements"""
    nels = elements.shape[0]
    ndof_per_el = 6  # Changed from 8 to 6 for triangles

    total_entries = nels * ndof_per_el * ndof_per_el
    rows = np.zeros(total_entries, dtype=np.int32)
    cols = np.zeros(total_entries, dtype=np.int32)
    stiff_vals = np.zeros(total_entries)
    mass_vals = np.zeros(total_entries)

    entry_idx = 0
    for el in range(nels):
        elem_nodes = elements[el, 3:6]  # Changed from 3:7 to 3:6 for triangles
        elcoor = np.zeros((3, 2))  # Changed from (4,2) to (3,2)
        for i in range(3):  # Changed from range(4)
            elcoor[i, 0] = nodes[elem_nodes[i], 1]
            elcoor[i, 1] = nodes[elem_nodes[i], 2]

        params = mats[elements[el, 2]]
        kloc, mloc = elast_tri_numba(elcoor, params)
        dme = assem_op[el, :6]  # Changed from :8 to :6

        for i in range(6):  # Changed from range(8)
            for j in range(6):  # Changed from range(8)
                if dme[i] != -1 and dme[j] != -1:
                    rows[entry_idx] = dme[i]
                    cols[entry_idx] = dme[j]
                    stiff_vals[entry_idx] = kloc[i, j]
                    mass_vals[entry_idx] = mloc[i, j]
                    entry_idx += 1

    return rows[:entry_idx], cols[:entry_idx], stiff_vals[:entry_idx], mass_vals[:entry_idx]


@jit(nopython=True)
def prepare_constraint_data_numba(nodes_xy, x_points, y_points, com, neq):
    """
    Numba-accelerated preparation of constraint data
    """
    # Calculate positions relative to center of mass
    r = np.zeros((neq, 2))
    for i in range(neq):
        if x_points[i] or y_points[i]:
            r[i, 0] = nodes_xy[i, 0] - com[1]
            r[i, 1] = nodes_xy[i, 1] - com[0]

    # Pre-calculate array sizes
    n_x = np.sum(x_points)
    n_y = np.sum(y_points)
    total_constraints = n_x + n_y + n_x + n_y  # Zero displacement + torque constraints

    # Initialize arrays for constraint matrix construction
    constraint_data = np.zeros(total_constraints)
    constraint_rows = np.zeros(total_constraints, dtype=np.int32)
    constraint_cols = np.zeros(total_constraints, dtype=np.int32)

    idx = 0

    # Zero displacement constraints for x
    for i in range(neq):
        if x_points[i]:
            constraint_data[idx] = 1.0
            constraint_rows[idx] = 0
            constraint_cols[idx] = i
            idx += 1

    # Zero displacement constraints for y
    for i in range(neq):
        if y_points[i]:
            constraint_data[idx] = 1.0
            constraint_rows[idx] = 1
            constraint_cols[idx] = i
            idx += 1

    # Torque constraints
    for i in range(neq):
        if x_points[i]:
            constraint_data[idx] = r[i, 1]  # y component for x DOF
            constraint_rows[idx] = 2
            constraint_cols[idx] = i
            idx += 1

        if y_points[i]:
            constraint_data[idx] = -r[i, 0]  # -x component for y DOF
            constraint_rows[idx] = 2
            constraint_cols[idx] = i
            idx += 1

    return constraint_data[:idx], constraint_rows[:idx], constraint_cols[:idx]


@jit(nopython=True)
def csr_matvec(data, indices, indptr, x, out):
    """Numba-optimized CSR matrix-vector multiplication"""
    for i in range(len(indptr) - 1):
        sum_val = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            sum_val += data[j] * x[indices[j]]
        out[i] = sum_val


@jit(nopython=True)
def project_to_nullspace(x, nodes_xy, x_points, y_points, com):
    """
    Project vector onto nullspace of constraint matrix with improved stability
    """
    # Calculate inertia tensor components
    I_xx = I_yy = I_xy = 0.0
    fx = fy = torque = 0.0
    n_x = n_y = 0

    for i in range(len(x)):
        if x_points[i]:
            fx += x[i]
            y_arm = nodes_xy[i, 1] - com[0]
            x_arm = nodes_xy[i, 0] - com[1]
            I_yy += y_arm * y_arm
            I_xy += x_arm * y_arm
            torque += y_arm * x[i]
            n_x += 1

        if y_points[i]:
            fy += x[i]
            y_arm = nodes_xy[i, 1] - com[0]
            x_arm = nodes_xy[i, 0] - com[1]
            I_xx += x_arm * x_arm
            I_xy += x_arm * y_arm
            torque -= x_arm * x[i]
            n_y += 1

    # Calculate corrections with improved stability
    fx_corr = fx / max(n_x, 1)
    fy_corr = fy / max(n_y, 1)

    # Calculate determinant of inertia tensor
    I_det = I_xx * I_yy - I_xy * I_xy
    eps = np.finfo(np.float64).eps * max(I_xx, I_yy)

    if I_det > eps:
        # Full inertia tensor approach
        I_inv_xx = I_yy / I_det
        I_inv_yy = I_xx / I_det
        I_inv_xy = -I_xy / I_det
        torque_scale = 1.0
    else:
        # Fallback for near-singular case
        I_inv_xx = I_inv_yy = 1.0 / (max(max(I_xx, I_yy), eps))
        I_inv_xy = 0.0
        torque_scale = 0.5  # Reduced influence of torque correction

    # Apply corrections with physical consistency
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x_points[i]:
            y_arm = nodes_xy[i, 1] - com[0]
            x_arm = nodes_xy[i, 0] - com[1]
            result[i] = x[i] - fx_corr - torque_scale * (I_inv_xx * y_arm * torque + I_inv_xy * x_arm * torque)

        if y_points[i]:
            y_arm = nodes_xy[i, 1] - com[0]
            x_arm = nodes_xy[i, 0] - com[1]
            result[i] = x[i] - fy_corr - torque_scale * (I_inv_xy * y_arm * torque + I_inv_yy * x_arm * torque)

    return result


@jit(nopython=True)
def block_diagonal_preconditioner(data, indices, indptr):
    """
    Create a block diagonal preconditioner with improved scaling and stability
    """
    n = len(indptr) - 1
    M_inv = np.zeros(n)

    # Extract diagonal entries and calculate statistics
    diag_vals = np.zeros(n)
    min_diag = np.inf
    max_diag = -np.inf

    for i in range(n):
        for j in range(indptr[i], indptr[i + 1]):
            if indices[j] == i:
                diag_vals[i] = abs(data[j])
                if diag_vals[i] > 0:
                    min_diag = min(min_diag, diag_vals[i])
                    max_diag = max(max_diag, diag_vals[i])
                break

    # Calculate scaling parameters
    if min_diag == np.inf:
        min_diag = 1.0
    if max_diag == -np.inf:
        max_diag = 1.0

    eps = min_diag * 1e-14
    threshold = max(eps, min_diag * 1e-7)

    # Compute inverse with stability checks
    for i in range(n):
        if diag_vals[i] > threshold:
            # Use actual diagonal value
            M_inv[i] = 1.0 / diag_vals[i]
        else:
            # Use stabilized value for near-zero or zero diagonals
            M_inv[i] = 1.0 / threshold

        # Additional scaling for better conditioning
        if M_inv[i] > 1.0 / eps:
            M_inv[i] = 1.0 / eps

    return M_inv

@jit(nopython=True)
def pcg_solver_with_constraints(data, indices, indptr, b, x0, nodes_xy, x_points, y_points, com, tol, max_iter):
    """
    PCG solver with improved constraint handling and convergence
    """
    n = len(b)
    x = x0.copy()

    # Initialize vectors
    r = np.zeros_like(b)
    p = np.zeros_like(b)
    z = np.zeros_like(b)
    Ap = np.zeros_like(b)

    # Create preconditioner with improved scaling
    M_inv = block_diagonal_preconditioner(data, indices, indptr)

    # Initial residual: r = b - Ax
    csr_matvec(data, indices, indptr, x, r)
    for i in range(n):
        r[i] = b[i] - r[i]

    # Project initial residual
    r = project_to_nullspace(r, nodes_xy, x_points, y_points, com)

    # Apply preconditioner
    for i in range(n):
        z[i] = M_inv[i] * r[i]

    # Project preconditioned residual
    z = project_to_nullspace(z, nodes_xy, x_points, y_points, com)
    p[:] = z[:]

    # Initial residual norm with improved stability
    rz = np.dot(r, z)
    initial_rz = max(abs(rz), np.finfo(np.float64).eps)

    # Convergence threshold with problem scaling
    scaled_tol = tol * np.sqrt(initial_rz)

    # Main iteration loop
    for iter_count in range(max_iter):
        # Matrix-vector product
        csr_matvec(data, indices, indptr, p, Ap)

        # Project Ap
        Ap = project_to_nullspace(Ap, nodes_xy, x_points, y_points, com)

        # Compute step size with stability check
        pAp = np.dot(p, Ap)
        if abs(pAp) < np.finfo(np.float64).eps * initial_rz:
            break

        alpha = rz / pAp

        # Update solution and residual
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]

        # Project residual
        r = project_to_nullspace(r, nodes_xy, x_points, y_points, com)

        # Apply preconditioner
        for i in range(n):
            z[i] = M_inv[i] * r[i]

        # Project preconditioned residual
        z = project_to_nullspace(z, nodes_xy, x_points, y_points, com)

        # Compute new residual norm
        rz_new = np.dot(r, z)

        # Check convergence with improved criteria
        rel_error = np.sqrt(abs(rz_new) / initial_rz)
        if rel_error < scaled_tol and iter_count > 10:  # Ensure minimum iterations
            return x, iter_count + 1

        # Update search direction
        beta = rz_new / rz
        rz = rz_new

        for i in range(n):
            p[i] = z[i] + beta * p[i]

        # Project search direction
        p = project_to_nullspace(p, nodes_xy, x_points, y_points, com)

    return x, max_iter


@jit(nopython=True)
def elast_diff_2d_gauss(r, s, coord):
    """Calculate B and H matrices for 2D elasticity with improved stability"""
    N, dN = shape_quad4_numba(r, s)

    J = np.zeros((2, 2))
    J[0, 0] = np.sum(dN[0] * coord[:, 0])
    J[0, 1] = np.sum(dN[0] * coord[:, 1])
    J[1, 0] = np.sum(dN[1] * coord[:, 0])
    J[1, 1] = np.sum(dN[1] * coord[:, 1])

    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    Jinv = np.array([
        [J[1, 1], -J[0, 1]],
        [-J[1, 0], J[0, 0]]
    ]) / det

    B = np.zeros((3, 8))
    dNx = Jinv[0, 0] * dN[0] + Jinv[0, 1] * dN[1]
    dNy = Jinv[1, 0] * dN[0] + Jinv[1, 1] * dN[1]

    for i in range(4):
        B[0, 2 * i] = dNx[i]
        B[1, 2 * i + 1] = dNy[i]
        B[2, 2 * i] = dNy[i]
        B[2, 2 * i + 1] = dNx[i]

    return N, B, det


@jit(nopython=True)
def calculate_element_stresses(el, elements, nodes, UC, mats):
    """Calculate stresses for triangular element"""
    el_nodes = elements[el, 3:6]
    el_coords = nodes[el_nodes, 1:3].astype(np.float64)
    el_disps = UC[el_nodes]
    mat_id = elements[el, 2]
    params = mats[mat_id]

    # Calculate constitutive matrix
    E = params[0]
    nu = params[1]
    fact = E / (1 - nu * nu)
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ]) * fact

    # Calculate stresses at centroid
    r, s = 1/3, 1/3
    N, B, det = elast_diff_tri_numba(r, s, el_coords)

    # Natural coordinates for triangle (same for all cases)
    natural_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    # Check for element quality
    area = abs(det) / 2
    if area < 1e-10:
        # Return zero stresses for degenerate elements
        stresses = np.zeros((3, 3), dtype=np.float64)
        return stresses, natural_coords, el_coords

    # Calculate strains and stresses
    strain = np.dot(B, el_disps.flatten())
    stress = np.dot(D, strain)

    # For triangles, stress is constant across element
    stresses = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        stresses[i, 0] = stress[0]
        stresses[i, 1] = stress[1]
        stresses[i, 2] = stress[2]

    return stresses, natural_coords, el_coords


@jit(nopython=True)
def extrapolate_to_nodes(gauss_stresses, natural_coords):
    """Extrapolate Gauss point stresses to nodes using optimal sampling"""
    # Vandermonde matrix for bilinear shape functions
    V = np.zeros((4, 4))
    for i in range(4):
        r, s = natural_coords[i]
        V[i, 0] = 1.0
        V[i, 1] = r
        V[i, 2] = s
        V[i, 3] = r * s

    # Node coordinates in natural space
    node_coords = np.array([
        [-1.0, -1.0],  # Node 1
        [1.0, -1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [-1.0, 1.0]  # Node 4
    ])

    # Solve for coefficients of stress interpolation
    # Use stable method for 4x4 system
    nodal_stresses = np.zeros((4, 3))

    for stress_comp in range(3):
        # Get stresses for this component
        stress_vals = gauss_stresses[:, stress_comp]

        # Solve Va=b using stable method for 4x4 system
        a = solve_4x4(V, stress_vals)

        # Evaluate at nodes
        for node in range(4):
            r, s = node_coords[node]
            nodal_stresses[node, stress_comp] = (a[0] + a[1] * r + a[2] * s + a[3] * r * s)

    return nodal_stresses


@jit(nopython=True)
def solve_4x4(A, b):
    """Solve 4x4 system using stable method"""
    x = np.zeros(4)

    # Simple Gaussian elimination with partial pivoting
    for i in range(3):
        # Find pivot
        pivot = i
        for j in range(i + 1, 4):
            if abs(A[j, i]) > abs(A[pivot, i]):
                pivot = j

        # Swap rows
        if pivot != i:
            A[i, :], A[pivot, :] = A[pivot, :].copy(), A[i, :].copy()
            b[i], b[pivot] = b[pivot], b[i]

        # Eliminate
        for j in range(i + 1, 4):
            factor = A[j, i] / A[i, i]
            b[j] -= factor * b[i]
            for k in range(i, 4):
                A[j, k] -= factor * A[i, k]

    # Back substitution
    for i in range(3, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, 4):
            sum_val += A[i, j] * x[j]
        x[i] = (b[i] - sum_val) / A[i, i]

    return x

def timer_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get parent function if it exists
        parent_func = getattr(self, '_current_func', None)

        # Set current function
        self._current_func = func.__name__

        # Initialize timing stats dict if it doesn't exist
        if not hasattr(self, 'timing_stats'):
            self.timing_stats = {}

        # Initialize nested calls dict if it doesn't exist
        if not hasattr(self, '_nested_calls'):
            self._nested_calls = {}

        # Track nested call depth
        func_name = func.__name__
        self._nested_calls[func_name] = self._nested_calls.get(func_name, 0) + 1

        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Only store timing for the outermost call of each function
        if self._nested_calls[func_name] == 1:
            self.timing_stats[func_name] = self.timing_stats.get(func_name, 0) + execution_time
            print(f"{func_name} took {execution_time:.4f} seconds to execute")

        # Decrement nested call count
        self._nested_calls[func_name] -= 1

        # Restore parent function
        self._current_func = parent_func

        return result

    return wrapper


@jit(nopython=True)
def shape_triangle_numba(r, s):
    """Shape functions and derivatives for linear triangle element"""
    N = np.array([
        1 - r - s,  # N1
        r,  # N2
        s  # N3
    ])

    dN = np.array([
        [-1, 1, 0],  # dN/dr
        [-1, 0, 1]  # dN/ds
    ])

    return N, dN


@jit(nopython=True)
def shape_tri_numba(r, s):
    """Shape functions and derivatives for linear triangle element"""
    N = np.array([
        1 - r - s,  # N1
        r,  # N2
        s  # N3
    ])

    dN = np.array([
        [-1, 1, 0],  # dN/dr
        [-1, 0, 1]  # dN/ds
    ])

    return N, dN

@jit(nopython=True)
def elast_diff_tri_numba(r, s, coord):
    """Calculate B and H matrices for 2D elasticity with triangular elements"""
    N, dN = shape_tri_numba(r, s)

    # Calculate Jacobian
    J = np.zeros((2, 2))
    J[0, 0] = np.sum(dN[0] * coord[:, 0])
    J[0, 1] = np.sum(dN[0] * coord[:, 1])
    J[1, 0] = np.sum(dN[1] * coord[:, 0])
    J[1, 1] = np.sum(dN[1] * coord[:, 1])

    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    Jinv = np.array([
        [J[1, 1], -J[0, 1]],
        [-J[1, 0], J[0, 0]]
    ]) / det

    B = np.zeros((3, 6))  # 3 strain components, 6 DOFs (2 per node)

    # Calculate derivatives of shape functions in global coordinates
    dNx = Jinv[0, 0] * dN[0] + Jinv[0, 1] * dN[1]
    dNy = Jinv[1, 0] * dN[0] + Jinv[1, 1] * dN[1]

    # Assemble B matrix
    for i in range(3):
        B[0, 2 * i] = dNx[i]  # εxx terms
        B[1, 2 * i + 1] = dNy[i]  # εyy terms
        B[2, 2 * i] = dNy[i]  # γxy terms
        B[2, 2 * i + 1] = dNx[i]

    H = np.zeros((2, 6))  # 2 displacement components, 6 DOFs
    for i in range(3):
        H[0, 2 * i] = N[i]
        H[1, 2 * i + 1] = N[i]

    return H, B, det


@jit(nopython=True)
def elast_tri_numba(coord, params):
    """Elastic triangular element calculation"""
    E = float(params[0])  # Ensure float conversion
    nu = float(params[1])
    dens = 1.0 if len(params) <= 2 else float(params[2])

    # Print actual values during first call
    if coord[0, 0] == 0:  # Only print for first element
        print(f"Element calculation with E={E}, nu={nu}")

    fact = E / (1 - nu * nu)
    C = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ]) * fact

    r, s = 1 / 3, 1 / 3
    H, B, det = elast_diff_tri_numba(r, s, coord)

    area = abs(det) / 2  # Use absolute value for robustness

    # Add stability check
    if area < 1e-10:
        print(f"Warning: Very small element area: {area}")
        area = 1e-10  # Minimum area to prevent instability

    K = area * (B.T @ C @ B)
    M = area * dens * (H.T @ H)

    return K, M
class MonolayerStressMicroscopy:
    def __init__(self, pixelsize, sigma=0.5, youngs_modulus=1, base_refinement=0.5, boundary_refinement=2.0, gradient_refinement=1.5):
        """
        Initialize MSM calculator with triangular elements

        Args:
            pixelsize: Pixel size in microns
            sigma: Poisson's ratio
            youngs_modulus: Young's modulus
            mesh_refinement: Mesh refinement factor (higher = finer mesh)
        """
        self.pixelsize = pixelsize
        self.sigma = sigma
        self.E = youngs_modulus
        self.mesh_generator = AdaptiveTriangleMeshGenerator(
            base_refinement=base_refinement,
            boundary_refinement=boundary_refinement,
            gradient_refinement=gradient_refinement
        )
        self.timing_stats = {}
        self._nested_calls = {}
        self._current_func = None

    @timer_decorator
    def calculate_stress_field(self, traction_x, traction_y, mask):
        """Calculate stress field using triangular elements"""
        # Input validation
        if np.all(np.isnan(traction_x)) or np.all(np.isnan(traction_y)):
            raise ValueError("Input tractions are all NaN")

        print(f"Input traction range x: [{np.nanmin(traction_x)}, {np.nanmax(traction_x)}]")
        print(f"Input traction range y: [{np.nanmin(traction_y)}, {np.nanmax(traction_y)}]")

        # Calculate effective pixelsize
        forcemap_pixelsize = self.pixelsize * 1e-6

        # Prepare forces
        f_x, f_y = self._prepare_forces(traction_x, traction_y, mask, forcemap_pixelsize)

        # Verify forces after preparation
        print(f"Prepared force range x: [{np.nanmin(f_x)}, {np.nanmax(f_x)}]")
        print(f"Prepared force range y: [{np.nanmin(f_y)}, {np.nanmax(f_y)}]")

        # Setup triangular mesh
        nodes, elements, loads, mats = self._grid_setup(mask, -f_x, -f_y)

        # Calculate stress tensor
        stress_tensor = self._fem_simulation(nodes, elements, loads, mats, mask)

        return stress_tensor

    @timer_decorator
    def _prepare_forces(self, tx, ty, mask, pixelsize):
        """Convert traction forces to point forces and correct for net force and torque"""
        # Convert to point force using exact same calculation as original
        f_x = tx * (pixelsize ** 2)
        f_y = ty * (pixelsize ** 2)

        # Mask forces
        f_x[~mask] = np.nan
        f_y[~mask] = np.nan

        # Remove mean force (force balance)
        f_x = f_x - np.nanmean(f_x)
        f_y = f_y - np.nanmean(f_y)

        # Correct torque
        f_x, f_y = self._correct_torque(f_x, f_y, mask)

        return f_x, f_y

    @timer_decorator
    def _correct_torque(self, fx, fy, mask):
        """Correct for net torque using the original implementation approach"""
        com = regionprops(mask.astype(int))[0].centroid
        com = (com[1], com[0])

        c_x, c_y = np.meshgrid(range(fx.shape[1]), range(fx.shape[0]))
        r = np.zeros((fx.shape[0], fx.shape[1], 2))
        r[:, :, 0] = c_x
        r[:, :, 1] = c_y
        r = r - np.array(com)

        f = np.zeros((fx.shape[0], fx.shape[1], 2))
        f[:, :, 0] = fx
        f[:, :, 1] = fy

        def get_torque_angle(p):
            q = np.zeros_like(f)
            q[:, :, 0] = np.cos(p) * f[:, :, 0] - np.sin(p) * f[:, :, 1]
            q[:, :, 1] = np.sin(p) * f[:, :, 0] + np.cos(p) * f[:, :, 1]
            return np.abs(np.nansum(np.cross(r, q, axisa=2, axisb=2)))

        # Use exact same optimization approach as original with error handling
        pstart = 0
        eps = np.finfo(float).eps
        try:
            p = least_squares(fun=get_torque_angle, x0=pstart, method="lm",
                              max_nfev=100000000, xtol=eps, ftol=eps, gtol=eps, args=())["x"][0]
        except KeyError:
            eps *= 5
            p = least_squares(fun=get_torque_angle, x0=pstart, method="lm",
                              max_nfev=100000000, xtol=eps, ftol=eps, gtol=eps, args=())["x"][0]

        fx_corr = np.cos(p) * fx - np.sin(p) * fy
        fy_corr = np.sin(p) * fx + np.cos(p) * fy

        return fx_corr, fy_corr

    @timer_decorator
    def _grid_setup(self, mask_area, f_x, f_y):
        """Setup triangular mesh and boundary conditions"""
        # Generate triangular mesh
        nodes_xy, elements = self.mesh_generator.generate_mesh(mask_area, f_x, f_y)

        print(f"Number of nodes: {len(nodes_xy)}")
        print(f"Number of elements: {len(elements)}")
        print(f"Max input force x: {np.nanmax(np.abs(f_x))}")
        print(f"Max input force y: {np.nanmax(np.abs(f_y))}")
        print(f"E: {self.E}, nu: {self.sigma}")

        # Convert to required format
        num_nodes = len(nodes_xy)
        nodes = np.zeros((num_nodes, 5))
        nodes[:, 0] = np.arange(num_nodes)  # node numbers
        nodes[:, 1:3] = nodes_xy  # x,y coordinates
        nodes[:, 3:] = 0  # BC flags

        # Debug: Check nodes
        print(f"Node coordinate range x: [{np.min(nodes[:, 1])}, {np.max(nodes[:, 1])}]")
        print(f"Node coordinate range y: [{np.min(nodes[:, 2])}, {np.max(nodes[:, 2])}]")

        # Convert element connectivity
        num_elements = len(elements)
        elements_formatted = np.zeros((num_elements, 6), dtype=np.int64)
        elements_formatted[:, 0] = np.arange(num_elements)  # element numbers
        elements_formatted[:, 1] = 1  # element type (1 for triangle)
        elements_formatted[:, 2] = 0  # material number
        elements_formatted[:, 3:] = elements  # connectivity

        # Debug: Check elements
        print(f"Element connectivity range: [{np.min(elements_formatted[:, 3:])}, {np.max(elements_formatted[:, 3:])}]")

        # Setup loads and interpolate forces
        loads = np.zeros((num_nodes, 3))
        loads[:, 0] = np.arange(num_nodes)

        # Create interpolation masks
        fx_valid = ~np.isnan(f_x)
        fy_valid = ~np.isnan(f_y)

        # Interpolate forces and track interpolation success
        force_assigned = 0
        for i in range(num_nodes):
            x, y = int(nodes[i, 1]), int(nodes[i, 2])
            if 0 <= x < f_x.shape[1] and 0 <= y < f_x.shape[0]:
                if fx_valid[y, x]:
                    loads[i, 1] = f_x[y, x]
                if fy_valid[y, x]:
                    loads[i, 2] = f_y[y, x]
                if not (np.isnan(loads[i, 1]) or np.isnan(loads[i, 2])):
                    force_assigned += 1

        print(f"Nodes with forces assigned: {force_assigned} out of {num_nodes}")
        print(f"Force range x: [{np.min(loads[:, 1])}, {np.max(loads[:, 1])}]")
        print(f"Force range y: [{np.min(loads[:, 2])}, {np.max(loads[:, 2])}]")

        # Setup materials
        mats = np.array([[self.E, self.sigma]])

        return nodes, elements_formatted, loads, mats
    def _assemble_custom_dme(self, nodes, elements):
        """Custom DME assembly for triangular elements"""
        nnodes = len(nodes)
        nels = len(elements)

        # Initialize arrays
        DME = np.zeros((nels, 6), dtype=np.int64)  # Changed from 8 to 6
        IBC = np.zeros((nnodes, 2), dtype=np.int64)

        # Process boundary conditions
        neq = 0
        for i in range(nnodes):
            for j in range(2):
                if nodes[i, j + 3] == 0:  # Free DOF
                    IBC[i, j] = neq
                    neq += 1
                else:  # Constrained DOF
                    IBC[i, j] = -1

        # Assemble DME
        for el in range(nels):
            elem_nodes = elements[el, 3:6]  # Changed from 3:7 to 3:6
            for i, node in enumerate(elem_nodes):
                DME[el, 2 * i:2 * i + 2] = IBC[node]

        return DME, IBC, neq

    @timer_decorator
    def _custom_assembler(self, elements, mats, nodes, neq, assem_op):
        """Custom assembler for triangular elements"""
        rows, cols, stiff_vals, mass_vals = numba_assembler_core(elements, mats, nodes, neq, assem_op)

        stiff = csr_matrix((stiff_vals, (rows, cols)), shape=(neq, neq))
        mass = csr_matrix((mass_vals, (rows, cols)), shape=(neq, neq))

        return stiff, mass

    @timer_decorator
    def _fem_simulation(self, nodes, elements, loads, mats, mask):
        """Main FEM simulation with simplified stress calculation and interpolation"""
        # System assembly
        DME, IBC, neq = self._assemble_custom_dme(nodes, elements)
        KG, MG = self._custom_assembler(elements, mats, nodes, neq, DME)
        RHSG = ass.loadasem(loads, IBC, neq)

        # Solve system
        UG_sol = self._custom_solver(KG, RHSG, mask, nodes, IBC)

        # Complete displacement calculation
        UC = pos.complete_disp(IBC, nodes, UG_sol)

        # Calculate nodal stresses
        num_nodes = len(nodes)
        nodal_stresses = np.zeros((num_nodes, 3))  # [σxx, σyy, σxy] for each node
        node_weights = np.zeros(num_nodes)

        # Calculate stresses at nodes by averaging from connected elements
        for el in range(len(elements)):
            el_nodes = elements[el, 3:6]  # indices of nodes for this element
            stresses, _, _ = calculate_element_stresses(el, elements, nodes, UC, mats)

            # Simple averaging - each element contributes equally to its nodes
            for i, node_idx in enumerate(el_nodes):
                nodal_stresses[node_idx] += stresses[i]
                node_weights[node_idx] += 1

        # Average the stresses at nodes
        valid_nodes = node_weights > 0
        nodal_stresses[valid_nodes] /= node_weights[valid_nodes, np.newaxis]

        # Interpolate stresses back to regular grid
        return self._interpolate_stress_field(nodes, nodal_stresses, mask)

    def _interpolate_stress_field(self, nodes, nodal_stresses, mask):
        """
        Simple linear interpolation of nodal stresses to regular grid.

        Args:
            nodes: Array of node coordinates and data
            nodal_stresses: Array of stress components at nodes [σxx, σyy, σxy]
            mask: Boolean mask indicating the region of interest

        Returns:
            stress_tensor: Array of shape (*mask.shape, 2, 2) containing interpolated stress tensor
        """
        from scipy.interpolate import LinearNDInterpolator

        # Setup interpolation points
        points = nodes[:, 1:3]  # node coordinates
        grid_y, grid_x = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        # Initialize stress tensor
        stress_tensor = np.zeros((*mask.shape, 2, 2))

        # Interpolate each stress component
        for i, comp in enumerate(['xx', 'yy', 'xy']):
            # Create interpolator for this stress component
            interpolator = LinearNDInterpolator(points, nodal_stresses[:, i])

            # Perform interpolation
            stress_field = interpolator(grid_points).reshape(mask.shape)

            # Fill any NaN values with nearest neighbor
            if np.any(np.isnan(stress_field)):
                from scipy.interpolate import NearestNDInterpolator
                nn_interpolator = NearestNDInterpolator(points, nodal_stresses[:, i])
                nan_mask = np.isnan(stress_field)
                stress_field[nan_mask] = nn_interpolator(grid_points).reshape(mask.shape)[nan_mask]

            # Apply mask
            stress_field[~mask] = 0.0

            # Assign to appropriate stress tensor component
            if comp == 'xx':
                stress_tensor[..., 0, 0] = stress_field
            elif comp == 'yy':
                stress_tensor[..., 1, 1] = stress_field
            else:  # xy component
                stress_tensor[..., 0, 1] = stress_field
                stress_tensor[..., 1, 0] = stress_field  # Symmetric tensor

        return stress_tensor

    def print_timing_stats(self):
        """Print detailed timing statistics"""
        print("\nDetailed Timing Statistics:")
        print("-" * 50)

        # Define parent functions to exclude from detailed stats
        parent_functions = {'calculate_stress_field', '_fem_simulation'}

        # Filter out parent functions for the detailed statistics
        filtered_stats = {op: duration for op, duration in self.timing_stats.items()
                          if op not in parent_functions}

        # Calculate total time using only leaf operations
        total_time = sum(filtered_stats.values())

        # Sort operations by duration
        sorted_stats = sorted(filtered_stats.items(),
                              key=lambda x: x[1],
                              reverse=True)

        # Print each operation's timing
        for operation, duration in sorted_stats:
            percentage = (duration / total_time) * 100
            print(f"{operation:20s}: {duration:8.4f} s ({percentage:5.1f}%)")

        print("-" * 50)
        print(f"{'Total time':20s}: {total_time:8.4f} s")

        # Optionally print overall execution time if available
        if hasattr(self, 'total_start_time'):
            overall_time = time.time() - self.total_start_time
            print(f"Total script execution time: {overall_time:.4f} seconds")

    def _find_eq_position(self, nodes, IBC, neq):
        """Find equilibrium positions for nodes with proper DOF handling"""
        nloads = IBC.shape[0]
        nodes_xy = np.zeros((neq, 2))  # Store x,y positions for each DOF
        x_points = np.zeros(neq, dtype=bool)  # x DOFs
        y_points = np.zeros(neq, dtype=bool)  # y DOFs

        for i in range(nloads):
            ilx = IBC[i, 0]  # x DOF number
            ily = IBC[i, 1]  # y DOF number

            if ilx != -1:  # If x DOF is free
                nodes_xy[ilx] = nodes[i, 1:3]  # Store node position
                x_points[ilx] = True

            if ily != -1:  # If y DOF is free
                nodes_xy[ily] = nodes[i, 1:3]  # Store node position
                y_points[ily] = True

        return nodes_xy, x_points, y_points

    def _custom_solver(self, KG, RHSG, mask, nodes, IBC):
        """Hybrid solver with preconditioning applied to system directly"""
        neq = KG.shape[0]

        print(f"System size: {neq}")
        print(f"Condition number estimate: {np.linalg.norm(KG.todense(), 1) * np.linalg.norm(np.linalg.pinv(KG.todense()), 1)}")
        print(f"Max RHS value: {np.max(np.abs(RHSG))}")
        print(f"Number of non-zeros in K: {KG.nnz}")

        # Get node positions and set up constraints using optimized method
        nodes_xy, x_points, y_points = self._find_eq_position(nodes, IBC, neq)

        # Calculate center of mass
        com = regionprops(mask.astype(int))[0].centroid

        # Use Numba-accelerated constraint preparation
        constraint_data, constraint_rows, constraint_cols = prepare_constraint_data_numba(
            nodes_xy, x_points, y_points, com, neq
        )

        # Create sparse constraint matrix
        constraints = csr_matrix(
            (constraint_data, (constraint_rows, constraint_cols)),
            shape=(3, neq)
        )

        # Create scaling based on diagonal of KG
        diag = np.array(KG.diagonal())
        scale = np.ones_like(diag)
        valid_diag = np.abs(diag) > 1e-10
        scale[valid_diag] = 1.0 / np.sqrt(np.abs(diag[valid_diag]))

        # Create scaling matrix
        S = diags(scale)

        # Scale system
        KG_scaled = KG.dot(S)

        # Scale constraints and stack
        constraint_scale = np.median(np.abs(diag[valid_diag]))


        KG_constrained = vstack([KG_scaled, constraints * constraint_scale], format="csr")
        RHSG_constrained = np.append(RHSG, np.zeros(3))

        # Solve scaled system using LSQR
        x_scaled = lsqr(KG_constrained, RHSG_constrained,
                        atol=1e-12,
                        btol=1e-12,
                        iter_lim=200000,
                        show=False)[0]

        # Unscale solution
        UG_sol = S.dot(x_scaled[:neq])

        return UG_sol

# Example usage
if __name__ == "__main__":


    # Start total execution timing
    total_start_time = time.time()

    # Load data
    data_path = r"C:\Users\aruppel\Desktop\test"

    # Load traction force data
    t_x = np.load(os.path.join(data_path, "t_x.npy"))[0, :, :]
    t_y = np.load(os.path.join(data_path, "t_y.npy"))[0, :, :]

    # Load reference stress tensor
    stress_tensor_ref = np.load(os.path.join(data_path, "stress_tensor_reference.npy"))

    # Load mask
    mask = tifffile.imread(os.path.join(data_path, "masks.tif")).astype(bool)[0, :, :]

    # Initialize MSM calculator
    pixelsize = 0.8  # microns per pixel
    msm = MonolayerStressMicroscopy(pixelsize=pixelsize)

    # Calculate stress field
    stress_tensor = msm.calculate_stress_field(t_x, t_y, mask) # in N/pixel
    stress_tensor = stress_tensor / (pixelsize * 1e-6)

    # Calculate principal stresses for both tensors
    def calculate_max_principal_stress(tensor):
        sigma_xx = tensor[:, :, 0, 0]
        sigma_yy = tensor[:, :, 1, 1]
        sigma_xy = tensor[:, :, 0, 1]
        return (sigma_xx + sigma_yy) / 2 + np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + sigma_xy ** 2)

    # Calculate maximum principal stresses
    sigma_max = calculate_max_principal_stress(stress_tensor)
    sigma_max_ref = calculate_max_principal_stress(stress_tensor_ref)

    # Extract normal stress components
    sigma_xx = stress_tensor[:, :, 0, 0]
    sigma_yy = stress_tensor[:, :, 1, 1]
    sigma_xx_ref = stress_tensor_ref[:, :, 0, 0]
    sigma_yy_ref = stress_tensor_ref[:, :, 1, 1]

    # Print timing statistics
    msm.print_timing_stats()

    # Calculate total execution time
    total_execution_time = time.time() - total_start_time
    print(f"\nTotal script execution time: {total_execution_time:.4f} seconds")

    # Create visualization with 3 rows (max principal, sigma_xx, sigma_yy)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # Function to plot stress component with consistent formatting
    def plot_stress(ax, data, title):
        vmax = 0.5 * np.nanmax(data)  # Scale to 0.5 * max value for each plot individually
        vmin = 0.5 * np.nanmin(data)  # Scale to 0.5 * max value for each plot individually
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Stress (Pa)')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        return im

    # Plot maximum principal stress
    plot_stress(axes[0, 0], sigma_max, 'Calculated Maximum Principal Stress')
    plot_stress(axes[0, 1], sigma_max_ref, 'Reference Maximum Principal Stress')

    # Plot sigma_xx
    plot_stress(axes[1, 0], sigma_xx, 'Calculated σxx')
    plot_stress(axes[1, 1], sigma_xx_ref, 'Reference σxx')

    # Plot sigma_yy
    plot_stress(axes[2, 0], sigma_yy, 'Calculated σyy')
    plot_stress(axes[2, 1], sigma_yy_ref, 'Reference σyy')

    plt.tight_layout()
    plt.show()