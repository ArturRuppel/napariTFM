import time
from functools import wraps

import solidspy.assemutil as ass
import solidspy.postprocesor as pos
import solidspy.solutil as sol
from numba import jit
from numpy import vstack
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr
from skimage.measure import regionprops



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
    """Numba-accelerated core assembly operations"""
    nels = elements.shape[0]
    ndof_per_el = 8

    total_entries = nels * ndof_per_el * ndof_per_el
    rows = np.zeros(total_entries, dtype=np.int32)
    cols = np.zeros(total_entries, dtype=np.int32)
    stiff_vals = np.zeros(total_entries)
    mass_vals = np.zeros(total_entries)

    entry_idx = 0
    for el in range(nels):
        elem_nodes = elements[el, 3:]
        elcoor = np.zeros((4, 2))
        for i in range(4):
            elcoor[i, 0] = nodes[elem_nodes[i], 1]
            elcoor[i, 1] = nodes[elem_nodes[i], 2]

        params = mats[elements[el, 2]]
        kloc, mloc = elast_quad4_numba(elcoor, params)
        dme = assem_op[el, :8]

        for i in range(8):
            for j in range(8):
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
def vector_operations(p, r, z, alpha, beta):
    """Fused vector operations for PCG"""
    n = len(p)
    rz_new = 0.0
    rz_old = 0.0
    pAp = 0.0

    for i in range(n):
        rz_old += r[i] * z[i]

    # Update solution and residual
    for i in range(n):
        p[i] = z[i] + beta * p[i]
        rz_new += r[i] * z[i]

    return rz_new, rz_old, pAp


@jit(nopython=True)
def pcg_solver_numba(data, indices, indptr, b, x0, M_inv, tol, max_iter):
    """
    Preconditioned Conjugate Gradient solver with Numba optimization

    Parameters:
    -----------
    data, indices, indptr : arrays
        CSR format matrix data
    b : array
        Right-hand side vector
    x0 : array
        Initial guess
    M_inv : array
        Diagonal preconditioner (inverse)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns:
    --------
    x : array
        Solution vector
    num_iter : int
        Number of iterations performed
    """
    n = len(b)
    x = x0.copy()

    # Allocate vectors
    r = np.zeros_like(b)
    p = np.zeros_like(b)
    z = np.zeros_like(b)
    Ap = np.zeros_like(b)

    # Initial residual: r = b - Ax
    csr_matvec(data, indices, indptr, x, r)
    for i in range(n):
        r[i] = b[i] - r[i]

    # Apply preconditioner: z = M⁻¹r
    for i in range(n):
        z[i] = M_inv[i] * r[i]

    # Initial search direction
    p[:] = z[:]

    # Initial residual norm
    rz = np.dot(r, z)
    initial_rz = rz

    # Main iteration loop
    for iter_count in range(max_iter):
        # Matrix-vector product: Ap = A*p
        csr_matvec(data, indices, indptr, p, Ap)

        # Compute step size alpha
        pAp = np.dot(p, Ap)
        if pAp == 0.0:
            break
        alpha = rz / pAp

        # Update solution and residual
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]

        # Apply preconditioner
        for i in range(n):
            z[i] = M_inv[i] * r[i]

        # Compute beta and update search direction
        rz_new = np.dot(r, z)
        beta = rz_new / rz

        # Check convergence
        if np.sqrt(rz_new) < tol * np.sqrt(initial_rz):
            return x, iter_count + 1

        # Update for next iteration
        rz = rz_new
        p = z + beta * p

    return x, max_iter


@jit(nopython=True)
def prepare_constraint_data_numba(nodes_xy, x_points, y_points, com, neq):
    """
    Prepare constraint data with improved numerical stability
    """
    # Calculate distances from center of mass with better precision
    r = np.zeros((neq, 2))
    for i in range(neq):
        if x_points[i] or y_points[i]:
            r[i, 0] = nodes_xy[i, 0] - com[1]  # x distance
            r[i, 1] = nodes_xy[i, 1] - com[0]  # y distance

    # Calculate constraint matrix sizes
    n_x = np.sum(x_points)
    n_y = np.sum(y_points)
    total_constraints = n_x + n_y + n_x + n_y

    # Initialize arrays
    constraint_data = np.zeros(total_constraints)
    constraint_rows = np.zeros(total_constraints, dtype=np.int32)
    constraint_cols = np.zeros(total_constraints, dtype=np.int32)

    idx = 0

    # Force balance constraints with improved scaling
    scale_factor = 1.0 / max(np.sqrt(n_x + n_y), 1.0)

    # X-direction force balance
    for i in range(neq):
        if x_points[i]:
            constraint_data[idx] = scale_factor
            constraint_rows[idx] = 0
            constraint_cols[idx] = i
            idx += 1

    # Y-direction force balance
    for i in range(neq):
        if y_points[i]:
            constraint_data[idx] = scale_factor
            constraint_rows[idx] = 1
            constraint_cols[idx] = i
            idx += 1

    # Moment balance with improved scaling
    I_xx = I_yy = I_xy = 0.0
    for i in range(neq):
        if x_points[i]:
            I_yy += r[i, 1] * r[i, 1]
            I_xy += r[i, 0] * r[i, 1]
        if y_points[i]:
            I_xx += r[i, 0] * r[i, 0]
            I_xy += r[i, 0] * r[i, 1]

    I_scale = np.sqrt(I_xx * I_yy)
    if I_scale > 0:
        moment_scale = scale_factor / np.sqrt(I_scale)
    else:
        moment_scale = scale_factor

    # Apply moment constraints with improved scaling
    for i in range(neq):
        if x_points[i]:
            constraint_data[idx] = r[i, 1] * moment_scale
            constraint_rows[idx] = 2
            constraint_cols[idx] = i
            idx += 1
        if y_points[i]:
            constraint_data[idx] = -r[i, 0] * moment_scale
            constraint_rows[idx] = 2
            constraint_cols[idx] = i
            idx += 1

    return constraint_data[:idx], constraint_rows[:idx], constraint_cols[:idx]


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
    """Calculate stresses at Gauss points with improved accuracy"""
    el_nodes = elements[el, 3:]
    el_coords = nodes[el_nodes, 1:3].astype(np.float64)
    el_disps = UC[el_nodes]
    mat_id = elements[el, 2]
    params = mats[mat_id]

    # 2x2 Gauss quadrature points
    gpts = np.array([
        [-0.577350269189626, -0.577350269189626],
        [0.577350269189626, -0.577350269189626],
        [0.577350269189626, 0.577350269189626],
        [-0.577350269189626, 0.577350269189626]
    ])

    # Calculate constitutive matrix
    E = params[0]
    nu = params[1]
    fact = E / (1 - nu * nu)
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ]) * fact

    # Calculate stresses at each Gauss point
    stresses = np.zeros((4, 3))  # 4 Gauss points, 3 stress components
    natural_coords = np.zeros((4, 2))  # Store natural coordinates

    for i in range(4):
        r, s = gpts[i]
        N, B, det = elast_diff_2d_gauss(r, s, el_coords)

        # Calculate strains and stresses
        strain = np.dot(B, el_disps.flatten())
        stresses[i] = np.dot(D, strain)
        natural_coords[i] = [r, s]

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


class MonolayerStressMicroscopy:
    def __init__(self, pixelsize, sigma=0.5, youngs_modulus=1):
        """
        Initialize MSM calculator
        Args:
            pixelsize: Pixel size in microns
            sigma: Poisson's ratio for the material
            youngs_modulus: Young's modulus for the material
        """
        self.pixelsize = pixelsize
        self.sigma = sigma
        self.E = youngs_modulus
        self.timing_stats = {}  # Store timing information
        self._nested_calls = {}  # Track nested function calls
        self._current_func = None  # Track current function

    @timer_decorator
    def calculate_stress_field(self, traction_x, traction_y, mask):
        """
        Calculate stress field from traction forces
        Args:
            traction_x: X component of traction forces
            traction_y: Y component of traction forces
            mask: Boolean mask of valid area
            downsample_factor: Factor to downsample the input data
        Returns:
            stress_tensor: Calculated stress tensor field
        """

        # Calculate effective pixelsize
        forcemap_pixelsize = self.pixelsize * 1e-6  # Convert to meters

        # Prepare forces with corrected pixelsize
        f_x, f_y = self._prepare_forces(traction_x, traction_y, mask, forcemap_pixelsize)

        # Setup FEM grid
        nodes, elements, loads, mats = self._grid_setup(mask, -f_x, -f_y)

        first_elem = elements[0]
        print("First element coordinates:")
        elem_coords = nodes[first_elem[3:], 1:]  # Getting coords for first element
        print("Shape:", elem_coords.shape)
        print("Values:\n", elem_coords)

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
    def _grid_setup(self, mask_area, f_x, f_y, edge_factor=0):
        coords = np.array(np.where(mask_area))

        # Node setup - explicitly separate coordinates from BCs
        nodes = np.zeros((coords.shape[1], 5), dtype=np.float64)  # Changed to float64
        nodes[:, 0] = np.arange(coords.shape[1])  # node numbers
        nodes[:, 1:3] = np.vstack((coords[1], coords[0])).T  # Only x,y coordinates
        nodes[:, 3:] = 0  # BC flags initialized to 0

        # Create node ID array
        ids = np.zeros(mask_area.shape, dtype=np.int64).T - 1
        ids[coords[0], coords[1]] = np.arange(coords.shape[1], dtype=np.int64)

        # Create elements with CCW ordering
        element_list = []
        element_id = 0

        for i in range(coords.shape[1]):
            x, y = coords[1][i], coords[0][i]
            if x > 0 and y > 0:
                # Get node IDs in CCW order
                node_ids = [
                    ids[y, x],  # current point (bottom right)
                    ids[y, x - 1],  # left point (bottom left)
                    ids[y - 1, x - 1],  # top-left point
                    ids[y - 1, x]  # top point (top right)
                ]

                if all(nid >= 0 for nid in node_ids):
                    # Create element - ensure proper coordinate extraction
                    element = np.array([
                        element_id,  # element id
                        1,  # element type (quad)
                        0,  # material id
                        node_ids[0],  # bottom right
                        node_ids[1],  # bottom left
                        node_ids[2],  # top left
                        node_ids[3]  # top right
                    ], dtype=np.int64)

                    element_list.append(element)
                    element_id += 1

        elements = np.array(element_list, dtype=np.int64)

        # Validation prints
        print("\nFirst element coordinates check:")
        first_elem = elements[0]
        elem_coords = nodes[first_elem[3:], 1:3]  # Only get x,y coordinates
        print("Coordinate shape:", elem_coords.shape)
        print("Coordinates:\n", elem_coords)

        # Setup loads and materials
        loads = np.zeros((len(nodes), 3))
        loads[:, 0] = np.arange(len(nodes))
        loads[:, 1] = f_x[coords[0], coords[1]]
        loads[:, 2] = f_y[coords[0], coords[1]]

        mats = np.array([[self.E, self.sigma]])

        return nodes, elements, loads, mats

    @timer_decorator
    def _custom_assembler(self, elements, mats, nodes, neq, assem_op):
        """
        Numba-accelerated assembler using optimized operations
        """
        rows, cols, stiff_vals, mass_vals = numba_assembler_core(elements, mats, nodes, neq, assem_op)

        stiff = csr_matrix((stiff_vals, (rows, cols)), shape=(neq, neq))
        mass = csr_matrix((mass_vals, (rows, cols)), shape=(neq, neq))

        return stiff, mass

    @timer_decorator
    def _fem_simulation(self, nodes, elements, loads, mats, mask):
        """Main FEM simulation with improved stress calculation"""
        DME, IBC, neq = ass.DME(nodes[:, 3:], elements)

        # System assembly
        KG, MG = self._custom_assembler(elements, mats, nodes, neq, DME)
        RHSG = ass.loadasem(loads, IBC, neq)

        # Solve system
        if np.sum(IBC == -1) < 3:
            UG_sol = self._custom_solver(KG, RHSG, mask, nodes, IBC)
        else:
            UG_sol = sol.static_sol(KG, RHSG)

        # Complete displacement field
        UC = pos.complete_disp(IBC, nodes, UG_sol)

        # Initialize stress arrays
        stress_tensor = np.zeros((*mask.shape, 2, 2))
        stress_counts = np.zeros(mask.shape, dtype=np.int32)

        # Calculate stresses for each element
        for el in range(len(elements)):
            # Get stresses at Gauss points
            gauss_stresses, natural_coords, el_coords = calculate_element_stresses(
                el, elements, nodes, UC, mats
            )

            # Extrapolate to nodes
            nodal_stresses = extrapolate_to_nodes(gauss_stresses, natural_coords)

            # Map element nodes to global coordinates
            el_nodes = elements[el, 3:]
            for i, node_id in enumerate(el_nodes):
                y, x = int(nodes[node_id, 2]), int(nodes[node_id, 1])
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    # Accumulate stress components
                    stress_tensor[y, x, 0, 0] += nodal_stresses[i, 0]  # σxx
                    stress_tensor[y, x, 1, 1] += nodal_stresses[i, 1]  # σyy
                    stress_tensor[y, x, 0, 1] += nodal_stresses[i, 2]  # σxy
                    stress_tensor[y, x, 1, 0] += nodal_stresses[i, 2]  # σyx = σxy
                    stress_counts[y, x] += 1

        # Average stresses where multiple elements contribute
        valid_points = stress_counts > 0
        for i in range(2):
            for j in range(2):
                stress_tensor[valid_points, i, j] /= stress_counts[valid_points]

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
        """Solver using Numba-optimized PCG"""
        neq = KG.shape[0]

        # Get node positions and set up constraints
        nodes_xy, x_points, y_points = self._find_eq_position(nodes, IBC, neq)

        # Calculate center of mass
        com = regionprops(mask.astype(int))[0].centroid

        # Prepare constraint matrix
        constraint_data, constraint_rows, constraint_cols = prepare_constraint_data_numba(
            nodes_xy, x_points, y_points, com, neq)

        # Create sparse constraint matrix
        constraints = csr_matrix(
            (constraint_data, (constraint_rows, constraint_cols)),
            shape=(3, neq)
        )

        # Stack matrices
        KG_constrained = vstack([KG, constraints], format="csr")
        RHSG_constrained = np.append(RHSG, np.zeros(3))

        # Convert to CSR format and extract data
        KG_csr = KG_constrained.tocsr()

        # Create diagonal preconditioner
        diag = KG_csr.diagonal()
        M_inv = 1.0 / (diag + 1e-12)  # Add small value to prevent division by zero

        # Initial guess
        x0 = np.zeros_like(RHSG_constrained)

        # Solve using PCG
        solution, n_iter = pcg_solver_numba(
            KG_csr.data, KG_csr.indices, KG_csr.indptr,
            RHSG_constrained, x0, M_inv,
            tol=1e-12,
            max_iter=200000
        )

        return solution


# Example usage
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import tifffile
    import numpy as np

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
    nodes, elements, loads, mats = msm._grid_setup(mask, -t_x, -t_y)
    print("Elements shape:", elements.shape)
    print("Sample element node connectivity:", elements[0, 3:])

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


    def compare_stress_tensors(stress_tensor1, stress_tensor2, name1="Optimized", name2="Original"):
        """
        Compare two stress tensors and print detailed statistics
        """

        def compute_stats(tensor):
            return {
                'max': np.nanmax(tensor),
                'min': np.nanmin(tensor),
                'mean': np.nanmean(tensor),
                'std': np.nanstd(tensor),
                'median': np.nanmedian(tensor)
            }

        # Compare each component
        components = [
            ('σxx', (0, 0)),
            ('σyy', (1, 1)),
            ('σxy', (0, 1))
        ]

        print("\nDetailed Stress Tensor Comparison:")
        print("-" * 80)
        print(f"{'Component':<10} {'Metric':<10} {name1:<15} {name2:<15} {'Diff':<15} {'Rel Diff %':<10}")
        print("-" * 80)

        for comp_name, indices in components:
            tensor1 = stress_tensor1[:, :, indices[0], indices[1]]
            tensor2 = stress_tensor2[:, :, indices[0], indices[1]]

            stats1 = compute_stats(tensor1)
            stats2 = compute_stats(tensor2)

            for metric in ['max', 'min', 'mean', 'median', 'std']:
                val1 = stats1[metric]
                val2 = stats2[metric]
                diff = val2 - val1
                rel_diff = (diff / val1 * 100) if val1 != 0 else float('inf')

                print(f"{comp_name:<10} {metric:<10} {val1:<15.3e} {val2:<15.3e} {diff:<15.3e} {rel_diff:<10.2f}")

        # Compute overall error metrics
        valid_mask = ~np.isnan(stress_tensor1) & ~np.isnan(stress_tensor2)
        rmse = np.sqrt(np.mean((stress_tensor1[valid_mask] - stress_tensor2[valid_mask]) ** 2))
        max_abs_error = np.max(np.abs(stress_tensor1[valid_mask] - stress_tensor2[valid_mask]))

        print("\nOverall Error Metrics:")
        print(f"RMSE: {rmse:.3e}")
        print(f"Maximum Absolute Error: {max_abs_error:.3e}")

    compare_stress_tensors(stress_tensor, stress_tensor_ref)
