import numpy as np
from numba import jit


@jit(nopython=True)
def numba_assembler_core(elements, mats, nodes, neq, assem_op):
    """Numba-accelerated core assembly operations for triangular finite elements.

    Efficiently assembles global stiffness and mass matrices for 2D linear triangular
    elements using Numba for performance optimization.

    Args:
        elements (np.ndarray): Element connectivity array (N x 6)
            columns: [element_id, element_type, material_id, node1, node2, node3]
        mats (np.ndarray): Material properties array
            columns: [Young's modulus, Poisson ratio]
        nodes (np.ndarray): Node coordinates array (M x 5)
            columns: [node_id, x, y, bc_flag_x, bc_flag_y]
        neq (int): Number of equations (degrees of freedom)
        assem_op (np.ndarray): Assembly operator array mapping local to global DOFs

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - rows: Row indices for sparse matrix assembly
            - cols: Column indices for sparse matrix assembly
            - stiff_vals: Values for stiffness matrix
            - mass_vals: Values for mass matrix

    Notes:
        This function is optimized for triangular elements with 6 DOFs per element
        (2 DOFs per node × 3 nodes). The assembly process follows the standard FEM
        procedure but uses pre-allocated arrays and direct indexing for efficiency.
    """
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
    """Prepare constraint data for FEM system with zero-displacement and torque constraints.

    Efficiently constructs constraint equations for:
    1. Zero mean displacement in x and y directions
    2. Zero net torque around the center of mass

    Args:
        nodes_xy (np.ndarray): Node coordinates array (N x 2)
        x_points (np.ndarray): Boolean mask for x DOFs
        y_points (np.ndarray): Boolean mask for y DOFs
        com (tuple): Center of mass coordinates (x, y)
        neq (int): Number of equations (degrees of freedom)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - constraint_data: Values for constraint matrix
            - constraint_rows: Row indices for sparse constraint matrix
            - constraint_cols: Column indices for sparse constraint matrix

    Notes:
        The constraints ensure physical validity of the solution by enforcing:
        - No rigid body translation (zero mean displacement)
        - No rigid body rotation (zero net torque)
        These constraints are essential for obtaining a unique solution in MSM.
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
def calculate_element_stresses(el, elements, nodes, UC, mats):
    """Calculate stress components for a triangular finite element.

    Computes stresses at element nodes using linear shape functions and the
    plane stress constitutive relationship.

    Args:
        el (int): Element index
        elements (np.ndarray): Element connectivity array
        nodes (np.ndarray): Node coordinates array
        UC (np.ndarray): Nodal displacement solution array
        mats (np.ndarray): Material properties array

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - stresses: Stress tensor components [σxx, σyy, σxy] at nodes
            - natural_coords: Natural coordinates of element nodes
            - el_coords: Physical coordinates of element nodes

    Notes:
        For linear triangular elements, stress is constant within each element.
        The stress tensor is computed using:
        σ = D * B * u
        where:
        - D is the constitutive matrix (plane stress)
        - B is the strain-displacement matrix
        - u is the nodal displacement vector
    """
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
    r, s = 1 / 3, 1 / 3
    N, B, det = elast_diff_tri_numba(r, s, el_coords)

    # Natural coordinates for triangle (same for all cases)
    natural_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

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
def shape_tri_numba(r, s):
    """Calculate shape functions and derivatives for linear triangular element.

    Computes the shape functions and their derivatives with respect to natural
    coordinates (r,s) for a linear triangular element.

    Args:
        r (float): First natural coordinate (0 ≤ r ≤ 1)
        s (float): Second natural coordinate (0 ≤ s ≤ 1, r + s ≤ 1)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - N: Shape functions [N1, N2, N3]
            - dN: Shape function derivatives [∂N/∂r, ∂N/∂s]

    Notes:
        The shape functions for linear triangles are:
        N1 = 1 - r - s
        N2 = r
        N3 = s
        These provide linear interpolation within the element.
    """
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
    """Calculate matrices for 2D elasticity using linear triangular elements.

    Computes the displacement interpolation (H) and strain-displacement (B)
    matrices for a linear triangular element in 2D elasticity.

    Args:
        r (float): First natural coordinate
        s (float): Second natural coordinate
        coord (np.ndarray): Physical coordinates of element nodes (3 x 2)

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - H: Displacement interpolation matrix
            - B: Strain-displacement matrix
            - det: Determinant of Jacobian matrix

    Notes:
        The matrices are computed at the point (r,s) in natural coordinates:
        - H matrix relates nodal displacements to displacements at any point
        - B matrix relates nodal displacements to strains at any point
        - Jacobian determinant gives the element area scaling factor
    """
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
    """Calculate stiffness and mass matrices for elastic triangular element.

    Computes element matrices for 2D plane stress elasticity using linear
    triangular elements.

    Args:
        coord (np.ndarray): Physical coordinates of element nodes (3 x 2)
        params (np.ndarray): Material parameters
            - params[0]: Young's modulus (E)
            - params[1]: Poisson's ratio (ν)
            - params[2]: Density (ρ) (optional, default=1.0)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - K: Element stiffness matrix (6 x 6)
            - M: Element mass matrix (6 x 6)

    Notes:
        The stiffness matrix is computed using:
        K = ∫ B^T D B dA = A * B^T D B
        where:
        - A is the element area
        - B is the strain-displacement matrix
        - D is the constitutive matrix for plane stress

        For efficiency, the integral is evaluated using one-point quadrature
        at the element centroid (r=1/3, s=1/3).
    """
    E = float(params[0])  # Ensure float conversion
    nu = float(params[1])
    dens = 1.0 if len(params) <= 2 else float(params[2])

    # Print actual values during first call
    # if coord[0, 0] == 0:  # Only print for first element
    #     print(f"Element calculation with E={E}, nu={nu}")

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
