import numpy as np
from numba import jit


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
    r, s = 1 / 3, 1 / 3
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
