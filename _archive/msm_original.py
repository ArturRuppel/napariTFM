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
from functools import wraps
import os
import solidspy.assemutil as ass
import solidspy.postprocesor as pos
import solidspy.solutil as sol
from numba import jit
from numpy import vstack
import numpy as np
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
        """Optimized FEM simulation using solidspy functions"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        import solidspy.femutil as fem

        # Get mask dimensions for later use
        if isinstance(mask, np.ndarray):
            mask_shape = mask.shape
        else:
            print("Mask type:", type(mask))
            print("Mask content:", mask)
            raise ValueError("Mask must be a numpy array")

        start_dme = time.time()
        DME, IBC, neq = ass.DME(nodes[:, 3:], elements)
        self.timing_stats['dme_setup'] = time.time() - start_dme


        # Optimized system assembly
        start_assembly = time.time()
        KG, MG = self._custom_assembler(elements, mats, nodes, neq, DME)
        self.timing_stats['system_assembly'] = time.time() - start_assembly


        # Efficient load assembly
        start_load = time.time()
        RHSG = ass.loadasem(loads, IBC, neq)
        self.timing_stats['load_assembly'] = time.time() - start_load

        # Optimized solver selection
        start_solve = time.time()
        if np.sum(IBC == -1) < 3:
            UG_sol = self._custom_solver(KG, RHSG, mask, nodes, IBC)
        else:
            UG_sol = sol.static_sol(KG, RHSG)
        self.timing_stats['system_solve'] = time.time() - start_solve

        # Complete displacement calculation
        UC = pos.complete_disp(IBC, nodes, UG_sol)
        stress_nodes = nodes.copy()[:, :3]

        def calculate_element_strains(el):
            """Calculate strains and stresses for a single element"""
            el_nodes = elements[el, 3:]
            el_coords = nodes[el_nodes, 1:3].astype(np.float64)
            el_disps = UC[el_nodes]
            mat_id = elements[el, 2]
            params = mats[mat_id]

            # Calculate element stiffness (we only need the B matrix)
            r = s = 0.0  # Center point of element
            H, B, det = fem.elast_diff_2d(r, s, el_coords, fem.shape_quad4)

            # Calculate strains
            strain = np.dot(B, el_disps.flatten())

            # Calculate stresses using constitutive relationship
            E, nu = params
            fact = E / (1 - nu * nu)
            D = np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ]) * fact

            stress = np.dot(D, strain)
            return strain, stress

        # Process elements in parallel
        n_cores = os.cpu_count()
        batch_size = max(1, len(elements) // (4 * n_cores))
        element_batches = np.array_split(np.arange(len(elements)), batch_size)

        E_elements = []
        S_elements = []
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = []
            for batch in element_batches:
                for el in batch:
                    futures.append(executor.submit(calculate_element_strains, el))

            for future in futures:
                strain, stress = future.result()
                E_elements.append(strain)
                S_elements.append(stress)

        # Convert to numpy arrays
        E_elements = np.array(E_elements)
        S_elements = np.array(S_elements)

        # Compute nodal averages
        num_nodes = len(nodes)
        E_nodes = np.zeros((num_nodes, 3))
        S_nodes = np.zeros((num_nodes, 3))
        node_counts = np.zeros(num_nodes)

        # Accumulate element contributions to nodes
        for el, (E_el, S_el) in enumerate(zip(E_elements, S_elements)):
            node_indices = elements[el, 3:]
            for node_idx in node_indices:
                E_nodes[node_idx] += E_el
                S_nodes[node_idx] += S_el
                node_counts[node_idx] += 1

        # Average the accumulated values
        valid_nodes = node_counts > 0
        E_nodes[valid_nodes] /= node_counts[valid_nodes, np.newaxis]
        S_nodes[valid_nodes] /= node_counts[valid_nodes, np.newaxis]

        # Create stress tensor with proper dimensions
        stress_tensor = np.zeros((*mask_shape, 2, 2))

        # Get y and x coordinates for node mapping
        y_coords = nodes[:, 2].astype(int)  # Row indices
        x_coords = nodes[:, 1].astype(int)  # Column indices

        # Validate indices
        valid_indices = (y_coords >= 0) & (y_coords < mask_shape[0]) & \
                        (x_coords >= 0) & (x_coords < mask_shape[1])

        # Assign stress components only for valid indices
        valid_y = y_coords[valid_indices]
        valid_x = x_coords[valid_indices]
        valid_s = S_nodes[valid_indices]

        stress_tensor[valid_y, valid_x, 0, 0] = valid_s[:, 0]  # σxx
        stress_tensor[valid_y, valid_x, 1, 1] = valid_s[:, 1]  # σyy
        stress_tensor[valid_y, valid_x, 0, 1] = valid_s[:, 2]  # σxy
        stress_tensor[valid_y, valid_x, 1, 0] = valid_s[:, 2]  # σyx = σxy

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
        """Solver optimized with Numba acceleration for constraint handling"""
        neq = KG.shape[0]

        # Get node positions and set up constraints
        nodes_xy, x_points, y_points = self._find_eq_position(nodes, IBC, neq)

        # Calculate center of mass
        com = regionprops(mask.astype(int))[0].centroid

        # Use Numba-accelerated function to prepare constraint data
        constraint_data, constraint_rows, constraint_cols = prepare_constraint_data_numba(
            nodes_xy, x_points, y_points, com, neq)

        # Create sparse constraint matrix directly
        constraints = csr_matrix(
            (constraint_data, (constraint_rows, constraint_cols)),
            shape=(3, neq)
        )

        # Stack matrices efficiently
        KG_constrained = vstack([KG, constraints], format="csr")
        RHSG_constrained = np.append(RHSG, np.zeros(3))

        # Use LSQR with optimized parameters
        UG_sol = lsqr(KG_constrained, RHSG_constrained,
                      atol=1e-12,
                      btol=1e-12,
                      iter_lim=200000,
                      show=False)[0]

        return UG_sol


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

    # Print max values for comparison
    print(f"Max calculated principal stress: {np.nanmax(sigma_max):.2f} Pa")
    print(f"Max reference principal stress: {np.nanmax(sigma_max_ref):.2f} Pa")
    print(f"\nMax calculated σxx: {np.nanmax(sigma_xx):.2f} Pa")
    print(f"Max reference σxx: {np.nanmax(sigma_xx_ref):.2f} Pa")
    print(f"\nMax calculated σyy: {np.nanmax(sigma_yy):.2f} Pa")
    print(f"Max reference σyy: {np.nanmax(sigma_yy_ref):.2f} Pa")