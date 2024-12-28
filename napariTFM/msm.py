import solidspy.assemutil as ass
import solidspy.postprocesor as pos
import solidspy.solutil as sol
import solidspy.uelutil as ue

from numpy import vstack
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr
from skimage.measure import regionprops
from skimage.transform import downscale_local_mean


class MonolayerStressMicroscopy:
    def __init__(self, pixelsize, sigma=0.5, youngs_modulus=1):
        """
        Initialize MSM calculator
        Args:
            pixelsize: Pixel size in microns
            sigma: Poisson's ratio for the material
            youngs_modulus: Young's modulus for the material
        """
        self.pixelsize = pixelsize  # Store original pixelsize without conversion
        self.sigma = sigma
        self.E = youngs_modulus

    def calculate_stress_field(self, traction_x, traction_y, mask, downsample_factor=1):
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
        # Downsample if needed
        if downsample_factor > 1:
            mask = downscale_local_mean(mask, (downsample_factor, downsample_factor)) > 0.5
            traction_x = downscale_local_mean(traction_x, (downsample_factor, downsample_factor))
            traction_y = downscale_local_mean(traction_y, (downsample_factor, downsample_factor))

        # Calculate effective pixelsize
        forcemap_pixelsize = self.pixelsize * downsample_factor * 1e-6  # Convert to meters

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

    def _custom_assembler(self, elements, mats, nodes, neq, assem_op):
        """
        Custom assembler that ensures proper coordinate handling

        Parameters
        ----------
        elements : ndarray (int)
            Array with the number for the nodes in each element
        mats : ndarray (float)
            Array with material profiles
        nodes : ndarray (float)
            Array with the nodal numbers and coordinates
        neq : int
            Number of active equations in the system
        assem_op : ndarray (int)
            Assembly operator

        Returns
        -------
        stiff : csr_matrix
            Global stiffness matrix
        mass : csr_matrix
            Global mass matrix
        """
        rows = []
        cols = []
        stiff_vals = []
        mass_vals = []
        nels = elements.shape[0]

        for ele in range(nels):
            # Extract only x,y coordinates for the current element
            elem_nodes = elements[ele, 3:]
            elcoor = nodes[elem_nodes, 1:3].astype(np.float64)  # Only get x,y coordinates

            # Get element parameters
            mat_id = elements[ele, 2]
            params = mats[mat_id, :]

            # Get element type and call appropriate element function
            elem_type = elements[ele, 1]
            if elem_type == 1:  # quad4 element
                kloc, mloc = ue.elast_quad4(elcoor, params)
            else:
                raise ValueError(f"Element type {elem_type} not supported")

            # Get DOFs for the element
            ndof = kloc.shape[0]
            dme = assem_op[ele, :ndof]

            # Assemble into global matrices
            for row in range(ndof):
                glob_row = dme[row]
                if glob_row != -1:
                    for col in range(ndof):
                        glob_col = dme[col]
                        if glob_col != -1:
                            rows.append(glob_row)
                            cols.append(glob_col)
                            stiff_vals.append(kloc[row, col])
                            mass_vals.append(mloc[row, col])

        # Create sparse matrices
        from scipy.sparse import coo_matrix
        stiff = coo_matrix((stiff_vals, (rows, cols)), shape=(neq, neq)).tocsr()
        mass = coo_matrix((mass_vals, (rows, cols)), shape=(neq, neq)).tocsr()
        return stiff, mass

    def _fem_simulation(self, nodes, elements, loads, mats, mask):
        """Perform FEM simulation using modified approach"""
        # Get boundary conditions
        DME, IBC, neq = ass.DME(nodes[:, 3:], elements)

        # System assembly with custom assembler
        KG, MG = self._custom_assembler(elements, mats, nodes, neq, DME)  # Removed sparse=True
        RHSG = ass.loadasem(loads, IBC, neq)

        # Solve system
        if np.sum(IBC == -1) < 3:
            UG_sol = self._custom_solver(KG, RHSG, mask, nodes, IBC)
        else:
            UG_sol = sol.static_sol(KG, RHSG)

        # Calculate stresses
        UC = pos.complete_disp(IBC, nodes, UG_sol)

        # For stress calculation, ensure we use only x,y coordinates
        stress_nodes = nodes.copy()
        stress_nodes = stress_nodes[:, :3]  # Keep only node number and x,y coords
        E_nodes, S_nodes = pos.strain_nodes(stress_nodes, elements, mats, UC)

        # Assemble stress tensor
        stress_tensor = np.zeros((mask.shape[0], mask.shape[1], 2, 2))
        stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 0, 0] = S_nodes[:, 0]
        stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 1, 1] = S_nodes[:, 1]
        stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 0, 1] = S_nodes[:, 2]
        stress_tensor[nodes[:, 2].astype(int), nodes[:, 1].astype(int), 1, 0] = S_nodes[:, 2]

        return stress_tensor

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
        """Custom solver with zero displacement/rotation constraints"""
        neq = KG.shape[0]
        zero_disp_x = np.zeros(neq)
        zero_disp_y = np.zeros(neq)
        zero_torque = np.zeros(neq)

        # Get node positions relative to center of mass
        com = regionprops(mask.astype(int))[0].centroid
        nodes_xy, x_points, y_points = self._find_eq_position(nodes, IBC, neq)

        # Calculate positions relative to center of mass
        r = np.zeros((neq, 2))
        r[x_points | y_points] = nodes_xy[x_points | y_points] - [com[1], com[0]]

        # Set up constraints
        zero_disp_x[x_points] = 1
        zero_disp_y[y_points] = 1

        # Set up torque constraints
        zero_torque[x_points] = r[x_points, 1]  # -r2 factor for x DOFs
        zero_torque[y_points] = -r[y_points, 0]  # r1 factor for y DOFs

        # Reshape arrays to 2D
        zero_disp_x = zero_disp_x.reshape(1, -1)
        zero_disp_y = zero_disp_y.reshape(1, -1)
        zero_torque = zero_torque.reshape(1, -1)

        # Add constraints to system
        if isinstance(KG, csr_matrix):
            # Convert to sparse and stack
            add_matrix = vstack([
                csr_matrix(zero_disp_x),
                csr_matrix(zero_disp_y),
                csr_matrix(zero_torque)
            ])
            KG = vstack([KG, add_matrix], format="csr")
        else:
            # Dense matrix case
            add_matrix = np.vstack([zero_disp_x, zero_disp_y, zero_torque])
            KG = np.vstack([KG, add_matrix])

        # Add zero RHS for constraints
        RHSG = np.append(RHSG, np.zeros(3))

        # Solve constrained system
        UG_sol = lsqr(KG, RHSG, atol=1e-12, btol=1e-12, iter_lim=200000)[0]

        return UG_sol


# Example usage
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import tifffile
    import numpy as np

    # Load data
    data_path = r"C:\Users\aruppel\Desktop\test"

    # Load traction force data
    t_x = np.load(os.path.join(data_path, "t_x.npy"))[0, :, :]
    t_y = np.load(os.path.join(data_path, "t_y.npy"))[0, :, :]

    # Load reference stress tensor
    stress_tensor_ref = np.load(os.path.join(data_path, "stress_tensor_reference.npy"))[0, :, :, :, :]

    # Load mask
    mask = tifffile.imread(os.path.join(data_path, "masks.tif")).astype(bool)[0, :, :]

    # Initialize MSM calculator
    pixelsize = 0.864  # microns per pixel
    msm = MonolayerStressMicroscopy(pixelsize=pixelsize)
    nodes, elements, loads, mats = msm._grid_setup(mask, -t_x, -t_y)
    print("Elements shape:", elements.shape)
    print("Sample element node connectivity:", elements[0, 3:])

    # Calculate stress field
    stress_tensor = msm.calculate_stress_field(t_x, t_y, mask)

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

    # Create visualization with 3 rows (max principal, sigma_xx, sigma_yy)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # Function to plot stress component with consistent formatting
    def plot_stress(ax, data, title):
        vmax = 0.5 * np.nanmax(data)  # Scale to 0.5 * max value for each plot individually
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=vmax)
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