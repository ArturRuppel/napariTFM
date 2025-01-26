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

import solidspy.assemutil as ass
import solidspy.postprocesor as pos
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, vstack, diags
from scipy.sparse.linalg import lsqr
from scipy.ndimage import binary_fill_holes, generate_binary_structure, binary_dilation, label, gaussian_filter, sum as ndimage_sum

from skimage.measure import regionprops
from skimage.transform import resize

from napariTFM.backend.mesh_generator import MeshParameters, MeshGenerator
from napariTFM.backend.msm_numba_functions import *



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
            # print(f"{func_name} took {execution_time:.4f} seconds to execute")

        # Decrement nested call count
        self._nested_calls[func_name] -= 1

        # Restore parent function
        self._current_func = parent_func

        return result

    return wrapper


class MonolayerStressMicroscopy:
    def __init__(self, mask, density_factor=0.02, algorithm=2,
                 use_optimization=False, poisson_ratio=0.5,
                 young_modulus=1, nodes=None, elements=None):
        """Initialize with optional pre-generated nodes/elements"""
        self.mask = mask
        self.poisson_ratio = poisson_ratio
        self.E = young_modulus
        self.timing_stats = {}
        self._nested_calls = {}

        if nodes is None or elements is None:
            # Fallback to generating mesh if not provided
            mesh_params = MeshParameters(
                mask=mask,
                density_factor=density_factor,
                algorithm=algorithm,
                use_optimization=use_optimization
            )
            self.mesh_generator = MeshGenerator(mesh_params)
            self.nodes, self.elements = self.mesh_generator.generate_mesh(mask)
        else:
            self.nodes = nodes
            self.elements = elements

    @staticmethod
    def create_mask_from_image(image, threshold_percentile=0, dilation=10, smoothing_sigma=10.0):
        """
        Create a binary mask from an input image using thresholding, dilation, and smoothing.

        Parameters
        ----------
        image : np.ndarray
            Input image
        threshold_percentile : float
            Percentile value for thresholding (0-100)
        dilation : int
            Number of pixels to dilate the mask
        smoothing_sigma : float
            Sigma value for Gaussian smoothing

        Returns
        -------
        np.ndarray
            Binary mask
        """
        # Convert image to float for consistent processing
        image_float = image.astype(float)

        # Calculate threshold value based on percentile
        # Ignore zero values when calculating percentile
        if threshold_percentile > 0:
            nonzero_mask = image_float > 0
            if np.any(nonzero_mask):
                threshold_value = np.percentile(image_float[nonzero_mask], threshold_percentile)
                thresholded_image = np.where(image_float > threshold_value, image_float, 0)
            else:
                thresholded_image = image_float
        else:
            thresholded_image = image_float

        # Basic thresholding to create binary mask
        mask = thresholded_image > 0

        # Fill holes
        filled_mask = binary_fill_holes(mask)

        # Smoothing
        if smoothing_sigma > 0:
            float_mask = filled_mask.astype(float)
            smoothed = gaussian_filter(float_mask, sigma=smoothing_sigma)
            smoothed_mask = smoothed > 0.5
            smoothed_mask = binary_fill_holes(smoothed_mask)
        else:
            smoothed_mask = filled_mask

        # Dilation
        if dilation > 0:
            struct = generate_binary_structure(2, 2)
            dilated_mask = binary_dilation(
                smoothed_mask,
                structure=struct,
                iterations=dilation
            )
        else:
            dilated_mask = smoothed_mask

        # Get largest connected component
        labels, num_features = label(dilated_mask)
        if num_features > 0:
            # Calculate sizes of each labeled region
            sizes = ndimage_sum(dilated_mask, labels, index=range(1, num_features + 1))
            largest_feature = np.argmax(sizes) + 1
            final_mask = labels == largest_feature
        else:
            final_mask = dilated_mask

        return final_mask

    @classmethod
    def create_mask_stack(cls, image_stack, threshold_percentile=0, dilation=10, smoothing_sigma=10.0,
                          target_shape=None, downscale_factor=1):
        """
        Create a stack of masks from an image stack with optional resizing.

        Parameters
        ----------
        image_stack : np.ndarray
            3D array of images or 2D single image
        threshold_percentile : float
            Percentile value for thresholding (0-100)
        dilation : int
            Number of pixels to dilate each mask
        smoothing_sigma : float
            Sigma value for Gaussian smoothing
        target_shape : tuple, optional
            Shape to resize masks to (height, width)
        downscale_factor : int, optional
            Factor to downscale masks by for analysis

        Returns
        -------
        tuple
            (analysis_mask_stack, visualization_mask_stack)
        """
        # Handle 2D input
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]

        # Create mask stack
        mask_stack = np.zeros_like(image_stack, dtype=bool)

        # Process each frame
        for frame in range(image_stack.shape[0]):
            mask_stack[frame] = cls.create_mask_from_image(
                image_stack[frame],
                threshold_percentile=threshold_percentile,
                dilation=dilation,
                smoothing_sigma=smoothing_sigma
            )

        # Initialize both stacks as the original
        analysis_mask_stack = mask_stack
        vis_mask_stack = mask_stack

        # Resize analysis masks if target shape is provided
        if target_shape is not None and mask_stack.shape[1:] != target_shape:
            analysis_mask_stack = resize(
                mask_stack.astype(float),
                (mask_stack.shape[0], *target_shape),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5

        # Create visualization masks if downscale factor is greater than 1
        if downscale_factor > 1:
            vis_shape = (
                analysis_mask_stack.shape[0],
                analysis_mask_stack.shape[1] * downscale_factor,
                analysis_mask_stack.shape[2] * downscale_factor
            )
            vis_mask_stack = resize(
                mask_stack.astype(float),
                vis_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5

        return analysis_mask_stack, vis_mask_stack

    def _grid_setup(self, nodes_xy, elements, f_x, f_y):
        """Setup triangular mesh with linear force interpolation and proper scaling"""
        # Convert to required format
        num_nodes = len(nodes_xy)
        nodes = np.zeros((num_nodes, 5))
        nodes[:, 0] = np.arange(num_nodes)  # node numbers
        nodes[:, 1:3] = nodes_xy
        nodes[:, 3:] = 0  # BC flags

        # Convert element connectivity
        num_elements = len(elements)
        elements_formatted = np.zeros((num_elements, 6), dtype=np.int64)
        elements_formatted[:, 0] = np.arange(num_elements)
        elements_formatted[:, 1] = 1  # element type (1 for triangle)
        elements_formatted[:, 2] = 0  # material number
        elements_formatted[:, 3:] = elements

        # Calculate nodal tributary areas
        node_areas = np.zeros(num_nodes)
        for el in range(num_elements):
            el_nodes = elements[el]
            # Calculate element area
            x = nodes_xy[el_nodes, 0]
            y = nodes_xy[el_nodes, 1]
            area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
            # Distribute area to nodes (1/3 to each node)
            node_areas[el_nodes] += area / 3.0

        # Setup loads with area-weighted interpolation
        loads = np.zeros((num_nodes, 3))
        loads[:, 0] = np.arange(num_nodes)

        # Create grid points for input forces
        grid_y, grid_x = np.mgrid[0:f_x.shape[0], 0:f_x.shape[1]]

        # Get valid force points
        valid_mask_x = ~np.isnan(f_x)
        valid_mask_y = ~np.isnan(f_y)

        # Points for interpolation
        points_x = np.column_stack((grid_x[valid_mask_x], grid_y[valid_mask_x]))
        values_x = f_x[valid_mask_x]
        points_y = np.column_stack((grid_x[valid_mask_y], grid_y[valid_mask_y]))
        values_y = f_y[valid_mask_y]

        # Create interpolators
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        interp_x = LinearNDInterpolator(points_x, values_x)
        interp_y = LinearNDInterpolator(points_y, values_y)
        nn_interp_x = NearestNDInterpolator(points_x, values_x)
        nn_interp_y = NearestNDInterpolator(points_y, values_y)

        # Interpolate forces to node positions with area weighting
        x_forces = interp_x(nodes[:, 1], nodes[:, 2])
        y_forces = interp_y(nodes[:, 1], nodes[:, 2])

        # Fill NaN values
        nan_mask_x = np.isnan(x_forces)
        nan_mask_y = np.isnan(y_forces)
        if np.any(nan_mask_x):
            x_forces[nan_mask_x] = nn_interp_x(nodes[nan_mask_x, 1], nodes[nan_mask_x, 2])
        if np.any(nan_mask_y):
            y_forces[nan_mask_y] = nn_interp_y(nodes[nan_mask_y, 1], nodes[nan_mask_y, 2])

        # Apply area weighting to forces
        loads[:, 1] = x_forces * node_areas
        loads[:, 2] = y_forces * node_areas

        # Setup materials
        mats = np.array([[self.E, self.poisson_ratio]])

        return nodes, elements_formatted, loads, mats

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

    def _custom_assembler(self, elements, mats, nodes, neq, assem_op):
        """Custom assembler for triangular elements"""
        rows, cols, stiff_vals, mass_vals = numba_assembler_core(elements, mats, nodes, neq, assem_op)

        stiff = csr_matrix((stiff_vals, (rows, cols)), shape=(neq, neq))
        mass = csr_matrix((mass_vals, (rows, cols)), shape=(neq, neq))

        return stiff, mass

    def _fem_simulation(self, nodes, elements, loads, mats, mask):
        """Main FEM simulation with simplified stress calculation and interpolation"""
        # System assembly
        DME, IBC, neq = self._assemble_custom_dme(nodes, elements)
        KG, MG = self._custom_assembler(elements, mats, nodes, neq, DME)
        RHSG = ass.loadasem(loads, IBC, neq)

        # Solve system and get metrics
        solver_output = self._custom_solver(KG, RHSG, mask, nodes, IBC)
        UG_sol = solver_output[0]  # Extract just the solution array
        condition_number = solver_output[1]
        residual = solver_output[2]

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
        stress_tensor = self._interpolate_stress_field(nodes, nodal_stresses, mask)

        return stress_tensor, condition_number, residual

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
        """Modified solver that also returns condition number and residual"""
        neq = KG.shape[0]
        nodes_xy, x_points, y_points = self._find_eq_position(nodes, IBC, neq)

        # Get constraint matrix
        com = regionprops(mask.astype(int))[0].centroid
        constraint_data, constraint_rows, constraint_cols = prepare_constraint_data_numba(
            nodes_xy, x_points, y_points, com, neq
        )
        constraints = csr_matrix(
            (constraint_data, (constraint_rows, constraint_cols)),
            shape=(3, neq)
        )

        # Create scaling based on diagonal of KG
        diag = np.array(KG.diagonal())
        scale = np.ones_like(diag)
        valid_diag = np.abs(diag) > 1e-10
        scale[valid_diag] = 1.0 / np.sqrt(np.abs(diag[valid_diag]))
        S = diags(scale)
        KG_scaled = KG.dot(S)

        # Stack scaled system
        constraint_scale = np.sqrt(np.sum(KG_scaled.data ** 2)) / np.sqrt(np.sum(constraints.data ** 2))
        KG_constrained = vstack([KG_scaled, constraints * constraint_scale], format="csr")
        RHSG_constrained = np.append(RHSG, np.zeros(3))

        # Estimate condition number (using power iteration for efficiency)
        def power_iteration(A, num_iterations=10):
            n = A.shape[0]
            v = np.random.rand(n)
            for _ in range(num_iterations):
                Av = A.dot(v)
                v_new = Av / np.linalg.norm(Av)
                if np.allclose(v, v_new):
                    break
                v = v_new
            return np.linalg.norm(A.dot(v)) / np.linalg.norm(v)

        AtA = KG_constrained.T.dot(KG_constrained)
        largest_eigval = power_iteration(AtA)
        smallest_eigval = 1 / power_iteration(csr_matrix(np.linalg.inv(AtA.toarray())))
        condition_number = np.sqrt(largest_eigval / smallest_eigval)

        # Solve system and get residual
        solution = lsqr(KG_constrained, RHSG_constrained,
                        atol=1e-16, btol=1e-16,
                        iter_lim=5000000,
                        show=False)
        x_scaled = solution[0]
        residual_norm = solution[3]  # Get the residual norm from LSQR output

        # Unscale solution
        UG_sol = S.dot(x_scaled[:neq])

        return UG_sol, condition_number, residual_norm

    def _prepare_forces(self, tx, ty, mask):
        """Convert traction forces to point forces while keeping spatial units in pixels"""
        # Keep the tractions as they are (in Pa) but work with pixel areas
        f_x = tx.copy()
        f_y = ty.copy()

        # Mask forces
        f_x[~mask] = np.nan
        f_y[~mask] = np.nan

        # Remove mean force (force balance)
        f_x = f_x - np.nanmean(f_x)
        f_y = f_y - np.nanmean(f_y)

        # Correct torque
        f_x, f_y = self._correct_torque(f_x, f_y, mask)

        # Debug output for force magnitudes
        # print(f"\nForce preparation:")
        # print(f"Traction range x: [{np.nanmin(f_x):.2e}, {np.nanmax(f_x):.2e}] Pa")
        # print(f"Traction range y: [{np.nanmin(f_y):.2e}, {np.nanmax(f_y):.2e}] Pa")

        return f_x, f_y

    def calculate_stress_field(self, traction_x, traction_y):
        """Calculate stress field and return quality metrics"""
        if np.all(np.isnan(traction_x)) or np.all(np.isnan(traction_y)):
            raise ValueError("Input tractions are all NaN")

        # Prepare forces
        f_x, f_y = self._prepare_forces(traction_x, traction_y, self.mask)

        # Format mesh for FEM solver
        nodes_formatted, elements_formatted, loads, mats = self._grid_setup(
            nodes_xy=self.nodes,
            elements=self.elements,
            f_x=-f_x,
            f_y=-f_y
        )

        # Calculate stress tensor and get metrics
        stress_tensor, condition_number, residual = self._fem_simulation(
            nodes_formatted, elements_formatted, loads, mats, self.mask
        )

        # Scale stress tensor
        stress_tensor = stress_tensor

        return stress_tensor, condition_number, residual
