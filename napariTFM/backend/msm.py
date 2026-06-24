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
and triangular instead of quadratic meshing while maintaining the accuracy of the
original MSM method. The solver employs an optimized LSQR implementation with careful
constraint handling for solving the resulting system of equations.

Masks are supplied externally (napariTFM does not generate them); the
implementation provides methods for:
1. Generating optimized FEM meshes from a supplied binary mask
2. Calculating stress fields from traction forces
3. Handling boundary conditions and numerical stability
"""

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple
import warnings

import numpy as np
import solidspy.assemutil as ass
import solidspy.postprocesor as pos
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, vstack, diags
from scipy.sparse.linalg import lsqr
from skimage.measure import regionprops
from skimage.transform import resize

from napariTFM.backend.mesh_generator import MeshGenerator
from napariTFM.backend.msm_numba_functions import *
from napariTFM.backend.parameter_dataclasses import MeshParameters, MSMParameters
from napariTFM.backend.parameter_validation import validate_msm_parameters


@dataclass
class MSMResult:
    """Results from Monolayer Stress Microscopy calculations."""
    stress_tensor: np.ndarray
    nodes: List[np.ndarray]
    elements: List[np.ndarray]
    condition_number: float
    residual: float
    parameters: MSMParameters
    physical_scale: dict
    original_shape: tuple
    stress_shape: tuple


def process_mask_data(
        mask_data: np.ndarray,
        force_field: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[str]]:
    """Convert mask data to a validated boolean stack, resizing to force shape when needed."""
    warnings_list = []

    if mask_data is None:
        raise ValueError("No mask data provided")

    unique_values = np.unique(mask_data)
    unique_values = unique_values[unique_values != 0]
    if len(unique_values) > 1:
        warnings_list.append("Multiple non-zero values detected in mask. Converting to binary (0 and 1).")

    mask_data = mask_data > 0

    if mask_data.ndim == 2:
        mask_data = mask_data[np.newaxis, ...]

    if mask_data.ndim != 3:
        raise ValueError(f"Mask data must be 2D or 3D, got shape {mask_data.shape}")

    if force_field is not None:
        force_shape = force_field.shape[1:3]
        if mask_data.shape[1:] != force_shape:
            mask_data = np.stack([
                resize(
                    frame.astype(float),
                    force_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5
                for frame in mask_data
            ])

    return mask_data, warnings_list


def generate_mesh_stack(
        mask_stack: np.ndarray,
        params: MSMParameters,
) -> Generator[Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int], None, List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]]:
    """Generate finite-element meshes for each mask frame."""
    is_valid, error_msg = validate_msm_parameters(params)
    if not is_valid:
        raise ValueError(error_msg)

    if mask_stack.ndim == 2:
        mask_stack = mask_stack[np.newaxis, ...]

    total_frames = mask_stack.shape[0]
    mesh_params = MeshParameters(
        mask=mask_stack[0],
        density_factor=params.density_factor,
        mesh_algorithm=_get_algorithm_code(params.mesh_algorithm),
        use_optimization=params.use_optimization
    )
    mesh_generator = MeshGenerator(mesh_params)
    mesh_results = []

    for frame in range(total_frames):
        nodes, elements = mesh_generator.generate_mesh(mask_stack[frame])
        quality_metrics = mesh_generator.analyze_mesh_quality(nodes, elements)
        mesh_results.append((nodes, elements, quality_metrics))
        yield nodes, elements, quality_metrics, frame, total_frames

    return mesh_results


def calculate_stresses(
        force_field: np.ndarray,
        masks: np.ndarray,
        params: MSMParameters,
        mesh_data: Optional[List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]] = None
) -> Generator[Tuple[MSMResult, int, int], None, MSMResult]:
    """Calculate MSM stress fields from traction forces and masks."""
    is_valid, error_msg = validate_msm_parameters(params)
    if not is_valid:
        raise ValueError(error_msg)

    if force_field.ndim == 3:
        force_field = force_field[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    total_frames = force_field.shape[0]
    stress_shape = (*force_field.shape[1:3], 2, 2)
    stress_stack = np.zeros((total_frames, *stress_shape), dtype=np.float32)
    nodes_stack = []
    elements_stack = []
    condition_numbers = []
    residuals = []
    physical_scale = _create_physical_scale(params)
    analyzer = MonolayerStressMicroscopy(params, mask=None)

    def process_frame(frame_idx):
        tx = force_field[frame_idx, ..., 0]
        ty = force_field[frame_idx, ..., 1]
        current_mask = masks[frame_idx]

        if current_mask.shape != tx.shape:
            current_mask = resize(
                current_mask.astype(float),
                tx.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5

        if mesh_data is not None:
            nodes, elements, _ = mesh_data[frame_idx]
        else:
            mesh_params = MeshParameters(
                mask=current_mask,
                density_factor=params.density_factor,
                mesh_algorithm=_get_algorithm_code(params.mesh_algorithm),
                use_optimization=params.use_optimization
            )
            mesh_generator = MeshGenerator(mesh_params)
            nodes, elements = mesh_generator.generate_mesh(current_mask)

        analyzer.mask = current_mask
        analyzer.nodes = nodes
        analyzer.elements = elements

        stress_tensor, condition_number, residual = analyzer.calculate_stress_field(tx, ty)
        stress_tensor *= params.downscale_factor * params.pixel_size * 1e-3

        return stress_tensor, nodes, elements, condition_number, residual

    for frame in range(total_frames):
        stress_tensor, nodes, elements, condition_number, residual = process_frame(frame)

        stress_stack[frame] = stress_tensor
        nodes_stack.append(nodes)
        elements_stack.append(elements)
        condition_numbers.append(condition_number)
        residuals.append(residual)

        yield MSMResult(
            stress_tensor=stress_stack[:frame + 1],
            nodes=nodes,
            elements=elements,
            condition_number=condition_number,
            residual=residual,
            parameters=params,
            physical_scale=physical_scale,
            original_shape=force_field.shape[1:3],
            stress_shape=stress_stack.shape[1:3]
        ), frame + 1, total_frames

    return MSMResult(
        stress_tensor=stress_stack,
        nodes=nodes_stack,
        elements=elements_stack,
        condition_number=np.mean(np.stack(condition_numbers)),
        residual=np.mean(residuals),
        parameters=params,
        physical_scale=physical_scale,
        original_shape=force_field.shape[1:3],
        stress_shape=stress_stack.shape[1:3]
    )


def _create_physical_scale(params: MSMParameters) -> dict:
    """Create physical scale information for MSM results."""
    return {
        'pixel_size': params.pixel_size,
        'grid_spacing': params.pixel_size * params.downscale_factor,
        'time_interval': params.frame_interval,
        'stress_units': 'mN/m',
        'grid_spacing_units': 'µm',
        'time_interval_units': 'min',
    }


def _get_algorithm_code(algorithm_name: str) -> int:
    """Convert configured mesh algorithm name to the gmsh algorithm code."""
    algorithm_map = {
        "frontal-del": 6,
        "delaunay": 5,
        "meshadapt": 1,
        "bamg": 7,
        "fd quads": 8,
        "para pack": 9
    }
    normalized_name = algorithm_name.lower().replace(".", "").replace("-", " ")
    return algorithm_map.get(normalized_name, 6)


class MonolayerStressMicroscopy:
    def __init__(self, params, mask=None, nodes=None, elements=None):
        """Initialize MonolayerStressMicroscopy calculator with material properties and mesh parameters.

        Args:
            params (MSMParameters): Configuration object containing:
                - young_modulus (float): Young's modulus of the cell monolayer in Pascals (Pa)
                - poisson_ratio_cells (float): Poisson ratio of the cell monolayer
                - density_factor (float): Controls mesh density (typical range: 0.005-0.05)
                - mesh_algorithm (str): One of ['frontal-del', 'delaunay', 'meshadapt',
                    'bamg', 'fd quads', 'para pack']
                - use_optimization (bool): Whether to optimize mesh for better numerical stability
            mask (np.ndarray, optional): Binary mask defining the cell monolayer region
            nodes (np.ndarray, optional): Pre-generated mesh nodes if available
            elements (np.ndarray, optional): Pre-generated mesh elements if available

        Example:
            >>> msm = MonolayerStressMicroscopy(MSMParameters(
            ...     young_modulus=1000,  # 1 kPa
            ...     poisson_ratio_cells=0.5,
            ...     density_factor=0.02,
            ...     mesh_algorithm='frontal-del',
            ...     use_optimization=True
            ... ))
        """

        self.poisson_ratio = params.poisson_ratio_cells
        self.E = params.young_modulus
        self.density_factor = params.density_factor
        self.mesh_algorithm = _get_algorithm_code(params.mesh_algorithm)
        self.use_optimization = params.use_optimization

        self.mask = mask
        self.nodes = nodes
        self.elements = elements

        # Initialize timing statistics
        self.timing_stats = {}
        self._nested_calls = {}

    def calculate_stress_field(self, traction_x, traction_y):
        """Calculate stress field from traction force measurements using FEM.

        This method implements the monolayer stress microscopy algorithm to compute
        internal stresses within a cell monolayer from measured traction forces.
        The calculation involves:
        1. Force preparation and balance correction
        2. Mesh generation (if not provided)
        3. FEM system assembly
        4. Constraint handling and numerical stabilization
        5. Solution of the mechanical equilibrium equations

        Args:
            traction_x (np.ndarray): X-component of traction forces (shape: H x W)
                Units: Pascals (Pa)
            traction_y (np.ndarray): Y-component of traction forces (shape: H x W)
                Units: Pascals (Pa)
                The traction fields should represent forces exerted by the cells
                on the substrate at each point.

        Returns:
            Tuple[np.ndarray, float, float]:
                - stress_tensor (np.ndarray): Calculated stress tensor (shape: H x W x 2 x 2)
                  Units: Pascals (Pa)
                  Components are:
                  - [..., 0, 0]: σxx (normal stress in x)
                  - [..., 1, 1]: σyy (normal stress in y)
                  - [..., 0, 1] and [..., 1, 0]: σxy (shear stress)
                - condition_number (float): Condition number of the system matrix,
                  indicating numerical stability
                - residual (float): Residual norm of the solution, indicating
                  solution accuracy

        Notes:
            The stress tensor is symmetric (σxy = σyx) and represents the internal
            stresses within the cell monolayer. These stresses arise from the
            mechanical equilibrium between cell-generated forces and measured
            traction forces.

        Example:
            >>> msm = MonolayerStressMicroscopy(params)
            >>> tx, ty = measure_traction_forces()  # Your measurement function
            >>> stress_tensor, cond, res = msm.calculate_stress_field(tx, ty)
            >>> # Access normal and shear stresses
            >>> sigma_xx = stress_tensor[..., 0, 0]
            >>> sigma_yy = stress_tensor[..., 1, 1]
            >>> sigma_xy = stress_tensor[..., 0, 1]
        """
        try:
            if self.nodes is None or self.elements is None:
                # Validate density_factor before mesh generation
                if self.density_factor < 0.005:
                    warnings.warn(
                        "Density factor is very low (< 0.005), which may lead to numerical instabilities and long runtimes.",
                        UserWarning
                    )

                # Proceed with mesh generation
                mesh_params = MeshParameters(
                    mask=self.mask,
                    density_factor=self.density_factor,
                    mesh_algorithm=self.mesh_algorithm,
                    use_optimization=self.use_optimization
                )
                self.mesh_generator = MeshGenerator(mesh_params)
                self.nodes, self.elements = self.mesh_generator.generate_mesh(self.mask)

            if np.all(np.isnan(traction_x)) or np.all(np.isnan(traction_y)):
                raise ValueError("Input tractions are all NaN")

            # Prepare forces
            self.mask = self.mask.astype(bool)
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

            # Check if stress tensor is invalid (all zeros might indicate failure)
            if np.all(stress_tensor == 0):
                warnings.warn("Solver returned zero stress tensor. Check inputs and mesh.",
                              RuntimeWarning)

            return stress_tensor, condition_number, residual

        except Exception as e:
            warnings.warn(f"Error in stress calculation: {str(e)}. Returning zero stress tensor.",
                          RuntimeWarning)
            stress_shape = (*self.mask.shape, 2, 2)
            zero_stress = np.zeros(stress_shape)
            return zero_stress, np.inf, np.inf

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
                        atol=1e-12, btol=1e-12,
                        iter_lim=100000,
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

        return f_x, f_y
