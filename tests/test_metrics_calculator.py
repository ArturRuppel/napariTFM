"""Contractile moment tensor: symmetry, and what symmetry buys.

The moment tensor is the one place in the metrics where a plausible-looking
expression (the raw first moment ``integral(x_i f_j) dA``) is not the quantity
the literature calls the contractile moment tensor. These tests pin the
difference down rather than leaving it to a docstring.
"""
import numpy as np
import pytest

from napariTFM.backend import metrics_calculator as mc


#: Nominal grid spacing and pixel area. Deliberately O(1) rather than the real
#: 1e-6 m / 1e-12 m^2: at true SI magnitudes an order-one dipole lands below
#: `calculate_polarization`'s 1e-15 near-zero-trace guard and every polarization
#: below comes back 0.0. The algebra under test is scale-free; the units are not.
SPACING = 1.0
AREA = 1.0


def _positions(shape, spacing_m=SPACING):
    """Pixel positions, centred on the field, ordered (x, y)."""
    ny, nx = shape
    row, col = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    pos = np.zeros((ny, nx, 2))
    pos[..., 0] = (col - (nx - 1) / 2) * spacing_m
    pos[..., 1] = (row - (ny - 1) / 2) * spacing_m
    return pos


def _dipole(shape=(9, 9), fx=0.0, fy=0.0, torque=0.0):
    """A traction field with a chosen contraction along x/y and a chosen torque.

    Two point forces on the x axis pulling inward with magnitude ``fx`` and two
    on the y axis with magnitude ``fy`` give a diagonal, force-balanced dipole.
    ``torque`` adds a force-balanced but rotational couple on top. All four of
    its forces are needed: tangential forces on the x axis alone give
    ``m01 != 0, m10 == 0``, which is half torque and half shear, and the shear
    half survives symmetrization. A pure couple needs ``m01 == -m10``.
    """
    force = np.zeros((*shape, 2))
    cy, cx = (shape[0] - 1) // 2, (shape[1] - 1) // 2
    force[cy, cx - 3, 0] = +fx      # left  edge pulls right (inward)
    force[cy, cx + 3, 0] = -fx
    force[cy - 3, cx, 1] = +fy      # top   edge pulls down  (inward)
    force[cy + 3, cx, 1] = -fy
    force[cy, cx - 3, 1] = +torque  # couple: tangential all the way round,
    force[cy, cx + 3, 1] = -torque  # zero net force, zero symmetric part,
    force[cy - 3, cx, 0] = -torque  # nonzero net torque
    force[cy + 3, cx, 0] = +torque
    return force


def test_moment_tensor_is_symmetric():
    force = _dipole(fx=1.0, fy=0.4, torque=0.7)
    mask = np.ones(force.shape[:2])
    m = mc.calculate_moment_tensor(force, mask, _positions(force.shape[:2]), AREA)
    assert m[0, 1] == pytest.approx(m[1, 0])


def test_symmetrization_removes_exactly_the_torque():
    """The dropped part is the net torque, and only that."""
    shape = (9, 9)
    pos, mask, area = _positions(shape), np.ones(shape), AREA
    quiet = _dipole(shape, fx=1.0, fy=0.4)
    spun = _dipole(shape, fx=1.0, fy=0.4, torque=0.7)

    # A pure couple changes the raw first moment but not the symmetric tensor.
    assert mc.calculate_moment_tensor(spun, mask, pos, area) == pytest.approx(
        mc.calculate_moment_tensor(quiet, mask, pos, area))
    # ...and the difference it makes is reported, not silently swallowed.
    assert mc.moment_tensor_torque(quiet, mask, pos, area) == pytest.approx(0.0)
    assert mc.moment_tensor_torque(spun, mask, pos, area) != pytest.approx(0.0)


def test_polarization_of_a_uniaxial_dipole_is_one():
    shape = (9, 9)
    force = _dipole(shape, fx=1.0, fy=0.0)
    m = mc.calculate_moment_tensor(force, np.ones(shape), _positions(shape), AREA)
    pi, l1, l2 = mc.calculate_polarization(m)
    assert pi == pytest.approx(1.0)
    assert l2 < 0 and l1 == pytest.approx(0.0, abs=1e-12)  # contraction: both <= 0


def test_polarization_of_an_isotropic_dipole_is_zero():
    shape = (9, 9)
    force = _dipole(shape, fx=1.0, fy=1.0)
    m = mc.calculate_moment_tensor(force, np.ones(shape), _positions(shape), AREA)
    pi, _l1, _l2 = mc.calculate_polarization(m)
    assert pi == pytest.approx(0.0, abs=1e-12)


def test_polarization_rejects_a_non_symmetric_tensor():
    """The guard exists because eigvals would return a complex pair here, and
    ``max`` of complex numbers is a lexicographic comparison, not an ordering."""
    with pytest.raises(ValueError, match="not symmetric"):
        mc.calculate_polarization(np.array([[-1.0, 5.0], [-5.0, -1.0]]))


def test_moment_tensor_origin_independent_when_force_balanced():
    """A balanced traction field gives the same tensor about any origin — which
    is what makes integrating over the WHOLE field, rather than over the cell
    footprint, the well-posed choice."""
    shape = (9, 9)
    force = _dipole(shape, fx=1.0, fy=0.4)
    mask, area = np.ones(shape), AREA
    centred = _positions(shape)
    shifted = centred + np.array([3.7, -2.1])
    assert mc.calculate_moment_tensor(force, mask, shifted, area) == pytest.approx(
        mc.calculate_moment_tensor(force, mask, centred, area))
