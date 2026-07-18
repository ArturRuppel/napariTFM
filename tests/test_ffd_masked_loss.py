"""FFD masked loss: with a foreground weight, the LNCC/MSE objective ignores
differences outside the mask, so the fit is driven only by the cell region.

Pure-torch metrics, so these run on CPU without a GPU (unlike the FFD analyzer,
which is GPU-only)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from napariTFM.backend._ffd_torch import _lncc, _mse


def _corrupt_outside(mask):
    """A textured pair (a, b) identical inside the mask, differing everywhere else."""
    torch.manual_seed(0)
    a = torch.rand(128, 128)
    noise = torch.rand(128, 128)
    b = torch.where(mask > 0, a, noise)   # in-mask: b == a; out-of-mask: b is unrelated
    return a, b


def test_masked_mse_ignores_out_of_mask():
    mask = torch.zeros(128, 128)
    mask[40:88, 40:88] = 1.0
    a, b = _corrupt_outside(mask)
    # The only differences are outside the mask, which the weight zeroes out.
    assert _mse(a, b, mask).item() < 1e-6
    # Unmasked, the corrupted background dominates.
    assert _mse(a, b).item() > 1e-2


def test_masked_lncc_ignores_out_of_mask():
    mask = torch.zeros(128, 128)
    mask[40:88, 40:88] = 1.0
    a, b = _corrupt_outside(mask)
    # In-mask a == b -> local CC ~1 -> loss ~0, even for windows straddling the
    # boundary (normalised convolution uses only their in-mask pixels).
    assert _lncc(a, b, mask).item() < 0.05
    # Unmasked, the decorrelated background pushes the loss up.
    assert _lncc(a, b).item() > 0.2


def test_masked_loss_matches_unmasked_when_weight_is_all_ones():
    """An all-ones weight must reproduce the plain loss (no silent behaviour change)."""
    torch.manual_seed(1)
    a = torch.rand(64, 64)
    b = torch.rand(64, 64)
    ones = torch.ones(64, 64)
    assert _mse(a, b, ones).item() == pytest.approx(_mse(a, b).item(), rel=1e-5)
    # LNCC's masked path divides by boxmean(m); for m==1 that is 1, so it matches
    # the unmasked reduction to within float error.
    assert _lncc(a, b, ones).item() == pytest.approx(_lncc(a, b).item(), abs=1e-4)
