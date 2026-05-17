import numpy as np

from _dev.optical_flow_comparison.adapters.dis import DISAdapter


def _shifted_pair(shape=(96, 96), shift=(3.0, 2.0), n_beads=20, seed=0):
    """Synthetic ref/deformed pair: random Gaussian beads, shifted by `shift`.

    Returns (ref, deformed, bead_positions_in_reference, shift).
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    margin = 10
    positions = rng.uniform(margin, min(h, w) - margin, size=(n_beads, 2)).astype(np.float32)

    def render(centers):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w), dtype=np.float32)
        sigma = 1.5
        for (x, y) in centers:
            img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        return img / max(img.max(), 1e-6)

    ref = render(positions)
    deformed = render(positions + np.array(shift, dtype=np.float32))
    return ref, deformed, positions, np.array(shift, dtype=np.float32)


def test_dis_adapter_has_name():
    assert DISAdapter().name == "DIS"


def test_dis_adapter_recovers_uniform_shift_within_tolerance():
    ref, deformed, beads, shift = _shifted_pair()
    adapter = DISAdapter()

    displacements, valid = adapter.displacements_at(ref, deformed, beads)

    assert displacements.shape == (len(beads), 2)
    assert valid.shape == (len(beads),)
    assert valid.all()
    mean = displacements.mean(axis=0)
    # DIS is approximate; allow 1 px tolerance per axis on this synthetic case.
    np.testing.assert_allclose(mean, shift, atol=1.0)
