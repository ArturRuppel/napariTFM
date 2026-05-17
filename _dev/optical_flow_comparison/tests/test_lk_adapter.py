import numpy as np

from _dev.optical_flow_comparison.adapters.lucas_kanade import LucasKanadeAdapter


def _shifted_pair(shape=(96, 96), shift=(3.0, 2.0), n_beads=20, seed=0):
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


def test_lk_adapter_has_name():
    assert LucasKanadeAdapter().name == "Lucas-Kanade"


def test_lk_adapter_recovers_uniform_shift_with_high_accuracy():
    ref, deformed, beads, shift = _shifted_pair()
    adapter = LucasKanadeAdapter()

    displacements, valid = adapter.displacements_at(ref, deformed, beads)

    assert displacements.shape == (len(beads), 2)
    assert valid.shape == (len(beads),)
    assert valid.sum() >= len(beads) - 2  # allow up to 2 dropouts on edges
    # LK should be sub-pixel accurate on this clean synthetic case.
    np.testing.assert_allclose(displacements[valid].mean(axis=0), shift, atol=0.2)
