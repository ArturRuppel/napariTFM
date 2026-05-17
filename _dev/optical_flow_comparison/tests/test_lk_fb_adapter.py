import numpy as np

from _dev.optical_flow_comparison.adapters.lucas_kanade_fb import LucasKanadeFBAdapter


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


def test_lk_fb_adapter_has_name():
    assert LucasKanadeFBAdapter().name == "Lucas-Kanade-FB"


def test_lk_fb_adapter_recovers_uniform_shift_with_high_accuracy():
    ref, deformed, beads, shift = _shifted_pair()
    adapter = LucasKanadeFBAdapter()

    displacements, valid = adapter.displacements_at(ref, deformed, beads)

    assert displacements.shape == (len(beads), 2)
    assert valid.shape == (len(beads),)
    assert valid.sum() >= len(beads) - 2
    np.testing.assert_allclose(displacements[valid].mean(axis=0), shift, atol=0.2)


def test_lk_fb_adapter_rejects_random_noise_pair():
    # Two independent random fields: nothing to track. FB should reject most
    # beads as inconsistent (round-trip distance huge), so valid coverage
    # should be low.
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 1, size=(96, 96)).astype(np.float32)
    deformed = rng.uniform(0, 1, size=(96, 96)).astype(np.float32)
    beads = rng.uniform(10, 80, size=(30, 2)).astype(np.float32)

    adapter = LucasKanadeFBAdapter()
    _, valid = adapter.displacements_at(ref, deformed, beads)

    # On pure noise the FB filter should reject the majority of beads.
    assert valid.sum() < len(beads) // 2
