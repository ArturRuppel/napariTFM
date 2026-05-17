import numpy as np
import trackpy as tp

DIAMETER = 7
SEPARATION = 8
AUTO_MINMASS_PERCENTILE = 30.0

tp.quiet()


def _to_uint16(image: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] image to uint16 for TrackPy."""
    return np.clip(image * 65535.0, 0, 65535).astype(np.uint16)


def _auto_minmass(image_u16: np.ndarray) -> float:
    """Estimate minmass: locate everything, take the 30th-percentile mass.

    Mirrors `_dev/spt_piv/spt_piv_displacement.py`.
    """
    feats = tp.locate(image_u16, diameter=DIAMETER, minmass=0, separation=SEPARATION)
    if feats.empty:
        return 0.0
    return float(np.percentile(feats["mass"].values, AUTO_MINMASS_PERCENTILE))


def detect_beads(image: np.ndarray) -> np.ndarray:
    """Detect beads using TrackPy with the project-standard parameters.

    Args:
        image: float32, shape (H, W), range [0, 1].

    Returns:
        float32 array of shape (N, 2), columns (x, y) in pixels.
    """
    image_u16 = _to_uint16(image)
    minmass = _auto_minmass(image_u16)
    feats = tp.locate(image_u16, diameter=DIAMETER, minmass=minmass, separation=SEPARATION)
    if feats.empty:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([feats["x"].values, feats["y"].values]).astype(np.float32)
