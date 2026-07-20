"""Diffraction-PSF forward model for the synthetic bead stacks (make_stacks.py).

Point emitters at random depth, imaged through a physical Gibson-Lanni scalar PSF
(Airy rings + correct defocus donut), then a measured camera (pedestal + gain*Poisson
+ read). The PSF *shape* comes from optics, not data; only a handful of scalars are
anchored to the rig.

Those scalars were calibrated ONCE, out of this repo, by matching a real bead image's
radial power spectrum and intensity histogram (that fit needs a private reference frame
and is not reproduced here). They are frozen below so the generator is self-contained
and deterministic. `flux_per_px` in particular was `mean(clip(real_crop - PED, 0))` on
the 512x512 centre crop of the calibration frame -- treat it like a measured gain, not
a free knob.

Needs `psfmodels` (validation-only dependency): pip install psfmodels.
"""
from functools import lru_cache

import numpy as np
import psfmodels as psfm

# Camera (measured on the rig): pedestal counts, read noise (counts), e-/count gain, frame size px.
PED, READ, GAIN, N = 177.0, 6.4, 2.17, 512
# Optics: pixel pitch um, emission um, sample RI (water), immersion RI.
DXY, WVL, NS, NI = 0.1612, 0.670, 1.33, 1.515
NPSF, DZ_STACK = 51, 0.4                            # PSF support px, defocus sampling um
# Per-pixel flux at expo=1, calibrated against the real frame (see module docstring). Frozen.
flux_per_px = 154.739609


@lru_cache(maxsize=16)
def psf_stack(NA, Z):
    """Diffraction PSF sampled across defocus [-Z, Z]; each plane normalized so the
    in-focus plane sums to 1 (preserves the slight physical defocus light-loss)."""
    zs = np.arange(-Z, Z + 1e-6, DZ_STACK)
    st = np.asarray(psfm.make_psf(z=zs, nx=NPSF, dxy=DXY, wvl=WVL, NA=NA, ns=NS, ni=NI, model="scalar"))
    st = st / st[len(zs) // 2].sum()               # in-focus plane -> unit flux
    return zs, st.astype(np.float32)
