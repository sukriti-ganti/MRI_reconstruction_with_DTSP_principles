"""Test images, synthetic phase and k-space generation.

The bundled images are magnitude-only. Two phase models are offered:

* ``"none"``   - the image is real, so its k-space is exactly Hermitian
                 (CO3, conjugate symmetry). An idealisation.
* ``"smooth"`` - the image is multiplied by a smooth low-order polynomial
                 phase map with peak-to-peak 2*pi, which breaks Hermitian
                 symmetry as real scanner phase does.

No network access is needed: ``skimage.data.shepp_logan_phantom`` ships with
scikit-image and the brain slices are vendored in ``data/brain_slices.npz``
(a copy of ``skimage.data.brain()``, used if scikit-image cannot fetch it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.transform import resize

from .transforms import fft2c

_VENDORED_BRAIN = Path(__file__).resolve().parent.parent / "data" / "brain_slices.npz"


def _brain_stack() -> np.ndarray:
    if _VENDORED_BRAIN.exists():
        return np.load(_VENDORED_BRAIN)["slices"]
    from skimage import data as skdata  # may need a one-off download

    return skdata.brain()


def load_image(source: str = "brain", slice_idx: int = 4, size: int = 256) -> np.ndarray:
    """Load a real, non-negative test image scaled to [0, 1], shape (size, size).

    ``source`` is ``"brain"`` (MRI slice ``slice_idx`` of skimage's brain stack)
    or ``"phantom"`` (Shepp-Logan). The phantom is resized with nearest
    neighbour and no anti-aliasing so its edges stay perfectly sharp, which
    makes Gibbs ringing unambiguous.
    """
    if source == "brain":
        img = _brain_stack()[slice_idx].astype(np.float64)
        if img.shape != (size, size):
            img = resize(img, (size, size), order=1, anti_aliasing=True)
    elif source == "phantom":
        from skimage.data import shepp_logan_phantom

        img = shepp_logan_phantom().astype(np.float64)
        img = resize(img, (size, size), order=0, anti_aliasing=False)
    else:
        raise ValueError(f"unknown image source {source!r}")
    img = np.clip(img, 0.0, None)
    return img / img.max()


def synthetic_phase_map(shape: tuple, peak_to_peak: float = 2 * np.pi) -> np.ndarray:
    """Smooth 2D quadratic polynomial phase map, rescaled to ``peak_to_peak``.

    Low-order => its spectrum is concentrated near k = 0, mimicking B0 and
    coil phase, which vary slowly across the field of view.
    """
    h, w = shape
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    p = 0.9 * x + 0.6 * y + 0.8 * x**2 - 0.5 * x * y + 0.7 * y**2
    p = (p - p.min()) / (p.max() - p.min())  # [0, 1]
    return peak_to_peak * (p - 0.5)


def apply_synthetic_phase(img: np.ndarray, mode: str = "none") -> np.ndarray:
    """Return a complex128 image ``img * exp(i*phi)``.

    ``mode="none"`` -> phi = 0 (real image, exactly Hermitian k-space).
    ``mode="smooth"`` -> phi = smooth polynomial, peak-to-peak 2*pi.
    """
    img = np.asarray(img, dtype=np.float64)
    if mode == "none":
        return img.astype(np.complex128)
    if mode == "smooth":
        return img * np.exp(1j * synthetic_phase_map(img.shape))
    raise ValueError(f"unknown phase mode {mode!r}")


def make_kspace(img_complex: np.ndarray) -> np.ndarray:
    """Fully-sampled centred k-space of a complex image (CO3, 2D DFT)."""
    return fft2c(img_complex)


def ground_truth(img_complex: np.ndarray) -> np.ndarray:
    """Magnitude ground truth: |IDFT(fully sampled k-space)| = |image|."""
    return np.abs(np.asarray(img_complex))
