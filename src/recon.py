"""Reconstruction methods for undersampled Cartesian k-space.

All functions take the (possibly already masked) centred k-space and the
boolean mask and return a complex128 image; convert to magnitude only when
computing metrics.
"""

from __future__ import annotations

import numpy as np

from .apodization import window_for_mask
from .masks import pe_lines
from .transforms import fft2c, hermitian_flip, ifft2c


def recon_zerofill(kspace_us: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-filled inverse DFT.

    DTSP (CO1/CO3): unsampled lines are zeros, so the result is exactly
    ``psf (*) image`` (circular convolution). Zero-filling interpolates on the
    full grid but adds no information.
    """
    return ifft2c(np.where(mask, kspace_us, 0))


def recon_apodized(kspace_us: np.ndarray, mask: np.ndarray, window: str = "hamming",
                   **kwargs) -> np.ndarray:
    """Zero-fill after multiplying sampled k-space by a PE window (CO4).

    The window spans the sampled k_y extent, so for truncated (low-pass)
    k-space it tapers to the truncation edge. The readout is fully sampled and
    therefore left untouched (no truncation there, no ringing to suppress).
    ``window="rect"`` reproduces :func:`recon_zerofill`.
    """
    win = window_for_mask(pe_lines(mask), window, **kwargs)
    return ifft2c(np.where(mask, kspace_us, 0) * win[:, None])


def recon_conjugate_symmetry(kspace_us: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Partial-Fourier fill using Hermitian symmetry K[-k] = K*[k] (CO3).

    Every missing sample whose mirror ``-k`` was acquired is replaced by the
    conjugate of the mirror. Exact for a real image; for a complex image the
    symmetry does not hold, so the filled half is wrong and the phase error
    turns into blur and intensity errors.
    """
    k = np.where(mask, kspace_us, 0)
    k_mirror = np.conj(hermitian_flip(k))
    m_mirror = hermitian_flip(mask)
    fill = (~mask) & m_mirror
    return ifft2c(np.where(fill, k_mirror, k))


def _symmetric_centre_lines(mask: np.ndarray) -> np.ndarray:
    """PE lines whose mirror is also sampled (the 'both halves' centre band)."""
    lines = pe_lines(mask)
    h = lines.size
    mirror = np.roll(lines[::-1], 1)  # index (h - i) % h
    both = lines & mirror
    # keep only the contiguous band around DC
    ky = np.arange(h) - h // 2
    kmax = 0
    while kmax + 1 < h // 2 and both[h // 2 + kmax + 1] and both[h // 2 - kmax - 1]:
        kmax += 1
    return np.abs(ky) <= kmax


def phase_estimate(kspace_us: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Low-resolution phase from the symmetrically sampled centre of k-space.

    A Hann window over the symmetric band suppresses Gibbs ripple in the
    estimate. Smooth (scanner-like) phase is concentrated at low k, so this
    estimate is accurate even though the band is narrow.
    """
    band = _symmetric_centre_lines(mask)
    h = band.size
    kmax = int(np.abs(np.arange(h) - h // 2)[band].max())
    win = np.zeros(h)
    win[h // 2 - kmax: h // 2 + kmax + 1] = np.hanning(2 * kmax + 1)
    return np.angle(ifft2c(np.where(mask, kspace_us, 0) * win[:, None]))


def recon_pocs(kspace_us: np.ndarray, mask: np.ndarray, n_iter: int = 20) -> np.ndarray:
    """Projection onto convex sets for partial Fourier (Haacke et al. 1991).

    Alternates two projections:
    1. image space: impose the low-resolution phase estimate,
       ``x <- |x| exp(i*phi_lr)``;
    2. k-space: data consistency, overwrite sampled lines with measurements.
    Unlike plain conjugate symmetry it accounts for the image phase, so it
    keeps working when the image is not real.
    """
    meas = np.where(mask, kspace_us, 0)
    phi = phase_estimate(kspace_us, mask)
    x = ifft2c(meas)
    for _ in range(n_iter):
        x = np.abs(x) * np.exp(1j * phi)
        k = fft2c(x)
        k = np.where(mask, meas, k)
        x = ifft2c(k)
    return x


# --------------------------------------------------------------------------
# Stretch goal: compressed sensing with an orthonormal Haar wavelet + FISTA
# --------------------------------------------------------------------------

def _haar_fwd_1level(x: np.ndarray) -> np.ndarray:
    s = 1 / np.sqrt(2)
    a = (x[0::2, :] + x[1::2, :]) * s
    d = (x[0::2, :] - x[1::2, :]) * s
    x = np.concatenate([a, d], axis=0)
    a = (x[:, 0::2] + x[:, 1::2]) * s
    d = (x[:, 0::2] - x[:, 1::2]) * s
    return np.concatenate([a, d], axis=1)


def _haar_inv_1level(c: np.ndarray) -> np.ndarray:
    s = 1 / np.sqrt(2)
    h, w = c.shape
    a, d = c[:, : w // 2], c[:, w // 2:]
    x = np.empty_like(c)
    x[:, 0::2] = (a + d) * s
    x[:, 1::2] = (a - d) * s
    a, d = x[: h // 2, :], x[h // 2:, :]
    y = np.empty_like(c)
    y[0::2, :] = (a + d) * s
    y[1::2, :] = (a - d) * s
    return y


def haar2d(x: np.ndarray, levels: int = 4) -> np.ndarray:
    """Orthonormal multi-level 2D Haar transform (Mallat layout)."""
    c = np.array(x, dtype=np.complex128)
    h, w = c.shape
    for lev in range(levels):
        hh, ww = h >> lev, w >> lev
        c[:hh, :ww] = _haar_fwd_1level(c[:hh, :ww])
    return c


def ihaar2d(c: np.ndarray, levels: int = 4) -> np.ndarray:
    """Inverse of :func:`haar2d`."""
    x = np.array(c, dtype=np.complex128)
    h, w = x.shape
    for lev in reversed(range(levels)):
        hh, ww = h >> lev, w >> lev
        x[:hh, :ww] = _haar_inv_1level(x[:hh, :ww])
    return x


def _soft(c: np.ndarray, t: float) -> np.ndarray:
    mag = np.abs(c)
    return np.where(mag > t, (1 - t / np.maximum(mag, 1e-300)) * c, 0)


def recon_cs_fista(kspace_us: np.ndarray, mask: np.ndarray, lam: float = 0.01,
                   n_iter: int = 100, levels: int = 4) -> np.ndarray:
    """L1-wavelet compressed sensing solved with FISTA (stretch goal).

    Minimises ``0.5*||M F x - y||^2 + lam*||W x||_1`` with F the unitary DFT
    and W an orthonormal Haar transform (coarsest band not penalised). The
    operator ``M F`` has Lipschitz constant 1, so the step size is 1.

    Sparsity can only separate signal from aliasing when the aliasing is
    incoherent (noise-like). With a uniform comb mask the aliases are exact
    shifted copies of the image, just as sparse as the image itself, so CS
    cannot remove them: the PSF shape decides whether CS can work.
    """
    meas = np.where(mask, kspace_us, 0)
    h, w = meas.shape
    coarse = (slice(0, h >> levels), slice(0, w >> levels))
    x = ifft2c(meas)
    z = x.copy()
    t = 1.0
    for _ in range(n_iter):
        g = ifft2c(np.where(mask, fft2c(z) - meas, 0))
        c = haar2d(z - g, levels)
        keep = c[coarse].copy()
        c = _soft(c, lam)
        c[coarse] = keep
        x_new = ihaar2d(c, levels)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


RECONS = {
    "zerofill": recon_zerofill,
    "apodized": recon_apodized,
    "conjugate_symmetry": recon_conjugate_symmetry,
    "pocs": recon_pocs,
    "cs_fista": recon_cs_fista,
}
