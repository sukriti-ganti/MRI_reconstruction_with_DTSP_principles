"""Point spread function analysis (CO1: convolution theorem).

Undersampling multiplies k-space by a mask M. By the convolution theorem
``IDFT(M . K) = psf (*) image`` with ``psf = IDFT(M)`` and (*) *circular*
convolution, because the DFT treats both domains as periodic. The PSF is the
whole story: its mainlobe sets blur, its sidelobes set the artefact, and the
*shape* of the sidelobes decides coherent ghosting versus incoherent noise.

Because every mask is constant along the readout, its PSF is a delta along
the readout and all the structure lies in the PE (axis 0) profile.
"""

from __future__ import annotations

import numpy as np

from .transforms import ifft2c


def compute_psf(mask: np.ndarray) -> np.ndarray:
    """IFFT of the sampling mask. This is the kernel convolved with the image.

    Returned complex, centred, with the unitary scaling of :func:`ifft2c`
    (the zero-displacement tap sits at ``(H//2, W//2)``).
    """
    return ifft2c(np.asarray(mask, dtype=np.complex128))


def psf_profile(psf: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalised magnitude profile through the PSF centre along ``axis``."""
    psf = np.asarray(psf)
    c = tuple(s // 2 for s in psf.shape)
    if axis == 0:
        prof = np.abs(psf[:, c[1]])
    else:
        prof = np.abs(psf[c[0], :])
    return prof / prof.max()


def circular_convolve_centred(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Direct (non-FFT) 2D circular convolution with a centred unitary PSF.

    ``out = (1/sqrt(HW)) * sum_m psf[m] * roll(img, m - centre)``.
    The 1/sqrt(HW) factor comes from the unitary DFT normalisation. Used to
    verify the convolution theorem without using the theorem itself.
    """
    img = np.asarray(img, dtype=np.complex128)
    psf = np.asarray(psf, dtype=np.complex128)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    out = np.zeros_like(img)
    for my, mx in zip(*np.nonzero(np.abs(psf) > 0)):
        out += psf[my, mx] * np.roll(img, (my - cy, mx - cx), axis=(0, 1))
    return out / np.sqrt(h * w)


def _mainlobe_bounds(prof: np.ndarray) -> tuple[int, int]:
    """Indices (lo, hi) of the first minima either side of the centre peak."""
    c = prof.size // 2
    lo = c
    while lo > 0 and prof[lo - 1] < prof[lo]:
        lo -= 1
    hi = c
    while hi < prof.size - 1 and prof[hi + 1] < prof[hi]:
        hi += 1
    return lo, hi


def _fwhm(prof: np.ndarray) -> float:
    """Full width at half maximum of the centre lobe, linear interpolation."""
    c = prof.size // 2
    half = 0.5 * prof[c]

    def cross(step: int) -> float:
        i = c
        while 0 <= i + step < prof.size and prof[i + step] >= half:
            i += step
        j = i + step
        if not 0 <= j < prof.size:
            return float(abs(i - c))
        # interpolate between i (>= half) and j (< half)
        t = (prof[i] - half) / (prof[i] - prof[j])
        return abs(i - c) + t

    return cross(-1) + cross(+1)


def psf_metrics(psf: np.ndarray) -> dict:
    """Scalar descriptors of a PSF, computed on its PE-axis profile.

    Returns
    -------
    mainlobe_fwhm : FWHM of the central lobe in pixels (blur).
    peak_sidelobe_ratio : largest sidelobe / mainlobe peak (PSR). 1.0 for a
        uniform comb: the ghosts are as strong as the image itself.
    sidelobe_energy_fraction : PSF energy outside the mainlobe / total energy.
        By Parseval this is the share of each object's energy that leaks away
        from its true location.
    coherence_index : peak-to-average ratio of the sidelobes,
        ``max(sidelobe |psf|) / mean(sidelobe |psf|)``. Large when leaked
        energy piles into a few discrete replicas (coherent ghosts), ~1-5 when
        spread thinly (incoherent noise). Defined as 0 when there are no
        sidelobes (fully sampled mask).
    """
    prof = psf_profile(psf, axis=0)
    lo, hi = _mainlobe_bounds(prof)
    side = np.concatenate([prof[:lo], prof[hi + 1:]])
    energy = prof**2
    total = energy.sum()
    side_e = float((side**2).sum() / total) if side.size else 0.0
    if side.size == 0 or side.max() < 1e-9:
        ci, psr = 0.0, 0.0
    else:
        ci = float(side.max() / side.mean())
        psr = float(side.max())
    return {
        "mainlobe_fwhm": _fwhm(prof),
        "peak_sidelobe_ratio": psr,
        "sidelobe_energy_fraction": side_e,
        "coherence_index": ci,
    }


def find_psf_peaks(psf: np.ndarray, rel_threshold: float = 0.5) -> np.ndarray:
    """PE positions (pixels, relative to the centre) of PSF peaks above threshold.

    Local maxima of the circular PE profile whose height is at least
    ``rel_threshold`` of the main peak, sorted by offset.
    """
    prof = psf_profile(psf, axis=0)
    left, right = np.roll(prof, 1), np.roll(prof, -1)
    peaks = np.flatnonzero((prof >= left) & (prof >= right) & (prof >= rel_threshold))
    return np.sort(peaks - prof.size // 2)


def ghost_spacing_from_psf(psf: np.ndarray) -> float:
    """Distance (pixels) from the mainlobe to the nearest strong PSF replica.

    The replica is the largest sidelobe local maximum; for a uniform mask of
    period R it is predicted at N/R (sampling theorem, CO3).
    """
    prof = psf_profile(psf, axis=0)
    n = prof.size
    c = n // 2
    lo, hi = _mainlobe_bounds(prof)
    side = prof.copy()
    side[lo:hi + 1] = 0
    # strong sidelobe maxima (>= half the largest); take the one nearest the
    # centre, since a comb has equal replicas at every multiple of N/R
    is_max = (side >= np.roll(side, 1)) & (side >= np.roll(side, -1))
    cand = np.flatnonzero(is_max & (side >= 0.5 * side.max()))
    dist = np.minimum(np.abs(cand - c), n - np.abs(cand - c))
    i = int(cand[np.argmin(dist)])
    # sub-pixel location via parabolic interpolation around the max
    a, b, d = side[(i - 1) % n], side[i], side[(i + 1) % n]
    denom = a - 2 * b + d
    delta = 0.5 * (a - d) / denom if denom != 0 else 0.0
    off = (i + delta) - c
    # report the replica nearest the centre (the comb is symmetric)
    return float(min(abs(off), n - abs(off)))
