"""Window functions for k-space apodization (CO4: FIR window design).

Truncating k-space is multiplication by a rectangular window, whose DTFT is a
Dirichlet kernel with -13 dB sidelobes: these are the Gibbs ripples. Exactly
as in FIR filter design, tapering the window lowers the sidelobes (less
ringing) at the price of a wider mainlobe (more blur).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

WINDOWS = ("rect", "hann", "hamming", "tukey", "blackman")


def window_1d(n: int, kind: str, **kwargs) -> np.ndarray:
    """Symmetric length-``n`` window. ``kind`` in rect/hann/hamming/tukey/blackman.

    ``tukey`` takes ``alpha`` (taper fraction, default 0.5).
    """
    if n <= 0:
        return np.zeros(0)
    if kind in ("rect", "boxcar", "rectangular"):
        return np.ones(n)
    if kind == "tukey":
        return get_window(("tukey", kwargs.get("alpha", 0.5)), n, fftbins=False)
    if kind in ("hann", "hamming", "blackman"):
        return get_window(kind, n, fftbins=False)
    raise ValueError(f"unknown window {kind!r}")


def window_2d_separable(shape: tuple, kind: str, **kwargs) -> np.ndarray:
    """Outer product of two 1D windows, ``w(ky) * w(kx)``."""
    return np.outer(window_1d(shape[0], kind, **kwargs), window_1d(shape[1], kind, **kwargs))


def window_2d_radial(shape: tuple, kind: str, **kwargs) -> np.ndarray:
    """Rotationally symmetric window ``w(r)`` sampled on the centred k grid.

    The 1D window's right half is used as a radial profile, with r = 1 at the
    edge of the inscribed circle; corners beyond r = 1 are zero.
    """
    h, w = shape
    n = 2049
    prof = window_1d(n, kind, **kwargs)[n // 2:]  # r in [0, 1]
    ky = (np.arange(h) - h // 2) / (h / 2)
    kx = (np.arange(w) - w // 2) / (w / 2)
    r = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    out = np.interp(r, np.linspace(0, 1, prof.size), prof, right=0.0)
    return out


def window_for_mask(mask_lines: np.ndarray, kind: str, **kwargs) -> np.ndarray:
    """1D PE window spanning the sampled k_y extent of a line mask.

    The window is centred on DC and its length is ``2*max|k_y| + 1`` over the
    sampled lines, so for a low-pass mask it tapers exactly to the truncation
    edge (that is where the rectangular discontinuity lives).
    """
    h = mask_lines.size
    ky = np.arange(h) - h // 2
    kmax = int(np.abs(ky[mask_lines]).max())
    win = np.zeros(h)
    seg = window_1d(2 * kmax + 1, kind, **kwargs)
    lo = h // 2 - kmax
    idx = np.arange(lo, lo + seg.size)
    keep = (idx >= 0) & (idx < h)
    win[idx[keep]] = seg[keep]
    return win


def window_response(kind: str, n: int = 64, nfft: int = 65536, **kwargs) -> dict:
    """Numerical DTFT characterisation of a window (CO4).

    The DTFT is sampled densely with a zero-padded FFT. Returns the mainlobe
    width between first nulls and at -3 dB (both in DFT bins of a length-n
    sequence, i.e. units of 2*pi/n rad), the peak sidelobe level in dB, and
    the sidelobe roll-off asymptote is left to the plot.
    """
    w = window_1d(n, kind, **kwargs)
    spec = np.abs(np.fft.fft(w, nfft))
    spec_db = 20 * np.log10(spec / spec[0] + 1e-300)
    half = spec_db[: nfft // 2]
    # first null: first local minimum moving away from DC
    i = 1
    while i < half.size - 1 and not (half[i] <= half[i - 1] and half[i] <= half[i + 1]):
        i += 1
    first_null = i
    bins_per_sample = nfft / n
    psl = float(half[first_null:].max())
    i3 = int(np.argmax(half < -3.0103))
    return {
        "window": kind,
        "n": n,
        "mainlobe_width_null_bins": 2 * first_null / bins_per_sample,
        "mainlobe_width_3db_bins": 2 * i3 / bins_per_sample,
        "peak_sidelobe_db": psl,
        "coherent_gain": float(w.sum() / n),
    }


def window_spectrum_db(kind: str, n: int = 64, nfft: int = 8192, **kwargs) -> tuple:
    """(normalised frequency in bins, magnitude in dB) of a window's DTFT."""
    w = window_1d(n, kind, **kwargs)
    spec = np.abs(np.fft.fftshift(np.fft.fft(w, nfft)))
    f = (np.arange(nfft) - nfft // 2) * n / nfft
    return f, 20 * np.log10(spec / spec.max() + 1e-12)
