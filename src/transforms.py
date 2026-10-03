"""Centred, unitary 2D DFT wrappers (CO3: 2D DFT, separability, Parseval).

MRI convention places the DC sample (k = 0) at the array centre, index
``N // 2`` on each axis. Every transform here is wrapped explicitly with
``ifftshift`` (move centre to index 0) before the DFT and ``fftshift``
(move index 0 back to the centre) after it, so no centring is ever implicit.

All transforms use ``norm="ortho"``, i.e. the DFT matrix is unitary and
Parseval's theorem ``||x||_2 == ||X||_2`` holds to machine precision.

Axis convention used throughout the project:
    axis 0 = phase-encode (PE) direction  (k_y, undersampled)
    axis 1 = readout / frequency-encode    (k_x, always fully sampled)
"""

from __future__ import annotations

import numpy as np

PE_AXIS = 0
RO_AXIS = 1


def fft2c(img: np.ndarray) -> np.ndarray:
    """Centred 2D forward DFT. Image domain -> k-space.

    DTSP concept (CO3): the 2D DFT is separable, so this equals a 1D DFT along
    every row followed by a 1D DFT along every column. ``norm="ortho"`` makes
    the transform unitary (Parseval).
    """
    x = np.asarray(img, dtype=np.complex128)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centred 2D inverse DFT. k-space -> image domain.

    DTSP concept (CO3): inverse of :func:`fft2c`; unitary, so
    ``ifft2c(fft2c(x)) == x`` to machine precision.
    """
    k = np.asarray(kspace, dtype=np.complex128)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))


def fft1c(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Centred unitary 1D forward DFT along ``axis``."""
    x = np.asarray(x, dtype=np.complex128)
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho"), axes=axis
    )


def ifft1c(k: np.ndarray, axis: int = -1) -> np.ndarray:
    """Centred unitary 1D inverse DFT along ``axis``."""
    k = np.asarray(k, dtype=np.complex128)
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(k, axes=axis), axis=axis, norm="ortho"), axes=axis
    )


def hermitian_flip(kspace: np.ndarray) -> np.ndarray:
    """Return ``K[-k]`` in centred indexing (both axes), *without* conjugation.

    In centred indexing sample index ``i`` holds frequency ``k = i - N//2``,
    so ``-k`` lives at index ``(N - i) mod N``. That is a flip followed by a
    roll of one sample. For a real image (CO3, conjugate symmetry)
    ``K == conj(hermitian_flip(K))``.
    """
    out = np.asarray(kspace)
    for ax in range(out.ndim):
        out = np.roll(np.flip(out, axis=ax), 1, axis=ax)
    return out


def zeropad_kspace(kspace: np.ndarray, factor: int = 2) -> np.ndarray:
    """Zero-pad centred k-space by an integer ``factor`` on every axis.

    DTSP concept (CO3): zero-padding the spectrum and inverse transforming is
    *band-limited (periodic-sinc / Dirichlet) interpolation* of the image. No
    new information is created; the samples are only interpolated.

    For even ``N`` the Nyquist bin (index 0, frequency -N/2) has no partner;
    it is split in half between +N/2 and -N/2 so that the interpolant is the
    symmetric trigonometric polynomial (real images stay real).
    """
    k = np.asarray(kspace, dtype=np.complex128)
    for ax in range(k.ndim):
        n = k.shape[ax]
        big = n * factor
        if n % 2 == 0:
            # split Nyquist: put half at index 0 and half at a new index n
            nyq = np.take(k, [0], axis=ax) / 2.0
            k = np.concatenate([nyq, np.take(k, range(1, n), axis=ax), nyq], axis=ax)
            m = n + 1
            start = big // 2 - n // 2
        else:
            m = n
            start = big // 2 - n // 2
        pad = [(0, 0)] * k.ndim
        pad[ax] = (start, big - start - m)
        k = np.pad(k, pad)
    return k


def upsample_by_zeropad(img: np.ndarray, factor: int = 2) -> np.ndarray:
    """Sinc-interpolate an image by zero-padding its centred k-space.

    The ``sqrt(factor**ndim)`` scale compensates the unitary normalisation so
    that, for even sizes, ``out[::factor, ::factor] == img`` exactly (the DC /
    centre pixel ``N//2`` maps to ``factor*N//2``).
    """
    img = np.asarray(img, dtype=np.complex128)
    if img.ndim == 1:
        k = fft1c(img)
        return ifft1c(zeropad_kspace(k, factor)) * np.sqrt(factor)
    k = fft2c(img)
    return ifft2c(zeropad_kspace(k, factor)) * factor ** (img.ndim / 2)


# --------------------------------------------------------------------------
# Reference implementations for the complexity benchmark (CO3)
# --------------------------------------------------------------------------

def naive_dft(x: np.ndarray) -> np.ndarray:
    """Direct O(N^2) DFT: ``X[k] = sum_n x[n] exp(-2j*pi*k*n/N)``.

    Builds the full N x N twiddle matrix (N^2 complex exponentials) and does a
    matrix-vector product (N^2 multiply-adds). Unnormalised, like ``np.fft.fft``.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    k = np.arange(n)
    w = np.exp(-2j * np.pi * np.outer(k, k) / n)
    return w @ x


def fft_radix2(x: np.ndarray) -> np.ndarray:
    """Iterative radix-2 decimation-in-time FFT, O(N log2 N), N a power of 2.

    Bit-reversal permutation followed by log2(N) butterfly stages; each stage
    is vectorised over all butterflies, so the cost is log2(N) passes of O(N).
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    stages = n.bit_length() - 1
    if 1 << stages != n:
        raise ValueError("length must be a power of two")
    rev = np.zeros(n, dtype=np.int64)
    for b in range(stages):
        rev |= ((np.arange(n) >> b) & 1) << (stages - 1 - b)
    a = x[rev].copy()
    size = 2
    while size <= n:
        half = size // 2
        tw = np.exp(-2j * np.pi * np.arange(half) / size)
        a = a.reshape(-1, size)
        top = a[:, :half].copy()
        bot = a[:, half:] * tw
        a[:, :half] = top + bot
        a[:, half:] = top - bot
        a = a.reshape(-1)
        size *= 2
    return a
