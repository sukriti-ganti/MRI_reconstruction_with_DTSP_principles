"""Numerical verification of the DTSP claims made in the report.

These are the project's evidence, not smoke tests.
"""

import numpy as np
import pytest

from src.apodization import window_response
from src.data import apply_synthetic_phase, load_image, make_kspace
from src.masks import MASKS, make_mask, mask_partial_fourier, mask_uniform, pe_lines, \
    realised_accel
from src.metrics import nrmse
from src.psf import circular_convolve_centred, compute_psf, find_psf_peaks, psf_profile
from src.recon import recon_conjugate_symmetry, recon_zerofill
from src.transforms import fft2c, hermitian_flip, ifft2c, upsample_by_zeropad


@pytest.fixture(scope="module")
def brain():
    return load_image("brain")


def test_parseval(brain):
    x = apply_synthetic_phase(brain, "smooth")
    assert abs(np.linalg.norm(x) - np.linalg.norm(fft2c(x))) < 1e-10


def test_roundtrip(brain):
    x = apply_synthetic_phase(brain, "smooth")
    assert np.max(np.abs(ifft2c(fft2c(x)) - x)) < 1e-10


@pytest.mark.parametrize("name", ["uniform", "random", "vardens", "lowpass"])
def test_convolution_theorem(name):
    """ifft2c(M . K) == psf (*) x, with (*) a *direct* circular convolution (CO1)."""
    rng = np.random.default_rng(0)
    shape = (24, 16)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    m = make_mask(name, shape, 3)
    lhs = ifft2c(m * fft2c(x))
    rhs = circular_convolve_centred(x, compute_psf(m))
    assert np.max(np.abs(lhs - rhs)) < 1e-8


@pytest.mark.parametrize("r", [2, 3, 4, 6, 8])
def test_ghost_spacing(r):
    """Uniform mask with R | N: PSF peaks are exactly N/R apart (sampling theorem)."""
    n = 240
    p = compute_psf(mask_uniform((n, n), r))
    peaks = find_psf_peaks(p, 0.5)
    gaps = np.diff(np.concatenate([peaks, [peaks[0] + n]]))
    assert np.all(gaps == n // r)


@pytest.mark.parametrize("r", [2, 3, 4, 6, 8])
def test_ghost_count(r):
    """PSF of mask_uniform(R) has exactly R peaks of equal magnitude, zero elsewhere."""
    n = 240
    prof = psf_profile(compute_psf(mask_uniform((n, n), r)))
    big = prof > 1e-9
    assert big.sum() == r
    assert np.allclose(prof[big], 1.0, atol=1e-10)


def test_hermitian_symmetry(brain):
    k = make_kspace(apply_synthetic_phase(brain, "none"))
    assert np.max(np.abs(k - np.conj(hermitian_flip(k)))) < 1e-10


@pytest.mark.parametrize("frac", [0.55, 0.625, 0.75])
def test_conjugate_recon_exact(brain, frac):
    x = apply_synthetic_phase(brain, "none")
    m = mask_partial_fourier(brain.shape, frac=frac)
    assert nrmse(np.abs(x), recon_conjugate_symmetry(make_kspace(x), m)) < 1e-6


@pytest.mark.parametrize("frac", [0.55, 0.625])
def test_conjugate_recon_degrades(brain, frac):
    x = apply_synthetic_phase(brain, "smooth")
    m = mask_partial_fourier(brain.shape, frac=frac)
    assert nrmse(np.abs(x), recon_conjugate_symmetry(make_kspace(x), m)) > 0.05


@pytest.mark.parametrize("n", [16, 17])
def test_zeropad_interpolates(n):
    """Zero-padding k-space x2 then IFFT == periodic-sinc (Dirichlet) interpolation."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    up = upsample_by_zeropad(x, 2)
    c, big = n // 2, 2 * n
    t = (np.arange(big) - big // 2) / 2 + c  # output sample positions, original units
    d = t[:, None] - np.arange(n)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        if n % 2:
            ker = np.sin(np.pi * d) / (n * np.sin(np.pi * d / n))
        else:
            ker = np.sin(np.pi * d) / (n * np.tan(np.pi * d / n))
    ker[np.isclose(np.mod(d, n), 0)] = 1.0
    expected = ker @ x
    assert np.max(np.abs(up - expected)) < 1e-10
    assert np.max(np.abs(up.imag)) < 1e-12


def test_realised_accel():
    shape = (256, 256)
    for name in MASKS:
        for r in (2, 3, 4, 6, 8):
            m = mask_partial_fourier(shape, frac=0.55) if name == "partial_fourier" \
                else make_mask(name, shape, r)
            assert realised_accel(m) == pytest.approx(256 / pe_lines(m).sum())
            assert realised_accel(m) == pytest.approx(256 / (m.sum() / 256))


def test_window_sidelobes():
    psl = {w: window_response(w)["peak_sidelobe_db"] for w in ("rect", "hann", "hamming")}
    assert psl["hamming"] < psl["hann"] < psl["rect"]
    assert psl["rect"] == pytest.approx(-13.26, abs=0.1)


def test_zerofill_is_psf_convolution(brain):
    """The full-size recon equals the image convolved with the mask PSF (CO1)."""
    x = apply_synthetic_phase(brain, "none")
    m = make_mask("uniform", brain.shape, 4)
    rec = recon_zerofill(make_kspace(x), m)
    # comb PSF: four equal taps at 0, +-64, 128 -> replicas of amplitude 1/4
    expected = sum(np.roll(x, s, axis=0) for s in (0, 64, 128, 192)) / 4
    assert np.max(np.abs(rec - expected)) < 1e-10
