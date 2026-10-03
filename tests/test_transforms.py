"""M1: centred unitary transforms and data generation."""

import numpy as np
import pytest

from src.data import apply_synthetic_phase, load_image, make_kspace
from src.transforms import (fft1c, fft2c, fft_radix2, hermitian_flip, ifft2c, naive_dft,
                            upsample_by_zeropad)


@pytest.fixture
def x():
    rng = np.random.default_rng(1)
    return rng.standard_normal((32, 48)) + 1j * rng.standard_normal((32, 48))


def test_roundtrip(x):
    assert np.max(np.abs(ifft2c(fft2c(x)) - x)) < 1e-10


def test_parseval(x):
    assert abs(np.linalg.norm(x) - np.linalg.norm(fft2c(x))) < 1e-10


def test_dc_is_centred():
    img = np.ones((16, 16))
    k = fft2c(img)
    assert np.argmax(np.abs(k)) == np.ravel_multi_index((8, 8), k.shape)
    assert np.isclose(np.abs(k[8, 8]), 16.0)  # sum / sqrt(N) = 256 / 16


def test_separability(x):
    """2D DFT = 1D DFT along rows then 1D DFT along columns (CO3)."""
    assert np.allclose(fft1c(fft1c(x, axis=1), axis=0), fft2c(x), atol=1e-12)


def test_hermitian_flip_index():
    k = np.arange(8)
    assert list(hermitian_flip(k)) == [0, 7, 6, 5, 4, 3, 2, 1]


def test_upsample_keeps_samples():
    img = np.random.default_rng(0).standard_normal((16, 16))
    up = upsample_by_zeropad(img, 2)
    assert np.allclose(up[::2, ::2], img, atol=1e-12)
    assert np.max(np.abs(up.imag)) < 1e-12


def test_reference_dfts_match_numpy():
    v = np.random.default_rng(0).standard_normal(256) + 0j
    assert np.allclose(naive_dft(v), np.fft.fft(v))
    assert np.allclose(fft_radix2(v), np.fft.fft(v))


@pytest.mark.parametrize("source", ["brain", "phantom"])
def test_load_image(source):
    img = load_image(source)
    assert img.shape == (256, 256) and img.min() >= 0 and np.isclose(img.max(), 1.0)


def test_smooth_phase_keeps_magnitude():
    img = load_image("phantom")
    xc = apply_synthetic_phase(img, "smooth")
    assert xc.dtype == np.complex128
    assert np.allclose(np.abs(xc), img)
    ph = np.angle(xc[img > 0.1])
    assert np.ptp(np.unwrap(ph)) > 3.0
    assert make_kspace(xc).dtype == np.complex128
