"""M3: PSF analysis."""

import numpy as np
import pytest

from src.masks import make_mask
from src.psf import compute_psf, find_psf_peaks, ghost_spacing_from_psf, psf_metrics

SHAPE = (256, 256)


def test_full_mask_psf_is_delta():
    p = compute_psf(make_mask("full", SHAPE))
    mag = np.abs(p)
    assert np.isclose(mag[128, 128], np.sqrt(256 * 256))
    mag[128, 128] = 0
    assert mag.max() < 1e-9
    assert psf_metrics(p)["coherence_index"] == 0.0


def test_psf_is_delta_along_readout():
    p = compute_psf(make_mask("random", SHAPE, 4))
    off = np.abs(p).copy()
    off[:, 128] = 0
    assert off.max() < 1e-9


def test_uniform_is_coherent_random_is_not():
    cu = psf_metrics(compute_psf(make_mask("uniform", SHAPE, 4)))
    cr = psf_metrics(compute_psf(make_mask("random", SHAPE, 4)))
    cv = psf_metrics(compute_psf(make_mask("vardens", SHAPE, 4)))
    assert cu["peak_sidelobe_ratio"] == pytest.approx(1.0)
    assert cu["coherence_index"] > 10 * max(cr["coherence_index"], cv["coherence_index"])


def test_lowpass_psf_sidelobe_is_dirichlet():
    """Truncation PSF: first sidelobe of a Dirichlet kernel ~ 0.217 (-13.3 dB)."""
    m = psf_metrics(compute_psf(make_mask("lowpass", SHAPE, 4)))
    assert m["peak_sidelobe_ratio"] == pytest.approx(0.217, abs=0.01)
    assert m["mainlobe_fwhm"] == pytest.approx(1.207 * 4, rel=0.05)


@pytest.mark.parametrize("r", [2, 3, 4, 6, 8])
def test_ghost_spacing_within_one_pixel_n256(r):
    p = compute_psf(make_mask("uniform", SHAPE, r))
    assert abs(ghost_spacing_from_psf(p) - 256 / r) < 1.0
    assert find_psf_peaks(p).size == r
