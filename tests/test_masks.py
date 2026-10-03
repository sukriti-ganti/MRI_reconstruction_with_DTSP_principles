"""M2: undersampling masks are phase-encode line patterns with honest R."""

import numpy as np
import pytest

from src.masks import MASKS, make_mask, mask_partial_fourier, pe_lines, realised_accel

SHAPE = (256, 256)


def _all_masks():
    for name in MASKS:
        if name == "partial_fourier":
            for f in (0.55, 0.625, 0.75):
                yield name, mask_partial_fourier(SHAPE, frac=f)
        else:
            for r in (2, 3, 4, 6, 8):
                yield name, make_mask(name, SHAPE, r)


@pytest.mark.parametrize("name,m", list(_all_masks()))
def test_masks_are_line_masks(name, m):
    assert m.shape == SHAPE and m.dtype == bool
    pe_lines(m)  # raises if any row is partially sampled
    assert m[SHAPE[0] // 2].all(), "DC line must be sampled"


@pytest.mark.parametrize("name", ["random", "vardens", "lowpass"])
@pytest.mark.parametrize("r", [2, 3, 4, 6, 8])
def test_exact_line_budget(name, r):
    m = make_mask(name, SHAPE, r)
    assert pe_lines(m).sum() == round(SHAPE[0] / r)


def test_uniform_spacing():
    lines = np.flatnonzero(pe_lines(make_mask("uniform", SHAPE, 4)))
    assert np.all(np.diff(lines) == 4)


def test_random_is_seeded():
    a = make_mask("random", SHAPE, 4, seed=3)
    b = make_mask("random", SHAPE, 4, seed=3)
    c = make_mask("random", SHAPE, 4, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_acs_adds_centre_block():
    m = pe_lines(make_mask("uniform_acs", SHAPE, 4, acs_lines=24))
    assert m[128 - 12:128 + 12].all()
    assert realised_accel(make_mask("uniform_acs", SHAPE, 4)) < 4


def test_partial_fourier_rejects_low_frac():
    with pytest.raises(ValueError):
        mask_partial_fourier(SHAPE, frac=0.4)
