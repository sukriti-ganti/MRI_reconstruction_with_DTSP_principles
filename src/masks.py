"""Cartesian k-space undersampling masks (CO1 impulse trains, CO3 sampling).

Physics: in 2D Cartesian MRI the readout (axis 1) is acquired for free; only
**phase-encode lines** (rows, axis 0) cost scan time. Every mask is therefore
a 1D pattern over PE lines broadcast across the readout direction, so masks
look like horizontal stripes, never like point noise.

Every function returns a boolean ``(H, W)`` array. Rounding means the realised
acceleration rarely equals the requested one, so use :func:`realised_accel`
(``total_lines / sampled_lines``) for all comparisons.

Centred indexing: PE line ``i`` holds frequency ``k_y = i - H//2``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _broadcast(lines: np.ndarray, shape: tuple) -> np.ndarray:
    h, w = shape
    lines = np.asarray(lines, dtype=bool)
    assert lines.shape == (h,)
    return np.repeat(lines[:, None], w, axis=1)


def _ky(h: int) -> np.ndarray:
    return np.arange(h) - h // 2


def pe_lines(mask: np.ndarray) -> np.ndarray:
    """1D boolean PE-line pattern of a line mask (asserts rows are all-or-none)."""
    mask = np.asarray(mask, dtype=bool)
    rows = mask.any(axis=1)
    if not np.array_equal(mask, _broadcast(rows, mask.shape)):
        raise ValueError("mask is not a phase-encode line mask")
    return rows


def realised_accel(mask: np.ndarray) -> float:
    """Actual acceleration ``total_PE_lines / sampled_PE_lines``."""
    rows = pe_lines(mask)
    return rows.size / max(int(rows.sum()), 1)


def mask_full(shape: tuple, accel: float = 1.0, **kwargs) -> np.ndarray:
    """Fully sampled baseline (R = 1). PSF is a Kronecker delta."""
    return np.ones(shape, dtype=bool)


def mask_uniform(shape: tuple, accel: float, **kwargs) -> np.ndarray:
    """Regular decimation: every R-th PE line, aligned so DC is sampled.

    DTSP (CO1/CO3): the line pattern is an impulse train (comb) of period R.
    Its IDFT is a comb of period N/R, so the image is replicated R times at
    spacing FOV/R -> coherent ghosts (violates Delta_k <= 1/FOV).
    """
    r = int(round(accel))
    lines = (_ky(shape[0]) % r) == 0
    return _broadcast(lines, shape)


def mask_uniform_acs(shape: tuple, accel: float, acs_lines: int = 24, **kwargs) -> np.ndarray:
    """Regular decimation plus a fully sampled central auto-calibration block.

    The ACS block restores the low-frequency (contrast) energy, so the ghosts
    carry less of the image energy; realised R is lower than requested.
    """
    lines = pe_lines(mask_uniform(shape, accel))
    ky = _ky(shape[0])
    lines |= (ky >= -(acs_lines // 2)) & (ky < acs_lines - acs_lines // 2)
    return _broadcast(lines, shape)


def _n_target(h: int, accel: float) -> int:
    return int(np.clip(round(h / accel), 1, h))


def mask_random(shape: tuple, accel: float, seed: int = 0, include_dc: bool = True,
                **kwargs) -> np.ndarray:
    """Uniform-density random PE lines (exactly round(H/R) lines).

    DTSP: random (non-periodic) sampling has no comb structure, so its PSF has
    no discrete replicas; the sidelobe energy is spread as noise-like,
    *incoherent* aliasing. The DC line is kept so mean intensity is preserved.
    """
    h = shape[0]
    rng = np.random.default_rng(seed)
    n = _n_target(h, accel)
    lines = np.zeros(h, dtype=bool)
    pool = np.arange(h)
    if include_dc:
        lines[h // 2] = True
        n -= 1
        pool = pool[pool != h // 2]
    lines[rng.choice(pool, size=max(n, 0), replace=False)] = True
    return _broadcast(lines, shape)


def mask_vardens(shape: tuple, accel: float, center_frac: float = 0.08,
                 poly_order: float = 3.0, seed: int = 0, **kwargs) -> np.ndarray:
    """Variable-density random PE lines (exactly round(H/R) lines).

    A fully sampled centre (``center_frac`` of the lines) plus random lines
    drawn without replacement with probability ~ (1 - |k|/k_max)^poly_order.
    DTSP: concentrates samples where the DFT energy of natural images is
    compacted (CO3), while keeping the PSF sidelobes incoherent.
    """
    h = shape[0]
    rng = np.random.default_rng(seed)
    n = _n_target(h, accel)
    ky = _ky(h)
    n_c = min(int(round(center_frac * h)), n)
    lines = (ky >= -(n_c // 2)) & (ky < n_c - n_c // 2)
    n_rem = n - int(lines.sum())
    if n_rem > 0:
        pool = np.flatnonzero(~lines)
        kmax = h / 2 + 1
        p = (1.0 - np.abs(ky[pool]) / kmax) ** poly_order
        lines[rng.choice(pool, size=n_rem, replace=False, p=p / p.sum())] = True
    return _broadcast(lines, shape)


def mask_partial_fourier(shape: tuple, accel: float | None = None,
                         frac: float | None = None, **kwargs) -> np.ndarray:
    """Partial Fourier: one half of k_y plus a fraction past the centre.

    Lines with ``i < round(frac*H)`` are acquired (frac > 0.5). R = 1/frac.
    DTSP (CO3): for a real image the missing half is recoverable from
    Hermitian symmetry K[-k] = K*[k]. Either ``frac`` or ``accel`` is given.
    """
    h = shape[0]
    if frac is None:
        if accel is None:
            raise ValueError("give frac or accel")
        frac = 1.0 / accel
    if not 0.5 < frac <= 1.0:
        raise ValueError("partial Fourier needs 0.5 < frac <= 1")
    n_s = int(round(frac * h))
    lines = np.arange(h) < n_s
    return _broadcast(lines, shape)


def mask_lowpass(shape: tuple, accel: float, frac: float | None = None, **kwargs) -> np.ndarray:
    """Centric low-pass: keep only the central round(H/R) (or frac*H) lines.

    DTSP (CO3/CO4): this *truncates* k-space, i.e. multiplies it by a
    rectangular window. The PSF is a Dirichlet (periodic sinc) kernel whose
    -13 dB sidelobes produce Gibbs ringing, and whose mainlobe width sets the
    blur. There is no replication, so there is no aliasing.
    """
    h = shape[0]
    n = int(round(frac * h)) if frac is not None else _n_target(h, accel)
    ky = _ky(h)
    lines = (ky >= -(n // 2)) & (ky < n - n // 2)
    return _broadcast(lines, shape)


MASKS: dict[str, Callable[..., np.ndarray]] = {
    "full": mask_full,
    "uniform": mask_uniform,
    "uniform_acs": mask_uniform_acs,
    "random": mask_random,
    "vardens": mask_vardens,
    "partial_fourier": mask_partial_fourier,
    "lowpass": mask_lowpass,
}


def make_mask(name: str, shape: tuple, accel: float | None = None, **kwargs) -> np.ndarray:
    """Dispatch to ``mask_<name>``."""
    if name not in MASKS:
        raise ValueError(f"unknown mask {name!r}")
    return MASKS[name](shape, accel, **kwargs)
