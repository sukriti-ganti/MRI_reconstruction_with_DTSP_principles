"""Image quality metrics.

All metrics compare **magnitude** images. Both images are divided by the
reference maximum, so the reference lies in [0, 1] and the test image keeps
its true relative scale (an intensity loss caused by discarding the DC
energy is an error, not something to normalise away).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


def _prep(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.abs(np.asarray(ref))
    test = np.abs(np.asarray(test))
    s = ref.max()
    return ref / s, test / s


def nrmse(ref: np.ndarray, test: np.ndarray) -> float:
    """``||test - ref||_2 / ||ref||_2`` on normalised magnitudes."""
    r, t = _prep(ref, test)
    return float(np.linalg.norm(t - r) / np.linalg.norm(r))


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Peak SNR in dB with peak 1 (the reference maximum)."""
    r, t = _prep(ref, test)
    mse = np.mean((t - r) ** 2)
    return float(np.inf if mse == 0 else 10 * np.log10(1.0 / mse))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """Structural similarity (skimage), data range 1."""
    r, t = _prep(ref, test)
    return float(structural_similarity(r, t, data_range=1.0))


def artifact_power(ref: np.ndarray, test: np.ndarray) -> float:
    """Energy of the error image normalised by reference energy (= NRMSE^2)."""
    r, t = _prep(ref, test)
    return float(np.sum((t - r) ** 2) / np.sum(r**2))


def edge_preservation(ref: np.ndarray, test: np.ndarray) -> float:
    """Pearson correlation between the Sobel gradient magnitudes."""
    r, t = _prep(ref, test)

    def grad(x: np.ndarray) -> np.ndarray:
        return np.hypot(ndimage.sobel(x, axis=0), ndimage.sobel(x, axis=1)).ravel()

    return float(np.corrcoef(grad(r), grad(t))[0, 1])


def edge_band_roi(ref: np.ndarray, d_min: float = 3.0, d_max: float = 20.0,
                  axis: int = 0, threshold: float = 0.05) -> np.ndarray:
    """Pixels lying ``d_min..d_max`` px (along ``axis``) from a sharp edge.

    Edges are steps of the normalised reference larger than ``threshold``
    along the PE axis; that is the direction in which k_y truncation rings.
    Pixels within ``d_min`` of an edge are excluded so the band measures the
    ripple, not the blurred edge itself.
    """
    r = np.abs(ref) / np.abs(ref).max()
    step = np.abs(np.diff(r, axis=axis)) > threshold
    edges = np.zeros_like(r, dtype=bool)
    if axis == 0:
        edges[:-1] |= step
        edges[1:] |= step
        sampling = (1.0, 1e6)  # distance only along PE
    else:
        edges[:, :-1] |= step
        edges[:, 1:] |= step
        sampling = (1e6, 1.0)
    dist = ndimage.distance_transform_edt(~edges, sampling=sampling)
    return (dist >= d_min) & (dist <= d_max)


def ringing_index(img: np.ndarray, roi: np.ndarray, ref: np.ndarray | None = None) -> float:
    """Oscillation amplitude near sharp edges (Gibbs ringing).

    RMS deviation inside ``roi`` (typically :func:`edge_band_roi`). With a
    reference it is the RMS of ``img - ref``; without one it is the RMS of
    ``img`` minus its local mean along the PE axis (a 9-px moving average).
    Values are in units of the reference / image maximum.
    """
    if ref is not None:
        r, t = _prep(ref, img)
        dev = t - r
    else:
        t = np.abs(img) / np.abs(img).max()
        dev = t - ndimage.uniform_filter1d(t, 9, axis=0)
    return float(np.sqrt(np.mean(dev[roi] ** 2)))


def all_metrics(ref: np.ndarray, test: np.ndarray, roi: np.ndarray | None = None) -> dict:
    out = {
        "nrmse": nrmse(ref, test),
        "psnr": psnr(ref, test),
        "ssim": ssim(ref, test),
        "artifact_power": artifact_power(ref, test),
        "edge_preservation": edge_preservation(ref, test),
    }
    if roi is not None:
        out["ringing_index"] = ringing_index(test, roi, ref)
    return out
