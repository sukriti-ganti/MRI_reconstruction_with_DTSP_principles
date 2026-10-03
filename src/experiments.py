"""Sweep orchestration for experiments E1-E7.

Every function takes the parsed YAML config and returns a tidy
``pandas.DataFrame``. ``scripts/run_all.py`` writes them to
``results/tables``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import data, masks, metrics, psf, recon
from .apodization import WINDOWS, window_response
from .transforms import fft_radix2, naive_dft

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Read ``config/experiments.yaml`` (or ``path``)."""
    path = Path(path) if path else ROOT / "config" / "experiments.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------- helpers

def load_images(cfg: dict) -> dict[str, np.ndarray]:
    """Real magnitude test images keyed by name."""
    return {
        im["name"]: data.load_image(im["source"], im.get("slice_idx", 4), im.get("size", 256))
        for im in cfg["images"]
    }


def mask_cases(cfg: dict, shape: tuple) -> list[dict]:
    """All (mask, accel/frac) cases of the E1 sweep, plus the full baseline."""
    e1 = cfg["e1"]
    cases = [{"mask": "full", "accel_requested": 1.0, "frac": np.nan,
              "m": masks.mask_full(shape)}]
    for name in e1["masks"]:
        kw = cfg["masks"].get(name, {})
        for r in e1["accels"]:
            cases.append({"mask": name, "accel_requested": float(r), "frac": np.nan,
                          "m": masks.make_mask(name, shape, r, **kw)})
    for f in e1["partial_fourier_fracs"]:
        cases.append({"mask": "partial_fourier", "accel_requested": 1.0 / f, "frac": f,
                      "m": masks.mask_partial_fourier(shape, frac=f)})
    return cases


def gallery_masks(cfg: dict, shape: tuple) -> dict[str, np.ndarray]:
    """The seven masks shown in F2-F5 (R = gallery_accel; PF at gallery_pf_frac)."""
    r = cfg["gallery_accel"]
    out = {}
    for name in ["full", "uniform", "uniform_acs", "random", "vardens", "partial_fourier",
                 "lowpass"]:
        kw = cfg["masks"].get(name, {})
        if name == "partial_fourier":
            out[name] = masks.mask_partial_fourier(shape, frac=cfg["gallery_pf_frac"])
        else:
            out[name] = masks.make_mask(name, shape, r, **kw)
    return out


def _mask_seed(cfg: dict, name: str) -> float:
    return cfg["masks"].get(name, {}).get("seed", np.nan)


def _run_recon(name: str, k: np.ndarray, m: np.ndarray, cfg: dict, **kw) -> np.ndarray:
    if name == "cs_fista":
        return recon.recon_cs_fista(k, m, **{**cfg["recon"]["cs_fista"], **kw})
    if name == "pocs":
        return recon.recon_pocs(k, m, n_iter=cfg["e5"]["pocs_iters"])
    return recon.RECONS[name](k, m, **kw)


def ringing_roi(cfg: dict, truth: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Edge band for the ringing index, in units of the truncation cell N/n_lines."""
    cell = masks.realised_accel(m)
    r = cfg["e4"]["ringing_roi"]
    return metrics.edge_band_roi(truth, d_min=r["d_min_cells"] * cell,
                                 d_max=r["d_max_cells"] * cell)


def _row(cfg, experiment, image, phase, case, rec_name, window, truth, rec):
    m = case["m"]
    roi = ringing_roi(cfg, truth, m)
    row = {
        "experiment": experiment,
        "image": image,
        "phase_mode": phase,
        "mask": case["mask"],
        "accel_requested": case["accel_requested"],
        "accel_realised": masks.realised_accel(m),
        "sampled_lines": int(masks.pe_lines(m).sum()),
        "frac": case.get("frac", np.nan),
        "recon": rec_name,
        "window": window,
        "seed": _mask_seed(cfg, case["mask"]),
    }
    row.update(metrics.all_metrics(truth, rec, roi))
    row.update(psf.psf_metrics(psf.compute_psf(m)))
    return row


# --------------------------------------------------------------------- E1

def run_e1(cfg: dict, images: dict | None = None) -> pd.DataFrame:
    """E1 acceleration sweep: every mask x R x recon x phase mode x image."""
    images = images or load_images(cfg)
    rows = []
    for iname, img in images.items():
        cases = mask_cases(cfg, img.shape)
        for phase in cfg["e1"]["phase_modes"]:
            x = data.apply_synthetic_phase(img, phase)
            k = data.make_kspace(x)
            truth = data.ground_truth(x)
            for case in cases:
                for rname in cfg["e1"]["recons"]:
                    rec = _run_recon(rname, k, case["m"], cfg)
                    rows.append(_row(cfg, "E1", iname, phase, case, rname, "rect", truth, rec))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- E2

def run_e2(cfg: dict, shape: tuple = (256, 256)) -> pd.DataFrame:
    """E2 PSF characterisation of every mask of the E1 sweep (image independent)."""
    rows = []
    for case in mask_cases(cfg, shape):
        p = psf.compute_psf(case["m"])
        rows.append({
            "mask": case["mask"],
            "accel_requested": case["accel_requested"],
            "frac": case["frac"],
            "accel_realised": masks.realised_accel(case["m"]),
            "sampled_lines": int(masks.pe_lines(case["m"]).sum()),
            "seed": _mask_seed(cfg, case["mask"]),
            **psf.psf_metrics(p),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- E3

def run_e3(e1: pd.DataFrame) -> pd.DataFrame:
    """E3 coherence vs error: rank correlations of predictors with NRMSE.

    For each image (phase "none") it reports the Spearman correlation of
    zero-fill NRMSE with realised R, with the PSF sidelobe energy fraction and
    with the coherence index, and the correlation of the coherence index with
    the fraction of the zero-fill error that CS-FISTA removes.
    """
    from scipy.stats import spearmanr

    rows = []
    sub = e1[(e1.phase_mode == "none") & (e1["mask"] != "full")]
    for iname, g in sub.groupby("image"):
        zf = g[g.recon == "zerofill"].set_index(["mask", "accel_requested"])
        cs = g[g.recon == "cs_fista"].set_index(["mask", "accel_requested"])
        gain = 1 - cs["nrmse"] / zf["nrmse"]
        aliasing = zf.index.get_level_values("mask").isin(["uniform", "uniform_acs", "random",
                                                            "vardens"])
        for pred in ["accel_realised", "sidelobe_energy_fraction", "coherence_index",
                     "peak_sidelobe_ratio"]:
            rows.append({"image": iname, "predictor": pred, "target": "zerofill_nrmse",
                         "subset": "all masks",
                         "spearman_rho": spearmanr(zf[pred], zf["nrmse"])[0]})
        for pred in ["coherence_index", "peak_sidelobe_ratio"]:
            rows.append({"image": iname, "predictor": pred, "target": "cs_error_reduction",
                         "subset": "aliasing masks",
                         "spearman_rho": spearmanr(zf[pred][aliasing], gain[aliasing])[0]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- E4

def run_e4(cfg: dict, images: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E4 apodization: low-pass frac x window (recon rows) + window DTFT table."""
    images = images or load_images(cfg)
    e4 = cfg["e4"]
    rows = []
    for iname, img in images.items():
        x = data.apply_synthetic_phase(img, "none")
        k = data.make_kspace(x)
        truth = data.ground_truth(x)
        for f in e4["lowpass_fracs"]:
            case = {"mask": "lowpass", "accel_requested": 1.0 / f, "frac": f,
                    "m": masks.mask_lowpass(img.shape, None, frac=f)}
            for win in e4["windows"]:
                kw = {"alpha": e4["tukey_alpha"]} if win == "tukey" else {}
                rec = recon.recon_apodized(k, case["m"], window=win, **kw)
                rows.append(_row(cfg, "E4", iname, "none", case, "apodized", win, truth, rec))
    resp = []
    for win in WINDOWS:
        kw = {"alpha": e4["tukey_alpha"]} if win == "tukey" else {}
        resp.append(window_response(win, n=e4["window_response_n"], **kw))
    return pd.DataFrame(rows), pd.DataFrame(resp)


# --------------------------------------------------------------------- E5

def run_e5(cfg: dict, images: dict | None = None) -> pd.DataFrame:
    """E5 partial Fourier: frac x {zero-fill, conjugate symmetry, POCS} x phase."""
    images = images or load_images(cfg)
    rows = []
    for iname, img in images.items():
        for phase in cfg["phase_modes"]:
            x = data.apply_synthetic_phase(img, phase)
            k = data.make_kspace(x)
            truth = data.ground_truth(x)
            for f in cfg["e5"]["fracs"]:
                case = {"mask": "partial_fourier", "accel_requested": 1.0 / f, "frac": f,
                        "m": masks.mask_partial_fourier(img.shape, frac=f)}
                for rname in cfg["e5"]["recons"]:
                    rec = _run_recon(rname, k, case["m"], cfg)
                    rows.append(_row(cfg, "E5", iname, phase, case, rname, "rect", truth, rec))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- E6

def _time(fn, x, repeats: int) -> float:
    best = np.inf
    for _ in range(repeats):
        # repeat fast calls enough times to beat timer resolution
        n_inner, t = 1, 0.0
        while True:
            t0 = time.perf_counter()
            for _ in range(n_inner):
                fn(x)
            t = time.perf_counter() - t0
            if t > 2e-3 or n_inner >= 1 << 16:
                break
            n_inner *= 4
        best = min(best, t / n_inner)
    return best


def run_e6(cfg: dict) -> pd.DataFrame:
    """E6 FFT complexity benchmark: naive O(N^2) DFT vs radix-2 FFT vs numpy."""
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for n in cfg["e6"]["sizes"]:
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        rows.append({
            "N": n,
            "naive_dft_s": _time(naive_dft, x, cfg["e6"]["repeats"]),
            "radix2_fft_s": _time(fft_radix2, x, cfg["e6"]["repeats"]),
            "numpy_fft_s": _time(np.fft.fft, x, cfg["e6"]["repeats"]),
            "N2": n * n,
            "NlogN": n * np.log2(n),
        })
    df = pd.DataFrame(rows)
    df["speedup_numpy_vs_naive"] = df.naive_dft_s / df.numpy_fft_s
    df["speedup_radix2_vs_naive"] = df.naive_dft_s / df.radix2_fft_s
    return df


# --------------------------------------------------------------------- E7

def measure_ghost_shift_image(recon_img: np.ndarray, truth: np.ndarray) -> float:
    """Ghost displacement (PE pixels) measured in the image domain.

    The in-place component ``alpha*x`` (least squares) is removed from the
    zero-filled reconstruction; the residual is then phase-correlated with the
    truth over every circular PE offset ``s``. The aliased replicas
    make the correlation peak at the ghost displacement; the peak nearest the
    centre is refined to sub-pixel accuracy with a parabola.
    """
    x = np.asarray(truth, dtype=np.complex128)
    y = np.asarray(recon_img, dtype=np.complex128)
    alpha = np.vdot(x, y) / np.vdot(x, x)
    res = y - alpha * x
    n = x.shape[0]
    # phase correlation along PE: whitening the cross-power spectrum removes
    # the broad autocorrelation of the anatomy and leaves sharp shift peaks
    X = np.fft.fft(x, axis=0)
    Rr = np.fft.fft(res, axis=0)
    cross = (Rr * np.conj(X)).sum(axis=1)
    cross /= np.maximum(np.abs(cross), 1e-12 * np.abs(cross).max())
    corr = np.abs(np.fft.ifft(cross))
    corr[0] = 0
    is_max = (corr >= np.roll(corr, 1)) & (corr >= np.roll(corr, -1))
    cand = np.flatnonzero(is_max & (corr >= 0.5 * corr.max()))
    dist = np.minimum(cand, n - cand)
    i = int(cand[np.argmin(dist)])
    a, b, d = corr[(i - 1) % n], corr[i], corr[(i + 1) % n]
    den = a - 2 * b + d
    delta = 0.5 * (a - d) / den if den != 0 else 0.0
    s = i + delta
    return float(min(s, n - s))


def run_e7(cfg: dict, images: dict | None = None) -> pd.DataFrame:
    """E7 aliasing period: measured ghost spacing vs the predicted N/R."""
    images = images or load_images(cfg)
    rows = []
    for iname, img in images.items():
        x = data.apply_synthetic_phase(img, "none")
        k = data.make_kspace(x)
        n = img.shape[0]
        for r in cfg["e7"]["accels"]:
            m = masks.mask_uniform(img.shape, r)
            p = psf.compute_psf(m)
            rec = recon.recon_zerofill(k, m)
            meas_psf = psf.ghost_spacing_from_psf(p)
            meas_img = measure_ghost_shift_image(rec, x)
            rows.append({
                "image": iname, "N": n, "R": r,
                "accel_realised": masks.realised_accel(m),
                "predicted_N_over_R": n / r,
                "measured_psf_px": meas_psf,
                "measured_image_px": meas_img,
                "abs_error_psf_px": abs(meas_psf - n / r),
                "abs_error_image_px": abs(meas_img - n / r),
                "n_psf_peaks": int(psf.find_psf_peaks(p, 0.5).size),
            })
    return pd.DataFrame(rows)
