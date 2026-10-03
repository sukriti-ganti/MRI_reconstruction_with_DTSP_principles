"""Generate figures F1-F12 from the result tables and the src/ modules.

Usage: python scripts/make_figures.py [--config config/experiments.yaml]

Each figure is a 300 dpi PNG in results/figures with its caption printed
under the plot and collected in results/figures/captions.md. Every caption
names the DTSP principle the figure demonstrates and the number it measures.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, masks, psf, recon  # noqa: E402
from src.apodization import WINDOWS, window_spectrum_db  # noqa: E402
from src.experiments import gallery_masks, load_config, load_images  # noqa: E402
from src.metrics import nrmse  # noqa: E402
from src.transforms import fft2c, ifft2c  # noqa: E402

# Fixed categorical order (colour follows the entity, never its rank) plus a
# distinct marker per mask so identity never depends on colour alone.
STYLE = {
    "uniform": ("#2a78d6", "o", "Uniform"),
    "uniform_acs": ("#eb6834", "s", "Uniform + ACS"),
    "random": ("#1baf7a", "^", "Random"),
    "vardens": ("#eda100", "D", "Variable density"),
    "partial_fourier": ("#e87ba4", "v", "Partial Fourier"),
    "lowpass": ("#008300", "P", "Low-pass (centric)"),
    "full": ("#4a3aa7", "*", "Fully sampled"),
}
WIN_COLORS = {"rect": "#2a78d6", "hann": "#eb6834", "hamming": "#1baf7a",
              "tukey": "#e87ba4", "blackman": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 2.0, "image.cmap": "gray",
})

CAPTIONS: dict[str, str] = {}


def save(fig, fid: str, name: str, caption: str, outdir: Path, dpi: int) -> None:
    CAPTIONS[fid] = caption
    wrapped = "\n".join(textwrap.wrap(f"{fid}. {caption}", 150))
    fig.text(0.01, 0.005, wrapped, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.savefig(outdir / f"{fid}_{name}.png", dpi=dpi)
    plt.close(fig)


def _img_ax(ax, im, title, vmax=None, cmap="gray"):
    ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax if vmax is not None else im.max())
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _label(name, m):
    return f"{STYLE[name][2]}\nR = {masks.realised_accel(m):.2f}"


# ---------------------------------------------------------------- F1

def fig_f1(img, out, dpi):
    k = fft2c(img)
    logk = np.log10(np.abs(k) + 1e-6)
    e = np.abs(k) ** 2
    ky, kx = np.meshgrid(*(np.arange(s) - s // 2 for s in img.shape), indexing="ij")
    r = np.hypot(ky, kx)
    frac_central = e[r <= 0.1 * img.shape[0] / 2].sum() / e.sum()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.14, wspace=0.05)
    _img_ax(axs[0], img, "Ground truth |x|")
    axs[1].imshow(logk, cmap="magma")
    axs[1].set_title("log10 |K| (centred 2D DFT)")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].add_patch(plt.Circle((img.shape[1] // 2, img.shape[0] // 2),
                                0.1 * img.shape[0] / 2, fill=False, color="w", lw=1))
    save(fig, "F1", "ground_truth_kspace",
         f"2D DFT and energy compaction (CO3). Brain slice and its centred log-magnitude "
         f"k-space. The circle of radius 0.1*k_max (0.8% of the k-space area) holds "
         f"{100 * frac_central:.1f}% of the signal energy (Parseval).", out, dpi)


# ---------------------------------------------------------------- F2-F5

def fig_f2_to_f5(cfg, img, out, dpi):
    gm = gallery_masks(cfg, img.shape)
    k = fft2c(img)
    names = list(gm)

    fig, axs = plt.subplots(1, 7, figsize=(14, 3.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.17, wspace=0.08)
    for ax, n in zip(axs, names):
        ax.imshow(gm[n], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(_label(n, gm[n]), fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "F2", "mask_gallery",
         "Mask gallery at R = 4 (partial Fourier at frac 0.55, R = 1.82, its practical "
         "limit). White = acquired phase-encode line; every mask is a 1D pattern over k_y "
         "broadcast along the readout k_x, so all masks are horizontal stripes. "
         "Titles give the realised R = lines / sampled lines.", out, dpi)

    fig, axs = plt.subplots(2, 7, figsize=(14, 5.4),
                            gridspec_kw={"height_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.2, wspace=0.3,
                        hspace=0.3)
    for j, n in enumerate(names):
        p = psf.compute_psf(gm[n])
        mag = np.abs(p) / np.abs(p).max()
        # the PSF is a delta along readout: show a 64-px wide strip around it
        c = img.shape[1] // 2
        strip = np.log10(mag[:, c - 32:c + 32] + 1e-4)
        axs[0, j].imshow(strip, cmap="magma", vmin=-4, vmax=0, aspect="auto")
        met = psf.psf_metrics(p)
        axs[0, j].set_title(f"{STYLE[n][2]}\nCI={met['coherence_index']:.1f}  "
                            f"PSR={met['peak_sidelobe_ratio']:.2f}", fontsize=8)
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        prof = psf.psf_profile(p)
        y = np.arange(prof.size) - prof.size // 2
        axs[1, j].plot(y, 20 * np.log10(prof + 1e-6), color=STYLE[n][0], lw=1)
        axs[1, j].set_ylim(-60, 3)
        axs[1, j].set_xlim(-128, 128)
        axs[1, j].set_xticks([-100, 0, 100])
        if j == 0:
            axs[1, j].set_ylabel("|PSF| (dB)")
        axs[1, j].set_xlabel("PE offset (px)")
    save(fig, "F3", "psf_gallery",
         "PSF gallery = IDFT of each mask (convolution theorem, CO1). Top: log10 |PSF| "
         "(a 64-px strip; the PSF is a delta along readout). Bottom: PE profile in dB. "
         "Uniform: comb of R equal peaks at multiples of N/R (CI = 84, PSR = 1). Random and "
         "variable density: thin, noise-like sidelobe floor (CI < 3). Low-pass: Dirichlet "
         "kernel, -13 dB first sidelobe. Partial Fourier: a broadened, complex mainlobe.",
         out, dpi)

    recs = {n: np.abs(recon.recon_zerofill(k, gm[n])) for n in names}
    fig, axs = plt.subplots(1, 7, figsize=(14, 2.95))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.17, wspace=0.05)
    for ax, n in zip(axs, names):
        _img_ax(ax, recs[n], f"{STYLE[n][2]}\nNRMSE={nrmse(img, recs[n]):.3f}", vmax=1.0)
        ax.title.set_fontsize(8.5)
    save(fig, "F4", "recon_gallery",
         "Zero-filled reconstructions at R = 4 (the image convolved with each F3 PSF). "
         "Uniform: coherent aliasing from a comb PSF, R-1 = 3 sharp replicas displaced by "
         "N/R = 64 px and wrapped circularly. Random: incoherent, noise-like aliasing from "
         "a flat sidelobe floor. Low-pass: truncation, so blur plus Gibbs ringing but no "
         "replicas.", out, dpi)

    fig, axs = plt.subplots(1, 7, figsize=(14, 2.95))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.17, wspace=0.05)
    for ax, n in zip(axs, names):
        err = np.abs(recs[n] - img)
        _img_ax(ax, err, f"{STYLE[n][2]}\nmax |err|={err.max():.2f}", vmax=0.5,
                cmap="inferno")
        ax.title.set_fontsize(8.5)
    save(fig, "F5", "error_maps",
         "Error maps | |recon| - |truth| | at R = 4, common scale [0, 0.5]. The error "
         "is (PSF - delta) convolved with the image: discrete shifted copies of the "
         "anatomy for the comb PSF (uniform), diffuse texture for random masks, and "
         "error confined to edges for low-pass and partial Fourier.", out, dpi)


# ---------------------------------------------------------------- F6

def fig_f6(e1, out, dpi):
    d = e1[(e1.recon == "zerofill") & (e1.phase_mode == "none") & (e1["mask"] != "full")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.94, bottom=0.12, hspace=0.18,
                        wspace=0.22)
    for i, image in enumerate(["brain", "phantom"]):
        g = d[d.image == image]
        for j, met in enumerate(["nrmse", "ssim"]):
            ax = axs[j, i]
            for n, (col, mk, lab) in STYLE.items():
                s = g[g["mask"] == n].sort_values("accel_realised")
                if s.empty:
                    continue
                ax.plot(s.accel_realised, s[met], color=col, marker=mk, ms=6, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1.33, 2, 3, 4, 6, 8])
            ax.set_xticklabels(["1.33", "2", "3", "4", "6", "8"])
            ax.grid(True, axis="y")
            ax.set_ylabel(met.upper())
            if j == 0:
                ax.set_title(image)
            else:
                ax.set_xlabel("realised acceleration R")
    axs[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "F6", "quality_vs_acceleration",
         "Quality-acceleration trade-off, zero-filled recon, plotted against realised R. "
         "At R = 4 on the brain, NRMSE ranges from 0.12 (low-pass) to 0.71 (uniform): "
         "the same amount of discarded data gives a 6x spread in error, so how lines are "
         "chosen matters more than how many.", out, dpi)


# ---------------------------------------------------------------- F7

def fig_f7(e1, e3, out, dpi, image="brain"):
    d = e1[(e1.phase_mode == "none") & (e1.image == image) & (e1["mask"] != "full")]
    zf = d[d.recon == "zerofill"].set_index(["mask", "accel_requested"])
    cs = d[d.recon == "cs_fista"].set_index(["mask", "accel_requested"])
    zf = zf.assign(cs_nrmse=cs["nrmse"])
    zf["gain"] = 1 - zf.cs_nrmse / zf.nrmse
    rho = e3[e3.image == image].set_index(["predictor", "target"])["spearman_rho"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.subplots_adjust(left=0.05, right=0.86, top=0.86, bottom=0.2, wspace=0.28)
    for (n, _), r in zf.iterrows():
        col, mk, _ = STYLE[n]
        size = 25 + 12 * r.accel_realised
        kw = dict(color=col, marker=mk, s=size, edgecolor="white", linewidth=1.0, zorder=3)
        axs[0].scatter(r.coherence_index, r.nrmse, **kw)
        axs[1].scatter(r.sidelobe_energy_fraction, r.nrmse, **kw)
        if n in ("uniform", "uniform_acs", "random", "vardens"):
            axs[2].scatter(r.coherence_index, r.gain, **kw)
    # arrows zero-fill -> CS on panel (a)
    for (n, _), r in zf.iterrows():
        axs[0].annotate("", xy=(r.coherence_index, r.cs_nrmse),
                        xytext=(r.coherence_index, r.nrmse),
                        arrowprops=dict(arrowstyle="->", color=STYLE[n][0], lw=0.8,
                                        alpha=0.6))
    axs[0].set_xscale("log")
    axs[0].set_xlim(1.5, 400)
    axs[0].set_xlabel("coherence index (PSF sidelobe peak / mean)")
    axs[0].set_ylabel("NRMSE")
    axs[0].set_title("(a) PSF coherence vs error\nmarker = zero-fill, arrow tip = CS-FISTA")
    axs[0].axvspan(1.5, 4, color=GRID, alpha=0.5, zorder=0)
    axs[0].text(1.6, 0.02, "incoherent", color=MUTED, fontsize=8)
    axs[0].text(30, 0.02, "coherent", color=MUTED, fontsize=8)
    axs[1].set_xlabel("PSF sidelobe energy fraction")
    axs[1].set_ylabel("zero-fill NRMSE")
    axs[1].set_title(
        f"(b) leaked energy predicts error\nSpearman rho: sidelobe energy "
        f"{rho[('sidelobe_energy_fraction', 'zerofill_nrmse')]:.2f}, realised R "
        f"{rho[('accel_realised', 'zerofill_nrmse')]:.2f}")
    axs[2].set_xscale("log")
    axs[2].set_xlim(1.5, 400)
    axs[2].axhline(0, color=MUTED, lw=0.8)
    axs[2].set_xlabel("coherence index")
    axs[2].set_ylabel("fraction of zero-fill error removed by CS")
    axs[2].set_title(
        f"(c) coherence decides recoverability\nSpearman rho (aliasing masks) "
        f"{rho[('coherence_index', 'cs_error_reduction')]:.2f}")
    for ax in axs:
        ax.grid(True)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", ms=7, label=lab)
               for k, (c, m, lab) in STYLE.items() if k != "full"]
    axs[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  title="marker size ~ R", title_fontsize=8)
    save(fig, "F7", "coherence_vs_error",
         f"Core thesis, {image}, all masks and R. (a) Every mask placed on the "
         "coherence-index / NRMSE plane: coherent masks (uniform, CI >= 18) sit top-right "
         "and CS-FISTA cannot move them (arrows of zero length), incoherent masks (CI < 3) "
         "move down. (b) Zero-fill NRMSE is set by the PSF sidelobe energy (Parseval), a "
         "PSF-shape property that ranks the error far better than R. (c) The coherence "
         "index decides whether the leaked energy is removable: comb PSFs give ~0 gain "
         "(aliasing masks only; low-pass truncation has no aliasing for CS to remove and "
         "CS increases its error by 5-153% instead, see results.csv).", out, dpi)


# ---------------------------------------------------------------- F8

def fig_f8(img, e7, out, dpi):
    rs = sorted(e7.R.unique())
    k = fft2c(img)
    n = img.shape[0]
    e = e7[e7.image == "brain"].set_index("R")
    fig, axs = plt.subplots(2, len(rs), figsize=(15, 6.4),
                            gridspec_kw={"height_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.14, wspace=0.15,
                        hspace=0.3)
    for j, r in enumerate(rs):
        m = masks.mask_uniform(img.shape, r)
        rec = np.abs(recon.recon_zerofill(k, m))
        _img_ax(axs[0, j], rec, f"R = {r}", vmax=rec.max())
        meas = e.loc[r, "measured_image_px"]
        for g in range(1, int(np.ceil(r))):
            y = (n // 2 + g * meas) % n
            axs[0, j].axhline(y, color="#eda100", lw=0.7, ls="--")
        axs[0, j].set_title(f"R = {r}: N/R = {n / r:.2f}, measured {meas:.2f} px",
                            fontsize=8.5)
        prof = psf.psf_profile(psf.compute_psf(m))
        y = np.arange(n) - n // 2
        axs[1, j].plot(y, prof, color=STYLE["uniform"][0], lw=1.2)
        for g in range(-int(r), int(r) + 1):
            axs[1, j].axvline(g * n / r, color="#eda100", lw=0.6, ls=":")
        axs[1, j].set_xlim(-n // 2, n // 2)
        axs[1, j].set_xlabel("PE offset (px)")
        if j == 0:
            axs[1, j].set_ylabel("|PSF| / peak")
    err = e[["abs_error_psf_px", "abs_error_image_px"]].to_numpy().max()
    save(fig, "F8", "ghost_spacing",
         f"Sampling theorem in frequency (CO3). Decimating k_y by R (Delta_k = R/FOV) "
         f"replicates the image with period FOV/R. Top: zero-filled uniform recon with "
         f"the ghost positions measured by phase correlation (dashed). Bottom: PSF "
         f"profile with predicted replica positions N/R (dotted). Max |measured - N/R| "
         f"= {err:.2f} px over R = 2..8; for R = 3, 6 (R does not divide 256) the comb is "
         f"not exactly periodic and the realised R is 3.01 / 5.95.", out, dpi)


# ---------------------------------------------------------------- F9

def fig_f9(cfg, img, e4, wresp, out, dpi):
    e4c = cfg["e4"]
    k = fft2c(img)
    frac = min(e4c["lowpass_fracs"])
    m = masks.mask_lowpass(img.shape, None, frac=frac)
    col = img.shape[1] // 2
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], left=0.06, right=0.98,
                          top=0.95, bottom=0.12, hspace=0.38, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    for w in WINDOWS:
        kw = {"alpha": e4c["tukey_alpha"]} if w == "tukey" else {}
        f, db = window_spectrum_db(w, n=e4c["window_response_n"], **kw)
        ax0.plot(f, db, color=WIN_COLORS[w], lw=1.2, label=w)
    ax0.set_xlim(0, 16)
    ax0.set_ylim(-100, 3)
    ax0.set_xlabel("frequency (DFT bins of the length-64 window)")
    ax0.set_ylabel("|W| (dB)")
    ax0.set_title("Window DTFTs (zero-padded FFT)")
    ax0.grid(True)
    ax0.legend(ncol=2, fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ring = e4[(e4.image == "phantom") & (e4.frac == frac)].set_index("window")
    ring2 = e4[(e4.image == "phantom") & (e4.frac == max(e4c["lowpass_fracs"]))].set_index(
        "window")
    rows = []
    for _, r in wresp.iterrows():
        w = r.window
        rows.append([w, f"{r.mainlobe_width_null_bins:.2f}", f"{r.peak_sidelobe_db:.1f}",
                     f"{1e3 * ring.loc[w, 'ringing_index']:.2f}",
                     f"{1e3 * ring2.loc[w, 'ringing_index']:.2f}",
                     f"{ring.loc[w, 'ssim']:.3f}"])
    tab = ax1.table(cellText=rows,
                    colLabels=["window", "mainlobe\n(null bins)", "PSL\n(dB)",
                               f"ring.x1e3\nf={frac}",
                               f"ring.x1e3\nf={max(e4c['lowpass_fracs'])}",
                               f"SSIM\nf={frac}"],
                    loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.9)
    ax1.set_title("Window response vs measured ringing (phantom)")

    ax2 = fig.add_subplot(gs[0, 2])
    _img_ax(ax2, np.abs(recon.recon_apodized(k, m, "rect")), f"Low-pass frac {frac}, rect")
    ax2.axvline(col, color="#eda100", lw=0.8)

    rows_y = slice(20, 110)
    truth_prof = img[rows_y, col]
    for j, ws in enumerate([["rect", "tukey"], ["hann", "hamming", "blackman"]]):
        ax = fig.add_subplot(gs[1, j])
        ax.plot(np.arange(rows_y.start, rows_y.stop), truth_prof, color=MUTED, lw=1.0,
                ls="--", label="truth")
        for w in ws:
            kw = {"alpha": e4c["tukey_alpha"]} if w == "tukey" else {}
            rec = np.abs(recon.recon_apodized(k, m, w, **kw))
            ax.plot(np.arange(rows_y.start, rows_y.stop), rec[rows_y, col],
                    color=WIN_COLORS[w], lw=1.4, label=w)
        ax.set_xlabel("PE row (px)")
        ax.set_ylabel("intensity")
        ax.set_title("Edge profile along PE (vertical line in the image)")
        ax.grid(True)
        ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    for w in WINDOWS:
        g = e4[(e4.image == "phantom") & (e4.window == w)].sort_values("frac")
        ax.plot(g.frac, 1e3 * g.ringing_index, color=WIN_COLORS[w], marker="o", label=w)
    ax.set_yscale("log")
    ax.set_xlabel("low-pass fraction of k_y kept")
    ax.set_ylabel("ringing index x1e3 (log)")
    ax.set_title("Ringing vs window and truncation")
    ax.grid(True)
    ax.legend(fontsize=8)
    r_rect = ring.loc["rect", "ringing_index"]
    r_bl = ring.loc["blackman", "ringing_index"]
    save(fig, "F9", "apodization",
         f"Windowing and Gibbs (CO4). Truncating k_y is a rectangular window whose -13 dB "
         f"Dirichlet sidelobes ring at every edge. Tapered windows trade mainlobe width "
         f"(blur) for sidelobe level: at frac {frac}, ringing drops from "
         f"{1e3 * r_rect:.1f}e-3 (rect) to {1e3 * r_bl:.2f}e-3 (Blackman, -58 dB) while the "
         f"mainlobe triples. Hann beats Hamming far from the edge despite a higher PSL "
         f"because its sidelobes roll off at 18 dB/octave versus Hamming's 6.", out, dpi)


# ---------------------------------------------------------------- F10

def fig_f10(cfg, images, e5, out, dpi, image="brain"):
    frac = min(cfg["e5"]["fracs"])
    img = images[image]
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.6),
                            gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    fig.subplots_adjust(left=0.04, right=0.94, top=0.9, bottom=0.14, wspace=0.12,
                        hspace=0.32)
    for i, phase in enumerate(["none", "smooth"]):
        x = data.apply_synthetic_phase(img, phase)
        k = data.make_kspace(x)
        m = masks.mask_partial_fourier(img.shape, frac=frac)
        ims = {
            "truth": np.abs(x),
            "zerofill": np.abs(recon.recon_zerofill(k, m)),
            "conjugate_symmetry": np.abs(recon.recon_conjugate_symmetry(k, m)),
            "pocs": np.abs(recon.recon_pocs(k, m, cfg["e5"]["pocs_iters"])),
        }
        for j, (name, im) in enumerate(ims.items()):
            t = name if name == "truth" else f"{name}\nNRMSE={nrmse(np.abs(x), im):.2e}"
            _img_ax(axs[i, j], im, t, vmax=1.0)
            axs[i, j].title.set_fontsize(8.5)
        axs[i, 0].set_ylabel(f"phase = {phase}", fontsize=10)
        if phase == "smooth":
            axs[i, 4].imshow(np.angle(x) * (np.abs(x) > 0.02), cmap="twilight",
                             vmin=-np.pi, vmax=np.pi)
            axs[i, 4].set_title("synthetic phase (rad)")
        else:
            d = e5[(e5.image == image)]
            ax = axs[i, 4]
            for rn, col in zip(cfg["e5"]["recons"], ["#2a78d6", "#eb6834", "#1baf7a"]):
                for ph, ls in [("none", "-"), ("smooth", "--")]:
                    s = d[(d.recon == rn) & (d.phase_mode == ph)].sort_values("frac")
                    ax.plot(s.frac, np.maximum(s.nrmse, 1e-17), color=col, ls=ls, marker="o",
                            ms=4, label=f"{rn}, {ph}")
            ax.set_yscale("log")
            ax.set_xlabel("partial-Fourier fraction")
            ax.set_ylabel("NRMSE")
            ax.legend(fontsize=6, loc="center right")
            ax.grid(True)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            continue
        axs[i, 4].set_xticks([])
        axs[i, 4].set_yticks([])
    g = e5[(e5.image == image) & (e5.frac == frac)].set_index(["phase_mode", "recon"])["nrmse"]
    save(fig, "F10", "partial_fourier",
         f"Conjugate (Hermitian) symmetry, CO3: {image}, frac = {frac}. For a real image "
         f"K[-k] = K*[k], so filling the missing half with conjugates is exact (NRMSE "
         f"{g[('none', 'conjugate_symmetry')]:.1e}). With smooth phase the symmetry is "
         f"broken and the same recon fails (NRMSE {g[('smooth', 'conjugate_symmetry')]:.3f}, "
         f"no better than zero-fill {g[('smooth', 'zerofill')]:.3f}); POCS imposes the "
         f"low-resolution phase and recovers ({g[('smooth', 'pocs')]:.3f}).", out, dpi)


# ---------------------------------------------------------------- F11

def fig_f11(e6, out, dpi):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.24)
    ax.loglog(e6.N, e6.naive_dft_s, color="#2a78d6", marker="o", label="naive DFT (N x N matrix)")
    ax.loglog(e6.N, e6.radix2_fft_s, color="#eb6834", marker="s", label="radix-2 FFT (own code)")
    ax.loglog(e6.N, e6.numpy_fft_s, color="#1baf7a", marker="^", label="numpy.fft (pocketfft)")
    i = e6.index[e6.N == 1024][0] if (e6.N == 1024).any() else e6.index[-1]
    a = e6.naive_dft_s[i] / e6.N2[i]
    b = e6.radix2_fft_s[i] / e6.NlogN[i]
    ax.loglog(e6.N, a * e6.N2, color=MUTED, ls="--", lw=1, label="~ N^2")
    ax.loglog(e6.N, b * e6.NlogN, color=MUTED, ls=":", lw=1.4, label="~ N log2 N")
    ax.set_xlabel("transform length N")
    ax.set_ylabel("time per transform (s)")
    ax.grid(True, which="major")
    ax.legend(fontsize=8)
    ax.set_title("DFT vs FFT computation time")
    s_np = e6.speedup_numpy_vs_naive[i]
    s_r2 = e6.speedup_radix2_vs_naive[i]
    cross = e6[e6.radix2_fft_s < e6.naive_dft_s].N.min()
    save(fig, "F11", "fft_benchmark",
         f"FFT complexity, N^2 -> N log2 N (CO3). At N = {e6.N[i]} numpy's FFT is "
         f"{s_np:,.0f}x faster than the direct DFT and our own radix-2 FFT {s_r2:,.0f}x. "
         f"The radix-2 code overtakes the direct DFT from N = {cross} (crossover); below "
         f"it, Python overhead per stage dominates. A 256 x 256 recon needs 512 length-256 "
         f"transforms.", out, dpi)


# ---------------------------------------------------------------- F12

def fig_f12(cfg, img, out, dpi):
    k = fft2c(img)
    h, w = img.shape
    side = np.sqrt(cfg["f12"]["centre_area_frac"])
    hy, hx = int(round(side * h / 2)), int(round(side * w / 2))
    box = np.zeros(img.shape, dtype=bool)
    box[h // 2 - hy:h // 2 + hy, w // 2 - hx:w // 2 + hx] = True
    centre = ifft2c(np.where(box, k, 0))
    edges = ifft2c(np.where(box, 0, k))
    e_c = (np.abs(k[box]) ** 2).sum() / (np.abs(k) ** 2).sum()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.16, wspace=0.05)
    _img_ax(axs[0], img, "Ground truth")
    _img_ax(axs[1], np.abs(centre), f"Centre only ({100 * box.mean():.0f}% of k-space,"
            f" {100 * e_c:.1f}% of energy)", vmax=1)
    _img_ax(axs[2], np.abs(edges), f"Periphery only ({100 * (1 - box.mean()):.0f}% of "
            f"k-space, {100 * (1 - e_c):.1f}% of energy)")
    save(fig, "F12", "centre_vs_edges",
         "What low and high k-space encode (energy compaction, CO3). The central 10% of "
         "k-space (a centred box) gives a blurred image with correct contrast; the outer "
         "90% gives only edges on a flat, near-zero background: the DC and low "
         "frequencies carry contrast, the periphery carries edges and fine detail.",
         out, dpi)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = ROOT / cfg["output"]["figures"]
    tables = ROOT / cfg["output"]["tables"]
    out.mkdir(parents=True, exist_ok=True)
    dpi = cfg["output"]["dpi"]

    res = pd.read_csv(tables / "results.csv")
    e1 = res[res.experiment == "E1"]
    e4 = res[res.experiment == "E4"]
    e5 = res[res.experiment == "E5"]
    e3 = pd.read_csv(tables / "e3_coherence_correlations.csv")
    e6 = pd.read_csv(tables / "e6_fft_benchmark.csv")
    e7 = pd.read_csv(tables / "e7_ghost_spacing.csv")
    wresp = pd.read_csv(tables / "e4_window_response.csv")

    images = load_images(cfg)
    img = images[cfg["default_image"]]
    fig_f1(img, out, dpi)
    fig_f2_to_f5(cfg, img, out, dpi)
    fig_f6(e1, out, dpi)
    fig_f7(e1, e3, out, dpi, image=cfg["default_image"])
    fig_f8(img, e7, out, dpi)
    fig_f9(cfg, images[cfg["controlled_image"]], e4, wresp, out, dpi)
    fig_f10(cfg, images, e5, out, dpi, image=cfg["default_image"])
    fig_f11(e6, out, dpi)
    fig_f12(cfg, img, out, dpi)

    with open(out / "captions.md", "w") as f:
        f.write("# Figure captions\n\n")
        for fid in sorted(CAPTIONS, key=lambda s: int(s[1:])):
            f.write(f"**{fid}.** {CAPTIONS[fid]}\n\n")
    print(f"wrote {len(CAPTIONS)} figures to {out}")


if __name__ == "__main__":
    main()
