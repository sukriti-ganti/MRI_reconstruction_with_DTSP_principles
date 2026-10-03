# k-Space Undersampling Strategies and MRI Reconstruction Quality

A DSP project for 24EC3112 *Discrete Time Signal Processing*: a reproducible framework that measures how k-space undersampling degrades MRI images and explains each artefact with DTSP theory.

## Thesis

Undersampling k-space multiplies it by a sampling mask. By the convolution theorem, the image is then **circularly convolved with the mask's point spread function** (PSF = IDFT of the mask). The PSF's shape, not the amount of data discarded, determines the artefact:

- Decimating phase-encode lines gives a comb PSF, which produces **coherent ghosts** spaced exactly N/R apart.
- Truncating them gives a Dirichlet PSF, which produces **blur and Gibbs ringing**.
- Random lines give a flat sidelobe floor, which produces **incoherent, noise-like aliasing**.

The measured results:

- At equal R, the PSF's sidelobe energy ranks the error far better than R does (Spearman ρ = 0.93 vs 0.58).
- PSF coherence decides whether compressed sensing can remove the error: 0% removed for a comb, 45% for variable density.

The full write-up is in [`report/report.md`](report/report.md).

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_all.py      # E1-E7 -> results/tables/*.csv, F1-F12 -> results/figures/*.png (~1.5 min)
pytest tests/                  # 98 tests, including the theory checks in tests/test_theory.py
python scripts/benchmark_fft.py   # E6 only
python scripts/make_figures.py    # figures only, from existing tables
```

No download is needed. The brain slices ship in `data/brain_slices.npz` (a copy of `skimage.data.brain()`), and the Shepp–Logan phantom ships with scikit-image. Every parameter is set in `config/experiments.yaml`.

## Modules and the DTSP concept each implements

| Module | What it does | DTSP concept (CO) |
|---|---|---|
| `src/transforms.py` | centred unitary `fft2c`/`ifft2c`, Hermitian flip, zero-pad interpolation, direct DFT and radix-2 FFT | 2D DFT and separability, Parseval, zero-padding ↔ sinc interpolation, FFT complexity (CO3) |
| `src/data.py` | test images, synthetic smooth phase, k-space | conjugate symmetry of real signals (CO3) |
| `src/masks.py` | 7 phase-encode line masks + realised R | impulse train / comb (CO1), sampling theorem Δk ≤ 1/FOV (CO3) |
| `src/psf.py` | PSF = IDFT(mask), FWHM, PSR, sidelobe energy, coherence index, direct circular convolution | convolution theorem, circular convolution (CO1) |
| `src/apodization.py` | rect/Hann/Hamming/Tukey/Blackman windows, DTFT mainlobe and sidelobe measurement | window design, mainlobe vs sidelobe trade-off (CO4) |
| `src/recon.py` | zero-fill, apodized, conjugate symmetry, POCS, CS-FISTA (Haar L1) | DTFT of a truncated sequence (CO3), filtering (CO4), Hermitian symmetry (CO3) |
| `src/metrics.py` | NRMSE, PSNR, SSIM, artefact power, edge preservation, ringing index | Parseval (error energy) |
| `src/experiments.py` | sweeps E1–E7 | — |

## Outputs

- `results/tables/results.csv`: one row per (experiment, image, phase mode, mask, R, recon, window), with every metric, the PSF metrics and the mask seed.
- `results/tables/e2_psf_metrics.csv`, `e3_coherence_correlations.csv`, `e4_window_response.csv`, `e6_fft_benchmark.csv`, `e7_ghost_spacing.csv`.
- `results/figures/F1…F12_*.png` at 300 dpi, each captioned with the principle it shows. `captions.md` collects the captions.

## Conventions

- Axis 0 is the phase-encode direction (undersampled); axis 1 is the readout (always fully sampled). Masks are horizontal stripes.
- DC sits at index N/2. All centring is explicit (`ifftshift` → DFT → `fftshift`), and every transform is unitary.
- Arrays stay `complex128` until metric time. Metrics use magnitudes divided by the reference maximum.
- All randomness is seeded, and the seeds are recorded in the results.
