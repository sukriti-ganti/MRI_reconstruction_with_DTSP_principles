# Figure captions

**F1.** 2D DFT and energy compaction (CO3). Brain slice and its centred log-magnitude k-space. The circle of radius 0.1*k_max (0.8% of the k-space area) holds 84.6% of the signal energy (Parseval).

**F2.** Mask gallery at R = 4 (partial Fourier at frac 0.55, R = 1.82, its practical limit). White = acquired phase-encode line; every mask is a 1D pattern over k_y broadcast along the readout k_x, so all masks are horizontal stripes. Titles give the realised R = lines / sampled lines.

**F3.** PSF gallery = IDFT of each mask (convolution theorem, CO1). Top: log10 |PSF| (a 64-px strip; the PSF is a delta along readout). Bottom: PE profile in dB. Uniform: comb of R equal peaks at multiples of N/R (CI = 84, PSR = 1). Random and variable density: thin, noise-like sidelobe floor (CI < 3). Low-pass: Dirichlet kernel, -13 dB first sidelobe. Partial Fourier: a broadened, complex mainlobe.

**F4.** Zero-filled reconstructions at R = 4 (the image convolved with each F3 PSF). Uniform: coherent aliasing from a comb PSF, R-1 = 3 sharp replicas displaced by N/R = 64 px and wrapped circularly. Random: incoherent, noise-like aliasing from a flat sidelobe floor. Low-pass: truncation, so blur plus Gibbs ringing but no replicas.

**F5.** Error maps | |recon| - |truth| | at R = 4, common scale [0, 0.5]. The error is (PSF - delta) convolved with the image: discrete shifted copies of the anatomy for the comb PSF (uniform), diffuse texture for random masks, and error confined to edges for low-pass and partial Fourier.

**F6.** Quality-acceleration trade-off, zero-filled recon, plotted against realised R. At R = 4 on the brain, NRMSE ranges from 0.12 (low-pass) to 0.71 (uniform): the same amount of discarded data gives a 6x spread in error, so how lines are chosen matters more than how many.

**F7.** Core thesis, brain, all masks and R. (a) Every mask placed on the coherence-index / NRMSE plane: coherent masks (uniform, CI >= 18) sit top-right and CS-FISTA cannot move them (arrows of zero length), incoherent masks (CI < 3) move down. (b) Zero-fill NRMSE is set by the PSF sidelobe energy (Parseval), a PSF-shape property that ranks the error far better than R. (c) The coherence index decides whether the leaked energy is removable: comb PSFs give ~0 gain (aliasing masks only; low-pass truncation has no aliasing for CS to remove and CS increases its error by 5-153% instead, see results.csv).

**F8.** Sampling theorem in frequency (CO3). Decimating k_y by R (Delta_k = R/FOV) replicates the image with period FOV/R. Top: zero-filled uniform recon with the ghost positions measured by phase correlation (dashed). Bottom: PSF profile with predicted replica positions N/R (dotted). Max |measured - N/R| = 0.24 px over R = 2..8; for R = 3, 6 (R does not divide 256) the comb is not exactly periodic and the realised R is 3.01 / 5.95.

**F9.** Windowing and Gibbs (CO4). Truncating k_y is a rectangular window whose -13 dB Dirichlet sidelobes ring at every edge. Tapered windows trade mainlobe width (blur) for sidelobe level: at frac 0.25, ringing drops from 15.1e-3 (rect) to 0.07e-3 (Blackman, -58 dB) while the mainlobe triples. Hann beats Hamming far from the edge despite a higher PSL because its sidelobes roll off at 18 dB/octave versus Hamming's 6.

**F10.** Conjugate (Hermitian) symmetry, CO3: brain, frac = 0.55. For a real image K[-k] = K*[k], so filling the missing half with conjugates is exact (NRMSE 3.1e-16). With smooth phase the symmetry is broken and the same recon fails (NRMSE 0.142, no better than zero-fill 0.141); POCS imposes the low-resolution phase and recovers (0.022).

**F11.** FFT complexity, N^2 -> N log2 N (CO3). At N = 1024 numpy's FFT is 2,959x faster than the direct DFT and our own radix-2 FFT 202x. The radix-2 code overtakes the direct DFT from N = 64 (crossover); below it, Python overhead per stage dominates. A 256 x 256 recon needs 512 length-256 transforms.

**F12.** What low and high k-space encode (energy compaction, CO3). The central 10% of k-space (a centred box) gives a blurred image with correct contrast; the outer 90% gives only edges on a flat, near-zero background: the DC and low frequencies carry contrast, the periphery carries edges and fine detail.

