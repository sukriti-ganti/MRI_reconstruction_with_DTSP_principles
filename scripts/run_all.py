"""Run every experiment (E1-E7), write all tables, then regenerate all figures.

Usage: python scripts/run_all.py [--config config/experiments.yaml]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import experiments as ex  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = ex.load_config(args.config)
    tables = ROOT / cfg["output"]["tables"]
    tables.mkdir(parents=True, exist_ok=True)
    images = ex.load_images(cfg)

    def step(label, fn):
        t0 = time.time()
        out = fn()
        print(f"{label:<40s} {time.time() - t0:6.1f} s")
        return out

    e1 = step("E1 acceleration sweep", lambda: ex.run_e1(cfg, images))
    e2 = step("E2 PSF characterisation", lambda: ex.run_e2(cfg))
    e3 = step("E3 coherence vs error", lambda: ex.run_e3(e1))
    e4, wresp = step("E4 apodization", lambda: ex.run_e4(cfg, images))
    e5 = step("E5 partial Fourier", lambda: ex.run_e5(cfg, images))
    e6 = step("E6 FFT benchmark", lambda: ex.run_e6(cfg))
    e7 = step("E7 ghost spacing", lambda: ex.run_e7(cfg, images))

    results = pd.concat([e1, e4, e5], ignore_index=True)
    results.to_csv(tables / "results.csv", index=False)
    e2.to_csv(tables / "e2_psf_metrics.csv", index=False)
    e3.to_csv(tables / "e3_coherence_correlations.csv", index=False)
    wresp.to_csv(tables / "e4_window_response.csv", index=False)
    e6.to_csv(tables / "e6_fft_benchmark.csv", index=False)
    e7.to_csv(tables / "e7_ghost_spacing.csv", index=False)
    print(f"results.csv: {len(results)} rows")

    from make_figures import main as make_figs  # noqa: E402

    step("figures F1-F12", lambda: make_figs(["--config", args.config] if args.config else []))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
