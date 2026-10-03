"""E6 - time a naive O(N^2) DFT against a radix-2 FFT and numpy.fft.

Usage: python scripts/benchmark_fft.py   (writes results/tables/e6_fft_benchmark.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments import load_config, run_e6  # noqa: E402


def main() -> None:
    cfg = load_config()
    df = run_e6(cfg)
    out = ROOT / cfg["output"]["tables"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "e6_fft_benchmark.csv", index=False)
    rep = df[df.N.isin(cfg["e6"]["report_sizes"])]
    print(rep[["N", "naive_dft_s", "radix2_fft_s", "numpy_fft_s",
               "speedup_numpy_vs_naive"]].to_string(index=False))


if __name__ == "__main__":
    main()
