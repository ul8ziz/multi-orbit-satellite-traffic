# -*- coding: utf-8 -*-
"""Run alpha calibration sweep on outputs/aggregated_lambda.csv."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _ensure_src_on_path() -> None:
    r = str(PROJECT_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def main() -> int:
    _ensure_src_on_path()
    from src.integration.alpha_calibration_from_aggregated import calibrate_alphas

    in_csv = PROJECT_ROOT / "outputs" / "aggregated_lambda.csv"
    out_csv = PROJECT_ROOT / "outputs" / "calibration_results.csv"
    out_path, _df, best = calibrate_alphas(in_csv, out_csv, ALPHAS)

    print(out_path.resolve())
    print(
        f"best_alpha={best['best_alpha']:.6g} "
        f"P_block={best['P_block']:.6f} "
        f"E[n]={best['E[n]']:.6f} "
        f"util_LEO={best['util_LEO']:.6f}"
    )
    print(best["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
