# -*- coding: utf-8 -*-
"""Числовая калибровка параметров CTMC Leontyeva (без изменения модели)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_src_on_path() -> None:
    r = str(PROJECT_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    configure_logging()

    from src.integration.calibration import run_calibration

    out = run_calibration(PROJECT_ROOT)

    rec = out["recommended"]
    print("", flush=True)
    print("=== RECOMMENDED CALIBRATED PARAMETER SETS (mean_load) ===", flush=True)
    if rec is not None and not rec.empty:
        for _, row in rec.iterrows():
            print(
                f"  {row.get('recommendation', '?')}: alpha={row['alpha']:.6g} "
                f"N={row['N']} M={row['M']} C_LEO={row['C_LEO']} "
                f"mu_LEO={row['mu_LEO']} mu_MEO={row['mu_MEO']} gamma={row['gamma']}",
                flush=True,
            )
            print(
                f"    P(n=N)={row['P(n=N)']:.6f} P(m=M)={row['P(m=M)']:.6f} "
                f"util_LEO={row['util_LEO']:.6f} util_MEO={row['util_MEO']:.6f}",
                flush=True,
            )
    else:
        print("  (no non-trivial recommendations; see calibration_grid_results.csv)", flush=True)

    print("", flush=True)
    print("Calibration output paths:", flush=True)
    for key in (
        "calibration_grid_results",
        "calibrated_scenarios",
        "calibration_summary",
        "blocking_vs_alpha",
        "utilization_vs_alpha",
        "calibration_tradeoff",
    ):
        p = out.get(key)
        if p is not None and hasattr(p, "resolve"):
            print(f"  {key}: {p.resolve()}", flush=True)
        elif isinstance(p, Path):
            print(f"  {key}: {p.resolve()}", flush=True)

    print("", flush=True)
    print(
        "Calibration completed successfully without changing the Leontyeva model.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
