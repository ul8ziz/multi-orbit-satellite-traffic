# -*- coding: utf-8 -*-
"""Extended alpha sweep for finding first useful operating region."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ALPHAS_EXTENDED = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]


def _ensure_src_on_path() -> None:
    r = str(PROJECT_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def main() -> int:
    _ensure_src_on_path()
    from src.integration.alpha_calibration_from_aggregated import (
        calibrate_alphas,
        pick_recommended_triplet,
    )

    in_csv = PROJECT_ROOT / "outputs" / "aggregated_lambda.csv"
    out_csv = PROJECT_ROOT / "outputs" / "calibration_results_extended.csv"
    out_path, df, _best = calibrate_alphas(in_csv, out_csv, ALPHAS_EXTENDED)
    picks = pick_recommended_triplet(df)

    fm = picks["first_moderate"]
    low = picks["recommended_low"]
    med = picks["recommended_medium"]
    high = picks["recommended_high"]

    print(out_path.resolve())
    if fm:
        print(
            "first_moderate_alpha={a} P_block={pb} E[n]={en} E[m]={em} util_LEO={ul} util_MEO={um}".format(
                a=_fmt(float(fm["alpha"])),
                pb=_fmt(float(fm["P_block"])),
                en=_fmt(float(fm["E[n]"])),
                em=_fmt(float(fm["E[m]"])),
                ul=_fmt(float(fm["util_LEO"])),
                um=_fmt(float(fm["util_MEO"])),
            )
        )
    else:
        print("first_moderate_alpha=none_in_grid")

    print(
        "recommended_low alpha={a} P_block={pb} E[n]={en} util_LEO={ul}".format(
            a=_fmt(float(low["alpha"])),
            pb=_fmt(float(low["P_block"])),
            en=_fmt(float(low["E[n]"])),
            ul=_fmt(float(low["util_LEO"])),
        )
    )
    print(
        "recommended_medium alpha={a} P_block={pb} E[n]={en} util_LEO={ul}".format(
            a=_fmt(float(med["alpha"])),
            pb=_fmt(float(med["P_block"])),
            en=_fmt(float(med["E[n]"])),
            ul=_fmt(float(med["util_LEO"])),
        )
    )
    if high:
        print(
            "recommended_high alpha={a} P_block={pb} E[n]={en} util_LEO={ul}".format(
                a=_fmt(float(high["alpha"])),
                pb=_fmt(float(high["P_block"])),
                en=_fmt(float(high["E[n]"])),
                ul=_fmt(float(high["util_LEO"])),
            )
        )
    else:
        print("recommended_high alpha=none_in_grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
