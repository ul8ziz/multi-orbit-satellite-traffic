"""CLI: regenerate Figures 1–12 (and Table 3) of the journal article from project data.

Outputs go to ``outputs/article_figures/`` (default):

* ``fig01_daily_profile.png``
* ``fig02_hourly_pdf.png``, ``fig03_hourly_cdf.png``          — as printed in the article
* ``fig02b_record_pdf.png``, ``fig03b_record_cdf.png``,
  ``table3_distribution_fits.csv``                           — per-record fits (caption / Table 3)
* ``fig04_pipeline.png``, ``fig05_service_rate.png``, ``fig06_state_diagram.png``
* ``fig07_forecast_vs_actual_eMBB.png``
* ``fig08_error_hist_eMBB.png``, ``fig09_error_hist_URLLC.png``, ``fig10_error_hist_mMTC.png``
* ``fig11_sensitivity.png`` + ``fig11_sensitivity.csv``
* ``fig12_quasi_stationary.png`` + ``fig12_quasi_stationary.csv``

Usage::

    python run_article_figures.py                    # all figures
    python run_article_figures.py --only 5 6 11 12   # a subset
    python run_article_figures.py --skip-records     # skip the parquet sample (Fig. 2b/3b, Table 3)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALL_FIGS = tuple(range(1, 13))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("run_article_figures")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="+", default=list(ALL_FIGS), help="Figure numbers to build.")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "article_figures")
    ap.add_argument("--tables", type=Path, default=PROJECT_ROOT / "outputs" / "tables")
    ap.add_argument("--parquet", type=Path, default=PROJECT_ROOT / "data" / "processed" / "cleaned_traffic.parquet")
    ap.add_argument("--skip-records", action="store_true", help="Skip per-record sample (Fig. 2b/3b, Table 3).")
    ap.add_argument("--sample-fraction", type=float, default=0.01, help="Per-record sample share (default 1%%).")
    ap.add_argument("--lang", choices=("en", "ru"), default="en", help="Fig. 4 label language.")
    ap.add_argument("--drop-missing-hours", action="store_true",
                    help="Fig. 8–10: exclude zero-filled hours (actual == 0) from the error histograms.")
    ap.add_argument("--target-mean-lambda", type=float, default=3.0,
                    help="Fig. 12: α is set so that the daily mean λ(h) equals this value.")
    ap.add_argument("--alpha", type=float, default=None, help="Fig. 12: explicit α (overrides the target).")
    ap.add_argument("--thesis-extras", action="store_true",
                    help="Also build thesis-only figures: Russian pipeline, LSTM/MLP architecture (en/ru), RMSE/MAE bars.")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    from src.markov.leontyeva import default_params
    from src.visualization import article_figures as af

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only)
    written: list[Path] = []

    if want & {1, 12}:
        daily = pd.read_csv(args.tables / "daily_profile.csv")
    if 1 in want:
        written.append(af.fig01_daily_profile(daily, out / "fig01_daily_profile.png"))

    if want & {2, 3}:
        hourly = pd.read_csv(args.tables / "artifact1_slice_aggregated.csv")
        if 2 in want:
            written.append(af.fig02_hourly_pdf(hourly, out / "fig02_hourly_pdf.png"))
        if 3 in want:
            written.append(af.fig03_hourly_cdf(hourly, out / "fig03_hourly_cdf.png"))
        if not args.skip_records:
            if args.parquet.is_file():
                rec = af.sample_records(args.parquet, fraction=args.sample_fraction)
                fits = af.fit_distributions(rec)
                p_t3 = out / "table3_distribution_fits.csv"
                fits.to_csv(p_t3, index=False)
                written.append(p_t3)
                if 2 in want:
                    written.append(af.fig02b_record_pdf(rec, fits, out / "fig02b_record_pdf.png"))
                if 3 in want:
                    written.append(af.fig03b_record_cdf(rec, fits, out / "fig03b_record_cdf.png"))
            else:
                log.warning("Parquet not found (%s): skipping Fig. 2b/3b and Table 3.", args.parquet)

    if 4 in want:
        written.append(af.fig04_pipeline(out / "fig04_pipeline.png", lang=args.lang))
    if 5 in want:
        written.append(af.fig05_service_rate(default_params(), out / "fig05_service_rate.png"))
    if 6 in want:
        written.append(af.fig06_state_diagram(out / "fig06_state_diagram.png", N=3, M=2))

    if want & {7, 8, 9, 10}:
        sar = pd.read_csv(args.tables / "forecast_predictions_sarima.csv")
        lstm = pd.read_csv(args.tables / "forecast_predictions_lstm.csv")
        preds = {s: af.test_predictions(sar, lstm, s) for s in ("eMBB", "URLLC", "mMTC")}
        n_zero = {s: int((d["actual"] == 0).sum()) for s, d in preds.items()}
        if any(n_zero.values()):
            log.warning(
                "Test horizon contains zero-filled hours with no data %s — these are outputs of the "
                "old pipeline; re-run run_forecasting.py (or use --drop-missing-hours).", n_zero,
            )
        if 7 in want:
            written.append(af.fig07_forecast_vs_actual(preds["eMBB"], "eMBB", out / "fig07_forecast_vs_actual_eMBB.png"))
        for num, s in ((8, "eMBB"), (9, "URLLC"), (10, "mMTC")):
            if num in want:
                written.append(af.fig08_10_error_hist(
                    preds[s], s, out / f"fig{num:02d}_error_hist_{s}.png", drop_missing=args.drop_missing_hours))

    if 11 in want:
        tab = af.sensitivity_table(np.arange(0.5, 8.0001, 0.25))
        p11 = out / "fig11_sensitivity.csv"
        tab.to_csv(p11, index=False)
        written += [p11, af.fig11_sensitivity(tab, out / "fig11_sensitivity.png")]

    if 12 in want:
        tab, alpha = af.quasi_stationary_table(daily, target_mean_lambda=args.target_mean_lambda, alpha=args.alpha)
        log.info("Fig. 12: alpha = %.6g (λ range %.3f … %.3f)", alpha, tab["lambda"].min(), tab["lambda"].max())
        p12 = out / "fig12_quasi_stationary.csv"
        tab.assign(alpha=alpha).to_csv(p12, index=False)
        written += [p12, af.fig12_quasi_stationary(tab, out / "fig12_quasi_stationary.png")]

    if args.thesis_extras:
        written.append(af.fig04_pipeline(out / "thesis_pipeline_ru.png", lang="ru"))
        written.append(af.fig_lstm_architecture(out / "thesis_lstm_architecture_ru.png", lang="ru"))
        written.append(af.fig_lstm_architecture(out / "thesis_lstm_architecture_en.png", lang="en"))
        cmp_df = pd.read_csv(args.tables / "forecast_model_comparison.csv")
        written.append(af.fig_rmse_mae_bars(cmp_df, out / "thesis_rmse_mae_bars.png"))

    for p in written:
        log.info("Wrote %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
