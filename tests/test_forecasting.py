"""Forecasting pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _synthetic_cleaned(project_root: Path) -> None:
    """Build multi-day hourly-ish cleaned traffic with slice_id."""
    rows: list[dict[str, object]] = []
    t0 = pd.Timestamp("2024-01-01")
    for d in range(5):
        for h in range(24):
            ts = t0 + pd.Timedelta(days=d, hours=h)
            for sid, scale in [("eMBB", 10.0), ("URLLC", 3.0), ("mMTC", 1.0)]:
                y = scale * (1 + 0.1 * h / 24.0) + d
                rows.append(
                    {
                        "MASKED_MSISDN": "U1",
                        "APP_CLASS": "Web",
                        "UPLOAD": y * 0.3,
                        "DOWNLOAD": y * 0.7,
                        "start_ts": ts,
                        "hour": h,
                        "date": ts.normalize(),
                        "total_mb": y,
                        "slice_id": sid,
                    }
                )
    df = pd.DataFrame(rows)
    proc = project_root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc / "cleaned_traffic.csv", index=False)


def test_run_forecasting_writes_csvs(tmp_path: Path) -> None:
    _synthetic_cleaned(tmp_path)
    from src.forecasting.pipeline import run_forecasting

    out = run_forecasting(tmp_path, test_ratio=0.2, seed=0)
    assert out["forecast_predictions_sarima"].is_file()
    assert out["forecast_predictions_lstm"].is_file()
    assert out["forecast_arima_metrics"].is_file()
    assert out["forecast_lstm_metrics"].is_file()
    assert out["forecast_model_comparison"].is_file()

    cmp_df = pd.read_csv(out["forecast_model_comparison"])
    assert "rmse" in cmp_df.columns and "model" in cmp_df.columns


def test_forecast_metrics_helpers() -> None:
    import numpy as np

    from src.metrics.forecast_metrics import mae, rmse, spike_mae

    y_t = np.array([1.0, 2.0, 10.0, 3.0])
    y_p = np.array([1.1, 2.1, 9.0, 2.9])
    train = np.array([0.5, 0.5, 0.5])
    assert rmse(y_t, y_p) >= 0
    assert mae(y_t, y_p) >= 0
    sm, n = spike_mae(y_t, y_p, train, quantile=0.75)
    assert n >= 0
