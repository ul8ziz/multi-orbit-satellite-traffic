"""Hours without records must be treated as missing data, never as zero traffic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _long_with_gap() -> pd.DataFrame:
    """6 days of hourly data per slice with a 30-hour hole on day 3."""
    ts = pd.date_range("2024-01-01", periods=6 * 24, freq="h")
    hole = (ts >= "2024-01-03 06:00") & (ts < "2024-01-04 12:00")
    rows = []
    for sid, scale in (("eMBB", 3e9), ("URLLC", 1e9), ("mMTC", 0.9e9)):
        y = scale * (1.5 + np.sin(2 * np.pi * ts.hour / 24))
        rows.append(pd.DataFrame({"ts": ts[~hole], "slice_id": sid, "y": y[~hole]}))
    return pd.concat(rows, ignore_index=True)


def test_missing_hours_are_nan_and_split_on_observed() -> None:
    from src.forecasting.timeseries import complete_hourly_series, temporal_train_test_split

    s = complete_hourly_series(_long_with_gap(), "eMBB")
    assert len(s) == 144 and int(s.isna().sum()) == 30
    assert (s.dropna() > 0).all()

    train, test = temporal_train_test_split(s, test_ratio=0.2)
    assert test.notna().all()
    assert len(test) == int(np.ceil(114 * 0.2))  # 20 % of the observed hours
    assert len(train) + len(test) == len(s)
    assert train.index.freq is not None  # statsmodels needs the hourly frequency

    s0 = complete_hourly_series(_long_with_gap(), "eMBB", fill_value=0.0)
    assert int((s0 == 0).sum()) == 30  # legacy behaviour still available


def test_spike_mask_ignores_nan() -> None:
    from src.metrics.forecast_metrics import spike_mask

    m = spike_mask(np.array([1.0, 2.0, np.nan, 3.0]), np.array([0.5, 10.0]), quantile=0.5)
    assert m.tolist() == [False, True]


@pytest.mark.parametrize("horizon", ["one_step", "multi_step"])
def test_models_forecast_across_gap(horizon: str) -> None:
    from src.forecasting.neural_forecast import forecast_train_test as nn_fc
    from src.forecasting.sarima_forecast import forecast_train_test as sar_fc
    from src.forecasting.timeseries import complete_hourly_series, temporal_train_test_split

    y = complete_hourly_series(_long_with_gap(), "eMBB")
    train, test = temporal_train_test_split(y, test_ratio=0.2)

    df_s, pred_s = sar_fc(y, train, test, horizon=horizon)
    df_n, pred_n, _ = nn_fc(y, train, test, seed=0, horizon=horizon)
    for pred in (pred_s, pred_n):
        assert len(pred) == len(test) and np.isfinite(pred).all()
        # A smooth daily cycle must be predicted far better than "always zero".
        assert np.sqrt(np.mean((pred - test.values) ** 2)) < 0.5 * np.sqrt(np.mean(test.values**2))
    for df in (df_s, df_n):
        tr = df[df["split"] == "train"]
        assert tr["y_actual"].isna().sum() == 30  # gap kept as missing, not 0
        assert (df["y_actual"].dropna() > 0).all()


def test_pipeline_metrics_on_observed_hours(tmp_path: Path) -> None:
    from src.forecasting.pipeline import run_forecasting

    long_df = _long_with_gap()
    hdf = pd.DataFrame(
        {
            "date": long_df["ts"].dt.normalize(),
            "hour": long_df["ts"].dt.hour.astype("int8"),
            "slice_id": long_df["slice_id"],
            "total_mb": long_df["y"].astype("int64"),
        }
    )
    p = tmp_path / "hourly.parquet"
    hdf.to_parquet(p)
    out = run_forecasting(tmp_path, hourly_slice_totals_path=p, test_ratio=0.2, seed=0)
    cmp_df = pd.read_csv(out["forecast_model_comparison"])
    assert (cmp_df["n_test"] == int(np.ceil(114 * 0.2))).all()
    assert cmp_df["rmse"].notna().all()
    pred = pd.read_csv(out["forecast_predictions_sarima"])
    assert not (pred["y_actual"] == 0).any()
