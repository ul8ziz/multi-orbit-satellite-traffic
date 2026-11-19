"""Build hourly per-slice time series from cleaned traffic for forecasting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def hourly_slice_totals_long(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``total_mb`` per calendar time and slice.

    Parameters
    ----------
    df:
        Cleaned traffic with ``date``, ``hour``, ``slice_id``, ``total_mb``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``ts`` (timezone-naive datetime), ``slice_id``, ``y`` (total_mb).
    """
    need = {"date", "hour", "slice_id", "total_mb"}
    missing = need - set(df.columns)
    if missing:
        msg = f"hourly_slice_totals_long: missing columns {missing}"
        raise ValueError(msg)

    g = (
        df.groupby(["date", "hour", "slice_id"], as_index=False)["total_mb"]
        .sum()
        .rename(columns={"total_mb": "y"})
    )
    g["ts"] = pd.to_datetime(g["date"]) + pd.to_timedelta(g["hour"].astype(int), unit="h")
    g = g.sort_values("ts").reset_index(drop=True)
    return g[["ts", "slice_id", "y"]]


def complete_hourly_series(
    long_df: pd.DataFrame,
    slice_id: Any,
    fill_value: float = np.nan,
) -> pd.Series:
    """Regular hourly grid from first to last timestamp.

    Hours without any record are **missing data**, not zero traffic: the dataset has no
    records from 2018-02-22 14:00 to 2018-03-01 15:00. They are kept as ``NaN`` by default so
    that models can skip them (SARIMAX's Kalman filter handles ``NaN`` natively) and metrics
    are computed on observed hours only. ``fill_value=0.0`` restores the old behaviour.
    """
    sub = long_df[long_df["slice_id"] == slice_id].sort_values("ts")
    if sub.empty:
        return pd.Series(dtype=float)

    t0, t1 = sub["ts"].min(), sub["ts"].max()
    full_idx = pd.date_range(t0, t1, freq="h")
    s = sub.set_index("ts")["y"].astype(float).reindex(full_idx, fill_value=fill_value)
    s.name = "y"
    n_missing = int(s.isna().sum()) if np.isnan(fill_value) else int((s == fill_value).sum())
    if n_missing:
        logger.info("Slice %s: %d of %d hours have no records (kept as %s).", slice_id, n_missing, len(s), fill_value)
    return s


def build_per_slice_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return a completed hourly series per ``slice_id``."""
    long_df = hourly_slice_totals_long(df)
    out: dict[str, pd.Series] = {}
    for sid in sorted(long_df["slice_id"].unique()):
        s = complete_hourly_series(long_df, sid)
        if len(s) > 0:
            out[str(sid)] = s
    logger.info("Built hourly series for slices: %s (lengths %s)", list(out.keys()), {k: len(v) for k, v in out.items()})
    return out


def temporal_train_test_split(
    y: pd.Series,
    test_ratio: float = 0.2,
    min_test: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Chronological split: the last ``test_ratio`` fraction of **observed** hours is test.

    Missing hours (``NaN``) are not counted, so the test segment holds
    ``ceil(n_observed · test_ratio)`` real observations and starts at the first of them.
    For a series without gaps this is the same as splitting by position.
    """
    obs = y.dropna()
    n = len(obs)
    if n < 3:
        msg = f"Series too short for train/test (n_observed={n})."
        raise ValueError(msg)
    n_test = max(min_test, int(np.ceil(n * test_ratio)))
    n_test = min(n_test, n - 1)
    k = int(y.index.get_loc(obs.index[-n_test]))
    train, test = y.iloc[:k], y.iloc[k:]  # iloc keeps the hourly freq for statsmodels
    return train, test
