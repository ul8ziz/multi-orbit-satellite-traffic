"""Per-slice descriptive statistics, PAPR, daily profiles, peak/off-peak, correlations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _hourly_slice_totals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, hour, slice_id) with summed ``total_mb``."""
    return (
        df.groupby(["date", "hour", "slice_id"], as_index=False)["total_mb"]
        .sum()
        .rename(columns={"total_mb": "total_mb"})
    )


def daily_profile_long(df: pd.DataFrame) -> pd.DataFrame:
    """Mean total traffic per clock hour and slice (averaged across calendar days)."""
    h = _hourly_slice_totals(df)
    if h.empty:
        return pd.DataFrame(columns=["slice_id", "hour", "mean_total"])

    prof = h.groupby(["slice_id", "hour"], as_index=False)["total_mb"].mean()
    prof = prof.rename(columns={"total_mb": "mean_total"})
    return prof.sort_values(["slice_id", "hour"]).reset_index(drop=True)


def compute_slice_stats_and_papr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-slice mean/variance/std, PAPR, peak vs off-peak averages, burstiness (CV²).

    PAPR is ``max(hourly mean profile) / mean(hourly mean profile)`` per slice.

    Peak / off-peak: hours with profile ≥ overall mean vs below mean (per slice).

    Returns
    -------
    slice_stats : pandas.DataFrame
        Columns include ``slice_id``, ``mean``, ``variance``, ``std``, ``papr``,
        ``peak_hour_avg``, ``off_peak_hour_avg``.
    burstiness_papr : pandas.DataFrame
        Columns: ``slice_id``, ``papr``, ``burstiness_cv2``.
    """
    prof = daily_profile_long(df)
    if prof.empty:
        empty_stats = pd.DataFrame(
            columns=[
                "slice_id",
                "mean",
                "variance",
                "std",
                "papr",
                "peak_hour_avg",
                "off_peak_hour_avg",
            ]
        )
        empty_b = pd.DataFrame(columns=["slice_id", "papr", "burstiness_cv2"])
        return empty_stats, empty_b

    rows: list[dict[str, Any]] = []
    burst_rows: list[dict[str, Any]] = []

    for sid, g in prof.groupby("slice_id"):
        y = g.sort_values("hour")["mean_total"].to_numpy(dtype=float)
        mu = float(np.mean(y))
        var = float(np.var(y, ddof=0))
        std = float(np.std(y, ddof=0))
        papr = float(np.max(y) / mu) if mu > 0 else float("nan")
        cv2 = float((std / mu) ** 2) if mu > 0 else float("nan")

        if mu > 0:
            peak_mask = y >= mu
            peak_avg = float(np.mean(y[peak_mask]))
            off_avg = float(np.mean(y[~peak_mask])) if (~peak_mask).any() else 0.0
        else:
            peak_avg = 0.0
            off_avg = 0.0

        rows.append(
            {
                "slice_id": sid,
                "mean": mu,
                "variance": var,
                "std": std,
                "papr": papr,
                "peak_hour_avg": peak_avg,
                "off_peak_hour_avg": off_avg,
            }
        )
        burst_rows.append(
            {
                "slice_id": sid,
                "papr": papr,
                "burstiness_cv2": cv2,
            }
        )

    stats_df = pd.DataFrame(rows)
    burst_df = pd.DataFrame(burst_rows)
    return stats_df, burst_df


def slice_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between slices using aligned (date, hour) totals."""
    h = _hourly_slice_totals(df)
    if h.empty:
        return pd.DataFrame()

    pivot = h.pivot_table(
        index=["date", "hour"],
        columns="slice_id",
        values="total_mb",
        aggfunc="sum",
        fill_value=0.0,
    )
    corr = pivot.corr()
    return corr
