"""Validate, clean, and harmonize loaded records before persistence to ``data/processed``."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_start_hour(series: pd.Series) -> pd.Series:
    """Parse ``START_HOUR`` into timezone-naive datetimes (coerce invalid to NaT)."""
    return pd.to_datetime(series, errors="coerce", utc=False)


def remove_extreme_outliers_total_mb(
    total_mb: pd.Series,
    iqr_factor: float = 3.0,
) -> pd.Series:
    """Boolean mask: True where ``total_mb`` is not an extreme outlier (Tukey-style IQR).

    Uses inner fences at ``Q1 - k*IQR`` and ``Q3 + k*IQR`` with ``k=iqr_factor`` (default 3
    for "far out" points). If IQR is zero, all finite positive values are kept.

    Parameters
    ----------
    total_mb:
        Non-negative traffic totals (same length as the frame being filtered).
    iqr_factor:
        Multiplier on IQR (3.0 corresponds to extreme / far-out outliers).

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``total_mb``.
    """
    s = pd.to_numeric(total_mb, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(False, index=total_mb.index)
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0 or not np.isfinite(iqr):
        return s.notna() & (s > 0)
    low = q1 - iqr_factor * iqr
    high = q3 + iqr_factor * iqr
    return s.notna() & (s >= low) & (s <= high)


def preprocess_step1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preprocess traffic: parse time, totals, drop non-positive and extreme outliers.

    Steps:

    #. Parse ``START_HOUR`` to ``start_ts``, ``hour`` (0--23), ``date``.
    #. ``total_mb = UPLOAD + DOWNLOAD`` (coerced numeric).
    #. Remove rows with ``total_mb <= 0`` or non-finite totals.
    #. Remove extreme outliers on ``total_mb`` (IQR-based).

    Parameters
    ----------
    df:
        Raw frame from :func:`loaders.load_raw_traffic`.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Cleaned frame and a small metadata dict (row counts, outlier thresholds).
    """
    n_raw = len(df)
    work = df.copy()
    work["UPLOAD"] = pd.to_numeric(work["UPLOAD"], errors="coerce")
    work["DOWNLOAD"] = pd.to_numeric(work["DOWNLOAD"], errors="coerce")
    work["start_ts"] = _parse_start_hour(work["START_HOUR"])
    work = work.dropna(subset=["start_ts"])
    n_after_time = len(work)
    work["hour"] = work["start_ts"].dt.hour.astype(np.int8)
    work["date"] = work["start_ts"].dt.normalize()
    work["total_mb"] = work["UPLOAD"] + work["DOWNLOAD"]
    work = work[np.isfinite(work["total_mb"])]
    work = work[work["total_mb"] > 0]
    n_after_positive = len(work)

    mask_out = remove_extreme_outliers_total_mb(work["total_mb"])
    removed = int((~mask_out).sum())
    work = work[mask_out].copy()
    n_final = len(work)

    meta: dict[str, Any] = {
        "rows_raw": n_raw,
        "rows_after_valid_time": n_after_time,
        "rows_after_positive_total": n_after_positive,
        "rows_final": n_final,
        "dropped_invalid_time": n_raw - n_after_time,
        "dropped_nonpositive_total": n_after_time - n_after_positive,
        "outliers_removed": removed,
    }
    logger.info(
        "Preprocessing: raw=%s valid_time=%s positive_total=%s final=%s (outliers=%s)",
        n_raw,
        n_after_time,
        n_after_positive,
        n_final,
        removed,
    )
    return work, meta
