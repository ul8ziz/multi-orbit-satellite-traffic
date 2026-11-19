"""Derive a single arrival rate λ from cleaned traffic (constant CTMC input)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def derive_lambda_scenarios(
    cleaned_csv: Path,
    *,
    alpha: float = 1.0,
) -> tuple[float, float, pd.DataFrame]:
    """Compute scenario arrival rates from real data.

    For each hourly bucket ``t`` (sum of ``total_mb`` over all rows in that hour):

    .. math::

        \\lambda_t = \\alpha \\cdot \\frac{\\text{total\\_mb}_t}{3600}

    Then:

    * **mean load:** ``λ = mean(λ_t)``
    * **peak load:** ``λ = max(λ_t)``

    Parameters
    ----------
    cleaned_csv:
        Stage A ``cleaned_traffic.csv`` with ``date``, ``hour``, ``total_mb``.
    alpha:
        Unit scaling (dimensionless).

    Returns
    -------
    lambda_mean, lambda_peak, hourly_table
        ``hourly_table`` has columns ``date``, ``hour``, ``total_mb``, ``lambda_t``.
    """

    p = Path(cleaned_csv)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p, columns=["date", "hour", "total_mb"])
    else:
        df = pd.read_csv(p)
    if "total_mb" not in df.columns:
        msg = "cleaned traffic must contain total_mb"
        raise ValueError(msg)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)

    hourly = df.groupby(["date", "hour"], as_index=False)["total_mb"].sum()
    hourly["lambda_t"] = alpha * hourly["total_mb"] / 3600.0

    lam_mean = float(hourly["lambda_t"].mean())
    lam_peak = float(hourly["lambda_t"].max())

    logger.info(
        "λ scenarios: mean=%s peak=%s (alpha=%s, n_hourly=%s)",
        lam_mean,
        lam_peak,
        alpha,
        len(hourly),
    )
    return lam_mean, lam_peak, hourly
