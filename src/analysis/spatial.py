"""500 km × 500 km synthetic spatial model and demand in long format."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRID_KM: float = 500.0
N_CELLS: int = 500
CELL_SIZE_KM: float = GRID_KM / N_CELLS
CLUSTER_SHARE: float = 0.8
N_CLUSTER_CENTERS: int = 5


def _clip_coord(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, GRID_KM - 1e-9)


def assign_user_coordinates(
    msisdns: pd.Series | np.ndarray | list[Any],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each user a position on a 500×500 km area and a 500×500 grid cell.

    * ``CLUSTER_SHARE`` (80%) of users are placed around random cluster centers with
      Gaussian jitter (σ = 15 km).
    * The remainder are uniform on ``[0, 500) × [0, 500)``.

    Grid indices ``grid_i``, ``grid_j`` are in ``0 .. N_CELLS-1`` (1 km cells).

    Parameters
    ----------
    msisdns:
        Unique subscriber identifiers (order preserved after unique).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: ``MASKED_MSISDN``, ``x_km``, ``y_km``, ``grid_i``, ``grid_j``.
    """
    ids = pd.Series(msisdns).drop_duplicates()
    n = len(ids)
    if n == 0:
        return pd.DataFrame(columns=["MASKED_MSISDN", "x_km", "y_km", "grid_i", "grid_j"])

    rng = np.random.default_rng(seed)
    n_cluster = int(round(CLUSTER_SHARE * n))
    n_cluster = min(max(n_cluster, 0), n)

    centers_x = rng.uniform(50.0, 450.0, size=N_CLUSTER_CENTERS)
    centers_y = rng.uniform(50.0, 450.0, size=N_CLUSTER_CENTERS)
    sigma_km = 15.0

    x = np.empty(n)
    y = np.empty(n)

    cluster_idx = rng.permutation(n)[:n_cluster]
    uniform_idx = np.setdiff1d(np.arange(n), cluster_idx, assume_unique=True)

    for k in cluster_idx:
        c = rng.integers(0, N_CLUSTER_CENTERS)
        x[k] = centers_x[c] + rng.normal(0.0, sigma_km)
        y[k] = centers_y[c] + rng.normal(0.0, sigma_km)

    x[uniform_idx] = rng.uniform(0.0, GRID_KM, size=len(uniform_idx))
    y[uniform_idx] = rng.uniform(0.0, GRID_KM, size=len(uniform_idx))

    x = _clip_coord(x)
    y = _clip_coord(y)
    grid_i = np.floor(x / CELL_SIZE_KM).astype(np.int32)
    grid_j = np.floor(y / CELL_SIZE_KM).astype(np.int32)
    grid_i = np.clip(grid_i, 0, N_CELLS - 1)
    grid_j = np.clip(grid_j, 0, N_CELLS - 1)

    out = pd.DataFrame(
        {
            "MASKED_MSISDN": ids.values,
            "x_km": x,
            "y_km": y,
            "grid_i": grid_i,
            "grid_j": grid_j,
        }
    )
    logger.info(
        "Spatial: %s users (%s clustered, %s uniform) on %sx%s km grid.",
        n,
        n_cluster,
        n - n_cluster,
        GRID_KM,
        GRID_KM,
    )
    return out


def build_demand_matrix_long(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Join traffic to coordinates and aggregate demand per cell, hour, and slice.

    Parameters
    ----------
    df:
        Preprocessed traffic with ``MASKED_MSISDN``, ``hour``, ``slice_id``,
        ``UPLOAD``, ``DOWNLOAD``, ``total_mb``.
    coords:
        Output of :func:`assign_user_coordinates`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``grid_i``, ``grid_j``, ``hour``, ``slice_id``, ``upload``,
        ``download``, ``total`` (sum of session ``total_mb`` in each cell bucket).
    """
    merged = df.merge(coords, on="MASKED_MSISDN", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "grid_i",
                "grid_j",
                "hour",
                "slice_id",
                "upload",
                "download",
                "total",
            ]
        )

    g = (
        merged.groupby(["grid_i", "grid_j", "hour", "slice_id"], as_index=False)[
            ["UPLOAD", "DOWNLOAD", "total_mb"]
        ]
        .sum()
        .rename(columns={"UPLOAD": "upload", "DOWNLOAD": "download", "total_mb": "total"})
    )
    return g.sort_values(["grid_i", "grid_j", "hour", "slice_id"]).reset_index(drop=True)
