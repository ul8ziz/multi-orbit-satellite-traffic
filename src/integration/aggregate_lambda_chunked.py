# -*- coding: utf-8 -*-
"""Chunked aggregation of telecom traffic into hourly lambda series."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

MAX_MB_PER_USER_PER_HOUR = 1_000.0
DEFAULT_CHUNK_ROWS = 500_000

_SLICE_MAP = {
    "web": "eMBB",
    "streaming": "eMBB",
    "file transfer": "eMBB",
    "music": "eMBB",
    "mail": "eMBB",
    "messaging": "URLLC",
    "games": "URLLC",
    "voip": "URLLC",
    "db": "URLLC",
    "others": "mMTC",
    "security": "mMTC",
    "terminals": "mMTC",
}


def _normalize_app_series(app: pd.Series) -> pd.Series:
    return app.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()


def _map_slice_id(app: pd.Series) -> pd.Series:
    norm = _normalize_app_series(app)
    return norm.map(_SLICE_MAP).fillna("mMTC")


def _iter_clean_chunks(
    input_csv: Path,
    chunk_rows: int,
    max_mb_per_user_per_hour: float,
) -> Iterable[pd.DataFrame]:
    usecols = ["START_HOUR", "APP_CLASS", "UPLOAD", "DOWNLOAD"]
    dtypes = {
        "APP_CLASS": "string",
        "UPLOAD": "float64",
        "DOWNLOAD": "float64",
    }
    reader = pd.read_csv(
        input_csv,
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunk_rows,
        engine="c",
    )
    for chunk in reader:
        work = chunk.copy()
        work["UPLOAD"] = pd.to_numeric(work["UPLOAD"], errors="coerce")
        work["DOWNLOAD"] = pd.to_numeric(work["DOWNLOAD"], errors="coerce")
        work["start_ts"] = pd.to_datetime(work["START_HOUR"], errors="coerce")
        work = work.dropna(subset=["start_ts", "UPLOAD", "DOWNLOAD"])

        work["total_mb"] = work["UPLOAD"] + work["DOWNLOAD"]
        work = work[work["total_mb"] > 0]
        work = work[work["total_mb"] <= max_mb_per_user_per_hour]
        if work.empty:
            continue

        work["hour_ts"] = work["start_ts"].dt.floor("h")
        work["hour"] = work["hour_ts"].dt.hour.astype("int8")
        work["slice_id"] = _map_slice_id(work["APP_CLASS"])

        yield work[["hour_ts", "hour", "slice_id", "total_mb"]]


def build_aggregated_lambda(
    input_csv: Path | str,
    output_csv: Path | str,
    *,
    alpha: float = 1.0,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_mb_per_user_per_hour: float = MAX_MB_PER_USER_PER_HOUR,
) -> Path:
    """Build `outputs/aggregated_lambda.csv` using chunked processing."""
    in_path = Path(input_csv).resolve()
    out_path = Path(output_csv).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    partials: list[pd.DataFrame] = []
    rows_in = 0
    rows_kept = 0

    usecols = ["START_HOUR"]
    for c in pd.read_csv(in_path, usecols=usecols, chunksize=chunk_rows):
        rows_in += len(c)

    for work in _iter_clean_chunks(in_path, chunk_rows, max_mb_per_user_per_hour):
        rows_kept += len(work)
        g = work.groupby(["hour_ts", "hour", "slice_id"], as_index=False)["total_mb"].sum()
        partials.append(g)

    if not partials:
        raise RuntimeError("No rows remained after cleaning filters.")

    agg = pd.concat(partials, ignore_index=True)
    agg = agg.groupby(["hour_ts", "hour", "slice_id"], as_index=False)["total_mb"].sum()

    full_hours = pd.date_range(agg["hour_ts"].min(), agg["hour_ts"].max(), freq="h")
    slices = pd.Index(["eMBB", "URLLC", "mMTC"], dtype="object")
    full_grid = (
        pd.MultiIndex.from_product([full_hours, slices], names=["hour_ts", "slice_id"])
        .to_frame(index=False)
        .sort_values(["hour_ts", "slice_id"])
        .reset_index(drop=True)
    )
    full_grid["hour"] = full_grid["hour_ts"].dt.hour.astype("int8")

    out = full_grid.merge(agg, on=["hour_ts", "hour", "slice_id"], how="left")
    out["total_mb"] = out["total_mb"].fillna(0.0)
    out["alpha"] = float(alpha)
    out["lambda_t"] = out["alpha"] * out["total_mb"] / 3600.0

    out = out[["hour_ts", "hour", "slice_id", "total_mb", "alpha", "lambda_t"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info(
        "aggregated lambda saved: %s (rows_in=%s, rows_kept=%s, rows_out=%s)",
        out_path,
        rows_in,
        rows_kept,
        len(out),
    )
    return out_path
