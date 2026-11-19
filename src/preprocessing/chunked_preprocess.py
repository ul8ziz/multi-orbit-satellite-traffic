# -*- coding: utf-8 -*-
"""Двухпроходная предобработка больших CSV: Tukey IQR и запись Parquet по чанкам."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.analysis.slices import add_slice_column, normalize_app_class
from src.preprocessing.cleaning import _parse_start_hour
from src.preprocessing.loaders import REQUIRED_COLUMNS, _sniff_csv_separator

logger = logging.getLogger(__name__)

# Размер чанка чтения CSV (строк данных; без заголовка в счётте итератора pandas)
DEFAULT_CHUNK_ROWS = 500_000
# Верхняя граница выборки для точных квантилей Tukey (полный массив может не поместиться в сортировку numpy)
IQR_SAMPLE_CAP = 8_000_000


def _step_log(
    name: str,
    rows_in: int | None,
    rows_out: int | None,
    elapsed: float,
    out_path: Path | None = None,
) -> None:
    extra = f" output={out_path}" if out_path else ""
    print(
        f"[Stage A step] {name} | rows_in={rows_in!s} rows_out={rows_out!s} "
        f"elapsed_s={elapsed:.3f}{extra}",
        flush=True,
    )


def _chunk_to_positive_totals(chunk: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Парсинг времени, положительный total_mb; без отсечения выбросов. Возвращает (work, n_in, n_pos)."""
    n_in = len(chunk)
    work = chunk.copy()
    work["UPLOAD"] = pd.to_numeric(work["UPLOAD"], errors="coerce")
    work["DOWNLOAD"] = pd.to_numeric(work["DOWNLOAD"], errors="coerce")
    work["start_ts"] = _parse_start_hour(work["START_HOUR"])
    work = work.dropna(subset=["start_ts"])
    work["hour"] = work["start_ts"].dt.hour.astype(np.int8)
    work["date"] = work["start_ts"].dt.normalize()
    work["total_mb"] = work["UPLOAD"] + work["DOWNLOAD"]
    work = work[np.isfinite(work["total_mb"])]
    work = work[work["total_mb"] > 0]
    return work, n_in, len(work)


def compute_tukey_bounds_from_sample(total_mb_flat: np.ndarray, iqr_factor: float = 3.0) -> tuple[float, float, dict[str, Any]]:
    """Квантили 25/75 на выборке (или полностью, если коротко), границы Tukey."""
    n = int(total_mb_flat.size)
    if n == 0:
        return 0.0, 0.0, {"n_used": 0, "method": "empty"}
    rng = np.random.default_rng(42)
    if n > IQR_SAMPLE_CAP:
        idx = rng.choice(n, size=IQR_SAMPLE_CAP, replace=False)
        sample = np.asarray(total_mb_flat[idx], dtype=np.float64)
        method = f"random_sample_n={IQR_SAMPLE_CAP}"
    else:
        sample = np.asarray(total_mb_flat, dtype=np.float64)
        method = "full"
    q1, q3 = np.percentile(sample, [25, 75])
    iqr = q3 - q1
    if iqr <= 0 or not np.isfinite(iqr):
        low, high = float(np.min(sample)), float(np.max(sample))
        return low, high, {"n_used": n, "method": method, "q1": q1, "q3": q3, "iqr": 0.0}
    low = float(q1 - iqr_factor * iqr)
    high = float(q3 + iqr_factor * iqr)
    return low, high, {"n_used": n, "method": method, "q1": float(q1), "q3": float(q3), "iqr": float(iqr)}


def pass1_collect_total_mb(
    path: Path,
    sep: str,
    scratch_dir: Path,
    chunk_rows: int,
) -> tuple[np.ndarray, int, float]:
    """Проход 1: собрать все положительные total_mb в memmap-файл для оценки квантилей."""
    t0 = time.perf_counter()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = scratch_dir / "total_mb_pass1.dat"
    if mmap_path.exists():
        mmap_path.unlink()

    rows_raw = 0
    rows_pos = 0
    with open(mmap_path, "wb") as bf:
        reader = pd.read_csv(
            path,
            sep=sep,
            chunksize=chunk_rows,
            engine="c",
            quoting=0,
            dtype={
                "MASKED_MSISDN": "string",
                "APP_CLASS": "string",
            },
        )
        for chunk in reader:
            rows_raw += len(chunk)
            missing = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
            if missing:
                raise ValueError(f"Missing columns {missing}; got {list(chunk.columns)}")
            work, _n_in, n_pos = _chunk_to_positive_totals(chunk)
            rows_pos += n_pos
            if n_pos == 0:
                continue
            arr = work["total_mb"].to_numpy(dtype=np.float64, copy=False)
            bf.write(arr.tobytes())

    n = rows_pos
    if n == 0:
        elapsed = time.perf_counter() - t0
        _step_log("pass1_collect_total_mb", rows_raw, 0, elapsed, mmap_path)
        return np.array([], dtype=np.float64), rows_raw, elapsed

    mm = np.memmap(mmap_path, dtype=np.float64, mode="r", shape=(n,))
    elapsed = time.perf_counter() - t0
    _step_log("pass1_collect_total_mb", rows_raw, n, elapsed, mmap_path)
    return mm, rows_raw, elapsed


def _tukey_mask(total_mb: pd.Series, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(total_mb, errors="coerce")
    return s.notna() & (s >= low) & (s <= high)


def pass2_write_parquet_and_partials(
    path: Path,
    sep: str,
    out_parquet: Path,
    low: float,
    high: float,
    scratch_dir: Path,
    chunk_rows: int,
) -> dict[str, Any]:
    """Проход 2: фильтр Tukey, slice_id, запись Parquet, частичные агрегаты."""
    t0 = time.perf_counter()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    parts_hs: list[pd.DataFrame] = []
    parts_dhs: list[pd.DataFrame] = []
    parts_user: list[pd.DataFrame] = []
    app_classes_seen: set[str] = set()
    rows_raw = 0
    rows_clean = 0

    writer: pq.ParquetWriter | None = None

    try:
        reader = pd.read_csv(
            path,
            sep=sep,
            chunksize=chunk_rows,
            engine="c",
            quoting=0,
            dtype={"MASKED_MSISDN": "string", "APP_CLASS": "string"},
        )
        for chunk in reader:
            rows_raw += len(chunk)
            missing = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
            if missing:
                raise ValueError(f"Missing columns {missing}")
            work, _ni, n_pos = _chunk_to_positive_totals(chunk)
            if n_pos == 0:
                continue
            mask = _tukey_mask(work["total_mb"], low, high)
            work = work[mask].copy()
            if work.empty:
                continue
            work = add_slice_column(work)
            for a in work["APP_CLASS"].astype("string").unique():
                normalized_app = normalize_app_class(a)
                if normalized_app:
                    app_classes_seen.add(normalized_app)

            g_hs = (
                work.groupby(["hour", "slice_id"], as_index=False)[["UPLOAD", "DOWNLOAD", "total_mb"]]
                .sum()
                .rename(columns={"UPLOAD": "upload", "DOWNLOAD": "download", "total_mb": "total"})
            )
            parts_hs.append(g_hs)

            g_dhs = work.groupby(["date", "hour", "slice_id"], as_index=False)["total_mb"].sum()
            parts_dhs.append(g_dhs)

            g_u = work.groupby("MASKED_MSISDN", as_index=False)["total_mb"].sum()
            parts_user.append(g_u)

            out = work[
                ["MASKED_MSISDN", "APP_CLASS", "UPLOAD", "DOWNLOAD", "start_ts", "hour", "date", "total_mb", "slice_id"]
            ].copy()

            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, table.schema, compression="snappy")
            writer.write_table(table)
            rows_clean += len(out)

    finally:
        if writer is not None:
            writer.close()

    if writer is None or rows_clean == 0:
        msg = "Проход 2 не записал ни одной строки (пустой датасет или все отфильтрованы)."
        raise RuntimeError(msg)

    elapsed = time.perf_counter() - t0
    _step_log("pass2_write_parquet_and_partials", rows_raw, rows_clean, elapsed, out_parquet)

    def _merge_parts(parts: list[pd.DataFrame], keys: list[str], sums: list[str] | None) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=keys + (sums or []))
        cat = pd.concat(parts, ignore_index=True)
        if sums:
            return cat.groupby(keys, as_index=False)[sums].sum()
        return cat.groupby(keys, as_index=False).sum()

    artifact1 = _merge_parts(parts_hs, ["hour", "slice_id"], ["upload", "download", "total"])
    artifact1 = artifact1.sort_values(["hour", "slice_id"]).reset_index(drop=True)

    hourly_dhs = _merge_parts(parts_dhs, ["date", "hour", "slice_id"], ["total_mb"])
    hourly_dhs = hourly_dhs.sort_values(["date", "hour", "slice_id"]).reset_index(drop=True)

    user_totals = _merge_parts(parts_user, ["MASKED_MSISDN"], ["total_mb"])

    return {
        "rows_raw_pass2": rows_raw,
        "rows_clean": rows_clean,
        "elapsed_pass2": elapsed,
        "artifact1": artifact1,
        "hourly_slice_totals": hourly_dhs,
        "user_totals": user_totals,
        "app_classes_seen": app_classes_seen,
        "tukey_low": low,
        "tukey_high": high,
    }


def merge_partial_groupby_sum(
    frames: list[pd.DataFrame],
    group_cols: list[str],
    sum_cols: list[str],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=group_cols + sum_cols)
    cat = pd.concat(frames, ignore_index=True)
    return cat.groupby(group_cols, as_index=False)[sum_cols].sum()


def write_checkpoint(dir_path: Path, name: str, payload: dict[str, Any]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def load_checkpoint(dir_path: Path, name: str) -> dict[str, Any] | None:
    p = dir_path / f"{name}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
