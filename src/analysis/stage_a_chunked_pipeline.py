# -*- coding: utf-8 -*-
"""Stage A для больших CSV: два прохода по сырому файлу, Parquet, чекпойнты, без полной загрузки в RAM."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.analysis.slice_statistics import (
    compute_slice_stats_and_papr,
    daily_profile_long,
    slice_correlation_matrix,
)
from src.analysis.slices import app_class_mapping_table, normalize_app_class
from src.analysis.spatial import assign_user_coordinates
from src.analysis.users import classify_user_profiles
from src.preprocessing.chunked_preprocess import (
    compute_tukey_bounds_from_sample,
    load_checkpoint,
    pass1_collect_total_mb,
    pass2_write_parquet_and_partials,
    write_checkpoint,
)
from src.preprocessing.loaders import (
    MIN_FULL_DATASET_ROWS,
    _assert_production_dataset_rules,
    _allow_alternate_dataset_for_dev,
    _sniff_csv_separator,
)
from src.visualization.stage_a_figures import (
    plot_correlation_heatmap,
    plot_heatmap_hour_slice,
    plot_histogram_binned,
    plot_time_series_hourly_slice,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK = 500_000


def _fingerprint_raw(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}


def _step_print(
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


def _write_app_class_mapping_report(path: Path, app_classes_seen: set[str] | None = None) -> None:
    df = app_class_mapping_table(app_classes_seen)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _step_print("app_class_mapping_report", len(df), len(df), 0.0, path)


def _normalized_app_classes_from_parquet(parquet_path: Path, batch_rows: int) -> set[str]:
    """Collect normalized APP_CLASS labels from cleaned Parquet for resumed runs."""
    pf = pq.ParquetFile(parquet_path)
    app_classes_seen: set[str] = set()
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["APP_CLASS"]):
        chunk = batch.to_pandas()
        for app_class in chunk["APP_CLASS"].astype("string").unique():
            normalized_app = normalize_app_class(app_class)
            if normalized_app:
                app_classes_seen.add(normalized_app)
    return app_classes_seen


def _demand_matrix_from_parquet(
    parquet_path: Path,
    coords: pd.DataFrame,
    batch_rows: int,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    pf = pq.ParquetFile(parquet_path)
    parts: list[pd.DataFrame] = []
    rows_in = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["MASKED_MSISDN", "hour", "slice_id", "UPLOAD", "DOWNLOAD", "total_mb"]):
        chunk = batch.to_pandas()
        rows_in += len(chunk)
        merged = chunk.merge(coords, on="MASKED_MSISDN", how="inner")
        if merged.empty:
            continue
        g = (
            merged.groupby(["grid_i", "grid_j", "hour", "slice_id"], as_index=False)[
                ["UPLOAD", "DOWNLOAD", "total_mb"]
            ]
            .sum()
            .rename(columns={"UPLOAD": "upload", "DOWNLOAD": "download", "total_mb": "total"})
        )
        parts.append(g)
    if not parts:
        out = pd.DataFrame(
            columns=["grid_i", "grid_j", "hour", "slice_id", "upload", "download", "total"]
        )
    else:
        cat = pd.concat(parts, ignore_index=True)
        out = (
            cat.groupby(["grid_i", "grid_j", "hour", "slice_id"], as_index=False)[
                ["upload", "download", "total"]
            ]
            .sum()
            .sort_values(["grid_i", "grid_j", "hour", "slice_id"])
            .reset_index(drop=True)
        )
    elapsed = time.perf_counter() - t0
    return out


def _histogram_total_mb_parquet(parquet_path: Path, bins: int, batch_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Накопление глобальной гистограммы total_mb по чанкам Parquet."""
    t0 = time.perf_counter()
    pf = pq.ParquetFile(parquet_path)
    tmin = np.inf
    tmax = -np.inf
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["total_mb"]):
        s = batch.to_pandas()["total_mb"].to_numpy(dtype=np.float64, copy=False)
        s = s[np.isfinite(s)]
        if s.size:
            tmin = min(tmin, float(s.min()))
            tmax = max(tmax, float(s.max()))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        edges = np.linspace(0, 1, bins + 1)
        counts = np.zeros(bins, dtype=np.float64)
        _step_print("histogram_scan_minmax", None, 0, time.perf_counter() - t0, parquet_path)
        return counts, edges
    edges = np.linspace(tmin, tmax, bins + 1)
    counts = np.zeros(bins, dtype=np.float64)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["total_mb"]):
        s = batch.to_pandas()["total_mb"].to_numpy(dtype=np.float64, copy=False)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        c, _ = np.histogram(s, bins=edges)
        counts += c.astype(np.float64)
    _step_print("histogram_accumulate", None, int(counts.sum()), time.perf_counter() - t0, parquet_path)
    return counts, edges


def run_stage_a_chunked(
    project_root: Path,
    input_path: Path,
    *,
    seed: int = 42,
    chunk_rows: int = DEFAULT_CHUNK,
    force: bool = False,
) -> dict[str, Path]:
    """Полный Stage A на полном датасете: чекпойнты, Parquet, логирование шагов."""
    project_root = Path(project_root).resolve()
    input_path = Path(input_path).resolve()

    processed_dir = project_root / "data" / "processed"
    scratch_dir = processed_dir / "scratch"
    ckpt_dir = processed_dir / "checkpoints"
    tables_dir = project_root / "outputs" / "tables"
    figures_dir = project_root / "outputs" / "figures"
    reports_dir = project_root / "outputs" / "reports"
    for d in (processed_dir, scratch_dir, ckpt_dir, tables_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not _allow_alternate_dataset_for_dev():
        _assert_production_dataset_rules(input_path)

    sep = _sniff_csv_separator(input_path)
    fp = _fingerprint_raw(input_path)

    cleaned_pq = processed_dir / "cleaned_traffic.parquet"
    hourly_pq = processed_dir / "hourly_slice_totals.parquet"
    user_tot_pq = processed_dir / "user_totals.parquet"
    preprocess_ck = load_checkpoint(ckpt_dir, "preprocess_done")

    meta_preprocess: dict[str, Any] = {}
    artifact1: pd.DataFrame
    hourly_slice_totals: pd.DataFrame
    user_totals: pd.DataFrame
    app_classes_seen: set[str]
    rows_raw_total = 0
    rows_clean = 0

    skip_preprocess = (
        not force
        and cleaned_pq.is_file()
        and preprocess_ck is not None
        and preprocess_ck.get("input_fingerprint") == fp
        and hourly_pq.is_file()
        and user_tot_pq.is_file()
    )

    if skip_preprocess:
        print("[Stage A] RESUME: пропуск предобработки (чекпойнт preprocess_done).", flush=True)
        t0 = time.perf_counter()
        artifact1 = pd.read_csv(tables_dir / "artifact1_slice_aggregated.csv")
        hourly_slice_totals = pd.read_parquet(hourly_pq)
        user_totals = pd.read_parquet(user_tot_pq)
        meta_preprocess = preprocess_ck.get("meta", {})
        rows_clean = int(meta_preprocess.get("rows_clean", len(user_totals)))
        rows_raw_total = int(meta_preprocess.get("rows_raw_pass2", rows_clean))
        stored_app_classes = meta_preprocess.get("app_class_labels_seen")
        if isinstance(stored_app_classes, list):
            app_classes_seen = {normalize_app_class(a) for a in stored_app_classes if normalize_app_class(a)}
        else:
            app_classes_seen = _normalized_app_classes_from_parquet(cleaned_pq, chunk_rows)
            meta_preprocess["app_class_labels_seen"] = sorted(app_classes_seen)
            meta_preprocess["unique_app_class_labels_seen"] = len(app_classes_seen)
            write_checkpoint(
                ckpt_dir,
                "preprocess_done",
                {
                    "input_fingerprint": fp,
                    "cleaned_parquet": str(cleaned_pq),
                    "meta": meta_preprocess,
                },
            )
        n_app_labels_resume = len(app_classes_seen)
        _step_print("resume_skip_preprocess", rows_raw_total, rows_clean, time.perf_counter() - t0, cleaned_pq)
    else:
        n_app_labels_resume = 0
        if force and cleaned_pq.is_file():
            cleaned_pq.unlink(missing_ok=True)
        t_pass1 = time.perf_counter()
        mm, rows_raw_p1, elapsed_p1 = pass1_collect_total_mb(input_path, sep, scratch_dir, chunk_rows)
        rows_raw_total = rows_raw_p1
        if not _allow_alternate_dataset_for_dev() and rows_raw_p1 < MIN_FULL_DATASET_ROWS:
            raise ValueError(f"Слишком мало строк в файле: {rows_raw_p1} < {MIN_FULL_DATASET_ROWS}")

        low, high, tukey_meta = compute_tukey_bounds_from_sample(np.asarray(mm), iqr_factor=3.0)
        del mm
        mmap_path = scratch_dir / "total_mb_pass1.dat"
        if mmap_path.exists():
            mmap_path.unlink()

        t_pass2_start = time.perf_counter()
        p2 = pass2_write_parquet_and_partials(
            input_path,
            sep,
            cleaned_pq,
            low,
            high,
            scratch_dir,
            chunk_rows,
        )
        elapsed_p2 = time.perf_counter() - t_pass2_start

        artifact1 = p2["artifact1"]
        hourly_slice_totals = p2["hourly_slice_totals"]
        user_totals = p2["user_totals"]
        app_classes_seen = p2["app_classes_seen"]
        rows_clean = int(p2["rows_clean"])
        rows_raw_total = int(p2["rows_raw_pass2"])

        hourly_slice_totals.to_parquet(hourly_pq, index=False)
        user_totals.to_parquet(user_tot_pq, index=False)
        _step_print("save_hourly_slice_totals", len(hourly_slice_totals), len(hourly_slice_totals), 0.0, hourly_pq)
        _step_print("save_user_totals", len(user_totals), len(user_totals), 0.0, user_tot_pq)

        meta_preprocess = {
            "rows_raw_pass1": rows_raw_p1,
            "rows_raw_pass2": rows_raw_total,
            "rows_clean": rows_clean,
            "elapsed_pass1": elapsed_p1,
            "elapsed_pass2": elapsed_p2,
            "tukey": tukey_meta,
            "tukey_low": low,
            "tukey_high": high,
            "unique_app_class_labels_seen": len(app_classes_seen),
            "app_class_labels_seen": sorted(app_classes_seen),
        }
        write_checkpoint(
            ckpt_dir,
            "preprocess_done",
            {
                "input_fingerprint": fp,
                "cleaned_parquet": str(cleaned_pq),
                "meta": meta_preprocess,
            },
        )

    n_app_labels = len(app_classes_seen) if app_classes_seen else n_app_labels_resume

    # --- Таблицы (имена по заданию + совместимость) ---
    t_tab = time.perf_counter()
    p_art = tables_dir / "artifact1_slice_aggregated.csv"
    artifact1.to_csv(p_art, index=False)
    _step_print("artifact1_slice_aggregated", len(artifact1), len(artifact1), time.perf_counter() - t_tab, p_art)

    p_map = reports_dir / "app_class_mapping_report.csv"
    _write_app_class_mapping_report(p_map, app_classes_seen)

    # user_profiles из user_totals (эквивалентно groupby на полном df)
    t_prof = time.perf_counter()
    up = classify_user_profiles(user_totals)
    p_prof = tables_dir / "user_profiles.csv"
    up.to_csv(p_prof, index=False)
    _step_print("user_profiles", len(user_totals), len(up), time.perf_counter() - t_prof, p_prof)

    coords = assign_user_coordinates(user_totals["MASKED_MSISDN"], seed=seed)
    p_coords = tables_dir / "user_coordinates.csv"
    t_coord = time.perf_counter()
    coords.to_csv(p_coords, index=False)
    _step_print("user_coordinates", len(user_totals), len(coords), time.perf_counter() - t_coord, p_coords)

    p_demand = tables_dir / "demand_matrix_long.csv"
    if (not skip_preprocess) or (not p_demand.is_file()) or force:
        t_dem = time.perf_counter()
        demand = _demand_matrix_from_parquet(cleaned_pq, coords, chunk_rows)
        demand.to_csv(p_demand, index=False)
        _step_print("demand_matrix_long", rows_clean, len(demand), time.perf_counter() - t_dem, p_demand)
    else:
        demand = pd.read_csv(p_demand)
        _step_print("demand_matrix_long_resume", None, len(demand), 0.0, p_demand)

    # daily / burst / slice stats / correlation — от почасовых агрегатов (малый df)
    mini = hourly_slice_totals.rename(columns={"total_mb": "total_mb"})
    t_daily = time.perf_counter()
    daily = daily_profile_long(mini)
    p_daily = tables_dir / "daily_profile.csv"
    daily.to_csv(p_daily, index=False)
    _step_print("daily_profile", len(mini), len(daily), time.perf_counter() - t_daily, p_daily)

    t_stats = time.perf_counter()
    slice_stats, burst_papr = compute_slice_stats_and_papr(mini)
    p_burst = tables_dir / "burstiness_papr.csv"
    burst_papr.to_csv(p_burst, index=False)
    p_slice_stats = tables_dir / "slice_descriptive_stats.csv"
    slice_stats.to_csv(p_slice_stats, index=False)
    p_slice_legacy = tables_dir / "slice_stats.csv"
    slice_stats.to_csv(p_slice_legacy, index=False)
    _step_print("slice_descriptive_stats", len(mini), len(slice_stats), time.perf_counter() - t_stats, p_slice_stats)

    t_corr = time.perf_counter()
    corr = slice_correlation_matrix(mini)
    p_corr = tables_dir / "slice_correlation_matrix.csv"
    corr.to_csv(p_corr)
    p_corr_legacy = tables_dir / "correlation.csv"
    corr.to_csv(p_corr_legacy)
    _step_print("slice_correlation_matrix", len(mini), corr.shape[0], time.perf_counter() - t_corr, p_corr)

    # preprocessing_summary.txt
    p_sum = reports_dir / "preprocessing_summary.txt"
    unique_users = int(user_totals["MASKED_MSISDN"].nunique())
    t_min = pd.to_datetime(hourly_slice_totals["date"]).min() if len(hourly_slice_totals) else None
    t_max = pd.to_datetime(hourly_slice_totals["date"]).max() if len(hourly_slice_totals) else None
    n_hourly = len(hourly_slice_totals)
    summary_lines = [
        "Stage A — preprocessing summary (chunked pipeline)",
        "==============================================",
        f"input_path: {input_path}",
        f"raw_rows_pass2: {rows_raw_total}",
        f"cleaned_rows: {rows_clean}",
        f"unique_users (MASKED_MSISDN): {unique_users}",
        f"unique APP_CLASS labels (seen): {n_app_labels}",
        f"date range (hourly aggregates): {t_min} .. {t_max}",
        f"hourly aggregate rows (date,hour,slice): {n_hourly}",
        f"tukey_bounds: low={meta_preprocess.get('tukey_low')} high={meta_preprocess.get('tukey_high')}",
        f"iqr_meta: {json.dumps(meta_preprocess.get('tukey', {}), ensure_ascii=False)}",
    ]
    p_sum.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _step_print("preprocessing_summary", rows_clean, rows_clean, 0.0, p_sum)

    h_agg = hourly_slice_totals.groupby(["date", "hour"], as_index=False)["total_mb"].sum()
    lam_t_series = h_agg["total_mb"] / 3600.0
    ts_combined = pd.to_datetime(h_agg["date"]) + pd.to_timedelta(h_agg["hour"].astype(int), unit="h")
    p_metrics = processed_dir / "stage_a_metrics.json"
    metrics_payload = {
        "n_records_cleaned": rows_clean,
        "n_raw_csv_pass2": rows_raw_total,
        "n_unique_users": unique_users,
        "n_unique_app_class": n_app_labels,
        "n_hourly_slice_rows": n_hourly,
        "n_hourly_buckets": len(h_agg),
        "time_min": ts_combined.min().isoformat(),
        "time_max": ts_combined.max().isoformat(),
        "lambda_t": {
            "mean": float(lam_t_series.mean()),
            "peak": float(lam_t_series.max()),
        },
    }
    p_metrics.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _step_print("stage_a_metrics.json", len(h_agg), len(h_agg), 0.0, p_metrics)

    # Figures
    t_fig = time.perf_counter()
    plot_time_series_hourly_slice(artifact1, figures_dir / "timeseries_hourly_by_slice.png")
    counts, edges = _histogram_total_mb_parquet(cleaned_pq, bins=50, batch_rows=chunk_rows)
    plot_histogram_binned(counts, edges, figures_dir / "histogram_total_mb.png")
    plot_heatmap_hour_slice(daily, figures_dir / "heatmap_hour_slice.png")
    plot_correlation_heatmap(corr, figures_dir / "correlation_heatmap.png")
    _step_print("figures", None, 4, time.perf_counter() - t_fig, figures_dir)

    manifest = reports_dir / "stage_a_manifest.txt"
    manifest_lines = [
        "Stage A outputs (chunked)",
        "========================",
        f"input: {input_path}",
        f"cleaned_traffic: {cleaned_pq}",
        f"hourly_slice_totals: {hourly_pq}",
        f"artifact1: {p_art}",
        f"app_class_mapping_report: {p_map}",
        f"preprocessing_summary: {p_sum}",
        f"user_profiles: {p_prof}",
        f"user_coordinates: {p_coords}",
        f"demand_matrix_long: {p_demand}",
        f"daily_profile: {p_daily}",
        f"burstiness_papr: {p_burst}",
        f"slice_descriptive_stats: {p_slice_stats}",
        f"slice_correlation_matrix: {p_corr}",
    ]
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    # --- Итоговая сводка ---
    print("", flush=True)
    print("=== STAGE A COMPLETION SUMMARY ===", flush=True)
    print(f"raw row count (CSV scan pass 2): {rows_raw_total}", flush=True)
    print(f"cleaned row count: {rows_clean}", flush=True)
    print(f"unique users: {unique_users}", flush=True)
    print(f"unique APP_CLASS count (seen in data): {n_app_labels}", flush=True)
    print(f"time range (hourly table dates): {t_min} .. {t_max}", flush=True)
    print(f"number of hourly aggregates (date×hour×slice rows): {n_hourly}", flush=True)
    print(f"number of hourly buckets (date×hour): {len(h_agg)}", flush=True)
    print("Generated outputs:", flush=True)
    for label, pth in [
        ("cleaned_traffic", cleaned_pq),
        ("hourly_slice_totals", hourly_pq),
        ("artifact1_slice_aggregated", p_art),
        ("app_class_mapping_report", p_map),
        ("preprocessing_summary", p_sum),
        ("user_profiles", p_prof),
        ("user_coordinates", p_coords),
        ("demand_matrix_long", p_demand),
        ("daily_profile", p_daily),
        ("burstiness_papr", p_burst),
        ("slice_descriptive_stats", p_slice_stats),
        ("slice_stats (legacy)", p_slice_legacy),
        ("slice_correlation_matrix", p_corr),
        ("correlation (legacy)", p_corr_legacy),
        ("manifest", manifest),
    ]:
        print(f"  {label}: {pth}", flush=True)
    for name in (
        "timeseries_hourly_by_slice.png",
        "histogram_total_mb.png",
        "heatmap_hour_slice.png",
        "correlation_heatmap.png",
    ):
        print(f"  figure: {figures_dir / name}", flush=True)
    print("  (forecasting: запустите python run_forecasting.py после Stage A)", flush=True)
    print("=== END STAGE A SUMMARY ===", flush=True)

    out: dict[str, Path] = {
        "cleaned_traffic": cleaned_pq,
        "hourly_slice_totals": hourly_pq,
        "artifact1_slice_aggregated": p_art,
        "app_class_mapping_report": p_map,
        "preprocessing_summary": p_sum,
        "user_profiles": p_prof,
        "user_coordinates": p_coords,
        "demand_matrix_long": p_demand,
        "daily_profile": p_daily,
        "burstiness_papr": p_burst,
        "slice_descriptive_stats": p_slice_stats,
        "slice_stats": p_slice_legacy,
        "slice_correlation_matrix": p_corr,
        "correlation": p_corr_legacy,
        "manifest": manifest,
        "stage_a_metrics": p_metrics,
    }
    return out
