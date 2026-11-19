# -*- coding: utf-8 -*-
"""Проверка наличия всех артефактов Stage A и печать структурированной сводки."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


@dataclass
class StageAVerification:
    ok: bool
    missing: list[str]
    metrics: dict[str, Any]
    output_paths: dict[str, str]


STAGE_A_REQUIRED = [
    "outputs/tables/artifact1_slice_aggregated.csv",
    "outputs/reports/app_class_mapping_report.csv",
    "outputs/reports/preprocessing_summary.txt",
    "outputs/tables/user_profiles.csv",
    "outputs/tables/user_coordinates.csv",
    "outputs/tables/demand_matrix_long.csv",
    "outputs/tables/daily_profile.csv",
    "outputs/tables/burstiness_papr.csv",
    "outputs/tables/slice_descriptive_stats.csv",
    "outputs/tables/slice_correlation_matrix.csv",
    "outputs/figures/timeseries_hourly_by_slice.png",
    "outputs/figures/histogram_total_mb.png",
    "outputs/figures/heatmap_hour_slice.png",
    "outputs/figures/correlation_heatmap.png",
]

FORECAST_REQUIRED = [
    "outputs/tables/forecast_model_comparison.csv",
    "outputs/tables/forecast_predictions_sarima.csv",
    "outputs/tables/forecast_predictions_lstm.csv",
    "outputs/tables/forecast_arima_metrics.csv",
    "outputs/tables/forecast_lstm_metrics.csv",
    "outputs/figures/forecast_vs_actual_eMBB.png",
    "outputs/figures/forecast_vs_actual_mMTC.png",
    "outputs/figures/forecast_vs_actual_URLLC.png",
]

# Один из двух: полный датасет в Parquet или legacy CSV
CLEANED_ANY_OF = [
    "data/processed/cleaned_traffic.parquet",
    "data/processed/cleaned_traffic.csv",
]


def _load_metrics(project_root: Path) -> dict[str, Any]:
    p = project_root / "data" / "processed" / "stage_a_metrics.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _time_range_from_parquet(project_root: Path) -> tuple[Any, Any] | tuple[None, None]:
    pq_path = project_root / "data" / "processed" / "cleaned_traffic.parquet"
    if not pq_path.is_file():
        return None, None
    t = pd.read_parquet(pq_path, columns=["start_ts"])
    ts = pd.to_datetime(t["start_ts"])
    return ts.min(), ts.max()


def verify_stage_a_outputs(project_root: Path, *, expect_forecast: bool = False) -> StageAVerification:
    """Проверяет файлы и собирает метрики для сводки.

    Если ``expect_forecast`` True, дополнительно требуются артефакты ``run_forecasting.py``.
    """
    project_root = Path(project_root).resolve()
    missing: list[str] = []
    required = list(STAGE_A_REQUIRED)
    if expect_forecast:
        required.extend(FORECAST_REQUIRED)
    for rel in required:
        if not (project_root / rel).is_file():
            missing.append(rel)

    cleaned_ok = any((project_root / r).is_file() for r in CLEANED_ANY_OF)
    if not cleaned_ok:
        missing.append("data/processed/cleaned_traffic.parquet|cleaned_traffic.csv")

    hourly_pq = project_root / "data" / "processed" / "hourly_slice_totals.parquet"
    if not hourly_pq.is_file():
        missing.append("data/processed/hourly_slice_totals.parquet")

    metrics = _load_metrics(project_root)
    if not metrics:
        missing.append("data/processed/stage_a_metrics.json (ожидается после chunked Stage A)")

    t_min = metrics.get("time_min")
    t_max = metrics.get("time_max")
    if t_min is None or t_max is None:
        tr = _time_range_from_parquet(project_root)
        if tr[0] is not None:
            t_min, t_max = tr[0].isoformat(), tr[1].isoformat()

    out_paths: dict[str, str] = {}
    _extra = CLEANED_ANY_OF + ["data/processed/hourly_slice_totals.parquet", "data/processed/stage_a_metrics.json"]
    for rel in STAGE_A_REQUIRED + FORECAST_REQUIRED + _extra:
        p = project_root / rel
        if p.is_file():
            out_paths[rel] = str(p.resolve())

    m = metrics or {}
    full_metrics: dict[str, Any] = {
        "raw_row_count": m.get("n_raw_csv_pass2"),
        "cleaned_row_count": m.get("n_records_cleaned"),
        "unique_MASKED_MSISDN": m.get("n_unique_users"),
        "unique_APP_CLASS": m.get("n_unique_app_class"),
        "min_START_HOUR": t_min,
        "max_START_HOUR": t_max,
        "n_hourly_slice_rows": m.get("n_hourly_slice_rows"),
        "n_hourly_buckets": m.get("n_hourly_buckets"),
    }

    return StageAVerification(
        ok=len(missing) == 0,
        missing=missing,
        metrics=full_metrics,
        output_paths=out_paths,
    )


def print_stage_a_verification_report(v: StageAVerification, project_root: Path) -> None:
    """Печать структурированной сводки Stage A."""
    project_root = Path(project_root).resolve()
    print("", flush=True)
    print("=== STAGE A OUTPUT SUMMARY ===", flush=True)
    for k, val in v.metrics.items():
        print(f"  {k}: {val}", flush=True)
    print("  produced outputs (key paths):", flush=True)
    for rel in sorted(v.output_paths.keys()):
        print(f"    {rel}", flush=True)
    if v.missing:
        print("  MISSING:", flush=True)
        for m in v.missing:
            print(f"    - {m}", flush=True)
    else:
        print("  verification: all required paths present", flush=True)
    print("=== END STAGE A OUTPUT SUMMARY ===", flush=True)
