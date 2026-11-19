"""Run per-slice forecasting (SARIMA + LSTM/MLP), metrics, tables, and figures."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.neural_forecast import forecast_train_test as lstm_forecast_train_test
from src.forecasting.sarima_forecast import forecast_train_test as sarima_forecast_train_test
from src.forecasting.timeseries import build_per_slice_series, temporal_train_test_split
from src.metrics.forecast_metrics import mae, rmse, spike_mae, spike_rmse
from src.visualization.forecast_figures import plot_forecast_comparison

logger = logging.getLogger(__name__)

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_slug(slice_id: str) -> str:
    s = SAFE_NAME.sub("_", str(slice_id)).strip("_")
    return s or "slice"


def _metrics_block(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    r = rmse(y_test, y_pred)
    a = mae(y_test, y_pred)
    sm, n_sp = spike_mae(y_test, y_pred, y_train)
    sr, _ = spike_rmse(y_test, y_pred, y_train)
    return {
        "rmse": r,
        "mae": a,
        "spike_mae": sm,
        "spike_rmse": sr,
        "n_spikes": n_sp,
        "n_test": int(np.isfinite(np.asarray(y_test, dtype=float)).sum()),  # observed hours only
    }


def run_forecasting(
    project_root: Path,
    *,
    cleaned_path: Path | None = None,
    hourly_slice_totals_path: Path | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    horizon: str = "one_step",
) -> dict[str, Path]:
    """Forecast each slice hourly series; write tables under ``outputs/tables`` and figures.

    Parameters
    ----------
    project_root:
        Project root (contains ``data/processed``).
    cleaned_path:
        Очищенный трафик (CSV или Parquet). Если задан ``hourly_slice_totals_path``, полный файл не читается.
    hourly_slice_totals_path:
        Агрегат ``date``, ``hour``, ``slice_id``, ``total_mb`` (Parquet), предпочтителен после chunked Stage A.
    test_ratio:
        Fraction of trailing timesteps held out for testing.
    seed:
        RNG seed for neural models.
    horizon:
        ``"one_step"`` (default) or ``"multi_step"`` — see the model modules.

    Hours without records are kept as missing (``NaN``), never as zero traffic; metrics use
    observed hours only.

    Returns
    -------
    dict
        Paths to written CSV files.
    """
    project_root = Path(project_root)
    proc_default = project_root / "data" / "processed" / "cleaned_traffic.csv"
    proc_pq = project_root / "data" / "processed" / "cleaned_traffic.parquet"
    proc = Path(cleaned_path) if cleaned_path else proc_default
    hourly_default = project_root / "data" / "processed" / "hourly_slice_totals.parquet"
    hp = Path(hourly_slice_totals_path) if hourly_slice_totals_path else hourly_default

    tables = project_root / "outputs" / "tables"
    figures = project_root / "outputs" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if hp.is_file():
        hdf = pd.read_parquet(hp)
        if not {"date", "hour", "slice_id", "total_mb"}.issubset(hdf.columns):
            msg = f"{hp} must contain date, hour, slice_id, total_mb"
            raise ValueError(msg)
        hdf = hdf.copy()
        hdf["date"] = pd.to_datetime(hdf["date"]).dt.normalize()
        hdf["hour"] = pd.to_numeric(hdf["hour"], errors="coerce").fillna(0).astype(int)
        series_map = build_per_slice_series(hdf)
    else:
        if not proc.is_file() and proc_pq.is_file():
            proc = proc_pq
        if not proc.is_file():
            msg = f"Need cleaned traffic at {proc} or hourly_slice_totals at {hp}; run Stage A first."
            raise FileNotFoundError(msg)

        if proc.suffix.lower() == ".parquet":
            df = pd.read_parquet(proc)
        else:
            df = pd.read_csv(proc)
        if "slice_id" not in df.columns:
            from src.analysis.slices import add_slice_column

            if "APP_CLASS" not in df.columns:
                msg = f"{proc} must include slice_id or APP_CLASS."
                raise ValueError(msg)
            df = add_slice_column(df)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
        series_map = build_per_slice_series(df)

    rows_sarima: list[pd.DataFrame] = []
    rows_lstm: list[pd.DataFrame] = []
    met_s: list[dict[str, Any]] = []
    met_l: list[dict[str, Any]] = []
    comp: list[dict[str, Any]] = []

    for sid, y in series_map.items():
        if len(y) < 8:
            logger.warning("Skipping slice %s: series too short (%s).", sid, len(y))
            continue

        train, test = temporal_train_test_split(y, test_ratio=test_ratio)
        y_train_arr = train.values.astype(float)
        y_test_arr = test.values.astype(float)

        df_s, pred_s = sarima_forecast_train_test(y, train, test, horizon=horizon)
        df_s.insert(0, "slice_id", sid)
        rows_sarima.append(df_s)

        mb = _metrics_block(y_train_arr, y_test_arr, pred_s)
        met_s.append({"slice_id": sid, **mb})

        df_l, pred_l, backend = lstm_forecast_train_test(y, train, test, seed=seed, horizon=horizon)
        df_l.insert(0, "slice_id", sid)
        df_l["backend"] = backend
        rows_lstm.append(df_l)

        mb_l = _metrics_block(y_train_arr, y_test_arr, pred_l)
        met_l.append({"slice_id": sid, "backend": backend, **mb_l})

        comp.append({"slice_id": sid, "model": "sarima", **{k: mb[k] for k in ("rmse", "mae", "spike_mae", "spike_rmse", "n_spikes", "n_test")}})
        comp.append(
            {
                "slice_id": sid,
                "model": "lstm" if backend == "lstm" else "mlp",
                **{k: mb_l[k] for k in ("rmse", "mae", "spike_mae", "spike_rmse", "n_spikes", "n_test")},
            }
        )

        plot_forecast_comparison(
            sid,
            test.index,
            y_test_arr,
            pred_s,
            pred_l,
            backend,
            figures / f"forecast_vs_actual_{_safe_slug(sid)}.png",
        )

    if not rows_sarima:
        msg = "No slice series were long enough to forecast."
        raise RuntimeError(msg)

    pred_sarima = pd.concat(rows_sarima, ignore_index=True)
    pred_lstm = pd.concat(rows_lstm, ignore_index=True)
    ms = pd.DataFrame(met_s)
    ml = pd.DataFrame(met_l)
    cmp_df = pd.DataFrame(comp)

    p_sarima = tables / "forecast_predictions_sarima.csv"
    p_lstm = tables / "forecast_predictions_lstm.csv"
    p_ms = tables / "forecast_arima_metrics.csv"
    p_ml = tables / "forecast_lstm_metrics.csv"
    p_cmp = tables / "forecast_model_comparison.csv"

    pred_sarima.to_csv(p_sarima, index=False)
    pred_lstm.to_csv(p_lstm, index=False)
    ms.to_csv(p_ms, index=False)
    ml.to_csv(p_ml, index=False)
    cmp_df.to_csv(p_cmp, index=False)

    logger.info(
        "Forecasting done: %s slices; wrote predictions and metrics to %s",
        len(met_s),
        tables,
    )

    return {
        "forecast_predictions_sarima": p_sarima,
        "forecast_predictions_lstm": p_lstm,
        "forecast_arima_metrics": p_ms,
        "forecast_lstm_metrics": p_ml,
        "forecast_model_comparison": p_cmp,
    }
