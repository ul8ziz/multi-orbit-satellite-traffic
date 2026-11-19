"""Forecast vs actual plots (test horizon)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_forecast_comparison(
    slice_id: str,
    test_index: pd.DatetimeIndex | pd.Index,
    y_test: np.ndarray,
    pred_sarima: np.ndarray,
    pred_lstm: np.ndarray,
    lstm_backend: str,
    path: Path,
) -> None:
    """Plot actual vs SARIMA vs neural forecast on the test segment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(y_test))
    if hasattr(test_index, "to_pydatetime"):
        x_lab = pd.to_datetime(test_index)
    else:
        x_lab = x

    # ``total_mb`` holds bytes (UPLOAD + DOWNLOAD); plot in GB. NaN (no data) leaves a gap.
    gb = 1e9
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(x_lab, np.asarray(y_test, dtype=float) / gb, label="Actual", color="black", linewidth=1.6)
    ax.plot(x_lab, np.asarray(pred_sarima, dtype=float) / gb, label="SARIMA", color="tab:blue", alpha=0.85)
    lbl = "LSTM" if lstm_backend == "lstm" else "MLP (fallback)"
    ax.plot(x_lab, np.asarray(pred_lstm, dtype=float) / gb, label=lbl, color="tab:orange", alpha=0.85)
    ax.set_title(f"Forecast vs actual — slice {slice_id} (test)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic (GB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_forecast_error_metrics(comparison: pd.DataFrame, path: Path) -> None:
    """Plot RMSE, MAE, spike RMSE, and spike MAE by slice and model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    required = {"slice_id", "model", "rmse", "mae", "spike_rmse", "spike_mae"}
    missing = required.difference(comparison.columns)
    if missing:
        msg = f"Forecast comparison table is missing columns: {sorted(missing)}"
        raise ValueError(msg)

    df = comparison.copy()
    df["model"] = df["model"].replace({"sarima": "SARIMA", "lstm": "LSTM", "mlp": "MLP"})
    slices = [s for s in ["URLLC", "eMBB", "mMTC"] if s in set(df["slice_id"])]
    if not slices:
        slices = list(df["slice_id"].drop_duplicates())
    models = [m for m in ["SARIMA", "LSTM", "MLP"] if m in set(df["model"])]
    metrics = [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("spike_rmse", "Spike RMSE"),
        ("spike_mae", "Spike MAE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=140)
    x = np.arange(len(slices))
    width = 0.35 if len(models) <= 2 else 0.25
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    for ax, (col, title) in zip(axes.ravel(), metrics):
        pivot = (
            df.pivot_table(index="slice_id", columns="model", values=col, aggfunc="first")
            .reindex(index=slices, columns=models)
            / 1e9
        )
        for offset, model in zip(offsets, models):
            ax.bar(x + offset, pivot[model].values, width=width, label=model)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(slices)
        ax.set_ylabel("Error, GB")
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle("Forecast error metrics by slice and model", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
