"""Stage A figures: time series, histograms, demand and correlation heatmaps."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 120,
            "font.size": 10,
        }
    )


def plot_time_series_hourly_slice(
    artifact1: pd.DataFrame,
    path: Path,
) -> None:
    """Line plot: hourly ``total`` per ``slice_id``."""
    _apply_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact1.empty:
        logger.warning("Skipping time series: empty artifact1.")
        return

    fig, ax = plt.subplots()
    for sid, g in artifact1.groupby("slice_id"):
        gg = g.sort_values("hour")
        ax.plot(gg["hour"], gg["total"], marker="o", markersize=3, label=sid)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total (MB)")
    ax.set_title("Hourly traffic totals by slice")
    ax.legend(title="slice_id")
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_histogram_binned(counts: np.ndarray, bin_edges: np.ndarray, path: Path) -> None:
    """Гистограмма по накопленным счётчикам и границам корзин (без хранения всех строк)."""
    _apply_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, ax = plt.subplots()
    ax.bar(centers, counts, width=np.diff(bin_edges), align="center", color="steelblue", edgecolor="white")
    ax.set_xlabel("total_mb")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of session total_mb (chunked aggregation)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_histogram_total_mb(df: pd.DataFrame, path: Path) -> None:
    """Histogram of per-row ``total_mb`` (after preprocessing)."""
    _apply_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "total_mb" not in df.columns:
        logger.warning("Skipping histogram: no data.")
        return

    s = df["total_mb"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots()
    ax.hist(s, bins=min(50, max(10, int(np.sqrt(len(s))))), color="steelblue", edgecolor="white")
    ax.set_xlabel("total_mb")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of session total_mb")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_heatmap_hour_slice(daily_profile: pd.DataFrame, path: Path) -> None:
    """Heatmap: rows = hour (0--23), columns = slice, values = mean_total."""
    _apply_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if daily_profile.empty:
        logger.warning("Skipping heatmap: empty daily profile.")
        return

    pivot = daily_profile.pivot(index="hour", columns="slice_id", values="mean_total")
    pivot = pivot.reindex(range(24)).fillna(0.0)
    data = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(range(24))
    ax.set_yticklabels(range(24))
    ax.set_ylabel("Hour")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=15, ha="right")
    ax.set_title("Mean traffic: hour × slice")
    fig.colorbar(im, ax=ax, label="mean_total (MB)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    """Heatmap of slice--slice Pearson correlation."""
    _apply_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if corr.empty:
        logger.warning("Skipping correlation heatmap: empty matrix.")
        return

    labels = list(corr.columns)
    data = corr.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Slice correlation (Pearson)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
