"""Forecast evaluation: RMSE, MAE, and error on traffic spikes."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    d = y_true[m] - y_pred[m]
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def spike_mask(
    y_train_actual: np.ndarray,
    y_eval_actual: np.ndarray,
    quantile: float = 0.9,
) -> np.ndarray:
    """Boolean mask: test points whose actual exceeds the train ``quantile`` level.

    Defines "spikes" relative to the training distribution (assignment-style peak load).
    Missing values (``NaN``) are ignored on both sides.
    """
    thr = float(np.nanquantile(np.asarray(y_train_actual, dtype=float), quantile))
    return np.asarray(y_eval_actual, dtype=float) > thr


def spike_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_for_threshold: np.ndarray,
    quantile: float = 0.9,
) -> tuple[float, int]:
    """MAE on spike timesteps only; returns (mae, n_spikes)."""
    mask = spike_mask(y_train_for_threshold, y_true, quantile=quantile)
    if not mask.any():
        return float("nan"), 0
    yt = np.asarray(y_true, dtype=float).ravel()[mask]
    yp = np.asarray(y_pred, dtype=float).ravel()[mask]
    return mae(yt, yp), int(mask.sum())


def spike_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_for_threshold: np.ndarray,
    quantile: float = 0.9,
) -> tuple[float, int]:
    """RMSE on spike timesteps only."""
    mask = spike_mask(y_train_for_threshold, y_true, quantile=quantile)
    if not mask.any():
        return float("nan"), 0
    yt = np.asarray(y_true, dtype=float).ravel()[mask]
    yp = np.asarray(y_pred, dtype=float).ravel()[mask]
    return rmse(yt, yp), int(mask.sum())
