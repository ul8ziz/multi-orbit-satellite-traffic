"""Common metric functions for comparing predictions or distributions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def score_predictions(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute RMSE and MAE for paired ground-truth and predicted values.

    Parameters
    ----------
    y_true:
        Observed targets.
    y_pred:
        Model predictions.

    Returns
    -------
    dict[str, float]
        Keys ``rmse`` and ``mae`` when inputs are array-like.
    """
    from src.metrics.forecast_metrics import mae as mae_fn
    from src.metrics.forecast_metrics import rmse as rmse_fn

    return {
        "rmse": rmse_fn(np.asarray(y_true), np.asarray(y_pred)),
        "mae": mae_fn(np.asarray(y_true), np.asarray(y_pred)),
    }
