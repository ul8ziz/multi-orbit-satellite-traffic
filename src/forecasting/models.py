"""Forecast model entry points."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.forecasting.pipeline import run_forecasting

logger = logging.getLogger(__name__)


def fit_forecast_model(
    project_root: Path,
    *,
    cleaned_path: Path | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Path]:
    """Run per-slice SARIMA and LSTM/MLP forecasting; write tables and figures.

    Parameters
    ----------
    project_root:
        Project directory.
    cleaned_path:
        Optional path to cleaned traffic CSV (default: ``data/processed/cleaned_traffic.csv``).
    test_ratio:
        Holdout fraction for chronological test split.
    seed:
        Random seed for neural models.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths to forecast CSV outputs under ``outputs/tables``.
    """
    return run_forecasting(project_root, cleaned_path=cleaned_path, test_ratio=test_ratio, seed=seed)
