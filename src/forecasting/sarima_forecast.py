"""SARIMAX / ARIMA baseline forecasts per slice."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


def _fit_sarimax(
    train: pd.Series,
) -> Any:
    """Fit a compact SARIMAX; fall back to simpler orders on failure."""
    y = train.astype(float)
    n = len(y)
    if n < 4:
        return None, "too_short"

    # Prefer seasonal model when enough history exists (hourly m=24).
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    if n >= 72:
        candidates.append(((1, 1, 1), (1, 0, 1, 24)))
        candidates.append(((1, 0, 1), (1, 0, 1, 24)))
    if n >= 48:
        candidates.append(((1, 1, 1), (1, 0, 0, 24)))
    candidates.append(((1, 1, 1), (0, 0, 0, 0)))
    candidates.append(((0, 1, 1), (0, 0, 0, 0)))

    last_err: str | None = None
    for order, seasonal_order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False, maxiter=100)
            logger.info("SARIMAX fit OK order=%s seasonal=%s aic=%s", order, seasonal_order, getattr(res, "aic", None))
            return res, None
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            continue

    logger.warning("SARIMAX failed for all candidates: %s", last_err)
    return None, last_err


def forecast_train_test(
    y: pd.Series,
    train: pd.Series,
    test: pd.Series,
    *,
    horizon: str = "one_step",
) -> tuple[pd.DataFrame, np.ndarray]:
    """In-sample fitted train values and test forecast.

    Missing hours (``NaN``) in ``train`` are skipped by the Kalman filter, both in estimation
    and in prediction.

    Parameters
    ----------
    horizon:
        ``"one_step"`` (default, as described in the article): the model fitted on ``train``
        is applied with the **same parameters** to the whole series and each test hour is
        predicted from all observations before it (Ŷ_{t+1|t}).
        ``"multi_step"``: one forecast of ``len(test)`` steps from the end of ``train``.

    Returns
    -------
    predictions_df
        Rows with ``ts``, ``y_actual``, ``y_predicted``, ``split``.
    test_pred
        Forecast aligned to ``test`` index (length ``len(test)``).
    """
    if horizon not in ("one_step", "multi_step"):
        raise ValueError("horizon must be 'one_step' or 'multi_step'.")
    res, err = _fit_sarimax(train)
    train_idx, test_idx = train.index, test.index

    if res is None:
        # Naive: last observed train value as constant forecast
        logger.warning("SARIMAX unavailable (%s); using naive last-value forecast.", err)
        last = float(train.dropna().iloc[-1])
        fitted_train = np.full(len(train), last)
        test_pred = np.full(len(test), last)
    else:
        fitted_train = np.asarray(res.fittedvalues, dtype=float).ravel()
        if len(fitted_train) != len(train):
            fitted_train = np.asarray(res.predict(start=0, end=len(train) - 1), dtype=float).ravel()
        fitted_train = pd.Series(fitted_train).bfill().ffill().values
        if horizon == "one_step":
            full = res.apply(pd.concat([train, test]).astype(float))
            pred = full.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
            test_pred = np.asarray(pred, dtype=float).ravel()
        else:
            fc = res.get_forecast(steps=len(test))
            test_pred = np.asarray(fc.predicted_mean, dtype=float).ravel()
        if len(test_pred) != len(test):
            test_pred = np.resize(test_pred, len(test))

    rows: list[dict[str, Any]] = []
    for i, t in enumerate(train_idx):
        rows.append(
            {
                "ts": t,
                "y_actual": float(train.loc[t]),
                "y_predicted": float(fitted_train[min(i, len(fitted_train) - 1)]),
                "split": "train",
            }
        )
    for i, t in enumerate(test_idx):
        rows.append(
            {
                "ts": t,
                "y_actual": float(test.loc[t]),
                "y_predicted": float(test_pred[i]),
                "split": "test",
            }
        )

    df = pd.DataFrame(rows)
    return df, test_pred
