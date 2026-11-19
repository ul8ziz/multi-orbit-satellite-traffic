"""LSTM forecaster with sklearn MLPRegressor fallback."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

WINDOW = 24
HIDDEN = 48
EPOCHS_LSTM = 120
LR = 0.01
MLP_HIDDEN = (64, 32)


def _build_xy(series_scaled: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for t in range(window, len(series_scaled)):
        x_list.append(series_scaled[t - window : t])
        y_list.append(series_scaled[t])
    if not x_list:
        return np.empty((0, window)), np.empty((0,))
    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=float)


def _window_size(n_train: int) -> int:
    return min(WINDOW, max(4, n_train // 3))


@dataclass
class _TorchPack:
    model: Any


def _train_lstm(x_tr: np.ndarray, y_tr: np.ndarray, *, seed: int) -> _TorchPack:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    class LSTMNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x: Any) -> Any:
            o, _ = self.lstm(x)
            return self.fc(o[:, -1, :])

    model = LSTMNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    x_t = torch.from_numpy(x_tr[:, :, np.newaxis].astype(np.float32))
    y_t = torch.from_numpy(y_tr[:, np.newaxis].astype(np.float32))
    model.train()
    for _ in range(EPOCHS_LSTM):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    model.eval()
    return _TorchPack(model=model)


def _train_mlp(x_tr: np.ndarray, y_tr: np.ndarray, *, seed: int) -> MLPRegressor:
    mlp = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN,
        max_iter=500,
        random_state=seed,
        early_stopping=True,
    )
    mlp.fit(x_tr, y_tr)
    return mlp


def _naive_forecast(
    y: pd.Series,
    train: pd.Series,
    test: pd.Series,
    last: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    for t in train.index:
        rows.append(
            {
                "ts": t,
                "y_actual": float(y.loc[t]),
                "y_predicted": last,
                "split": "train",
            }
        )
    pred = np.full(len(test), last)
    for k, t in enumerate(test.index):
        rows.append(
            {
                "ts": t,
                "y_actual": float(y.loc[t]),
                "y_predicted": float(pred[k]),
                "split": "test",
            }
        )
    return pd.DataFrame(rows), pred


def seasonal_fill(y: pd.Series, reference: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Fill ``NaN`` hours with the hour-of-day mean of the observed ``reference`` values.

    Used **only for model inputs** (lag windows) of the neural forecaster, which cannot take
    missing values; imputed hours are never used as training targets or in the metrics.
    Returns the filled series and a boolean mask of the imputed positions.
    """
    miss = y.isna().to_numpy()
    ref = reference.dropna()
    if not miss.any() or ref.empty:
        return y.astype(float), miss
    if isinstance(y.index, pd.DatetimeIndex) and isinstance(ref.index, pd.DatetimeIndex):
        prof = ref.groupby(ref.index.hour).mean()
        fill = pd.Series(y.index.hour, index=y.index).map(prof)
        filled = y.astype(float).fillna(fill)
    else:
        filled = y.astype(float)
    return filled.fillna(float(ref.mean())), miss


def _build_xy_masked(
    series_scaled: np.ndarray, imputed: np.ndarray, window: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Windows ending before ``end`` whose inputs and target are all observed."""
    x_list, y_list = [], []
    for t in range(window, end):
        if imputed[t] or imputed[t - window : t].any():
            continue
        x_list.append(series_scaled[t - window : t])
        y_list.append(series_scaled[t])
    if not x_list:
        return np.empty((0, window)), np.empty((0,))
    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=float)


def forecast_train_test(
    y: pd.Series,
    train: pd.Series,
    test: pd.Series,
    *,
    seed: int = 42,
    horizon: str = "one_step",
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Min–max scaled LSTM (or MLP fallback) forecast of the test segment.

    ``horizon="one_step"`` (default, as in the article): each test hour is predicted from the
    actual previous ``window`` hours. ``horizon="multi_step"``: recursive forecast that feeds
    its own predictions back (the previous behaviour).

    Missing hours (``NaN``) are handled as follows: training uses only windows whose inputs
    and target are all observed; at prediction time missing input hours are filled with the
    hour-of-day mean of the training data (:func:`seasonal_fill`).
    """
    if horizon not in ("one_step", "multi_step"):
        raise ValueError("horizon must be 'one_step' or 'multi_step'.")
    scaler = MinMaxScaler()
    train_obs = train.dropna().values.astype(float).reshape(-1, 1)
    scaler.fit(train_obs)
    y_in, imputed = seasonal_fill(y, train)
    y_scaled = scaler.transform(y_in.values.astype(float).reshape(-1, 1)).ravel()

    n_train = len(train)
    window = _window_size(len(train_obs))
    x_tr, y_tr = _build_xy_masked(y_scaled, imputed, window, n_train)
    if imputed[:n_train].any():
        logger.info("Neural forecast: %d training windows fully observed (missing hours excluded).", len(y_tr))
    test_len = len(test)

    if len(y_tr) < 2:
        last = float(train.dropna().iloc[-1])
        df, arr = _naive_forecast(y, train, test, last)
        return df, arr, "mlp"

    predict_one_step: Callable[[np.ndarray], np.ndarray]
    backend: str

    torch_ok = False
    pack: _TorchPack | None = None
    try:
        if len(y_tr) >= 4:
            pack = _train_lstm(x_tr, y_tr, seed=seed)
            torch_ok = True
    except ImportError:
        logger.info("PyTorch not installed; using MLPRegressor for sequence forecast.")
    except Exception as e:  # noqa: BLE001
        logger.warning("LSTM training failed (%s); using MLPRegressor.", e)

    if torch_ok and pack is not None:
        import torch as T

        def predict_one_step(xb: np.ndarray) -> np.ndarray:
            xt = T.from_numpy(xb[:, :, np.newaxis].astype(np.float32))
            with T.no_grad():
                return pack.model(xt).numpy().ravel()

        backend = "lstm"
    else:
        mlp = _train_mlp(x_tr, y_tr, seed=seed)

        def predict_one_step(xb: np.ndarray) -> np.ndarray:
            return mlp.predict(xb)

        backend = "mlp"

    n = len(y_scaled)
    fitted_scaled = np.full(n, np.nan)
    if n > window:
        idx = np.arange(window, n)
        wins = np.stack([y_scaled[t - window : t] for t in idx], axis=0)
        fitted_scaled[idx] = predict_one_step(wins)

    if horizon == "one_step":
        test_pred_scaled = fitted_scaled[n_train : n_train + test_len].copy()
        if not np.isfinite(test_pred_scaled).all():  # test starts before the first full window
            test_pred_scaled = pd.Series(test_pred_scaled).bfill().ffill().fillna(0.0).to_numpy()
    else:
        buf = list(y_scaled[n_train - window : n_train])
        test_pred_scaled = np.zeros(test_len)
        for k in range(test_len):
            xin = np.asarray(buf[-window:], dtype=float).reshape(1, -1)
            p = predict_one_step(xin)[0]
            test_pred_scaled[k] = p
            buf.append(p)

    test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

    rows: list[dict[str, Any]] = []
    for t in train.index:
        i = y.index.get_loc(t)
        ya = float(y.loc[t])
        if np.isfinite(fitted_scaled[i]):
            yp = float(scaler.inverse_transform(np.array([[fitted_scaled[i]]]))[0, 0])
        else:
            yp = ya
        rows.append({"ts": t, "y_actual": ya, "y_predicted": yp, "split": "train"})
    for k, t in enumerate(test.index):
        rows.append(
            {
                "ts": t,
                "y_actual": float(y.loc[t]),
                "y_predicted": float(test_pred[k]),
                "split": "test",
            }
        )

    return pd.DataFrame(rows), test_pred, backend
