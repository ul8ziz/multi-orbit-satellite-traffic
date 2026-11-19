"""Article figures: every figure function runs on small synthetic data and writes a PNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.markov.leontyeva import default_params
from src.visualization import article_figures as af

SL = ("eMBB", "URLLC", "mMTC")


def _daily() -> pd.DataFrame:
    h = np.arange(24)
    rows = [
        {"slice_id": s, "hour": int(x), "mean_total": (k + 1) * 1e9 * (1.5 + np.sin(x / 24 * 2 * np.pi))}
        for k, s in enumerate(SL)
        for x in h
    ]
    return pd.DataFrame(rows)


def _preds() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2018-03-01", periods=30, freq="h").astype(str)
    rng = np.random.default_rng(0)
    out = []
    for name in ("sarima", "lstm"):
        rows = []
        for s in SL:
            y = np.r_[np.zeros(5), rng.uniform(1e9, 3e9, 25)]
            for t, v in zip(ts, y):
                rows.append({"slice_id": s, "ts": t, "y_actual": v, "y_predicted": v * 0.9, "split": "test"})
        out.append(pd.DataFrame(rows))
    return out[0], out[1]


def test_simple_figures(tmp_path: Path) -> None:
    daily = _daily()
    hourly = daily.rename(columns={"mean_total": "total"})
    for p in (
        af.fig01_daily_profile(daily, tmp_path / "1.png"),
        af.fig02_hourly_pdf(hourly, tmp_path / "2.png"),
        af.fig03_hourly_cdf(hourly, tmp_path / "3.png"),
        af.fig04_pipeline(tmp_path / "4.png"),
        af.fig04_pipeline(tmp_path / "4ru.png", lang="ru"),
        af.fig05_service_rate(default_params(), tmp_path / "5.png"),
        af.fig06_state_diagram(tmp_path / "6.png"),
    ):
        assert p.is_file() and p.stat().st_size > 1000


def test_forecast_figures(tmp_path: Path) -> None:
    sar, lstm = _preds()
    pred = af.test_predictions(sar, lstm, "eMBB")
    assert list(pred.columns) == ["ts", "actual", "sarima", "lstm"]
    assert len(pred) == 30
    assert af.fig07_forecast_vs_actual(pred, "eMBB", tmp_path / "7.png").is_file()
    assert af.fig08_10_error_hist(pred, "eMBB", tmp_path / "8.png").is_file()
    assert af.fig08_10_error_hist(pred, "eMBB", tmp_path / "8d.png", drop_missing=True).is_file()


def test_sensitivity_monotone(tmp_path: Path) -> None:
    tab = af.sensitivity_table([1.0, 2.0, 4.0], configs=((3, 1), (5, 2)))
    for _, g in tab.groupby("N"):
        assert g["P_loss"].is_monotonic_increasing
        assert g["util_LEO"].is_monotonic_increasing
    # Larger system blocks less at the same λ.
    at4 = tab[tab["lambda"] == 4.0].set_index("N")
    assert at4.loc[5, "P_loss"] < at4.loc[3, "P_loss"]
    assert af.fig11_sensitivity(tab, tmp_path / "11.png").is_file()


def test_quasi_stationary_alpha(tmp_path: Path) -> None:
    tab, alpha = af.quasi_stationary_table(_daily(), target_mean_lambda=3.0)
    assert len(tab) == 24 and alpha > 0
    assert tab["lambda"].mean() == pytest.approx(3.0)
    tab2, a2 = af.quasi_stationary_table(_daily(), alpha=alpha / 2)
    assert a2 == pytest.approx(alpha / 2)
    assert tab2["lambda"].mean() == pytest.approx(1.5)
    assert af.fig12_quasi_stationary(tab, tmp_path / "12.png").is_file()


def test_distribution_fits_recover_lognormal(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    rec = pd.DataFrame(
        {
            "slice_id": np.repeat(SL, 4000),
            "total_mb": np.concatenate([rng.lognormal(np.log(sc), s, 4000) for s, sc in
                                        ((1.6, 32000), (1.9, 5600), (1.8, 9700))]),
        }
    )
    fits = af.fit_distributions(rec).set_index("slice_id")
    assert fits.loc["eMBB", "lognorm_shape"] == pytest.approx(1.6, rel=0.05)
    assert fits.loc["URLLC", "lognorm_scale"] == pytest.approx(5600, rel=0.1)
    assert (fits["lognorm_KS"] < fits["normal_KS"]).all()
    assert (fits["lognorm_KS"] < fits["gamma_KS"]).all()
    assert af.fig02b_record_pdf(rec, fits.reset_index(), tmp_path / "2b.png").is_file()
    assert af.fig03b_record_cdf(rec, fits.reset_index(), tmp_path / "3b.png").is_file()
