"""Figures 1–12 of the journal article (docs/Dabwan_2026_Journal_Article.docx).

Every ``fig*`` function takes already-loaded data and an output path, so it can be reused and
tested on synthetic data; :mod:`run_article_figures` loads the project data and calls them all.

Units: the column ``total_mb`` / ``total`` in this project actually holds **bytes**
(``UPLOAD + DOWNLOAD``); figures convert with ``BYTES_PER_GB = 1e9``.

=====  ==========================================================  =====================================
Fig.   Content                                                     Data source
=====  ==========================================================  =====================================
1      Average daily traffic profile per slice                     outputs/tables/daily_profile.csv
2, 3   PDF / CDF of hourly aggregated traffic + Normal fit         outputs/tables/artifact1_slice_aggregated.csv
2b,3b  PDF / CDF of per-record volume + Normal/LogNormal/Gamma     data/processed/cleaned_traffic.parquet (sample)
       (and Table 3: KS statistics)
4      Forecasting pipeline diagram                                —
5      Service rate μ(n), LEO and MEO contributions                LeontyevaParams
6      CTMC state-transition diagram (N=3, M=2)                    —
7      Forecast vs actual (test)                                   outputs/tables/forecast_predictions_*.csv
8–10   Forecast error histograms SARIMA vs LSTM/MLP                outputs/tables/forecast_predictions_*.csv
11     P_loss and util_LEO vs λ for several (N, C_LEO)             CTMC (src/markov)
12     Quasi-stationary hourly λ(t), utilisation, blocking         daily_profile.csv + CTMC
=====  ==========================================================  =====================================
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from src.markov.ctmc import build_generator_matrix, stationary_distribution
from src.markov.leontyeva import LeontyevaParams, default_params, mu_n
from src.markov.stationary import compute_metrics

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1e9
SLICES: tuple[str, ...] = ("eMBB", "URLLC", "mMTC")
SLICE_COLORS = {"eMBB": "#2196F3", "URLLC": "#FF5722", "mMTC": "#4CAF50"}
HIST_COLORS = {"eMBB": "#64B5F6", "URLLC": "#FF8A65", "mMTC": "#81C784"}
LEO_COLOR, MEO_COLOR = "#0D47A1", "#BF360C"
FIG11_CONFIGS: tuple[tuple[int, int], ...] = ((3, 1), (5, 2), (7, 3), (10, 5), (15, 7))
NUMERICAL_FLOOR = 1e-14  # P_loss below this is round-off of the linear solver, not signal


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def _present(df: pd.DataFrame, col: str = "slice_id") -> list[str]:
    have = set(df[col].unique())
    return [s for s in SLICES if s in have]


# --------------------------------------------------------------------------- Fig. 1


def fig01_daily_profile(daily_profile: pd.DataFrame, path: Path) -> Path:
    """Fig. 1 — mean traffic per hour of day per slice (GB). Columns: slice_id, hour, mean_total."""

    _style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for s in _present(daily_profile):
        g = daily_profile[daily_profile["slice_id"] == s].sort_values("hour")
        ax.plot(g["hour"], g["mean_total"] / BYTES_PER_GB, marker="o", ms=4, lw=2,
                color=SLICE_COLORS[s], label=s)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Traffic (GB)")
    ax.set_title("Average Daily Traffic Profile per 3GPP Slice")
    ax.legend(loc="upper left")
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 2 / 3 (as in article)


def _hourly_gb(hourly: pd.DataFrame, s: str) -> np.ndarray:
    return hourly.loc[hourly["slice_id"] == s, "total"].to_numpy(dtype=float) / BYTES_PER_GB


def _normal_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _normal_cdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    from math import erf, sqrt

    return np.array([0.5 * (1 + erf((v - mu) / (sd * sqrt(2)))) for v in np.atleast_1d(x)])


def fig02_hourly_pdf(hourly: pd.DataFrame, path: Path, bins: int = 12) -> Path:
    """Fig. 2 — histogram (density) of hourly aggregated traffic per slice with a Normal fit.

    ``hourly``: columns ``slice_id``, ``total`` (bytes). In the article this is
    ``artifact1_slice_aggregated.csv`` — traffic per hour-of-day summed over all days (24 values).
    """

    _style()
    slices = _present(hourly)
    fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 4))
    for ax, s in zip(np.atleast_1d(axes), slices):
        x = _hourly_gb(hourly, s)
        mu, sd = float(x.mean()), float(x.std(ddof=0))
        ax.hist(x, bins=bins, density=True, color=HIST_COLORS[s], edgecolor="white", label="Empirical")
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, _normal_pdf(xs, mu, sd), "k--", lw=1.2, label=f"Normal (μ={mu:.2f}, σ={sd:.2f})")
        ax.set_title(f"{s} — Hourly PDF")
        ax.set_xlabel("Hourly Traffic (GB)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("PDF of Hourly Aggregated Traffic per Slice", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def fig03_hourly_cdf(hourly: pd.DataFrame, path: Path) -> Path:
    """Fig. 3 — empirical CDF of hourly aggregated traffic per slice vs Normal CDF."""

    _style()
    slices = _present(hourly)
    fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 4))
    for ax, s in zip(np.atleast_1d(axes), slices):
        x = np.sort(_hourly_gb(hourly, s))
        ecdf = np.arange(1, x.size + 1) / x.size
        mu, sd = float(x.mean()), float(x.std(ddof=0))
        ax.plot(x, ecdf, color=SLICE_COLORS[s], lw=1.6, label="Empirical CDF")
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, _normal_cdf(xs, mu, sd), "k--", lw=1.2, label="Normal CDF")
        ax.set_title(f"{s} — Hourly CDF")
        ax.set_xlabel("Hourly Traffic (GB)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("CDF of Hourly Aggregated Traffic per Slice", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 2b / 3b + Table 3


def sample_records(parquet_path: Path, fraction: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Uniform random sample of (slice_id, total_mb) from the cleaned parquet, row group by row group."""

    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(parquet_path)
    parts = []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["slice_id", "total_mb"]).to_pandas()
        parts.append(t[rng.random(len(t)) < fraction])
    df = pd.concat(parts, ignore_index=True)
    logger.info("Sampled %d records (fraction=%s) from %s", len(df), fraction, parquet_path)
    return df


def fit_distributions(records: pd.DataFrame) -> pd.DataFrame:
    """Table 3 — MLE fits (Normal, LogNormal, Gamma; loc fixed at 0) and KS statistics per slice.

    ``records``: columns ``slice_id``, ``total_mb`` (bytes per record). Non-positive values are
    dropped (log-normal and gamma have positive support); their share is reported.
    """

    from scipy import stats

    rows = []
    for s in _present(records):
        x_all = records.loc[records["slice_id"] == s, "total_mb"].to_numpy(dtype=float)
        x = x_all[x_all > 0]
        mu, sig = stats.norm.fit(x)
        ln_s, _, ln_sc = stats.lognorm.fit(x, floc=0)
        g_a, _, g_sc = stats.gamma.fit(x, floc=0)
        rows.append(
            {
                "slice_id": s,
                "n": int(x.size),
                "share_nonpositive_dropped": float(1 - x.size / max(x_all.size, 1)),
                "normal_mu": mu,
                "normal_sigma": sig,
                "normal_KS": stats.kstest(x, stats.norm(mu, sig).cdf).statistic,
                "lognorm_shape": ln_s,
                "lognorm_scale": ln_sc,
                "lognorm_KS": stats.kstest(x, stats.lognorm(ln_s, 0, ln_sc).cdf).statistic,
                "gamma_a": g_a,
                "gamma_scale": g_sc,
                "gamma_KS": stats.kstest(x, stats.gamma(g_a, 0, g_sc).cdf).statistic,
            }
        )
    return pd.DataFrame(rows)


def _fitted(fits: pd.DataFrame, s: str):
    from scipy import stats

    r = fits.set_index("slice_id").loc[s]
    return {
        "Normal": stats.norm(r["normal_mu"], r["normal_sigma"]),
        "LogNormal": stats.lognorm(r["lognorm_shape"], 0, r["lognorm_scale"]),
        "Gamma": stats.gamma(r["gamma_a"], 0, r["gamma_scale"]),
    }


def fig02b_record_pdf(records: pd.DataFrame, fits: pd.DataFrame, path: Path, q_max: float = 0.99) -> Path:
    """Fig. 2 (per-record variant matching the caption and Table 3): PDF with three fitted laws."""

    _style()
    slices = _present(records)
    fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 4))
    for ax, s in zip(np.atleast_1d(axes), slices):
        x = records.loc[records["slice_id"] == s, "total_mb"].to_numpy(dtype=float)
        x = x[x > 0] / 1e3  # KB
        hi = float(np.quantile(x, q_max))
        ax.hist(x[x <= hi], bins=80, density=True, color=HIST_COLORS[s], edgecolor="none", label="Empirical")
        xs = np.linspace(hi / 400, hi, 400)
        for (name, d), ls in zip(_fitted(fits, s).items(), ("k:", "k-", "k--")):
            ax.plot(xs, d.pdf(xs * 1e3) * 1e3, ls, lw=1.2, label=name)
        ax.set_title(f"{s} — per-record PDF")
        ax.set_xlabel(f"Traffic per record (KB), ≤ {int(q_max * 100)}th percentile")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle("PDF of Per-Record Traffic Volume by Slice", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def fig03b_record_cdf(records: pd.DataFrame, fits: pd.DataFrame, path: Path) -> Path:
    """Fig. 3 (per-record variant): empirical CDF (log x-axis) with three fitted laws."""

    _style()
    slices = _present(records)
    fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 4))
    for ax, s in zip(np.atleast_1d(axes), slices):
        x = np.sort(records.loc[records["slice_id"] == s, "total_mb"].to_numpy(dtype=float))
        x = x[x > 0]
        step = max(1, x.size // 2000)
        ax.plot(x[::step] / 1e3, (np.arange(1, x.size + 1) / x.size)[::step],
                color=SLICE_COLORS[s], lw=2, label="Empirical CDF")
        xs = np.geomspace(x[0], x[-1], 400)
        for (name, d), ls in zip(_fitted(fits, s).items(), ("k:", "k-", "k--")):
            ax.plot(xs / 1e3, d.cdf(xs), ls, lw=1.2, label=name)
        ax.set_xscale("log")
        ax.set_title(f"{s} — per-record CDF")
        ax.set_xlabel("Traffic per record (KB, log scale)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("CDF of Per-Record Traffic Volume by Slice", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 4


PIPELINE_LABELS = {
    "en": ("Hourly series\nY_t per slice", "Train/Test\nsplit", "Normalisation\nfor LSTM/MLP",
           "SARIMA\n(1,1,1)(1,0,1,24)", "24-hour window\nLSTM/MLP", "Forecast\nŶ_(t+1)",
           "Evaluation\nRMSE, MAE", "Forecasting Pipeline Architecture"),
    "ru": ("Почасовой ряд\nY_t по слайсу", "Train/Test\nразделение", "Нормализация\nдля LSTM/MLP",
           "SARIMA\n(1,1,1)(1,0,1,24)", "Окно 24 часа\nLSTM/MLP", "Прогноз\nŶ_(t+1)",
           "Оценка\nRMSE, MAE", "Схема конвейера прогнозирования трафика"),
}


def fig04_pipeline(path: Path, lang: str = "en", sarima_label: str | None = None) -> Path:
    """Fig. 4 — forecasting pipeline block diagram (``lang`` = 'en' or 'ru')."""

    _style()
    lab = list(PIPELINE_LABELS[lang])
    if sarima_label:
        lab[3] = sarima_label
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    w, h = 1.55, 0.8
    pos = {0: (0.1, 1.4), 1: (1.95, 1.4), 2: (3.8, 1.4), 3: (5.9, 2.5), 4: (5.9, 0.3),
           5: (8.0, 1.4), 6: (9.85, 1.4)}
    for i, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="#E8EEF7", ec="#3A4A5E", lw=1.2))
        ax.text(x + w / 2, y + h / 2, lab[i], ha="center", va="center", fontsize=9)

    def arrow(a: int, b: int) -> None:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((xa + w, ya + h / 2), (xb, yb + h / 2),
                                     arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#3A4A5E"))

    for a, b in ((0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)):
        arrow(a, b)
    ax.set_title(lab[7], fontweight="bold")
    return _save(fig, path)


def fig_lstm_architecture(path: Path, lang: str = "en") -> Path:
    """Input window and LSTM / MLP architecture (thesis Fig. 2.10), from the model's constants."""

    from src.forecasting.neural_forecast import HIDDEN, MLP_HIDDEN, WINDOW

    _style()
    w0 = WINDOW - 1
    if lang == "ru":
        lab = (f"Входное окно\n[Y_(t-{w0}), …, Y_t]\n{WINDOW} часов", f"LSTM вход\nshape = ({WINDOW}, 1)",
               f"LSTM\n{HIDDEN} скрытых нейронов", "Dense\n1 выход", f"MLP вход\nвектор = {WINDOW}",
               f"MLP\nскрытые слои {MLP_HIDDEN}", "Linear\n1 выход", "Прогноз\nŶ_(t+1)",
               "Входное окно и архитектура LSTM/MLP для одношагового прогноза")
    else:
        lab = (f"Input window\n[Y_(t-{w0}), …, Y_t]\n{WINDOW} hours", f"LSTM input\nshape = ({WINDOW}, 1)",
               f"LSTM\n{HIDDEN} hidden units", "Dense\n1 output", f"MLP input\nvector = {WINDOW}",
               f"MLP\nhidden layers {MLP_HIDDEN}", "Linear\n1 output", "Forecast\nŶ_(t+1)",
               "Input window and LSTM/MLP architecture for one-step-ahead forecasting")
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    w, h = 2.1, 0.9
    pos = {0: (0.05, 1.65), 1: (2.6, 2.85), 2: (5.1, 2.85), 3: (7.6, 2.85), 4: (2.6, 0.45),
           5: (5.1, 0.45), 6: (7.6, 0.45), 7: (10.8, 1.65)}
    for i, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="#E8EEF7", ec="#3A4A5E", lw=1.2))
        ax.text(x + w / 2, y + h / 2, lab[i], ha="center", va="center", fontsize=9)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 7)):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((xa + w, ya + h / 2), (xb, yb + h / 2),
                                     arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#3A4A5E"))
    ax.set_title(lab[8], fontweight="bold")
    return _save(fig, path)


def fig_rmse_mae_bars(comparison: pd.DataFrame, path: Path) -> Path:
    """RMSE and MAE per slice, SARIMA vs LSTM/MLP, in GB (thesis Fig. 2.18).

    ``comparison``: ``forecast_model_comparison.csv`` (columns slice_id, model, rmse, mae; bytes).
    """

    _style()
    df = comparison.copy()
    df["model"] = df["model"].replace({"sarima": "SARIMA", "lstm": "LSTM/MLP", "mlp": "LSTM/MLP"})
    slices = [s for s in ("URLLC", "eMBB", "mMTC") if s in set(df["slice_id"])]
    x = np.arange(len(slices))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (col, name) in zip(axes, (("rmse", "RMSE"), ("mae", "MAE"))):
        pv = df.pivot_table(index="slice_id", columns="model", values=col, aggfunc="first").reindex(slices)
        for off, (m, c) in zip((-0.18, 0.18), (("SARIMA", "#1976D2"), ("LSTM/MLP", "#E64A19"))):
            ax.bar(x + off, pv[m] / BYTES_PER_GB, width=0.36, color=c, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(slices)
        ax.set_ylabel(f"{name} (GB)")
        ax.set_title(f"Forecast {name} per Slice")
        ax.legend()
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 5


def fig05_service_rate(p: LeontyevaParams, path: Path) -> Path:
    """Fig. 5 — stacked bars of μ(n) = min(n,C)·μ_LEO + max(0,n−C)·μ_MEO for n = 0..N."""

    _style()
    n = np.arange(p.N + 1)
    leo = np.minimum(n, p.C_LEO) * p.mu_LEO
    meo = np.maximum(0, n - p.C_LEO) * p.mu_MEO
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(n, leo, color=LEO_COLOR, edgecolor="white", label=f"LEO: min(n,{p.C_LEO})·μ_LEO")
    ax.bar(n, meo, bottom=leo, color=MEO_COLOR, edgecolor="white", label=f"MEO: max(0,n−{p.C_LEO})·μ_MEO")
    for k in n:
        tot = mu_n(int(k), p)
        ax.text(k, tot + 0.15, f"{tot:g}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(n)
    ax.set_xlabel("n (active sessions)")
    ax.set_ylabel("μ(n)")
    ax.set_title("Service Rate μ(n) — LEO and MEO Contributions")
    ax.set_ylim(0, (leo + meo).max() * 1.1 + 0.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 6


def fig06_state_diagram(path: Path, N: int = 3, M: int = 2) -> Path:
    """Fig. 6 — state-transition diagram of the (n, m) CTMC for small N, M.

    Columns are n = 0..N, rows m = 0..M. Arrows: λ (n→n+1), μ(n) (n→n−1), m·γ retrial
    ((n,m)→(n+1,m−1), dashed), and λ into the retrial orbit at n = N ((N,m)→(N,m+1)).
    """

    _style()
    dx, dy, r = 2.2, 1.9, 0.48
    fig, ax = plt.subplots(figsize=(2.3 * (N + 1) + 1, 1.9 * (M + 1) + 0.6))
    ax.set_aspect("equal")
    ax.axis("off")
    xy = {(n, m): (n * dx, -m * dy) for n in range(N + 1) for m in range(M + 1)}

    def arr(p0, p1, color, ls="-", rad=0.0, lw=1.6):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, lw=lw,
                                     color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=0, shrinkB=0))

    for (n, m), (x, y) in xy.items():
        if n < N:  # arrival accepted, and service completion back
            arr((x + r, y + 0.1), (x + dx - r, y + 0.1), "black")
            ax.text(x + dx / 2, y + 0.32, "λ", color="#1F4E9A", ha="center", fontsize=13)
            arr((x + dx - r, y - 0.1), (x + r, y - 0.1), "#2E7D32")
            ax.text(x + dx / 2, y - 0.45, "μ(n)", color="#2E7D32", ha="center", fontsize=12)
        if n == N and m < M:  # blocked arrival joins retrial orbit
            arr((x, y - r), (x, y - dy + r), "black")
            ax.text(x + 0.15, y - dy / 2, "λ", color="#C62828", fontsize=13, va="center")
        if m > 0 and n < N:  # retrial success
            x1, y1 = xy[(n + 1, m - 1)]
            v = np.array([x1 - x, y1 - y])
            v /= np.linalg.norm(v)
            arr((x + v[0] * r, y + v[1] * r), (x1 - v[0] * r, y1 - v[1] * r), "#DAA520", ls="--", lw=1.3)
            ax.text(x + 0.42 * (x1 - x) + 0.1, y + 0.42 * (y1 - y), "m·γ", color="#DAA520", fontsize=11)
    for (n, m), (x, y) in xy.items():
        ax.add_patch(Circle((x, y), r, fc="#CFE8FC", ec="#1F4E9A", lw=1.8, zorder=3))
        ax.text(x, y, f"({n},{m})", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    ax.set_xlim(-0.8, N * dx + 0.8)
    ax.set_ylim(-M * dy - 0.8, 0.9)
    ax.set_title(f"State Transition Diagram for CTMC (simplified N={N}, M={M})")
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 7 – 10


def test_predictions(sarima: pd.DataFrame, lstm: pd.DataFrame, slice_id: str) -> pd.DataFrame:
    """Merge test-split predictions: columns ts, actual, sarima, lstm (bytes)."""

    def pick(df: pd.DataFrame, name: str) -> pd.DataFrame:
        d = df[(df["slice_id"] == slice_id) & (df["split"] == "test")]
        return d[["ts", "y_actual", "y_predicted"]].rename(columns={"y_actual": "actual", "y_predicted": name})

    out = pick(sarima, "sarima").merge(pick(lstm, "lstm").drop(columns="actual"), on="ts")
    out["ts"] = pd.to_datetime(out["ts"])
    return out.sort_values("ts").reset_index(drop=True)


def missing_mask(pred: pd.DataFrame) -> np.ndarray:
    """Hours without data: ``NaN`` (current pipeline) or exactly 0 (older zero-filled outputs)."""

    a = pred["actual"]
    return (a.isna() | (a == 0)).to_numpy()


def fig07_forecast_vs_actual(pred: pd.DataFrame, slice_id: str, path: Path, mark_missing: bool = True) -> Path:
    """Fig. 7 — actual vs SARIMA vs LSTM/MLP on the test horizon (GB).

    With ``mark_missing`` hours without data are shaded (see :func:`missing_mask`).
    """

    _style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = pred["ts"]
    miss = missing_mask(pred)
    if mark_missing and miss.any():
        ax.fill_between(t, 0, 1, where=miss, transform=ax.get_xaxis_transform(),
                        color="grey", alpha=0.15, step="mid", label="no data (zero-filled)")
    ax.plot(t, pred["actual"] / BYTES_PER_GB, color="black", lw=1.6, label="Actual")
    ax.plot(t, pred["sarima"] / BYTES_PER_GB, color="tab:blue", lw=1.2, label="SARIMA")
    ax.plot(t, pred["lstm"] / BYTES_PER_GB, color="tab:orange", lw=1.2, label="LSTM/MLP")
    ax.set_title(f"Forecast vs actual — slice {slice_id} (test)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic (GB)")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, path)


def fig08_10_error_hist(pred: pd.DataFrame, slice_id: str, path: Path, drop_missing: bool = False,
                        bins: int = 25) -> Path:
    """Fig. 8–10 — histograms of forecast error (predicted − actual, GB), SARIMA vs LSTM/MLP.

    ``NaN`` hours never enter the histogram; ``drop_missing`` also drops zero-filled hours of
    older outputs.
    """

    _style()
    d = pred[~missing_mask(pred)] if drop_missing else pred[pred["actual"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (col, name, color) in zip(axes, (("sarima", "SARIMA", "#1E88E5"), ("lstm", "LSTM/MLP", "#F4511E"))):
        err = (d[col] - d["actual"]) / BYTES_PER_GB
        ax.hist(err, bins=bins, color=color, edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.set_title(f"{name} Error — {slice_id}")
        ax.set_xlabel("Error (GB)")
        ax.set_ylabel("Frequency")
    if drop_missing:
        fig.suptitle(f"{slice_id}: {len(pred) - len(d)} zero-filled hours excluded", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 11


def sensitivity_table(
    lambdas: Iterable[float],
    configs: Sequence[tuple[int, int]] = FIG11_CONFIGS,
    base: LeontyevaParams | None = None,
) -> pd.DataFrame:
    """P_loss, P(n=N), util_LEO, util_MEO over λ for each (N, C_LEO)."""

    base = base or default_params()
    rows = []
    for N, C in configs:
        for lam in lambdas:
            p = replace(base, N=N, C_LEO=C, lam=float(lam))
            met = compute_metrics(stationary_distribution(build_generator_matrix(p)), p)
            rows.append({"N": N, "C_LEO": C, "lambda": float(lam), **met})
    return pd.DataFrame(rows)


def fig11_sensitivity(table: pd.DataFrame, path: Path) -> Path:
    """Fig. 11 — P_loss (log) and util_LEO vs λ. Values below the solver's round-off are hidden."""

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (N, C), g in table.groupby(["N", "C_LEO"], sort=True):
        ploss = g["P_loss"].where(g["P_loss"] > NUMERICAL_FLOOR)
        axes[0].plot(g["lambda"], ploss, marker=".", ms=4, label=f"N={N}")
        axes[1].plot(g["lambda"], g["util_LEO"], marker=".", ms=4, label=f"N={N}, C={C}")
    axes[0].set_yscale("log")
    axes[0].set_title("Loss Probability vs λ for Different N")
    axes[0].set_ylabel("P_loss")
    axes[1].set_title("LEO Utilization vs λ for Different N")
    axes[1].set_ylabel("LEO Utilization")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("λ (arrival rate)")
        ax.legend()
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 12


def quasi_stationary_table(
    daily_profile: pd.DataFrame,
    base: LeontyevaParams | None = None,
    target_mean_lambda: float = 3.0,
    alpha: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """λ(h) = α · traffic(h) / 3600 per hour of day, and the stationary metrics for each hour.

    ``traffic(h)`` is the mean hourly volume summed over slices (bytes). If ``alpha`` is not
    given it is chosen so that the daily mean of λ(h) equals ``target_mean_lambda``
    (3.0 reproduces the article's Fig. 12: λ ≈ 1.8 … 3.6 req/s).
    """

    base = base or default_params()
    tot = daily_profile.groupby("hour")["mean_total"].sum().sort_index()
    if alpha is None:
        alpha = target_mean_lambda * 3600.0 / float(tot.mean())
    rows = []
    for h, v in tot.items():
        lam = alpha * float(v) / 3600.0
        p = replace(base, lam=lam)
        met = compute_metrics(stationary_distribution(build_generator_matrix(p)), p)
        rows.append({"hour": int(h), "traffic_bytes": float(v), "lambda": lam, **met})
    return pd.DataFrame(rows), float(alpha)


def fig12_quasi_stationary(table: pd.DataFrame, path: Path) -> Path:
    """Fig. 12 — three stacked panels: λ(h); util_LEO / util_MEO; P_loss and P_block = P(n=N)."""

    _style()
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    h = table["hour"]
    axes[0].plot(h, table["lambda"], "o-", color="#1F5FA6", ms=4)
    axes[0].set_title("Calibrated Inflow Intensity")
    axes[0].set_ylabel("lambda_h, req/s")
    axes[1].plot(h, table["util_LEO"], "o-", color="#2E8B57", ms=4, label="util_LEO")
    axes[1].plot(h, table["util_MEO"], "s-", color="#E8732A", ms=4, label="util_MEO")
    axes[1].set_title("Orbital Resources Utilization")
    axes[1].set_ylabel("Utilization")
    axes[1].legend(loc="center right")
    axes[2].semilogy(h, table["P_loss"], "o-", color="#C62828", ms=4, label="P_loss")
    axes[2].semilogy(h, table["P(n=N)"], "s-", color="#7E57C2", ms=4, label="P_block = P(n = N)")
    axes[2].set_title("Blocking and Loss Probabilities")
    axes[2].set_ylabel("Probability")
    axes[2].set_xlabel("Hour of Day")
    axes[2].legend(loc="lower right")
    axes[2].set_xticks(range(0, 24, 2))
    fig.suptitle("Quasi-Stationary Hourly Assessment of LEO/MEO Resources", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
