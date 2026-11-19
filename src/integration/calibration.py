# -*- coding: utf-8 -*-
"""
Числовая калибровка параметров Leontyeva CTMC без изменения структуры Q, μ(n) и пространства состояний.

Меняются только: alpha, N, M, C_LEO, mu_LEO, mu_MEO, gamma — и масштаб λ через alpha
(λ_t = alpha * total_mb / 3600 на реальных данных).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.integration.lambda_from_traffic import derive_lambda_scenarios
from src.markov.ctmc import build_generator_matrix, stationary_distribution
from src.markov.leontyeva import LeontyevaParams
from src.markov.stationary import compute_metrics

logger = logging.getLogger(__name__)

# Сетка alpha (логарифмическая по порядку величины)
GRID_ALPHA: list[float] = [
    1e-9,
    3e-9,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
]

GRID_N = [5, 10, 20, 40, 80]
GRID_M = [4, 10, 20, 40]
GRID_GAMMA = [0.5, 1.0, 2.0, 5.0, 10.0]
GRID_MU_LEO = [3, 5, 10, 20, 50, 100]
GRID_MU_MEO = [2, 4, 8, 15, 30, 60]

SATURATION_PN = 0.99
TRIVIAL_PN = 1e-5
TRIVIAL_UTIL = 0.05


def _cleaned_path(project_root: Path) -> Path:
    proc = Path(project_root) / "data" / "processed"
    pq = proc / "cleaned_traffic.parquet"
    csv = proc / "cleaned_traffic.csv"
    if pq.is_file():
        return pq
    if csv.is_file():
        return csv
    raise FileNotFoundError(f"Нет cleaned_traffic.parquet или cleaned_traffic.csv в {proc}")


def _lambda_at_alpha(lam_at_alpha1: float, alpha: float) -> float:
    """λ(mean или peak) масштабируется линейно с alpha относительно alpha=1."""
    return float(alpha) * float(lam_at_alpha1)


def _candidate_c_leo(n: int) -> list[int]:
    opts = {min(2, n), max(1, n // 4), max(1, n // 2), n}
    return sorted({x for x in opts if 1 <= x <= n})


def _mu_pairs() -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in product(GRID_MU_LEO, GRID_MU_MEO):
        if a >= b:
            out.append((float(a), float(b)))
    return out


def solve_metrics(p: LeontyevaParams) -> dict[str, Any]:
    """Точное решение πQ=0 и метрики; при ошибке — поля error."""
    try:
        Q = build_generator_matrix(p)
        pi = stationary_distribution(Q)
        m = compute_metrics(pi, p)
        out = {k: float(v) for k, v in m.items()}
        out["error"] = ""
        return out
    except Exception as e:  # noqa: BLE001
        return {
            "P_loss": float("nan"),
            "P(n=N)": float("nan"),
            "P(m=M)": float("nan"),
            "E[n]": float("nan"),
            "E[m]": float("nan"),
            "util_LEO": float("nan"),
            "util_MEO": float("nan"),
            "error": str(e),
        }


def _flags(pn: float, util_leo: float) -> tuple[bool, bool]:
    sat = np.isfinite(pn) and pn >= SATURATION_PN
    under = (
        np.isfinite(pn)
        and np.isfinite(util_leo)
        and pn <= TRIVIAL_PN
        and util_leo <= TRIVIAL_UTIL
    )
    return sat, under


def run_stage_alpha_only(
    lam_mean_1: float,
    lam_peak_1: float,
    base: LeontyevaParams,
) -> pd.DataFrame:
    """Этап 1: только сетка alpha при фиксированных N,M,... из base (через replace lam)."""
    rows: list[dict[str, Any]] = []
    for alpha in GRID_ALPHA:
        for scenario, lam1 in (("mean_load", lam_mean_1), ("peak_load", lam_peak_1)):
            lam = _lambda_at_alpha(lam1, alpha)
            p = replace(base, lam=lam)
            met = solve_metrics(p)
            pn = met["P(n=N)"]
            ul = met["util_LEO"]
            sat, und = _flags(pn, ul)
            rows.append(
                {
                    "stage": "alpha_sweep_default_params",
                    "alpha": alpha,
                    "N": p.N,
                    "M": p.M,
                    "C_LEO": p.C_LEO,
                    "mu_LEO": p.mu_LEO,
                    "mu_MEO": p.mu_MEO,
                    "gamma": p.gamma,
                    "scenario": scenario,
                    "lambda": lam,
                    "lambda_mean_implied": _lambda_at_alpha(lam_mean_1, alpha),
                    "lambda_peak_implied": _lambda_at_alpha(lam_peak_1, alpha),
                    "P_loss": met.get("P_loss", float("nan")),
                    "P(n=N)": met["P(n=N)"],
                    "P(m=M)": met["P(m=M)"],
                    "E[n]": met["E[n]"],
                    "E[m]": met["E[m]"],
                    "util_LEO": met["util_LEO"],
                    "util_MEO": met["util_MEO"],
                    "saturation_flag": sat,
                    "trivial_underload_flag": und,
                    "error": met.get("error", ""),
                }
            )
    return pd.DataFrame(rows)


def run_stage_full_grid(
    lam_mean_1: float,
    lam_peak_1: float,
    alphas: list[float],
) -> pd.DataFrame:
    """Этап 2: сетка ёмкости/обслуживания при подмножестве alpha (сжатая, ~8k строк)."""
    # Усечённая сетка, чтобы уложиться в разумное время; полный декартов продукт слишком велик.
    grid_n = [5, 20, 40, 80]
    grid_m = [4, 20, 40]
    grid_gamma = [0.5, 1.0, 5.0]
    mu_pairs_subset = [
        (3.0, 2.0),
        (5.0, 2.0),
        (10.0, 4.0),
        (10.0, 8.0),
        (20.0, 8.0),
        (20.0, 15.0),
        (50.0, 30.0),
        (100.0, 60.0),
    ]
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for N in grid_n:
            for M in grid_m:
                for C_LEO in sorted({min(2, N), max(1, N // 2)}):
                    if C_LEO > N:
                        continue
                    for mu_LEO, mu_MEO in mu_pairs_subset:
                        for gamma in grid_gamma:
                            for scenario, lam1 in (
                                ("mean_load", lam_mean_1),
                                ("peak_load", lam_peak_1),
                            ):
                                lam = _lambda_at_alpha(lam1, alpha)
                                p = LeontyevaParams(
                                    N=N,
                                    M=M,
                                    lam=lam,
                                    gamma=gamma,
                                    C_LEO=C_LEO,
                                    mu_LEO=mu_LEO,
                                    mu_MEO=mu_MEO,
                                )
                                met = solve_metrics(p)
                                pn = met["P(n=N)"]
                                ul = met["util_LEO"]
                                sat, und = _flags(pn, ul)
                                rows.append(
                                    {
                                        "stage": "full_grid",
                                        "alpha": alpha,
                                        "N": N,
                                        "M": M,
                                        "C_LEO": C_LEO,
                                        "mu_LEO": mu_LEO,
                                        "mu_MEO": mu_MEO,
                                        "gamma": gamma,
                                        "scenario": scenario,
                                        "lambda": lam,
                                        "lambda_mean_implied": _lambda_at_alpha(lam_mean_1, alpha),
                                        "lambda_peak_implied": _lambda_at_alpha(lam_peak_1, alpha),
                                        "P_loss": met.get("P_loss", float("nan")),
                                        "P(n=N)": met["P(n=N)"],
                                        "P(m=M)": met["P(m=M)"],
                                        "E[n]": met["E[n]"],
                                        "E[m]": met["E[m]"],
                                        "util_LEO": met["util_LEO"],
                                        "util_MEO": met["util_MEO"],
                                        "saturation_flag": sat,
                                        "trivial_underload_flag": und,
                                        "error": met.get("error", ""),
                                    }
                                )
    return pd.DataFrame(rows)


def _pick_recommended(df: pd.DataFrame) -> pd.DataFrame:
    """Отбор recommended_low / medium / high по сценарию mean_load и не тривиальных строк."""
    err_ok = df["error"].fillna("").astype(str) == ""
    sub = df[(df["scenario"] == "mean_load") & err_ok].copy()
    if sub.empty:
        return pd.DataFrame()

    def ok_row(r: pd.Series) -> bool:
        return (
            np.isfinite(r["P(n=N)"])
            and not r["saturation_flag"]
            and not r["trivial_underload_flag"]
        )

    usable = sub[sub.apply(ok_row, axis=1)]
    if usable.empty:
        usable = sub[np.isfinite(sub["P(n=N)"])].copy()

    picks: list[dict[str, Any]] = []

    # low: минимальный P(n=N) среди «умеренной» утилизации
    low_cand = usable[(usable["P(n=N)"] < 0.08) & (usable["util_LEO"] >= 0.08)]
    if not low_cand.empty:
        j = low_cand["P(n=N)"].idxmin()
        r = usable.loc[j].to_dict()
        r["recommendation"] = "recommended_low"
        picks.append(r)
    elif not usable.empty:
        j = usable["P(n=N)"].idxmin()
        r = usable.loc[j].to_dict()
        r["recommendation"] = "recommended_low"
        picks.append(r)

    # medium: близко к P(n=N) ~ 0.05–0.2
    med = usable[(usable["P(n=N)"] >= 0.01) & (usable["P(n=N)"] <= 0.25)]
    if not med.empty:
        med = med.assign(_score=(med["P(n=N)"] - 0.08).abs())
        j = med["_score"].idxmin()
        r = med.drop(columns=["_score"]).loc[j].to_dict()
        r["recommendation"] = "recommended_medium"
        picks.append(r)

    # high: высокая загрузка, но не тривиальное насыщение (оставляем запас до 1)
    hi = usable[
        (usable["P(n=N)"] >= 0.25)
        & (usable["P(n=N)"] <= 0.92)
        & (usable["util_LEO"] >= 0.35)
    ]
    if not hi.empty:
        j = hi["P(n=N)"].idxmax()
        r = hi.loc[j].to_dict()
        r["recommendation"] = "recommended_high"
        picks.append(r)
    else:
        hi2 = usable[(usable["P(n=N)"] >= 0.2) & (usable["P(n=N)"] < SATURATION_PN)]
        if not hi2.empty:
            j = hi2["P(n=N)"].idxmax()
            r = hi2.loc[j].to_dict()
            r["recommendation"] = "recommended_high"
            picks.append(r)

    if not picks:
        return pd.DataFrame()

    return pd.DataFrame(picks)


def _plot_blocking_vs_alpha(stage1: pd.DataFrame, out_dir: Path) -> Path:
    s = stage1[stage1["scenario"] == "mean_load"].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.semilogx(s["alpha"], s["P(n=N)"], "o-", label="mean_load P(n=N)")
    sp = stage1[stage1["scenario"] == "peak_load"].sort_values("alpha")
    ax.semilogx(sp["alpha"], sp["P(n=N)"], "s-", label="peak_load P(n=N)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("P(n=N)")
    ax.set_title("Blocking vs alpha (default N,M,C_LEO,mu,gamma)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "blocking_vs_alpha.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _plot_util_vs_alpha(stage1: pd.DataFrame, out_dir: Path) -> Path:
    s = stage1[stage1["scenario"] == "mean_load"].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.semilogx(s["alpha"], s["util_LEO"], "o-", label="util_LEO (mean_load)")
    ax.semilogx(s["alpha"], s["util_MEO"], "^-", label="util_MEO (mean_load)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Utilization")
    ax.set_title("Utilization vs alpha (default params)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "utilization_vs_alpha.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _plot_tradeoff(full: pd.DataFrame, out_dir: Path) -> Path:
    sub = full[(full["scenario"] == "mean_load") & (full["error"] == "")]
    if sub.empty:
        sub = full[full["scenario"] == "mean_load"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sc = ax.scatter(
        sub["util_LEO"],
        sub["P(n=N)"],
        c=np.log10(sub["alpha"].astype(float)),
        cmap="viridis",
        alpha=0.5,
        s=12,
    )
    plt.colorbar(sc, ax=ax, label="log10(alpha)")
    ax.set_xlabel("util_LEO")
    ax.set_ylabel("P(n=N)")
    ax.set_title("Calibration tradeoff (mean_load)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "calibration_tradeoff.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def write_calibration_summary(
    out_path: Path,
    *,
    cleaned_path: Path,
    lam_raw_mean: float,
    lam_raw_peak: float,
    alpha_baseline: float,
    base_params: LeontyevaParams,
    stage1: pd.DataFrame,
    recommended: pd.DataFrame,
) -> None:
    """Текстовый отчёт: сырой масштаб vs калибровка, без изменения модели."""
    lines = [
        "CALIBRATION SUMMARY — Leontyeva LEO/MEO CTMC",
        "=============================================",
        "",
        "1) Математическая модель не менялась: то же пространство состояний (n,m),",
        "   те же переходы Q, то же mu(n)=min(n,C_LEO)*mu_LEO+max(0,n-C_LEO)*mu_MEO,",
        "   постоянный lambda внутри каждого сценария (нет Q(t)).",
        "",
        "2) Менялись только ЧИСЛОВЫЕ параметры: alpha, N, M, C_LEO, mu_LEO, mu_MEO, gamma.",
        "",
        "3) Почему «сырой» результат при alpha=1 выглядел насыщенным:",
        f"   На реальных данных при alpha={alpha_baseline} получены lambda_mean={lam_raw_mean:.6g},",
        f"   lambda_peak={lam_raw_peak:.6g} (агрегат почасовых total_mb).",
        "   При малых mu и N эти величины несовместимы по масштабу с выбранной интенсивностью",
        "   обслуживания в CTMC: отношение lambda к эффективной мощности обслуживания огромно,",
        "   поэтому P(n=N) -> 1. Это не ошибка Q, а несогласованность единиц/масштаба нагрузки и mu.",
        "",
        "4) Как alpha меняет lambda:",
        "   lambda_t = alpha * total_mb/3600; среднее и пик по часам масштабируются линейно с alpha",
        "   относительно того же ряда total_mb.",
        "",
        f"5) Базовые параметры для сравнения (до калибровки сетки): N={base_params.N}, M={base_params.M},",
        f"   C_LEO={base_params.C_LEO}, mu_LEO={base_params.mu_LEO}, mu_MEO={base_params.mu_MEO}, gamma={base_params.gamma}.",
        "",
        "6) Рекомендованные сценарии (mean_load, не тривиальные): см. calibrated_scenarios.csv.",
        "   Они подобраны так, чтобы blocking и утилизация были интерпретируемыми (не 0 и не 1).",
        "",
        "7) Исходные данные:",
        f"   cleaned_traffic: {cleaned_path.resolve()}",
        "",
        "Alpha sweep (default params) — диапазон P(n=N) по alpha:",
    ]
    s1m = stage1[stage1["scenario"] == "mean_load"]
    if not s1m.empty:
        lines.append(
            f"   min P(n=N)={s1m['P(n=N)'].min():.6g}, max P(n=N)={s1m['P(n=N)'].max():.6g}"
        )
    lines.extend(
        [
            "",
            "Файлы:",
            "  - calibration_grid_results.csv — полная сетка",
            "  - calibrated_scenarios.csv — recommended_low/medium/high",
            "  - blocking_vs_alpha.png, utilization_vs_alpha.png, calibration_tradeoff.png",
            "",
        ]
    )
    if not recommended.empty:
        lines.append("Рекомендации (кратко):")
        for _, r in recommended.iterrows():
            lines.append(
                f"  {r.get('recommendation', '?')}: alpha={r['alpha']}, N={r['N']}, M={r['M']}, "
                f"C_LEO={r['C_LEO']}, mu_LEO={r['mu_LEO']}, mu_MEO={r['mu_MEO']}, gamma={r['gamma']}, "
                f"P(n=N)={r['P(n=N)']:.4f}, util_LEO={r['util_LEO']:.4f}"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_calibration(project_root: Path | str) -> dict[str, Any]:
    """
    Полный цикл калибровки: этап alpha, этап полной сетки, отбор сценариев, графики, summary.

    Сохраняет outputs/integration/calibration_grid_results.csv, calibrated_scenarios.csv,
    calibration_summary.txt и три PNG.
    """
    project_root = Path(project_root).resolve()
    out_dir = project_root / "outputs" / "integration"
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned = _cleaned_path(project_root)
    lam_mean_1, lam_peak_1, hourly = derive_lambda_scenarios(cleaned, alpha=1.0)
    alpha_baseline = 1.0

    # «Сырой» прогон при alpha=1 и параметрах по умолчанию (для отчёта)
    from src.markov.leontyeva import default_params

    base = default_params()
    p_raw_mean = replace(base, lam=_lambda_at_alpha(lam_mean_1, alpha_baseline))
    p_raw_peak = replace(base, lam=_lambda_at_alpha(lam_peak_1, alpha_baseline))
    raw_mean_met = solve_metrics(p_raw_mean)
    raw_peak_met = solve_metrics(p_raw_peak)

    stage1 = run_stage_alpha_only(lam_mean_1, lam_peak_1, base)
    # Подмножество alpha для этапа 2: не перегружать; берём 7 значений по диапазону
    stage2_alphas = [1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]
    stage2 = run_stage_full_grid(lam_mean_1, lam_peak_1, stage2_alphas)

    grid_all = pd.concat([stage1, stage2], ignore_index=True)
    # Сырой эмпирический результат (alpha=1, default params) — отдельные строки
    raw_rows = pd.DataFrame(
        [
            {
                "stage": "raw_empirical_alpha_1_default_params",
                "alpha": alpha_baseline,
                "N": base.N,
                "M": base.M,
                "C_LEO": base.C_LEO,
                "mu_LEO": base.mu_LEO,
                "mu_MEO": base.mu_MEO,
                "gamma": base.gamma,
                "scenario": "mean_load",
                "lambda": p_raw_mean.lam,
                "lambda_mean_implied": lam_mean_1,
                "lambda_peak_implied": lam_peak_1,
                **{k: raw_mean_met[k] for k in ("P_loss", "P(n=N)", "P(m=M)", "E[n]", "E[m]", "util_LEO", "util_MEO")},
                "saturation_flag": _flags(raw_mean_met["P(n=N)"], raw_mean_met["util_LEO"])[0],
                "trivial_underload_flag": _flags(raw_mean_met["P(n=N)"], raw_mean_met["util_LEO"])[1],
                "error": raw_mean_met.get("error", ""),
            },
            {
                "stage": "raw_empirical_alpha_1_default_params",
                "alpha": alpha_baseline,
                "N": base.N,
                "M": base.M,
                "C_LEO": base.C_LEO,
                "mu_LEO": base.mu_LEO,
                "mu_MEO": base.mu_MEO,
                "gamma": base.gamma,
                "scenario": "peak_load",
                "lambda": p_raw_peak.lam,
                "lambda_mean_implied": lam_mean_1,
                "lambda_peak_implied": lam_peak_1,
                **{k: raw_peak_met[k] for k in ("P_loss", "P(n=N)", "P(m=M)", "E[n]", "E[m]", "util_LEO", "util_MEO")},
                "saturation_flag": _flags(raw_peak_met["P(n=N)"], raw_peak_met["util_LEO"])[0],
                "trivial_underload_flag": _flags(raw_peak_met["P(n=N)"], raw_peak_met["util_LEO"])[1],
                "error": raw_peak_met.get("error", ""),
            },
        ]
    )

    grid_all = pd.concat([raw_rows, grid_all], ignore_index=True)

    p_grid = out_dir / "calibration_grid_results.csv"
    grid_all.to_csv(p_grid, index=False)

    recommended = _pick_recommended(grid_all[grid_all["stage"] != "raw_empirical_alpha_1_default_params"])
    if recommended.empty:
        recommended = _pick_recommended(grid_all)
    p_rec = out_dir / "calibrated_scenarios.csv"
    recommended.to_csv(p_rec, index=False)

    pb = _plot_blocking_vs_alpha(stage1, out_dir)
    pu = _plot_util_vs_alpha(stage1, out_dir)
    pt = _plot_tradeoff(grid_all[grid_all["stage"] != "raw_empirical_alpha_1_default_params"], out_dir)

    p_sum = out_dir / "calibration_summary.txt"
    write_calibration_summary(
        p_sum,
        cleaned_path=cleaned,
        lam_raw_mean=lam_mean_1,
        lam_raw_peak=lam_peak_1,
        alpha_baseline=alpha_baseline,
        base_params=base,
        stage1=stage1,
        recommended=recommended,
    )

    logger.info("Calibration wrote %s, %s, %s", p_grid, p_rec, p_sum)

    return {
        "calibration_grid_results": p_grid,
        "calibrated_scenarios": p_rec,
        "calibration_summary": p_sum,
        "blocking_vs_alpha": pb,
        "utilization_vs_alpha": pu,
        "calibration_tradeoff": pt,
        "recommended": recommended,
        "n_hourly": len(hourly),
        "lam_mean_at_alpha1": lam_mean_1,
        "lam_peak_at_alpha1": lam_peak_1,
    }
