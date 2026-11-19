# -*- coding: utf-8 -*-
"""Alpha calibration from pre-aggregated hourly traffic table."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.integration.calibration import solve_metrics
from src.markov.leontyeva import default_params


def _hourly_total_from_aggregated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"hour_ts", "slice_id", "total_mb"}
    miss = required.difference(df.columns)
    if miss:
        raise ValueError(f"aggregated lambda missing columns: {sorted(miss)}")
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], errors="coerce")
    df["total_mb"] = pd.to_numeric(df["total_mb"], errors="coerce")
    df = df.dropna(subset=["hour_ts", "total_mb"])
    hourly = df.groupby("hour_ts", as_index=False)["total_mb"].sum().sort_values("hour_ts")
    return hourly


def _score_row(row: pd.Series, n: int) -> float:
    if not bool(row["stable_flag"]):
        return 1e6
    target_block = 0.08
    target_util = 0.60
    target_en = 0.50 * n
    score = (
        abs(float(row["P_block"]) - target_block)
        + 0.6 * abs(float(row["util_LEO"]) - target_util)
        + 0.25 * abs(float(row["E[n]"]) - target_en) / max(float(n), 1.0)
    )
    if not bool(row["nontrivial_block_flag"]):
        score += 1.0
    if not bool(row["realistic_load_flag"]):
        score += 0.6
    return float(score)


def pick_recommended_triplet(res: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Pick first moderate alpha, plus low/medium/high recommendations."""
    if res.empty:
        raise ValueError("empty calibration table")
    x = res.sort_values("alpha").reset_index(drop=True)

    stable = x[x["stable_flag"]]
    moderate = stable[(stable["P_block"] >= 0.01) & (stable["P_block"] <= 0.20)]

    first_moderate = moderate.iloc[0] if not moderate.empty else None

    if not moderate.empty:
        med_pool = moderate.copy()
    else:
        med_pool = stable[(stable["P_block"] >= 0.005) & (stable["P_block"] <= 0.30)]
        if med_pool.empty:
            med_pool = stable if not stable.empty else x
    med_pool = med_pool.assign(
        _score=(med_pool["P_block"] - 0.08).abs() + 0.4 * (med_pool["util_LEO"] - 0.60).abs()
    )
    medium = med_pool.sort_values(["_score", "alpha"]).iloc[0]

    lower = stable[stable["alpha"] < float(medium["alpha"])]
    if not lower.empty:
        low_pool = lower[(lower["P_block"] >= 0.0) & (lower["P_block"] < float(medium["P_block"]))]
        low = low_pool.sort_values(["P_block", "alpha"]).iloc[0] if not low_pool.empty else lower.iloc[-1]
    else:
        low = medium

    higher = stable[(stable["alpha"] > float(medium["alpha"])) & (stable["P_block"] < 0.95)]
    if not higher.empty:
        hi_pool = higher[higher["P_block"] > float(medium["P_block"])]
        high = hi_pool.sort_values(["alpha"]).iloc[0] if not hi_pool.empty else higher.iloc[0]
    else:
        high = None

    def _as_dict(r: pd.Series | None) -> dict[str, Any]:
        if r is None:
            return {}
        return {
            "alpha": float(r["alpha"]),
            "P_block": float(r["P_block"]),
            "E[n]": float(r["E[n]"]),
            "E[m]": float(r["E[m]"]),
            "P(m=M)": float(r["P(m=M)"]),
            "util_LEO": float(r["util_LEO"]),
            "util_MEO": float(r["util_MEO"]),
        }

    return {
        "first_moderate": _as_dict(first_moderate),
        "recommended_low": _as_dict(low),
        "recommended_medium": _as_dict(medium),
        "recommended_high": _as_dict(high),
    }


def calibrate_alphas(
    aggregated_lambda_csv: Path | str,
    output_csv: Path | str,
    alphas: list[float],
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    src = Path(aggregated_lambda_csv).resolve()
    dst = Path(output_csv).resolve()
    hourly = _hourly_total_from_aggregated(src)

    base = default_params()
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        lambda_t = float(alpha) * hourly["total_mb"] / 3600.0
        lam_mean = float(lambda_t.mean())
        lam_peak = float(lambda_t.max())

        met_mean = solve_metrics(replace(base, lam=lam_mean))
        met_peak = solve_metrics(replace(base, lam=lam_peak))

        p_block = float(met_mean["P(n=N)"])
        p_m_full = float(met_mean["P(m=M)"])
        util = float(met_mean["util_LEO"])
        util_meo = float(met_mean["util_MEO"])
        e_n = float(met_mean["E[n]"])
        e_m = float(met_mean["E[m]"])

        stable = bool(np.isfinite(p_block) and np.isfinite(p_m_full) and p_block < 0.99 and p_m_full < 0.99)
        nontrivial = bool(np.isfinite(p_block) and 0.01 <= p_block <= 0.30)
        realistic = bool(
            np.isfinite(util)
            and np.isfinite(util_meo)
            and 0.10 <= util <= 0.95
            and 0.0 <= util_meo <= 0.95
            and 0.2 <= e_n <= (base.N - 0.2)
        )

        row = {
            "alpha": float(alpha),
            "lambda_mean": lam_mean,
            "lambda_peak": lam_peak,
            "E[n]": e_n,
            "E[m]": e_m,
            "P_block": p_block,
            "P(m=M)": p_m_full,
            "util_LEO": util,
            "util_MEO": util_meo,
            "E[n]_peak": float(met_peak["E[n]"]),
            "E[m]_peak": float(met_peak["E[m]"]),
            "P_block_peak": float(met_peak["P(n=N)"]),
            "P(m=M)_peak": float(met_peak["P(m=M)"]),
            "util_LEO_peak": float(met_peak["util_LEO"]),
            "util_MEO_peak": float(met_peak["util_MEO"]),
            "stable_flag": stable,
            "nontrivial_block_flag": nontrivial,
            "realistic_load_flag": realistic,
            "error": met_mean.get("error", ""),
            "N": base.N,
            "M": base.M,
            "gamma": base.gamma,
            "C_LEO": base.C_LEO,
            "mu_LEO": base.mu_LEO,
            "mu_MEO": base.mu_MEO,
        }
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    res["score"] = res.apply(lambda r: _score_row(r, base.N), axis=1)

    preferred = res[res["stable_flag"] & res["nontrivial_block_flag"] & res["realistic_load_flag"]]
    if preferred.empty:
        preferred = res[res["stable_flag"] & res["nontrivial_block_flag"]]
    if preferred.empty:
        preferred = res[res["stable_flag"]]
    if preferred.empty:
        preferred = res

    best_idx = int(preferred["score"].idxmin())
    res["is_best"] = False
    res.loc[best_idx, "is_best"] = True

    best = res.loc[best_idx].to_dict()
    best_summary = {
        "best_alpha": float(best["alpha"]),
        "reason": (
            "Выбран alpha с минимальным интегральным отклонением от целевой области "
            "умеренной блокировки (около 0.08), реалистичной загрузки LEO и устойчивого режима."
        ),
        "P_block": float(best["P_block"]),
        "E[n]": float(best["E[n]"]),
        "util_LEO": float(best["util_LEO"]),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(dst, index=False)
    return dst, res, best_summary
