"""Stationary distribution table and Leontyeva performance metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markov.ctmc import build_generator_matrix, stationary_distribution
from src.markov.leontyeva import LeontyevaParams, index_to_nm, state_index

logger = logging.getLogger(__name__)


def distribution_table(pi: np.ndarray, p: LeontyevaParams) -> pd.DataFrame:
    """``stationary_distribution.csv`` rows: ``state_id``, ``n``, ``m``, ``pi``."""

    N, M = p.N, p.M
    rows: list[dict[str, Any]] = []
    for k in range(len(pi)):
        n, m = index_to_nm(k, M)
        rows.append({"state_id": k, "n": n, "m": m, "pi": float(pi[k])})
    return pd.DataFrame(rows)


def compute_metrics(pi: np.ndarray, p: LeontyevaParams) -> dict[str, float]:
    """Blocking, expectations, LEO/MEO utilizations."""

    N, M = p.N, p.M
    c = p.C_LEO

    P_n_eq_N = float(sum(pi[state_index(N, m, M)] for m in range(M + 1)))
    P_m_eq_M = float(sum(pi[state_index(n, M, M)] for n in range(N + 1)))
    P_loss = float(pi[state_index(N, M, M)])

    E_n = 0.0
    E_m = 0.0
    E_min_nc = 0.0
    E_excess = 0.0
    for k in range(len(pi)):
        n, m = index_to_nm(k, M)
        pk = float(pi[k])
        E_n += pk * n
        E_m += pk * m
        E_min_nc += pk * min(n, c)
        E_excess += pk * max(0, n - c)

    denom_leo = float(c) if c > 0 else float("nan")
    util_leo = float(E_min_nc / denom_leo) if c > 0 else float("nan")

    cap_meo = max(0, N - c)
    denom_meo = float(cap_meo) if cap_meo > 0 else float("nan")
    util_meo = float(E_excess / denom_meo) if cap_meo > 0 else 0.0

    return {
        "P_loss": P_loss,
        "P(n=N)": P_n_eq_N,
        "P(m=M)": P_m_eq_M,
        "E[n]": float(E_n),
        "E[m]": float(E_m),
        "util_LEO": util_leo,
        "util_MEO": util_meo,
    }


def run_leontyeva(
    p: LeontyevaParams,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Solve ``π Q = 0``, write ``stationary_distribution.csv`` and ``metrics.csv``."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Q = build_generator_matrix(p)
    pi = stationary_distribution(Q)

    df_pi = distribution_table(pi, p)
    p_pi = out_dir / "stationary_distribution.csv"
    df_pi.to_csv(p_pi, index=False)

    met = compute_metrics(pi, p)
    met["N"] = p.N
    met["M"] = p.M
    met["lambda"] = p.lam
    met["gamma"] = p.gamma
    met["C_LEO"] = p.C_LEO
    met["mu_LEO"] = p.mu_LEO
    met["mu_MEO"] = p.mu_MEO
    df_m = pd.DataFrame([met])
    p_m = out_dir / "metrics.csv"
    df_m.to_csv(p_m, index=False)

    logger.info("Wrote %s and %s", p_pi, p_m)
    return p_pi, p_m
