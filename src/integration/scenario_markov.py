"""Run the unchanged Leontyeva CTMC once per scenario (constant λ per run)."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.markov.ctmc import build_generator_matrix, stationary_distribution
from src.markov.leontyeva import LeontyevaParams, default_params
from src.markov.stationary import compute_metrics

logger = logging.getLogger(__name__)


def run_scenarios(
    lambda_mean: float,
    lambda_peak: float,
    *,
    base: LeontyevaParams | None = None,
    out_dir: Path | None = None,
    alpha: float = 1.0,
) -> tuple[Path, Path]:
    """Solve stationary CTMC separately for mean-load and peak-load λ.

    The generator structure and all parameters except ``lam`` match ``base``
    (default :func:`default_params`). Each scenario uses a **constant** λ (no Q(t)).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base or default_params()

    scenarios: list[tuple[str, float]] = [
        ("mean_load", lambda_mean),
        ("peak_load", lambda_peak),
    ]

    rows: list[dict[str, Any]] = []
    blocking: list[tuple[str, float]] = []

    for name, lam in scenarios:
        p = replace(base, lam=float(lam))
        Q = build_generator_matrix(p)
        pi = stationary_distribution(Q)
        met = compute_metrics(pi, p)
        row = {
            "scenario": name,
            "alpha": alpha,
            "lambda": float(lam),
            **met,
            "N": p.N,
            "M": p.M,
            "gamma": p.gamma,
            "C_LEO": p.C_LEO,
            "mu_LEO": p.mu_LEO,
            "mu_MEO": p.mu_MEO,
        }
        rows.append(row)
        blocking.append((name, met["P_loss"], met["P(n=N)"]))

    df = pd.DataFrame(rows)
    p_csv = out_dir / "scenario_metrics.csv"
    df.to_csv(p_csv, index=False)

    names = [b[0] for b in blocking]
    p_loss_vals = [b[1] for b in blocking]
    p_busy_vals = [b[2] for b in blocking]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, p_loss_vals, w, label="P_loss = π(N,M)", color=["steelblue", "coral"])
    ax.bar(x + w / 2, p_busy_vals, w, label="P(n=N)", color=["lightsteelblue", "lightsalmon"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Blocking vs scenario (Leontyeva CTMC)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p_png = out_dir / "blocking_vs_scenario.png"
    fig.savefig(p_png, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote %s and %s", p_csv, p_png)
    return p_csv, p_png
