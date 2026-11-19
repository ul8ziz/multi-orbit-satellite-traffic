"""Leontyeva CTMC stationary check."""

from __future__ import annotations

import numpy as np

from src.markov.ctmc import build_generator_matrix, stationary_distribution
from src.markov.leontyeva import LeontyevaParams


def test_pi_Q_zero_and_metrics() -> None:
    p = LeontyevaParams(N=2, M=2, lam=1.0, gamma=0.5, C_LEO=1, mu_LEO=2.0, mu_MEO=1.0)
    Q = build_generator_matrix(p)
    pi = stationary_distribution(Q)
    assert abs(pi.sum() - 1.0) < 1e-9
    assert np.linalg.norm(pi @ Q) < 1e-6
    assert np.all(pi >= -1e-12)

    from src.markov.stationary import compute_metrics

    m = compute_metrics(pi, p)
    assert "P(n=N)" in m and "E[n]" in m
    assert 0.0 <= m["P(n=N)"] <= 1.0
    assert "P_loss" in m
    assert 0.0 <= m["P_loss"] <= m["P(n=N)"]
