"""CTMC generator :math:`Q` and stationary distribution :math:`\\pi Q=0`, :math:`\\sum\\pi=1`."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.markov.leontyeva import LeontyevaParams, mu_n, state_index

logger = logging.getLogger(__name__)


def build_generator_matrix(p: LeontyevaParams) -> np.ndarray:
    """Build infinitesimal generator ``Q`` for the Leontyeva model.

    States are ``(n,m)`` with ``n=0..N``, ``m=0..M``. Off-diagonal ``Q[i,j]`` is the
    transition rate **from** state **i** **to** state **j**; rows sum to zero.

    Transitions (from state ``(n,m)``):

    1. ``λ`` to ``(n+1,m)`` if ``n < N``.
    2. ``λ`` to ``(n,m+1)`` if ``n = N`` and ``m < M``.
    3. ``μ(n)`` to ``(n-1,m)`` if ``n > 0``, with ``μ(n) = min(n,C)·μ_LEO + max(0,n-C)·μ_MEO``.
    4. ``m·γ`` to ``(n+1,m-1)`` if ``m > 0`` and ``n < N``.
    """

    N, M = p.N, p.M
    S = (N + 1) * (M + 1)
    Q = np.zeros((S, S), dtype=float)

    for n in range(N + 1):
        for m in range(M + 1):
            i = state_index(n, m, M)

            if n < N:
                j = state_index(n + 1, m, M)
                Q[i, j] += p.lam

            if n == N and m < M:
                j = state_index(N, m + 1, M)
                Q[i, j] += p.lam

            if n > 0:
                j = state_index(n - 1, m, M)
                Q[i, j] += mu_n(n, p)

            if m > 0 and n < N:
                j = state_index(n + 1, m - 1, M)
                Q[i, j] += m * p.gamma

    for i in range(S):
        Q[i, i] = -np.sum(Q[i, :])

    return Q


def stationary_distribution(Q: np.ndarray) -> np.ndarray:
    """Solve ``π Q = 0`` with ``sum π = 1`` (``π`` a row vector)."""

    S = Q.shape[0]
    A = np.vstack([Q.T[: S - 1, :], np.ones((1, S))])
    b = np.zeros(S)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s <= 0:
        msg = "Degenerate stationary distribution."
        raise RuntimeError(msg)
    pi = pi / s
    err = np.linalg.norm(pi @ Q)
    if err > 1e-7:
        logger.info("|π Q| = %s (should be ~0)", err)
    return pi


def build_generator_matrix_legacy(states: Any) -> Any:
    """``states`` must be a :class:`LeontyevaParams` instance."""

    if not isinstance(states, LeontyevaParams):
        msg = "states must be LeontyevaParams"
        raise TypeError(msg)
    return build_generator_matrix(states)
