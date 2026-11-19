"""Parameters for the Leontyeva LEO/MEO CTMC on state (n,m), n=0..N, m=0..M."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeontyevaParams:
    """Fixed numerical parameters for the exact Leontyeva model."""

    N: int
    M: int
    lam: float  # λ: arrival rate
    gamma: float  # γ: per-handoff intensity in the m-chain
    C_LEO: int  # LEO capacity threshold in μ(n)
    mu_LEO: float
    mu_MEO: float

    def __post_init__(self) -> None:
        if self.N < 0 or self.M < 0:
            raise ValueError("N and M must be non-negative.")
        if self.C_LEO < 0:
            raise ValueError("C_LEO must be non-negative.")
        if self.lam < 0 or self.gamma < 0:
            raise ValueError("λ and γ must be non-negative.")
        if self.mu_LEO < 0 or self.mu_MEO < 0:
            raise ValueError("μ_LEO and μ_MEO must be non-negative.")


def mu_n(n: int, p: LeontyevaParams) -> float:
    """Service rate μ(n) = min(n,C_LEO)·μ_LEO + max(0,n-C_LEO)·μ_MEO."""

    c = p.C_LEO
    return float(min(n, c) * p.mu_LEO + max(0, n - c) * p.mu_MEO)


def model_spec() -> dict[str, Any]:
    """Default specification dict (same numbers as :class:`default_params`)."""

    p = default_params()
    return {
        "N": p.N,
        "M": p.M,
        "lambda": p.lam,
        "gamma": p.gamma,
        "C_LEO": p.C_LEO,
        "mu_LEO": p.mu_LEO,
        "mu_MEO": p.mu_MEO,
    }


def default_params() -> LeontyevaParams:
    """Illustrative defaults (override via CLI or code)."""

    return LeontyevaParams(N=5, M=4, lam=2.0, gamma=1.0, C_LEO=2, mu_LEO=3.0, mu_MEO=2.0)


def state_index(n: int, m: int, M: int) -> int:
    """Row-major flat index for (n,m), m=0..M."""

    return n * (M + 1) + m


def index_to_nm(idx: int, M: int) -> tuple[int, int]:
    """Inverse of :func:`state_index`."""

    m = idx % (M + 1)
    n = idx // (M + 1)
    return n, m
