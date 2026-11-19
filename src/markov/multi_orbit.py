"""Multi-orbit (LEO/MEO/GEO) retrial CTMC — extension of the Leontyeva two-orbit model.

Two formulations are provided.

1. **Aggregated** (:func:`build_aggregated_generator`) — the Leontyeva state ``(n, m)`` kept
   unchanged, only the service rate becomes a K-segment function::

       μ(n) = Σ_j  min(max(0, n - C_1 - … - C_{j-1}), C_j) · μ_j

   For K = 2 this is *exactly* the model in :mod:`src.markov.ctmc`. The formula implicitly
   assumes that when a channel on a faster orbit frees up, a session from a slower orbit is
   moved to it instantly (re-packing). That is acceptable for LEO/MEO but questionable once
   GEO is involved.

2. **Exact per-orbit occupancy, multi-class** (:class:`MultiOrbitParams`,
   :func:`build_generator`) — state ``(k_1, …, k_K, m_1, …, m_C)`` where ``k_j`` is the
   number of busy channels on orbit *j* and ``m_c`` the number of class-*c* requests in that
   class's retrial orbit. Each traffic class has its own arrival rate and its own orbit
   traversal order (e.g. URLLC: LEO→MEO, eMBB: MEO→LEO→GEO, mMTC: GEO→MEO), which is what
   3GPP TR 22.887 use case 5.20 describes. With a single class this is the single-class
   model with an arbitrary traversal order (LEO→MEO→GEO, MEO→LEO→GEO, …).

Transitions of the exact model, from state ``(k, m)``:

* class-*c* arrival (``λ_c``): occupy the first orbit in ``order_c`` with a free channel;
  if none is free and ``m_c < M_c`` join the class-*c* retrial orbit; otherwise lost;
* class-*c* retrial (``m_c · γ_c``): if an orbit in ``order_c`` is free, occupy the first
  such orbit and ``m_c -= 1``; otherwise nothing happens (the request stays in orbit);
* service completion on orbit *j* (``k_j · μ_j``): ``k_j -= 1``.

Only numpy is required; scipy's sparse solver is used when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DENSE_STATE_LIMIT = 6000


# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class Orbit:
    """One orbital layer: number of channels, per-channel service rate, round-trip delay."""

    name: str
    channels: int
    mu: float
    delay_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError(f"Orbit {self.name}: channels must be >= 1.")
        if self.mu <= 0:
            raise ValueError(f"Orbit {self.name}: mu must be > 0.")
        if self.delay_ms < 0:
            raise ValueError(f"Orbit {self.name}: delay_ms must be >= 0.")


@dataclass(frozen=True)
class TrafficClass:
    """A traffic class with its arrival rate, orbit traversal order and retrial orbit."""

    name: str
    lam: float
    order: tuple[str, ...]
    M: int = 4
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.lam < 0 or self.gamma < 0:
            raise ValueError(f"Class {self.name}: lam and gamma must be >= 0.")
        if self.M < 0:
            raise ValueError(f"Class {self.name}: M must be >= 0.")
        if not self.order:
            raise ValueError(f"Class {self.name}: order must list at least one orbit.")
        if len(set(self.order)) != len(self.order):
            raise ValueError(f"Class {self.name}: order contains duplicates.")


@dataclass(frozen=True)
class MultiOrbitParams:
    """Orbits plus traffic classes for the exact multi-orbit model."""

    orbits: tuple[Orbit, ...]
    classes: tuple[TrafficClass, ...]
    _orbit_index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.orbits or not self.classes:
            raise ValueError("At least one orbit and one class are required.")
        names = [o.name for o in self.orbits]
        if len(set(names)) != len(names):
            raise ValueError("Orbit names must be unique.")
        idx = {n: i for i, n in enumerate(names)}
        for c in self.classes:
            unknown = [o for o in c.order if o not in idx]
            if unknown:
                raise ValueError(f"Class {c.name}: unknown orbit(s) {unknown}.")
        object.__setattr__(self, "_orbit_index", idx)

    def orbit_index(self, name: str) -> int:
        return self._orbit_index[name]

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(o.channels + 1 for o in self.orbits) + tuple(c.M + 1 for c in self.classes)

    @property
    def n_states(self) -> int:
        return int(np.prod(self.dims))


def leo_meo_geo(
    *,
    c_leo: int = 2,
    c_meo: int = 3,
    c_geo: int = 2,
    mu_leo: float = 3.0,
    mu_meo: float = 2.0,
    mu_geo: float = 1.0,
    delay_leo_ms: float = 30.0,
    delay_meo_ms: float = 130.0,
    delay_geo_ms: float = 550.0,
) -> tuple[Orbit, Orbit, Orbit]:
    """Default three orbits. LEO/MEO numbers match :func:`src.markov.leontyeva.default_params`.

    Delays are illustrative round-trip propagation values (UE–satellite–gateway); override
    them for a concrete constellation.
    """

    return (
        Orbit("LEO", c_leo, mu_leo, delay_leo_ms),
        Orbit("MEO", c_meo, mu_meo, delay_meo_ms),
        Orbit("GEO", c_geo, mu_geo, delay_geo_ms),
    )


def single_class(
    orbits: tuple[Orbit, ...],
    lam: float,
    order: tuple[str, ...] | None = None,
    *,
    M: int = 4,
    gamma: float = 1.0,
) -> MultiOrbitParams:
    """One traffic class traversing ``order`` (default: orbits in the given sequence)."""

    order = tuple(order) if order is not None else tuple(o.name for o in orbits)
    return MultiOrbitParams(orbits=orbits, classes=(TrafficClass("all", lam, order, M, gamma),))


def scenario_5_20(
    orbits: tuple[Orbit, ...],
    lam_total: float,
    *,
    shares: dict[str, float] | None = None,
    orders: dict[str, tuple[str, ...]] | None = None,
    M: int = 3,
    gamma: float = 1.0,
) -> MultiOrbitParams:
    """Three-class model following 3GPP TR 22.887 use case 5.20.

    Default routing (preferred orbit first, then fall-back):

    * URLLC (gaming, latency-sensitive) — LEO → MEO
    * eMBB  (video)                     — MEO → LEO → GEO
    * mMTC  (messaging, delay-tolerant) — GEO → MEO

    Default shares are the mean hourly volume shares of the dataset
    (``outputs/tables/slice_descriptive_stats.csv``): URLLC 18.6 %, eMBB 64.6 %, mMTC 16.8 %.
    """

    shares = shares or {"URLLC": 0.186, "eMBB": 0.646, "mMTC": 0.168}
    orders = orders or {
        "URLLC": ("LEO", "MEO"),
        "eMBB": ("MEO", "LEO", "GEO"),
        "mMTC": ("GEO", "MEO"),
    }
    total = sum(shares.values())
    classes = tuple(
        TrafficClass(name, lam_total * shares[name] / total, tuple(orders[name]), M, gamma)
        for name in shares
    )
    return MultiOrbitParams(orbits=orbits, classes=classes)


# --------------------------------------------------------------------------- solver


def solve_stationary(Q: Any) -> np.ndarray:
    """Solve ``π Q = 0``, ``Σ π = 1`` for a dense ndarray or a scipy sparse matrix."""

    S = Q.shape[0]
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        if sp.issparse(Q):
            A = sp.vstack([Q.T.tocsr()[: S - 1, :], sp.csr_matrix(np.ones((1, S)))]).tocsc()
            b = np.zeros(S)
            b[-1] = 1.0
            pi = spla.spsolve(A, b)
            return _normalise(pi, Q)
    except ImportError:  # pragma: no cover - scipy optional
        pass

    Qd = np.asarray(Q.toarray() if hasattr(Q, "toarray") else Q, dtype=float)
    A = np.vstack([Qd.T[: S - 1, :], np.ones((1, S))])
    b = np.zeros(S)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return _normalise(pi, Qd)


def _normalise(pi: np.ndarray, Q: Any) -> np.ndarray:
    pi = np.clip(np.asarray(pi, dtype=float).ravel(), 0.0, None)
    s = pi.sum()
    if s <= 0:
        raise RuntimeError("Degenerate stationary distribution.")
    pi = pi / s
    err = float(np.linalg.norm(Q.T @ pi)) if hasattr(Q, "toarray") else float(np.linalg.norm(pi @ Q))
    if err > 1e-7:
        logger.info("|π Q| = %s (should be ~0)", err)
    return pi


def _assemble(rows: list[int], cols: list[int], vals: list[float], S: int) -> Any:
    """Off-diagonal triplets → generator (diagonal = −row sums). Sparse when scipy exists."""

    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    v = np.asarray(vals, dtype=float)
    out = np.bincount(r, weights=v, minlength=S)
    try:
        import scipy.sparse as sp

        diag = np.arange(S)
        return sp.csr_matrix(
            (np.concatenate([v, -out]), (np.concatenate([r, diag]), np.concatenate([c, diag]))),
            shape=(S, S),
        )
    except ImportError:  # pragma: no cover - scipy optional
        if S > DENSE_STATE_LIMIT:
            raise MemoryError(f"{S} states need scipy for a sparse solve.") from None
        Q = np.zeros((S, S))
        np.add.at(Q, (r, c), v)
        Q[np.diag_indices(S)] = -out
        return Q


# --------------------------------------------------------------------------- aggregated model


def mu_n_multi(n: int, channels: tuple[int, ...], mus: tuple[float, ...]) -> float:
    """K-segment service rate: fill orbit 1 first, then orbit 2, … (Leontyeva for K = 2)."""

    rate, left = 0.0, n
    for c, mu in zip(channels, mus):
        take = min(left, c)
        rate += take * mu
        left -= take
    return float(rate)


def build_aggregated_generator(
    lam: float, gamma: float, M: int, channels: tuple[int, ...], mus: tuple[float, ...]
) -> np.ndarray:
    """Generator on ``(n, m)`` with ``N = Σ channels`` and the K-segment ``μ(n)``."""

    if len(channels) != len(mus):
        raise ValueError("channels and mus must have equal length.")
    N = int(sum(channels))
    S = (N + 1) * (M + 1)
    Q = np.zeros((S, S))
    for n in range(N + 1):
        for m in range(M + 1):
            i = n * (M + 1) + m
            if n < N:
                Q[i, (n + 1) * (M + 1) + m] += lam
            if n == N and m < M:
                Q[i, n * (M + 1) + m + 1] += lam
            if n > 0:
                Q[i, (n - 1) * (M + 1) + m] += mu_n_multi(n, channels, mus)
            if m > 0 and n < N:
                Q[i, (n + 1) * (M + 1) + m - 1] += m * gamma
    Q[np.diag_indices(S)] = -Q.sum(axis=1)
    return Q


def aggregated_metrics(
    pi: np.ndarray, M: int, channels: tuple[int, ...], names: tuple[str, ...] | None = None
) -> dict[str, float]:
    """Leontyeva metrics for the aggregated model plus ``util_<orbit>`` for every layer."""

    names = names or tuple(f"orbit{j}" for j in range(len(channels)))
    N = int(sum(channels))
    P = np.asarray(pi).reshape(N + 1, M + 1)
    n = np.arange(N + 1)
    pn = P.sum(axis=1)
    out = {
        "P_loss": float(P[N, M]),
        "P(n=N)": float(pn[N]),
        "P(m=M)": float(P[:, M].sum()),
        "E[n]": float(pn @ n),
        "E[m]": float(P.sum(axis=0) @ np.arange(M + 1)),
    }
    lower = 0
    for name, c in zip(names, channels):
        busy = np.clip(n - lower, 0, c)
        out[f"util_{name}"] = float(pn @ busy / c)
        lower += c
    return out


# --------------------------------------------------------------------------- exact model


def _first_free(k: tuple[int, ...], order_idx: tuple[int, ...], caps: tuple[int, ...]) -> int | None:
    for j in order_idx:
        if k[j] < caps[j]:
            return j
    return None


def build_generator(p: MultiOrbitParams) -> tuple[Any, np.ndarray]:
    """Generator of the exact model and the admission-rate matrix.

    Returns
    -------
    Q
        ``S × S`` generator (scipy CSR if scipy is installed, dense ndarray otherwise).
    admit
        ``S × C × K`` array: rate at which class-*c* requests (new + retrial) are admitted to
        orbit *j* while in state *s*. Used for orbit shares and mean delay.
    """

    K, C = len(p.orbits), len(p.classes)
    caps = tuple(o.channels for o in p.orbits)
    mus = tuple(o.mu for o in p.orbits)
    dims = p.dims
    S = p.n_states
    orders = [tuple(p.orbit_index(n) for n in c.order) for c in p.classes]
    if S > 200_000:
        raise MemoryError(f"State space too large ({S}); reduce channels or M.")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    admit = np.zeros((S, C, K))

    def add(i: int, state: list[int], rate: float) -> None:
        if rate > 0:
            rows.append(i)
            cols.append(int(np.ravel_multi_index(tuple(state), dims)))
            vals.append(rate)

    for i in range(S):
        s = list(np.unravel_index(i, dims))
        k = tuple(int(x) for x in s[:K])
        m = [int(x) for x in s[K:]]
        for c, cls in enumerate(p.classes):
            j = _first_free(k, orders[c], caps)
            if j is not None:
                t = list(s)
                t[j] += 1
                add(i, t, cls.lam)
                admit[i, c, j] += cls.lam
                if m[c] > 0:
                    t2 = list(t)
                    t2[K + c] -= 1
                    add(i, t2, m[c] * cls.gamma)
                    admit[i, c, j] += m[c] * cls.gamma
            elif m[c] < cls.M:
                t = list(s)
                t[K + c] += 1
                add(i, t, cls.lam)
        for j in range(K):
            if k[j] > 0:
                t = list(s)
                t[j] -= 1
                add(i, t, k[j] * mus[j])

    return _assemble(rows, cols, vals, S), admit


def compute_metrics(pi: np.ndarray, p: MultiOrbitParams, admit: np.ndarray) -> dict[str, Any]:
    """Per-class and per-orbit QoS metrics of the exact model.

    Per class *c* (PASTA applies, arrivals are Poisson):

    * ``P_block[c]`` — all orbits in ``order_c`` busy (request goes to retrial orbit or is lost);
    * ``P_loss[c]``  — all orbits in ``order_c`` busy and the class-*c* retrial orbit full;
    * ``E_m[c]``     — mean class-*c* retrial-orbit length;
    * ``W_retrial[c]`` — mean time in the retrial orbit per admitted request (Little: E_m / λ_acc);
    * ``share[c][orbit]`` — fraction of admitted class-*c* sessions served by each orbit;
    * ``delay_ms[c]`` — mean round-trip propagation delay of admitted class-*c* sessions.

    Per orbit: ``util[orbit]`` = E[k_j] / channels_j. Plus traffic-weighted overall values.
    """

    K = len(p.orbits)
    caps = np.array([o.channels for o in p.orbits])
    delays = np.array([o.delay_ms for o in p.orbits])
    dims = p.dims
    grid = np.indices(dims).reshape(len(dims), -1)  # (K + C) × S
    k = grid[:K]
    pi = np.asarray(pi)

    out: dict[str, Any] = {"n_states": int(pi.size), "util": {}, "classes": {}}
    for j, o in enumerate(p.orbits):
        out["util"][o.name] = float(pi @ k[j] / caps[j])
    out["E[n]"] = float(pi @ k.sum(axis=0))

    adm = np.einsum("s,scj->cj", pi, admit)
    lam_tot = sum(c.lam for c in p.classes)
    acc_tot, w_delay, w_block, w_loss, w_wait = 0.0, 0.0, 0.0, 0.0, 0.0
    for c, cls in enumerate(p.classes):
        oi = [p.orbit_index(n) for n in cls.order]
        all_busy = np.all(k[oi] == caps[oi, None], axis=0)
        m_c = grid[K + c]
        p_block = float(pi[all_busy].sum())
        p_loss = float(pi[all_busy & (m_c == cls.M)].sum())
        e_m = float(pi @ m_c)
        lam_acc = cls.lam * (1.0 - p_loss)
        tot_adm = adm[c].sum()
        share = adm[c] / tot_adm if tot_adm > 0 else np.zeros(K)
        delay = float(share @ delays)
        wait = e_m / lam_acc if lam_acc > 0 else float("nan")
        out["classes"][cls.name] = {
            "lambda": cls.lam,
            "P_block": p_block,
            "P_loss": p_loss,
            "E_m": e_m,
            "W_retrial": wait,
            "delay_ms": delay,
            "share": {o.name: float(share[j]) for j, o in enumerate(p.orbits)},
        }
        acc_tot += lam_acc
        w_delay += lam_acc * delay
        w_block += cls.lam * p_block
        w_loss += cls.lam * p_loss
        w_wait += e_m
    out["P_block"] = w_block / lam_tot if lam_tot > 0 else 0.0
    out["P_loss"] = w_loss / lam_tot if lam_tot > 0 else 0.0
    out["delay_ms"] = w_delay / acc_tot if acc_tot > 0 else float("nan")
    out["W_retrial"] = w_wait / acc_tot if acc_tot > 0 else float("nan")
    return out


def solve(p: MultiOrbitParams) -> dict[str, Any]:
    """Build, solve and evaluate the exact model in one call."""

    Q, admit = build_generator(p)
    pi = solve_stationary(Q)
    return compute_metrics(pi, p, admit)


def flatten_metrics(res: dict[str, Any]) -> dict[str, float]:
    """One flat row (for CSV) from :func:`compute_metrics` output."""

    row: dict[str, float] = {
        "n_states": res["n_states"],
        "P_block": res["P_block"],
        "P_loss": res["P_loss"],
        "E[n]": res["E[n]"],
        "W_retrial": res["W_retrial"],
        "delay_ms": res["delay_ms"],
    }
    for o, u in res["util"].items():
        row[f"util_{o}"] = u
    for c, d in res["classes"].items():
        for key in ("lambda", "P_block", "P_loss", "E_m", "W_retrial", "delay_ms"):
            row[f"{c}.{key}"] = d[key]
        for o, s in d["share"].items():
            row[f"{c}.share_{o}"] = s
    return row
