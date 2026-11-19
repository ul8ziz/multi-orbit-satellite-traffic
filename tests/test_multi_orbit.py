"""Multi-orbit (LEO/MEO/GEO) CTMC: consistency with the two-orbit Leontyeva model."""

from __future__ import annotations

import numpy as np
import pytest

from src.markov.ctmc import build_generator_matrix, stationary_distribution
from src.markov.leontyeva import default_params
from src.markov.multi_orbit import (
    Orbit,
    aggregated_metrics,
    build_aggregated_generator,
    build_generator,
    leo_meo_geo,
    scenario_5_20,
    single_class,
    solve,
    solve_stationary,
)


def _dense(Q):
    return Q.toarray() if hasattr(Q, "toarray") else np.asarray(Q)


def test_aggregated_k2_equals_leontyeva() -> None:
    p = default_params()
    Q_ref = build_generator_matrix(p)
    Q = build_aggregated_generator(p.lam, p.gamma, p.M, (p.C_LEO, p.N - p.C_LEO), (p.mu_LEO, p.mu_MEO))
    assert np.allclose(Q, Q_ref)


def test_aggregated_k2_reproduces_article_table6() -> None:
    p = default_params()
    Q = build_aggregated_generator(p.lam, p.gamma, p.M, (2, 3), (3.0, 2.0))
    met = aggregated_metrics(stationary_distribution(Q), p.M, (2, 3), ("LEO", "MEO"))
    assert met["P_loss"] == pytest.approx(8.43e-7, rel=1e-2)
    assert met["P(n=N)"] == pytest.approx(0.000973, rel=1e-3)
    assert met["E[n]"] == pytest.approx(0.680935, rel=1e-5)
    assert met["util_LEO"] == pytest.approx(0.319064, rel=1e-5)
    assert met["util_MEO"] == pytest.approx(0.014269, rel=1e-4)


def test_aggregated_three_orbits_valid() -> None:
    Q = build_aggregated_generator(2.0, 1.0, 4, (2, 3, 2), (3.0, 2.0, 1.0))
    pi = stationary_distribution(Q)
    assert np.linalg.norm(pi @ Q) < 1e-8
    met = aggregated_metrics(pi, 4, (2, 3, 2), ("LEO", "MEO", "GEO"))
    assert met["util_LEO"] >= met["util_MEO"] >= met["util_GEO"] >= 0.0


def test_exact_single_orbit_equals_aggregated() -> None:
    """With one orbit there is no re-packing, so both formulations must coincide."""

    orb = (Orbit("LEO", 3, 2.0),)
    res = solve(single_class(orb, lam=4.0, M=3, gamma=0.7))
    Q = build_aggregated_generator(4.0, 0.7, 3, (3,), (2.0,))
    met = aggregated_metrics(stationary_distribution(Q), 3, (3,), ("LEO",))
    assert res["P_block"] == pytest.approx(met["P(n=N)"], rel=1e-9)
    assert res["P_loss"] == pytest.approx(met["P_loss"], rel=1e-9)
    assert res["util"]["LEO"] == pytest.approx(met["util_LEO"], rel=1e-9)
    assert res["classes"]["all"]["E_m"] == pytest.approx(met["E[m]"], rel=1e-9)


def test_exact_generator_and_stationarity() -> None:
    p = single_class(leo_meo_geo(), lam=3.0)
    Q, _ = build_generator(p)
    Qd = _dense(Q)
    assert Qd.shape == (p.n_states, p.n_states)
    assert np.allclose(Qd.sum(axis=1), 0.0)
    pi = solve_stationary(Q)
    assert pi.sum() == pytest.approx(1.0)
    assert np.linalg.norm(pi @ Qd) < 1e-8


def test_exact_metrics_ranges_and_order_effect() -> None:
    orbits = leo_meo_geo()
    leo_first = solve(single_class(orbits, 6.0, ("LEO", "MEO", "GEO")))
    geo_first = solve(single_class(orbits, 6.0, ("GEO", "MEO", "LEO")))
    for r in (leo_first, geo_first):
        assert 0.0 <= r["P_loss"] <= r["P_block"] <= 1.0
        assert sum(r["classes"]["all"]["share"].values()) == pytest.approx(1.0)
        assert all(0.0 <= u <= 1.0 for u in r["util"].values())
    # LEO is fastest and has the shortest delay, so serving it first must win on both.
    assert leo_first["delay_ms"] < geo_first["delay_ms"]
    assert leo_first["P_block"] < geo_first["P_block"]


def test_scenario_5_20_routing() -> None:
    p = scenario_5_20(leo_meo_geo(), lam_total=2.0)
    res = solve(p)
    assert set(res["classes"]) == {"URLLC", "eMBB", "mMTC"}
    assert sum(c["lambda"] for c in res["classes"].values()) == pytest.approx(2.0)
    # URLLC is never routed to GEO, mMTC never to LEO.
    assert res["classes"]["URLLC"]["share"]["GEO"] == 0.0
    assert res["classes"]["mMTC"]["share"]["LEO"] == 0.0
    # At low load each class lands mostly on its preferred orbit.
    assert res["classes"]["URLLC"]["share"]["LEO"] > 0.8
    assert res["classes"]["eMBB"]["share"]["MEO"] > 0.8
    assert res["classes"]["mMTC"]["share"]["GEO"] > 0.8
    assert res["classes"]["URLLC"]["delay_ms"] < res["classes"]["mMTC"]["delay_ms"]


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        Orbit("LEO", 0, 1.0)
    with pytest.raises(ValueError):
        single_class(leo_meo_geo(), 1.0, ("LEO", "XEO"))
