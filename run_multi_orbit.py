"""CLI: three-orbit (LEO/MEO/GEO) retrial CTMC — order comparison and 3GPP TR 22.887 §5.20.

Outputs (default ``outputs/multi_orbit/``):

* ``order_comparison.csv`` / ``.png`` — one traffic class, all 6 orbit traversal orders, sweep of λ;
* ``scenario_5_20.csv`` / ``.png``   — three classes (URLLC/eMBB/mMTC): service-based routing of
  use case 5.20 versus "LEO first for everyone", sweep of λ;
* ``aggregated_vs_exact.csv``        — Leontyeva-style aggregated μ(n) versus the exact
  per-orbit model for LEO→MEO→GEO (shows the effect of the re-packing assumption).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _lambda_grid(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 6) for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("run_multi_orbit")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-leo", type=int, default=2)
    ap.add_argument("--c-meo", type=int, default=3)
    ap.add_argument("--c-geo", type=int, default=2)
    ap.add_argument("--mu-leo", type=float, default=3.0)
    ap.add_argument("--mu-meo", type=float, default=2.0)
    ap.add_argument("--mu-geo", type=float, default=1.0)
    ap.add_argument("--delay-leo", type=float, default=30.0, help="Round-trip delay, ms.")
    ap.add_argument("--delay-meo", type=float, default=130.0, help="Round-trip delay, ms.")
    ap.add_argument("--delay-geo", type=float, default=550.0, help="Round-trip delay, ms.")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=4, help="Retrial orbit capacity (single class).")
    ap.add_argument("--M-class", type=int, default=3, help="Retrial orbit capacity per class (5.20).")
    ap.add_argument("--lam-min", type=float, default=0.5)
    ap.add_argument("--lam-max", type=float, default=12.0)
    ap.add_argument("--lam-step", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "multi_orbit")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.markov.ctmc import stationary_distribution
    from src.markov.multi_orbit import (
        aggregated_metrics,
        build_aggregated_generator,
        flatten_metrics,
        leo_meo_geo,
        scenario_5_20,
        single_class,
        solve,
    )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    orbits = leo_meo_geo(
        c_leo=args.c_leo, c_meo=args.c_meo, c_geo=args.c_geo,
        mu_leo=args.mu_leo, mu_meo=args.mu_meo, mu_geo=args.mu_geo,
        delay_leo_ms=args.delay_leo, delay_meo_ms=args.delay_meo, delay_geo_ms=args.delay_geo,
    )
    names = tuple(o.name for o in orbits)
    lams = _lambda_grid(args.lam_min, args.lam_max, args.lam_step)

    # 1) single class: every traversal order -------------------------------------------
    rows = []
    for order in itertools.permutations(names):
        label = "→".join(order)
        for lam in lams:
            res = solve(single_class(orbits, lam, order, M=args.M, gamma=args.gamma))
            rows.append({"order": label, "lambda": lam, **flatten_metrics(res)})
    df_ord = pd.DataFrame(rows)
    p_ord = out / "order_comparison.csv"
    df_ord.to_csv(p_ord, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=140)
    for label, g in df_ord.groupby("order", sort=False):
        axes[0].plot(g["lambda"], g["P_block"], marker=".", label=label)
        axes[1].plot(g["lambda"], g["delay_ms"], marker=".", label=label)
        axes[2].plot(g["lambda"], g["W_retrial"], marker=".", label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1e-12)
    axes[0].set_title("Blocking probability P_block")
    axes[1].set_title("Mean round-trip propagation delay, ms")
    axes[2].set_title("Mean time in retrial orbit W")
    for ax in axes:
        ax.set_xlabel("λ (arrival rate)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, title="Traversal order")
    fig.suptitle(
        f"Single class, LEO/MEO/GEO = {args.c_leo}/{args.c_meo}/{args.c_geo} channels, "
        f"μ = {args.mu_leo}/{args.mu_meo}/{args.mu_geo}"
    )
    fig.tight_layout()
    p_ord_png = out / "order_comparison.png"
    fig.savefig(p_ord_png, bbox_inches="tight")
    plt.close(fig)

    # 2) scenario 5.20: service-based routing vs LEO-first ------------------------------
    policies = {
        "5.20 service-based": None,  # defaults of scenario_5_20
        "LEO-first for all": {c: ("LEO", "MEO", "GEO") for c in ("URLLC", "eMBB", "mMTC")},
    }
    rows = []
    for pol, orders in policies.items():
        for lam in lams:
            p = scenario_5_20(orbits, lam, orders=orders, M=args.M_class, gamma=args.gamma)
            rows.append({"policy": pol, "lambda": lam, **flatten_metrics(solve(p))})
    df_520 = pd.DataFrame(rows)
    p_520 = out / "scenario_5_20.csv"
    df_520.to_csv(p_520, index=False)

    classes = ("URLLC", "eMBB", "mMTC")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=140, sharex=True)
    styles = {"5.20 service-based": "-", "LEO-first for all": "--"}
    for pol, g in df_520.groupby("policy", sort=False):
        for j, c in enumerate(classes):
            axes[0, j].plot(g["lambda"], g[f"{c}.P_block"], styles[pol], marker=".", label=pol)
            axes[1, j].plot(g["lambda"], g[f"{c}.delay_ms"], styles[pol], marker=".", label=pol)
    for j, c in enumerate(classes):
        axes[0, j].set_title(f"{c}: P_block")
        axes[0, j].set_yscale("log")
        axes[0, j].set_ylim(bottom=1e-12)
        axes[1, j].set_title(f"{c}: mean delay, ms")
        axes[1, j].set_xlabel("λ total (arrival rate)")
        for i in range(2):
            axes[i, j].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("3GPP TR 22.887 §5.20: URLLC→LEO, eMBB→MEO, mMTC→GEO vs LEO-first routing")
    fig.tight_layout()
    p_520_png = out / "scenario_5_20.png"
    fig.savefig(p_520_png, bbox_inches="tight")
    plt.close(fig)

    # 3) aggregated (re-packing) vs exact for LEO→MEO→GEO -------------------------------
    caps = tuple(o.channels for o in orbits)
    mus = tuple(o.mu for o in orbits)
    rows = []
    for lam in lams:
        agg = aggregated_metrics(
            stationary_distribution(build_aggregated_generator(lam, args.gamma, args.M, caps, mus)),
            args.M, caps, names,
        )
        ex = solve(single_class(orbits, lam, names, M=args.M, gamma=args.gamma))
        row = {"lambda": lam, "agg.P_block": agg["P(n=N)"], "exact.P_block": ex["P_block"],
               "agg.P_loss": agg["P_loss"], "exact.P_loss": ex["P_loss"]}
        for n in names:
            row[f"agg.util_{n}"] = agg[f"util_{n}"]
            row[f"exact.util_{n}"] = ex["util"][n]
        rows.append(row)
    p_cmp = out / "aggregated_vs_exact.csv"
    pd.DataFrame(rows).to_csv(p_cmp, index=False)

    for path in (p_ord, p_ord_png, p_520, p_520_png, p_cmp):
        log.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
