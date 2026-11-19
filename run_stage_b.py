"""CLI: exact Leontyeva LEO/MEO CTMC — stationary distribution and metrics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_src_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    configure_logging()
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description="Leontyeva CTMC: πQ=0, outputs under outputs/markov/")
    ap.add_argument("--N", type=int, default=None, help="Max n (default from model_spec).")
    ap.add_argument("--M", type=int, default=None, help="Max m (default from model_spec).")
    ap.add_argument("--lambda", dest="lam", type=float, default=None, help="Arrival rate λ.")
    ap.add_argument("--gamma", type=float, default=None, help="Handoff intensity γ.")
    ap.add_argument("--c-leo", dest="C_LEO", type=int, default=None, help="C_LEO.")
    ap.add_argument("--mu-leo", type=float, default=None, help="μ_LEO.")
    ap.add_argument("--mu-meo", type=float, default=None, help="μ_MEO.")
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "markov",
        help="Directory for stationary_distribution.csv and metrics.csv",
    )
    args = ap.parse_args(argv)

    from src.markov.leontyeva import LeontyevaParams, default_params
    from src.markov.stationary import run_leontyeva

    d = default_params()
    p = LeontyevaParams(
        N=args.N if args.N is not None else d.N,
        M=args.M if args.M is not None else d.M,
        lam=args.lam if args.lam is not None else d.lam,
        gamma=args.gamma if args.gamma is not None else d.gamma,
        C_LEO=args.C_LEO if args.C_LEO is not None else d.C_LEO,
        mu_LEO=args.mu_leo if args.mu_leo is not None else d.mu_LEO,
        mu_MEO=args.mu_meo if args.mu_meo is not None else d.mu_MEO,
    )

    paths = run_leontyeva(p, args.out)
    logger.info("Stage B done: %s", paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
