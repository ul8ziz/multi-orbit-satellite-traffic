"""CLI: per-slice SARIMA + LSTM/MLP forecasts (no Markov linkage)."""

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

    p = argparse.ArgumentParser(description="Per-slice traffic forecasting.")
    p.add_argument(
        "--cleaned",
        type=Path,
        default=None,
        help="Path to cleaned_traffic.csv (default: data/processed/cleaned_traffic.csv).",
    )
    p.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of tail timesteps for test.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for neural models.")
    p.add_argument(
        "--horizon",
        choices=("one_step", "multi_step"),
        default="one_step",
        help="one_step: Ŷ(t+1|t) on each test hour (article); multi_step: one forecast from train end.",
    )
    args = p.parse_args(argv)

    from src.forecasting.pipeline import run_forecasting

    out = run_forecasting(
        PROJECT_ROOT, cleaned_path=args.cleaned, test_ratio=args.test_ratio, seed=args.seed, horizon=args.horizon
    )
    logger.info("Forecast outputs: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
