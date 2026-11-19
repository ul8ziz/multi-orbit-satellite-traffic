"""CLI: derive λ from Stage A data and run mean/peak Leontyeva scenarios."""

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

    ap = argparse.ArgumentParser(description="Integration: traffic λ → Leontyeva scenarios.")
    ap.add_argument(
        "--cleaned",
        type=Path,
        default=None,
        help="Path to cleaned_traffic.csv (default: data/processed/cleaned_traffic.csv).",
    )
    ap.add_argument("--alpha", type=float, default=1.0, help="Scale in λ_t = α·total_mb/3600.")
    args = ap.parse_args(argv)

    from src.integration.pipeline import run_integration

    out = run_integration(PROJECT_ROOT, cleaned_path=args.cleaned, alpha=args.alpha)
    logger.info("Integration: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
