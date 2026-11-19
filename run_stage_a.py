"""CLI entry point for Stage A: traffic data analysis (assignment requirements)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_src_on_path() -> None:
    """Insert project root so ``import src.…`` resolves when run as a script."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def configure_logging() -> None:
    """Configure root logging for local runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    """Run Stage A: preprocessing, slices, users, spatial demand, stats, figures."""
    _ensure_src_on_path()
    configure_logging()
    logger = logging.getLogger(__name__)

    from src.preprocessing.dataset_path import resolve_real_dataset_path

    default_input = resolve_real_dataset_path(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Stage A: traffic analysis pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=(
            "Только Applications_data_sample.csv (по умолчанию: рядом с каталогом project/ "
            "или project/data/raw/Applications_data_sample.csv)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the spatial model (80/20 clusters vs uniform).",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Размер чанка чтения CSV (по умолчанию 500000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Игнорировать чекпойнты и пересчитать предобработку с сырого CSV.",
    )
    args = parser.parse_args(argv)

    input_resolved = Path(args.input).resolve()
    print(f"REAL DATASET PATH (Stage A input): {input_resolved}", flush=True)

    from src.analysis.stage_a_pipeline import run_stage_a

    paths = run_stage_a(
        PROJECT_ROOT,
        input_resolved,
        seed=args.seed,
        force=args.force,
        chunk_rows=args.chunk_rows,
    )
    logger.info("Stage A complete. Key outputs: %s", paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
