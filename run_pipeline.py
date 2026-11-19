"""Full pipeline: Stage A → Forecast → Markov → Integration; list generated files."""

from __future__ import annotations

import argparse
import logging
import subprocess
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


def print_generated_files(project_root: Path) -> None:
    """Print paths of files under ``outputs/`` and ``data/processed/``."""

    root = project_root.resolve()
    print("\n--- Generated files ---")
    for sub in ("outputs", "data/processed"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                print(rel.as_posix())
    print("--- End ---\n")


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    configure_logging()
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description="Run Stage A → Forecast → Markov → Integration.")
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scale α in λ_t = α·total_mb/3600 for integration scenarios.",
    )
    args = ap.parse_args(argv)

    import run_forecasting as forecasting
    import run_stage_a as stage_a
    import run_stage_b as stage_b

    from src.integration.pipeline import run_integration
    from src.preprocessing.dataset_path import resolve_real_dataset_path

    ds_path = resolve_real_dataset_path(PROJECT_ROOT)
    logger.info("Real dataset (Stage A will load): %s", ds_path)
    print(f"REAL DATASET PATH (pipeline): {ds_path.resolve()}", flush=True)

    logger.info("Pipeline: Stage A")
    stage_a.main([])

    logger.info("Pipeline: Forecast")
    forecasting.main([])

    logger.info("Pipeline: Markov (default λ baseline)")
    stage_b.main([])

    logger.info("Pipeline: Integration (λ from data, mean/peak scenarios)")
    run_integration(PROJECT_ROOT, alpha=args.alpha)

    logger.info("Pipeline: Final report (DOCX)")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "build_final_report.py")],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if r.returncode != 0:
        logger.warning("build_final_report.py exited with code %s", r.returncode)

    print_generated_files(PROJECT_ROOT)
    logger.info("Pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
