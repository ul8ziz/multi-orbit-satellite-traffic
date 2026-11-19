# -*- coding: utf-8 -*-
"""
Полный прогон по заданию: Stage A (полный датасет) -> проверка -> прогноз -> Stage B -> интеграция.

Использование:
  python run_assignment_pipeline.py
  python run_assignment_pipeline.py --alpha 1.0
  python run_assignment_pipeline.py --skip-stage-a   # только прогноз, Markov, интеграция (Stage A уже есть)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_src_on_path() -> None:
    r = str(PROJECT_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _verify_stage_b(project_root: Path) -> tuple[bool, list[str]]:
    need = [
        "outputs/markov/stationary_distribution.csv",
        "outputs/markov/metrics.csv",
        "outputs/integration/scenario_metrics.csv",
        "outputs/integration/blocking_vs_scenario.png",
        "outputs/integration/integration_summary.txt",
    ]
    missing = [x for x in need if not (project_root / x).is_file()]
    return len(missing) == 0, missing


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    configure_logging()
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description="Assignment pipeline: Stage A verify -> Forecast -> Stage B -> Integration.")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha in lambda_t = alpha * total_mb / 3600")
    ap.add_argument(
        "--skip-stage-a",
        action="store_true",
        help="Не запускать Stage A (ожидаются готовые data/processed/* и outputs).",
    )
    ap.add_argument(
        "--skip-report",
        action="store_true",
        help="Не запускать build_final_report.py в конце.",
    )
    args = ap.parse_args(argv)

    import run_forecasting as forecasting
    import run_stage_a as stage_a
    import run_stage_b as stage_b

    from src.analysis.verify_stage_a import print_stage_a_verification_report, verify_stage_a_outputs
    from src.integration.pipeline import run_integration
    from src.preprocessing.dataset_path import resolve_real_dataset_path

    ds = resolve_real_dataset_path(PROJECT_ROOT)
    print(f"REAL DATASET PATH: {ds.resolve()}", flush=True)

    if not args.skip_stage_a:
        logger.info("Running Stage A (chunked, full dataset)...")
        rc = stage_a.main([])
        if rc != 0:
            print("Stage A failed.", file=sys.stderr)
            return rc

    v = verify_stage_a_outputs(PROJECT_ROOT, expect_forecast=False)
    print_stage_a_verification_report(v, PROJECT_ROOT)
    if not v.ok:
        logger.error("Stage A verification failed (missing outputs).")
        return 1

    print("", flush=True)
    print("=== STAGE A VERIFIED ===", flush=True)
    print("", flush=True)

    logger.info("Running forecasting...")
    rc = forecasting.main([])
    if rc != 0:
        return rc

    v_f = verify_stage_a_outputs(PROJECT_ROOT, expect_forecast=True)
    print_stage_a_verification_report(v_f, PROJECT_ROOT)
    if not v_f.ok:
        logger.error("Forecast outputs missing.")
        return 1

    logger.info("Running Stage B (Leontyeva baseline)...")
    rc = stage_b.main([])
    if rc != 0:
        return rc

    logger.info("Running integration (lambda scenarios)...")
    run_integration(PROJECT_ROOT, alpha=args.alpha)

    ok_b, miss_b = _verify_stage_b(PROJECT_ROOT)
    if not ok_b:
        logger.error("Stage B / integration verification failed: %s", miss_b)
        return 1

    print("", flush=True)
    print("=== STAGE B COMPLETED ===", flush=True)
    print("  outputs/markov/stationary_distribution.csv", flush=True)
    print("  outputs/markov/metrics.csv", flush=True)
    print("  outputs/integration/scenario_metrics.csv", flush=True)
    print("  outputs/integration/blocking_vs_scenario.png", flush=True)
    print("  outputs/integration/integration_summary.txt", flush=True)
    print("", flush=True)

    if not args.skip_report:
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "build_final_report.py")],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if r.returncode != 0:
            logger.warning("build_final_report.py exit %s", r.returncode)

    logger.info("Assignment pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
