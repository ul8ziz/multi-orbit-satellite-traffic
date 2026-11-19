"""Integration pipeline: Stage A traffic → λ scenarios → constant-λ Leontyeva solves."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.integration.lambda_from_traffic import derive_lambda_scenarios
from src.integration.scenario_markov import run_scenarios
from src.markov.leontyeva import LeontyevaParams, default_params

logger = logging.getLogger(__name__)


def run_integration(
    project_root: Path,
    *,
    cleaned_path: Path | None = None,
    alpha: float = 1.0,
    base_params: LeontyevaParams | None = None,
) -> dict[str, Any]:
    """Derive λ from cleaned traffic and run mean/peak Leontyeva scenarios.

    Does not alter the CTMC structure; only ``lam`` varies between scenario runs.
    """

    project_root = Path(project_root)
    if cleaned_path is not None:
        cleaned = Path(cleaned_path)
    else:
        pq = project_root / "data" / "processed" / "cleaned_traffic.parquet"
        csv = project_root / "data" / "processed" / "cleaned_traffic.csv"
        cleaned = pq if pq.is_file() else csv
    cleaned = Path(cleaned)
    if not cleaned.is_file():
        msg = f"Need {cleaned}; run Stage A first."
        raise FileNotFoundError(msg)

    lam_mean, lam_peak, _hourly = derive_lambda_scenarios(cleaned, alpha=alpha)
    out_dir = project_root / "outputs" / "integration"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base_params or default_params()
    p_csv, p_png = run_scenarios(
        lam_mean,
        lam_peak,
        base=base,
        out_dir=out_dir,
        alpha=alpha,
    )

    p_summary = _write_integration_summary(
        project_root,
        cleaned_path=cleaned,
        lam_mean=lam_mean,
        lam_peak=lam_peak,
        alpha=alpha,
        scenario_metrics_csv=p_csv,
        blocking_png=p_png,
        base_params=base,
    )

    return {
        "lambda_mean": lam_mean,
        "lambda_peak": lam_peak,
        "scenario_metrics": p_csv,
        "blocking_vs_scenario": p_png,
        "integration_summary": p_summary,
        "alpha": alpha,
        "base_params": base,
    }


def _write_integration_summary(
    project_root: Path,
    *,
    cleaned_path: Path,
    lam_mean: float,
    lam_peak: float,
    alpha: float,
    scenario_metrics_csv: Path,
    blocking_png: Path,
    base_params: LeontyevaParams,
) -> Path:
    """Сохраняет ``integration_summary.txt`` с путями и ключевыми числами."""
    project_root = Path(project_root)
    out_dir = project_root / "outputs" / "integration"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "integration_summary.txt"

    try:
        df = pd.read_csv(scenario_metrics_csv)
        scen_block = df.to_string(index=False)
    except Exception as e:  # noqa: BLE001
        scen_block = f"(could not read scenario_metrics: {e})"

    lines = [
        "Integration: Stage A traffic -> lambda scenarios -> Leontyeva CTMC (constant lambda per scenario)",
        "=============================================================================================",
        f"cleaned_traffic_input: {cleaned_path.resolve()}",
        f"alpha (lambda_t = alpha * total_mb / 3600): {alpha}",
        f"lambda_mean (scenario mean_load): {lam_mean}",
        f"lambda_peak (scenario peak_load): {lam_peak}",
        "",
        "Base CTMC parameters (unchanged except lambda per scenario):",
        f"  N={base_params.N} M={base_params.M} gamma={base_params.gamma} C_LEO={base_params.C_LEO}",
        f"  mu_LEO={base_params.mu_LEO} mu_MEO={base_params.mu_MEO}",
        "",
        "Outputs:",
        f"  scenario_metrics.csv: {scenario_metrics_csv.resolve()}",
        f"  blocking_vs_scenario.png: {blocking_png.resolve()}",
        "",
        "scenario_metrics table:",
        scen_block,
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", p)
    return p
