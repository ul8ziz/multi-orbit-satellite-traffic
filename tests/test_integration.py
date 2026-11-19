"""Integration: λ from traffic and scenario CTMC."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_derive_lambda_and_scenarios(tmp_path: Path) -> None:
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01 12:00:00"],
                format="mixed",
            ).normalize(),
            "hour": [0, 0, 12],
            "total_mb": [3600.0, 3600.0, 7200.0],
            "slice_id": ["eMBB", "eMBB", "eMBB"],
        }
    )
    p = proc / "cleaned_traffic.csv"
    df.to_csv(p, index=False)

    from src.integration.lambda_from_traffic import derive_lambda_scenarios
    from src.integration.scenario_markov import run_scenarios

    lam_m, lam_p, h = derive_lambda_scenarios(p, alpha=1.0)
    assert lam_m > 0 and lam_p >= lam_m

    out = tmp_path / "outputs" / "integration"
    run_scenarios(lam_m, lam_p, out_dir=out, alpha=1.0)
    assert (out / "scenario_metrics.csv").is_file()
    assert (out / "blocking_vs_scenario.png").is_file()
