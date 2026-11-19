"""Stage A pipeline integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def test_stage_a_pipeline_writes_artifacts(tmp_path: Path) -> None:
    # Малый CSV не соответствует правилам production (имя файла и число строк).
    prev = os.environ.get("PROJECT_ALLOW_ALT_DATASET")
    os.environ["PROJECT_ALLOW_ALT_DATASET"] = "1"
    try:
        _run_tiny_stage_a(tmp_path)
    finally:
        if prev is None:
            os.environ.pop("PROJECT_ALLOW_ALT_DATASET", None)
        else:
            os.environ["PROJECT_ALLOW_ALT_DATASET"] = prev


def _run_tiny_stage_a(tmp_path: Path) -> None:
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "\n".join(
            [
                "START_HOUR,MASKED_MSISDN,APP_CLASS,UPLOAD,DOWNLOAD",
                "2024-01-01 10:00:00,A,Web Applications,1,2",
                "2024-01-01 11:00:00,A,Streaming Applications,3,4",
                "2024-01-01 10:00:00,B,Games,5,6",
                "2024-01-02 10:00:00,B,Instant Messaging Applications,0.5,0.5",
            ]
        ),
        encoding="utf-8",
    )

    from src.analysis.stage_a_pipeline import run_stage_a

    out = run_stage_a(tmp_path, csv, seed=7)
    assert out["artifact1_slice_aggregated"].is_file()
    assert out["user_profiles"].is_file()
    assert out["demand_matrix_long"].is_file()
    assert out["slice_stats"].is_file()
    assert out["correlation"].is_file()

    a1 = pd.read_csv(out["artifact1_slice_aggregated"])
    assert set(a1.columns) >= {"hour", "slice_id", "upload", "download", "total"}


def test_map_app_to_slice() -> None:
    from src.analysis.slices import map_app_to_slice

    assert map_app_to_slice("Web Applications") == "eMBB"
    assert map_app_to_slice("Streaming Applications") == "eMBB"
    assert map_app_to_slice("Music Streaming") == "eMBB"
    assert map_app_to_slice("Instant Messaging Applications") == "URLLC"
    assert map_app_to_slice("games") == "URLLC"
    assert map_app_to_slice("DB Transactions") == "URLLC"
    assert map_app_to_slice("Security") == "mMTC"
    assert map_app_to_slice("Network Operation") == "mMTC"
    assert map_app_to_slice("UnknownApp") == "mMTC"


def test_app_class_mapping_table_filters_normalized_seen_classes() -> None:
    from src.analysis.slices import app_class_mapping_table

    table = app_class_mapping_table(["Network Operation", "Network operation", "Web Applications"])

    assert table["canonical_APP_CLASS"].tolist() == ["network operation", "web applications"]
